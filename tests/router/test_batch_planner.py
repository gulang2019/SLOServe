import sys
from types import SimpleNamespace

import pytest


def _fake_remote(obj=None, **kwargs):
    if obj is None:
        return lambda actual: actual
    return obj


sys.modules.setdefault("SLOsServe_C", SimpleNamespace(AdmCtrlScheduler=None))
sys.modules.setdefault("ray", SimpleNamespace(remote=_fake_remote))

from SLOsServe.router import adm_ctrl


class _DummyAdmCtrlScheduler:

    def __init__(self, *args, **kwargs):
        pass

    def set_ar_planner(self, *args, **kwargs):
        pass


class _FakePerfModel:

    def __init__(self):
        self.hardware_params = [1e-5, 0.0, 0.0, 0.0, 0.0]

    def get_batch_time(self, batch):
        total_tokens = sum(n_tokens for _n_past, n_tokens in batch)
        return max(1e-4, total_tokens * 1e-5)

    def get_bs(self,
               t: float,
               num_reqs: int = 1,
               num_past_tokens: int = 0,
               num_decode_steps: int = 1):
        del num_reqs, num_past_tokens, num_decode_steps
        return max(1, int(t / 1e-5))


def _make_planner(monkeypatch: pytest.MonkeyPatch,
                  max_batch_size: int) -> adm_ctrl.BatchPlanner:
    monkeypatch.setattr(adm_ctrl.SLOsServe_C, "AdmCtrlScheduler",
                        _DummyAdmCtrlScheduler)
    planner = adm_ctrl.BatchPlanner(
        _perf_model=_FakePerfModel(),
        _block_size=16,
        _max_decode_length=1024,
        _num_free_blocks=4096,
        _max_batch_size=max_batch_size,
    )
    planner._now = lambda: 0.0
    planner.batch_id = -1
    return planner


def test_refresh_fast_caps_feasible_batch_size(monkeypatch: pytest.MonkeyPatch):
    planner = _make_planner(monkeypatch, max_batch_size=128)
    planner._requests["req-0"] = adm_ctrl.Request(
        request_id="req-0",
        num_prompt_tokens=20000,
        prefill_ddl=1.0,
        slo_tpot=0.1,
        prefill_only=True,
        kv_ready_time=None,
        output_length=0,
    )

    is_feasible, batches, _ = planner._refresh_fast()

    assert is_feasible
    assert len(batches) == 1
    assert batches[0].n_scheduled_tokens["req-0"] > 0
    assert sum(batches[0].n_scheduled_tokens.values()) <= 128


def test_refresh_fast_caps_overdue_recovery_batch_size(
        monkeypatch: pytest.MonkeyPatch):
    planner = _make_planner(monkeypatch, max_batch_size=128)
    planner._requests["req-0"] = adm_ctrl.Request(
        request_id="req-0",
        num_prompt_tokens=20000,
        prefill_ddl=0.001,
        slo_tpot=0.1,
        prefill_only=True,
        kv_ready_time=None,
        output_length=0,
    )

    is_feasible, batches, _ = planner._refresh_fast()

    assert is_feasible
    assert len(batches) == 1
    assert batches[0].n_scheduled_tokens["req-0"] > 0
    assert sum(batches[0].n_scheduled_tokens.values()) <= 128


def test_must_admit_defaults_to_best_effort(
        monkeypatch: pytest.MonkeyPatch):
    planner = _make_planner(monkeypatch, max_batch_size=128)
    planner._cpp_feasible_with_new = lambda new_req, now: (False, "CMP")

    admitted = planner.add_request(
        request_id="req-0",
        num_prompt_tokens=1024,
        num_computed_tokens=0,
        prefill_ddl=1.0,
        slo_tpot=0.1,
        prefill_only=False,
        kv_ready_time=None,
        must_admit=True,
        output_length=32,
    )

    assert admitted
    assert planner._requests["req-0"].service_tier == "best_effort"


def test_best_effort_only_uses_leftover_capacity(
        monkeypatch: pytest.MonkeyPatch):
    planner = _make_planner(monkeypatch, max_batch_size=32)
    planner._requests["req-default"] = adm_ctrl.Request(
        request_id="req-default",
        num_prompt_tokens=20000,
        prefill_ddl=1.0,
        slo_tpot=0.1,
        prefill_only=True,
        kv_ready_time=None,
        output_length=0,
    )
    planner._requests["req-best-effort"] = adm_ctrl.Request(
        request_id="req-best-effort",
        num_prompt_tokens=20000,
        prefill_ddl=0.1,
        slo_tpot=0.1,
        prefill_only=True,
        kv_ready_time=None,
        output_length=0,
        service_tier="best_effort",
    )

    is_feasible, batches, _ = planner._refresh_fast()

    assert is_feasible
    assert len(batches) == 1
    assert batches[0].n_scheduled_tokens["req-default"] > 0
    assert "req-best-effort" not in batches[0].n_scheduled_tokens
