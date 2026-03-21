import pytest

from SLOsServe.perf_model import PerfModel
from SLOsServe.router import adm_ctrl


class _DummyAdmCtrlScheduler:

    def __init__(self, *args, **kwargs):
        pass

    def set_ar_planner(self, *args, **kwargs):
        pass


def _make_planner(monkeypatch: pytest.MonkeyPatch,
                  max_batch_size: int) -> adm_ctrl.BatchPlanner:
    monkeypatch.setattr(adm_ctrl.SLOsServe_C, "AdmCtrlScheduler",
                        _DummyAdmCtrlScheduler)
    planner = adm_ctrl.BatchPlanner(
        _perf_model=PerfModel("Qwen/Qwen2.5-7B-Instruct", [1e-5, 0.0, 0.0, 0.0,
                                                           0.0]),
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
