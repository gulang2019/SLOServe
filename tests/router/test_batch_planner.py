import sys
from types import SimpleNamespace

import pytest

from Dataset.dataset import Request as DatasetRequest


def _fake_remote(obj=None, **kwargs):
    if obj is None:
        return lambda actual: actual
    return obj


sys.modules.setdefault(
    "SLOsServe_C",
    SimpleNamespace(AdmCtrlScheduler=None, Request=None),
)
_fake_ray_queue = SimpleNamespace(Queue=None)
_fake_ray_util = SimpleNamespace(queue=_fake_ray_queue)
sys.modules.setdefault(
    "ray",
    SimpleNamespace(
        remote=_fake_remote,
        util=_fake_ray_util,
        ObjectRef=object,
        get=lambda x: x,
    ),
)
sys.modules.setdefault("ray.util", _fake_ray_util)
sys.modules.setdefault("ray.util.queue", _fake_ray_queue)

from SLOsServe.router import adm_ctrl
from SLOsServe.decode_length_predictor import (
    BucketedQuantileDecodeLengthPredictor,
    DecodeLengthPredictorPlugin,
)


class _DummyAdmCtrlScheduler:

    def __init__(self, *args, **kwargs):
        pass

    def set_ar_planner(self, *args, **kwargs):
        pass

    def adm_ctrl(self, reqs, _num_free_blocks, _current_time):
        return True, [True] * len(reqs)

    def adm_ctrl_with_reason(self, reqs, _num_free_blocks, _current_time):
        return True, [True] * len(reqs), None


class _CaptureAdmCtrlScheduler:

    def __init__(self):
        self.last_reqs = []
        self.last_num_free_blocks = None

    def adm_ctrl_with_reason(self, reqs, num_free_blocks, _current_time):
        self.last_reqs = list(reqs)
        self.last_num_free_blocks = num_free_blocks
        return True, [True] * len(reqs), None


class _DummyCppRequest:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


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

    def configure_cpp_ar_planner(self, *args, **kwargs):
        pass


def _make_planner(monkeypatch: pytest.MonkeyPatch,
                  max_batch_size: int) -> adm_ctrl.BatchPlanner:
    monkeypatch.setattr(adm_ctrl.SLOsServe_C, "AdmCtrlScheduler",
                        _DummyAdmCtrlScheduler)
    monkeypatch.setattr(adm_ctrl.SLOsServe_C, "Request", _DummyCppRequest)
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


def test_refresh_fast_uses_earliest_deadline_batch_size_when_feasible(
        monkeypatch: pytest.MonkeyPatch):
    planner = _make_planner(monkeypatch, max_batch_size=128)
    planner._requests["req-0"] = adm_ctrl.Request(
        request_id="req-0",
        num_prompt_tokens=1,
        prefill_ddl=0.001,
        slo_tpot=0.1,
        prefill_only=True,
        kv_ready_time=None,
        output_length=0,
    )
    planner._requests["req-1"] = adm_ctrl.Request(
        request_id="req-1",
        num_prompt_tokens=1,
        prefill_ddl=0.5,
        slo_tpot=0.1,
        prefill_only=True,
        kv_ready_time=None,
        output_length=0,
    )

    is_feasible, batches, _ = planner._refresh_fast()

    assert is_feasible
    assert len(batches) == 1
    assert batches[0].n_scheduled_tokens["req-0"] == 1
    assert batches[0].n_scheduled_tokens["req-1"] == 1
    assert batches[0].unscheduled_tokens == 98


def test_refresh_fast_enters_recovery_when_any_regular_deadline_infeasible(
        monkeypatch: pytest.MonkeyPatch):
    planner = _make_planner(monkeypatch, max_batch_size=128)
    planner._cpp_feasible = lambda now, num_free_blocks_override=None: (False, {"req-0"})
    planner._requests["req-0"] = adm_ctrl.Request(
        request_id="req-0",
        num_prompt_tokens=80,
        prefill_ddl=0.001,
        slo_tpot=0.1,
        prefill_only=True,
        kv_ready_time=None,
        output_length=0,
    )
    planner._requests["req-1"] = adm_ctrl.Request(
        request_id="req-1",
        num_prompt_tokens=80,
        prefill_ddl=0.001,
        slo_tpot=0.1,
        prefill_only=True,
        kv_ready_time=None,
        output_length=0,
    )

    is_feasible, batches, _ = planner._refresh_fast()

    assert is_feasible
    assert len(batches) == 1
    assert batches[0].n_scheduled_tokens["req-0"] == 80
    assert batches[0].n_scheduled_tokens["req-1"] == 48
    assert sum(batches[0].n_scheduled_tokens.values()) == 128


def test_must_admit_defaults_to_best_effort(
        monkeypatch: pytest.MonkeyPatch):
    planner = _make_planner(monkeypatch, max_batch_size=128)
    planner._cpp_feasible_with_new = (
        lambda new_req, now, num_free_blocks_override=None, predictor_override=None: (
            False,
            "CMP",
        )
    )

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


def test_cpp_feasible_regular_request_ignores_best_effort_state(
        monkeypatch: pytest.MonkeyPatch):
    planner = _make_planner(monkeypatch, max_batch_size=128)
    monkeypatch.setattr(
        adm_ctrl.SLOsServe_C,
        "Request",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )
    capture = _CaptureAdmCtrlScheduler()
    planner._adm_ctrler = capture
    planner._ensure_cpp_planner = lambda _tpot: None
    planner._requests["req-regular"] = adm_ctrl.Request(
        request_id="req-regular",
        num_prompt_tokens=128,
        num_computed_tokens=16,
        prefill_ddl=1.0,
        slo_tpot=0.1,
        prefill_only=False,
        kv_ready_time=None,
        output_length=32,
    )
    planner._requests["req-best-effort"] = adm_ctrl.Request(
        request_id="req-best-effort",
        num_prompt_tokens=256,
        num_computed_tokens=32,
        prefill_ddl=0.5,
        slo_tpot=0.1,
        prefill_only=False,
        kv_ready_time=None,
        output_length=32,
        service_tier="best_effort",
    )

    feasible, reason = planner._cpp_feasible_with_new(
        adm_ctrl.Request(
            request_id="req-new",
            num_prompt_tokens=64,
            num_computed_tokens=0,
            prefill_ddl=1.5,
            slo_tpot=0.1,
            prefill_only=False,
            kv_ready_time=None,
            output_length=16,
        ),
        now=0.0,
        num_free_blocks_override=77,
    )

    assert feasible is True
    assert reason is None
    assert capture.last_num_free_blocks == 77
    assert [req.id for req in capture.last_reqs] == ["req-regular", "req-new"]


def test_cpp_request_uses_decode_length_predictor(
        monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(adm_ctrl.SLOsServe_C, "AdmCtrlScheduler",
                        _DummyAdmCtrlScheduler)
    monkeypatch.setattr(adm_ctrl.SLOsServe_C, "Request", _DummyCppRequest)
    predictor = BucketedQuantileDecodeLengthPredictor.fit_from_requests(
        [
            DatasetRequest(input_length=64, output_length=10),
            DatasetRequest(input_length=64, output_length=20),
            DatasetRequest(input_length=64, output_length=48),
        ],
        workload_type="test",
        prompt_bucket_uppers=(128,),
    )
    planner = adm_ctrl.BatchPlanner(
        _perf_model=_FakePerfModel(),
        _block_size=16,
        _max_decode_length=1024,
        _num_free_blocks=4096,
        _decode_length_predictor=DecodeLengthPredictorPlugin.quantile_plugin(
            predictor,
            0.95,
        ),
    )
    capture = _CaptureAdmCtrlScheduler()
    planner._adm_ctrler = capture
    planner._ensure_cpp_planner = lambda _tpot: None
    planner._now = lambda: 0.0

    admitted = planner.add_request(
        request_id="req-0",
        num_prompt_tokens=64,
        num_computed_tokens=0,
        prefill_ddl=1.0,
        slo_tpot=0.1,
        prefill_only=False,
        output_length=12,
    )

    assert admitted
    assert capture.last_num_free_blocks == 4096
    assert len(capture.last_reqs) == 1
    assert capture.last_reqs[0].max_tokens == 48
    assert capture.last_reqs[0].mem == 7
