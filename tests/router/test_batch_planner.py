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

    def adm_ctrl(self, reqs, _num_free_blocks, _current_time, _online_slack=0.0):
        return True, [True] * len(reqs)

    def adm_ctrl_with_reason(self, reqs, _num_free_blocks, _current_time, _online_slack=0.0):
        return True, [True] * len(reqs), None


class _FlipAdmCtrlScheduler:

    def __init__(self, *args, **kwargs):
        self.calls = 0

    def set_ar_planner(self, *args, **kwargs):
        pass

    def adm_ctrl_with_reason(self, reqs, _num_free_blocks, _current_time, _online_slack=0.0):
        self.calls += 1
        if self.calls == 1:
            return True, [True] * len(reqs), "NONE"
        return False, [False] * len(reqs), "CMP"


class _CaptureAdmCtrlScheduler:

    def __init__(self):
        self.last_reqs = []
        self.last_num_free_blocks = None
        self.last_online_slack = None
        self.last_schedule_online_slack = None

    def adm_ctrl_with_reason(self, reqs, num_free_blocks, _current_time, online_slack=0.0):
        self.last_reqs = list(reqs)
        self.last_num_free_blocks = num_free_blocks
        self.last_online_slack = online_slack
        return True, [True] * len(reqs), None

    def schedule(self, reqs, _current_time, _max_time, _verbose, online_slack=0.0):
        self.last_reqs = list(reqs)
        self.last_schedule_online_slack = online_slack
        return True, []


class _DummyCppRequest:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakePerfModel:

    def __init__(self):
        self.hardware_params = [1e-5, 0.0, 0.0, 0.0, 0.0]
        self._online_average_delay = 0.0
        self._online_spike_slack = 0.0

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

    def get_cpp_planner_config(self):
        return {
            "type": "linear",
            "hardware_params": list(self.hardware_params),
        }

    def update(self, delta):
        self._online_average_delay = delta
        self._online_spike_slack = max(delta, 0.0)


class _PiecewisePerfModel(_FakePerfModel):

    def get_cpp_planner_config(self):
        return {
            "type": "piecewise_current_tokens",
            "breakpoints": [64, 256],
            "segment_hardware_params": [
                [1e-5, 0.0, 0.0, 0.0, 1e-4],
                [2e-5, 1e-6, 0.0, 0.0, 2e-4],
                [3e-5, 2e-6, 1e-7, 0.0, 3e-4],
            ],
        }


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


def test_admitted_remote_request_signal_survives_until_arrival(
        monkeypatch: pytest.MonkeyPatch):
    planner = _make_planner(monkeypatch, max_batch_size=128)
    planner.register_request(
        request_id="req-remote",
        num_prompt_tokens=64,
        num_computed_tokens=64,
        prefill_ddl=1.0,
        slo_tpot=0.1,
        kv_ready_time=10.0,
        output_length=32,
    )
    planner.register_request(
        request_id="req-local",
        num_prompt_tokens=64,
        num_computed_tokens=64,
        prefill_ddl=1.0,
        slo_tpot=0.1,
        kv_ready_time=None,
        output_length=32,
    )

    _scheduled, admitted, _exec_plan = planner.get_next_batch_and_admitted_reqs()

    assert admitted == {"req-local"}
    assert planner._admitted_requests == ["req-remote"]

    planner.request_arrive("req-remote")
    _scheduled, admitted, _exec_plan = planner.get_next_batch_and_admitted_reqs()

    assert "req-remote" in admitted
    assert planner._admitted_requests == []


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
    recovery_state = planner.get_backend_admission_state()
    assert recovery_state.pop("backend_earliest_deadline_laxity_ms") == pytest.approx(1.0)
    assert recovery_state == {
        "backend_admission_checked": 1,
        "backend_recovery_state": 1,
        "backend_regular_feasible": 0,
        "backend_regular_count": 2,
        "backend_accepted_regular_count": 1,
        "backend_rejected_regular_count": 1,
        "backend_threshold_enabled": 0,
        "backend_threshold_limit": 0,
        "backend_threshold_active_count": 0,
        "backend_threshold_arrived_count": 0,
        "backend_threshold_pending_count": 0,
        "backend_threshold_decode_only_count": 0,
        "backend_threshold_non_decode_count": 0,
        "backend_pending_admitted_count": 0,
    }


def test_refresh_fast_threshold_schedules_existing_over_limit_decode_requests(
        monkeypatch: pytest.MonkeyPatch):
    planner = _make_planner(monkeypatch, max_batch_size=128)
    planner._threshold_admission_request_limit = 1

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("threshold decode-only path should bypass this")

    planner._cpp_feasible = _fail_if_called
    planner._fast_sched_get_bs = _fail_if_called
    planner._requests["req-0"] = adm_ctrl.Request(
        request_id="req-0",
        num_prompt_tokens=64,
        num_computed_tokens=64,
        prefill_ddl=1.0,
        slo_tpot=0.1,
        prefill_only=False,
        kv_ready_time=None,
        output_length=32,
    )
    planner._requests["req-1"] = adm_ctrl.Request(
        request_id="req-1",
        num_prompt_tokens=64,
        num_computed_tokens=64,
        prefill_ddl=1.1,
        slo_tpot=0.1,
        prefill_only=False,
        kv_ready_time=None,
        output_length=32,
    )

    is_feasible, batches, _ = planner._refresh_fast()

    assert is_feasible
    assert len(batches) == 1
    assert batches[0].n_scheduled_tokens == {"req-0": 1, "req-1": 1}
    state = planner.get_backend_admission_state()
    assert state["backend_recovery_state"] == 0
    assert state["backend_regular_feasible"] == 1
    assert state["backend_threshold_limit"] == 1
    assert state["backend_threshold_active_count"] == 2
    assert planner._profile_events[-1]["extra_args"]["batch_size_source"] == (
        "threshold_decode_only"
    )


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


def test_cpp_feasible_forwards_online_slack_to_cpp_admission(
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
    planner._perf_model._online_spike_slack = 0.25
    planner._requests["req-regular"] = adm_ctrl.Request(
        request_id="req-regular",
        num_prompt_tokens=128,
        num_computed_tokens=64,
        prefill_ddl=1.0,
        slo_tpot=0.1,
        prefill_only=False,
        kv_ready_time=None,
        output_length=32,
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
    )

    assert feasible is True
    assert reason is None
    assert capture.last_online_slack == pytest.approx(0.25)


def test_refresh_fast_cpp_feasible_ignores_unarrived_regular_requests(
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
    planner._requests["req-arrived"] = adm_ctrl.Request(
        request_id="req-arrived",
        num_prompt_tokens=64,
        num_computed_tokens=64,
        prefill_ddl=1.0,
        slo_tpot=0.1,
        prefill_only=False,
        kv_ready_time=None,
        output_length=32,
    )
    planner._requests["req-future"] = adm_ctrl.Request(
        request_id="req-future",
        num_prompt_tokens=64,
        num_computed_tokens=64,
        prefill_ddl=1.1,
        slo_tpot=0.1,
        prefill_only=False,
        kv_ready_time=10.0,
        output_length=32,
    )

    is_feasible, batches, _ = planner._refresh_fast()

    assert is_feasible
    assert len(batches) == 1
    assert batches[0].n_scheduled_tokens == {"req-arrived": 1}
    assert [req.id for req in capture.last_reqs] == ["req-arrived"]
    assert planner._profile_events[-1]["extra_args"]["batch_size_source"] == (
        "earliest_regular_deadline"
    )
    assert planner._profile_events[-1]["extra_args"]["accepted_regular_count"] == 1
    assert planner._profile_events[-1]["extra_args"]["rejected_regular_count"] == 0


def test_refresh_c_forwards_online_slack_to_cpp_schedule(
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
    planner._perf_model._online_spike_slack = 0.15
    planner._requests["req-arrived"] = adm_ctrl.Request(
        request_id="req-arrived",
        num_prompt_tokens=64,
        num_computed_tokens=64,
        prefill_ddl=1.0,
        slo_tpot=0.1,
        prefill_only=False,
        kv_ready_time=None,
        output_length=32,
    )

    is_feasible, batches, reason = planner._refresh_c()

    assert is_feasible is True
    assert batches == []
    assert reason == "|"
    assert capture.last_schedule_online_slack == pytest.approx(0.15)


def test_update_perf_model_no_longer_requires_cpp_update(
        monkeypatch: pytest.MonkeyPatch):
    planner = _make_planner(monkeypatch, max_batch_size=128)

    planner.update_perf_model(0.4)

    assert planner._perf_model._online_average_delay == pytest.approx(0.4)
    assert planner._perf_model._online_spike_slack == pytest.approx(0.4)


def test_refresh_fast_flip_capture_writes_replay_artifacts(
        monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("SLOSERVE_PLANNER_SCH_FLIP_DUMP_LIMIT", "1")
    monkeypatch.setattr(adm_ctrl, "_LOCAL_DUMP_DIR", str(tmp_path))
    monkeypatch.setattr(adm_ctrl.SLOsServe_C, "AdmCtrlScheduler",
                        _FlipAdmCtrlScheduler)
    monkeypatch.setattr(adm_ctrl.SLOsServe_C, "Request", _DummyCppRequest)
    planner = adm_ctrl.BatchPlanner(
        _perf_model=_FakePerfModel(),
        _block_size=16,
        _max_decode_length=1024,
        _num_free_blocks=4096,
        _max_batch_size=128,
    )
    planner._now = lambda: 0.0
    planner.batch_id = 0
    planner._requests["req-0"] = adm_ctrl.Request(
        request_id="req-0",
        num_prompt_tokens=64,
        num_computed_tokens=64,
        prefill_ddl=1.0,
        slo_tpot=0.1,
        prefill_only=False,
        kv_ready_time=None,
        output_length=32,
    )
    planner._requests["req-1"] = adm_ctrl.Request(
        request_id="req-1",
        num_prompt_tokens=64,
        num_computed_tokens=64,
        prefill_ddl=1.1,
        slo_tpot=0.1,
        prefill_only=False,
        kv_ready_time=None,
        output_length=32,
    )

    planner._refresh_fast()
    planner._refresh_fast()
    planner._emit_planner_sch_event({
        "batch_id": 0,
        "load_ddls": [],
        "sch": {},
        "unsched_tokens": 0,
        "accepted_regular_count": 0,
        "rejected_regular_count": 0,
        "batch_size_source": "earliest_regular_deadline",
        "batch_time_budget": 0.0,
        "selected_deadline_req_id": None,
        "selected_deadline_cumulative_load": 0,
        "selected_deadline_total_past_tokens": 0,
        "selected_deadline_num_reqs": 0,
        "regular_feasible": False,
        "fast_sched_perf_mode": planner._fast_sched_perf_mode,
        "total_past_tokens": 0,
        "total_scheduled_tokens": 0,
        "num_scheduled_reqs": 0,
    })

    prev_dump = tmp_path / "schedule_planner_0_planner_sch_flip_planner_0_1_prev.txt"
    curr_dump = tmp_path / "schedule_planner_0_planner_sch_flip_planner_0_1_curr.txt"
    assert (tmp_path / "planner_sch_flip_planner_0_1.json").exists()
    assert prev_dump.exists()
    assert curr_dump.exists()
    assert "SLOPACKER_SCHEDULE_DUMP_V2" in prev_dump.read_text()
    assert 'planner_config_type "linear"' in curr_dump.read_text()


def test_dump_schedule_inputs_writes_piecewise_cpp_planner_config(
        monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setattr(adm_ctrl, "_LOCAL_DUMP_DIR", str(tmp_path))
    monkeypatch.setattr(adm_ctrl.SLOsServe_C, "AdmCtrlScheduler",
                        _DummyAdmCtrlScheduler)
    monkeypatch.setattr(adm_ctrl.SLOsServe_C, "Request", _DummyCppRequest)
    planner = adm_ctrl.BatchPlanner(
        _perf_model=_PiecewisePerfModel(),
        _block_size=16,
        _max_decode_length=1024,
        _num_free_blocks=4096,
        _max_batch_size=128,
    )
    planner._adm_ctrler_tpot = 0.1

    dump_path = planner._dump_schedule_inputs(
        [
            _DummyCppRequest(
                id="req-0",
                is_new_req=False,
                ddl=1.0,
                input_length=64,
                n_computed_tokens=64,
                max_tokens=32,
                profit=0.0,
                mem=1,
                tpot_idx=0,
                prefill_mem=0,
                prefill_device_id=-1,
                decode_device_id=-1,
                prefill_only=False,
                arrival_time=0.0,
            )
        ],
        ["req-0"],
        num_free_blocks=4096,
        current_time=0.0,
        is_feasible=True,
        surfix="piecewise_test",
    )

    assert dump_path is not None
    dump_text = (tmp_path / "schedule_planner_0_piecewise_test.txt").read_text()
    assert "SLOPACKER_SCHEDULE_DUMP_V2" in dump_text
    assert 'planner_config_type "piecewise_current_tokens"' in dump_text
    assert "current_token_breakpoints 2 64 256" in dump_text
    assert "segment_hardware_params 3" in dump_text
    assert "segment_params 5 1e-05 0.0 0.0 0.0 0.0001" in dump_text


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


def test_cpp_request_clamps_decode_horizon_past_already_computed_tokens(
        monkeypatch: pytest.MonkeyPatch):
    class _ShortPredictor:

        def predict_length(self, _req):
            return 2

    monkeypatch.setattr(adm_ctrl.SLOsServe_C, "AdmCtrlScheduler",
                        _DummyAdmCtrlScheduler)
    monkeypatch.setattr(adm_ctrl.SLOsServe_C, "Request", _DummyCppRequest)
    planner = adm_ctrl.BatchPlanner(
        _perf_model=_FakePerfModel(),
        _block_size=16,
        _max_decode_length=1024,
        _num_free_blocks=4096,
        _decode_length_predictor=_ShortPredictor(),
    )

    req = adm_ctrl.Request(
        request_id="req-0",
        num_prompt_tokens=64,
        num_computed_tokens=72,
        prefill_ddl=1.0,
        slo_tpot=0.1,
        prefill_only=False,
        kv_ready_time=None,
        output_length=12,
    )
    cpp_req = planner._to_cpp_request(req, is_new_req=False, now=0.0)

    assert cpp_req.max_tokens == 9
    assert cpp_req.mem == 1
