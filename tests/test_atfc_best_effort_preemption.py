from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3rdparty" / "vllm"))

from SLOsServe.router.execplan_bus import ExecPlan, make_default_load_stats
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler_adm_ctrl import SchedulerAdmCtrl
from vllm.v1.request import Request, RequestStatus


class _FakeSchedRequest:

    def __init__(
        self,
        request_id: str,
        *,
        is_best_effort: bool,
        status: RequestStatus = RequestStatus.RUNNING,
        arrival_time: float = 0.0,
        num_computed_tokens: int = 0,
        num_prompt_tokens: int = 0,
    ) -> None:
        self.request_id = request_id
        self.is_best_effort = is_best_effort
        self.status = status
        self.arrival_time = arrival_time
        self.num_computed_tokens = num_computed_tokens
        self.num_prompt_tokens = num_prompt_tokens
        self.stop_reason = None

    def is_finished(self) -> bool:
        return False


class _FakeKVCacheManager:

    def __init__(
        self,
        block_map: dict[str, tuple[list[int], ...]],
        *,
        free_blocks: int = 0,
        freed_blocks: dict[str, int] | None = None,
    ) -> None:
        self.block_map = dict(block_map)
        self.free_blocks = free_blocks
        self.freed_blocks = dict(freed_blocks or {})

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        return self.block_map.get(request_id, ([],))

    def get_num_free_blocks(self) -> int:
        return self.free_blocks

    def get_num_freed_blocks_after_free(self, request: _FakeSchedRequest) -> int:
        return self.freed_blocks.get(request.request_id, 0)


def test_vllm_request_exposes_best_effort_service_tier():
    request = Request(
        request_id="req-best-effort",
        prompt_token_ids=[1, 2, 3],
        multi_modal_kwargs=None,
        multi_modal_hashes=None,
        multi_modal_placeholders=None,
        sampling_params=SamplingParams(
            max_tokens=4,
            extra_args={"service_tier": "best_effort"},
        ),
        pooling_params=None,
        eos_token_id=None,
    )

    assert request.service_tier == "best_effort"
    assert request.is_best_effort is True


def test_effective_free_blocks_adds_only_exclusive_best_effort_blocks():
    scheduler = SchedulerAdmCtrl.__new__(SchedulerAdmCtrl)
    scheduler._assume_no_be_block_overlap = False
    scheduler.requests = {
        "regular": _FakeSchedRequest("regular", is_best_effort=False),
        "be-1": _FakeSchedRequest("be-1", is_best_effort=True),
        "be-2": _FakeSchedRequest("be-2", is_best_effort=True),
    }
    scheduler.kv_cache_manager = _FakeKVCacheManager(
        {
            "regular": ([1, 2],),
            "be-1": ([2, 3],),
            "be-2": ([3, 4],),
        },
        free_blocks=5,
    )

    assert scheduler._get_best_effort_reclaimable_blocks() == 2
    assert scheduler._get_effective_num_free_blocks_for_admission() == 7


def test_get_load_statistics_reports_regular_and_best_effort_counts():
    scheduler = SchedulerAdmCtrl.__new__(SchedulerAdmCtrl)
    scheduler._load_stat_counters = {
        "n_oom_rejects": 0,
        "n_arrival_oom_rejects": 0,
        "n_post_admission_oom_rejects": 0,
    }
    waiting_regular = _FakeSchedRequest("wait-regular", is_best_effort=False)
    waiting_be = _FakeSchedRequest("wait-be", is_best_effort=True)
    running_regular = _FakeSchedRequest("run-regular", is_best_effort=False)
    running_be = _FakeSchedRequest("run-be", is_best_effort=True)
    scheduler.waiting_attainable = [waiting_regular]
    scheduler.waiting_kv_xfer = [waiting_be]
    scheduler.running = [running_regular, running_be]
    scheduler.requests = {}
    scheduler.kv_cache_manager = _FakeKVCacheManager({}, free_blocks=5)

    expected = make_default_load_stats()
    expected.update({
        "num_free_blocks": 5,
        "effective_num_free_blocks": 5,
        "n_waitings": 2,
        "n_running": 2,
        "n_waiting_attainable": 1,
        "n_waiting_kv_xfer": 1,
        "n_waiting_unattainable": 0,
        "n_regular_waitings": 1,
        "n_regular_running": 1,
        "n_best_effort_waitings": 1,
        "n_best_effort_running": 1,
    })

    assert scheduler.get_load_statistics() == expected


def test_control_online_slack_is_published_after_timing_update():
    class _FakePerfModel:
        def __init__(self) -> None:
            self._online_average_delay = 0.0
            self._online_spike_slack = 0.0

        def update(self, delay: float) -> None:
            self._online_average_delay = delay
            self._online_spike_slack = max(delay, 0.0)

    class _FakeAtfcPlanner:
        def __init__(self) -> None:
            self._perf_model = _FakePerfModel()

        def update_perf_model(self, delta: float) -> None:
            self._perf_model.update(delta)

    scheduler = SchedulerAdmCtrl.__new__(SchedulerAdmCtrl)
    scheduler.scheduler_config = SimpleNamespace(scheduling_policy="atfc")
    scheduler.perf_model = _FakePerfModel()
    scheduler.atfc_planner = _FakeAtfcPlanner()
    scheduler._load_stat_counters = {
        "n_oom_rejects": 0,
        "n_arrival_oom_rejects": 0,
        "n_post_admission_oom_rejects": 0,
    }
    scheduler.waiting_attainable = []
    scheduler.waiting_kv_xfer = []
    scheduler.running = []
    scheduler.requests = {}
    scheduler.kv_cache_manager = _FakeKVCacheManager({}, free_blocks=5)
    scheduler._control_time_last_schedule_point_s = 1.0
    scheduler._control_time_last_estimated_s = 0.2
    scheduler._control_time_last_delta_s = None
    scheduler._control_time_elapsed_avg_s = None
    scheduler._control_time_estimated_avg_s = None
    scheduler._control_time_delta_avg_s = None
    scheduler._control_time_delta_samples = 0

    scheduler._observe_control_timing_delta(1.35)
    stats = scheduler.get_load_statistics()

    assert scheduler.perf_model._online_spike_slack == pytest.approx(0.15)
    assert scheduler.atfc_planner._perf_model._online_spike_slack == pytest.approx(0.15)
    assert stats["control_time_delta_last_ms"] == pytest.approx(150.0)
    assert stats["control_online_slack_ms"] == pytest.approx(150.0)


def test_control_timing_update_applies_scheduler_safety_margin_only_to_learning():
    class _FakePerfModel:
        def __init__(self) -> None:
            self._decay_factor = 0.99
            self.updated_delays: list[float] = []

        def update(self, delay: float) -> None:
            self.updated_delays.append(float(delay))

    class _FakeAtfcPlanner:
        def __init__(self) -> None:
            self.updated_delays: list[float] = []

        def update_perf_model(self, delta: float) -> None:
            self.updated_delays.append(float(delta))

    scheduler = SchedulerAdmCtrl.__new__(SchedulerAdmCtrl)
    scheduler.scheduler_config = SimpleNamespace(
        scheduling_policy="atfc",
        scheduling_safety_margin=0.002,
    )
    scheduler.perf_model = _FakePerfModel()
    scheduler.atfc_planner = _FakeAtfcPlanner()
    scheduler._control_time_last_schedule_point_s = 1.0
    scheduler._control_time_last_estimated_s = 0.2
    scheduler._control_time_last_delta_s = None
    scheduler._control_time_elapsed_avg_s = None
    scheduler._control_time_estimated_avg_s = None
    scheduler._control_time_delta_avg_s = None
    scheduler._control_time_delta_samples = 0

    scheduler._observe_control_timing_delta(1.35)

    assert scheduler._control_time_last_delta_s == pytest.approx(0.15)
    assert scheduler.perf_model.updated_delays == pytest.approx([0.152])
    assert scheduler.atfc_planner.updated_delays == pytest.approx([0.152])


def test_stop_clock_monitor_preserves_seeded_control_slack():
    scheduler = SchedulerAdmCtrl.__new__(SchedulerAdmCtrl)
    scheduler._clock_monitor = None
    scheduler._clock_monitor_config = SimpleNamespace(enabled=False)
    scheduler._clock_monitor_applied_slack_s = 0.0
    scheduler.perf_model = SimpleNamespace(_online_spike_slack=0.005)
    scheduler.execution_perf_model = SimpleNamespace(_online_spike_slack=0.005)
    scheduler.atfc_planner = SimpleNamespace(
        _perf_model=SimpleNamespace(_online_spike_slack=0.005)
    )

    scheduler._stop_clock_monitor()

    assert scheduler.perf_model._online_spike_slack == pytest.approx(0.005)
    assert scheduler.execution_perf_model._online_spike_slack == pytest.approx(0.005)
    assert scheduler.atfc_planner._perf_model._online_spike_slack == pytest.approx(0.005)

    scheduler._apply_clock_monitor_slack(0.002)
    assert scheduler.perf_model._online_spike_slack == pytest.approx(0.007)
    assert scheduler.execution_perf_model._online_spike_slack == pytest.approx(0.007)
    assert scheduler.atfc_planner._perf_model._online_spike_slack == pytest.approx(0.007)

    scheduler._stop_clock_monitor()
    assert scheduler.perf_model._online_spike_slack == pytest.approx(0.005)
    assert scheduler.execution_perf_model._online_spike_slack == pytest.approx(0.005)
    assert scheduler.atfc_planner._perf_model._online_spike_slack == pytest.approx(0.005)


def test_empty_schedule_clears_control_timing_anchor():
    scheduler = SchedulerAdmCtrl.__new__(SchedulerAdmCtrl)
    scheduler._control_time_last_schedule_point_s = 3.0
    scheduler._control_time_last_estimated_s = 0.2

    empty_scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[],
                                              num_computed_tokens=[]),
        num_scheduled_tokens={},
    )

    scheduler._record_control_schedule_point(4.0, empty_scheduler_output)

    assert scheduler._control_time_last_schedule_point_s is None
    assert scheduler._control_time_last_estimated_s is None


def test_router_exec_plan_filters_best_effort_only_batches():
    scheduler = SchedulerAdmCtrl.__new__(SchedulerAdmCtrl)
    scheduler._snapshot_use_effective_free_blocks = True
    scheduler._assume_no_be_block_overlap = False
    scheduler.requests = {
        "regular": _FakeSchedRequest("regular", is_best_effort=False),
        "be": _FakeSchedRequest("be", is_best_effort=True),
    }
    scheduler.kv_cache_manager = _FakeKVCacheManager(
        {
            "regular": ([1],),
            "be": ([2],),
        },
        free_blocks=3,
    )
    scheduler._exec_plan = ExecPlan(
        req_plans={
            "be": [(1, 0)],
            "regular": [(4, 1)],
        },
        batch_times=[1.0, 2.0],
        num_free_blocks=1,
        batch_id=7,
    )

    router_plan = scheduler.get_router_exec_plan()

    assert router_plan.batch_id == 7
    assert router_plan.num_free_blocks == 4
    assert dict(router_plan.req_plans) == {
        "regular": [(4, 0)],
    }
    assert router_plan.batch_times == [2.0]


def test_schedule_stateless_atfc_respects_admitted_and_resumed_ids():
    scheduler = SchedulerAdmCtrl.__new__(SchedulerAdmCtrl)
    waiting_new = _FakeSchedRequest(
        "wait-new",
        is_best_effort=False,
        status=RequestStatus.WAITING,
    )
    waiting_other = _FakeSchedRequest(
        "wait-other",
        is_best_effort=False,
        status=RequestStatus.WAITING,
    )
    resumed = _FakeSchedRequest(
        "resume-old",
        is_best_effort=False,
        status=RequestStatus.PREEMPTED,
    )
    scheduler.waiting_attainable = [waiting_new, waiting_other]
    scheduler.waiting_unattainable = [resumed]
    scheduler.atfc_planner = SimpleNamespace(
        get_next_batch_and_admitted_reqs=lambda: (
            {"wait-new": 16, "resume-old": 1},
            {"wait-new"},
            object(),
        )
    )

    (preempted, admitted, rejected, resumed_reqs), num_scheduled_tokens = (
        scheduler._schedule_stateless_atfc()
    )

    assert preempted == []
    assert rejected == []
    assert num_scheduled_tokens == {"wait-new": 16, "resume-old": 1}
    assert [req.request_id for req in admitted] == ["wait-new"]
    assert [req.request_id for req in resumed_reqs] == ["resume-old"]


def test_schedule_stateless_atfc_threshold_promotes_ready_decode_waiters():
    scheduler = SchedulerAdmCtrl.__new__(SchedulerAdmCtrl)
    waiting_ready = _FakeSchedRequest(
        "wait-ready",
        is_best_effort=False,
        status=RequestStatus.WAITING,
        num_computed_tokens=64,
        num_prompt_tokens=64,
    )
    waiting_prefill = _FakeSchedRequest(
        "wait-prefill",
        is_best_effort=False,
        status=RequestStatus.WAITING,
        num_computed_tokens=32,
        num_prompt_tokens=64,
    )
    scheduler.waiting_attainable = [waiting_ready, waiting_prefill]
    scheduler.waiting_unattainable = []
    scheduler.scheduler_config = SimpleNamespace(
        threshold_admission_request_limit=64,
    )
    scheduler.atfc_planner = SimpleNamespace(
        get_next_batch_and_admitted_reqs=lambda: ({}, set(), object())
    )

    (preempted, admitted, rejected, resumed_reqs), num_scheduled_tokens = (
        scheduler._schedule_stateless_atfc()
    )

    assert preempted == []
    assert rejected == []
    assert resumed_reqs == []
    assert num_scheduled_tokens == {}
    assert [req.request_id for req in admitted] == ["wait-ready"]


def test_schedule_stateless_atfc_promotes_decode_ready_waiters_without_backend_threshold():
    scheduler = SchedulerAdmCtrl.__new__(SchedulerAdmCtrl)
    waiting_ready = _FakeSchedRequest(
        "wait-ready",
        is_best_effort=False,
        status=RequestStatus.WAITING,
        num_computed_tokens=64,
        num_prompt_tokens=64,
    )
    waiting_ready_later = _FakeSchedRequest(
        "wait-ready-later",
        is_best_effort=False,
        status=RequestStatus.WAITING,
        num_computed_tokens=80,
        num_prompt_tokens=64,
    )
    scheduler.waiting_attainable = [waiting_ready, waiting_ready_later]
    scheduler.waiting_unattainable = []
    scheduler.scheduler_config = SimpleNamespace(
        threshold_admission_request_limit=None,
    )
    scheduler.atfc_planner = SimpleNamespace(
        get_next_batch_and_admitted_reqs=lambda: ({}, set(), object())
    )

    (_preempted, admitted, _rejected, _resumed), num_scheduled_tokens = (
        scheduler._schedule_stateless_atfc()
    )

    assert num_scheduled_tokens == {}
    assert [req.request_id for req in admitted] == [
        "wait-ready",
        "wait-ready-later",
    ]


def test_schedule_stateless_atfc_does_not_promote_mixed_prefill_waiters():
    scheduler = SchedulerAdmCtrl.__new__(SchedulerAdmCtrl)
    waiting_ready = _FakeSchedRequest(
        "wait-ready",
        is_best_effort=False,
        status=RequestStatus.WAITING,
        num_computed_tokens=64,
        num_prompt_tokens=64,
    )
    waiting_prefill = _FakeSchedRequest(
        "wait-prefill",
        is_best_effort=False,
        status=RequestStatus.WAITING,
        num_computed_tokens=32,
        num_prompt_tokens=64,
    )
    scheduler.waiting_attainable = [waiting_ready, waiting_prefill]
    scheduler.waiting_unattainable = []
    scheduler.scheduler_config = SimpleNamespace(
        threshold_admission_request_limit=None,
    )
    scheduler.atfc_planner = SimpleNamespace(
        get_next_batch_and_admitted_reqs=lambda: ({}, set(), object())
    )

    (_preempted, admitted, _rejected, _resumed), num_scheduled_tokens = (
        scheduler._schedule_stateless_atfc()
    )

    assert num_scheduled_tokens == {}
    assert admitted == []


def test_allocate_slots_with_best_effort_fallback_drops_best_effort_and_prunes_state():
    scheduler = SchedulerAdmCtrl.__new__(SchedulerAdmCtrl)
    regular = _FakeSchedRequest(
        "regular",
        is_best_effort=False,
        status=RequestStatus.RUNNING,
    )
    best_effort = _FakeSchedRequest(
        "best-effort",
        is_best_effort=True,
        status=RequestStatus.RUNNING,
        arrival_time=10.0,
        num_computed_tokens=4,
    )
    scheduler.requests = {
        "regular": regular,
        "best-effort": best_effort,
    }
    scheduler.kv_cache_manager = _FakeKVCacheManager(
        {
            "regular": ([1, 2],),
            "best-effort": ([10, 11],),
        },
        freed_blocks={"best-effort": 2},
    )
    finished_in_planner: list[str] = []
    scheduler.scheduler_config = SimpleNamespace(scheduling_policy="atfc")
    scheduler.atfc_planner = SimpleNamespace(
        finish_request=lambda request_id: finished_in_planner.append(request_id)
    )
    scheduler._profile_events = []
    rejected: list[str] = []

    def _reject_requests(request_ids):
        ids = [request_ids] if isinstance(request_ids, str) else list(request_ids)
        for request_id in ids:
            rejected.append(request_id)
            scheduler.requests.pop(request_id, None)

    scheduler.reject_requests = _reject_requests

    num_scheduled_tokens = {"regular": 8, "best-effort": 4}
    admitted_reqs: list[_FakeSchedRequest] = []
    resumed_reqs: list[_FakeSchedRequest] = []
    scheduled_running_reqs = [best_effort]
    req_to_new_blocks = {"best-effort": object()}
    new_requests: list[_FakeSchedRequest] = []
    attempts = {"count": 0}

    def _allocate_once():
        attempts["count"] += 1
        if attempts["count"] == 1:
            return None
        return object()

    result = scheduler._allocate_slots_with_best_effort_fallback(
        request=regular,
        allocate_once=_allocate_once,
        num_scheduled_tokens=num_scheduled_tokens,
        admitted_reqs=admitted_reqs,
        resumed_reqs=resumed_reqs,
        scheduled_running_reqs=scheduled_running_reqs,
        req_to_new_blocks=req_to_new_blocks,
        new_requests=new_requests,
    )

    assert result is not None
    assert attempts["count"] == 2
    assert rejected == ["best-effort"]
    assert finished_in_planner == ["best-effort"]
    assert "best-effort" not in scheduler.requests
    assert "best-effort" not in num_scheduled_tokens
    assert scheduled_running_reqs == []
    assert req_to_new_blocks == {}
