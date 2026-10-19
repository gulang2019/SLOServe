# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from SLOsServe import perf_model as perf_model_module
from SLOsServe.router import api_server_ray
from SLOsServe.router.execplan_bus import (ExecPlan, extract_load_statistics,
                                           make_default_load_stats,
                                           make_engine_state_entry,
                                           normalize_exec_plan)
from SLOsServe.router.mock_engine import (
    _abort_scheduler_request_for_mock,
    _copy_scheduler_execution_perf_model,
    _clear_scheduler_control_timing_anchor,
    _get_scheduler_exec_plan_snapshot,
    _get_scheduler_load_stats_snapshot,
    _scheduler_has_requests,
)


def _make_router_request(request_id: str, *, session_id: str | None = None,
                         cached_tokens: int = 0,
                         service_tier: str = "default"):
    payload = {
        "model": "test-model",
        "prompt": [1, 2, 3],
        "max_tokens": 8,
        "stream": True,
        "vllm_xargs": {
            "input_length": 64,
            "output_length": 8,
            "prefill_ddl": 50.0,
            "profit": 1.0,
            "slo_tpot": 0.05,
            "cached_tokens": cached_tokens,
            "service_tier": service_tier,
            "initial_service_tier": service_tier,
        },
    }
    if session_id is not None:
        payload["vllm_xargs"]["session_id"] = session_id
    return api_server_ray.RequestInstance(
        request_id=request_id,
        payload=payload,
        response_queue=None,
        arrival_time=0.0,
    )


def _expected_load_stats(**overrides):
    expected = make_default_load_stats()
    expected.update(overrides)
    return expected


def test_request_slo_violation_snapshot_waits_until_deadline():
    request = _make_router_request("req-0")

    assert request.get_slo_violation_snapshot(now=49.99) is None

    violation = request.get_slo_violation_snapshot(now=50.01)

    assert violation is not None
    assert violation["violation"] == "ttft"
    assert violation["expected_num_computed_tokens"] == 64
    assert violation["deadline"] == pytest.approx(50.0)
    assert violation["lateness_s"] == pytest.approx(0.01)


def test_request_slo_violation_snapshot_reports_late_observed_tpot():
    request = _make_router_request("req-0")
    request.update({"num_computed_tokens": 64, "timestamp": 50.0})
    request.update({"num_computed_tokens": 65, "timestamp": 50.2})

    violation = request.get_slo_violation_snapshot(now=50.2)

    assert violation is not None
    assert violation["violation"] == "tpot"
    assert violation["expected_num_computed_tokens"] == 65
    assert violation["deadline"] == pytest.approx(50.05)
    assert violation["observed_timestamp"] == pytest.approx(50.2)
    assert violation["lateness_s"] == pytest.approx(0.15)
    assert request.is_slo_violation is True


def test_request_pool_logs_backend_recovery_transitions(monkeypatch):
    pool = api_server_ray.RequestPool.__new__(api_server_ray.RequestPool)
    pool.n_devices = 2
    pool._profile_events = []
    pool._backend_recovery_state_by_device = {}
    warnings = []
    infos = []
    monkeypatch.setattr(
        api_server_ray.logger,
        "warning",
        lambda *args, **kwargs: warnings.append((args, kwargs)),
    )
    monkeypatch.setattr(
        api_server_ray.logger,
        "info",
        lambda *args, **kwargs: infos.append((args, kwargs)),
    )

    recovery_stats = [
        _expected_load_stats(
            backend_admission_checked=1,
            backend_recovery_state=1,
            backend_regular_feasible=0,
            backend_regular_count=3,
            backend_accepted_regular_count=1,
            backend_rejected_regular_count=2,
            backend_earliest_deadline_laxity_ms=0.0,
        ),
        _expected_load_stats(),
    ]
    api_server_ray.RequestPool._log_backend_recovery_states(
        pool,
        recovery_stats,
        now=50.2,
    )
    api_server_ray.RequestPool._log_backend_recovery_states(
        pool,
        recovery_stats,
        now=50.3,
    )

    assert len(warnings) == 1
    assert "Backend admission entered recovery" in warnings[0][0][0]
    assert len(pool._profile_events) == 1
    event = pool._profile_events[0]
    assert event["event_type"] == "backend_recovery_start"
    assert event["device_id"] == 0
    assert event["extra_args"]["backend_recovery_state"] == 1
    assert event["extra_args"]["backend_rejected_regular_count"] == 2

    recovered_stats = [_expected_load_stats(), _expected_load_stats()]
    api_server_ray.RequestPool._log_backend_recovery_states(
        pool,
        recovered_stats,
        now=50.4,
    )

    assert len(infos) == 1
    assert "Backend admission left recovery" in infos[0][0][0]
    assert pool._profile_events[-1]["event_type"] == "backend_recovery_end"


def test_make_engine_state_entry_defaults_load_stats():
    plan = ExecPlan()

    entry = make_engine_state_entry(3, 1.5, plan)

    assert entry["device_id"] == 3
    assert entry["timestamp"] == 1.5
    assert entry["exec_plan"] is plan
    assert entry["load_stats"] == _expected_load_stats()


def test_normalize_exec_plan_accepts_snapshot_dict():
    plan = normalize_exec_plan({
        "req_plans": {
            "req-0": [[1, 0], (2, 1)],
        },
        "batch_times": [1, 2.5],
        "num_free_blocks": "9",
    })

    assert isinstance(plan, ExecPlan)
    assert plan.req_plans["req-0"] == [(1, 0), (2, 1)]
    assert plan.batch_times == [1.0, 2.5]
    assert plan.num_free_blocks == 9


def test_extract_load_statistics_requires_load_stats_for_all_devices():
    states = {
        0: make_engine_state_entry(
            0,
            1.0,
            None,
            {"num_free_blocks": "7", "n_waitings": 2, "n_running": 1},
        ),
        1: {
            "device_id": 1,
            "timestamp": 1.0,
            "exec_plan": None,
        },
    }

    assert extract_load_statistics(states, 2) is None

    states[1] = make_engine_state_entry(
        1,
        1.0,
        None,
        {"num_free_blocks": 5, "n_waitings": 0, "n_running": 4},
    )

    assert extract_load_statistics(states, 2) == [
        _expected_load_stats(
            num_free_blocks=7,
            n_waitings=2,
            n_running=1,
        ),
        _expected_load_stats(
            num_free_blocks=5,
            n_waitings=0,
            n_running=4,
        ),
    ]


def test_mock_engine_scheduler_snapshots_have_safe_defaults():
    plan = ExecPlan()
    scheduler = SimpleNamespace(
        get_exec_plan=lambda: plan,
        waiting=[1, 2],
        running=[3],
        kv_cache_manager=SimpleNamespace(get_num_free_blocks=lambda: 11),
    )

    assert _get_scheduler_exec_plan_snapshot(scheduler) is plan
    assert _get_scheduler_load_stats_snapshot(scheduler) == _expected_load_stats(
        num_free_blocks=11,
        n_waitings=2,
        n_running=1,
    )


def test_mock_engine_scheduler_snapshot_prefers_explicit_method():
    scheduler = SimpleNamespace(
        get_load_statistics=lambda: {
            "num_free_blocks": 9,
            "n_waitings": 4,
            "n_running": 3,
        })

    assert _get_scheduler_load_stats_snapshot(scheduler) == _expected_load_stats(
        num_free_blocks=9,
        n_waitings=4,
        n_running=3,
    )


def test_mock_engine_scheduler_snapshot_prefers_router_exec_plan():
    regular_plan = ExecPlan(batch_times=[3.0], num_free_blocks=9)
    mixed_plan = ExecPlan(batch_times=[1.0], num_free_blocks=2)
    scheduler = SimpleNamespace(
        get_router_exec_plan=lambda: regular_plan,
        get_exec_plan=lambda: mixed_plan,
    )

    assert _get_scheduler_exec_plan_snapshot(scheduler) is regular_plan


def test_mock_engine_idle_helpers_reset_control_anchor():
    calls = {"count": 0}
    scheduler = SimpleNamespace(
        has_requests=lambda: False,
        clear_control_timing_anchor=lambda: calls.__setitem__(
            "count",
            calls["count"] + 1,
        ),
    )

    assert _scheduler_has_requests(scheduler) is False
    _clear_scheduler_control_timing_anchor(scheduler)
    assert calls["count"] == 1


def test_mock_engine_idle_helper_defaults_to_non_idle_without_probe():
    assert _scheduler_has_requests(SimpleNamespace()) is True


def test_mock_engine_prefers_scheduler_execution_perf_model_copy():
    execution_perf_model = SimpleNamespace(
        hardware_params=[1.0, 2.0],
        nested={"value": 1},
    )
    scheduler = SimpleNamespace(execution_perf_model=execution_perf_model)

    copied = _copy_scheduler_execution_perf_model(
        scheduler,
        "Qwen/Qwen2.5-7B-Instruct",
    )

    assert copied is not execution_perf_model
    assert copied.hardware_params == [1.0, 2.0]
    copied.nested["value"] = 9
    assert execution_perf_model.nested["value"] == 1


def test_abort_scheduler_request_for_mock_uses_finish_requests():
    class FakePlanner:
        def __init__(self):
            self.finished_request_ids = []

        def finish_request(self, request_id):
            self.finished_request_ids.append(request_id)

    class FakeScheduler:
        def __init__(self):
            self.requests = {"req-0": object()}
            self.finished = []
            self.atfc_planner = FakePlanner()

        def finish_requests(self, request_id, finished_status):
            self.finished.append((request_id, str(finished_status)))
            self.requests.pop(request_id, None)

    scheduler = FakeScheduler()

    assert _abort_scheduler_request_for_mock(scheduler, "req-0") is True

    assert scheduler.finished == [("req-0", "FINISHED_ABORTED")]
    assert scheduler.atfc_planner.finished_request_ids == ["req-0"]
    assert scheduler.requests == {}


def test_abort_scheduler_request_for_mock_ignores_unknown_request():
    scheduler = SimpleNamespace(requests={})

    assert _abort_scheduler_request_for_mock(scheduler, "missing") is False


@pytest.mark.asyncio
async def test_request_pool_get_load_statistics_prefers_bus_state():
    pool = api_server_ray.RequestPool.__new__(api_server_ray.RequestPool)
    pool.n_devices = 2
    pool.stat_window = 1.0
    now = 10.0
    pool._get_engine_states = AsyncMock(return_value={
        0: make_engine_state_entry(
            0,
            now,
            None,
            {"num_free_blocks": 8, "n_waitings": 1, "n_running": 2},
        ),
        1: make_engine_state_entry(
            1,
            now,
            None,
            {"num_free_blocks": 6, "n_waitings": 0, "n_running": 3},
        ),
    })
    pool._get_load_statistics_from_engines = AsyncMock(
        side_effect=AssertionError("RPC fallback should not be used"))
    original_time = api_server_ray.time.time
    api_server_ray.time.time = lambda: now

    try:
        stats = await api_server_ray.RequestPool.get_load_statistics(pool)
    finally:
        api_server_ray.time.time = original_time

    assert stats == [
        _expected_load_stats(
            num_free_blocks=8,
            n_waitings=1,
            n_running=2,
        ),
        _expected_load_stats(
            num_free_blocks=6,
            n_waitings=0,
            n_running=3,
        ),
    ]


@pytest.mark.asyncio
async def test_request_pool_get_load_statistics_falls_back_when_bus_state_is_placeholder_zero():
    pool = api_server_ray.RequestPool.__new__(api_server_ray.RequestPool)
    pool.n_devices = 2
    pool.stat_window = 1.0
    now = 10.0
    pool._get_engine_states = AsyncMock(return_value={
        0: make_engine_state_entry(0, now, None),
        1: make_engine_state_entry(1, now, None),
    })
    pool._get_load_statistics_from_engines = AsyncMock(return_value=[
        _expected_load_stats(
            num_free_blocks=3,
            effective_num_free_blocks=3,
            n_waitings=2,
            n_running=1,
            n_regular_waitings=2,
            n_regular_running=1,
        ),
        _expected_load_stats(
            num_free_blocks=4,
            effective_num_free_blocks=4,
            n_waitings=1,
            n_running=0,
            n_regular_waitings=1,
            n_regular_running=0,
        ),
    ])
    original_time = api_server_ray.time.time
    api_server_ray.time.time = lambda: now

    try:
        stats = await api_server_ray.RequestPool.get_load_statistics(pool)
    finally:
        api_server_ray.time.time = original_time

    assert stats == [
        _expected_load_stats(
            num_free_blocks=3,
            effective_num_free_blocks=3,
            n_waitings=2,
            n_running=1,
            n_regular_waitings=2,
            n_regular_running=1,
        ),
        _expected_load_stats(
            num_free_blocks=4,
            effective_num_free_blocks=4,
            n_waitings=1,
            n_running=0,
            n_regular_waitings=1,
            n_regular_running=0,
        ),
    ]
    pool._get_load_statistics_from_engines.assert_awaited_once()


@pytest.mark.asyncio
async def test_request_pool_get_load_statistics_falls_back_to_rpc():
    pool = api_server_ray.RequestPool.__new__(api_server_ray.RequestPool)
    pool.n_devices = 2
    pool._get_engine_states = AsyncMock(return_value={})
    pool._get_load_statistics_from_engines = AsyncMock(return_value=[
        _expected_load_stats(
            num_free_blocks=3,
            effective_num_free_blocks=3,
            n_waitings=2,
            n_running=1,
            n_regular_waitings=2,
            n_regular_running=1,
        ),
        _expected_load_stats(
            num_free_blocks=4,
            effective_num_free_blocks=4,
            n_waitings=1,
            n_running=0,
            n_regular_waitings=1,
            n_regular_running=0,
        ),
    ])

    stats = await api_server_ray.RequestPool.get_load_statistics(pool)

    assert stats == [
        _expected_load_stats(
            num_free_blocks=3,
            effective_num_free_blocks=3,
            n_waitings=2,
            n_running=1,
            n_regular_waitings=2,
            n_regular_running=1,
        ),
        _expected_load_stats(
            num_free_blocks=4,
            effective_num_free_blocks=4,
            n_waitings=1,
            n_running=0,
            n_regular_waitings=1,
            n_regular_running=0,
        ),
    ]
    pool._get_load_statistics_from_engines.assert_awaited_once()


def test_slosserve_router_get_engine_states_handles_bus_dict(monkeypatch):
    router = api_server_ray.SLOsServeRouter.__new__(api_server_ray.SLOsServeRouter)
    router.n_devices = 2
    router.n_block = 16

    now = 10.0
    plan = ExecPlan(
        req_plans={"req-0": [(2, 0), (4, 1)]},
        batch_times=[9.0, 12.0],
        num_free_blocks=7,
        batch_id=3,
    )
    bus_state = {
        0: make_engine_state_entry(
            0,
            now,
            plan,
            {
                "num_free_blocks": 8,
                "n_waitings": 1,
                "n_running": 1,
                "control_online_slack_ms": 12.5,
            },
        ),
    }
    fake_bus = SimpleNamespace(get_all=SimpleNamespace(remote=lambda: bus_state))

    monkeypatch.setattr(api_server_ray, "execplan_bus_actor", fake_bus)
    monkeypatch.setattr(api_server_ray.ray, "get", lambda x: x)
    monkeypatch.setattr(api_server_ray.time, "time", lambda: now)

    states = api_server_ray.SLOsServeRouter.get_engine_states(router)

    assert set(states) == {0, 1}
    assert states[0].batch_id == 3
    assert states[0].num_free_blocks == 7
    assert states[0].next_batch_time == 12.0
    assert states[0].num_computed_tokens == {"req-0": 4}
    assert states[0].control_online_slack_ms == pytest.approx(12.5)
    assert states[0].control_online_slack_s == pytest.approx(0.0125)
    assert states[1].batch_id is None
    assert states[1].num_free_blocks == 16
    assert states[1].next_batch_time == 10.0
    assert states[1].control_online_slack_ms == pytest.approx(0.0)
    assert states[1].control_online_slack_s == pytest.approx(0.0)


def test_slosserve_router_get_engine_states_prefers_effective_free_blocks(
    monkeypatch,
):
    router = api_server_ray.SLOsServeRouter.__new__(api_server_ray.SLOsServeRouter)
    router.n_devices = 1
    router.n_block = 16

    now = 10.0
    bus_state = {
        0: make_engine_state_entry(
            0,
            now,
            None,
            {
                "num_free_blocks": 3,
                "effective_num_free_blocks": 9,
                "n_waitings": 1,
                "n_running": 2,
                "control_online_slack_ms": 6.0,
            },
        ),
    }
    fake_bus = SimpleNamespace(get_all=SimpleNamespace(remote=lambda: bus_state))

    monkeypatch.setattr(api_server_ray, "execplan_bus_actor", fake_bus)
    monkeypatch.setattr(api_server_ray.ray, "get", lambda x: x)
    monkeypatch.setattr(api_server_ray.time, "time", lambda: now)

    states = api_server_ray.SLOsServeRouter.get_engine_states(router)

    assert states[0].num_free_blocks == 9
    assert states[0].control_online_slack_ms == pytest.approx(6.0)
    assert states[0].control_online_slack_s == pytest.approx(0.006)


def test_slosserve_router_run_with_planner_passes_engine_state():
    router = api_server_ray.SLOsServeRouter.__new__(api_server_ray.SLOsServeRouter)
    router.n_devices = 1
    router.n_group = 1
    router.group_size = 1
    router.is_pd_disagg = False
    router.strict_pd_separation = False
    router.group_idx = 0
    router.n_lb = 1
    router.lb_indices_per_group = [0]
    router._prefetched_raw_engine_states = None

    engine_state = api_server_ray.EngineState(next_batch_time=1.0,
                                              num_free_blocks=5)
    router.get_engine_states = lambda raw_states=None: {0: engine_state}

    captured: dict[str, object] = {}

    def fake_run_pre_adm_planner(did, running_requests, waiting_requests,
                                 engine_state, mode):
        captured["did"] = did
        captured["running_requests"] = running_requests
        captured["waiting_requests"] = waiting_requests
        captured["engine_state"] = engine_state
        captured["mode"] = mode
        return []

    router._run_pre_adm_planner = fake_run_pre_adm_planner

    waiting_request = SimpleNamespace(prefill_device_id=-1, session_id=None)

    api_server_ray.SLOsServeRouter.run_with_planner(router,
                                                    [waiting_request], [])

    assert captured["did"] == 0
    assert captured["waiting_requests"] == [waiting_request]
    assert captured["engine_state"] is engine_state
    assert captured["mode"] == "normal"


def test_request_pool_filters_best_effort_from_regular_running_requests():
    pool = api_server_ray.RequestPool.__new__(api_server_ray.RequestPool)
    regular = _make_router_request("req-regular")
    best_effort = _make_router_request("req-be", service_tier="best_effort")
    pool.running_pool = [regular, best_effort]

    assert pool._get_regular_running_requests() == [regular]


def test_slosserve_router_applies_perf_model_err_to_control_model(monkeypatch):
    captured_hardware_params: list[list[float]] = []
    base_params = [1.0, 2.0, 3.0, 4.0, 5.0]

    class FakeAdmCtrlScheduler:
        def __init__(self, *args, **kwargs):
            pass

        def set_ar_planner(self, *, tpots, hardware_params, fixed_bs, max_bs=None):
            captured_hardware_params.append(list(hardware_params))

    monkeypatch.setattr(api_server_ray.SLOsServe_C, "AdmCtrlScheduler",
                        FakeAdmCtrlScheduler)
    monkeypatch.setattr(
        perf_model_module.PerfModel,
        "get_perf_model",
        staticmethod(
            lambda model_name, task="default": perf_model_module.PerfModel(
                model_name,
                base_params,
            )),
    )

    router = api_server_ray.SLOsServeRouter(2, {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "tpot": 0.05,
        "device_mem": 16,
        "block_size": 16,
        "max_decode_length": 128,
        "scheduling_overhead": 0.3,
        "perf_model_err": 1.2,
    })

    expected_params = [
        1.2,
        2.4,
        3.6,
        4.8,
        6.0,
    ]

    assert router.hardware_params == pytest.approx(expected_params)
    assert router.perf_model._online_average_delay == pytest.approx(0.0)
    assert router.perf_model._online_spike_slack == pytest.approx(0.3)
    assert len(captured_hardware_params) == 2
    for hardware_params in captured_hardware_params:
        assert hardware_params == pytest.approx(expected_params)


def test_slosserve_router_run_with_planner_prefers_session_home_device():
    router = api_server_ray.SLOsServeRouter.__new__(api_server_ray.SLOsServeRouter)
    router.n_devices = 2
    router.n_group = 2
    router.group_size = 1
    router.is_pd_disagg = False
    router.strict_pd_separation = False
    router.group_idx = 1
    router.n_lb = 1
    router.lb_indices_per_group = [0, 0]
    router._session_prefill_device_map = {"session-1": 0}
    router._session_decode_device_map = {"session-1": 0}
    router._prefetched_raw_engine_states = None

    engine_state = api_server_ray.EngineState(next_batch_time=1.0,
                                              num_free_blocks=5)
    router.get_engine_states = (
        lambda raw_states=None: {0: engine_state, 1: engine_state}
    )

    captured: dict[str, object] = {}

    def fake_run_pre_adm_planner(did, running_requests, waiting_requests,
                                 engine_state, mode):
        if waiting_requests:
            captured["did"] = did
            captured["waiting_requests"] = waiting_requests
        return []

    router._run_pre_adm_planner = fake_run_pre_adm_planner

    waiting_request = _make_router_request("req-0", session_id="session-1",
                                           cached_tokens=16)

    api_server_ray.SLOsServeRouter.run_with_planner(router,
                                                    [waiting_request], [])

    assert captured["did"] == 0
    assert captured["waiting_requests"] == [waiting_request]


def test_slosserve_router_run_with_planner_strict_pd_separation_blocks_decode_spill():
    router = api_server_ray.SLOsServeRouter.__new__(api_server_ray.SLOsServeRouter)
    router.n_devices = 4
    router.n_group = 1
    router.group_size = 4
    router.is_pd_disagg = True
    router.strict_pd_separation = True
    router.group_idx = 0
    router.decode_group_idx = 0
    router.n_lb = 1
    router.lb_indices_per_group = [0]
    router.n_prefill_or_mixed_per_group = 2
    router._prefetched_raw_engine_states = None
    router.get_engine_states = lambda raw_states=None: {
        device_id: api_server_ray.EngineState(next_batch_time=0.0,
                                              num_free_blocks=16)
        for device_id in range(4)
    }

    decode_request = _make_router_request("req-decode")
    decode_request.state = api_server_ray.RequestState.PREFILL_FINISHED
    decode_request.prefill_device_id = 0

    decode_calls: list[int] = []

    def fake_run_pre_adm_planner(did, running_requests, waiting_requests,
                                 engine_state, mode):
        if mode == "decode_only" and waiting_requests:
            decode_calls.append(did)
            return list(waiting_requests)
        return []

    router._run_pre_adm_planner = fake_run_pre_adm_planner

    api_server_ray.SLOsServeRouter.run_with_planner(router,
                                                    [decode_request], [])

    assert decode_calls == [3, 2]


def test_slosserve_router_run_with_planner_strict_pd_decode_retry_stays_in_decode_region():
    router = api_server_ray.SLOsServeRouter.__new__(api_server_ray.SLOsServeRouter)
    router.n_devices = 4
    router.n_group = 1
    router.group_size = 4
    router.is_pd_disagg = True
    router.strict_pd_separation = True
    router.enable_decode_compaction = False
    router._decode_migration_plan = {}
    router.group_idx = 0
    router.decode_group_idx = 0
    router.n_lb = 1
    router.lb_indices_per_group = [0]
    router.n_prefill_or_mixed_per_group = 2
    router._prefetched_raw_engine_states = None
    router.get_engine_states = lambda raw_states=None: {
        device_id: api_server_ray.EngineState(next_batch_time=0.0,
                                              num_free_blocks=16)
        for device_id in range(4)
    }

    decode_request = _make_router_request("req-decode")
    decode_request.state = api_server_ray.RequestState.PREFILL_FINISHED
    decode_request.prefill_device_id = 0
    decode_request.decode_device_id = 2

    decode_calls: list[int] = []

    def fake_run_pre_adm_planner(did, running_requests, waiting_requests,
                                 engine_state, mode):
        if mode == "decode_only" and waiting_requests:
            decode_calls.append(did)
            return list(waiting_requests)
        return []

    router._run_pre_adm_planner = fake_run_pre_adm_planner

    api_server_ray.SLOsServeRouter.run_with_planner(router,
                                                    [decode_request], [])

    assert decode_calls == [3, 2]


def test_request_instance_fork_for_decode_migration_clears_kv_and_copies_resume_state():
    response_queue = asyncio.Queue()
    request = _make_router_request("req-parent")
    request.response_queue = response_queue
    request.state = api_server_ray.RequestState.PREFILL_FINISHED
    request.prefill_device_id = 0
    request.decode_device_id = 2
    request.generated_token_ids = [101, 102]
    request.generated_text = "hello"
    request.num_computed_tokens = request.num_prompt_tokens + 2
    request.timestamps = [(request.num_prompt_tokens, 50.0)]
    request.kv_transfer_params = {"num_tokens": 64}
    request.payload["kv_transfer_params"] = {"num_tokens": 64}
    request.payload["vllm_xargs"]["kv_transfer_params"] = {"num_tokens": 64}

    child = request.fork_for_decode_migration(3)
    child.sync_resume_state()

    assert child.request_id == request.request_id
    assert child.response_queue is response_queue
    assert child.state == api_server_ray.RequestState.PREFILL_FINISHED
    assert child.prefill_device_id == 0
    assert child.decode_device_id == 3
    assert child.is_migration_request
    assert child.migration_parent is request
    assert child.generated_token_ids == [101, 102]
    assert child.payload["vllm_xargs"]["existing_output_token_ids"] == [101, 102]
    assert child.kv_transfer_params is None
    assert "kv_transfer_params" not in child.payload
    assert "kv_transfer_params" not in child.payload["vllm_xargs"]


def test_request_pool_enqueue_decode_migration_freezes_parent_and_shares_queue():
    pool = api_server_ray.RequestPool.__new__(api_server_ray.RequestPool)
    pool.running_pool = []
    pool.waiting_pool = []
    pool._profile_events = []

    parent = _make_router_request("req-parent")
    parent.response_queue = asyncio.Queue()
    parent.state = api_server_ray.RequestState.PREFILL_FINISHED
    parent.prefill_device_id = 0
    parent.decode_device_id = 2
    parent.generated_token_ids = [42]
    pool.running_pool.append(parent)

    pool._enqueue_decode_migrations({"req-parent": 3})

    assert parent.migration_in_progress
    assert parent.migration_child is not None
    assert parent.migration_buffer == []
    assert len(pool.waiting_pool) == 1
    child = pool.waiting_pool[0]
    assert child.admitted is True
    assert child.decode_device_id == 3
    assert child.response_queue is parent.response_queue
    assert child.migration_parent is parent
    assert child.generated_token_ids == [42]


def test_slosserve_router_plans_decode_compaction_with_dry_run():
    router = api_server_ray.SLOsServeRouter.__new__(api_server_ray.SLOsServeRouter)
    router.n_devices = 4
    router.group_size = 4
    router.is_pd_disagg = True
    router.use_planner = True
    router.enable_decode_compaction = True
    router.decode_compaction_interval_s = 0.0
    router._last_decode_compaction_ts = 0.0
    router._decode_migration_plan = {}
    router._profile_events = []

    source = _make_router_request("req-source")
    source.state = api_server_ray.RequestState.PREFILL_FINISHED
    source.prefill_device_id = 0
    source.decode_device_id = 2
    source.generated_token_ids = [1, 2]

    new_decode = _make_router_request("req-new")
    new_decode.state = api_server_ray.RequestState.PREFILL_FINISHED
    new_decode.prefill_device_id = 0
    new_decode.decode_device_id = 3
    new_decode.admitted = True

    running_by_device = [[], [], [source], [new_decode]]
    engine_states = {
        did: api_server_ray.EngineState(next_batch_time=0.0, num_free_blocks=16)
        for did in range(4)
    }

    def fake_run_pre_adm_planner(did, running_requests, waiting_requests,
                                 engine_state, mode):
        assert mode == "decode_migration"
        if did == 3:
            for request in waiting_requests:
                request.decode_device_id = did
                request.admitted = True
                running_requests.append(request)
            return []
        return list(waiting_requests)

    router._run_pre_adm_planner = fake_run_pre_adm_planner

    router._maybe_plan_decode_compaction_for_group(
        group_start=0,
        group_end=4,
        running_requests_by_device=running_by_device,
        engine_states=engine_states,
        real_decode_touched_devices={3},
    )

    assert router.consume_decode_migration_plan() == {"req-source": 3}


def test_slosserve_router_skips_compaction_when_new_decode_touches_current_lowest():
    router = api_server_ray.SLOsServeRouter.__new__(api_server_ray.SLOsServeRouter)
    router.n_devices = 4
    router.group_size = 4
    router.is_pd_disagg = True
    router.use_planner = True
    router.enable_decode_compaction = True
    router.decode_compaction_interval_s = 0.0
    router._last_decode_compaction_ts = 0.0
    router._decode_migration_plan = {}
    router._profile_events = []

    source = _make_router_request("req-source")
    source.state = api_server_ray.RequestState.PREFILL_FINISHED
    source.prefill_device_id = 0
    source.decode_device_id = 2
    running_by_device = [[], [], [source], []]
    engine_states = {
        did: api_server_ray.EngineState(next_batch_time=0.0, num_free_blocks=16)
        for did in range(4)
    }

    router._run_pre_adm_planner = (
        lambda did, running_requests, waiting_requests, engine_state, mode:
        list(waiting_requests)
    )

    router._maybe_plan_decode_compaction_for_group(
        group_start=0,
        group_end=4,
        running_requests_by_device=running_by_device,
        engine_states=engine_states,
        real_decode_touched_devices={2, 3},
    )

    assert router.consume_decode_migration_plan() == {}


def test_slosserve_router_select_asap_server_reuses_last_device():
    router = api_server_ray.SLOsServeRouter.__new__(api_server_ray.SLOsServeRouter)
    router.is_pd_disagg = False

    unscheduled_request = _make_router_request("req-new")
    assert router.select_asap_server(None, request=unscheduled_request) == (0, 0)

    placed_request = _make_router_request("req-old")
    placed_request.prefill_device_id = 1
    placed_request.decode_device_id = 1
    assert router.select_asap_server(None, request=placed_request) == (1, 1)


def test_slosserve_router_select_asap_server_reuses_last_pd_devices():
    router = api_server_ray.SLOsServeRouter.__new__(api_server_ray.SLOsServeRouter)
    router.is_pd_disagg = True
    router.group_size = 4
    router.n_prefill_or_mixed_per_group = 2

    unscheduled_request = _make_router_request("req-new")
    assert router.select_asap_server(None, request=unscheduled_request) == (0, 3)

    prefill_only_request = _make_router_request("req-prefill")
    prefill_only_request.prefill_device_id = 4
    assert router.select_asap_server(None, request=prefill_only_request) == (4, 7)

    placed_request = _make_router_request("req-old")
    placed_request.prefill_device_id = 4
    placed_request.decode_device_id = 7
    assert router.select_asap_server(None, request=placed_request) == (4, 7)


def test_request_pool_routes_best_effort_to_emptiest_server():
    pool = api_server_ray.RequestPool.__new__(api_server_ray.RequestPool)
    pool.router = SimpleNamespace(
        select_asap_server=lambda load_stat, request=None: (2, 2)
    )
    pool.load_stat = SimpleNamespace()

    best_effort_request = _make_router_request(
        "req-best-effort",
        service_tier="best_effort",
    )

    api_server_ray.RequestPool._route_best_effort_requests(
        pool,
        [best_effort_request],
    )

    assert best_effort_request.admitted is True
    assert best_effort_request.prefill_device_id == 2
    assert best_effort_request.decode_device_id == 2
    assert best_effort_request.payload["vllm_xargs"]["must_admit"] is True
    assert best_effort_request.payload["vllm_xargs"]["service_tier"] == "best_effort"


def test_slosserve_router_pre_adm_planner_uses_cached_tokens_on_home_device(
    monkeypatch,
):
    router = api_server_ray.SLOsServeRouter.__new__(api_server_ray.SLOsServeRouter)
    router.ablation = False
    router.kv_xfer_delay = 0.05
    router.block_size = 16
    router.max_decode_length = 32
    router.admission_max_decode_length = 32
    router.oracle_mem = False
    router.is_pd_disagg = False
    router.router_safety_margin_s = 0.0
    router._session_prefill_device_map = {"session-hit": 1}
    router._session_decode_device_map = {"session-hit": 1}
    captured: dict[str, object] = {}

    class FakeCRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeAdmPlanner:
        def adm_ctrl(self, c_reqs, num_free_blocks, now, online_slack=0.0):
            captured["c_reqs"] = c_reqs
            captured["online_slack"] = online_slack
            return True, [False for _ in c_reqs]

    monkeypatch.setattr(api_server_ray.SLOsServe_C, "Request", FakeCRequest)
    router.adm_planner = FakeAdmPlanner()

    hit_request = _make_router_request("req-hit", session_id="session-hit",
                                       cached_tokens=16)
    miss_request = _make_router_request("req-miss", session_id="session-miss",
                                        cached_tokens=16)

    router._run_pre_adm_planner(
        did=1,
        running_requests=[],
        waiting_requests=[hit_request, miss_request],
        engine_state=api_server_ray.EngineState(next_batch_time=0.0,
                                                num_free_blocks=128,
                                                control_online_slack_s=0.2,
                                                control_online_slack_ms=200.0),
        mode="normal",
    )

    c_reqs = captured["c_reqs"]
    assert c_reqs[0].id == "req-hit"
    assert c_reqs[0].n_computed_tokens == 16
    assert c_reqs[0].mem == 5
    assert c_reqs[1].id == "req-miss"
    assert c_reqs[1].n_computed_tokens == 0
    assert c_reqs[1].mem == 6
    assert captured["online_slack"] == pytest.approx(0.2)
