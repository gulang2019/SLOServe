# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from SLOsServe import perf_model as perf_model_module
from SLOsServe.router import api_server_ray
from SLOsServe.router.execplan_bus import (ExecPlan, extract_load_statistics,
                                           make_engine_state_entry,
                                           normalize_exec_plan)
from SLOsServe.router.mock_engine import (_get_scheduler_exec_plan_snapshot,
                                          _get_scheduler_load_stats_snapshot)


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


def test_make_engine_state_entry_defaults_load_stats():
    plan = ExecPlan()

    entry = make_engine_state_entry(3, 1.5, plan)

    assert entry["device_id"] == 3
    assert entry["timestamp"] == 1.5
    assert entry["exec_plan"] is plan
    assert entry["load_stats"] == {
        "num_free_blocks": 0,
        "effective_num_free_blocks": 0,
        "n_waitings": 0,
        "n_running": 0,
        "n_regular_waitings": 0,
        "n_regular_running": 0,
        "n_best_effort_waitings": 0,
        "n_best_effort_running": 0,
    }


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
        {
            "num_free_blocks": 7,
            "effective_num_free_blocks": 0,
            "n_waitings": 2,
            "n_running": 1,
            "n_regular_waitings": 0,
            "n_regular_running": 0,
            "n_best_effort_waitings": 0,
            "n_best_effort_running": 0,
        },
        {
            "num_free_blocks": 5,
            "effective_num_free_blocks": 0,
            "n_waitings": 0,
            "n_running": 4,
            "n_regular_waitings": 0,
            "n_regular_running": 0,
            "n_best_effort_waitings": 0,
            "n_best_effort_running": 0,
        },
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
    assert _get_scheduler_load_stats_snapshot(scheduler) == {
        "num_free_blocks": 11,
        "effective_num_free_blocks": 0,
        "n_waitings": 2,
        "n_running": 1,
        "n_regular_waitings": 0,
        "n_regular_running": 0,
        "n_best_effort_waitings": 0,
        "n_best_effort_running": 0,
    }


def test_mock_engine_scheduler_snapshot_prefers_explicit_method():
    scheduler = SimpleNamespace(
        get_load_statistics=lambda: {
            "num_free_blocks": 9,
            "n_waitings": 4,
            "n_running": 3,
        })

    assert _get_scheduler_load_stats_snapshot(scheduler) == {
        "num_free_blocks": 9,
        "effective_num_free_blocks": 0,
        "n_waitings": 4,
        "n_running": 3,
        "n_regular_waitings": 0,
        "n_regular_running": 0,
        "n_best_effort_waitings": 0,
        "n_best_effort_running": 0,
    }


def test_mock_engine_scheduler_snapshot_prefers_router_exec_plan():
    regular_plan = ExecPlan(batch_times=[3.0], num_free_blocks=9)
    mixed_plan = ExecPlan(batch_times=[1.0], num_free_blocks=2)
    scheduler = SimpleNamespace(
        get_router_exec_plan=lambda: regular_plan,
        get_exec_plan=lambda: mixed_plan,
    )

    assert _get_scheduler_exec_plan_snapshot(scheduler) is regular_plan


@pytest.mark.asyncio
async def test_request_pool_get_load_statistics_prefers_bus_state():
    pool = api_server_ray.RequestPool.__new__(api_server_ray.RequestPool)
    pool.n_devices = 2
    pool._get_engine_states = AsyncMock(return_value={
        0: make_engine_state_entry(
            0,
            1.0,
            None,
            {"num_free_blocks": 8, "n_waitings": 1, "n_running": 2},
        ),
        1: make_engine_state_entry(
            1,
            1.0,
            None,
            {"num_free_blocks": 6, "n_waitings": 0, "n_running": 3},
        ),
    })
    pool._get_load_statistics_from_engines = AsyncMock(
        side_effect=AssertionError("RPC fallback should not be used"))

    stats = await api_server_ray.RequestPool.get_load_statistics(pool)

    assert stats == [
        {
            "num_free_blocks": 8,
            "effective_num_free_blocks": 0,
            "n_waitings": 1,
            "n_running": 2,
            "n_regular_waitings": 0,
            "n_regular_running": 0,
            "n_best_effort_waitings": 0,
            "n_best_effort_running": 0,
        },
        {
            "num_free_blocks": 6,
            "effective_num_free_blocks": 0,
            "n_waitings": 0,
            "n_running": 3,
            "n_regular_waitings": 0,
            "n_regular_running": 0,
            "n_best_effort_waitings": 0,
            "n_best_effort_running": 0,
        },
    ]


@pytest.mark.asyncio
async def test_request_pool_get_load_statistics_falls_back_to_rpc():
    pool = api_server_ray.RequestPool.__new__(api_server_ray.RequestPool)
    pool.n_devices = 2
    pool._get_engine_states = AsyncMock(return_value={})
    pool._get_load_statistics_from_engines = AsyncMock(return_value=[
        {
            "num_free_blocks": 3,
            "effective_num_free_blocks": 3,
            "n_waitings": 2,
            "n_running": 1,
            "n_regular_waitings": 2,
            "n_regular_running": 1,
            "n_best_effort_waitings": 0,
            "n_best_effort_running": 0,
        },
        {
            "num_free_blocks": 4,
            "effective_num_free_blocks": 4,
            "n_waitings": 1,
            "n_running": 0,
            "n_regular_waitings": 1,
            "n_regular_running": 0,
            "n_best_effort_waitings": 0,
            "n_best_effort_running": 0,
        },
    ])

    stats = await api_server_ray.RequestPool.get_load_statistics(pool)

    assert stats == [
        {
            "num_free_blocks": 3,
            "effective_num_free_blocks": 3,
            "n_waitings": 2,
            "n_running": 1,
            "n_regular_waitings": 2,
            "n_regular_running": 1,
            "n_best_effort_waitings": 0,
            "n_best_effort_running": 0,
        },
        {
            "num_free_blocks": 4,
            "effective_num_free_blocks": 4,
            "n_waitings": 1,
            "n_running": 0,
            "n_regular_waitings": 1,
            "n_regular_running": 0,
            "n_best_effort_waitings": 0,
            "n_best_effort_running": 0,
        },
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
            {"num_free_blocks": 8, "n_waitings": 1, "n_running": 1},
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
    assert states[1].batch_id is None
    assert states[1].num_free_blocks == 16
    assert states[1].next_batch_time == 10.0


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
            },
        ),
    }
    fake_bus = SimpleNamespace(get_all=SimpleNamespace(remote=lambda: bus_state))

    monkeypatch.setattr(api_server_ray, "execplan_bus_actor", fake_bus)
    monkeypatch.setattr(api_server_ray.ray, "get", lambda x: x)
    monkeypatch.setattr(api_server_ray.time, "time", lambda: now)

    states = api_server_ray.SLOsServeRouter.get_engine_states(router)

    assert states[0].num_free_blocks == 9


def test_slosserve_router_run_with_planner_passes_engine_state():
    router = api_server_ray.SLOsServeRouter.__new__(api_server_ray.SLOsServeRouter)
    router.n_devices = 1
    router.n_group = 1
    router.group_size = 1
    router.is_pd_disagg = False
    router.group_idx = 0
    router.n_lb = 1
    router.lb_indices_per_group = [0]

    engine_state = api_server_ray.EngineState(next_batch_time=1.0,
                                              num_free_blocks=5)
    router.get_engine_states = lambda: {0: engine_state}

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
        6.0 + 0.3,
    ]

    assert router.hardware_params == pytest.approx(expected_params)
    assert len(captured_hardware_params) == 2
    for hardware_params in captured_hardware_params:
        assert hardware_params == pytest.approx(expected_params)


def test_slosserve_router_run_with_planner_prefers_session_home_device():
    router = api_server_ray.SLOsServeRouter.__new__(api_server_ray.SLOsServeRouter)
    router.n_devices = 2
    router.n_group = 2
    router.group_size = 1
    router.is_pd_disagg = False
    router.group_idx = 1
    router._session_prefill_device_map = {"session-1": 0}
    router._session_decode_device_map = {"session-1": 0}

    engine_state = api_server_ray.EngineState(next_batch_time=1.0,
                                              num_free_blocks=5)
    router.get_engine_states = lambda: {0: engine_state, 1: engine_state}

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
    router._session_prefill_device_map = {"session-hit": 1}
    router._session_decode_device_map = {"session-hit": 1}

    captured: dict[str, object] = {}

    class FakeCRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeAdmPlanner:
        def adm_ctrl(self, c_reqs, num_free_blocks, now):
            captured["c_reqs"] = c_reqs
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
                                                num_free_blocks=128),
        mode="normal",
    )

    c_reqs = captured["c_reqs"]
    assert c_reqs[0].id == "req-hit"
    assert c_reqs[0].n_computed_tokens == 16
    assert c_reqs[0].mem == 5
    assert c_reqs[1].id == "req-miss"
    assert c_reqs[1].n_computed_tokens == 0
    assert c_reqs[1].mem == 6
