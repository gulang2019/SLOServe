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


def test_make_engine_state_entry_defaults_load_stats():
    plan = ExecPlan()

    entry = make_engine_state_entry(3, 1.5, plan)

    assert entry["device_id"] == 3
    assert entry["timestamp"] == 1.5
    assert entry["exec_plan"] is plan
    assert entry["load_stats"] == {
        "num_free_blocks": 0,
        "n_waitings": 0,
        "n_running": 0,
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
        {"num_free_blocks": 7, "n_waitings": 2, "n_running": 1},
        {"num_free_blocks": 5, "n_waitings": 0, "n_running": 4},
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
        "n_waitings": 2,
        "n_running": 1,
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
        "n_waitings": 4,
        "n_running": 3,
    }


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
        {"num_free_blocks": 8, "n_waitings": 1, "n_running": 2},
        {"num_free_blocks": 6, "n_waitings": 0, "n_running": 3},
    ]


@pytest.mark.asyncio
async def test_request_pool_get_load_statistics_falls_back_to_rpc():
    pool = api_server_ray.RequestPool.__new__(api_server_ray.RequestPool)
    pool.n_devices = 2
    pool._get_engine_states = AsyncMock(return_value={})
    pool._get_load_statistics_from_engines = AsyncMock(return_value=[
        {"num_free_blocks": 3, "n_waitings": 2, "n_running": 1},
        {"num_free_blocks": 4, "n_waitings": 1, "n_running": 0},
    ])

    stats = await api_server_ray.RequestPool.get_load_statistics(pool)

    assert stats == [
        {"num_free_blocks": 3, "n_waitings": 2, "n_running": 1},
        {"num_free_blocks": 4, "n_waitings": 1, "n_running": 0},
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


def test_slosserve_router_run_with_planner_passes_engine_state():
    router = api_server_ray.SLOsServeRouter.__new__(api_server_ray.SLOsServeRouter)
    router.n_devices = 1
    router.n_group = 1
    router.group_size = 1
    router.is_pd_disagg = False
    router.group_idx = 0

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

    waiting_request = SimpleNamespace(prefill_device_id=-1)

    api_server_ray.SLOsServeRouter.run_with_planner(router,
                                                    [waiting_request], [])

    assert captured["did"] == 0
    assert captured["waiting_requests"] == [waiting_request]
    assert captured["engine_state"] is engine_state
    assert captured["mode"] == "normal"


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
        6.0 + 0.3 + api_server_ray.PERF_MODEL_HEADROOM,
    ]

    assert router.hardware_params == pytest.approx(expected_params)
    assert len(captured_hardware_params) == 2
    for hardware_params in captured_hardware_params:
        assert hardware_params == pytest.approx(expected_params)
