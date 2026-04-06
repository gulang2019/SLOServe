from __future__ import annotations

import copy
from typing import Any

import ray

from dataclasses import dataclass, field 

from collections import defaultdict


DEFAULT_LOAD_STATS = {
    "num_free_blocks": 0,
    "effective_num_free_blocks": 0,
    "n_waitings": 0,
    "n_running": 0,
    "n_regular_waitings": 0,
    "n_regular_running": 0,
    "n_best_effort_waitings": 0,
    "n_best_effort_running": 0,
    "n_oom_rejects": 0,
    "n_arrival_oom_rejects": 0,
    "n_post_admission_oom_rejects": 0,
}


def make_default_load_stats() -> dict[str, int]:
    return dict(DEFAULT_LOAD_STATS)


def normalize_load_stats(load_stats: Any | None) -> dict[str, int]:
    normalized = make_default_load_stats()
    if not isinstance(load_stats, dict):
        return normalized

    for key, default_value in DEFAULT_LOAD_STATS.items():
        value = load_stats.get(key, default_value)
        try:
            normalized[key] = int(value)
        except (TypeError, ValueError):
            normalized[key] = default_value
    return normalized


@dataclass
class ExecPlan:
    req_plans: dict[str, list[tuple[int, int]]] = field(default_factory=lambda :
                                                        defaultdict(list))
    '''
    dict[req_id, list[(n_computed_tokens, bid)]]
    '''
    batch_times: list[float] = field(default_factory=list)
    num_free_blocks: int | None = None
    batch_id: int | None = None



def normalize_exec_plan(execplan: Any | None) -> ExecPlan | None:
    if execplan is None:
        return None
    if isinstance(execplan, ExecPlan):
        return execplan
    if not isinstance(execplan, dict):
        return None

    normalized = ExecPlan()

    req_plans = execplan.get("req_plans")
    if isinstance(req_plans, dict):
        normalized_req_plans = defaultdict(list)
        for req_id, req_plan in req_plans.items():
            if not isinstance(req_plan, list):
                continue
            normalized_steps: list[tuple[int, int]] = []
            for step in req_plan:
                if not isinstance(step, (list, tuple)) or len(step) != 2:
                    continue
                try:
                    normalized_steps.append((int(step[0]), int(step[1])))
                except (TypeError, ValueError):
                    continue
            normalized_req_plans[str(req_id)] = normalized_steps
        normalized.req_plans = normalized_req_plans

    batch_times = execplan.get("batch_times")
    if isinstance(batch_times, list):
        normalized.batch_times = [
            float(batch_time) for batch_time in batch_times
            if isinstance(batch_time, (int, float))
        ]

    num_free_blocks = execplan.get("num_free_blocks")
    if num_free_blocks is not None:
        try:
            normalized.num_free_blocks = int(num_free_blocks)
        except (TypeError, ValueError):
            normalized.num_free_blocks = None

    return normalized


def make_engine_state_entry(
    device_id: int,
    timestamp: float,
    execplan: "ExecPlan | None",
    load_stats: Any | None = None,
) -> dict[str, Any]:
    return {
        "device_id": int(device_id),
        "timestamp": float(timestamp),
        "exec_plan": normalize_exec_plan(execplan),
        "load_stats": normalize_load_stats(load_stats),
    }


def extract_load_statistics(
    engine_states: dict[int, dict[str, Any]] | None,
    n_devices: int,
) -> list[dict[str, int]] | None:
    if not isinstance(engine_states, dict):
        return None

    load_statistics: list[dict[str, int]] = []
    for device_id in range(n_devices):
        state = engine_states.get(device_id)
        if not isinstance(state, dict) or "load_stats" not in state:
            return None
        load_statistics.append(normalize_load_stats(state.get("load_stats")))

    return load_statistics

@ray.remote
class ExecPlanBus:
    """A latest-only shared state actor for per-device batch plans."""

    def __init__(self):
        self._latest: dict[int, dict[str, Any]] = {}

    def publish(self,
                device_id: int,
                timestamp: float,
                execplan: ExecPlan | None,
                load_stats: Any | None = None) -> bool:
        entry = make_engine_state_entry(device_id,
                                        timestamp,
                                        execplan,
                                        load_stats=load_stats)
        self._latest[entry["device_id"]] = entry
        return True

    def get_latest(self, device_id: int) -> dict[str, Any] | None:
        latest = self._latest.get(int(device_id))
        if latest is None:
            return None
        return copy.deepcopy(latest)

    def get_all(self) -> dict[int, dict[str, Any]]:
        return copy.deepcopy(self._latest)

    def reset(self):
        self._latest = {}
