from __future__ import annotations

import copy
from typing import Any

import ray

from dataclasses import dataclass, field 

from collections import defaultdict

@dataclass
class ExecPlan:
    req_plans: dict[str, list[tuple[int, int]]] = field(default_factory=lambda : defaultdict(list))
    '''
    dict[req_id, list[(n_computed_tokens, bid)]]
    '''
    batch_times: list[float] = field(default_factory=list)
    num_free_blocks: int | None = None 
@ray.remote
class ExecPlanBus:
    """A latest-only shared state actor for per-device batch plans."""

    def __init__(self):
        self._latest: dict[int, dict[str, Any]] = {}

    def publish(self,
                device_id: int,
                timestamp: float,
                execplan: ExecPlan) -> bool:
        device_id = int(device_id)
        timestamp = float(timestamp)
        
        self._latest[device_id] = {
            "device_id": device_id,
            "timestamp": timestamp,
            "exec_plan": execplan,
        }
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
