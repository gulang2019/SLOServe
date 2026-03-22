#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SLOsServe.router.api_server_ray import (  # noqa: E402
    LoadStat,
    RequestInstance,
    RequestState,
    create_router,
)


def _build_payload(row: dict, default_prefill_ddl: float) -> dict:
    input_length = int(row.get("input_length") or 0)
    output_length = int(row.get("output_length") or 0)
    prefill_ddl = float(row.get("prefill_ddl") or default_prefill_ddl)
    router_arrival_time = float(row.get("router_arrival_time") or time.time())
    return {
        "max_tokens": output_length,
        "prompt": [],
        "vllm_xargs": {
            "input_length": input_length,
            "output_length": output_length,
            "prefill_ddl": prefill_ddl,
            "router_arrival_time": router_arrival_time,
            "profit": 1.0,
            "slo_tpot": 0.01,
            "slo_ttft": max(prefill_ddl - router_arrival_time, 0.0),
        },
    }


def _restore_request(row: dict, default_prefill_ddl: float) -> RequestInstance:
    request_id = row["request_id"]
    payload = _build_payload(row, default_prefill_ddl)
    arrival_time = float(row.get("arrival_time") or time.time())
    req = RequestInstance(request_id=request_id, payload=payload, response_queue=None, arrival_time=arrival_time)
    req.admitted = row.get("admitted")
    req.prefill_device_id = int(row.get("prefill_device_id", -1))
    req.decode_device_id = int(row.get("decode_device_id", -1))
    req.num_computed_tokens = int(row.get("num_computed_tokens") or 0)
    state_name = row.get("state", "WAITING")
    req.state = RequestState[state_name] if state_name in RequestState.__members__ else RequestState.WAITING
    return req


def _snapshot_decisions(requests: list[RequestInstance]) -> dict[str, dict]:
    out = {}
    for r in requests:
        out[r.request_id] = {
            "state": r.state.name,
            "admitted": r.admitted,
            "prefill_device_id": r.prefill_device_id,
            "decode_device_id": r.decode_device_id,
        }
    return out


def _infer_n_devices(snapshot: dict) -> int:
    max_did = -1
    for key in ("waiting_requests", "running_requests"):
        for row in snapshot.get(key, []):
            for did_key in ("prefill_device_id", "decode_device_id"):
                did = int(row.get(did_key, -1))
                if did >= 0:
                    max_did = max(max_did, did)
    return max_did + 1 if max_did >= 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one router.run() from routing_overhead debug dump.")
    parser.add_argument("--snapshot", required=True, help="Path to routing_overhead_*.json dump")
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    with snapshot_path.open("r") as f:
        snapshot = json.load(f)

    n_devices = 16

    router = create_router("slosserve", n_devices, {
        'model_name': 'Qwen/Qwen2.5-7B-Instruct',
        'scheduling_overhead': 0.005,
        'tpot': 0.05,
        'device_mem': 10000,
        'block_size': 16,
        'max_decode_length': 100,
        'use_planner': True
    })
    
    router.set_load_stat(LoadStat(max_window=10, n_devices=n_devices))

    default_prefill_ddl = time.time() + 3600.0
    waiting = [_restore_request(row, default_prefill_ddl) for row in snapshot.get("waiting_requests", [])]
    running = [_restore_request(row, default_prefill_ddl) for row in snapshot.get("running_requests", [])]

    before = _snapshot_decisions(waiting)
    start = time.time()
    router.run(waiting, running)
    elapsed = time.time() - start
    after = _snapshot_decisions(waiting)

    changed = []
    for rid in before:
        if before[rid] != after[rid]:
            changed.append({"request_id": rid, "before": before[rid], "after": after[rid]})

    print(json.dumps({
        "snapshot": str(snapshot_path),
        "n_devices": n_devices,
        "routing_policy": args.routing_policy,
        "router_run_elapsed_s": elapsed,
        "waiting_size": len(waiting),
        "running_size": len(running),
        "n_changed_waiting_requests": len(changed),
        "changed_waiting_requests": changed,
    }, indent=2))


if __name__ == "__main__":
    main()
