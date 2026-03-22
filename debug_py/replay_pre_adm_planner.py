#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SLOsServe.router.api_server_ray import RequestInstance, RequestState, create_router  # noqa: E402


def _restore_request(row: dict) -> RequestInstance:
    xargs = row.get("vllm_xargs", {})
    input_length = int(xargs.get("input_length") or 0)
    output_length = int(xargs.get("output_length") or row.get("max_tokens") or 0)
    prefill_ddl = float(xargs.get("prefill_ddl") or (time.time() + 3600.0))
    arrival_time = float(row.get("arrival_time") or xargs.get("router_arrival_time") or time.time())

    payload = {
        "max_tokens": int(row.get("max_tokens") or output_length),
        "prompt": [],
        "vllm_xargs": {
            "input_length": input_length,
            "output_length": output_length,
            "prefill_ddl": prefill_ddl,
            "router_arrival_time": float(xargs.get("router_arrival_time") or arrival_time),
            "profit": float(xargs.get("profit") or 1.0),
            "slo_tpot": float(xargs.get("slo_tpot") or 0.01),
            "slo_ttft": float(xargs.get("slo_ttft") or max(prefill_ddl - arrival_time, 0.0)),
        },
    }
    req = RequestInstance(
        request_id=row["request_id"],
        payload=payload,
        response_queue=None,
        arrival_time=arrival_time,
    )
    req.admitted = row.get("admitted")
    req.prefill_device_id = int(row.get("prefill_device_id", -1))
    req.decode_device_id = int(row.get("decode_device_id", -1))
    req.num_computed_tokens = int(row.get("num_computed_tokens") or 0)
    state_name = row.get("state", "WAITING")
    req.state = RequestState[state_name] if state_name in RequestState.__members__ else RequestState.WAITING
    return req


def _restore_exec_plan(dump_exec_plan: dict | None) -> dict | None:
    if dump_exec_plan is None:
        return None
    plan = dump_exec_plan.get("exec_plan")
    if plan is None:
        return {"timestamp": dump_exec_plan.get("timestamp"), "exec_plan": None}
    req_plans = {
        rid: [tuple(x) for x in req_plan]
        for rid, req_plan in plan.get("req_plans", {}).items()
    }
    exec_plan_obj = SimpleNamespace(
        num_free_blocks=plan.get("num_free_blocks"),
        batch_times=list(plan.get("batch_times", [])),
        req_plans=req_plans,
    )
    return {"timestamp": dump_exec_plan.get("timestamp"), "exec_plan": exec_plan_obj}


def _snapshot(requests: list[RequestInstance]) -> dict[str, dict]:
    out = {}
    for req in requests:
        out[req.request_id] = {
            "admitted": req.admitted,
            "prefill_device_id": req.prefill_device_id,
            "decode_device_id": req.decode_device_id,
            "state": req.state.name,
        }
    return out


def _default_router_kwargs() -> dict[str, Any]:
    return {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "scheduling_overhead": 0.005,
        "tpot": 0.05,
        "device_mem": 10000,
        "block_size": 16,
        "max_decode_length": 100,
        "group_size": 16,
        "n_lb": 1,
        "use_planner": True,
    }


def _replay_one(path: Path, n_devices_override: int | None) -> dict[str, Any]:
    with path.open("r") as f:
        data = json.load(f)

    router_cfg = data.get("router_config", {})
    n_devices = int(n_devices_override or router_cfg.get("n_devices") or 16)
    did = int(data["did"])
    mode = data["mode"]
    original_elapsed = data.get("elapsed_s")

    router = create_router("slosserve", n_devices, _default_router_kwargs())
    waiting_requests = [_restore_request(r) for r in data.get("waiting_requests", [])]
    running_requests = [_restore_request(r) for r in data.get("running_requests", [])]
    exec_plan = _restore_exec_plan(data.get("exec_plan"))

    before_waiting = _snapshot(waiting_requests)
    t0 = time.time()
    remained_waiting = router._run_pre_adm_planner(
        did=did,
        running_requests=running_requests,
        waiting_requests=waiting_requests,
        exec_plan=exec_plan,
        mode=mode,
    )
    replay_elapsed = time.time() - t0
    after_waiting = _snapshot(waiting_requests)
    remained_ids = [r.request_id for r in remained_waiting]

    changed = []
    for rid, before in before_waiting.items():
        after = after_waiting.get(rid)
        if after != before:
            changed.append({"request_id": rid, "before": before, "after": after})

    return {
        "dump": str(path),
        "did": did,
        "mode": mode,
        "n_devices": n_devices,
        "original_elapsed_s": original_elapsed,
        "replayed_elapsed_s": replay_elapsed,
        "elapsed_delta_s": (replay_elapsed - original_elapsed) if isinstance(original_elapsed, (int, float)) else None,
        "elapsed_ratio": (replay_elapsed / original_elapsed) if isinstance(original_elapsed, (int, float)) and original_elapsed > 0 else None,
        "waiting_size": len(waiting_requests),
        "running_size": len(running_requests),
        "remained_waiting_size": len(remained_ids),
        "remained_waiting_ids": remained_ids,
        "changed_waiting_requests": changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay dumped _run_pre_adm_planner input(s).")
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--dump", help="Path to one pre_adm_planner_*.json")
    src_group.add_argument("--dump-dir", help="Directory containing pre_adm_planner dumps")
    parser.add_argument("--pattern", default="pre_adm_planner_*.json", help="Glob for --dump-dir mode")
    parser.add_argument("--n-devices", type=int, default=None, help="Override n_devices")
    args = parser.parse_args()

    if args.dump:
        result = _replay_one(Path(args.dump), args.n_devices)
        print(json.dumps(result, indent=2))
        return

    dump_dir = Path(args.dump_dir)
    files = sorted(dump_dir.glob(args.pattern))
    results = []
    for path in files:
        try:
            results.append(_replay_one(path, args.n_devices))
        except Exception as e:
            results.append({"dump": str(path), "error": str(e)})

    compared = [r for r in results if isinstance(r.get("original_elapsed_s"), (int, float))]
    summary = {
        "dump_dir": str(dump_dir),
        "pattern": args.pattern,
        "n_files": len(files),
        "n_replayed": len([r for r in results if "error" not in r]),
        "n_failed": len([r for r in results if "error" in r]),
        "n_with_original_elapsed": len(compared),
        "avg_original_elapsed_s": (
            sum(r["original_elapsed_s"] for r in compared) / len(compared)
            if compared else None
        ),
        "avg_replayed_elapsed_s": (
            sum(r["replayed_elapsed_s"] for r in compared) / len(compared)
            if compared else None
        ),
    }
    print(json.dumps({"summary": summary, "results": results}, indent=2))


if __name__ == "__main__":
    main()
