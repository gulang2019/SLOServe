import argparse
import csv
import os
import sys
import types
from dataclasses import replace


try:
    import dotenv  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    sys.modules["dotenv"] = types.ModuleType("dotenv")

from Dataset.dataset import ArrivalTimes, Request, Requests
from SLOsServe.analysis.headroom_analysis import EventQueue, Instance, RequestInstance


def _load_requests(
    *,
    arrival_pattern: str,
    length_pattern: str,
    n_req: int,
    load_scale: float,
    include_thinking_in_output: bool,
) -> tuple[list[float], list[Request], float]:
    arrival_times = ArrivalTimes.load(arrival_pattern).arrival_times[:n_req]
    requests = Requests.load(length_pattern).requests[: len(arrival_times)]
    if not arrival_times or not requests:
        return [], [], 0.0

    scaled_arrivals = [t / load_scale for t in arrival_times]
    duration = arrival_times[-1] - arrival_times[0]
    base_rps = len(arrival_times) / duration if duration > 0 else 0.0

    normalized_requests: list[Request] = []
    for req in requests:
        output_length = int(req.output_length)
        if include_thinking_in_output:
            output_length += int(getattr(req, "thinking_length", 0))
        normalized_requests.append(
            replace(
                req,
                input_length=int(req.input_length - getattr(req, "cached_length", 0)),
                output_length=output_length,
                cached_length=0,
                thinking_length=0,
            )
        )
    return scaled_arrivals, normalized_requests, base_rps


def _run_once(
    *,
    arrival_pattern: str,
    length_pattern: str,
    model_name: str,
    slo_ttft_scale: float,
    slo_ttft_constant: float,
    slo_tpot: float,
    n_req: int,
    load_scale: float,
    kv_cache_mem_gb: float,
    is_oracle: bool,
    include_thinking_in_output: bool,
) -> dict:
    arrival_times, requests, base_rps = _load_requests(
        arrival_pattern=arrival_pattern,
        length_pattern=length_pattern,
        n_req=n_req,
        load_scale=load_scale,
        include_thinking_in_output=include_thinking_in_output,
    )
    if not arrival_times or not requests:
        raise RuntimeError("No arrivals or requests loaded")

    kv_cache_mem = kv_cache_mem_gb * 1e9
    max_decode_length = max(req.output_length for req in requests)

    event_queue = EventQueue()
    instance = Instance(
        device_id=0,
        event_queue=event_queue,
        slo_ttft_scale=slo_ttft_scale,
        slo_ttft_constant=slo_ttft_constant,
        slo_tpot=slo_tpot,
        model_name=model_name,
        kv_cache_mem=kv_cache_mem,
        max_decode_length=max_decode_length,
        is_oracle=is_oracle,
    )
    for i, (t, req) in enumerate(zip(arrival_times, requests)):
        event_queue.push(
            t=t,
            event_type="arrival",
            device_id=-1,
            obj=RequestInstance(req, f"req-{i}", mode="normal"),
        )

    rejected = 0
    finished = 0
    while len(event_queue):
        now, event_type, device_id, obj = event_queue.pop()
        if event_type == "arrival":
            if not instance.add_request(now, obj):
                rejected += 1
        elif event_type == "batch_finish":
            instance.on_batch_finish(now, obj)
        elif event_type == "request_finish":
            finished += 1
        else:
            raise RuntimeError(f"Unsupported event_type={event_type}")

    offered_rps = base_rps * load_scale
    row = {
        "policy": "oracle_mem" if is_oracle else "baseline",
        "arrival_pattern": arrival_pattern,
        "length_pattern": length_pattern,
        "model_name": model_name,
        "slo_ttft_scale": slo_ttft_scale,
        "slo_ttft_constant": slo_ttft_constant,
        "slo_tpot": slo_tpot,
        "n_req": len(requests),
        "load_scale": load_scale,
        "offered_rps": offered_rps,
        "kv_cache_mem_gb": kv_cache_mem_gb,
        "include_thinking_in_output": int(include_thinking_in_output),
        "rejected": rejected,
        "finished": finished,
        "violation_rate": rejected / len(requests),
        "goodput_rps": finished / len(requests) * offered_rps,
        "fail_comp": int(instance.failure_reasons.get("comp", 0)),
        "fail_mem": int(instance.failure_reasons.get("mem", 0)),
        "fail_oom": int(instance.failure_reasons.get("oom", 0)),
    }
    return row


def _print_rows(rows: list[dict]) -> None:
    for row in rows:
        print(
            f"{row['policy']:>10} "
            f"kv={row['kv_cache_mem_gb']:>5g}GB "
            f"load={row['load_scale']:<6g} "
            f"offered={row['offered_rps']:.6f}rps "
            f"viol={row['violation_rate']:.3f} "
            f"goodput={row['goodput_rps']:.6f}rps "
            f"rej={row['rejected']} "
            f"fin={row['finished']} "
            f"fail(comp={row['fail_comp']},mem={row['fail_mem']},oom={row['fail_oom']})"
        )


def _write_csv(rows: list[dict], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a baseline vs oracle admission probe on a paired arrival/length trace."
    )
    parser.add_argument("--arrival-pattern", default="azure_chat_23")
    parser.add_argument("--length-pattern", default="deepseek-r1")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--slo-ttft-scale", type=float, default=10.0)
    parser.add_argument("--slo-ttft-constant", type=float, default=0.0)
    parser.add_argument("--slo-tpot", type=float, default=0.05)
    parser.add_argument("--n-req", type=int, default=1000)
    parser.add_argument("--load-scales", type=float, nargs="+", default=[0.05])
    parser.add_argument("--kv-cache-mem-gb", type=float, nargs="+", default=[20.0, 60.0])
    parser.add_argument("--include-thinking-in-output", action="store_true")
    parser.add_argument("--csv-path", default="")
    args = parser.parse_args()

    include_thinking_in_output = (
        args.include_thinking_in_output or args.length_pattern == "deepseek-r1"
    )

    rows: list[dict] = []
    for kv_cache_mem_gb in args.kv_cache_mem_gb:
        for load_scale in args.load_scales:
            for is_oracle in (False, True):
                Instance._printed_kv_cache_info = False
                row = _run_once(
                    arrival_pattern=args.arrival_pattern,
                    length_pattern=args.length_pattern,
                    model_name=args.model_name,
                    slo_ttft_scale=args.slo_ttft_scale,
                    slo_ttft_constant=args.slo_ttft_constant,
                    slo_tpot=args.slo_tpot,
                    n_req=args.n_req,
                    load_scale=load_scale,
                    kv_cache_mem_gb=kv_cache_mem_gb,
                    is_oracle=is_oracle,
                    include_thinking_in_output=include_thinking_in_output,
                )
                rows.append(row)

    _print_rows(rows)
    if args.csv_path:
        _write_csv(rows, args.csv_path)
        print(f"Wrote {args.csv_path}")


if __name__ == "__main__":
    main()
