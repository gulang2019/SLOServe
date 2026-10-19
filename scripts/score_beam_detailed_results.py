#!/usr/bin/env python3
"""Convert BEAM's ``--save-detailed`` result into SLOServe SLO outcomes.

BEAM records client-side per-request measurements in the request order of its
``traced`` CSV.  SLOServe regards a request as successful when its TTFT meets
the scaled zero-load prefill budget and the 90th percentile of its per-token
decode deadline laxities is non-positive.  This utility applies that rule to a
BEAM result without modifying the BEAM submodule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


def percentile_90(values: list[float]) -> float | None:
    """Match numpy.percentile(values, 90) (linear interpolation)."""
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = 0.9 * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def read_trace(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"ContextTokens", "GeneratedTokens"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} is not a BEAM traced CSV (missing {required})")
    return rows


def item(values: list[Any], index: int, default: Any = None) -> Any:
    return values[index] if index < len(values) else default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True,
                        help="BEAM JSON emitted with --save-result --save-detailed")
    parser.add_argument("--trace", type=Path, required=True,
                        help="BEAM traced CSV supplied to the benchmark")
    parser.add_argument("--output", type=Path, required=True,
                        help="per-request JSONL output")
    parser.add_argument("--summary", type=Path,
                        help="aggregate JSON output (default: OUTPUT with .summary.json)")
    parser.add_argument("--ttft-scale", type=float, required=True,
                        help="SLOServe Figure-7 TTFT scale, e.g. 3 or 5")
    parser.add_argument("--zero-load-ttft-per-token", type=float, required=True,
                        help="unscaled SLOServe zero-load prefill slope in seconds/token")
    parser.add_argument("--zero-load-ttft-constant", type=float, required=True,
                        help="unscaled SLOServe zero-load prefill intercept in seconds")
    parser.add_argument("--slo-tpot", type=float, required=True,
                        help="SLOServe TPOT target in seconds/token")
    parser.add_argument("--routing-overhead", type=float, default=0.0,
                        help="additive TTFT allowance in seconds (default: 0)")
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    trace = read_trace(args.trace)
    trace_start = datetime.fromisoformat(
        trace[0].get("TIMESTAMP", trace[0].get("timestamp", "")).replace("Z", "+00:00")
    )
    ttfts = result.get("ttfts", [])
    itls = result.get("itls", [])
    e2el = result.get("e2el", [])
    entry_times = result.get("entry_times", [])
    output_lens = result.get("output_lens", [])
    successes = result.get("successes", [])
    errors = result.get("errors", [])
    count = max(len(ttfts), len(itls), len(e2el), len(entry_times), len(successes))
    if count != len(trace):
        raise ValueError(
            f"result contains {count} requests but trace contains {len(trace)}; "
            "score the exact trace passed to BEAM"
        )

    rows: list[dict[str, Any]] = []
    for index, source in enumerate(trace):
        trace_timestamp = source.get("TIMESTAMP", source.get("timestamp"))
        planned_arrival = (
            datetime.fromisoformat(str(trace_timestamp).replace("Z", "+00:00"))
            - trace_start
        ).total_seconds()
        prompt_tokens = int(source["ContextTokens"])
        target_output_tokens = int(source["GeneratedTokens"])
        budget = args.ttft_scale * (
            args.zero_load_ttft_constant
            + args.zero_load_ttft_per_token * prompt_tokens
        ) + args.routing_overhead
        ttft = item(ttfts, index)
        request_itls = item(itls, index, []) or []
        request_itls = [float(value) for value in request_itls]
        p90_itl = percentile_90(request_itls)
        entry_time = item(entry_times, index)
        first_token_time = (
            None if entry_time is None or ttft is None
            else float(entry_time) + float(ttft)
        )
        token_times = []
        if first_token_time is not None:
            token_times.append(first_token_time)
            for itl in request_itls:
                token_times.append(token_times[-1] + itl)
        success = bool(item(successes, index, False))
        if not success:
            violation = "error"
        elif ttft is None:
            violation = "missing_ttft"
        elif float(ttft) > budget:
            violation = "ttft"
        # This is the same criterion as _evaluate_saved_request_slo(): p90
        # decode-token deadline laxity must be <= 0, i.e. p90(ITL) <= TPOT.
        elif p90_itl is not None and p90_itl > args.slo_tpot:
            violation = "tpot"
        else:
            violation = "none"
        rows.append({
            "request_index": index,
            "trace_timestamp": trace_timestamp,
            "planned_arrival_s": planned_arrival,
            "client_entry_time_s": entry_time,
            "first_token_time_s": first_token_time,
            "token_timestamps_s": token_times,
            "prompt_tokens": prompt_tokens,
            "target_output_tokens": target_output_tokens,
            "actual_output_tokens": item(output_lens, index),
            "ttft_s": ttft,
            "e2e_s": item(e2el, index),
            "itls_s": request_itls,
            "p90_itl_s": p90_itl,
            "ttft_budget_s": budget,
            "slo_tpot_s": args.slo_tpot,
            "ttft_laxity_s": None if ttft is None else float(ttft) - budget,
            "p90_tpot_laxity_s": None if p90_itl is None else p90_itl - args.slo_tpot,
            "success": success,
            "error": item(errors, index),
            "violation": violation,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["violation"]] = counts.get(row["violation"], 0) + 1
    summary = {
        "result": str(args.result), "trace": str(args.trace), "requests": len(rows),
        "slo_satisfied": counts.get("none", 0),
        "slo_satisfaction_rate": counts.get("none", 0) / len(rows) if rows else 0.0,
        "violation_counts": counts,
        "ttft_scale": args.ttft_scale,
        "zero_load_ttft_per_token_s": args.zero_load_ttft_per_token,
        "zero_load_ttft_constant_s": args.zero_load_ttft_constant,
        "slo_tpot_s": args.slo_tpot,
        "routing_overhead_s": args.routing_overhead,
    }
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
