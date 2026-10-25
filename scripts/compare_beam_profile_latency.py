#!/usr/bin/env python3
"""Compare measured batch times with a BEAM profile at one locked clock.

Run any non-BEAM policy (e.g. round_robin:vllm) with the GPU locked at
--clock (``sudo nvidia-smi -i <gpu> -lgc <clock>``), then pass its
``*.events.jsonl``. For each batch, the measured execution time
(extra_args.to_finish - to_launch) is compared with the profile's ground
truth (BeamProfile.simulate) and with BEAM's own decision estimate
(get_time_taken, which rounds batch sizes up), split by batch kind.

A ratio near 1 means BEAM's decisions on this engine use the right latency
model; a large gap means the profile came from a different engine speed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SLOsServe.beam import BeamProfile  # noqa: E402


def quantiles(values: list[float]) -> str:
    values = sorted(values)
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]
    return (f"n={len(values):6d}  p10={pick(0.1):.3f}  p50={pick(0.5):.3f}  "
            f"p90={pick(0.9):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("events", type=Path, nargs="+")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--clock", type=int, required=True,
                        help="clock (MHz) the GPU was locked at during the run")
    args = parser.parse_args()

    profile = BeamProfile(str(args.profile))
    rows = {"decode-only": [], "with prefill": []}
    for path in args.events:
        for event in json.loads(path.read_text()):
            if event.get("event_type") != "batch":
                continue
            extra = event.get("extra_args") or {}
            if "to_finish" not in extra or "to_launch" not in extra:
                continue
            scheduled = list(event["num_scheduled_tokens"].values())
            tokens = sum(scheduled)
            if tokens <= 0 or tokens > profile.batch_sizes[-1]:
                continue
            measured = float(extra["to_finish"]) - float(extra["to_launch"])
            if measured <= 0:
                continue
            ctx = sum(event["num_computed_tokens"])
            truth, _ = profile.simulate(tokens, ctx, args.clock)
            has_prefill = any(n > 1 for n in scheduled)
            decision = profile.get_time_taken(tokens, args.clock,
                                              None if has_prefill else ctx)
            rows["with prefill" if has_prefill else "decode-only"].append(
                (measured, truth, decision))

    if not any(rows.values()):
        raise SystemExit("No usable batch events (need extra_args.to_launch/to_finish).")
    print(f"profile={profile.path}\nclock={args.clock} MHz")
    for kind, samples in rows.items():
        if not samples:
            continue
        print(f"\n[{kind}]")
        print("  measured / profile          ", quantiles([m / t for m, t, _ in samples]))
        print("  measured / BEAM estimate    ", quantiles([m / d for m, _, d in samples]))
        print("  measured - profile (ms)     ",
              f"median={1000 * statistics.median(m - t for m, t, _ in samples):.2f}")


if __name__ == "__main__":
    main()
