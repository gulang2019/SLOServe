#!/usr/bin/env python3
"""Tabulate SLOServe attainment and energy for bench_beam.sh runs.

Every directory under ROOT that holds per_request_slo.summary.json is one
run; its BEAM control targets come from slo_config.txt and its energy from
power_timeline_gpu_*.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_config(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--csv", type=Path, help="also write the table as CSV")
    args = parser.parse_args()

    rows = []
    for summary_path in sorted(args.root.rglob("per_request_slo.summary.json")):
        run = summary_path.parent
        summary = json.loads(summary_path.read_text())
        config = read_config(run / "slo_config.txt")
        result = json.loads((run / "beam_detailed.json").read_text())
        energy_j = mean_clock = None
        timelines = sorted(run.glob("power_timeline_gpu_*.csv"))
        if timelines:
            with timelines[0].open(newline="") as handle:
                samples = list(csv.DictReader(handle))
            energy_j = (float(samples[-1]["cumulative_energy_mj"])
                        - float(samples[0]["cumulative_energy_mj"])) / 1000
            mean_clock = sum(float(s["graphics_clock_mhz"]) for s in samples) / len(samples)
        counts = summary["violation_counts"]
        tokens = result.get("total_output_tokens") or 0
        rows.append({
            "run": str(run.relative_to(args.root)),
            "beam_ttft_s": float(config["beam_global_ttft_s"]),
            "beam_tbt_s": float(config.get("beam_tbt_s", config["slo_tpot_s"])),
            "requests": summary["requests"],
            "attainment": summary["slo_satisfaction_rate"],
            "ttft_viol": counts.get("ttft", 0),
            "tpot_viol": counts.get("tpot", 0),
            "errors": counts.get("error", 0),
            "energy_j": energy_j,
            "mj_per_token": energy_j * 1000 / tokens if energy_j and tokens else None,
            "mean_clock_mhz": mean_clock,
        })

    rows.sort(key=lambda r: (-r["attainment"], r["energy_j"] or 0))
    header = f"{'run':<40} {'TTFT_s':>7} {'TBT_s':>6} {'attain':>7} {'ttftV':>6} {'tpotV':>6} {'err':>4} {'energy_J':>9} {'mJ/tok':>7} {'MHz':>5}"
    print(header)
    for r in rows:
        print(f"{r['run']:<40} {r['beam_ttft_s']:>7.3f} {r['beam_tbt_s']:>6.3f} "
              f"{r['attainment']:>7.1%} {r['ttft_viol']:>6} {r['tpot_viol']:>6} {r['errors']:>4} "
              f"{r['energy_j'] or 0:>9.0f} {r['mj_per_token'] or 0:>7.0f} {r['mean_clock_mhz'] or 0:>5.0f}")
    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
