#!/usr/bin/env python3
"""Merge sliced round-robin runs of the same setting into one point.

A run with --n_devices N on k GPUs replays shards offset .. offset+k-1 of
the N-way split and reports energy x N/k. Runs at different offsets of the
same setting are merged here: raw energies are summed and scaled by
N / (shards covered), so a complete cover gives the measured total;
violation rates are weighted by each run's request count.

Unsliced runs pass through unchanged. Overlapping shards are an error.

    python scripts/merge_beam_shard_runs.py runs/a100sxm_fig7b/*/results.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

KEY = ("trace_used", "scheduling_policy", "routing_policy",
       "requested_n_device", "ttft_slo_scale", "slo_tpot",
       "slo_routing_overhead", "tensor_parallel_size")
FIELDS = list(KEY) + ["shards_covered", "complete", "offsets", "n_requests",
                      "slo_violation_rate", "energy_consumption"]


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["_source"] = str(path)
                rows.append(row)
    return rows


def merge(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        row.setdefault("requested_n_device", row["n_device"])
        groups[tuple(row.get(k) for k in KEY)].append(row)

    merged = []
    for key, members in groups.items():
        n = int(key[KEY.index("requested_n_device")])
        shards: set[int] = set()
        energy = violations = requests = 0.0
        for row in members:
            k = int(row.get("effective_n_device") or row["n_device"])
            offset = int(row.get("rr_shard_offset") or 0)
            if row.get("rr_sliced") and "rr_shard_offset" not in row:
                print(f"warning: {row['_source']} predates rr_shard_offset; "
                      "assuming offset 0", file=sys.stderr)
            covered = set(range(offset, offset + k))
            if shards & covered:
                raise SystemExit(f"shards {sorted(shards & covered)} of {key} "
                                 f"appear in more than one run")
            shards |= covered
            count = row.get("rr_slice_kept_request_count") or 1
            energy += float(row["energy_consumption"]) * k / n
            violations += float(row["slo_violation_rate"]) * count
            requests += count
        merged.append(dict(zip(KEY, key)) | {
            "shards_covered": len(shards),
            "complete": len(shards) == n,
            "offsets": " ".join(str(o) for o in sorted(
                int(r.get("rr_shard_offset") or 0) for r in members)),
            "n_requests": int(requests),
            "slo_violation_rate": violations / requests,
            "energy_consumption": energy * n / len(shards),
        })
    return sorted(merged, key=lambda r: tuple(str(r[k]) for k in KEY))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="+",
                        help="results.jsonl files")
    parser.add_argument("--output", type=Path, help="write the table as CSV")
    args = parser.parse_args()
    merged = merge(load_rows(args.results))
    out = open(args.output, "w", newline="") if args.output else sys.stdout
    writer = csv.DictWriter(out, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(merged)
    if args.output:
        out.close()


if __name__ == "__main__":
    main()
