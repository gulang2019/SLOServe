#!/usr/bin/env python3
"""Split Figure 7 BEAM traces into deterministic round-robin server loads.

The output layout is ``<output-dir>/<experiment-name>/<server-count>/<id>.csv``.
Each CSV retains global timestamps, so a multi-server replay coordinator can
preserve the original inter-server timing relationship.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "3rdparty/BEAM/benchmarks/energy/datasets"
DEFAULT_OUTPUT_DIR = DATASET_DIR / "sloserve_fig7_round_robin"

# These are the server-count sweeps present in the Figure 7 input CSVs.
EXPERIMENTS: dict[str, tuple[str, tuple[int, ...]]] = {
    "chat_7b": ("sloserve_fig7_chat_7b_t1200_1800.csv", tuple(range(1, 9))),
    "code_7b": (
        "sloserve_fig7_code_7b_t1200_1800.csv",
        tuple(range(1, 9)) + (10, 12, 14, 16),
    ),
    "mixed_7b": (
        "sloserve_fig7_mixed_7b_t1200_1800.csv",
        tuple(range(1, 9)) + (10, 12, 14, 16),
    ),
    "chat_32b": (
        "sloserve_fig7_chat_32b_t1200_1800.csv",
        tuple(range(2, 9)),
    ),
    "multiround_chat": (
        "sloserve_fig7_multiround_chat_t1200_1800_flattened.csv",
        (8, 12, 16),
    ),
    "large_scale_chat": (
        "sloserve_fig7_large_scale_chat_t1200_1800.csv",
        (4, 6, 8, 10, 12, 14, 16),
    ),
}


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")
        required = {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} is missing required BEAM traced columns")
        return reader.fieldnames, list(reader)


def write_split(
    output_dir: Path,
    experiment_name: str,
    server_count: int,
    header: list[str],
    rows: list[dict[str, str]],
) -> None:
    server_dir = output_dir / experiment_name / str(server_count)
    server_dir.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        writers = []
        for server_id in range(server_count):
            handle = (server_dir / f"{server_id}.csv").open(
                "w", newline="", encoding="utf-8"
            )
            handles.append(handle)
            writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
            writer.writeheader()
            writers.append(writer)
        for request_index, row in enumerate(rows):
            writers[request_index % server_count].writerow(row)
    finally:
        for handle in handles:
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    for experiment_name, (source_name, server_counts) in EXPERIMENTS.items():
        source_path = DATASET_DIR / source_name
        header, rows = load_rows(source_path)
        for server_count in server_counts:
            write_split(args.output_dir, experiment_name, server_count, header, rows)
            counts = [len(rows) // server_count] * server_count
            for server_id in range(len(rows) % server_count):
                counts[server_id] += 1
            print(
                f"{experiment_name}/{server_count}: {len(rows)} requests, "
                f"per-server=[{min(counts)}, {max(counts)}]"
            )


if __name__ == "__main__":
    main()
