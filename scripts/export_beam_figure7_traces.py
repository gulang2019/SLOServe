#!/usr/bin/env python3
"""Export Figure 7 trace inputs in BEAM's ``traced`` CSV format.

Each output retains the arrival pattern selected by ``--window t1200:1800``
and pairs it with the same request-length source used by SLOServe.  BEAM's
traced input has no session-dependency field, so the multi-round export is a
timestamp-preserving flattened replay; it is not session-aware.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Dataset.dataset import ArrivalTimes, Requests  # noqa: E402


WINDOW_START = "t1200"
WINDOW_END = "1800"
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class TraceComponent:
    request_source: str
    arrival_source: str


@dataclass(frozen=True)
class BeamTrace:
    filename: str
    components: tuple[TraceComponent, ...]
    note: str
    # SLOServe excludes a request when request.total_length + 1000 exceeds
    # the model context limit. Qwen2.5-7B uses a 24K-token limit here.
    max_request_total_tokens: int | None = None


FIGURE7_TRACES = (
    BeamTrace(
        "sloserve_fig7_chat_7b_t1200_1800.csv",
        (TraceComponent("azure_chat_23", "azure_chat_23"),),
        "ChatBot (7B): AzureChat23 lengths and arrivals.",
        23000,
    ),
    BeamTrace(
        "sloserve_fig7_code_7b_t1200_1800.csv",
        (TraceComponent("azure_code_23", "azure_code_23"),),
        "Coder (7B): AzureCode23 lengths and arrivals.",
        23000,
    ),
    BeamTrace(
        "sloserve_fig7_mixed_7b_t1200_1800.csv",
        (
            TraceComponent("azure_chat_23", "azure_chat_23"),
            TraceComponent("azure_code_23", "azure_code_23"),
        ),
        "Mixed (7B): timestamp-merged AzureChat23 and AzureCode23 streams.",
        23000,
    ),
    BeamTrace(
        "sloserve_fig7_chat_32b_t1200_1800.csv",
        (TraceComponent("azure_chat_23", "azure_chat_23"),),
        "ChatBot (32B): AzureChat23 lengths and arrivals.",
    ),
    BeamTrace(
        "sloserve_fig7_multiround_chat_t1200_1800_flattened.csv",
        (TraceComponent("sharegpt_chat", "azure_chat_23"),),
        "Multi-Round ChatBot: ShareGPT lengths with AzureChat23 arrivals; flattened for BEAM.",
        23000,
    ),
    BeamTrace(
        "sloserve_fig7_large_scale_chat_t1200_1800.csv",
        (TraceComponent("azure_chat", "azure_chat"),),
        "Large-Scale Chat: AzureChat lengths and arrivals.",
        23000,
    ),
)


def load_component(
    component: TraceComponent, max_request_total_tokens: int | None
) -> list[tuple[float, int, int]]:
    arrivals = ArrivalTimes.load(
        component.arrival_source,
        load_scale=1.0,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
    ).arrival_times
    requests = Requests.load(component.request_source).requests
    if max_request_total_tokens is not None:
        requests = [
            request
            for request in requests
            if request.total_length <= max_request_total_tokens
        ]
    count = min(len(arrivals), len(requests))
    return [
        (
            float(arrivals[index]),
            int(requests[index].input_length),
            int(requests[index].output_length),
        )
        for index in range(count)
    ]


def write_trace(spec: BeamTrace, output_dir: Path) -> tuple[Path, int]:
    rows = [
        row
        for component in spec.components
        for row in load_component(component, spec.max_request_total_tokens)
    ]
    rows.sort(key=lambda row: row[0])
    first_arrival = rows[0][0]
    path = output_dir / spec.filename
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("TIMESTAMP", "ContextTokens", "GeneratedTokens", "timestamp"))
        for arrival_s, input_length, output_length in rows:
            timestamp = EPOCH + timedelta(seconds=arrival_s - first_arrival)
            formatted = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "+0000"
            writer.writerow((formatted, input_length, output_length, formatted))
    return path, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "3rdparty/BEAM/benchmarks/energy/datasets",
        help="Directory for BEAM traced CSVs (default: %(default)s)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for spec in FIGURE7_TRACES:
        path, count = write_trace(spec, args.output_dir)
        print(f"{path}: {count} requests ({spec.note})")


if __name__ == "__main__":
    main()
