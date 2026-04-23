#!/usr/bin/env python3
"""Extract batch prediction time versus measured execution time from events."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO


CSV_FIELDS = [
    "source_file",
    "batch_id",
    "timestamp_s",
    "device_id",
    "batch_size_reqs",
    "batch_category",
    "total_scheduled_tokens",
    "total_past_tokens",
    "batch_prediction_time_s",
    "prediction_source",
    "estimated_time_s",
    "control_estimated_time_s",
    "batch_execution_time_s",
    "batch_kernel_execution_time_s",
    "batch_total_elapsed_s",
    "scheduling_overhead_s",
    "output_processing_elapsed_s",
    "publish_overhead_s",
    "prediction_minus_execution_s",
    "prediction_over_execution",
    "abs_relative_error",
    "to_launch_s",
    "to_finish_s",
    "to_est_finish_s",
]


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@contextmanager
def _open_event_text(path: Path, zip_member: str | None = None) -> Iterator[TextIO]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            member = zip_member or _default_zip_member(archive)
            with archive.open(member) as raw_file:
                yield io.TextIOWrapper(raw_file, encoding="utf-8")
        return

    with path.open("r", encoding="utf-8") as file_obj:
        yield file_obj


def _default_zip_member(archive: zipfile.ZipFile) -> str:
    names = [name for name in archive.namelist() if not name.endswith("/")]
    for suffix in (".events.jsonl", ".jsonl", ".events.json", ".json"):
        matches = [name for name in names if name.endswith(suffix)]
        if matches:
            return matches[0]
    if len(names) == 1:
        return names[0]
    raise ValueError(
        "Could not infer event file inside zip. Use --zip-member explicitly."
    )


def _normalize_payload(payload: Any) -> Iterator[dict[str, Any]]:
    if isinstance(payload, dict):
        events = payload.get("events")
        if isinstance(events, list):
            for item in events:
                if isinstance(item, dict):
                    yield item
            return
        yield payload
        return

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item


def _iter_events(file_obj: TextIO) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""

    while True:
        chunk = file_obj.read(1024 * 1024)
        if not chunk:
            break
        buffer += chunk
        while True:
            buffer = buffer.lstrip()
            if not buffer:
                break
            if buffer[0] in "[,":
                buffer = buffer[1:]
                continue
            if buffer[0] == "]":
                return
            try:
                payload, offset = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                break
            yield from _normalize_payload(payload)
            buffer = buffer[offset:]

    buffer = buffer.strip()
    if not buffer or buffer == "]":
        return
    if buffer.startswith(","):
        buffer = buffer[1:].lstrip()
    if buffer and buffer != "]":
        payload = json.loads(buffer)
        yield from _normalize_payload(payload)


def _scheduled_tokens_by_req(event: dict[str, Any]) -> list[int]:
    req_ids = event.get("req_ids")
    scheduled_tokens = event.get("num_scheduled_tokens")
    if not isinstance(scheduled_tokens, dict):
        return []

    if isinstance(req_ids, list):
        values: list[int] = []
        for req_id in req_ids:
            raw_value = scheduled_tokens.get(req_id)
            if raw_value is None:
                raw_value = scheduled_tokens.get(str(req_id))
            value = _coerce_int(raw_value)
            if value is not None and value > 0:
                values.append(value)
        return values

    values = []
    for raw_value in scheduled_tokens.values():
        value = _coerce_int(raw_value)
        if value is not None and value > 0:
            values.append(value)
    return values


def _total_past_tokens(event: dict[str, Any]) -> int | None:
    computed_tokens = event.get("num_computed_tokens")
    if not isinstance(computed_tokens, list):
        return None
    total = 0
    for raw_value in computed_tokens:
        value = _coerce_int(raw_value)
        if value is not None and value > 0:
            total += value
    return total


def _batch_category(current_tokens: list[int]) -> str:
    if not current_tokens:
        return "unknown"
    if max(current_tokens) == 1:
        return "decode"
    if min(current_tokens) > 1:
        return "prefill"
    return "mixed"


def _choose_prediction(
    event: dict[str, Any],
    prediction_field: str,
) -> tuple[float | None, str | None]:
    estimated = _coerce_float(event.get("estimated_time"))
    control_estimated = _coerce_float(event.get("control_estimated_time"))
    fields = {
        "estimated_time": estimated,
        "control_estimated_time": control_estimated,
    }

    if prediction_field == "auto":
        if control_estimated is not None:
            return control_estimated, "control_estimated_time"
        if estimated is not None:
            return estimated, "estimated_time"
        return None, None

    prediction = fields[prediction_field]
    if prediction is not None:
        return prediction, prediction_field

    fallback_field = (
        "control_estimated_time"
        if prediction_field == "estimated_time"
        else "estimated_time"
    )
    fallback = fields[fallback_field]
    if fallback is not None:
        return fallback, f"{fallback_field}_fallback"
    return None, None


def _event_to_row(
    event: dict[str, Any],
    source_file: str,
    prediction_field: str,
) -> dict[str, Any] | None:
    if event.get("event_type") != "batch":
        return None

    elapsed = _coerce_float(event.get("elapsed"))
    scheduling_overhead = _coerce_float(event.get("scheduling_overhead"))
    if elapsed is None:
        return None

    execution_time = elapsed
    if scheduling_overhead is not None:
        execution_time = max(0.0, elapsed - scheduling_overhead)

    extra_args = event.get("extra_args")
    if not isinstance(extra_args, dict):
        extra_args = {}
    to_launch = _coerce_float(extra_args.get("to_launch"))
    to_finish = _coerce_float(extra_args.get("to_finish"))
    to_est_finish = _coerce_float(extra_args.get("to_est_finish"))
    kernel_execution_time = None
    if to_launch is not None and to_finish is not None:
        kernel_execution_time = max(0.0, to_finish - to_launch)

    prediction_time, prediction_source = _choose_prediction(event, prediction_field)
    prediction_minus_execution = None
    prediction_over_execution = None
    abs_relative_error = None
    if prediction_time is not None:
        prediction_minus_execution = prediction_time - execution_time
        if execution_time > 0:
            prediction_over_execution = prediction_time / execution_time
            abs_relative_error = abs(prediction_minus_execution / execution_time)

    current_tokens = _scheduled_tokens_by_req(event)
    req_ids = event.get("req_ids")
    batch_size = len(req_ids) if isinstance(req_ids, list) else len(current_tokens)

    return {
        "source_file": source_file,
        "batch_id": event.get("batch_id"),
        "timestamp_s": _coerce_float(event.get("timestamp")),
        "device_id": event.get("device_id"),
        "batch_size_reqs": batch_size,
        "batch_category": _batch_category(current_tokens),
        "total_scheduled_tokens": sum(current_tokens) if current_tokens else None,
        "total_past_tokens": _total_past_tokens(event),
        "batch_prediction_time_s": prediction_time,
        "prediction_source": prediction_source,
        "estimated_time_s": _coerce_float(event.get("estimated_time")),
        "control_estimated_time_s": _coerce_float(event.get("control_estimated_time")),
        "batch_execution_time_s": execution_time,
        "batch_kernel_execution_time_s": kernel_execution_time,
        "batch_total_elapsed_s": elapsed,
        "scheduling_overhead_s": scheduling_overhead,
        "output_processing_elapsed_s": _coerce_float(
            event.get("output_processing_elapsed")
        ),
        "publish_overhead_s": _coerce_float(event.get("publish_overhead")),
        "prediction_minus_execution_s": prediction_minus_execution,
        "prediction_over_execution": prediction_over_execution,
        "abs_relative_error": abs_relative_error,
        "to_launch_s": to_launch,
        "to_finish_s": to_finish,
        "to_est_finish_s": to_est_finish,
    }


def _default_output_path(input_path: Path) -> Path:
    name = input_path.name
    for suffix in (".events.jsonl", ".events.json", ".jsonl", ".json", ".zip"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return input_path.with_name(f"{name}.batch_prediction_vs_execution.csv")


def extract_csv(
    input_path: Path,
    output_path: Path,
    *,
    prediction_field: str,
    zip_member: str | None,
) -> int:
    row_count = 0
    with _open_event_text(input_path, zip_member=zip_member) as event_file:
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for event in _iter_events(event_file):
                row = _event_to_row(
                    event,
                    source_file=str(input_path),
                    prediction_field=prediction_field,
                )
                if row is None:
                    continue
                writer.writerow(row)
                row_count += 1
    return row_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-batch prediction and measured execution timing from an "
            "events JSON/JSONL trace."
        )
    )
    parser.add_argument("events_file", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--prediction-field",
        choices=("estimated_time", "control_estimated_time", "auto"),
        default="estimated_time",
        help=(
            "Field to use as batch_prediction_time_s. The default matches the "
            "runtime event's estimated_time field."
        ),
    )
    parser.add_argument(
        "--zip-member",
        help="Event file member to read when events_file is a .zip archive.",
    )
    args = parser.parse_args()

    input_path = args.events_file.expanduser()
    output_path = (
        args.output.expanduser()
        if args.output is not None
        else _default_output_path(input_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = extract_csv(
        input_path,
        output_path,
        prediction_field=args.prediction_field,
        zip_member=args.zip_member,
    )
    print(f"Wrote {row_count} batch rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
