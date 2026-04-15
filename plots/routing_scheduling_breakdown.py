#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import zipfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime.
    tqdm = None


class _NullProgressBar:
    def __enter__(self) -> "_NullProgressBar":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def update(self, _: int | float = 1) -> None:
        return

    def close(self) -> None:
        return


def _progress_enabled() -> bool:
    return tqdm is not None and sys.stderr.isatty()


def _iter_progress(
    iterable: Any,
    *,
    desc: str,
    total: int | None = None,
    unit: str = "item",
) -> Any:
    if not _progress_enabled():
        return iterable
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        dynamic_ncols=True,
        leave=False,
    )


def _progress_bar(
    *,
    desc: str,
    total: int | None = None,
    unit: str = "item",
    unit_scale: bool = False,
) -> Any:
    if not _progress_enabled():
        return _NullProgressBar()
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        unit_scale=unit_scale,
        dynamic_ncols=True,
        leave=False,
    )


def _log_phase(message: str, *, run_start_s: float) -> None:
    elapsed_s = time.monotonic() - run_start_s
    print(f"[phase +{elapsed_s:8.2f}s] {message}", file=sys.stderr, flush=True)


@contextmanager
def _phase(name: str, *, run_start_s: float) -> Any:
    phase_start_s = time.monotonic()
    _log_phase(f"START {name}", run_start_s=run_start_s)
    try:
        yield
    except Exception:
        phase_elapsed_s = time.monotonic() - phase_start_s
        _log_phase(
            f"FAIL  {name} after {phase_elapsed_s:.2f}s",
            run_start_s=run_start_s,
        )
        raise
    phase_elapsed_s = time.monotonic() - phase_start_s
    _log_phase(
        f"END   {name} ({phase_elapsed_s:.2f}s)",
        run_start_s=run_start_s,
    )


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.lstrip().startswith("["):
        rows = json.loads(raw)
    else:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise ValueError(f"expected a list of JSON rows in {path}")
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(
                f"expected every row in {path} to be a JSON object, "
                f"got {type(row).__name__}"
            )
        result.append(row)
    return result


def _strip_known_analysis_suffix(path: Path) -> Path:
    path_str = str(path)
    for suffix in (
        ".events.summary.json",
        ".summary.json",
        ".events.breakdown.csv",
        ".breakdown.csv",
        ".events.tries_distribution.csv",
        ".tries_distribution.csv",
        ".slo_categories.csv",
        ".events.zip",
        ".zip",
        ".events.jsonl",
        ".events.json",
        ".reqs.jsonl",
        ".reqs.json",
    ):
        if path_str.endswith(suffix):
            return Path(path_str[: -len(suffix)])
    return path


def _resolve_existing_companion(
    prefix: Path | None,
    suffixes: tuple[str, ...],
) -> Path | None:
    if prefix is None:
        return None
    for suffix in suffixes:
        candidate = Path(f"{prefix}{suffix}")
        if candidate.exists():
            return candidate
    return None


def _infer_slo_params_from_path(path: Path | None) -> tuple[float | None, float | None]:
    if path is None:
        return (None, None)
    match = re.search(
        r"_arrival_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)(?:_|$)",
        path.name,
    )
    if match is None:
        return (None, None)
    return (float(match.group(1)), float(match.group(2)))


def _pct(values: list[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * percentile
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return ordered[low]
    weight = pos - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    mean = sum(values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "std": (
            0.0
            if len(values) == 1
            else math.sqrt(
                sum((value - mean) ** 2 for value in values) / len(values)
            )
        ),
        "min": min(values),
        "p50": _pct(values, 0.50),
        "p90": _pct(values, 0.90),
        "p95": _pct(values, 0.95),
        "p99": _pct(values, 0.99),
        "max": max(values),
    }


def _build_request_prefill_tokens(
    events: list[dict[str, Any]]
) -> dict[str, int]:
    prefill_tokens_by_request: dict[str, int] = {}
    for event in _iter_progress(
        events,
        desc="Indexing prefill tokens",
        total=len(events),
        unit="event",
    ):
        if str(event.get("event_type", "")) != "arrival":
            continue
        request_id_raw = event.get("request_id")
        if request_id_raw is None:
            continue
        prompt_tokens = _coerce_int(event.get("prompt_tokens"))
        if prompt_tokens is None:
            continue
        num_cached_tokens = _coerce_int(event.get("num_cached_tokens")) or 0
        prefill_tokens_by_request[str(request_id_raw)] = max(
            prompt_tokens - num_cached_tokens,
            0,
        )
    return prefill_tokens_by_request


def _classify_batch_type(
    event: dict[str, Any],
    prefill_tokens_by_request: dict[str, int],
) -> str:
    req_ids = [str(req_id) for req_id in (event.get("req_ids") or [])]
    if not req_ids:
        return "unknown"

    computed_tokens_raw = list(event.get("num_computed_tokens") or [])
    if len(computed_tokens_raw) < len(req_ids):
        computed_tokens_raw.extend([None] * (len(req_ids) - len(computed_tokens_raw)))

    scheduled_tokens_by_request = event.get("num_scheduled_tokens") or {}
    stages: list[str] = []
    for req_id, num_computed_raw in zip(req_ids, computed_tokens_raw):
        num_computed_tokens = _coerce_int(num_computed_raw)
        prefill_tokens = prefill_tokens_by_request.get(req_id)
        if num_computed_tokens is not None and prefill_tokens is not None:
            stages.append(
                "prefill" if num_computed_tokens < prefill_tokens else "decode"
            )
            continue

        scheduled_tokens = _coerce_int(scheduled_tokens_by_request.get(req_id))
        if scheduled_tokens is None:
            return "unknown"
        stages.append("prefill" if scheduled_tokens > 1 else "decode")

    unique_stages = set(stages)
    if unique_stages == {"prefill"}:
        return "prefill_only"
    if unique_stages == {"decode"}:
        return "decode_only"
    if unique_stages == {"prefill", "decode"}:
        return "mixed"
    return "unknown"


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 1000.0:9.3f}"


def _duration(end: float | None, start: float | None) -> float | None:
    if end is None or start is None:
        return None
    return end - start


def load_events(
    path: Path,
    *,
    allowed_event_types: set[str] | None = None,
) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        first_non_ws = None
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_non_ws = ch
                break
        f.seek(0)
        if first_non_ws == "[":
            payload = json.load(f)
            if not isinstance(payload, list):
                raise ValueError(f"{path} does not contain a JSON array")
            events: list[dict[str, Any]] = []
            for event in _iter_progress(
                payload,
                desc="Filtering loaded events",
                total=len(payload),
                unit="event",
            ):
                if not isinstance(event, dict):
                    continue
                if (
                    allowed_event_types is not None
                    and str(event.get("event_type", "")) not in allowed_event_types
                ):
                    continue
                events.append(event)
            return events

        events: list[dict[str, Any]] = []
        total_bytes = path.stat().st_size
        with _progress_bar(
            desc=f"Loading {path.name}",
            total=total_bytes,
            unit="B",
            unit_scale=True,
        ) as progress:
            for line_no, raw_line in enumerate(f, start=1):
                progress.update(len(raw_line))
                line = raw_line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(
                        f"{path}:{line_no} is not a JSON object: {type(item).__name__}"
                    )
                if (
                    allowed_event_types is not None
                    and str(item.get("event_type", "")) not in allowed_event_types
                ):
                    continue
                events.append(item)
        return events


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object summary")
    return payload


def _build_tpot_deadline_offsets(
    output_tokens: int,
    thinking_length: int,
) -> list[int]:
    output_tokens = max(int(output_tokens), 0)
    thinking_length = max(int(thinking_length), 0)
    if output_tokens <= 0:
        return []
    if thinking_length > 0:
        return list(range(thinking_length, thinking_length + output_tokens))
    return list(range(1, output_tokens))


def _evaluate_saved_request_slo(
    req_row: dict[str, Any],
    *,
    ttft_budget_s: float,
    slo_tpot: float,
    routing_overhead: float = -1.0,
    lag_cutoff_s: float = 0.10,
) -> dict[str, Any]:
    if routing_overhead < 0.0:
        arrival_time = float(req_row.get("arrival_time", 0.0))
    else:
        arrival_time = (
            float(req_row.get("engine_arrival_time", req_row.get("arrival_time", 0.0)))
            + float(routing_overhead)
        )

    prompt_tokens = int(req_row.get("prompt_tokens", 0))
    cached_tokens = int(req_row.get("num_cached_tokens", 0))
    output_tokens = max(int(req_row.get("output_tokens", 0)), 0)
    thinking_length = max(int(req_row.get("thinking_length", 0) or 0), 0)
    uncached_prompt_tokens = max(prompt_tokens - cached_tokens, 0)

    corrected_schedules: list[tuple[float, int]] = []
    lag = 0.0
    prev_timestamp: float | None = None
    for schedule in req_row.get("schedules", []):
        if not isinstance(schedule, dict):
            continue
        timestamp = float(schedule.get("timestamp", arrival_time)) - lag
        if (
            prev_timestamp is not None
            and prev_timestamp + float(lag_cutoff_s) < timestamp
        ):
            delta = timestamp - prev_timestamp - float(lag_cutoff_s)
            lag += delta
            timestamp -= delta
        corrected_schedules.append(
            (
                timestamp,
                int(schedule.get("num_scheduled_tokens", 0)),
            )
        )
        prev_timestamp = timestamp

    tpot_deadline_offsets = _build_tpot_deadline_offsets(
        output_tokens,
        thinking_length,
    )
    required_tokens = [(arrival_time + float(ttft_budget_s), uncached_prompt_tokens)]
    for offset in tpot_deadline_offsets:
        required_tokens.append(
            (
                arrival_time + float(ttft_budget_s) + float(offset) * float(slo_tpot),
                1,
            )
        )

    schedule_idx = 0
    available_tokens = 0
    completion_timestamp = arrival_time
    ttft_laxity = 0.0
    tpot_laxities: list[float] = []

    for required_idx, (deadline, required_token_count) in enumerate(required_tokens):
        while (
            schedule_idx < len(corrected_schedules)
            and available_tokens < required_token_count
        ):
            completion_timestamp, num_scheduled_tokens = corrected_schedules[schedule_idx]
            available_tokens += num_scheduled_tokens
            schedule_idx += 1
        available_tokens -= required_token_count
        laxity = float(completion_timestamp - deadline)
        if required_idx == 0:
            ttft_laxity = laxity
        else:
            tpot_laxities.append(laxity)

    finish_reason = req_row.get("finish_reason")
    if finish_reason != "length":
        violation = str(finish_reason or "unfinished")
    elif ttft_laxity > 0:
        violation = "ttft"
    elif tpot_laxities and _pct(tpot_laxities, 0.90) > 0:
        violation = "tpot"
    else:
        violation = "none"

    return {
        "req_id": req_row.get("req_id"),
        "finish_reason": finish_reason,
        "violation": violation,
        "ttft_laxity": float(ttft_laxity),
        "max_tpot_laxity": float(max(tpot_laxities, default=0.0)),
    }


def _build_reason_rows(
    counts: dict[str, int],
    *,
    total: int,
    scope: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category in sorted(counts):
        count = int(counts[category])
        rows.append(
            {
                "scope": scope,
                "category": category,
                "request_count": count,
                "fraction_of_requests": (0.0 if total == 0 else count / total),
            }
        )
    return rows


def analyze_saved_requests(
    reqs_file: Path,
    *,
    ttft_slo_scale: float,
    slo_tpot: float,
    ttft_overhead: float = 0.0,
    routing_overhead: float = -1.0,
) -> dict[str, Any]:
    req_rows = _load_json_rows(reqs_file)
    violation_counts: Counter[str] = Counter()
    finish_reason_counts: Counter[str] = Counter()
    ttft_budgets: list[float] = []
    ttft_laxities: list[float] = []
    max_tpot_laxities: list[float] = []

    for req_row in req_rows:
        ttft_budget_s = (
            float(req_row.get("zero_load_ttft", 0.0)) * float(ttft_slo_scale)
            + float(ttft_overhead)
        )
        evaluation = _evaluate_saved_request_slo(
            req_row,
            ttft_budget_s=ttft_budget_s,
            slo_tpot=float(slo_tpot),
            routing_overhead=float(routing_overhead),
        )
        violation_counts[str(evaluation["violation"])] += 1
        finish_reason_counts[str(req_row.get("finish_reason") or "unfinished")] += 1
        ttft_budgets.append(ttft_budget_s)
        ttft_laxities.append(float(evaluation["ttft_laxity"]))
        max_tpot_laxities.append(float(evaluation["max_tpot_laxity"]))

    total_requests = len(req_rows)
    attained = int(violation_counts.get("none", 0))
    non_terminal_reasons = {
        reason: count
        for reason, count in violation_counts.items()
        if reason not in {"none", "ttft", "tpot"}
    }
    category_rows = _build_reason_rows(
        dict(violation_counts),
        total=total_requests,
        scope="violation_reason",
    )
    category_rows.extend(
        _build_reason_rows(
            dict(finish_reason_counts),
            total=total_requests,
            scope="finish_reason",
        )
    )

    return {
        "reqs_file": str(reqs_file),
        "ttft_mode": "scale",
        "ttft_slo_scale": float(ttft_slo_scale),
        "slo_tpot": float(slo_tpot),
        "ttft_overhead": float(ttft_overhead),
        "routing_overhead": float(routing_overhead),
        "total_requests": total_requests,
        "slo_attainment_rate": (
            float(attained / total_requests) if total_requests else 0.0
        ),
        "slo_violation_rate": (
            float(1.0 - attained / total_requests) if total_requests else 0.0
        ),
        "ttft_violation_rate": (
            float(violation_counts.get("ttft", 0) / total_requests)
            if total_requests
            else 0.0
        ),
        "tpot_violation_rate": (
            float(violation_counts.get("tpot", 0) / total_requests)
            if total_requests
            else 0.0
        ),
        "unfinished_or_rejected_rate": (
            float(sum(non_terminal_reasons.values()) / total_requests)
            if total_requests
            else 0.0
        ),
        "violation_reason_counts": dict(sorted(violation_counts.items())),
        "violation_reason_rates": (
            {
                reason: float(count / total_requests)
                for reason, count in sorted(violation_counts.items())
            }
            if total_requests
            else {}
        ),
        "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
        "finish_reason_rates": (
            {
                reason: float(count / total_requests)
                for reason, count in sorted(finish_reason_counts.items())
            }
            if total_requests
            else {}
        ),
        "ttft_budget_s_summary": _stats(ttft_budgets),
        "ttft_laxity_summary": _stats(ttft_laxities),
        "max_tpot_laxity_summary": _stats(max_tpot_laxities),
        "category_rows": category_rows,
    }


@dataclass
class Attempt:
    request_id: str
    attempt_index: int
    start_reason: str
    router_entry_ts: float | None = None
    routing_ts: float | None = None
    routing_iter: int | None = None
    routing_overhead_s: float | None = None
    routing_waiting_time_s: float | None = None
    routing_compute_time_s: float | None = None
    routing_start_ts: float | None = None
    router_decision_ts: float | None = None
    dispatch_ts: float | None = None
    dispatch_type: str | None = None
    target_prefill_device_id: int | None = None
    target_decode_device_id: int | None = None
    engine_arrival_ts: float | None = None
    engine_device_id: int | None = None
    engine_add_request_ts: float | None = None
    engine_add_request_elapsed_s: float | None = None
    engine_admitted: bool | None = None
    reject_finish_ts: float | None = None
    reject_finish_reason: str | None = None
    reject_finish_rejection_reason: str | None = None
    reject_finish_scheduling_overhead_s: float | None = None
    router_ack_ts: float | None = None
    router_ack_type: str | None = None
    rescheduling_ts: float | None = None
    service_batch_id: int | None = None
    service_batch_device_id: int | None = None
    first_batch_ts: float | None = None
    first_batch_elapsed_s: float | None = None
    first_batch_scheduling_overhead_s: float | None = None
    first_batch_to_launch_s: float | None = None
    service_launch_ts_actual: float | None = None
    service_ready_ts: float | None = None
    finish_ts: float | None = None
    finish_reason: str | None = None
    terminal_status: str | None = None

    def refresh_service_ready(self) -> None:
        if self.service_launch_ts_actual is None:
            return
        if self.router_ack_ts is None:
            self.service_ready_ts = self.service_launch_ts_actual
            return
        self.service_ready_ts = max(self.service_launch_ts_actual, self.router_ack_ts)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.update(
            {
                "router_entry_to_routing_s": _duration(
                    self.routing_ts, self.router_entry_ts
                ),
                "queueing_time_s": _duration(
                    self.routing_start_ts, self.router_entry_ts
                ),
                "routing_compute_time_s": self.routing_compute_time_s,
                "routing_to_dispatch_s": _duration(self.dispatch_ts, self.routing_ts),
                "dispatch_to_engine_arrival_s": _duration(
                    self.engine_arrival_ts, self.dispatch_ts
                ),
                "engine_arrival_to_add_request_s": _duration(
                    self.engine_add_request_ts, self.engine_arrival_ts
                ),
                "feasibility_check_s": _duration(
                    self.engine_add_request_ts, self.engine_arrival_ts
                ),
                "engine_add_request_to_router_ack_s": _duration(
                    self.router_ack_ts, self.engine_add_request_ts
                ),
                "route_back_to_server_s": _duration(
                    self.router_ack_ts, self.engine_add_request_ts
                ),
                "add_request_to_server_s": _duration(
                    self.router_ack_ts, self.engine_add_request_ts
                ),
                "reject_add_request_to_finish_s": _duration(
                    self.reject_finish_ts, self.engine_add_request_ts
                ),
                "dispatch_to_rescheduling_s": _duration(
                    self.rescheduling_ts, self.dispatch_ts
                ),
                "dispatch_to_reject_finish_s": _duration(
                    self.reject_finish_ts, self.dispatch_ts
                ),
                "engine_arrival_to_rescheduling_s": _duration(
                    self.rescheduling_ts, self.engine_arrival_ts
                ),
                "reject_finish_to_rescheduling_s": _duration(
                    self.rescheduling_ts, self.reject_finish_ts
                ),
                "router_ack_to_service_ready_s": _duration(
                    self.service_ready_ts, self.router_ack_ts
                ),
                "engine_add_request_to_service_launch_actual_s": _duration(
                    self.service_launch_ts_actual, self.engine_add_request_ts
                ),
                "service_launch_minus_router_ack_s": (
                    None
                    if self.service_launch_ts_actual is None or self.router_ack_ts is None
                    else self.service_launch_ts_actual - self.router_ack_ts
                ),
                "router_entry_to_service_ready_s": _duration(
                    self.service_ready_ts, self.router_entry_ts
                ),
                "router_entry_to_rescheduling_s": _duration(
                    self.rescheduling_ts, self.router_entry_ts
                ),
            }
        )
        return row


@dataclass
class RequestSummary:
    request_id: str
    attempts_total: int
    retries_total: int
    accepted: bool
    first_router_entry_ts: float | None
    final_service_ready_ts: float | None
    final_finish_ts: float | None
    final_status: str
    time_to_service_ready_s: float | None
    time_to_final_finish_s: float | None
    total_routing_overhead_s: float
    total_reject_finish_scheduling_overhead_s: float
    first_serving_batch_scheduling_overhead_s: float | None

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def _new_attempt(
    request_id: str,
    attempts_by_request: dict[str, list[Attempt]],
    *,
    start_reason: str,
    router_entry_ts: float | None,
) -> Attempt:
    attempts = attempts_by_request.setdefault(request_id, [])
    attempt = Attempt(
        request_id=request_id,
        attempt_index=len(attempts) + 1,
        start_reason=start_reason,
        router_entry_ts=router_entry_ts,
    )
    attempts.append(attempt)
    return attempt


def _current_attempt(
    request_id: str,
    attempts_by_request: dict[str, list[Attempt]],
    *,
    create_reason: str = "unknown",
    create_ts: float | None = None,
) -> Attempt:
    attempts = attempts_by_request.get(request_id)
    if attempts:
        return attempts[-1]
    return _new_attempt(
        request_id,
        attempts_by_request,
        start_reason=create_reason,
        router_entry_ts=create_ts,
    )


def build_attempts(events: list[dict[str, Any]]) -> dict[str, list[Attempt]]:
    attempts_by_request: dict[str, list[Attempt]] = {}
    routing_by_iter: dict[int, dict[str, float | int | None]] = {}
    last_routing_context: dict[str, float | int | None] | None = None

    for event in _iter_progress(
        events,
        desc="Reconstructing attempts",
        total=len(events),
        unit="event",
    ):
        event_type = str(event.get("event_type", ""))
        timestamp = _coerce_float(event.get("timestamp"))

        if event_type == "routing":
            extra_args = event.get("extra_args") or {}
            routing_iter = _coerce_int(extra_args.get("routing_iter"))
            ctx = {
                "timestamp": timestamp,
                "routing_overhead_s": _coerce_float(event.get("routing_overhead")),
                "routing_waiting_time_s": _coerce_float(extra_args.get("waiting_time")),
                "routing_iter": routing_iter,
            }
            last_routing_context = ctx
            if routing_iter is not None:
                routing_by_iter[routing_iter] = ctx
            continue

        request_id_raw = event.get("request_id")
        request_id = None if request_id_raw is None else str(request_id_raw)

        if event_type == "arrival-router":
            if request_id is None:
                continue
            attempts = attempts_by_request.get(request_id)
            if not attempts:
                _new_attempt(
                    request_id,
                    attempts_by_request,
                    start_reason="arrival-router",
                    router_entry_ts=timestamp,
                )
            else:
                current = attempts[-1]
                if (
                    current.router_entry_ts is None
                    and current.start_reason == "unknown"
                    and timestamp is not None
                ):
                    current.start_reason = "arrival-router"
                    current.router_entry_ts = timestamp
                elif current.rescheduling_ts is not None:
                    _new_attempt(
                        request_id,
                        attempts_by_request,
                        start_reason="arrival-router",
                        router_entry_ts=timestamp,
                    )
            continue

        if request_id is None and event_type != "batch":
            continue

        if event_type == "router_decision":
            assert request_id is not None
            attempt = _current_attempt(request_id, attempts_by_request, create_ts=timestamp)
            extra_args = event.get("extra_args") or {}
            routing_iter = _coerce_int(extra_args.get("routing_iter"))
            ctx = None
            if routing_iter is not None:
                ctx = routing_by_iter.get(routing_iter)
            if ctx is None:
                ctx = last_routing_context
            if ctx is not None:
                attempt.routing_ts = _coerce_float(ctx.get("timestamp"))
                attempt.routing_overhead_s = _coerce_float(ctx.get("routing_overhead_s"))
                attempt.routing_waiting_time_s = _coerce_float(
                    ctx.get("routing_waiting_time_s")
                )
                attempt.routing_iter = _coerce_int(ctx.get("routing_iter"))
                if (
                    attempt.routing_overhead_s is not None
                    and attempt.routing_waiting_time_s is not None
                ):
                    attempt.routing_compute_time_s = max(
                        0.0,
                        attempt.routing_overhead_s - attempt.routing_waiting_time_s,
                    )
                elif attempt.routing_overhead_s is not None:
                    attempt.routing_compute_time_s = attempt.routing_overhead_s
                if (
                    attempt.routing_ts is not None
                    and attempt.routing_compute_time_s is not None
                ):
                    attempt.routing_start_ts = (
                        attempt.routing_ts - attempt.routing_compute_time_s
                    )
            attempt.router_decision_ts = timestamp
            attempt.target_prefill_device_id = _coerce_int(event.get("prefill_device_id"))
            attempt.target_decode_device_id = _coerce_int(event.get("decode_device_id"))
            continue

        if event_type in {"dispatch-both", "dispatch-prefill", "dispatch-decode"}:
            assert request_id is not None
            attempt = _current_attempt(request_id, attempts_by_request, create_ts=timestamp)
            attempt.dispatch_ts = timestamp
            attempt.dispatch_type = event_type
            if attempt.target_prefill_device_id is None:
                attempt.target_prefill_device_id = _coerce_int(
                    event.get("prefill_device_id")
                )
            if attempt.target_decode_device_id is None:
                attempt.target_decode_device_id = _coerce_int(
                    event.get("decode_device_id")
                )
            continue

        if event_type == "arrival":
            assert request_id is not None
            attempt = _current_attempt(request_id, attempts_by_request, create_ts=timestamp)
            if attempt.engine_arrival_ts is None:
                attempt.engine_arrival_ts = timestamp
                attempt.engine_device_id = _coerce_int(event.get("device_id"))
            continue

        if event_type == "add_request":
            assert request_id is not None
            attempt = _current_attempt(request_id, attempts_by_request, create_ts=timestamp)
            attempt.engine_add_request_ts = timestamp
            extra_args = event.get("extra_args") or {}
            attempt.engine_add_request_elapsed_s = _coerce_float(extra_args.get("elapsed"))
            admitted = extra_args.get("admitted")
            if isinstance(admitted, bool):
                attempt.engine_admitted = admitted
                if admitted:
                    attempt.terminal_status = "engine-admitted"
                else:
                    attempt.terminal_status = "engine-rejected"
            continue

        if event_type == "admitted":
            assert request_id is not None
            attempt = _current_attempt(request_id, attempts_by_request, create_ts=timestamp)
            attempt.router_ack_ts = timestamp
            attempt.router_ack_type = "admitted"
            attempt.refresh_service_ready()
            if attempt.service_ready_ts is not None:
                attempt.terminal_status = "service-ready"
            else:
                attempt.terminal_status = "router-acknowledged"
            continue

        if event_type == "rescheduling":
            assert request_id is not None
            attempt = _current_attempt(request_id, attempts_by_request, create_ts=timestamp)
            attempt.rescheduling_ts = timestamp
            attempt.router_ack_ts = timestamp
            attempt.router_ack_type = "rescheduling"
            attempt.terminal_status = "rescheduled"
            _new_attempt(
                request_id,
                attempts_by_request,
                start_reason="rescheduling",
                router_entry_ts=timestamp,
            )
            continue

        if event_type == "finish":
            assert request_id is not None
            attempt = _current_attempt(request_id, attempts_by_request, create_ts=timestamp)
            finish_reason = None if event.get("finish_reason") is None else str(event.get("finish_reason"))
            attempt.finish_ts = timestamp
            attempt.finish_reason = finish_reason
            if finish_reason is not None and finish_reason.startswith("rejected-arrival"):
                attempt.reject_finish_ts = timestamp
                attempt.reject_finish_reason = finish_reason
                attempt.reject_finish_rejection_reason = (
                    None
                    if event.get("rejection_reason") is None
                    else str(event.get("rejection_reason"))
                )
                attempt.reject_finish_scheduling_overhead_s = _coerce_float(
                    event.get("scheduling_overhead")
                )
                attempt.terminal_status = "engine-rejected"
            elif finish_reason == "router_rejection":
                attempt.router_ack_ts = timestamp
                attempt.router_ack_type = "router_rejection"
                attempt.terminal_status = "router-rejected"
            elif finish_reason is not None and finish_reason != "length":
                attempt.terminal_status = f"finished:{finish_reason}"
            continue

        if event_type != "batch":
            continue

        batch_ts = timestamp
        batch_elapsed = _coerce_float(event.get("elapsed"))
        batch_id = _coerce_int(event.get("batch_id"))
        device_id = _coerce_int(event.get("device_id"))
        scheduling_overhead = _coerce_float(event.get("scheduling_overhead"))
        extra_args = event.get("extra_args") or {}
        to_launch = _coerce_float(extra_args.get("to_launch"))
        if batch_ts is None or batch_elapsed is None:
            continue
        batch_start = batch_ts - batch_elapsed
        service_launch_ts_actual = (
            None if to_launch is None else batch_start + to_launch
        )

        for req_id_raw in event.get("req_ids") or []:
            req_id = str(req_id_raw)
            attempts = attempts_by_request.get(req_id)
            if not attempts:
                continue
            for attempt in attempts:
                if attempt.engine_admitted is not True:
                    continue
                if attempt.service_batch_id is not None:
                    continue
                if attempt.engine_arrival_ts is not None and batch_ts < attempt.engine_arrival_ts:
                    continue
                if (
                    attempt.engine_device_id is not None
                    and device_id is not None
                    and attempt.engine_device_id != device_id
                ):
                    continue
                attempt.service_batch_id = batch_id
                attempt.service_batch_device_id = device_id
                attempt.first_batch_ts = batch_ts
                attempt.first_batch_elapsed_s = batch_elapsed
                attempt.first_batch_scheduling_overhead_s = scheduling_overhead
                attempt.first_batch_to_launch_s = to_launch
                attempt.service_launch_ts_actual = service_launch_ts_actual
                attempt.refresh_service_ready()
                if attempt.service_ready_ts is not None:
                    attempt.terminal_status = "service-ready"
                break

    for attempts in attempts_by_request.values():
        for attempt in attempts:
            if attempt.terminal_status is not None:
                continue
            if attempt.service_ready_ts is not None:
                attempt.terminal_status = "service-ready"
            elif attempt.service_launch_ts_actual is not None:
                attempt.terminal_status = "service-launch-observed"
            elif attempt.rescheduling_ts is not None:
                attempt.terminal_status = "rescheduled"
            elif attempt.reject_finish_ts is not None:
                attempt.terminal_status = "engine-rejected"
            elif attempt.router_ack_type == "router_rejection":
                attempt.terminal_status = "router-rejected"
            elif attempt.engine_admitted is True:
                attempt.terminal_status = "engine-admitted"
            elif attempt.engine_admitted is False:
                attempt.terminal_status = "engine-rejected"
            else:
                attempt.terminal_status = "incomplete"

    return attempts_by_request


def build_request_summaries(
    attempts_by_request: dict[str, list[Attempt]]
) -> list[RequestSummary]:
    summaries: list[RequestSummary] = []
    sorted_attempts = sorted(
        attempts_by_request.items(),
        key=lambda item: int(item[0]) if item[0].isdigit() else item[0],
    )
    for request_id, attempts in _iter_progress(
        sorted_attempts,
        desc="Summarizing requests",
        total=len(sorted_attempts),
        unit="request",
    ):
        first_router_entry_ts = next(
            (attempt.router_entry_ts for attempt in attempts if attempt.router_entry_ts is not None),
            None,
        )
        accepted_attempt = next(
            (attempt for attempt in attempts if attempt.service_ready_ts is not None),
            None,
        )
        final_finish_ts = next(
            (
                attempt.finish_ts
                for attempt in reversed(attempts)
                if attempt.finish_ts is not None
            ),
            None,
        )
        final_status = attempts[-1].terminal_status or "incomplete"
        first_service_batch_scheduling_overhead_s = (
            None
            if accepted_attempt is None
            else accepted_attempt.first_batch_scheduling_overhead_s
        )
        summaries.append(
            RequestSummary(
                request_id=request_id,
                attempts_total=len(attempts),
                retries_total=max(0, len(attempts) - 1),
                accepted=accepted_attempt is not None,
                first_router_entry_ts=first_router_entry_ts,
                final_service_ready_ts=(
                    None if accepted_attempt is None else accepted_attempt.service_ready_ts
                ),
                final_finish_ts=final_finish_ts,
                final_status=(
                    "accepted"
                    if accepted_attempt is not None
                    else final_status
                ),
                time_to_service_ready_s=(
                    None
                    if accepted_attempt is None
                    else _duration(accepted_attempt.service_ready_ts, first_router_entry_ts)
                ),
                time_to_final_finish_s=_duration(final_finish_ts, first_router_entry_ts),
                total_routing_overhead_s=sum(
                    attempt.routing_overhead_s or 0.0 for attempt in attempts
                ),
                total_reject_finish_scheduling_overhead_s=sum(
                    attempt.reject_finish_scheduling_overhead_s or 0.0
                    for attempt in attempts
                ),
                first_serving_batch_scheduling_overhead_s=(
                    first_service_batch_scheduling_overhead_s
                ),
            )
        )
    return summaries


def summarize_attempts(
    attempts_by_request: dict[str, list[Attempt]]
) -> dict[str, Any]:
    all_attempts = [attempt for attempts in attempts_by_request.values() for attempt in attempts]
    accepted_attempts = [
        attempt for attempt in all_attempts if attempt.service_ready_ts is not None
    ]
    rescheduled_attempts = [
        attempt for attempt in all_attempts if attempt.rescheduling_ts is not None
    ]
    router_rejected_attempts = [
        attempt for attempt in all_attempts if attempt.router_ack_type == "router_rejection"
    ]

    def collect(attempts: list[Attempt], builders: dict[str, Any]) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for name, builder in builders.items():
            values: list[float] = []
            for attempt in attempts:
                value = builder(attempt)
                if value is None:
                    continue
                values.append(float(value))
            metrics[name] = _stats(values)
        return metrics

    accepted_metrics = collect(
        accepted_attempts,
        {
            "router_entry_to_routing_s": lambda a: _duration(a.routing_ts, a.router_entry_ts),
            "queueing_time_s": lambda a: _duration(a.routing_start_ts, a.router_entry_ts),
            "routing_overhead_s": lambda a: a.routing_overhead_s,
            "routing_waiting_time_s": lambda a: a.routing_waiting_time_s,
            "routing_compute_time_s": lambda a: a.routing_compute_time_s,
            "routing_to_dispatch_s": lambda a: _duration(a.dispatch_ts, a.routing_ts),
            "dispatch_to_engine_arrival_s": lambda a: _duration(a.engine_arrival_ts, a.dispatch_ts),
            "engine_arrival_to_add_request_s": lambda a: _duration(a.engine_add_request_ts, a.engine_arrival_ts),
            "engine_add_request_to_router_ack_s": lambda a: _duration(a.router_ack_ts, a.engine_add_request_ts),
            "router_ack_to_service_ready_s": lambda a: _duration(a.service_ready_ts, a.router_ack_ts),
            "engine_add_request_to_service_launch_actual_s": lambda a: _duration(a.service_launch_ts_actual, a.engine_add_request_ts),
            "service_launch_minus_router_ack_s": (
                lambda a: None
                if a.service_launch_ts_actual is None or a.router_ack_ts is None
                else a.service_launch_ts_actual - a.router_ack_ts
            ),
            "router_entry_to_service_ready_s": lambda a: _duration(a.service_ready_ts, a.router_entry_ts),
            "first_batch_scheduling_overhead_s": lambda a: a.first_batch_scheduling_overhead_s,
            "first_batch_elapsed_s": lambda a: a.first_batch_elapsed_s,
            "first_batch_to_launch_s": lambda a: a.first_batch_to_launch_s,
        },
    )

    rescheduled_metrics = collect(
        rescheduled_attempts,
        {
            "router_entry_to_routing_s": lambda a: _duration(a.routing_ts, a.router_entry_ts),
            "queueing_time_s": lambda a: _duration(a.routing_start_ts, a.router_entry_ts),
            "routing_overhead_s": lambda a: a.routing_overhead_s,
            "routing_waiting_time_s": lambda a: a.routing_waiting_time_s,
            "routing_compute_time_s": lambda a: a.routing_compute_time_s,
            "routing_to_dispatch_s": lambda a: _duration(a.dispatch_ts, a.routing_ts),
            "dispatch_to_engine_arrival_s": lambda a: _duration(a.engine_arrival_ts, a.dispatch_ts),
            "engine_arrival_to_add_request_s": lambda a: _duration(a.engine_add_request_ts, a.engine_arrival_ts),
            "dispatch_to_rescheduling_s": lambda a: _duration(a.rescheduling_ts, a.dispatch_ts),
            "dispatch_to_reject_finish_s": lambda a: _duration(a.reject_finish_ts, a.dispatch_ts),
            "engine_arrival_to_rescheduling_s": lambda a: _duration(a.rescheduling_ts, a.engine_arrival_ts),
            "reject_add_request_to_finish_s": lambda a: _duration(a.reject_finish_ts, a.engine_add_request_ts),
            "engine_add_request_to_router_ack_s": lambda a: _duration(a.router_ack_ts, a.engine_add_request_ts),
            "reject_finish_to_rescheduling_s": lambda a: _duration(a.rescheduling_ts, a.reject_finish_ts),
            "router_entry_to_rescheduling_s": lambda a: _duration(a.rescheduling_ts, a.router_entry_ts),
            "reject_finish_scheduling_overhead_s": lambda a: a.reject_finish_scheduling_overhead_s,
        },
    )

    request_summaries = build_request_summaries(attempts_by_request)
    accepted_requests = [summary for summary in request_summaries if summary.accepted]
    request_metrics = {
        "attempts_total": _stats([float(summary.attempts_total) for summary in request_summaries]),
        "retries_total": _stats([float(summary.retries_total) for summary in request_summaries]),
        "time_to_service_ready_s": _stats(
            [
                float(summary.time_to_service_ready_s)
                for summary in accepted_requests
                if summary.time_to_service_ready_s is not None
            ]
        ),
        "total_routing_overhead_s": _stats(
            [float(summary.total_routing_overhead_s) for summary in request_summaries]
        ),
        "first_serving_batch_scheduling_overhead_s": _stats(
            [
                float(summary.first_serving_batch_scheduling_overhead_s)
                for summary in accepted_requests
                if summary.first_serving_batch_scheduling_overhead_s is not None
            ]
        ),
    }

    return {
        "counts": {
            "requests_total": len(request_summaries),
            "accepted_requests": len(accepted_requests),
            "requests_with_retries": sum(
                1 for summary in request_summaries if summary.retries_total > 0
            ),
            "attempts_total": len(all_attempts),
            "accepted_attempts": len(accepted_attempts),
            "rescheduled_attempts": len(rescheduled_attempts),
            "router_rejected_attempts": len(router_rejected_attempts),
        },
        "accepted_attempt_metrics": accepted_metrics,
        "rescheduled_attempt_metrics": rescheduled_metrics,
        "rescheduled_round_trip_metrics": {
            "queueing_time_s": rescheduled_metrics["queueing_time_s"],
            "routing_compute_time_s": rescheduled_metrics["routing_compute_time_s"],
            "routing_elapsed_s": rescheduled_metrics["routing_overhead_s"],
            "dispatch_to_engine_arrival_s": (
                rescheduled_metrics["dispatch_to_engine_arrival_s"]
            ),
            "engine_arrival_to_rescheduling_s": (
                rescheduled_metrics["engine_arrival_to_rescheduling_s"]
            ),
        },
        "rescheduled_engine_breakdown_metrics": {
            "engine_arrival_to_rescheduling_s": (
                rescheduled_metrics["engine_arrival_to_rescheduling_s"]
            ),
            "feasibility_check_s": (
                rescheduled_metrics["engine_arrival_to_add_request_s"]
            ),
            "route_back_to_server_s": (
                rescheduled_metrics["engine_add_request_to_router_ack_s"]
            ),
            "reject_finish_to_rescheduling_s": (
                rescheduled_metrics["reject_finish_to_rescheduling_s"]
            ),
        },
        "request_metrics": request_metrics,
        "request_summaries": [summary.to_row() for summary in request_summaries],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_zip(path: Path, files: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            if not file_path.exists():
                continue
            zf.write(file_path, arcname=file_path.name)


def _stats_row(
    segment: str,
    stats: dict[str, Any],
    *,
    value_unit: str = "s",
    **extra: Any,
) -> dict[str, Any]:
    row = {
        **extra,
        "segment": segment,
        "value_unit": value_unit,
        "count": stats.get("count"),
        "mean_value": stats.get("mean"),
        "std_value": stats.get("std"),
        "min_value": stats.get("min"),
        "p50_value": stats.get("p50"),
        "p90_value": stats.get("p90"),
        "p95_value": stats.get("p95"),
        "p99_value": stats.get("p99"),
        "max_value": stats.get("max"),
        "mean_s": stats.get("mean"),
        "std_s": stats.get("std"),
        "min_s": stats.get("min"),
        "p50_s": stats.get("p50"),
        "p90_s": stats.get("p90"),
        "p95_s": stats.get("p95"),
        "p99_s": stats.get("p99"),
        "max_s": stats.get("max"),
    }
    if value_unit != "s":
        row.update(
            {
                "mean_s": None,
                "std_s": None,
                "min_s": None,
                "p50_s": None,
                "p90_s": None,
                "p95_s": None,
                "p99_s": None,
                "max_s": None,
            }
        )
    return row


def _extract_batch_time_breakdown(event: dict[str, Any]) -> dict[str, float | None]:
    elapsed = _coerce_float(event.get("elapsed"))
    scheduling_overhead = _coerce_float(event.get("scheduling_overhead"))
    output_processing_elapsed = _coerce_float(event.get("output_processing_elapsed"))
    publish_overhead = _coerce_float(event.get("publish_overhead"))
    extra_args = event.get("extra_args") or {}
    to_launch = _coerce_float(extra_args.get("to_launch"))
    to_finish = _coerce_float(extra_args.get("to_finish"))

    kernel_execution_time = None
    other_control_logic_time = None
    pre_launch_control_time = None
    post_kernel_control_time = None

    if elapsed is not None and to_launch is not None and to_finish is not None:
        kernel_execution_time = max(0.0, to_finish - to_launch)
        if scheduling_overhead is not None:
            pre_launch_control_time = max(0.0, to_launch - scheduling_overhead)
            other_control_logic_time = max(
                0.0,
                elapsed - scheduling_overhead - kernel_execution_time,
            )
        post_kernel_control_time = max(0.0, elapsed - to_finish)
        if other_control_logic_time is None and pre_launch_control_time is not None:
            other_control_logic_time = (
                pre_launch_control_time + post_kernel_control_time
            )
    elif elapsed is not None:
        known_control = 0.0
        has_known_control = False
        if output_processing_elapsed is not None:
            known_control += max(0.0, output_processing_elapsed)
            has_known_control = True
        if publish_overhead is not None:
            known_control += max(0.0, publish_overhead)
            has_known_control = True
        if scheduling_overhead is not None:
            kernel_execution_time = max(
                0.0,
                elapsed - scheduling_overhead - known_control,
            )
        elif has_known_control:
            kernel_execution_time = max(0.0, elapsed - known_control)
        if has_known_control:
            other_control_logic_time = known_control

    execution_without_scheduling = (
        None
        if elapsed is None or scheduling_overhead is None
        else max(0.0, elapsed - scheduling_overhead)
    )
    return {
        "batch_total_elapsed_s": elapsed,
        "batch_scheduling_overhead_s": scheduling_overhead,
        "batch_execution_time_s": execution_without_scheduling,
        "batch_kernel_execution_time_s": kernel_execution_time,
        "batch_other_control_logic_time_s": other_control_logic_time,
        "batch_output_processing_elapsed_s": output_processing_elapsed,
        "batch_publish_overhead_s": publish_overhead,
        "batch_pre_launch_control_logic_time_s": pre_launch_control_time,
        "batch_post_kernel_control_logic_time_s": post_kernel_control_time,
    }


def _collect_batch_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    run_start_s = time.monotonic()
    batch_metric_values: dict[str, list[float]] = {
        "batch_total_elapsed_s": [],
        "batch_scheduling_overhead_s": [],
        "batch_execution_time_s": [],
        "batch_kernel_execution_time_s": [],
        "batch_other_control_logic_time_s": [],
        "batch_output_processing_elapsed_s": [],
        "batch_publish_overhead_s": [],
        "batch_pre_launch_control_logic_time_s": [],
        "batch_post_kernel_control_logic_time_s": [],
    }
    observed_engine_phase_values: dict[str, list[float]] = {
        "process_input_elapsed_s": [],
        "engine_step_elapsed_s": [],
    }
    batch_sizes_by_type: dict[str, list[float]] = {
        "prefill_only": [],
        "decode_only": [],
        "mixed": [],
    }
    with _phase("collect_batch_stats.build_request_prefill_tokens", run_start_s=run_start_s):
        prefill_tokens_by_request = _build_request_prefill_tokens(events)
    with _phase("collect_batch_stats.scan_batch_events", run_start_s=run_start_s):
        for event in _iter_progress(
            events,
            desc="Collecting batch stats",
            total=len(events),
            unit="event",
        ):
            event_type = str(event.get("event_type", ""))
            if event_type == "process_input":
                elapsed = _coerce_float(event.get("elapsed"))
                if elapsed is not None:
                    observed_engine_phase_values["process_input_elapsed_s"].append(
                        elapsed
                    )
                continue
            if event_type == "engine_step":
                elapsed = _coerce_float(event.get("elapsed"))
                if elapsed is not None:
                    observed_engine_phase_values["engine_step_elapsed_s"].append(
                        elapsed
                    )
                continue
            if event_type != "batch":
                continue
            breakdown = _extract_batch_time_breakdown(event)
            if breakdown["batch_total_elapsed_s"] is None:
                continue
            for metric_name, metric_value in breakdown.items():
                if metric_value is not None:
                    batch_metric_values[metric_name].append(metric_value)
            batch_type = _classify_batch_type(event, prefill_tokens_by_request)
            if batch_type in batch_sizes_by_type:
                batch_sizes_by_type[batch_type].append(
                    float(len(event.get("req_ids") or []))
                )
    with _phase("collect_batch_stats.reduce_stats", run_start_s=run_start_s):
        return {
            **{
                metric_name: _stats(values)
                for metric_name, values in batch_metric_values.items()
            },
            "observed_engine_phase_metrics": {
                metric_name: _stats(values)
                for metric_name, values in observed_engine_phase_values.items()
            },
            "batch_size_distribution_metrics": {
                "prefill_only_batch_size_reqs": _stats(
                    batch_sizes_by_type["prefill_only"]
                ),
                "decode_only_batch_size_reqs": _stats(
                    batch_sizes_by_type["decode_only"]
                ),
                "mixed_batch_size_reqs": _stats(batch_sizes_by_type["mixed"]),
            },
        }


def _build_segment_stats_rows(
    summary: dict[str, Any],
    batch_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    segments = [
        (
            "queueing_time_s",
            summary["rescheduled_round_trip_metrics"]["queueing_time_s"],
            "rescheduled_attempts",
        ),
        (
            "routing_compute_time_s",
            summary["rescheduled_round_trip_metrics"]["routing_compute_time_s"],
            "rescheduled_attempts",
        ),
        (
            "routing_elapsed_s",
            summary["rescheduled_round_trip_metrics"]["routing_elapsed_s"],
            "rescheduled_attempts",
        ),
        (
            "dispatch_to_engine_arrival_s",
            summary["rescheduled_round_trip_metrics"]["dispatch_to_engine_arrival_s"],
            "rescheduled_attempts",
        ),
        (
            "engine_arrival_to_rescheduling_s",
            summary["rescheduled_round_trip_metrics"]["engine_arrival_to_rescheduling_s"],
            "rescheduled_attempts",
        ),
        (
            "feasibility_check_s",
            summary["rescheduled_engine_breakdown_metrics"]["feasibility_check_s"],
            "rescheduled_attempts",
        ),
        (
            "add_request_to_server_s",
            summary["rescheduled_engine_breakdown_metrics"]["route_back_to_server_s"],
            "rescheduled_attempts",
        ),
        (
            "batch_scheduling_overhead_s",
            batch_stats["batch_scheduling_overhead_s"],
            "all_batches",
        ),
        (
            "batch_execution_time_s",
            batch_stats["batch_execution_time_s"],
            "all_batches",
        ),
        (
            "batch_total_elapsed_s",
            batch_stats["batch_total_elapsed_s"],
            "all_batches",
        ),
        (
            "batch_kernel_execution_time_s",
            batch_stats["batch_kernel_execution_time_s"],
            "all_batches",
        ),
        (
            "batch_other_control_logic_time_s",
            batch_stats["batch_other_control_logic_time_s"],
            "all_batches",
        ),
        (
            "batch_output_processing_elapsed_s",
            batch_stats["batch_output_processing_elapsed_s"],
            "all_batches",
        ),
        (
            "batch_publish_overhead_s",
            batch_stats["batch_publish_overhead_s"],
            "all_batches",
        ),
        (
            "batch_pre_launch_control_logic_time_s",
            batch_stats["batch_pre_launch_control_logic_time_s"],
            "all_batches",
        ),
        (
            "batch_post_kernel_control_logic_time_s",
            batch_stats["batch_post_kernel_control_logic_time_s"],
            "all_batches",
        ),
    ]
    for segment, stats, cohort in segments:
        rows.append(
            _stats_row(
                segment,
                stats,
                scope="overall_segment",
                cohort=cohort,
                device_id=None,
            )
        )
    for segment, stats in batch_stats["observed_engine_phase_metrics"].items():
        rows.append(
            _stats_row(
                segment,
                stats,
                scope="observed_engine_phase",
                cohort="observed_engine_phase_events",
                device_id=None,
            )
        )
    for segment, stats in batch_stats["batch_size_distribution_metrics"].items():
        rows.append(
            _stats_row(
                segment,
                stats,
                scope="batch_size_distribution",
                cohort="all_batches",
                device_id=None,
                value_unit="requests_per_batch",
            )
        )
    return rows


def _build_batch_size_rows(batch_stats: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in [
        "batch_total_elapsed_s",
        "batch_scheduling_overhead_s",
        "batch_execution_time_s",
        "batch_kernel_execution_time_s",
        "batch_other_control_logic_time_s",
        "batch_output_processing_elapsed_s",
        "batch_publish_overhead_s",
        "batch_pre_launch_control_logic_time_s",
        "batch_post_kernel_control_logic_time_s",
    ]:
        rows.append(
            _stats_row(
                segment,
                batch_stats[segment],
                scope="overall_segment",
                cohort="all_batches",
                device_id=None,
            )
        )
    for segment, stats in batch_stats["observed_engine_phase_metrics"].items():
        rows.append(
            _stats_row(
                segment,
                stats,
                scope="observed_engine_phase",
                cohort="observed_engine_phase_events",
                device_id=None,
            )
        )
    for segment, stats in batch_stats["batch_size_distribution_metrics"].items():
        rows.append(
            _stats_row(
                segment,
                stats,
                scope="batch_size_distribution",
                cohort="all_batches",
                device_id=None,
                value_unit="requests_per_batch",
            )
        )
    return rows


def _build_device_breakdown_rows(
    attempts_by_request: dict[str, list[Attempt]],
) -> list[dict[str, Any]]:
    all_attempts = [attempt for attempts in attempts_by_request.values() for attempt in attempts]
    cohorts = {
        "accepted_attempts": [
            attempt
            for attempt in all_attempts
            if attempt.service_ready_ts is not None and attempt.engine_device_id is not None
        ],
        "rescheduled_attempts": [
            attempt
            for attempt in all_attempts
            if attempt.rescheduling_ts is not None and attempt.engine_device_id is not None
        ],
        "all_attempts_with_server_ack": [
            attempt
            for attempt in all_attempts
            if attempt.router_ack_ts is not None and attempt.engine_device_id is not None
        ],
    }

    rows: list[dict[str, Any]] = []
    for cohort, attempts in cohorts.items():
        by_device: dict[int, dict[str, list[float]]] = {}
        for attempt in attempts:
            device_id = attempt.engine_device_id
            if device_id is None:
                continue
            device_bucket = by_device.setdefault(
                device_id,
                {
                    "dispatch_to_engine_arrival_s": [],
                    "add_request_to_server_s": [],
                },
            )
            dispatch_to_arrival = _duration(attempt.engine_arrival_ts, attempt.dispatch_ts)
            add_request_to_server = _duration(
                attempt.router_ack_ts, attempt.engine_add_request_ts
            )
            if dispatch_to_arrival is not None:
                device_bucket["dispatch_to_engine_arrival_s"].append(dispatch_to_arrival)
            if add_request_to_server is not None:
                device_bucket["add_request_to_server_s"].append(add_request_to_server)

        for device_id in sorted(by_device):
            for segment, values in by_device[device_id].items():
                rows.append(
                    _stats_row(
                        segment,
                        _stats(values),
                        scope="per_device_segment",
                        cohort=cohort,
                        device_id=device_id,
                    )
                )
    return rows


def _build_combined_breakdown_rows(
    summary: dict[str, Any],
    batch_stats: dict[str, Any],
    attempts_by_request: dict[str, list[Attempt]],
) -> list[dict[str, Any]]:
    rows = _build_segment_stats_rows(summary, batch_stats)
    rows.extend(_build_device_breakdown_rows(attempts_by_request))
    return rows


def _build_tries_distribution_rows(
    request_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts: dict[int, dict[str, int]] = {}
    total = len(request_rows)
    for row in request_rows:
        tries = int(row.get("attempts_total", 0) or 0)
        bucket = counts.setdefault(
            tries,
            {
                "request_count": 0,
                "accepted_count": 0,
                "not_accepted_count": 0,
            },
        )
        bucket["request_count"] += 1
        if bool(row.get("accepted")):
            bucket["accepted_count"] += 1
        else:
            bucket["not_accepted_count"] += 1

    result: list[dict[str, Any]] = []
    for tries in sorted(counts):
        bucket = counts[tries]
        request_count = bucket["request_count"]
        accepted_count = bucket["accepted_count"]
        not_accepted_count = bucket["not_accepted_count"]
        result.append(
            {
                "tries": tries,
                "request_count": request_count,
                "fraction_of_requests": (
                    0.0 if total == 0 else request_count / total
                ),
                "accepted_count": accepted_count,
                "not_accepted_count": not_accepted_count,
                "accepted_fraction_within_tries": (
                    0.0 if request_count == 0 else accepted_count / request_count
                ),
            }
        )
    return result


def _print_metric_block(title: str, metrics: dict[str, Any]) -> None:
    if not metrics:
        return
    print()
    print(title)
    print(
        f"{'metric':36} {'count':>7} {'mean_ms':>10} {'p50_ms':>10} "
        f"{'p90_ms':>10} {'p99_ms':>10} {'max_ms':>10}"
    )
    for name, stat in metrics.items():
        count = int(stat.get("count", 0) or 0)
        print(
            f"{name:36} {count:7d} "
            f"{_fmt_ms(stat.get('mean'))} {_fmt_ms(stat.get('p50'))} "
            f"{_fmt_ms(stat.get('p90'))} {_fmt_ms(stat.get('p99'))} "
            f"{_fmt_ms(stat.get('max'))}"
        )


def _print_count_metric_block(title: str, metrics: dict[str, Any]) -> None:
    if not metrics:
        return
    print()
    print(title)
    print(
        f"{'metric':36} {'count':>7} {'mean':>10} {'p50':>10} "
        f"{'p90':>10} {'p99':>10} {'max':>10}"
    )
    for name, stat in metrics.items():
        count = int(stat.get("count", 0) or 0)
        mean = stat.get("mean")
        p50 = stat.get("p50")
        p90 = stat.get("p90")
        p99 = stat.get("p99")
        max_value = stat.get("max")
        print(
            f"{name:36} {count:7d} "
            f"{('-' if mean is None else f'{mean:10.3f}')} "
            f"{('-' if p50 is None else f'{p50:10.3f}')} "
            f"{('-' if p90 is None else f'{p90:10.3f}')} "
            f"{('-' if p99 is None else f'{p99:10.3f}')} "
            f"{('-' if max_value is None else f'{max_value:10.3f}')}"
        )


def _print_reason_block(
    title: str,
    counts: dict[str, Any],
    rates: dict[str, Any],
) -> None:
    if not counts:
        return
    print()
    print(title)
    print(f"{'category':24} {'count':>10} {'rate_pct':>10}")
    for category in sorted(counts):
        count = int(counts.get(category, 0) or 0)
        rate = float(rates.get(category, 0.0) or 0.0) * 100.0
        print(f"{category:24} {count:10d} {rate:10.3f}")


def _print_request_detail(request_id: str, attempts: list[Attempt]) -> None:
    print()
    print(f"Request {request_id}")
    for attempt in attempts:
        row = attempt.to_row()
        print(
            "  "
            f"attempt={attempt.attempt_index} start={attempt.start_reason} "
            f"status={attempt.terminal_status} "
            f"target=({attempt.target_prefill_device_id}->{attempt.target_decode_device_id}) "
            f"engine={attempt.engine_device_id}"
        )
        print(
            "    "
            f"router_entry={attempt.router_entry_ts} routing={attempt.routing_ts} "
            f"dispatch={attempt.dispatch_ts} engine_arrival={attempt.engine_arrival_ts} "
            f"add_request={attempt.engine_add_request_ts} router_ack={attempt.router_ack_ts} "
            f"service_launch_actual={attempt.service_launch_ts_actual} "
            f"service_ready={attempt.service_ready_ts}"
        )
        print(
            "    "
            f"router->routing={row['router_entry_to_routing_s']} "
            f"routing->dispatch={row['routing_to_dispatch_s']} "
            f"dispatch->arrival={row['dispatch_to_engine_arrival_s']} "
            f"arrival->add_request={row['engine_arrival_to_add_request_s']} "
            f"add_request->router_ack={row['engine_add_request_to_router_ack_s']} "
            f"router_ack->service_ready={row['router_ack_to_service_ready_s']}"
        )


def _resolve_analysis_inputs(args: argparse.Namespace) -> dict[str, Path | None]:
    input_path = args.input_path
    events_file = args.events_file
    reqs_file = args.reqs_file
    summary_file: Path | None = None

    prefix = args.prefix
    if prefix is None and input_path is not None:
        prefix = _strip_known_analysis_suffix(input_path)

    if input_path is not None:
        input_path_str = str(input_path)
        if input_path_str.endswith(".events.summary.json") or input_path_str.endswith(
            ".summary.json"
        ):
            summary_file = input_path
        elif input_path_str.endswith(".events.jsonl") or input_path_str.endswith(".events.json"):
            if events_file is None:
                events_file = input_path
        elif input_path_str.endswith(".reqs.jsonl") or input_path_str.endswith(".reqs.json"):
            if reqs_file is None:
                reqs_file = input_path

    if events_file is None:
        events_file = _resolve_existing_companion(
            prefix,
            (".events.jsonl", ".events.json"),
        )
    if reqs_file is None:
        reqs_file = _resolve_existing_companion(
            prefix,
            (".reqs.jsonl", ".reqs.json"),
        )
    if summary_file is None:
        summary_file = _resolve_existing_companion(
            prefix,
            (".events.summary.json", ".summary.json"),
        )

    if events_file is None and reqs_file is None and summary_file is None:
        raise ValueError(
            "no analyzable inputs found; pass an events file, reqs file, summary file, "
            "or a run prefix with matching companions"
        )

    return {
        "prefix": prefix,
        "events_file": events_file,
        "reqs_file": reqs_file,
        "summary_file": summary_file,
    }


def _default_output_prefix_from_sources(
    *,
    prefix: Path | None,
    events_file: Path | None,
    reqs_file: Path | None,
    summary_file: Path | None,
) -> Path:
    if prefix is not None:
        source_name = _strip_known_analysis_suffix(prefix).name
    elif events_file is not None:
        source_name = _strip_known_analysis_suffix(events_file).name
    elif reqs_file is not None:
        source_name = _strip_known_analysis_suffix(reqs_file).name
    elif summary_file is not None:
        source_name = _strip_known_analysis_suffix(summary_file).name
    else:
        source_name = "routing_scheduling"
    return Path("plots/out") / f"{source_name}.routing_scheduling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze router/engine control-path timing and optional reqs-side "
            "SLO categories. The input can be an events file, a reqs file, a "
            "precomputed .events.summary.json, or a run prefix that resolves "
            "to matching companions."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        help=(
            "Optional path to an events file, reqs file, summary file, or run prefix."
        ),
    )
    parser.add_argument(
        "--prefix",
        type=Path,
        default=None,
        help="Run prefix used to infer companion files like .events.jsonl and .reqs.jsonl.",
    )
    parser.add_argument(
        "--events-file",
        type=Path,
        default=None,
        help="Explicit events trace path. Supports JSON arrays and JSONL.",
    )
    parser.add_argument(
        "--reqs-file",
        type=Path,
        default=None,
        help="Explicit saved reqs path. Supports JSON arrays and JSONL.",
    )
    parser.add_argument(
        "--ttft-slo-scale",
        type=float,
        default=None,
        help=(
            "TTFT SLO scale for reqs-side analysis. If omitted, the script "
            "tries to infer it from the run prefix."
        ),
    )
    parser.add_argument(
        "--slo-tpot",
        type=float,
        default=None,
        help=(
            "TPOT SLO for reqs-side analysis. If omitted, the script tries to "
            "infer it from the run prefix."
        ),
    )
    parser.add_argument(
        "--ttft-overhead",
        type=float,
        default=0.0,
        help="Optional TTFT overhead added to reqs-side TTFT budgets.",
    )
    parser.add_argument(
        "--routing-overhead",
        type=float,
        default=-1.0,
        help=(
            "Routing overhead used by reqs-side SLO evaluation. The default -1 "
            "matches the existing benchmark helper and uses request arrival_time."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help=(
            "Prefix for generated artifacts. Defaults to "
            "plots/out/<trace_stem>.routing_scheduling"
        ),
    )
    parser.add_argument(
        "--request-id",
        action="append",
        default=[],
        help="Optional request id to print a detailed reconstructed timeline for.",
    )
    parser.add_argument(
        "--zip-results",
        action="store_true",
        help=(
            "Also write a zip archive containing the generated artifacts for this run."
        ),
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help=(
            "Optional output path for the zip archive. Defaults to "
            "<output_prefix>.zip when --zip-results is set."
        ),
    )
    parser.add_argument(
        "--batch-only",
        action="store_true",
        help=(
            "Skip request-level reconstruction and export only batch-focused "
            "timing and batch-size metrics."
        ),
    )
    return parser.parse_args()


def main() -> None:
    run_start_s = time.monotonic()
    args = parse_args()
    input_sources = _resolve_analysis_inputs(args)
    prefix = input_sources["prefix"]
    events_file = input_sources["events_file"]
    reqs_file = input_sources["reqs_file"]
    summary_file = input_sources["summary_file"]

    inferred_ttft_slo_scale, inferred_slo_tpot = (None, None)
    for candidate in (prefix, events_file, reqs_file, summary_file):
        inferred_ttft_slo_scale, inferred_slo_tpot = _infer_slo_params_from_path(
            candidate
        )
        if inferred_ttft_slo_scale is not None and inferred_slo_tpot is not None:
            break
    ttft_slo_scale = (
        inferred_ttft_slo_scale
        if args.ttft_slo_scale is None
        else float(args.ttft_slo_scale)
    )
    slo_tpot = (
        inferred_slo_tpot
        if args.slo_tpot is None
        else float(args.slo_tpot)
    )

    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = _default_output_prefix_from_sources(
            prefix=prefix,
            events_file=events_file,
            reqs_file=reqs_file,
            summary_file=summary_file,
        )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    routing_payload: dict[str, Any] | None = None
    attempts_by_request: dict[str, list[Attempt]] = {}
    breakdown_rows: list[dict[str, Any]] = []
    tries_distribution_rows: list[dict[str, Any]] = []

    if events_file is not None:
        allowed_event_types = (
            {"arrival", "batch", "process_input", "engine_step"}
            if args.batch_only
            else None
        )
        with _phase("load_events", run_start_s=run_start_s):
            events = load_events(events_file, allowed_event_types=allowed_event_types)
        _log_phase(
            f"Loaded {len(events)} events from {events_file}",
            run_start_s=run_start_s,
        )

        with _phase("collect_batch_stats", run_start_s=run_start_s):
            batch_stats = _collect_batch_stats(events)
        if args.batch_only:
            summary: dict[str, Any] = {
                "mode": "batch_only",
                "counts": {
                    "events_loaded": len(events),
                },
            }
            request_rows: list[dict[str, Any]] = []
            breakdown_rows = _build_batch_size_rows(batch_stats)
            tries_distribution_rows = []
            _log_phase(
                "Skipping request reconstruction (--batch-only)",
                run_start_s=run_start_s,
            )
        else:
            with _phase("build_attempts", run_start_s=run_start_s):
                attempts_by_request = build_attempts(events)
            _log_phase(
                f"Reconstructed {sum(len(v) for v in attempts_by_request.values())} "
                f"attempts across {len(attempts_by_request)} requests",
                run_start_s=run_start_s,
            )
            with _phase("summarize_attempts", run_start_s=run_start_s):
                summary = summarize_attempts(attempts_by_request)
            request_rows = summary["request_summaries"]
            breakdown_rows = _build_combined_breakdown_rows(
                summary,
                batch_stats,
                attempts_by_request,
            )
            tries_distribution_rows = _build_tries_distribution_rows(request_rows)

        routing_payload = {
            **summary,
            "batch_metrics": batch_stats,
            "breakdown_rows": breakdown_rows,
            "request_rows": request_rows,
            "tries_distribution_rows": tries_distribution_rows,
        }
    elif summary_file is not None:
        with _phase("load_summary", run_start_s=run_start_s):
            routing_payload = load_summary(summary_file)
        breakdown_rows = list(routing_payload.get("breakdown_rows", []))
        tries_distribution_rows = list(
            routing_payload.get("tries_distribution_rows", [])
        )
        _log_phase(
            f"Loaded precomputed routing summary from {summary_file}",
            run_start_s=run_start_s,
        )

    reqs_analysis: dict[str, Any] | None = None
    slo_category_rows: list[dict[str, Any]] = []
    if reqs_file is not None:
        if ttft_slo_scale is None or slo_tpot is None:
            _log_phase(
                f"Skipping reqs-side SLO analysis for {reqs_file}: missing "
                f"ttft_slo_scale/slo_tpot (use --ttft-slo-scale and --slo-tpot)",
                run_start_s=run_start_s,
            )
        else:
            with _phase("analyze_saved_requests", run_start_s=run_start_s):
                reqs_analysis = analyze_saved_requests(
                    reqs_file,
                    ttft_slo_scale=float(ttft_slo_scale),
                    slo_tpot=float(slo_tpot),
                    ttft_overhead=float(args.ttft_overhead),
                    routing_overhead=float(args.routing_overhead),
                )
            slo_category_rows = list(reqs_analysis.get("category_rows", []))

    cross_source_consistency: dict[str, Any] | None = None
    if routing_payload is not None and reqs_analysis is not None:
        routing_request_count = (
            routing_payload.get("counts", {}).get("requests_total")
            if isinstance(routing_payload.get("counts"), dict)
            else None
        )
        reqs_request_count = reqs_analysis.get("total_requests")
        if routing_request_count is not None and reqs_request_count is not None:
            cross_source_consistency = {
                "routing_request_count": int(routing_request_count),
                "reqs_request_count": int(reqs_request_count),
                "request_count_delta": int(reqs_request_count) - int(routing_request_count),
                "request_count_match": (
                    int(routing_request_count) == int(reqs_request_count)
                ),
            }

    breakdown_csv = output_prefix.with_suffix(".breakdown.csv")
    tries_distribution_csv = output_prefix.with_suffix(".tries_distribution.csv")
    slo_categories_csv = output_prefix.with_suffix(".slo_categories.csv")
    summary_json = output_prefix.with_suffix(".summary.json")
    zip_path = None
    if args.zip_results:
        zip_path = args.zip_path or output_prefix.with_suffix(".zip")
    legacy_paths = [
        output_prefix.with_suffix(".attempts.csv"),
        output_prefix.with_suffix(".requests.csv"),
        output_prefix.with_suffix(".segment_stats.csv"),
        output_prefix.with_suffix(".device_breakdown.csv"),
    ]
    for legacy_path in legacy_paths:
        if legacy_path.exists():
            legacy_path.unlink()

    if routing_payload is not None:
        with _phase("write_breakdown_csv", run_start_s=run_start_s):
            _write_csv(breakdown_csv, breakdown_rows)
        with _phase("write_tries_distribution_csv", run_start_s=run_start_s):
            _write_csv(tries_distribution_csv, tries_distribution_rows)
    if reqs_analysis is not None:
        with _phase("write_slo_categories_csv", run_start_s=run_start_s):
            _write_csv(slo_categories_csv, slo_category_rows)
    with _phase("write_summary_json", run_start_s=run_start_s):
        summary_payload: dict[str, Any] = {
            "analysis_sources": {
                "prefix": None if prefix is None else str(prefix),
                "events_file": None if events_file is None else str(events_file),
                "events_summary_file": (
                    None if summary_file is None else str(summary_file)
                ),
                "reqs_file": None if reqs_file is None else str(reqs_file),
            },
        }
        if ttft_slo_scale is not None or slo_tpot is not None:
            summary_payload["inferred_or_configured_slo"] = {
                "ttft_slo_scale": ttft_slo_scale,
                "slo_tpot": slo_tpot,
                "ttft_overhead": float(args.ttft_overhead),
                "routing_overhead": float(args.routing_overhead),
            }
        if routing_payload is not None:
            summary_payload.update(routing_payload)
        if reqs_analysis is not None:
            summary_payload["reqs_analysis"] = reqs_analysis
            summary_payload["slo_category_rows"] = slo_category_rows
        if cross_source_consistency is not None:
            summary_payload["cross_source_consistency"] = cross_source_consistency
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, sort_keys=True)
    if zip_path is not None:
        with _phase("write_results_zip", run_start_s=run_start_s):
            files_to_zip = [summary_json]
            if routing_payload is not None:
                files_to_zip.extend([breakdown_csv, tries_distribution_csv])
            if reqs_analysis is not None:
                files_to_zip.append(slo_categories_csv)
            _write_zip(zip_path, files_to_zip)

    if events_file is not None:
        print(f"Loaded events from {events_file}")
    elif summary_file is not None:
        print(f"Loaded routing summary from {summary_file}")
    if reqs_file is not None:
        print(f"Loaded reqs from {reqs_file}")

    if routing_payload is not None:
        if args.batch_only and events_file is not None:
            print("Batch-only mode: exported batch timing and batch-size metrics.")
        elif "counts" in routing_payload:
            counts = routing_payload["counts"]
            if "requests_total" in counts:
                print(
                    "Requests: "
                    f"total={counts['requests_total']} "
                    f"accepted={counts['accepted_requests']} "
                    f"with_retries={counts['requests_with_retries']}"
                )
                print(
                    "Attempts: "
                    f"total={counts['attempts_total']} "
                    f"accepted={counts['accepted_attempts']} "
                    f"rescheduled={counts['rescheduled_attempts']} "
                    f"router_rejected={counts['router_rejected_attempts']}"
                )
                print()
                print(
                    "service_ready = max(router admitted/rescheduling ack, first batch launch), "
                    "where first batch launch = batch.timestamp - batch.elapsed + batch.extra_args.to_launch"
                )
                print(
                    "routing_elapsed_s is the raw router-loop elapsed time from the trace. "
                    "queueing_time_s is arrival-router -> routing_start, and "
                    "routing_compute_time_s is routing_start -> routing_finish."
                )
                print(
                    "batch_kernel_execution_time_s is derived from "
                    "batch.extra_args.to_finish - batch.extra_args.to_launch when present. "
                    "batch_other_control_logic_time_s is the remaining batch.elapsed after "
                    "subtracting scheduling_overhead and kernel execution."
                )

                _print_metric_block(
                    "Accepted Attempt Breakdown",
                    routing_payload["accepted_attempt_metrics"],
                )
                _print_metric_block(
                    "Rejected Attempt Breakdown Before Rescheduling",
                    routing_payload["rescheduled_attempt_metrics"],
                )
                _print_metric_block(
                    "Rejected Attempt Round Trip",
                    routing_payload["rescheduled_round_trip_metrics"],
                )
                _print_metric_block(
                    "Rejected Attempt Engine Breakdown",
                    routing_payload["rescheduled_engine_breakdown_metrics"],
                )
                _print_count_metric_block(
                    "Per-Request Attempt Counts",
                    {
                        "attempts_total": routing_payload["request_metrics"]["attempts_total"],
                        "retries_total": routing_payload["request_metrics"]["retries_total"],
                    },
                )
                _print_metric_block(
                    "Per-Request Latency Summary",
                    {
                        "time_to_service_ready_s": (
                            routing_payload["request_metrics"]["time_to_service_ready_s"]
                        ),
                        "total_routing_overhead_s": (
                            routing_payload["request_metrics"]["total_routing_overhead_s"]
                        ),
                        "first_serving_batch_scheduling_overhead_s": (
                            routing_payload["request_metrics"][
                                "first_serving_batch_scheduling_overhead_s"
                            ]
                        ),
                    },
                )
        batch_metrics = routing_payload.get("batch_metrics")
        if batch_metrics:
            _print_metric_block(
                "Batch Timing Summary",
                {
                    "batch_total_elapsed_s": batch_metrics["batch_total_elapsed_s"],
                    "batch_scheduling_overhead_s": batch_metrics["batch_scheduling_overhead_s"],
                    "batch_execution_time_s": batch_metrics["batch_execution_time_s"],
                    "batch_kernel_execution_time_s": (
                        batch_metrics["batch_kernel_execution_time_s"]
                    ),
                    "batch_other_control_logic_time_s": (
                        batch_metrics["batch_other_control_logic_time_s"]
                    ),
                    "batch_output_processing_elapsed_s": (
                        batch_metrics["batch_output_processing_elapsed_s"]
                    ),
                    "batch_publish_overhead_s": batch_metrics["batch_publish_overhead_s"],
                    "batch_pre_launch_control_logic_time_s": (
                        batch_metrics["batch_pre_launch_control_logic_time_s"]
                    ),
                    "batch_post_kernel_control_logic_time_s": (
                        batch_metrics["batch_post_kernel_control_logic_time_s"]
                    ),
                },
            )
            _print_metric_block(
                "Observed Engine Phase Events",
                batch_metrics["observed_engine_phase_metrics"],
            )
            _print_count_metric_block(
                "Batch Size Distribution",
                batch_metrics["batch_size_distribution_metrics"],
            )

    if reqs_analysis is not None:
        print()
        print(
            "Reqs-side SLO config: "
            f"ttft_slo_scale={reqs_analysis['ttft_slo_scale']} "
            f"slo_tpot={reqs_analysis['slo_tpot']} "
            f"routing_overhead={reqs_analysis['routing_overhead']}"
        )
        print(
            "Reqs-side SLO summary: "
            f"total={reqs_analysis['total_requests']} "
            f"violation_rate={reqs_analysis['slo_violation_rate']:.6f} "
            f"ttft_rate={reqs_analysis['ttft_violation_rate']:.6f} "
            f"tpot_rate={reqs_analysis['tpot_violation_rate']:.6f}"
        )
        _print_reason_block(
            "Reqs-side Violation Categories",
            reqs_analysis["violation_reason_counts"],
            reqs_analysis["violation_reason_rates"],
        )
        _print_reason_block(
            "Reqs-side Finish Reasons",
            reqs_analysis["finish_reason_counts"],
            reqs_analysis["finish_reason_rates"],
        )
    if cross_source_consistency is not None:
        print()
        print(
            "Cross-source request counts: "
            f"routing={cross_source_consistency['routing_request_count']} "
            f"reqs={cross_source_consistency['reqs_request_count']} "
            f"delta={cross_source_consistency['request_count_delta']}"
        )
        if not cross_source_consistency["request_count_match"]:
            print(
                "Warning: routing/request counts do not match across events and reqs inputs."
            )

    if routing_payload is not None and not args.batch_only:
        if attempts_by_request:
            for request_id in args.request_id:
                attempts = attempts_by_request.get(str(request_id))
                if not attempts:
                    print()
                    print(f"Request {request_id}: not found")
                    continue
                _print_request_detail(str(request_id), attempts)
        elif args.request_id:
            print()
            print(
                "Detailed request timelines require a raw events file; "
                "they are unavailable when loading a precomputed summary."
            )

    print()
    if routing_payload is not None:
        print(f"Wrote {breakdown_csv}")
        print(f"Wrote {tries_distribution_csv}")
    if reqs_analysis is not None:
        print(f"Wrote {slo_categories_csv}")
    print(f"Wrote {summary_json}")
    if zip_path is not None:
        print(f"Wrote {zip_path}")


if __name__ == "__main__":
    main()
