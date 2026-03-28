import tqdm
import time
import asyncio
import os
import copy
from typing import Tuple, List, Dict, Any
import subprocess
from dataclasses import dataclass, field, asdict
import numpy as np
import pprint
import json
from itertools import product
import logging
import uuid
import random
import bisect
from collections import Counter, defaultdict, deque
import aiohttp
import httpx
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from motivation.events_analysis import (
    _compute_active_device_series,
    _compute_measured_power_series,
    analyze_events,
    analyze_slo_violation,
    build_active_requests_step,
    count_at_times,
)
from motivation.auto_scaling import eval_auto_scaling
from Dataset.dataset import ArrivalTimes, Requests, Request

from SLOsServe.client_spec import count_client_spec, normalize_client_spec
from SLOsServe.fitting_utils import fit_linear_perf_model
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

FRONTEND_DELAY=0.03
FRONTEND_HEALTH_CHECK_INTERVAL_S = 5.0
REQUEST_TIMEOUT_S = 60.0
REQUEST_CANCEL_GRACE_S = 5.0
BENCHMARK_METRIC_WINDOW_S = 1.0
BENCHMARK_IDLE_POWER_PER_GPU_W = 70.0
BENCHMARK_BATCH_POWER_MATCH_TOLERANCE_S = 0.2

from SLOsServe.perf_model import (
    PerfModel,
    build_piecewise_current_token_hardware_params,
    extract_batch_perf_sample as extract_perf_model_batch_sample,
    fit_piecewise_current_token_model,
    get_easy_name,
    get_model_max_tokens,
    iter_current_token_piece_segments,
)


class BenchmarkOverloadedError(RuntimeError):
    """Raised when the serving stack dies mid-benchmark."""


def _request_id_sort_key(request_id: Any) -> tuple[int, int | str]:
    request_id_str = str(request_id)
    try:
        return (0, int(request_id_str))
    except (TypeError, ValueError):
        return (1, request_id_str)


def _normalize_rejection_reason(reason: Any) -> str | None:
    if reason is None:
        return None
    raw = str(reason).strip()
    if not raw:
        return None
    normalized = raw.upper()
    if normalized in {"CMP", "COMPUTE", "REJECTED-COMPUTE"}:
        return "compute"
    if normalized in {"MEM", "MEMORY", "REJECTED-MEMORY"}:
        return "memory"
    if normalized in {"OOM", "REJECTED-OOM", "OUT_OF_MEMORY"}:
        return "oom"
    if normalized in {"ROUTER", "ROUTER_REJECTION"}:
        return "router"
    if normalized == "UNKNOWN":
        return "unknown"
    return raw.lower()


def _extract_batch_perf_sample(
    event: Dict[str, Any],
) -> tuple[Dict[str, Any], list[tuple[int, int]]] | None:
    if event.get("event_type") != "batch":
        return None

    req_ids = event.get("req_ids")
    computed_tokens = event.get("num_computed_tokens")
    scheduled_tokens = event.get("num_scheduled_tokens")
    if not isinstance(req_ids, list) or not isinstance(computed_tokens, list) or not isinstance(scheduled_tokens, dict):
        return None

    batch = []
    for idx, req_id in enumerate(req_ids):
        try:
            past_tokens = max(0, int(computed_tokens[idx]))
        except (IndexError, TypeError, ValueError):
            past_tokens = 0
        try:
            current_tokens = max(0, int(scheduled_tokens.get(req_id, 0)))
        except (TypeError, ValueError):
            current_tokens = 0
        if past_tokens == 0 and current_tokens == 0:
            continue
        batch.append({
            "past_tokens": past_tokens,
            "scheduled_tokens": current_tokens,
        })

    if not batch:
        return None

    try:
        elapsed = float(event.get("elapsed", 0.0))
    except (TypeError, ValueError):
        elapsed = 0.0
    try:
        scheduling_overhead = float(event.get("scheduling_overhead", 0.0))
    except (TypeError, ValueError):
        scheduling_overhead = 0.0
    try:
        estimated_time = float(event.get("estimated_time", 0.0))
    except (TypeError, ValueError):
        estimated_time = 0.0
    control_estimated_time_raw = event.get("control_estimated_time")
    try:
        control_estimated_time = (
            float(control_estimated_time_raw)
            if control_estimated_time_raw is not None else None
        )
    except (TypeError, ValueError):
        control_estimated_time = None

    try:
        device_id = int(event.get("device_id", 0))
    except (TypeError, ValueError):
        device_id = 0
    try:
        batch_id = int(event.get("batch_id", -1))
    except (TypeError, ValueError):
        batch_id = -1
    try:
        timestamp = float(event.get("timestamp", 0.0))
    except (TypeError, ValueError):
        timestamp = 0.0

    row = {
        "device_id": device_id,
        "batch_id": batch_id,
        "timestamp": timestamp,
        "batch_size": len(batch),
        "total_current_tokens": sum(item["scheduled_tokens"] for item in batch),
        "total_past_tokens": sum(item["past_tokens"] for item in batch),
        "estimated_time": max(0.0, estimated_time),
        "measured_time": max(0.0, elapsed - scheduling_overhead),
        "elapsed_time": max(0.0, elapsed),
        "scheduling_overhead": max(0.0, scheduling_overhead),
        "estimated_full_time": max(
            0.0,
            control_estimated_time
            if control_estimated_time is not None
            else estimated_time + scheduling_overhead,
        ),
    }
    if control_estimated_time is not None:
        row["control_estimated_time"] = max(0.0, control_estimated_time)

    return row, [
        (int(item["past_tokens"]), int(item["scheduled_tokens"]))
        for item in batch
    ]


def _extract_batch_perf_error_row(event: Dict[str, Any]) -> Dict[str, Any] | None:
    sample = _extract_batch_perf_sample(event)
    if sample is None:
        return None
    row, _ = sample
    return row


def _collect_batch_fit_samples(
    events: List[Dict[str, Any]],
) -> list[tuple[list[tuple[int, int]], float]]:
    batch_times: list[tuple[list[tuple[int, int]], float]] = []

    for event in events:
        sample = _extract_batch_perf_sample(event)
        if sample is None:
            continue
        row, batch = sample
        batch_times.append((batch, float(row["measured_time"])))

    return batch_times


def _collect_batch_perf_error_rows(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    error_rows: List[Dict[str, Any]] = []

    for event in events:
        row = _extract_batch_perf_error_row(event)
        if row is None:
            continue
        error_s = float(row["estimated_time"] - row["measured_time"])
        abs_error_s = float(abs(error_s))
        measured_time = float(row["measured_time"])
        relative_error = None
        abs_relative_error = None
        estimated_over_measured = None
        if measured_time > 0.0:
            relative_error = float(error_s / measured_time)
            abs_relative_error = float(abs_error_s / measured_time)
            estimated_over_measured = float(row["estimated_time"] / measured_time)
        row.update({
            "error_s": error_s,
            "abs_error_s": abs_error_s,
            "relative_error": relative_error,
            "abs_relative_error": abs_relative_error,
            "estimated_over_measured": estimated_over_measured,
        })
        full_error_s = float(row["estimated_full_time"] - row["elapsed_time"])
        abs_full_error_s = float(abs(full_error_s))
        elapsed_time = float(row["elapsed_time"])
        full_relative_error = None
        abs_full_relative_error = None
        estimated_full_over_elapsed = None
        if elapsed_time > 0.0:
            full_relative_error = float(full_error_s / elapsed_time)
            abs_full_relative_error = float(abs_full_error_s / elapsed_time)
            estimated_full_over_elapsed = float(
                row["estimated_full_time"] / elapsed_time
            )
        row.update({
            "full_error_s": full_error_s,
            "abs_full_error_s": abs_full_error_s,
            "full_relative_error": full_relative_error,
            "abs_full_relative_error": abs_full_relative_error,
            "estimated_full_over_elapsed": estimated_full_over_elapsed,
        })
        error_rows.append(row)

    return error_rows


def _summarize_batch_scheduling_overhead_rows(
    error_rows: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    if not error_rows:
        return None

    summary = {
        "overhead_s": _summarize_distribution([
            float(row["scheduling_overhead"]) for row in error_rows
        ]),
        "relative_to_measured_time": _summarize_distribution([
            float(row["scheduling_overhead"] / row["measured_time"])
            for row in error_rows
            if float(row["measured_time"]) > 0.0
        ]),
    }
    return summary


def _summarize_batch_scheduling_overhead(
    events: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    return _summarize_batch_scheduling_overhead_rows(
        _collect_batch_perf_error_rows(events)
    )


def _summarize_distribution(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}

    data = np.asarray(values, dtype=float)
    percentiles = {
        "p01": 1,
        "p05": 5,
        "p10": 10,
        "p25": 25,
        "p50": 50,
        "p75": 75,
        "p90": 90,
        "p95": 95,
        "p99": 99,
    }
    summary = {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }
    for key, percentile in percentiles.items():
        summary[key] = float(np.percentile(data, percentile))
    return summary


def _write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return output_path


def _save_estimated_vs_measured_figure(
    estimated_times: List[float],
    measured_times: List[float],
    path: str | Path,
    *,
    title: str,
    x_label: str = "Measured Time (s)",
    y_label: str = "Estimated Time (s)",
) -> Path | None:
    if not estimated_times or not measured_times:
        return None

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    estimated = np.asarray(estimated_times, dtype=float)
    measured = np.asarray(measured_times, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
    ax.scatter(measured, estimated, s=14, alpha=0.7)
    lo = float(min(measured.min(), estimated.min()))
    hi = float(max(measured.max(), estimated.max()))
    if hi > lo:
        ax.plot([lo, hi], [lo, hi], "--r", linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _compute_regressed_hardware_params_delta(
    old_hardware_params: Any,
    regressed_hardware_params: Any,
) -> Any:
    if isinstance(old_hardware_params, list) and isinstance(regressed_hardware_params, list):
        return [
            float(new_param - old_param)
            for old_param, new_param in zip(
                old_hardware_params,
                regressed_hardware_params,
            )
        ]

    if not isinstance(old_hardware_params, dict) or not isinstance(regressed_hardware_params, dict):
        return None
    if (
        old_hardware_params.get("type") != "piecewise_current_tokens"
        or regressed_hardware_params.get("type") != "piecewise_current_tokens"
    ):
        return None
    if old_hardware_params.get("breakpoints") != regressed_hardware_params.get("breakpoints"):
        return None

    old_segments = old_hardware_params.get("segment_params", {})
    new_segments = regressed_hardware_params.get("segment_params", {})
    if not isinstance(old_segments, dict) or not isinstance(new_segments, dict):
        return None

    delta_segments: dict[str, list[float]] = {}
    for segment_key, old_segment_params in old_segments.items():
        new_segment_params = new_segments.get(segment_key)
        if not isinstance(old_segment_params, list) or not isinstance(new_segment_params, list):
            return None
        delta_segments[segment_key] = [
            float(new_param - old_param)
            for old_param, new_param in zip(old_segment_params, new_segment_params)
        ]

    return build_piecewise_current_token_hardware_params(
        delta_segments,
        breakpoints=old_hardware_params["breakpoints"],
    )


def _piecewise_perf_model_breakpoints(
    problem: "Problem",
) -> list[int] | None:
    if problem.perf_model_piecewise_breakpoints:
        return [int(point) for point in problem.perf_model_piecewise_breakpoints]
    if problem.enable_piecewise_perf_model_regression:
        return [512, 2048]
    return None


def _materialize_piecewise_regressed_hardware_params(
    old_perf_model: PerfModel,
    piecewise_fit_result: Dict[str, Any],
) -> tuple[Dict[str, Any], list[str], list[str]]:
    breakpoints = piecewise_fit_result["breakpoints"]
    segment_params: dict[str, list[float]] = {}
    regressed_segment_keys: list[str] = []
    fallback_segment_keys: list[str] = []

    for descriptor in iter_current_token_piece_segments(breakpoints):
        segment_key = descriptor["segment_key"]
        segment_report = piecewise_fit_result["segments"].get(segment_key)
        if segment_report is not None:
            segment_params[segment_key] = [
                float(param)
                for param in segment_report["hardware_params"]
            ]
            regressed_segment_keys.append(segment_key)
            continue

        if descriptor["max_current_tokens"] is None:
            fallback_hint = descriptor["min_current_tokens"]
        else:
            fallback_hint = descriptor["max_current_tokens"]
        segment_params[segment_key] = old_perf_model.get_active_hardware_params(
            int(fallback_hint)
        )
        fallback_segment_keys.append(segment_key)

    piecewise_hardware_params = build_piecewise_current_token_hardware_params(
        segment_params,
        breakpoints=breakpoints,
    )
    piecewise_hardware_params["fitted_segments"] = list(regressed_segment_keys)
    piecewise_hardware_params["fallback_segments"] = list(fallback_segment_keys)
    return piecewise_hardware_params, regressed_segment_keys, fallback_segment_keys


def _log_perf_model_errors_from_batch_events(
    problem: "Problem",
    events: List[Dict[str, Any]],
    output_path: str | Path,
    *,
    event_file: str,
    include_time_lists: bool = False,
    draw_figure: bool = True,
    figure_path: str | Path | None = None,
    regression_figure_path: str | Path | None = None,
    full_elapsed_figure_path: str | Path | None = None,
) -> Dict[str, Any] | None:
    error_rows = _collect_batch_perf_error_rows(events)
    if not error_rows:
        return None
    batch_times = _collect_batch_fit_samples(events)
    if not batch_times:
        return None
    regression_rows = []
    for event in events:
        sample = extract_perf_model_batch_sample(
            event,
            subtract_scheduling_overhead=True,
        )
        if sample is not None:
            regression_rows.append(sample)
    if not regression_rows:
        return None

    old_perf_model = PerfModel.get_perf_model(
        problem.model_name,
        problem.perf_model_task,
    )
    old_hardware_params = old_perf_model.describe_hardware_params()
    piecewise_breakpoints = _piecewise_perf_model_breakpoints(problem)

    if piecewise_breakpoints is not None:
        fit_result = fit_piecewise_current_token_model(
            regression_rows,
            breakpoints=piecewise_breakpoints,
        )
        (
            regressed_hardware_params,
            regressed_segment_keys,
            fallback_segment_keys,
        ) = _materialize_piecewise_regressed_hardware_params(
            old_perf_model,
            fit_result,
        )
        regression_model_type = "piecewise_current_tokens"
        regression_stats = {
            "aggregate": fit_result["aggregate_stats"],
            "segments": {
                segment_key: {
                    "label": segment_report["label"],
                    "used_sample_count": segment_report["used_sample_count"],
                    "hardware_params": segment_report["hardware_params"],
                    "fit_stats": segment_report["fit_stats"],
                    "fitted_estimator_stats": (
                        segment_report["fitted_estimator_stats"]
                    ),
                    "existing_estimator_stats": (
                        segment_report["existing_estimator_stats"]
                    ),
                }
                for segment_key, segment_report in fit_result["segments"].items()
            },
        }
        regression_predicted_times = fit_result["predicted_times"]
        regression_measured_times = [
            float(row["measured_time"]) for row in regression_rows
        ]
    else:
        fit_result = fit_linear_perf_model(batch_times)
        regressed_hardware_params = [
            float(param) for param in fit_result["hardware_params"]
        ]
        regressed_segment_keys = []
        fallback_segment_keys = []
        regression_model_type = "linear"
        regression_stats = fit_result["stats"]
        regression_predicted_times = fit_result["predicted_times"]
        regression_measured_times = fit_result["measured_times"]

    regressed_hardware_params_delta = _compute_regressed_hardware_params_delta(
        old_hardware_params,
        regressed_hardware_params,
    )
    empirical_scheduling_overhead = _summarize_batch_scheduling_overhead_rows(
        error_rows
    )
    estimated_times = [float(row["estimated_time"]) for row in error_rows]
    measured_times = [float(row["measured_time"]) for row in error_rows]
    estimated_full_times = [
        float(row["estimated_full_time"]) for row in error_rows
    ]
    elapsed_times = [float(row["elapsed_time"]) for row in error_rows]

    summary = {
        "model_name": problem.model_name,
        "length_pattern": problem.length_pattern,
        "perf_model_task": problem.perf_model_task,
        "store_prefix": problem.store_prefix,
        "event_file": event_file,
        "perf_model_err": float(problem.perf_model_err),
        "configured_scheduling_overhead_s": float(problem.scheduling_overhead),
        "old_hardware_params": old_hardware_params,
        "regression_model_type": regression_model_type,
        "regressed_hardware_params": regressed_hardware_params,
        "regressed_hardware_params_delta": regressed_hardware_params_delta,
        "regression_stats": regression_stats,
        "relative_error_denominator": "measured_time",
        "estimated_minus_measured_s": _summarize_distribution(
            [float(row["error_s"]) for row in error_rows]
        ),
        "abs_estimated_minus_measured_s": _summarize_distribution(
            [float(row["abs_error_s"]) for row in error_rows]
        ),
        "estimated_minus_measured_relative": _summarize_distribution([
            float(row["relative_error"])
            for row in error_rows
            if row["relative_error"] is not None
        ]),
        "abs_estimated_minus_measured_relative": _summarize_distribution([
            float(row["abs_relative_error"])
            for row in error_rows
            if row["abs_relative_error"] is not None
        ]),
        "estimated_with_overhead_minus_elapsed_s": _summarize_distribution(
            [float(row["full_error_s"]) for row in error_rows]
        ),
        "abs_estimated_with_overhead_minus_elapsed_s": _summarize_distribution(
            [float(row["abs_full_error_s"]) for row in error_rows]
        ),
        "estimated_with_overhead_minus_elapsed_relative": (
            _summarize_distribution([
                float(row["full_relative_error"])
                for row in error_rows
                if row["full_relative_error"] is not None
            ])
        ),
        "abs_estimated_with_overhead_minus_elapsed_relative": (
            _summarize_distribution([
                float(row["abs_full_relative_error"])
                for row in error_rows
                if row["abs_full_relative_error"] is not None
            ])
        ),
    }
    if piecewise_breakpoints is not None:
        summary["regression_breakpoints"] = list(piecewise_breakpoints)
        summary["regressed_hardware_params_fitted_segments"] = regressed_segment_keys
        summary["regressed_hardware_params_fallback_segments"] = fallback_segment_keys
    if empirical_scheduling_overhead is not None:
        summary["empirical_scheduling_overhead"] = empirical_scheduling_overhead
    if include_time_lists:
        summary["estimated_time_list"] = estimated_times
        summary["measured_time_list"] = measured_times

    plotted_figure_path = None
    if draw_figure:
        plotted_figure_path = _save_estimated_vs_measured_figure(
            estimated_times,
            measured_times,
            figure_path or Path(output_path).with_suffix(".png"),
            title=f"{get_easy_name(problem.model_name)} [{problem.perf_model_task}]",
        )
        if plotted_figure_path is not None:
            summary["estimated_vs_measured_figure_path"] = str(plotted_figure_path)

    full_elapsed_plot_path = None
    if draw_figure:
        full_elapsed_plot_path = _save_estimated_vs_measured_figure(
            estimated_full_times,
            elapsed_times,
            full_elapsed_figure_path
            or Path(output_path).with_name(
                f"{Path(output_path).stem}.elapsed.png"
            ),
            title=(
                f"{get_easy_name(problem.model_name)} "
                f"[{problem.perf_model_task}] + Overhead"
            ),
            x_label="Elapsed Time (s)",
            y_label="Estimated + Overhead (s)",
        )
        if full_elapsed_plot_path is not None:
            summary["estimated_with_overhead_vs_elapsed_figure_path"] = str(
                full_elapsed_plot_path
            )

    regression_plot_path = None
    if draw_figure:
        regression_plot_path = _save_estimated_vs_measured_figure(
            regression_predicted_times,
            regression_measured_times,
            regression_figure_path
            or Path(output_path).with_name(
                f"{Path(output_path).stem}.regression.png"
            ),
            title=(
                f"{get_easy_name(problem.model_name)} "
                f"[{problem.perf_model_task}] "
                f"{'Piecewise Regression' if piecewise_breakpoints is not None else 'Regression'}"
            ),
            y_label="Regressed Time (s)",
        )
        if regression_plot_path is not None:
            summary["regression_figure_path"] = str(regression_plot_path)

    rows = [{
        "record_type": "summary",
        **summary,
    }]
    rows.extend({
        "record_type": "batch_error",
        **row,
    } for row in error_rows)
    output_path = _write_jsonl(output_path, list(rows))

    return {
        "path": str(output_path),
        "summary": summary,
        "figure_path": (
            str(plotted_figure_path)
            if plotted_figure_path is not None else None
        ),
        "full_elapsed_figure_path": (
            str(full_elapsed_plot_path)
            if full_elapsed_plot_path is not None else None
        ),
        "regression_figure_path": (
            str(regression_plot_path)
            if regression_plot_path is not None else None
        ),
    }


@dataclass
class Problem:
    
    # problem
    model_name: str = 'Qwen/Qwen2.5-7B-Instruct'
    arrival_pattern: str = 'azure_code_23'
    length_pattern: str = 'azure_code_23'
    trace_spec: str = 'azure_code_23'
    perf_model_task: str = 'azure_code_23'
    window: str = '0:10'
    perf_model_err: float = 1.0 # the factor between used performance model and real performance model.
    scheduling_overhead: float = 0.0
    
    # runtime config
    load_scale: float = 1.0 
    n_devices: int = 2  # logical replicas
    tensor_parallel_size: int = 1
    
    # slo model
    ttft_slo_scale: float = 1.0
    slo_ttft_per_token: float = 2e-4
    slo_ttft_constant: float = 0.1
    slo_tpot: float = 0.05
    slo_routing_overhead: float = 0.16
    enable_session_replay: bool = False
    session_pause_s: float = 0.0
     
    # profit model 
    profit_per_input_token: float = 0.0
    profit_per_output_token: float = 0.0
    profit_base: float = 1.0
    
    # scheduling mode
    admission_mode: str = 'arrival'
    
    # policies
    routing_policy: str = 'slo'
    routing_kwargs: dict = field(default_factory=lambda: {'hardware_params': [4.1e-5, 0, 1.3e-2], 'tpot': 0.05, 'device_mem': 16384, 'block_size': 16})
    routing_overhead: float = -1.0
    routing_fallback_policy: str = 'asap'
    
    scheduling_policy: str = 'vllm'
    scheduling_kwargs: dict = field(default_factory=lambda: {'max_num_seqs': 128, 'max_num_batched_tokens': 512, 'long_prefill_token_threshold': 256, 'enable_chunked_prefill': False, 'enable_admission': True, 'allow_rejection': True})
    
    # store_prefix
    store_prefix: str = 'problem'
    record_events: bool = False
    log_perf_model_errors: bool = True
    include_perf_model_time_lists: bool = False
    draw_perf_model_error_figure: bool = True
    enable_piecewise_perf_model_regression: bool = False
    perf_model_piecewise_breakpoints: list[int] | None = None

    def get_expected_profit(self, input_length: int):
        return float(self.profit_per_input_token * input_length + self.profit_per_output_token * average_output_length + self.profit_base)


def compute_ttft_slo(
    prompt_tokens: int,
    cached_tokens: int,
    *,
    slo_ttft_per_token: float,
    slo_ttft_constant: float,
    slo_routing_overhead: float,
) -> float:
    new_tokens = max(int(prompt_tokens) - int(cached_tokens), 0)
    return (
        float(slo_ttft_constant)
        + float(slo_ttft_per_token) * new_tokens
        + float(slo_routing_overhead)
    )


def _perf_model_regression_suffix(
    enable_piecewise_perf_model_regression: bool,
    perf_model_piecewise_breakpoints: list[int] | None,
) -> str:
    if perf_model_piecewise_breakpoints:
        normalized = "-".join(str(int(point)) for point in perf_model_piecewise_breakpoints)
        return f"_pmreg_piecewise_{normalized}"
    if enable_piecewise_perf_model_regression:
        return "_pmreg_piecewise_default"
    return ""


def _split_trace_components(trace_spec: str) -> list[str]:
    trace_spec = str(trace_spec).strip()
    if not trace_spec:
        raise ValueError("trace spec must be non-empty")

    if "+" in trace_spec:
        components = [component.strip() for component in trace_spec.split("+")]
        components = [component for component in components if component]
        if not components:
            raise ValueError(f"invalid mixed trace spec: {trace_spec}")
        return components

    if trace_spec.count(":") <= 1:
        return [trace_spec]

    if "-" in trace_spec:
        components = [component.strip() for component in trace_spec.split("-")]
        if components and all(component and component.count(":") <= 1 for component in components):
            return components
        raise ValueError(
            "ambiguous mixed trace spec using '-'. "
            "Use '+' between components when dataset names contain '-': "
            f"{trace_spec}"
        )

    raise ValueError(
        "invalid mixed trace spec. Expected TRACE or TRACE+TRACE where each TRACE is "
        "LENGTH[:ARRIVAL]. "
        f"Got: {trace_spec}"
    )


def _parse_trace_spec(trace_spec: str) -> list[tuple[str, str]]:
    components: list[tuple[str, str]] = []
    for component in _split_trace_components(trace_spec):
        if ":" in component:
            length_pattern, arrival_pattern = component.split(":", 1)
        else:
            length_pattern = arrival_pattern = component
        length_pattern = length_pattern.strip()
        arrival_pattern = arrival_pattern.strip()
        if not length_pattern or not arrival_pattern:
            raise ValueError(f"invalid trace component: {component}")
        components.append((length_pattern, arrival_pattern))
    return components


def _perf_model_task_for_trace_spec(trace_spec: str) -> str:
    components = _parse_trace_spec(trace_spec)
    unique_length_patterns = {length_pattern for length_pattern, _ in components}
    if len(unique_length_patterns) == 1:
        return components[0][0]
    return "default"


def _load_trace_inputs(
    trace_spec: str,
    *,
    model_name: str,
    load_scale: float,
    window: str,
) -> tuple[list[Request], list[float], list[tuple[str, str]]]:
    components = _parse_trace_spec(trace_spec)
    window_start, window_end = window.split(":")
    max_tokens = get_model_max_tokens(model_name)

    if len(components) == 1:
        length_pattern, arrival_pattern = components[0]
        arrival_times = ArrivalTimes.load(
            arrival_pattern,
            load_scale,
            window_start=window_start,
            window_end=window_end,
        ).arrival_times
        requests = Requests.load(
            length_pattern,
            window_start=0,
            window_end=len(arrival_times),
            max_tokens=max_tokens,
        ).requests
        n_items = min(len(requests), len(arrival_times))
        requests = [copy.deepcopy(request) for request in requests[:n_items]]
        arrival_times = list(arrival_times[:n_items])
        return requests, arrival_times, components

    merged_pairs: list[tuple[float, int, Request]] = []
    for source_idx, (length_pattern, arrival_pattern) in enumerate(components):
        component_arrivals = ArrivalTimes.load(
            arrival_pattern,
            load_scale,
            window_start=window_start,
            window_end=window_end,
        ).arrival_times
        component_requests = Requests.load(
            length_pattern,
            window_start=0,
            window_end=len(component_arrivals),
            max_tokens=max_tokens,
        ).requests
        n_items = min(len(component_requests), len(component_arrivals))
        component_arrivals = list(component_arrivals[:n_items])
        component_requests = [copy.deepcopy(request) for request in component_requests[:n_items]]
        for request in component_requests:
            if request.session_id is not None:
                request.session_id = f"mix{source_idx}:{request.session_id}"
        logger.info(
            "Mixed trace component %d: lengths=%s arrivals=%s kept=%d",
            source_idx,
            length_pattern,
            arrival_pattern,
            n_items,
        )
        merged_pairs.extend(
            (float(arrival_time), source_idx, request)
            for arrival_time, request in zip(component_arrivals, component_requests)
        )

    merged_pairs.sort(key=lambda item: (item[0], item[1]))
    merged_arrival_times = [arrival_time for arrival_time, _, _ in merged_pairs]
    merged_requests = [request for _, _, request in merged_pairs]
    return merged_requests, merged_arrival_times, components


def _build_request_payload(
    *,
    model_name: str,
    prompt: str | list[int],
    input_length: int,
    output_length: int,
    zero_load_ttft: float,
    cached_tokens: int,
    session_id: str | None,
    ttft_slo: float,
    slo_tpot: float,
    expected_profit: float,
    request_id: str,
) -> dict[str, Any]:
    vllm_xargs = {
        'input_length': input_length,
        'output_length': output_length,
        'zero_load_ttft': zero_load_ttft,
        'cached_tokens': cached_tokens,
        'slo_ttft': ttft_slo,
        'slo_tpot': slo_tpot,
        'profit': expected_profit,
        'request_id': request_id,
    }
    if session_id is not None:
        vllm_xargs['session_id'] = session_id
    return {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": output_length,
        "stream": True,
        "ignore_eos": True,
        "vllm_xargs": vllm_xargs,
    }


def _split_ready_request_indices(
    pending_indices: list[int],
    requests: list[Request],
    elapsed_time: float,
    session_ready_at: dict[str, float],
    enable_session_replay: bool,
) -> tuple[list[int], list[int]]:
    if not enable_session_replay:
        return list(pending_indices), []

    ready_indices: list[int] = []
    blocked_indices: list[int] = []
    occupied_sessions: set[str] = set()
    for idx in pending_indices:
        session_id = getattr(requests[idx], "session_id", None)
        if not session_id:
            ready_indices.append(idx)
            continue
        ready_at = session_ready_at.get(session_id, 0.0)
        if session_id in occupied_sessions or elapsed_time < ready_at:
            blocked_indices.append(idx)
            continue
        ready_indices.append(idx)
        occupied_sessions.add(session_id)
    return ready_indices, blocked_indices
    
@dataclass
class ExecutionResult:
    request: Request
    timestamps: List[float]
    request_id: str
    rejection_reason: str | None = None
    slo_result: str | None = None
    laxities: List[float] = field(default_factory=list)
    expected_finish_time: List[float] = field(default_factory=list)
    
    
@dataclass
class ExecutionResults:
    problem: Problem
    execution_results: List[ExecutionResult]
    results: Dict[str, Any]
    event_file: str
    energy_consumption: float = field(default=0.0)
    per_gpu_energy_consumption: List[float] = field(default_factory=list)
    
    # stats
    slo_violation_rate: float = field(init=False)
    profit: float = field(init=False)
    
    def get_slo_result(self, exec_result: ExecutionResult):
        slo_ttft = compute_ttft_slo(
            exec_result.request.input_length,
            exec_result.request.cached_length,
            slo_ttft_per_token=self.problem.slo_ttft_per_token,
            slo_ttft_constant=self.problem.slo_ttft_constant,
            slo_routing_overhead=self.problem.slo_routing_overhead,
        )
        # print('slo_ttft', slo_ttft, 'input_length', exec_result.request.input_length)
        expected_finish_time = [exec_result.timestamps[0], exec_result.timestamps[0] + slo_ttft]
        
        for _ in range(exec_result.request.output_length - 1):
            expected_finish_time.append(self.problem.slo_tpot + expected_finish_time[-1])
        
        exec_result.expected_finish_time = expected_finish_time
        
        if len(exec_result.timestamps) < len(expected_finish_time):
            return 'unfinished'
        
        if not len(exec_result.timestamps) == len(expected_finish_time):
            logger.warning(f"Request {exec_result.request_id} has {len(exec_result.timestamps)} timestamps but {len(expected_finish_time)} expected finish times")
            exec_result.timestamps = exec_result.timestamps[:len(expected_finish_time)]
        
        laxities = np.array(exec_result.timestamps) - np.array(expected_finish_time)
        exec_result.laxities = laxities.tolist()
        
        for i in range(len(expected_finish_time)):
            if exec_result.timestamps[i] > expected_finish_time[i]:
                return 'slo_violation'
        return 'slo_attained'
        
    def __post_init__(self):
        from collections import Counter
        for exec_result in self.execution_results:
            exec_result.slo_result = self.get_slo_result(exec_result)
        slo_results = [exec_result.slo_result for exec_result in self.execution_results]
        print(f"SLO results histogram: {Counter(slo_results)}")
        is_slo_violation = [slo_result != 'slo_attained' for slo_result in slo_results]
        self.slo_violation_rate = sum(is_slo_violation) / len(is_slo_violation)
        
        profits = [self.problem.profit_per_input_token * exec_result.request.input_length +\
            self.problem.profit_per_output_token * exec_result.request.output_length +\
                self.problem.profit_base for exec_result in self.execution_results]
    
        self.profit = (np.array(profits) * 1 - np.array(is_slo_violation)).sum() / len(self.execution_results)


@dataclass
class TerminalRunResult:
    profit: float
    results: Dict[str, Any]
    event_file: str
    energy_consumption: float = 0.0
    per_gpu_energy_consumption: List[float] = field(default_factory=list)


def _make_overload_run_result(
    problem: Problem,
    *,
    requested_n_devices: int,
    error: Exception,
) -> TerminalRunResult:
    trace_spec = problem.trace_spec or problem.length_pattern
    _, arrival_times, _ = _load_trace_inputs(
        trace_spec,
        model_name=problem.model_name,
        load_scale=problem.load_scale,
        window=problem.window,
    )
    if len(arrival_times) > 1 and arrival_times[-1] > arrival_times[0]:
        rps = len(arrival_times) / (arrival_times[-1] - arrival_times[0])
    else:
        rps = 0.0
    request_count = len(arrival_times)
    rr_sliced = problem.n_devices != requested_n_devices

    event_prefix = f"{problem.store_prefix}.overloaded"
    event_path = f"{event_prefix}.events.jsonl"
    with open(event_path, "w", encoding="utf-8") as f:
        json.dump([
            {
                "event_type": "benchmark_terminal",
                "timestamp": time.time(),
                "status": "overloaded",
                "reason": "server_unhealthy",
                "error_type": type(error).__name__,
                "error": str(error),
                "requested_n_device": requested_n_devices,
                "effective_n_device": problem.n_devices,
            }
        ], f, indent=4)

    zero_metrics = [0.0 for _ in range(max(0, int(problem.n_devices)))]
    return TerminalRunResult(
        profit=-1.0,
        results={
            "rps": rps,
            "requested_n_device": requested_n_devices,
            "effective_n_device": problem.n_devices,
            "rr_sliced": rr_sliced,
            "rr_slice_kept_request_count": request_count,
            "rr_slice_total_request_count": request_count,
            "slo_attainment_rate": 0.0,
            "run_status": "overloaded",
            "overloaded": True,
            "rejection_reason_counts": {},
            "rejected_request_reasons": {},
            "overload_reason": "server_unhealthy",
            "overload_error_type": type(error).__name__,
            "overload_error": str(error),
            "extra_metrics": {
                "energy_consumption_active": 0.0,
                "energy_consumption_non_idle": 0.0,
                "per_server_energy_consumption": zero_metrics.copy(),
                "per_server_power": zero_metrics.copy(),
                "per_server_rps": zero_metrics.copy(),
                "benchmark_figure_replay": {},
                "window_time_pct_vs_active_requests_figure": None,
                "window_time_pct_vs_active_requests_figure_pdf": None,
                "power_vs_active_servers_and_batch_tokens_figure": None,
                "power_vs_active_servers_and_batch_tokens_figure_pdf": None,
            },
        },
        event_file=event_prefix,
    )
    

async def run_request(client: httpx.AsyncClient,
                    request_id: str,
                    model_name: str,
                    prompt: str | list[int], 
                    input_length: int,
                    output_length: int,
                    zero_load_ttft: float,
                    cached_tokens: int,
                    session_id: str | None,
                    ttft_slo: float,
                    slo_tpot: float, 
                    expected_profit: float,
                    real_arrival_times: dict) -> Tuple[bool, str | None, str, List[float]]:
        real_arrival_times[request_id] = time.time()
        timestamps = []
        response_text = ""
        rejection_reason: str | None = None
        
        headers = {
            "Content-Type": "application/json",
            "X-Request-Id": request_id
        }
        payload = _build_request_payload(
            model_name=model_name,
            prompt=prompt,
            input_length=input_length,
            output_length=output_length,
            zero_load_ttft=zero_load_ttft,
            cached_tokens=cached_tokens,
            session_id=session_id,
            ttft_slo=ttft_slo,
            slo_tpot=slo_tpot,
            expected_profit=expected_profit,
            request_id=request_id,
        )
        
        chunks = []
        is_rejected = False
        async with client.stream("POST",
                            '/v1/completions',
                            json=payload,
                            headers=headers) as response:
            logger.info(f"Streaming response opened: request_id={request_id}, status_code={response.status_code}")
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line is None:
                    continue
                logger.debug(f"Streaming line for request_id={request_id}, line_len={len(line)}")
                if not line:
                    # SSE message separator (blank line)
                    continue
                if line.strip() == "[done]":
                    break
                # Support both 'data: {...}' and raw '{...}' payloads
                payload_text = line[6:] if line.startswith('data: ') else line
                try:
                    obj = json.loads(payload_text)
                    if 'finish_reason' in obj:
                        finish_reason = str(obj['finish_reason'] or '')
                        finish_reason_lower = finish_reason.lower()
                        is_rejected = 'reject' in finish_reason_lower
                        if is_rejected:
                            rejection_reason = _normalize_rejection_reason(
                                obj.get('rejection_reason')
                                or obj.get('stop_reason')
                                or (
                                    'ROUTER'
                                    if finish_reason_lower == 'router_rejection'
                                    else None
                                )
                            )
                        if obj['finish_reason'] == 'error':
                            raise RuntimeError(obj.get('error', 'backend error'))
                except RuntimeError:
                    raise
                except Exception as e:
                    if not 'done' in line.lower():
                        logger.error(f"Error parsing SSE line for request_id={request_id}: {e}, line: {line}")
                    timestamps.append(time.time())
                    continue
                n_tokens = 1
                if 'token_ids' in obj:
                    n_tokens = len(obj['token_ids'])
                chunk_ts = obj.get('timestamp', None)
                if not isinstance(chunk_ts, (int, float)):
                    chunk_ts = None
                for _ in range(n_tokens):
                    timestamps.append(float(chunk_ts) if chunk_ts is not None else time.time())
                chunks.append(obj)
            logger.info(f"Streaming response finished: request_id={request_id}")
        # print(f'Request {request_id} finished with {len(timestamps)} timestamps IL: {input_length} OL: {output_length} .')
        return is_rejected, rejection_reason, response_text, timestamps

def summarize_energy_events(
    events: List[Dict[str, Any] | Any],
    n_devices: int | None = None,
) -> tuple[List[float], float]:
    per_gpu: Dict[int, float] = {}
    limit = None if n_devices is None else max(0, int(n_devices))
    for event in events:
        if _event_value(event, "event_type") != "energy":
            continue
        device_id = int(_event_value(event, "device_id", 0))
        if device_id < 0:
            continue
        if limit is not None and device_id >= limit:
            continue
        per_gpu[device_id] = per_gpu.get(device_id, 0.0) + float(
            _event_value(event, "energy", 0.0) or 0.0)
    if not per_gpu:
        if limit is not None:
            return [0.0 for _ in range(limit)], 0.0
        return [], 0.0
    if limit is not None:
        ordered = [per_gpu.get(i, 0.0) for i in range(limit)]
    else:
        ordered = [per_gpu.get(i, 0.0) for i in range(max(per_gpu) + 1)]
    return ordered, sum(ordered)


def _event_value(event: Dict[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(event, dict):
        return event.get(key, default)
    return getattr(event, key, default)


def _summarize_per_server_energy_metrics(
    events: List[Dict[str, Any] | Any],
    n_devices: int,
) -> Dict[str, List[float]]:
    per_server_energy = [0.0 for _ in range(max(0, int(n_devices)))]
    per_server_elapsed = [0.0 for _ in range(len(per_server_energy))]
    per_server_power_samples: List[List[float]] = [
        [] for _ in range(len(per_server_energy))
    ]

    for event in events:
        if _event_value(event, "event_type") != "energy":
            continue
        try:
            device_id = int(_event_value(event, "device_id", -1))
        except (TypeError, ValueError):
            continue
        if not 0 <= device_id < len(per_server_energy):
            continue

        try:
            energy = float(_event_value(event, "energy", 0.0) or 0.0)
        except (TypeError, ValueError):
            energy = 0.0
        try:
            power = float(_event_value(event, "power", 0.0) or 0.0)
        except (TypeError, ValueError):
            power = 0.0

        per_server_energy[device_id] += energy
        if power > 0.0:
            per_server_power_samples[device_id].append(power)
            if energy > 0.0:
                per_server_elapsed[device_id] += energy / power

    per_server_power: List[float] = []
    for energy, elapsed, power_samples in zip(
        per_server_energy,
        per_server_elapsed,
        per_server_power_samples,
    ):
        if elapsed > 0.0:
            per_server_power.append(float(energy / elapsed))
        elif power_samples:
            per_server_power.append(float(np.mean(power_samples)))
        else:
            per_server_power.append(0.0)

    return {
        "per_server_energy_consumption": per_server_energy,
        "per_server_power": per_server_power,
    }


def _make_time_bins(
    start_time: float,
    end_time: float,
    window_size: float,
) -> np.ndarray:
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if end_time <= start_time:
        end_time = start_time + window_size
    n_windows = max(1, int(np.ceil((end_time - start_time) / window_size)))
    return start_time + np.arange(n_windows + 1, dtype=np.float64) * window_size


def _resolve_benchmark_window_bounds(
    events: List[Dict[str, Any] | Any],
    reqs: Dict[str, Any],
    *,
    window_size: float,
) -> tuple[float, float]:
    start_candidates: list[float] = []
    end_candidates: list[float] = []

    for event in events:
        event_type = _event_value(event, "event_type")
        timestamp = _event_value(event, "timestamp")
        if timestamp is None:
            continue
        ts = float(timestamp)
        if event_type == "energy":
            start_candidates.append(ts)
            end_candidates.append(ts + 1e-9)
        elif event_type == "batch":
            elapsed = float(_event_value(event, "elapsed", 0.0) or 0.0)
            start_candidates.append(ts - elapsed)
            end_candidates.append(ts + 1e-9)

    for req in reqs.values():
        arrival_time = getattr(req, "engine_arrival_time", -1.0)
        if arrival_time is None or arrival_time < 0:
            arrival_time = getattr(req, "arrival_time", -1.0)
        if arrival_time is not None and arrival_time >= 0:
            start_candidates.append(float(arrival_time))
        finish_times = [
            float(_event_value(event, "timestamp", arrival_time))
            for event in getattr(req, "events", [])
            if _event_value(event, "event_type") == "finish"
        ]
        if finish_times:
            end_candidates.append(max(finish_times) + 1e-9)

    if not start_candidates or not end_candidates:
        return 0.0, window_size
    return min(start_candidates), max(end_candidates)


def _extract_request_intervals(
    reqs: Dict[str, Any],
    *,
    end_time: float,
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for req in reqs.values():
        arrival_time = getattr(req, "engine_arrival_time", -1.0)
        if arrival_time is None or arrival_time < 0:
            arrival_time = getattr(req, "arrival_time", -1.0)
        if arrival_time is None or arrival_time < 0:
            continue

        finish_times = [
            float(_event_value(event, "timestamp", arrival_time))
            for event in getattr(req, "events", [])
            if _event_value(event, "event_type") == "finish"
        ]
        finish_time = max(finish_times) if finish_times else float(end_time)
        if finish_time < arrival_time:
            finish_time = float(arrival_time)
        intervals.append((float(arrival_time), float(finish_time)))
    return intervals


def _summarize_active_request_windows(
    reqs: Dict[str, Any],
    *,
    start_time: float,
    end_time: float,
    window_size: float,
) -> Dict[str, Any]:
    bins = _make_time_bins(start_time, end_time, window_size)
    n_windows = len(bins) - 1
    centers = bins[:-1] + 0.5 * window_size
    intervals = _extract_request_intervals(reqs, end_time=end_time)

    center_counts = np.zeros(n_windows, dtype=np.int64)
    any_active = np.zeros(n_windows, dtype=bool)
    if intervals:
        event_times, counts_after = build_active_requests_step(intervals)
        center_counts = count_at_times(event_times, counts_after, centers).astype(
            np.int64
        )

        diff = np.zeros(n_windows + 1, dtype=np.int64)
        for arrival_time, finish_time in intervals:
            if finish_time <= start_time or arrival_time >= end_time:
                continue
            left = max(0, int(np.floor((arrival_time - start_time) / window_size)))
            right = min(
                n_windows,
                int(np.ceil((finish_time - start_time) / window_size)),
            )
            if right <= left:
                if 0 <= left < n_windows:
                    right = left + 1
                else:
                    continue
            diff[left] += 1
            diff[right] -= 1
        any_active = np.cumsum(diff[:-1]) > 0

    distribution = [
        {
            "active_requests": int(active_requests),
            "window_count": int(count),
            "window_time_pct": float(100.0 * count / n_windows) if n_windows else 0.0,
        }
        for active_requests, count in sorted(Counter(center_counts.tolist()).items())
    ]
    return {
        "time": bins[:-1] - start_time,
        "center_counts": center_counts,
        "any_active": any_active,
        "distribution": distribution,
    }


def _summarize_boxplot(values: List[float], *, label: str) -> Dict[str, Any] | None:
    if not values:
        return None
    data = np.asarray(values, dtype=float)
    return {
        "label": label,
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "q1": float(np.percentile(data, 25)),
        "med": float(np.percentile(data, 50)),
        "q3": float(np.percentile(data, 75)),
        "whislo": float(np.percentile(data, 10)),
        "whishi": float(np.percentile(data, 90)),
        "fliers": [],
    }


def _match_batch_power_rows(
    events: List[Dict[str, Any] | Any],
    *,
    tolerance_s: float = BENCHMARK_BATCH_POWER_MATCH_TOLERANCE_S,
) -> List[Dict[str, Any]]:
    energy_times: Dict[int, List[float]] = defaultdict(list)
    energy_powers: Dict[int, List[float]] = defaultdict(list)
    for event in events:
        if _event_value(event, "event_type") != "energy":
            continue
        try:
            device_id = int(_event_value(event, "device_id", -1))
        except (TypeError, ValueError):
            continue
        if device_id < 0:
            continue
        energy_times[device_id].append(float(_event_value(event, "timestamp", 0.0)))
        energy_powers[device_id].append(float(_event_value(event, "power", 0.0) or 0.0))

    rows: List[Dict[str, Any]] = []
    for event in events:
        if _event_value(event, "event_type") != "batch":
            continue
        try:
            device_id = int(_event_value(event, "device_id", -1))
        except (TypeError, ValueError):
            continue
        if device_id < 0 or device_id not in energy_times:
            continue
        ts = float(_event_value(event, "timestamp", 0.0))
        idx = bisect.bisect_left(energy_times[device_id], ts)
        best_delta = None
        best_power = None
        for probe_idx in (idx - 1, idx):
            if not 0 <= probe_idx < len(energy_times[device_id]):
                continue
            delta = abs(energy_times[device_id][probe_idx] - ts)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_power = energy_powers[device_id][probe_idx]
        if best_delta is None or best_power is None or best_delta > tolerance_s:
            continue
        scheduled_tokens = _event_value(event, "num_scheduled_tokens", {}) or {}
        if not isinstance(scheduled_tokens, dict) or not scheduled_tokens:
            continue
        rows.append({
            "device_id": device_id,
            "tokens": int(sum(scheduled_tokens.values(), start=0)),
            "power": float(best_power),
        })
    return rows


def _build_token_bins(tokens: List[int]) -> Dict[str, Any]:
    if not tokens:
        return {"kind": "empty", "bins": []}
    unique_tokens = sorted({int(token) for token in tokens if int(token) > 0})
    if not unique_tokens:
        return {"kind": "empty", "bins": []}

    if len(unique_tokens) <= 16 and unique_tokens[-1] <= 64:
        return {
            "kind": "exact",
            "bins": [
                {
                    "label": str(token),
                    "low": int(token),
                    "high": int(token),
                }
                for token in unique_tokens
            ],
        }

    bins: list[dict[str, int | str]] = []
    power_of_two_edges = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    prev = 0
    min_token = unique_tokens[0]
    max_token = unique_tokens[-1]
    for edge in power_of_two_edges:
        low = prev + 1
        high = edge
        prev = edge
        if high < min_token:
            continue
        bins.append({
            "label": str(high) if low == high else f"{low}-{high}",
            "low": int(low),
            "high": int(high),
        })
        if high >= max_token:
            break
    if not bins:
        bins.append({
            "label": str(max_token),
            "low": int(max_token),
            "high": int(max_token),
        })
    elif int(bins[-1]["high"]) < max_token:
        low = int(bins[-1]["high"]) + 1
        bins.append({
            "label": f"{low}-{max_token}",
            "low": int(low),
            "high": int(max_token),
        })
    return {
        "kind": "powers_of_two",
        "bins": bins,
    }


def _summarize_power_vs_active_servers(
    active_server_counts: np.ndarray,
    total_power: np.ndarray,
) -> Dict[str, Any]:
    grouped_power: Dict[int, List[float]] = defaultdict(list)
    for active_servers, power in zip(active_server_counts.tolist(), total_power.tolist()):
        grouped_power[int(active_servers)].append(float(power))
    stats = [
        _summarize_boxplot(values, label=str(active_servers))
        for active_servers, values in sorted(grouped_power.items())
    ]
    return {
        "xlabel": "# Active Servers",
        "ylabel": "Total Power (W)",
        "stats": [item for item in stats if item is not None],
    }


def _summarize_power_vs_batch_tokens(
    batch_power_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    token_bins = _build_token_bins([int(row["tokens"]) for row in batch_power_rows])
    grouped_power: Dict[str, List[float]] = defaultdict(list)
    grouped_ranges: Dict[str, tuple[int, int]] = {}
    for row in batch_power_rows:
        token_count = int(row["tokens"])
        for bucket in token_bins["bins"]:
            low = int(bucket["low"])
            high = int(bucket["high"])
            if low <= token_count <= high:
                label = str(bucket["label"])
                grouped_power[label].append(float(row["power"]))
                grouped_ranges[label] = (low, high)
                break

    stats: list[dict[str, Any]] = []
    for bucket in token_bins["bins"]:
        label = str(bucket["label"])
        summary = _summarize_boxplot(grouped_power.get(label, []), label=label)
        if summary is None:
            continue
        low, high = grouped_ranges.get(label, (int(bucket["low"]), int(bucket["high"])))
        summary["low"] = int(low)
        summary["high"] = int(high)
        stats.append(summary)
    return {
        "xlabel": "# Tokens in Batch",
        "ylabel": "Server Power (W)",
        "binning": token_bins,
        "stats": stats,
    }


def _save_window_time_pct_figure(
    summary: Dict[str, Any],
    path: str | Path,
    *,
    title: str,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = summary.get("points", [])

    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
    if points:
        x = [int(point["active_requests"]) for point in points]
        y = [float(point["window_time_pct"]) for point in points]
        ax.bar(x, y, width=0.8, color="#1f78b4", alpha=0.9)
        ax.set_xticks(x)
    else:
        ax.text(0.5, 0.5, "No request activity", ha="center", va="center")
    ax.set_xlabel("Active Requests")
    ax.set_ylabel("Window Time (%)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def _draw_serialized_boxplot(
    ax: plt.Axes,
    stats: List[Dict[str, Any]],
    *,
    facecolor: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if not stats:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        return
    artists = ax.bxp(stats, showfliers=False, patch_artist=True)
    for box in artists["boxes"]:
        box.set_facecolor(facecolor)
        box.set_alpha(0.65)
    for median in artists["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)


def _save_power_distribution_figure(
    active_servers_summary: Dict[str, Any],
    batch_tokens_summary: Dict[str, Any],
    path: str | Path,
    *,
    title_prefix: str,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), tight_layout=True)
    _draw_serialized_boxplot(
        axes[0],
        active_servers_summary.get("stats", []),
        facecolor="#c73e1d",
        title=f"{title_prefix}: Power vs # Active Servers",
        xlabel=active_servers_summary.get("xlabel", "# Active Servers"),
        ylabel=active_servers_summary.get("ylabel", "Power (W)"),
    )
    _draw_serialized_boxplot(
        axes[1],
        batch_tokens_summary.get("stats", []),
        facecolor="#1f78b4",
        title=f"{title_prefix}: Power vs # Tokens in Batch",
        xlabel=batch_tokens_summary.get("xlabel", "# Tokens in Batch"),
        ylabel=batch_tokens_summary.get("ylabel", "Power (W)"),
    )
    if len(batch_tokens_summary.get("stats", [])) > 6:
        axes[1].tick_params(axis="x", rotation=45)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_benchmark_figures_from_result_row(
    result: Dict[str, Any],
    output_prefix: str | Path,
) -> Dict[str, str]:
    replay = result.get("benchmark_figure_replay")
    if not isinstance(replay, dict):
        return {}

    prefix = str(output_prefix)
    title_prefix = (
        f"{result.get('scheduling_policy', 'benchmark')} / "
        f"{result.get('routing_policy', 'policy')}"
    )
    window_png = _save_window_time_pct_figure(
        replay.get("window_time_pct_vs_active_requests", {}),
        f"{prefix}.window_time_pct_vs_active_requests.png",
        title=f"{title_prefix}: Window Time % vs Active Requests",
    )
    window_pdf = _save_window_time_pct_figure(
        replay.get("window_time_pct_vs_active_requests", {}),
        f"{prefix}.window_time_pct_vs_active_requests.pdf",
        title=f"{title_prefix}: Window Time % vs Active Requests",
    )
    power_png = _save_power_distribution_figure(
        replay.get("power_vs_active_servers", {}),
        replay.get("power_vs_batch_tokens", {}),
        f"{prefix}.power_vs_active_servers_and_batch_tokens.png",
        title_prefix=title_prefix,
    )
    power_pdf = _save_power_distribution_figure(
        replay.get("power_vs_active_servers", {}),
        replay.get("power_vs_batch_tokens", {}),
        f"{prefix}.power_vs_active_servers_and_batch_tokens.pdf",
        title_prefix=title_prefix,
    )
    return {
        "window_time_pct_vs_active_requests_figure": str(window_png),
        "window_time_pct_vs_active_requests_figure_pdf": str(window_pdf),
        "power_vs_active_servers_and_batch_tokens_figure": str(power_png),
        "power_vs_active_servers_and_batch_tokens_figure_pdf": str(power_pdf),
    }


def _summarize_benchmark_energy_and_figures(
    events: List[Dict[str, Any] | Any],
    reqs: Dict[str, Any],
    *,
    n_devices: int,
    tensor_parallel_size: int,
    output_prefix: str,
    scheduling_policy: str,
    routing_policy: str,
    window_size: float = BENCHMARK_METRIC_WINDOW_S,
    idle_power_per_gpu: float = BENCHMARK_IDLE_POWER_PER_GPU_W,
) -> Dict[str, Any]:
    start_time, end_time = _resolve_benchmark_window_bounds(
        events,
        reqs,
        window_size=window_size,
    )
    power_summary = _compute_measured_power_series(
        events,
        n_device=n_devices,
        window_size=window_size,
        start_time=start_time,
        end_time=end_time,
    )
    total_power = np.asarray(power_summary["total_power"], dtype=float)
    total_energy = total_power * window_size
    idle_power_per_replica = float(
        max(0.0, float(idle_power_per_gpu)) * max(1, int(tensor_parallel_size))
    )

    active_req_summary = _summarize_active_request_windows(
        reqs,
        start_time=start_time,
        end_time=end_time,
        window_size=window_size,
    )
    _, active_server_counts, _, _ = _compute_active_device_series(
        events,
        window_size=window_size,
        start_time=start_time,
        end_time=end_time,
        n_device=n_devices,
    )
    active_server_counts = np.asarray(active_server_counts, dtype=np.int64)

    n_windows = min(
        len(total_energy),
        len(active_req_summary["any_active"]),
        len(active_server_counts),
    )
    total_power = total_power[:n_windows]
    total_energy = total_energy[:n_windows]
    active_server_counts = active_server_counts[:n_windows]
    any_active = np.asarray(active_req_summary["any_active"][:n_windows], dtype=bool)

    energy_consumption_active = float(np.sum(total_energy[any_active]))
    idle_energy_per_window = float(
        idle_power_per_replica * max(0, int(n_devices)) * window_size
    )
    energy_consumption_non_idle = float(
        np.sum(np.maximum(total_energy - idle_energy_per_window, 0.0))
    )

    active_req_points = active_req_summary["distribution"]
    window_time_pct_summary = {
        "kind": "bar",
        "window_size_seconds": float(window_size),
        "xlabel": "Active Requests",
        "ylabel": "Window Time (%)",
        "points": active_req_points,
    }
    active_servers_summary = _summarize_power_vs_active_servers(
        active_server_counts,
        total_power,
    )
    batch_tokens_summary = _summarize_power_vs_batch_tokens(
        _match_batch_power_rows(events)
    )

    replay = {
        "window_size_seconds": float(window_size),
        "idle_power_per_gpu_w": float(idle_power_per_gpu),
        "idle_power_per_replica_w": float(idle_power_per_replica),
        "tensor_parallel_size": int(tensor_parallel_size),
        "power_source": str(power_summary.get("source", "measured")),
        "window_time_pct_vs_active_requests": window_time_pct_summary,
        "power_vs_active_servers": active_servers_summary,
        "power_vs_batch_tokens": batch_tokens_summary,
    }
    figure_paths = save_benchmark_figures_from_result_row(
        {
            "benchmark_figure_replay": replay,
            "scheduling_policy": scheduling_policy,
            "routing_policy": routing_policy,
        },
        output_prefix,
    )
    return {
        "energy_consumption_active": energy_consumption_active,
        "energy_consumption_non_idle": energy_consumption_non_idle,
        "benchmark_window_size_seconds": float(window_size),
        "benchmark_idle_power_per_gpu_w": float(idle_power_per_gpu),
        "benchmark_idle_power_per_replica_w": float(idle_power_per_replica),
        "benchmark_power_source": str(power_summary.get("source", "measured")),
        "benchmark_figure_replay": replay,
        **figure_paths,
    }


def _summarize_per_server_rps(
    events: List[Dict[str, Any] | Any],
    n_devices: int,
    duration_s: float,
) -> List[float]:
    per_server_dispatches = [0 for _ in range(max(0, int(n_devices)))]

    for event in events:
        event_type = _event_value(event, "event_type")
        if event_type == "dispatch-both":
            device_key = "prefill_device_id"
        elif event_type == "dispatch-prefill":
            device_key = "prefill_device_id"
        elif event_type == "dispatch-decode":
            device_key = "decode_device_id"
        else:
            continue

        try:
            device_id = int(_event_value(event, device_key, -1))
        except (TypeError, ValueError):
            continue
        if 0 <= device_id < len(per_server_dispatches):
            per_server_dispatches[device_id] += 1

    if duration_s <= 0.0:
        return [0.0 for _ in per_server_dispatches]
    return [float(count / duration_s) for count in per_server_dispatches]


def _get_rr_disagg_server_roles(
    n_devices: int,
    routing_policy: str,
    routing_kwargs: Dict[str, Any] | str | None,
) -> Dict[str, List[int]] | None:
    if routing_policy != "round_robin":
        return None

    routing_kwargs_dict = _routing_kwargs_to_dict(routing_kwargs)
    if not routing_kwargs_dict.get("is_pd_disagg", False):
        return None

    try:
        group_size = int(routing_kwargs_dict.get("group_size", n_devices))
        n_prefill_per_group = int(
            routing_kwargs_dict.get("n_prefill_per_group", group_size)
        )
    except (TypeError, ValueError):
        return None

    if (
        n_devices <= 0
        or group_size <= 0
        or group_size > n_devices
        or n_devices % group_size != 0
        or n_prefill_per_group <= 0
        or n_prefill_per_group >= group_size
    ):
        return None

    prefill_server_ids: List[int] = []
    decode_server_ids: List[int] = []
    for group_start in range(0, n_devices, group_size):
        prefill_server_ids.extend(
            range(group_start, group_start + n_prefill_per_group)
        )
        decode_server_ids.extend(
            range(group_start + n_prefill_per_group, group_start + group_size)
        )

    return {
        "prefill_server_ids": prefill_server_ids,
        "decode_server_ids": decode_server_ids,
    }


def _summarize_rr_disagg_power_metrics(
    *,
    n_devices: int,
    routing_policy: str,
    routing_kwargs: Dict[str, Any] | str | None,
    per_server_power: List[float],
) -> Dict[str, Any]:
    roles = _get_rr_disagg_server_roles(
        n_devices=n_devices,
        routing_policy=routing_policy,
        routing_kwargs=routing_kwargs,
    )
    if roles is None:
        return {}

    prefill_power = [
        float(per_server_power[device_id])
        for device_id in roles["prefill_server_ids"]
        if 0 <= device_id < len(per_server_power)
    ]
    decode_power = [
        float(per_server_power[device_id])
        for device_id in roles["decode_server_ids"]
        if 0 <= device_id < len(per_server_power)
    ]

    return {
        "per_prefill_power": (
            float(np.mean(prefill_power)) if prefill_power else 0.0
        ),
        "per_decode_power": (
            float(np.mean(decode_power)) if decode_power else 0.0
        ),
        "per_prefill_server_power": prefill_power,
        "per_decode_server_power": decode_power,
    }

def ensure_prompts_present(requests: List[Request], model_name: str) -> None:
    """Ensure prompts exist as token-id lists for all requests.

    Benchmarks in this repo operate on prompt lengths, not prompt semantics.
    To avoid backend-side tokenization variability, normalize every request to
    a synthetic token-id prompt unless it already has a token-id list.
    """
    if not requests:
        return
    has_token_prompts = [
        isinstance(req.prompt, list)
        and len(req.prompt) == req.input_length
        and all(isinstance(token, int) for token in req.prompt)
        for req in requests
    ]
    if all(has_token_prompts):
        return

    if any(req.prompt is not None for req in requests):
        logger.info(
            "Normalizing prompts to synthetic token-id lists for all requests."
        )
        for req in requests:
            req.prompt = None

    rng = np.random.default_rng()
    indices_by_length: Dict[int, List[int]] = {}
    for idx, req in enumerate(requests):
        indices_by_length.setdefault(req.input_length, []).append(idx)
    for length, indices in indices_by_length.items():
        num = len(indices)
        tokens_batch = rng.integers(1000, 2001, size=(num, length), dtype=np.int64)
        for i, req_idx in enumerate(indices):
            requests[req_idx].prompt = tokens_batch[i].tolist()


def _generate_random_prompt_tokens(length: int, rng: random.Random) -> list[int]:
    return [rng.randint(1000, 2000) for _ in range(max(0, int(length)))]


def generate_session_replay_prompts(
    requests: List[Request],
    seed: int = 0,
) -> None:
    """Build synthetic prompts that preserve cached prefixes within a session."""
    rng = random.Random(seed)
    session_prompts: Dict[str, list[int]] = {}
    warned_missing_session = False

    for request in requests:
        input_length = max(0, int(request.input_length))
        cached_length = min(max(0, int(request.cached_length)), input_length)
        session_id = request.session_id

        if not session_id:
            if cached_length > 0 and not warned_missing_session:
                logger.warning(
                    "enable_session_replay is on, but some requests have "
                    "cached_length > 0 and no session_id; generating "
                    "independent random prompts for those requests."
                )
                warned_missing_session = True
            request.prompt = _generate_random_prompt_tokens(input_length, rng)
            continue

        prefix_tokens: list[int] = []
        previous_prompt = session_prompts.get(session_id)
        if previous_prompt is not None and cached_length > 0:
            prefix_tokens = list(previous_prompt[:cached_length])
        if len(prefix_tokens) < cached_length:
            prefix_tokens.extend(
                _generate_random_prompt_tokens(cached_length - len(prefix_tokens), rng)
            )

        prompt_tokens = prefix_tokens + _generate_random_prompt_tokens(
            input_length - cached_length,
            rng,
        )
        request.prompt = prompt_tokens
        session_prompts[session_id] = prompt_tokens


def prepare_request_prompts(
    requests: List[Request],
    model_name: str,
    enable_session_replay: bool,
) -> None:
    if enable_session_replay:
        generate_session_replay_prompts(requests)
        return
    ensure_prompts_present(requests, model_name)


def build_session_replay_arrivals(
    requests: List[Request],
    arrival_times: List[float],
    enable_session_replay: bool,
) -> tuple[list[int], list[float], dict[str, deque[int]], dict[str, int]]:
    if len(requests) != len(arrival_times):
        raise ValueError(f"{len(requests)=} != {len(arrival_times)=}")
    if not enable_session_replay:
        return list(range(len(requests))), list(arrival_times), {}, {}

    trace_request_indices: list[int] = []
    trace_arrival_times: list[float] = []
    session_turn_queues: dict[str, deque[int]] = {}
    session_lengths: dict[str, int] = {}
    seen_sessions: set[str] = set()

    for idx, (request, arrival_time) in enumerate(zip(requests, arrival_times)):
        session_id = request.session_id
        if not session_id:
            trace_request_indices.append(idx)
            trace_arrival_times.append(arrival_time)
            continue

        session_lengths[session_id] = session_lengths.get(session_id, 0) + 1
        if session_id not in seen_sessions:
            seen_sessions.add(session_id)
            session_turn_queues[session_id] = deque()
            trace_request_indices.append(idx)
            trace_arrival_times.append(arrival_time)
            continue
        session_turn_queues[session_id].append(idx)

    return (
        trace_request_indices,
        trace_arrival_times,
        session_turn_queues,
        session_lengths,
    )


def release_ready_session_turns(
    pending_indices: list[int],
    requests: list[Request],
    elapsed_time: float,
    session_ready_at: dict[str, float],
    session_turn_queues: dict[str, deque[int]],
) -> list[int]:
    if not session_turn_queues:
        return []

    pending_sessions = {
        requests[idx].session_id
        for idx in pending_indices
        if requests[idx].session_id
    }
    released_indices: list[int] = []
    for session_id, queue in session_turn_queues.items():
        if not queue or session_id in pending_sessions:
            continue
        ready_at = session_ready_at.get(session_id)
        if ready_at is None or ready_at == float("inf") or elapsed_time < ready_at:
            continue
        request_idx = queue.popleft()
        pending_indices.append(request_idx)
        pending_sessions.add(session_id)
        released_indices.append(request_idx)
    return released_indices

async def _wait_for_server_ready(
    endpoint: str,
    timeout_s: float = 30.0,
    interval_s: float = 0.25,
) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.time() < deadline:
            try:
                response = await client.get(f"{endpoint}/health_check")
                if response.status_code == 200:
                    return
            except Exception as exc:
                last_error = exc
            await asyncio.sleep(interval_s)
    if last_error is not None:
        raise RuntimeError(
            f"Server at {endpoint} did not become ready within {timeout_s}s"
        ) from last_error
    raise RuntimeError(
        f"Server at {endpoint} did not become ready within {timeout_s}s"
    )


async def _get_server_health(
    endpoint: str,
    timeout_s: float = 5.0,
) -> tuple[bool, str]:
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(f"{endpoint}/health_check")
        if response.status_code == 200:
            return True, "ok"
        detail = response.text.strip() or f"status={response.status_code}"
        return False, detail
    except Exception as exc:
        return False, repr(exc)

async def _configure_router_endpoint(
    problem: Problem,
    endpoint: str,
    clients: str | None,
    *,
    update_clients: bool,
) -> None:
    print(f"Posting problem to endpoint: {endpoint}")
    print(f"Problem: {problem}")
    timeout = aiohttp.ClientTimeout(total=3000.0)

    async with aiohttp.ClientSession() as session:
        if clients is not None and update_clients:
            response = await session.post(
                endpoint + "/update_clients",
                json={"clients": clients},
                timeout=timeout,
            )
            response.raise_for_status()
        response = await session.post(
            endpoint + "/update_config",
            json=asdict(problem),
            timeout=timeout,
        )
        response.raise_for_status()


async def main(
    problem: Problem,
    endpoint: str,
    clients: str | None,
    *,
    update_clients: bool = True,
):
    trace_spec = problem.trace_spec or problem.length_pattern
    requests, arrival_times, trace_components = _load_trace_inputs(
        trace_spec,
        model_name=problem.model_name,
        load_scale=problem.load_scale,
        window=problem.window,
    )
    if trace_components:
        print(f"trace_components: {trace_components}")
    print('arrival_times:', arrival_times[0], '->', arrival_times[-1])
    arrival_base_time = arrival_times[0]
    requested_n_devices = problem.n_devices
    original_request_count = len(requests)
    effective_n_devices, rr_sliced = _resolve_rr_effective_n_devices(
        requested_n_devices,
        problem.routing_policy,
        problem.routing_kwargs,
        clients,
    )
    if rr_sliced:
        requests, arrival_times, _ = _slice_rr_workload(
            requests,
            arrival_times,
            requested_n_devices,
            effective_n_devices,
        )
        routing_kwargs = _routing_kwargs_to_dict(problem.routing_kwargs)
        if "group_size" in routing_kwargs:
            routing_kwargs["group_size"] = min(
                int(routing_kwargs["group_size"]),
                effective_n_devices,
            )
        problem.routing_kwargs = routing_kwargs
        problem.n_devices = effective_n_devices
        problem.store_prefix = (
            f"{problem.store_prefix}_rrslice_eff{effective_n_devices}"
        )
        print(
            "RR slicing enabled:",
            f"requested_n_devices={requested_n_devices}",
            f"effective_n_devices={effective_n_devices}",
            f"kept_requests={len(requests)}",
            f"original_requests={original_request_count}",
        )
    (
        trace_request_indices,
        trace_arrival_times,
        session_turn_queues,
        session_lengths,
    ) = build_session_replay_arrivals(
        requests,
        arrival_times,
        problem.enable_session_replay,
    )
    arrival_base_time = trace_arrival_times[0]
    arrival_times = [t - arrival_base_time for t in trace_arrival_times]
    rps = len(trace_request_indices) / (arrival_times[-1] - arrival_times[0])
    
    from SLOsServe.perf_model import PerfModel
    perf_model = PerfModel.get_perf_model(problem.model_name, problem.perf_model_task)
    
    prepare_request_prompts(
        requests,
        problem.model_name,
        problem.enable_session_replay,
    )
    
    # requests = requests.requests[window_start:window_end]
    # arrival_times = arrival_times.arrival_times[window_start:window_end]
    
    import numpy as np
    global average_input_length, average_output_length
    average_input_length = np.mean([request.input_length for request in requests])
    average_output_length = np.mean([request.output_length for request in requests])
    print(f'#Requests: {len(requests)}')
    print(f'average_input_length: {average_input_length}')
    print(f'average_output_length: {average_output_length}')
    if problem.enable_session_replay:
        session_length_values = list(session_lengths.values())
        if session_length_values:
            session_request_count = sum(session_length_values)
            print(
                "Session replay:",
                f"sessions={len(session_length_values)}",
                f"session_requests={session_request_count}",
                f"non_session_requests={len(requests) - session_request_count}",
                f"session_head_arrivals={len(session_length_values)}",
                f"mean_len={float(np.mean(session_length_values)):.2f}",
                f"median_len={float(np.median(session_length_values)):.2f}",
                f"p90_len={float(np.percentile(session_length_values, 90)):.2f}",
                f"max_len={max(session_length_values)}",
            )
        else:
            print("Session replay: no session_ids found in requests")
    
    
    arrival_idx = 0
    window_size = 0.001
    
    execution_results: List[ExecutionResult] = []
    
    await _configure_router_endpoint(
        problem,
        endpoint,
        clients,
        update_clients=update_clients,
    )
    await _wait_for_server_ready(endpoint)
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                endpoint + "/start_energy_profile",
                json={"store_prefix": problem.store_prefix},
                timeout=aiohttp.ClientTimeout(total=3000.0),
            ) as response:
                if response.status != 404:
                    response.raise_for_status()
        except aiohttp.ClientResponseError as exc:
            if exc.status != 404:
                raise

    arrival_bar_desc = 'Session Arrival' if problem.enable_session_replay else 'Arrival'
    arrival_bar = tqdm.tqdm(total = len(trace_request_indices), desc = arrival_bar_desc)
    finished_bar = tqdm.tqdm(total = len(requests), desc = 'Finished Requests')
    
    global_start_time = time.time()
    print(f'global_start_time: {global_start_time}')
    
    tasks = []
    pending_indices: list[int] = []
    time_offset = 0
    time_offsets = [(global_start_time, 0)]
    timeout = REQUEST_TIMEOUT_S
    cancel_grace = REQUEST_CANCEL_GRACE_S
    last_health_check_time = global_start_time - FRONTEND_HEALTH_CHECK_INTERVAL_S
    dump_profile_response: Dict[str, Any] = {}
    
    real_arrival_times = {}
    
    bid_to_id = {}
    timed_out_requests: set[str] = set()
    session_ready_at: dict[str, float] = {}
    n_rejected = 0
    n_timed_out = 0
    rejection_reason_counts: Counter[str] = Counter()
    rejected_request_reasons: dict[str, str] = {}
    
    try:
        async with httpx.AsyncClient(timeout=3600, base_url=endpoint) as client:
            while finished_bar.n < len(requests):
                elapsed_time = time.time() - global_start_time + time_offset
                
                while arrival_idx < len(trace_request_indices) and arrival_times[arrival_idx] <= elapsed_time:
                    pending_indices.append(trace_request_indices[arrival_idx])
                    arrival_bar.update(1)
                    arrival_bar.set_description(f'Arrival Time: {arrival_times[arrival_idx]:.2f}, Elapsed Time: {elapsed_time:.2f}')
                    arrival_idx += 1

                if problem.enable_session_replay:
                    release_ready_session_turns(
                        pending_indices,
                        requests,
                        elapsed_time,
                        session_ready_at,
                        session_turn_queues,
                    )

                ready_indices, pending_indices = _split_ready_request_indices(
                    pending_indices,
                    requests,
                    elapsed_time,
                    session_ready_at,
                    problem.enable_session_replay,
                )
                for request_idx in ready_indices:
                    request = requests[request_idx]
                    assert request.prompt is not None
                    prompt = request.prompt
                    assert prompt is not None
                    task_start_time = time.time()
                    request_id_backend = str(uuid.uuid1()) # str(request_idx)
                    request_id = str(request_idx)
                    bid_to_id[request_id_backend] = request_id
                    if problem.enable_session_replay and request.session_id:
                        session_ready_at[request.session_id] = float("inf")

                    task = asyncio.create_task(run_request(
                        client,
                        request_id_backend,
                        problem.model_name,
                        prompt,
                        request.input_length,
                        request.output_length,
                        zero_load_ttft=perf_model.get_zero_load_ttft(
                            request.input_length,
                            request.cached_length,
                        ),
                        cached_tokens=request.cached_length,
                        session_id=request.session_id,
                        ttft_slo=compute_ttft_slo(
                            request.input_length,
                            request.cached_length,
                            slo_ttft_per_token=problem.slo_ttft_per_token,
                            slo_ttft_constant=problem.slo_ttft_constant,
                            slo_routing_overhead=problem.slo_routing_overhead,
                        ),
                        slo_tpot=problem.slo_tpot,
                        expected_profit=problem.get_expected_profit(
                            request.input_length - request.cached_length,
                        ),
                        real_arrival_times=real_arrival_times,
                    ))

                    tasks.append((task, request, task_start_time, request_id))
                    
                real_time = time.time()
                elapsed_time = real_time - global_start_time + time_offset
                if finished_bar.n == arrival_bar.n and arrival_idx < len(trace_request_indices) and (arrival_times[arrival_idx] - elapsed_time > 10):
                    time_offset += arrival_times[arrival_idx] - elapsed_time -1
                    time_offsets.append((real_time, time_offset))
                    continue
                # Only check finished requests without waiting, and keep unfinished tasks in the list
                new_tasks = []
                current_time = time.time()

                for task, request, task_start_time, request_id in tasks:
                    if task.done():
                        completion_elapsed_time = current_time - global_start_time + time_offset
                        timed_out_before = request_id in timed_out_requests
                        finished_bar.update(1)
                        finished_bar.set_description(f'Finished: {finished_bar.n}, Rejected: {n_rejected}, Timed Out: {n_timed_out}')

                        # ---- explicit status checks ----
                        if task.cancelled():
                            logger.info(f"Request {request_id} cancelled before completion")
                            execution_results.append(ExecutionResult(request, [task_start_time], request_id))
                            if problem.enable_session_replay and request.session_id:
                                session_ready_at[request.session_id] = completion_elapsed_time + problem.session_pause_s
                            timed_out_requests.discard(request_id)
                            continue

                        exc = task.exception()  # safe here because task.done() is True and not cancelled
                        if exc is not None:
                            # ===== task FAILED =====
                            logger.error(f"Task for request {request_id} failed: {exc!r}")
                            import traceback
                            logger.error("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
                            if problem.enable_session_replay and request.session_id:
                                session_ready_at[request.session_id] = completion_elapsed_time + problem.session_pause_s
                            timed_out_requests.discard(request_id)
                            raise BenchmarkOverloadedError(
                                "Request task failed after the server became unhealthy. "
                                f"request_id={request_id} error={exc!r}"
                            ) from exc

                        # ===== task SUCCEEDED =====
                        try:
                            is_rejected, rejection_reason, response_text, timestamps = task.result()
                            normalized_rejection_reason = (
                                rejection_reason if is_rejected else None
                            )
                            if is_rejected:
                                n_rejected += 1
                                reason_key = normalized_rejection_reason or "unknown"
                                rejection_reason_counts[reason_key] += 1
                                rejected_request_reasons[request_id] = reason_key
                            if not len(timestamps):
                                timestamps = [task_start_time]
                            execution_results.append(
                                ExecutionResult(
                                    request,
                                    timestamps,
                                    request_id,
                                    rejection_reason=normalized_rejection_reason,
                                )
                            )
                            if timed_out_before:
                                logger.info(f"Request {request_id} returned after timeout")
                            if problem.enable_session_replay and request.session_id:
                                session_ready_at[request.session_id] = completion_elapsed_time + problem.session_pause_s
                        finally:
                            timed_out_requests.discard(request_id)

                    elif request_id in timed_out_requests:
                        if current_time - task_start_time > timeout + cancel_grace:
                            completion_elapsed_time = current_time - global_start_time + time_offset
                            logger.error(
                                "Request %s remained pending %.1fs after timeout; treating it as failed and moving on",
                                request_id,
                                current_time - task_start_time,
                            )
                            finished_bar.update(1)
                            finished_bar.set_description(f'Finished: {finished_bar.n}, Rejected: {n_rejected}, Timed Out: {n_timed_out}')
                            execution_results.append(ExecutionResult(request, [task_start_time], request_id))
                            if problem.enable_session_replay and request.session_id:
                                session_ready_at[request.session_id] = completion_elapsed_time + problem.session_pause_s
                            timed_out_requests.discard(request_id)
                            continue
                        new_tasks.append((task, request, task_start_time, request_id))

                    elif current_time - task_start_time > timeout:
                        n_timed_out += 1
                        task.cancel()
                        timed_out_requests.add(request_id)
                        if problem.enable_session_replay and request.session_id:
                            session_ready_at[request.session_id] = elapsed_time + problem.session_pause_s
                        new_tasks.append((task, request, task_start_time, request_id))

                    else:
                        new_tasks.append((task, request, task_start_time, request_id))
                tasks = new_tasks

                if tasks and current_time - last_health_check_time >= FRONTEND_HEALTH_CHECK_INTERVAL_S:
                    last_health_check_time = current_time
                    healthy, health_detail = await _get_server_health(endpoint)
                    if not healthy:
                        for task, *_ in tasks:
                            task.cancel()
                        raise BenchmarkOverloadedError(
                            "Server became unhealthy during benchmark; aborting this run. "
                            f"endpoint={endpoint} detail={health_detail}"
                        )

                await asyncio.sleep(window_size)
    finally:
        for task, *_ in tasks:
            if not task.done():
                task.cancel()
        arrival_bar.close()
        finished_bar.close()
    
    def apply_time_offsets(t: float):
        idx = len(time_offsets) - 1
        while idx >= 0 and time_offsets[idx][0] > t:
            idx -= 1
        if idx < 0:
            idx = 0
        return time_offsets[idx][1] + t - global_start_time

    for result in execution_results:
        result.timestamps = [apply_time_offsets(t) for t in result.timestamps]
        
    i = 0
    filename = f'{problem.store_prefix}.{i}.events.jsonl'
    while os.path.exists(filename):
        i += 1
        filename = f'{problem.store_prefix}.{i}.events.jsonl'
    
    admission_filename = f'{problem.store_prefix}.{i}.admission_history.jsonl'

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{endpoint}/dump_profile_events",
            json={
                "filename": filename,
                "admission_filename": admission_filename,
                "timeout": 3000.0,
                "include_energy_csv": False,
            },
            timeout=3000.0
        )
        response.raise_for_status()
        dump_profile_response = response.json()
        
    
    
    with open(filename, 'r') as f:
        events = json.load(f)
        events = sorted(events, key=lambda x: x['timestamp'])
        # print(bid_to_id.keys())
        backend_id_2_id = lambda id: bid_to_id.get(('-'.join(id.split('-')[1:-1]) if id.startswith('cmpl-') else id), '-1')
        for event in events:
            if 'request_id' in event:
                event['request_id'] = backend_id_2_id(event['request_id'])  
            if event['event_type'] == 'batch':
                event['req_ids'] = [backend_id_2_id(req_id) for req_id in event['req_ids']]
                event['num_scheduled_tokens'] = {backend_id_2_id(req_id): num_scheduled_tokens for req_id, num_scheduled_tokens in event['num_scheduled_tokens'].items()}
                if 'rejected_reqs' in event:
                    event['rejected_reqs'] = [backend_id_2_id(req_id) for req_id in event['rejected_reqs']]
            if event['event_type'] == 'req_state':
                event['ddl'] = apply_time_offsets(event['ddl'])
            if event['event_type'] == 'arrival':
                event['add_req_time'] = apply_time_offsets(event['add_req_time'])
            if event['event_type'] == 'schedule_problem':
                for req in event['reqs']:
                    req['id'] = backend_id_2_id(req['id'])
                event['accepted_ids'] = [backend_id_2_id(req_id) for req_id in event['accepted_ids']]
                for batch in event['batch_schedule']:
                    batch['id'] = backend_id_2_id(batch['id'])
        control_estimated_batch_times = {}
        for event in events:
            if event['event_type'] == 'schedule_problem':
                control_estimated_batch_times[
                    (event.get('device_id', 0), event['batch_id'])
                ] = event.get('estimated_time')
        for event in events:
            if event['event_type'] == 'batch':
                batch_key = (event.get('device_id', 0), event['batch_id'])
                control_estimated_time = control_estimated_batch_times.get(
                    batch_key
                )
                if control_estimated_time is not None:
                    event['control_estimated_time'] = control_estimated_time
                if (
                    'estimated_time' not in event
                    or event.get('estimated_time') is None
                ):
                    event['estimated_time'] = (
                        control_estimated_time
                        if control_estimated_time is not None else 0
                    )
        with open(filename, 'w') as f:
            json.dump(events, f, indent=4)
    
    with open(filename, 'r') as f:
        events = json.load(f)
        for event in events:
            event['timestamp'] = apply_time_offsets(event['timestamp'])
        for req_id, arrival_time in real_arrival_times.items():
            events.append({
                'event_type': 'global_arrival',
                'request_id': backend_id_2_id(req_id),
                'timestamp': apply_time_offsets(arrival_time),
            })
        for event in events:
            if 'prefill_ddl' in event:
                event['prefill_ddl'] = apply_time_offsets(event['prefill_ddl'])
            if 'kv_ready_time' in event:
                event['kv_ready_time'] = apply_time_offsets(event['kv_ready_time'])
        events = sorted(events, key=lambda x: x['timestamp'])
    with open(filename, 'w') as f:
        json.dump(events, f, indent = 4)
    print(f'Saved {filename}')

    perf_model_error_filename = f'{problem.store_prefix}.{i}.perf_model_errors.jsonl'
    perf_model_error_figure_filename = (
        f'{problem.store_prefix}.{i}.perf_model_estimated_vs_measured.png'
    )
    perf_model_full_elapsed_figure_filename = (
        f'{problem.store_prefix}.{i}.perf_model_estimated_with_overhead_vs_elapsed.png'
    )
    perf_model_regression_figure_filename = (
        f'{problem.store_prefix}.{i}.perf_model_regression.png'
    )
    perf_model_error_artifacts = None
    if problem.log_perf_model_errors:
        perf_model_error_artifacts = _log_perf_model_errors_from_batch_events(
            problem,
            events,
            perf_model_error_filename,
            event_file=filename,
            include_time_lists=problem.include_perf_model_time_lists,
            draw_figure=problem.draw_perf_model_error_figure,
            figure_path=perf_model_error_figure_filename,
            regression_figure_path=perf_model_regression_figure_filename,
            full_elapsed_figure_path=perf_model_full_elapsed_figure_filename,
        )
        if perf_model_error_artifacts is not None:
            print(
                "Saved perf-model error log to "
                f"{perf_model_error_artifacts['path']}"
            )
            error_summary = perf_model_error_artifacts["summary"]
            empirical_scheduling_overhead = error_summary.get(
                "empirical_scheduling_overhead", {}
            ).get("overhead_s", {})
            print(
                "Perf-model error summary:",
                f"old_params={error_summary['old_hardware_params']}",
                f"regressed_params={error_summary['regressed_hardware_params']}",
                f"signed_mean={error_summary['estimated_minus_measured_s'].get('mean')}",
                f"abs_mean={error_summary['abs_estimated_minus_measured_s'].get('mean')}",
                f"relative_p50={error_summary['estimated_minus_measured_relative'].get('p50')}",
                f"scheduling_overhead_mean={empirical_scheduling_overhead.get('mean')}",
                f"scheduling_overhead_p95={empirical_scheduling_overhead.get('p95')}",
            )
            if perf_model_error_artifacts.get("figure_path"):
                print(
                    "Saved perf-model estimated-vs-pure-execution figure to "
                    f"{perf_model_error_artifacts['figure_path']}"
                )
            if perf_model_error_artifacts.get("full_elapsed_figure_path"):
                print(
                    "Saved perf-model estimated-with-overhead-vs-elapsed figure to "
                    f"{perf_model_error_artifacts['full_elapsed_figure_path']}"
                )
            if perf_model_error_artifacts.get("regression_figure_path"):
                print(
                    "Saved perf-model regression figure to "
                    f"{perf_model_error_artifacts['regression_figure_path']}"
                )

    # TODO(Yi): add oom fail rate analysis here.
    empirical_scheduling_overhead_summary = _summarize_batch_scheduling_overhead(
        events
    )
    events, reqs = analyze_events(filename, verbose = True)
    results = analyze_slo_violation(reqs, events, 
                                    model_name = problem.model_name, 
                                    length_pattern = problem.perf_model_task,
                                    ttft_slo_scale = problem.ttft_slo_scale, 
                                    slo_tpot = problem.slo_tpot, 
                                    slo_ttft_overhead = problem.slo_routing_overhead,
                                    slo_ttft_per_token = problem.slo_ttft_per_token,
                                    slo_ttft_constant = problem.slo_ttft_constant,
                                    prefix = problem.store_prefix, 
                                    routing_overhead = problem.routing_overhead,
                                    n_device = problem.n_devices,
                                    group_size = problem.routing_kwargs.get('group_size'),
                                    draw = True)
    ttft_laxity_percentiles = results.get('ttft_laxity_percentiles', {})
    ttft_parts = [
        f"{key}={ttft_laxity_percentiles[key]:.6f}"
        for key in ('p20', 'p50', 'p80', 'p90', 'p95', 'p99', 'max')
        if ttft_laxity_percentiles.get(key) is not None
    ]
    if ttft_parts:
        print('Benchmark TTFT laxity percentiles:', ', '.join(ttft_parts))
    tpot_slo_violation_rate_sweep = results.get(
        'tpot_slo_violation_rate_sweep',
        {},
    )
    tpot_parts = [
        f"{label}({summary['slo_tpot']:.3f})={summary['violation_rate']:.6f}"
        for label, summary in tpot_slo_violation_rate_sweep.items()
    ]
    if tpot_parts:
        print('Benchmark TPOT SLO violation sweep:', ', '.join(tpot_parts))
    if rejection_reason_counts:
        rejection_parts = [
            f"{reason}={count}"
            for reason, count in sorted(rejection_reason_counts.items())
        ]
        print('Benchmark rejection reasons:', ', '.join(rejection_parts))
    else:
        print('Benchmark rejection reasons: none')
    results['rps'] = rps
    results['requested_n_device'] = requested_n_devices
    results['effective_n_device'] = problem.n_devices
    results['rr_sliced'] = rr_sliced
    results['rr_slice_kept_request_count'] = len(requests)
    results['rr_slice_total_request_count'] = original_request_count
    results['run_status'] = 'completed'
    results['overloaded'] = False
    results['rejection_reason_counts'] = dict(
        sorted(rejection_reason_counts.items())
    )
    results['rejected_request_reasons'] = {
        request_id: rejected_request_reasons[request_id]
        for request_id in sorted(
            rejected_request_reasons,
            key=_request_id_sort_key,
        )
    }
    results['configured_scheduling_overhead_s'] = float(
        problem.scheduling_overhead
    )
    if empirical_scheduling_overhead_summary is not None:
        results['empirical_scheduling_overhead_summary'] = (
            empirical_scheduling_overhead_summary
        )
    extra_metrics = results.setdefault('extra_metrics', {})
    extra_metrics.update(_summarize_per_server_energy_metrics(
        events,
        problem.n_devices,
    ))
    benchmark_energy_figure_metrics = _summarize_benchmark_energy_and_figures(
        events,
        reqs,
        n_devices=problem.n_devices,
        tensor_parallel_size=problem.tensor_parallel_size,
        output_prefix=f'{problem.store_prefix}.{i}',
        scheduling_policy=problem.scheduling_policy,
        routing_policy=problem.routing_policy,
    )
    extra_metrics.update(benchmark_energy_figure_metrics)
    arrival_duration_s = (
        float(arrival_times[-1] - arrival_times[0])
        if len(arrival_times) > 1 else 0.0
    )
    extra_metrics['per_server_rps'] = _summarize_per_server_rps(
        events,
        problem.n_devices,
        arrival_duration_s,
    )
    extra_metrics.update(_summarize_rr_disagg_power_metrics(
        n_devices=problem.n_devices,
        routing_policy=problem.routing_policy,
        routing_kwargs=problem.routing_kwargs,
        per_server_power=extra_metrics.get('per_server_power', []),
    ))
    if perf_model_error_artifacts is not None:
        results['perf_model_error_summary'] = perf_model_error_artifacts["summary"]
        results['perf_model_error_file'] = perf_model_error_artifacts["path"]
        if perf_model_error_artifacts.get("figure_path") is not None:
            results['perf_model_error_figure'] = perf_model_error_artifacts[
                "figure_path"]
        if perf_model_error_artifacts.get("full_elapsed_figure_path") is not None:
            results['perf_model_full_elapsed_figure'] = (
                perf_model_error_artifacts["full_elapsed_figure_path"])
        if perf_model_error_artifacts.get("regression_figure_path") is not None:
            results['perf_model_regression_figure'] = (
                perf_model_error_artifacts["regression_figure_path"])
        results['perf_model_old_hardware_params'] = perf_model_error_artifacts[
            "summary"]["old_hardware_params"]
        results['perf_model_regressed_hardware_params'] = (
            perf_model_error_artifacts["summary"]["regressed_hardware_params"])
        results['perf_model_regressed_hardware_params_delta'] = (
            perf_model_error_artifacts["summary"][
                "regressed_hardware_params_delta"])
        results['perf_model_regression_stats'] = (
            perf_model_error_artifacts["summary"]["regression_stats"])
        if "estimated_time_list" in perf_model_error_artifacts["summary"]:
            results["perf_model_estimated_time_list"] = (
                perf_model_error_artifacts["summary"]["estimated_time_list"])
        if "measured_time_list" in perf_model_error_artifacts["summary"]:
            results["perf_model_measured_time_list"] = (
                perf_model_error_artifacts["summary"]["measured_time_list"])
    if os.path.exists(admission_filename):
        with open(admission_filename, 'r') as f:
            admission_history = json.load(f)
        for event in admission_history:
            event['request_id'] = backend_id_2_id(event['request_id'])
            event['slo_violation'] = reqs[event['request_id']].is_violate_slo()
        with open(admission_filename, 'w') as f:
            json.dump(admission_history, f, indent=4)
        if 'auto_scaling' in problem.routing_policy:
            threshold = problem.routing_kwargs.get('threshold', 0.5)
            model_key = problem.routing_kwargs.get('model_key', 'all')
            auto_scaling_analysis = eval_auto_scaling(model_key, admission_filename,
                                                    threshold = threshold)
        else: 
            auto_scaling_analysis = {}
        results['auto_scaling_analysis'] = auto_scaling_analysis
        
    execution_results = sorted(
        execution_results,
        key=lambda x: _request_id_sort_key(x.request_id),
    )
    raw_per_gpu_energy = dump_profile_response.get("per_gpu_energy_consumption", [])
    per_gpu_energy_consumption = [
        float(value or 0.0)
        for value in list(raw_per_gpu_energy)[:max(0, int(problem.n_devices))]
    ]
    if len(per_gpu_energy_consumption) < max(0, int(problem.n_devices)):
        per_gpu_energy_consumption.extend(
            [0.0] * (int(problem.n_devices) - len(per_gpu_energy_consumption))
        )
    energy_consumption = float(sum(per_gpu_energy_consumption))
    if not per_gpu_energy_consumption or all(
        energy <= 0.0 for energy in per_gpu_energy_consumption
    ):
        per_gpu_energy_consumption, energy_consumption = summarize_energy_events(
            events,
            n_devices=problem.n_devices,
        )
    results, energy_consumption = _scale_rr_energy_results(
        results=results,
        energy_consumption=energy_consumption,
        requested_n_devices=requested_n_devices,
        effective_n_devices=problem.n_devices,
        rr_sliced=rr_sliced,
    )
    results =  ExecutionResults(
        problem,
        execution_results,
        results,
        f'{problem.store_prefix}.{i}',
        energy_consumption=energy_consumption,
        per_gpu_energy_consumption=per_gpu_energy_consumption,
    )
    reqs = sorted(list(reqs.values()), key=lambda x: _request_id_sort_key(x.req_id))
    with open(f'{problem.store_prefix}.reqs.jsonl', 'w') as f:
        json.dump([asdict(req) for req in reqs], f, indent=4)
    print(f'Saved {problem.store_prefix}.{i}.reqs.jsonl')
    return results


def _normalize_router_clients_arg(clients_arg: str | None) -> str | None:
    return normalize_client_spec(clients_arg)


def _count_router_clients_arg(clients_arg: str | None) -> int | None:
    return count_client_spec(clients_arg)


def _routing_kwargs_to_dict(
    routing_kwargs: dict[str, Any] | str | None,
) -> dict[str, Any]:
    if routing_kwargs is None:
        return {}
    if isinstance(routing_kwargs, str):
        return json.loads(routing_kwargs)
    return dict(routing_kwargs)


def _supports_rr_workload_slicing(
    routing_policy: str,
    routing_kwargs: dict[str, Any] | str | None,
) -> bool:
    if routing_policy not in {"round_robin", "round_robin_retry", "round_robin_session"}:
        return False
    return not _routing_kwargs_to_dict(routing_kwargs).get("is_pd_disagg", False)


def _resolve_rr_effective_n_devices(
    requested_n_devices: int,
    routing_policy: str,
    routing_kwargs: dict[str, Any] | str | None,
    clients_arg: str | None,
) -> tuple[int, bool]:
    if not _supports_rr_workload_slicing(routing_policy, routing_kwargs):
        return requested_n_devices, False
    available_clients = _count_router_clients_arg(clients_arg)
    if available_clients is None or available_clients <= 0:
        return requested_n_devices, False
    if available_clients >= requested_n_devices:
        return requested_n_devices, False
    return available_clients, True


def _slice_rr_workload(
    requests: list[Any],
    arrival_times: list[float],
    requested_n_devices: int,
    effective_n_devices: int,
) -> tuple[list[Any], list[float], list[int]]:
    assert len(requests) == len(arrival_times), (
        f"{len(requests)=} != {len(arrival_times)=}")
    if effective_n_devices >= requested_n_devices:
        kept_indices = list(range(len(requests)))
        return list(requests), list(arrival_times), kept_indices

    kept_indices = [
        idx for idx in range(len(requests))
        if idx % requested_n_devices < effective_n_devices
    ]
    return (
        [requests[idx] for idx in kept_indices],
        [arrival_times[idx] for idx in kept_indices],
        kept_indices,
    )


def _rr_slice_energy_multiplier(
    requested_n_devices: int,
    effective_n_devices: int,
    rr_sliced: bool,
) -> float:
    if not rr_sliced:
        return 1.0
    if effective_n_devices <= 0 or effective_n_devices >= requested_n_devices:
        return 1.0
    return requested_n_devices / effective_n_devices


def _scale_rr_energy_results(
    results: dict[str, Any],
    energy_consumption: float,
    requested_n_devices: int,
    effective_n_devices: int,
    rr_sliced: bool,
) -> tuple[dict[str, Any], float]:
    multiplier = _rr_slice_energy_multiplier(
        requested_n_devices=requested_n_devices,
        effective_n_devices=effective_n_devices,
        rr_sliced=rr_sliced,
    )
    if multiplier == 1.0:
        return results, float(energy_consumption)

    scaled_results = dict(results)
    extra_metrics = scaled_results.get("extra_metrics")
    if isinstance(extra_metrics, dict):
        scaled_extra_metrics = dict(extra_metrics)
        for key in (
            "energy_est",
            "energy_consumption_active",
            "energy_consumption_non_idle",
        ):
            if key in scaled_extra_metrics:
                scaled_extra_metrics[key] = float(scaled_extra_metrics[key]) * multiplier
        scaled_results["extra_metrics"] = scaled_extra_metrics
    return scaled_results, float(energy_consumption) * multiplier


def _session_replay_suffix(enable_session_replay: bool, session_pause_s: float) -> str:
    if not enable_session_replay:
        return ""
    pause = str(session_pause_s).replace("-", "m").replace(".", "p")
    return f"_sessionreplay_{pause}s"


def _baseline_cap_suffix(baseline_decode_cap: int | None) -> str:
    if baseline_decode_cap is None:
        return ""
    return f"_bcap{int(baseline_decode_cap)}"


def build_problems(
    model_name: str,
    trace: str,
    ttft_slo_scale: float,
    slo_tpot: float,
    profit: str,
    scheduling_policy: str,
    routing_policy: str,
    n_device: int,  
    tensor_parallel_size: int,
    window: str,
    load_scale: float,
    experiment_dir: str,
    slo_routing_overhead: float = 0.08,
    admission_mode: str = 'arrival',
    perf_model_err: float = 1.0,
    routing_overhead: float = -1.0,
    scheduling_overhead: float = 0.0,
    routing_fallback_policy: str = "asap",
    kv_xfer_delay: float = 0.05,
    enable_session_replay: bool = False,
    session_pause_s: float = 0.0,
    log_perf_model_errors: bool = True,
    include_perf_model_time_lists: bool = False,
    draw_perf_model_error_figure: bool = True,
    enable_piecewise_perf_model_regression: bool = False,
    perf_model_piecewise_breakpoints: list[int] | None = None,
    baseline_decode_cap: int | None = None,
):
    session_replay_suffix = _session_replay_suffix(
        enable_session_replay,
        session_pause_s,
    )
    baseline_cap_suffix = _baseline_cap_suffix(baseline_decode_cap)
    perf_model_regression_suffix = _perf_model_regression_suffix(
        enable_piecewise_perf_model_regression,
        perf_model_piecewise_breakpoints,
    )
    store_prefix = (
        f'{experiment_dir}/{scheduling_policy}_{routing_policy}_{load_scale}_'
        f'{n_device}_tp{tensor_parallel_size}_{admission_mode}_'
        f'{ttft_slo_scale}_{slo_tpot}_{routing_fallback_policy}'
        f'{session_replay_suffix}{baseline_cap_suffix}'
        f'{perf_model_regression_suffix}')
    requests_trace = trace
    arrival_times_trace = trace
    perf_model_task = _perf_model_task_for_trace_spec(trace)
    requests, arrival_times, trace_components = _load_trace_inputs(
        trace,
        model_name=model_name,
        load_scale=load_scale,
        window=window,
    )
    print(f'trace_components: {trace_components}')
    
    input_lengths = np.fromiter((request.input_length for request in requests), dtype=np.float64, count=len(requests))
    output_lengths = np.fromiter((request.output_length for request in requests), dtype=np.float64, count=len(requests))
    average_input_length = float(np.mean(input_lengths))
    average_output_length = float(np.mean(output_lengths))
    output_length_percentiles = np.percentile(output_lengths, [50, 75, 85, 90, 95, 99])
    max_output_length = int(max(output_lengths))
    input_length_percentiles = np.percentile(input_lengths, [50, 75, 85, 90, 95, 99])
    print(f'average_input_length: {average_input_length}')
    print(f'average_output_length: {average_output_length}')
    print(
        f'input_length_percentiles: '
        f'p50={input_length_percentiles[0]}, '
        f'p75={input_length_percentiles[1]}, '
        f'p85={input_length_percentiles[2]}, '
        f'p90={input_length_percentiles[3]}, '
        f'p95={input_length_percentiles[4]}, '
        f'p99={input_length_percentiles[5]}'
    )
    print(
        f'output_length_percentiles: '
        f'p50={output_length_percentiles[0]}, '
        f'p75={output_length_percentiles[1]}, '
        f'p85={output_length_percentiles[2]}, '
        f'p90={output_length_percentiles[3]}, '
        f'p95={output_length_percentiles[4]}, '
        f'p99={output_length_percentiles[5]}'
    )
    print(f'max_output_length: {max_output_length}')
    from SLOsServe.perf_model import PerfModel
    perf_model = PerfModel.get_perf_model(model_name, perf_model_task)
    loaded_perf_model_params = perf_model.describe_hardware_params()
    loaded_perf_model_type = (
        "piecewise_current_tokens"
        if perf_model.is_piecewise_current_tokens else "linear"
    )
    loaded_perf_model_breakpoints = (
        loaded_perf_model_params.get("breakpoints")
        if isinstance(loaded_perf_model_params, dict) else None
    )
    print(
        "loaded_perf_model:",
        f"model={model_name}",
        f"task={perf_model_task}",
        f"type={loaded_perf_model_type}",
        f"breakpoints={loaded_perf_model_breakpoints}",
    )
    max_decode_batch_size = perf_model.get_max_decode_batch_size(slo_tpot, average_input_length)
    decode_zero_load = perf_model.get_batch_time([(0, 1)])
    
    assert max_decode_batch_size > 0
    
    print(f'max_decode_batch_size: {max_decode_batch_size}')
    
    slo_ttft_per_token, slo_ttft_constant = perf_model.get_zero_load_prefill_affine_params()
    slo_ttft_per_token *= ttft_slo_scale
    slo_ttft_constant *= ttft_slo_scale
    assert slo_tpot >= decode_zero_load
    
    average_prefill_time = perf_model.get_batch_time([(0, average_input_length)])
    average_decode_time = slo_tpot * average_output_length / max_decode_batch_size
    optimal_prefill_ratio = average_prefill_time / (average_prefill_time + average_decode_time)
        
    if profit == 'constant': 
        profit_per_input_token = 0.0
        profit_per_output_token = 0.0
        profit_base = 1.0
    elif profit == 'weighted':
        profit_per_input_token = 1.25e-6
        profit_per_output_token = 10.0e-6
        profit_base = 0
    
    raw_default_capped_baseline_tokens = max(
        1,
        min(max_decode_batch_size - 10, 16384),
    )
    default_capped_baseline_tokens = min(
        16384,
        ((raw_default_capped_baseline_tokens + 63) // 64) * 64,
    )
    capped_baseline_tokens = (
        int(baseline_decode_cap)
        if baseline_decode_cap is not None else default_capped_baseline_tokens
    )
    if capped_baseline_tokens <= 0:
        raise ValueError(
            f"baseline_decode_cap must be positive, got {baseline_decode_cap}"
        )
    print(
        f'capped_baseline_tokens: {capped_baseline_tokens} '
        f'(default={default_capped_baseline_tokens}, '
        f'raw_default={raw_default_capped_baseline_tokens})'
    )
    ablation_no_global = False
    ablation_no_local = False
    oracle_mem = False
    scheduling_kwargss = []
    if scheduling_policy == 'vllm':
        scheduling_kwargss.append({
            'scheduling_policy': 'vllm',
            'max_num_batched_tokens': 16384,
            'long_prefill_token_threshold': 16384,
            'max_num_seqs': 512,
            'enable_chunked_prefill': False,
            'enable_admission': False,
            'allow_rejection': False
        })
    elif scheduling_policy == 'vllm+': 
        scheduling_kwargss.append({
            'scheduling_policy': 'vllm+',
            'max_num_batched_tokens': 16384,
            'long_prefill_token_threshold': 16384,
            'max_num_seqs': 512,
            'enable_chunked_prefill': False,
            'enable_admission': True,
            'allow_rejection': True
        })
    elif scheduling_policy == 'sarathi':
        scheduling_kwargss.append({
            'scheduling_policy': 'vllm-sarathi',
            'max_num_batched_tokens': capped_baseline_tokens,
            'long_prefill_token_threshold': capped_baseline_tokens,
            'max_num_seqs': 512,
            'enable_chunked_prefill': True,
            'enable_admission': False,
            'allow_rejection': False,
        })
        
    elif scheduling_policy == 'sarathi+':
        scheduling_kwargss.append({
            'max_num_batched_tokens': capped_baseline_tokens,
            'long_prefill_token_threshold': capped_baseline_tokens,
            'max_num_seqs': 512,
            'enable_chunked_prefill': True,
            'enable_admission': False,
            'allow_rejection': False,
            'scheduling_policy': 'vllm-edf-sarathi+'
        })
    
    elif scheduling_policy == 'qlm':
        # for maximum_queue_length in [10, 20, 50]:
        scheduling_kwargss.append({
            'enable_chunked_prefill': True,
            'max_num_batched_tokens': 16384,
            'max_num_seqs': 512,
            'long_prefill_token_threshold': 16384,
            'enable_admission': False,
            'allow_rejection': False,
            'scheduling_policy': 'vllm-edf',
        })
    elif scheduling_policy == 'qlm+':
        scheduling_kwargss.append({
            'enable_chunked_prefill': False,
            'max_num_batched_tokens': 16384,
            'max_num_seqs': 512,
            'long_prefill_token_threshold': 16384,
            'enable_admission': True,
            'allow_rejection': True,
            'scheduling_policy': 'vllm-edf',
        })

    elif scheduling_policy == 'slosserve-edf':
        scheduling_kwargss.append({
            'scheduling_policy': 'edf',
            'enable_admission': True,
            "allow_rejection": True,
            "scheduling_overhead": scheduling_overhead,
            "slosserve_token_headroom": 1,
            "max_num_batched_tokens": 16384
        })
        
    elif scheduling_policy == 'atfc':
        scheduling_kwargss.append({
            'scheduling_policy': 'atfc',
            'enable_admission': True,
            "allow_rejection": True,
            "scheduling_overhead": scheduling_overhead,
            "max_num_batched_tokens": 16384,
            "length_pattern": perf_model_task
        })

    elif scheduling_policy == 'slosserve-dp':
        scheduling_kwargss.append({
            'scheduling_policy': 'dp',
            'enable_admission': True,
            "allow_rejection": True,
            "scheduling_overhead": 0.003,
            "slosserve_token_headroom": 1
        })
        
    elif scheduling_policy == 'slosserve-edf-fair':
        scheduling_kwargss.append({
            'scheduling_policy': 'edf',
            'fifo_fair': True,
            'enable_admission': True,
            "allow_rejection": True,
            "scheduling_overhead": 0.000,
            "slosserve_token_headroom": 1
        })

    for sch_kwargs in scheduling_kwargss:
        sch_kwargs['max_decoding_length'] = max_output_length
    
    routing_kwargss = []
    if routing_policy == 'round_robin_session':
        admission_mode = "off" # w/ round robin, we let each request ends
        routing_kwargss = [{
            "enable_rerouting": False,
            "enable_rescheduling": False,
            "sticky_sessions": True,
        }]
    elif 'round_robin' in routing_policy:
        _args = routing_policy.split('-')
        routing_policy = 'round_robin'
        admission_mode = "off" # w/ round robin, we let each request ends 
        group_size = n_device
        extra_kwargs = {}
        if len(_args) >= 2:
            assert _args[1] == 'disagg'
            opt_n_prefill_devices = int(optimal_prefill_ratio * group_size)
            opt_n_prefill_devices = min(max(opt_n_prefill_devices, 1), n_device - 1)
            print(f'opt_n_prefill_devices: {opt_n_prefill_devices}')
            extra_kwargs = {
                'group_size': group_size,
                'is_pd_disagg': True, 
                'n_prefill_per_group': opt_n_prefill_devices,
            }
        routing_kwargss = [{"enable_rerouting": False,
                            "enable_rescheduling": False} | extra_kwargs]
    elif 'llumnix_load' in routing_policy:
        _args = routing_policy.split('-')
        routing_policy = 'llumnix_load'
        admission_mode = "off" # w/ round robin, we let each request ends 
        group_size = n_device
        extra_kwargs = {}
        if len(_args) >= 2:
            assert _args[1] == 'disagg'
            opt_n_prefill_devices = int(optimal_prefill_ratio * group_size)
            opt_n_prefill_devices = min(max(opt_n_prefill_devices, 1), n_device - 1)
            print(f'opt_n_prefill_devices: {opt_n_prefill_devices}')
            extra_kwargs = {
                'group_size': group_size,
                'is_pd_disagg': True, 
                'n_prefill_per_group': opt_n_prefill_devices,
            }
        routing_kwargss = [{"enable_rerouting": False,
                            "enable_rescheduling": False} | extra_kwargs]
    elif routing_policy == 'lightest_first':
        routing_policy = 'lightest_first'
        routing_kwargss = [{"enable_rerouting": False,
                            "enable_rescheduling": False}]
    elif routing_policy == 'lightest_first_retry':
        routing_policy = 'lightest_first'
        routing_kwargss = [{"enable_rerouting": False,
                            "enable_rescheduling": True}]
    elif 'disagg_auto_scaling' in routing_policy:
        _args = routing_policy.split('-')
        if len(_args) == 2:
            _, feature = _args
            threshold = None
        elif len(_args) == 3:
            _, feature, threshold = _args
            threshold = float(threshold)
        routing_kwargss = [{"enable_rescheduling": 'resch' in routing_policy,
                            "enable_rerouting": True,
                            "model_path": "auto_scaling_model.json",
                            "model_key": feature,
                            "threshold": threshold,
                            "max_decode_batch_size": max_decode_batch_size}]
    elif 'auto_scaling' in routing_policy:
        _args = routing_policy.split('-')
        if len(_args) == 2:
            _, feature = _args
            threshold = None
        elif len(_args) == 3:
            _, feature, threshold = _args
            threshold = float(threshold)
        routing_kwargss = [{"enable_rescheduling": 'resch' in routing_policy,
                            "enable_rerouting": False,
                            "model_path": "auto_scaling_model.json",
                            "model_key": feature,
                            "threshold": threshold}]
    elif routing_policy == 'round_robin_retry':
        routing_policy = 'round_robin_retry'
        routing_kwargss = [{
            "enable_rescheduling": True,
            "enable_rerouting": False,
            "round_robin_init": True
        }]
    elif routing_policy in ['disaggregated', 'disaggregated-edf']:
        opt_n_prefill_devices = int(optimal_prefill_ratio * n_device)
        print(f'opt_n_prefill_devices: {opt_n_prefill_devices}')
        tx = lambda n: max(min(n, n_device - 1), 1)
        for num_prefill_devices in {tx(opt_n_prefill_devices),
                                    tx(opt_n_prefill_devices) - 1,
                                    tx(opt_n_prefill_devices) + 1}:
            num_decode_devices = n_device - num_prefill_devices
            if num_decode_devices > 0 and num_prefill_devices > 0:
                routing_kwargss.append(f"{num_prefill_devices}P{num_decode_devices}D")
        
    # elif routing_policy == 'slosserve':
    #     routing_policy = 'slosserve'
    #     routing_kwargss.append({
    #         "hardware_params": hardware_params,
    #         "device_mem": 23949,
    #         "tpot": slo_tpot,
    #         "block_size": 16,
    #         "routing_overhead": slo_routing_overhead
    #     })
    elif routing_policy == 'renaming':
        routing_policy = 'renaming'
        routing_kwargss.append({
            "max_decode_batch_size": max_decode_batch_size,
            "enable_rerouting": True
        })
    
    elif 'slosserve' in routing_policy:
        # Supported forms:
        # - slosserve
        # - slosserve_<n_group>[_<n_lb>]
        # - slosserve[_<n_group>[_<n_lb>]]_(disagg|diagg)_<n_prefill_per_group>
        # - any above + _planner suffix (planner path enabled)
        # - any above + _oracle_mem suffix (use per-request decode length for memory bounds)
        # - any above + _ablation_no_global suffix (without any global admission)
        # - any above + _ablation_no_local suffix (ATFC accepts regardless of local feasibility)
        # - plain _ablation is kept as an alias for _ablation_no_global
        # Also accepts '-' as separator.
        logger.info(f'parsing routing policy {routing_policy}')
        normalized_policy = routing_policy.replace('-', '_')
        if 'oracle_mem' in normalized_policy:
            oracle_mem = True
            normalized_policy = normalized_policy.replace('_oracle_mem', '')
        if 'ablation_no_global' in normalized_policy:
            ablation_no_global = True
            normalized_policy = normalized_policy.replace('_ablation_no_global', '')
        if 'ablation_no_local' in normalized_policy:
            ablation_no_local = True
            normalized_policy = normalized_policy.replace('_ablation_no_local', '')
        _args = [x for x in normalized_policy.split('_') if x]
        if not _args or _args[0] != 'slosserve':
            raise ValueError(f"Invalid slosserve routing policy: {routing_policy}")

        group_size = n_device
        n_lb = 1
        extra_kwargs = {}
        use_planner = False
        if 'planner' in _args:
            use_planner = True
            _args = [x for x in _args if x != 'planner']
        if 'ablation' in _args:
            ablation_no_global = True
            _args = [x for x in _args if x != 'ablation']
        disagg_tokens = [tok for tok in ('disagg', 'diagg') if tok in _args]
        if disagg_tokens:
            disagg_idx = min(_args.index(tok) for tok in disagg_tokens)
            numeric_prefix = _args[1:disagg_idx]
            if len(numeric_prefix) >= 1:
                group_size = int(numeric_prefix[0])
            if len(numeric_prefix) >= 2:
                n_lb = int(numeric_prefix[1])
            if len(numeric_prefix) > 2:
                raise ValueError(
                    f"Too many numeric fields before disagg in routing policy: {routing_policy}"
                )
            if disagg_idx + 2 < len(_args):
                raise ValueError(
                    f"Unexpected trailing fields in disagg routing policy: {routing_policy}"
                )
            if disagg_idx + 1 < len(_args):
                n_prefill_server_per_group = int(_args[disagg_idx + 1])
            else:
                # Allow shorthand like slosserve_4_diagg_planner by defaulting to one decode slot/group.
                n_prefill_server_per_group = max(1, group_size - 1)
            extra_kwargs = {
                'is_pd_disagg': True,
                'n_prefill_per_group': n_prefill_server_per_group,
                'max_decode_bs': default_capped_baseline_tokens,
                "enable_rerouting": True,
                'use_planner': use_planner,
                'ablation': ablation_no_global
            }
        else:
            numeric_fields = _args[1:]
            if len(numeric_fields) >= 1:
                group_size = int(numeric_fields[0])
            if len(numeric_fields) >= 2:
                n_lb = int(numeric_fields[1])
            if len(numeric_fields) > 2:
                raise ValueError(
                    f"Too many numeric fields in slosserve routing policy: {routing_policy}"
                )

        routing_kwargss.append({
            "max_decode_length": max_output_length,
            "enable_rescheduling": True,
            "enable_rerouting": False,
            'device_mem': 1248576, # TODO: make it concrete. 
            'block_size': 16, # TODO: make it concrete.
            'model_name': model_name,
            'tpot': slo_tpot,
            'scheduling_overhead': scheduling_overhead,
            'group_size': group_size,
            'n_lb': n_lb,
            'use_planner': use_planner,
            'oracle_mem': oracle_mem,
            'ablation': ablation_no_global
            # 'routing_overhead': slo_routing_overhead
        } | extra_kwargs)

    if scheduling_policy == 'atfc':
        for sch_kwargs in scheduling_kwargss:
            sch_kwargs['ablation_no_local'] = ablation_no_local
            sch_kwargs['oracle_mem'] = oracle_mem

    for routing_kwargs in routing_kwargss:
        if isinstance(routing_kwargs, dict):
            routing_kwargs.setdefault("perf_model_task", perf_model_task)
            routing_kwargs.setdefault("kv_xfer_delay", kv_xfer_delay)
            routing_kwargs.setdefault("perf_model_err", perf_model_err)
    
    return [Problem(
        model_name = model_name,
        arrival_pattern = arrival_times_trace,
        length_pattern = requests_trace,
        trace_spec = trace,
        perf_model_task = perf_model_task,
        window = window,
        load_scale = load_scale,
        n_devices = n_device,
        tensor_parallel_size = tensor_parallel_size,
        ttft_slo_scale = ttft_slo_scale,
        slo_ttft_per_token = slo_ttft_per_token,
        slo_ttft_constant = slo_ttft_constant,
        slo_tpot = slo_tpot,
        slo_routing_overhead = slo_routing_overhead,
        profit_per_input_token = profit_per_input_token,
        profit_per_output_token = profit_per_output_token,
        profit_base = profit_base,
        routing_policy = routing_policy,
        routing_kwargs = routing_kwargs,
        scheduling_policy = scheduling_policy,
        scheduling_kwargs = scheduling_kwargs,
        store_prefix = store_prefix,
        admission_mode = admission_mode,
        perf_model_err = perf_model_err,
        scheduling_overhead = scheduling_overhead,
        routing_overhead = routing_overhead,
        routing_fallback_policy = routing_fallback_policy,
        enable_session_replay = enable_session_replay,
        session_pause_s = session_pause_s,
        log_perf_model_errors = log_perf_model_errors,
        include_perf_model_time_lists = include_perf_model_time_lists,
        draw_perf_model_error_figure = draw_perf_model_error_figure,
        enable_piecewise_perf_model_regression = (
            enable_piecewise_perf_model_regression
        ),
        perf_model_piecewise_breakpoints = (
            None if perf_model_piecewise_breakpoints is None
            else [int(point) for point in perf_model_piecewise_breakpoints]
        ),
    ) for (scheduling_kwargs, routing_kwargs) in product(
        scheduling_kwargss, routing_kwargss)]



SCHEDULING_POLICIES = ['vllm-no_rejection', 'vllm-fcfs', 'vllm-edf', 'slosserve-edf', 'slosserve-dp']
ROUTING_POLICIES = ['round_robin', 'round_robin_session', 'disaggregated', 'disaggregated-edf', 'slosserve', 'renaming']

def run(
    model_name: str,
    ttft_slo_scales: list[float],
    slo_tpots: list[float],
    profit: str,
    trace: str,
    window: str,
    load_scales: list[float],
    n_devices: list[int],
    tensor_parallel_size: int,
    endpoint: str,
    clients: str | None,
    policies: list[str],
    overwrite: bool,
    slo_routing_overhead: float,
    admission_mode: str,
    scheduling_overhead: float,
    output_dir: str,
    perf_model_errs: list[float],
    routing_overhead: float,
    routing_fallback_policy: str,
    kv_xfer_delay: float,
    enable_session_replay: bool,
    session_pause_s: float,
    log_perf_model_errors: bool,
    include_perf_model_time_lists: bool,
    draw_perf_model_error_figure: bool,
    enable_piecewise_perf_model_regression: bool,
    perf_model_piecewise_breakpoints: list[int] | None,
    baseline_decode_cap: int | None,
    update_clients: bool = True,
):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    model_name_easy = get_easy_name(model_name)
    global experiment_dir
    if 'bursty' in trace:
        # cutoff what after bursty
        _trace_name, burstiness_level = trace.split('bursty_')
        _trace_name += 'bursty'
        burstiness_level = float(burstiness_level)
    else:
        _trace_name = trace
        burstiness_level = 0.0
    experiment_dir = os.path.abspath(
        f"{output_dir}/{model_name_easy}_{profit}_{_trace_name}_{window}_{admission_mode}_{slo_routing_overhead}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    print('--Problem Grid--')
    print(f"model_name: {model_name_easy}")
    print(f"ttft_slo_scales: {ttft_slo_scales}")
    print(f"slo_tpots: {slo_tpots}")
    print(f"profit: {profit}")
    print(f"trace: {trace}")
    print(f"window: {window}")
    print(f"load_scales: {load_scales}")
    print(f"n_devices: {n_devices}")
    print(f"tensor_parallel_size: {tensor_parallel_size}")
    print(f"policies: {policies}")
    print(f"Experiment directory: {experiment_dir}")    
    print(f"admission_mode: {admission_mode}")
    print(f"scheduling_overhead: {scheduling_overhead}")
    print(f'performance model errors: {perf_model_errs}')
    print(f'routing fallback policy: {routing_fallback_policy}')
    print(f'kv_xfer_delay: {kv_xfer_delay}')
    print(f'enable_session_replay: {enable_session_replay}')
    print(f'session_pause_s: {session_pause_s}')
    print(f'log_perf_model_errors: {log_perf_model_errors}')
    print(f'include_perf_model_time_lists: {include_perf_model_time_lists}')
    print(f'draw_perf_model_error_figure: {draw_perf_model_error_figure}')
    print(
        'enable_piecewise_perf_model_regression: '
        f'{enable_piecewise_perf_model_regression}'
    )
    print(
        'perf_model_piecewise_breakpoints: '
        f'{perf_model_piecewise_breakpoints}'
    )
    print(f'baseline_decode_cap: {baseline_decode_cap}')
    print('--End of Problem Grid--')
    results = {}
    if os.path.exists(f'{experiment_dir}/results.jsonl'):
        print(f'Loading cached results from {experiment_dir}/results.jsonl')
        with open(f'{experiment_dir}/results.jsonl', 'r') as f:
            results = [json.loads(line) for line in f]
            results = {
                (
                    r['load_scale'],
                    r.get('requested_n_device', r['n_device']),
                    r.get('effective_n_device', r['n_device']),
                    r.get('tensor_parallel_size', 1),
                    r['scheduling_policy'],
                    r['routing_policy'],
                    r['ttft_slo_scale'],
                    r['slo_tpot'],
                    r['perf_model_err'],
                    r.get('baseline_decode_cap'),
                    r.get('enable_piecewise_perf_model_regression', False),
                    tuple(r.get('perf_model_piecewise_breakpoints') or ()),
                    r.get('enable_session_replay', False),
                    r.get('session_pause_s', 0.0),
                ): r for r in results
            }
    else:
        results = {}
    
    for ttft_slo_scale, slo_tpot, load_scale, n_device, policy, perf_model_err in product(\
        ttft_slo_scales, slo_tpots, load_scales, n_devices, policies, perf_model_errs):
        if ':' in policy:
            routing_policy, scheduling_policy = policy.split(':')
        else:
            scheduling_policy = policy
            routing_policy = 'round_robin'
        if n_device == 1 and 'disaggregated' in routing_policy:
            print(f'Skipping {load_scale}, {n_device}, {scheduling_policy}, {routing_policy}, {ttft_slo_scale}, {slo_tpot} because n_device is 1 and routing policy is disaggregated')
            continue
        effective_n_device, _ = _resolve_rr_effective_n_devices(
            n_device,
            routing_policy,
            None,
            clients,
        )
        cache_key = (
            load_scale,
            n_device,
            effective_n_device,
            tensor_parallel_size,
            scheduling_policy,
            routing_policy,
            ttft_slo_scale,
            slo_tpot,
            perf_model_err,
            baseline_decode_cap,
            enable_piecewise_perf_model_regression,
            tuple(perf_model_piecewise_breakpoints or ()),
            enable_session_replay,
            session_pause_s,
        )
        if not overwrite and cache_key in results:
            print(f'Skipping {load_scale}, {n_device}, {scheduling_policy}, {routing_policy}, {ttft_slo_scale}, {slo_tpot}, {perf_model_err} because it already exists')
            continue
        problems = build_problems(
            model_name,
            trace,
            ttft_slo_scale,
            slo_tpot,
            profit,
            scheduling_policy,
            routing_policy,
            n_device,
            tensor_parallel_size,
            window,
            load_scale,
            experiment_dir,
            slo_routing_overhead,
            admission_mode,
            perf_model_err,
            routing_overhead,
            scheduling_overhead = scheduling_overhead,
            routing_fallback_policy=routing_fallback_policy,
            kv_xfer_delay=kv_xfer_delay,
            enable_session_replay=enable_session_replay,
            session_pause_s=session_pause_s,
            log_perf_model_errors=log_perf_model_errors,
            include_perf_model_time_lists=include_perf_model_time_lists,
            draw_perf_model_error_figure=draw_perf_model_error_figure,
            enable_piecewise_perf_model_regression=(
                enable_piecewise_perf_model_regression
            ),
            perf_model_piecewise_breakpoints=perf_model_piecewise_breakpoints,
            baseline_decode_cap=baseline_decode_cap,
        )
        if not len(problems):
            logger.error(f'No problems found for {load_scale=}, {n_device=}, {scheduling_policy=}, {routing_policy=}, {ttft_slo_scale=}, {slo_tpot}')
            continue
        run_results = []
        for problem in problems:
            # print(f"Running problem: {problem}")
            problem.scheduling_kwargs['scheduling_overhead'] = scheduling_overhead
            try:
                exec_result = asyncio.run(
                    main(
                        problem,
                        endpoint,
                        clients,
                        update_clients=update_clients,
                    )
                )
            except (BenchmarkOverloadedError,
                    httpx.HTTPError,
                    aiohttp.ClientError,
                    asyncio.TimeoutError) as exc:
                logger.error(
                    "Benchmark run terminated as overloaded for %s: %r",
                    problem.store_prefix,
                    exc,
                )
                exec_result = _make_overload_run_result(
                    problem,
                    requested_n_devices=n_device,
                    error=exc,
                )
            run_results.append(exec_result)
        best_result = max(
            run_results,
            key=lambda x: (
                not bool(x.results.get('overloaded', False)),
                x.profit,
            ),
        )
        requested_n_device = best_result.results.get('requested_n_device', n_device)
        effective_n_device = best_result.results.get('effective_n_device', n_device)
        result = {
            'load_scale': load_scale,
            'rps': best_result.results['rps'],
            'n_device': effective_n_device,
            'requested_n_device': requested_n_device,
            'effective_n_device': effective_n_device,
            'rr_sliced': best_result.results.get('rr_sliced', False),
            'tensor_parallel_size': tensor_parallel_size,
            'total_gpus': effective_n_device * tensor_parallel_size,
            'requested_total_gpus': requested_n_device * tensor_parallel_size,
            'scheduling_policy': scheduling_policy,
            'routing_policy': routing_policy,
            'profit': best_result.profit,
            'ttft_slo_scale': ttft_slo_scale,
            'slo_tpot': slo_tpot,
            'slo_violation_rate': 1 - best_result.results['slo_attainment_rate'],
            'perf_model_err': perf_model_err,
            'enable_piecewise_perf_model_regression': (
                enable_piecewise_perf_model_regression
            ),
            'perf_model_piecewise_breakpoints': (
                None if perf_model_piecewise_breakpoints is None
                else [int(point) for point in perf_model_piecewise_breakpoints]
            ),
            'baseline_decode_cap': baseline_decode_cap,
            'enable_session_replay': enable_session_replay,
            'session_pause_s': session_pause_s,
            
            'event_file': f'{problems[0].store_prefix}.events.jsonl',
            'energy_consumption': best_result.energy_consumption,
            'per_gpu_energy_consumption': best_result.per_gpu_energy_consumption,
            'scheduling_overhead': scheduling_overhead,
            'burstiness_level': burstiness_level,
            'run_status': best_result.results.get('run_status', 'completed'),
            'overloaded': bool(best_result.results.get('overloaded', False)),
            'rr_slice_kept_request_count': best_result.results.get(
                'rr_slice_kept_request_count'),
            'rr_slice_total_request_count': best_result.results.get(
                'rr_slice_total_request_count'),
        }
        if 'overload_reason' in best_result.results:
            result['overload_reason'] = best_result.results['overload_reason']
        if 'overload_error_type' in best_result.results:
            result['overload_error_type'] = best_result.results[
                'overload_error_type']
        if 'overload_error' in best_result.results:
            result['overload_error'] = best_result.results['overload_error']
        if 'auto_scaling_analysis' in best_result.results:
            result.update(best_result.results['auto_scaling_analysis'])
        if 'extra_metrics' in best_result.results:
            result.update(best_result.results['extra_metrics'])
        if result.get('window_time_pct_vs_active_requests_figure') is not None:
            result['window_time_pct_vs_active_requests_figure'] = (
                f'{problems[0].store_prefix}.window_time_pct_vs_active_requests.png'
            )
        if result.get('window_time_pct_vs_active_requests_figure_pdf') is not None:
            result['window_time_pct_vs_active_requests_figure_pdf'] = (
                f'{problems[0].store_prefix}.window_time_pct_vs_active_requests.pdf'
            )
        if result.get('power_vs_active_servers_and_batch_tokens_figure') is not None:
            result['power_vs_active_servers_and_batch_tokens_figure'] = (
                f'{problems[0].store_prefix}.power_vs_active_servers_and_batch_tokens.png'
            )
        if result.get('power_vs_active_servers_and_batch_tokens_figure_pdf') is not None:
            result['power_vs_active_servers_and_batch_tokens_figure_pdf'] = (
                f'{problems[0].store_prefix}.power_vs_active_servers_and_batch_tokens.pdf'
            )
        if 'perf_model_error_summary' in best_result.results:
            result['perf_model_error_summary'] = best_result.results[
                'perf_model_error_summary']
        if 'perf_model_error_file' in best_result.results:
            result['perf_model_error_file'] = (
                f'{problems[0].store_prefix}.perf_model_errors.jsonl')
        if 'perf_model_error_figure' in best_result.results:
            result['perf_model_error_figure'] = (
                f'{problems[0].store_prefix}.perf_model_estimated_vs_measured.png')
        if 'perf_model_full_elapsed_figure' in best_result.results:
            result['perf_model_full_elapsed_figure'] = (
                f'{problems[0].store_prefix}.'
                'perf_model_estimated_with_overhead_vs_elapsed.png')
        if 'perf_model_regression_figure' in best_result.results:
            result['perf_model_regression_figure'] = (
                f'{problems[0].store_prefix}.perf_model_regression.png')
        if 'perf_model_old_hardware_params' in best_result.results:
            result['perf_model_old_hardware_params'] = best_result.results[
                'perf_model_old_hardware_params']
        if 'perf_model_regressed_hardware_params' in best_result.results:
            result['perf_model_regressed_hardware_params'] = best_result.results[
                'perf_model_regressed_hardware_params']
        if 'perf_model_regressed_hardware_params_delta' in best_result.results:
            result['perf_model_regressed_hardware_params_delta'] = (
                best_result.results[
                    'perf_model_regressed_hardware_params_delta'])
        if 'perf_model_regression_stats' in best_result.results:
            result['perf_model_regression_stats'] = best_result.results[
                'perf_model_regression_stats']
        if 'perf_model_estimated_time_list' in best_result.results:
            result['perf_model_estimated_time_list'] = best_result.results[
                'perf_model_estimated_time_list']
        if 'perf_model_measured_time_list' in best_result.results:
            result['perf_model_measured_time_list'] = best_result.results[
                'perf_model_measured_time_list']
        if 'configured_scheduling_overhead_s' in best_result.results:
            result['configured_scheduling_overhead_s'] = best_result.results[
                'configured_scheduling_overhead_s']
        if 'empirical_scheduling_overhead_summary' in best_result.results:
            result['empirical_scheduling_overhead_summary'] = (
                best_result.results['empirical_scheduling_overhead_summary'])
        print('--Result--')
        pprint.pprint(result)
        print('--End of Result--')
        
        results[cache_key] = result
        with open(f'{experiment_dir}/results.jsonl', 'a') as f:
            f.write(json.dumps(result) + '\n')            
            
        for surfix in [
            'events',
            'reqs',
            'admission_history',
            'perf_model_errors',
        ]:
            src = f'{best_result.event_file}.{surfix}.jsonl'
            dst = f'{problems[0].store_prefix}.{surfix}.jsonl'
            if os.path.exists(src):
                os.system(f'cp {src} {dst}')
        src = f'{best_result.event_file}.perf_model_estimated_vs_measured.png'
        dst = f'{problems[0].store_prefix}.perf_model_estimated_vs_measured.png'
        if os.path.exists(src):
            os.system(f'cp {src} {dst}')
        src = (
            f'{best_result.event_file}.'
            'perf_model_estimated_with_overhead_vs_elapsed.png'
        )
        dst = (
            f'{problems[0].store_prefix}.'
            'perf_model_estimated_with_overhead_vs_elapsed.png'
        )
        if os.path.exists(src):
            os.system(f'cp {src} {dst}')
        src = f'{best_result.event_file}.perf_model_regression.png'
        dst = f'{problems[0].store_prefix}.perf_model_regression.png'
        if os.path.exists(src):
            os.system(f'cp {src} {dst}')
        for figure_suffix in [
            'window_time_pct_vs_active_requests.png',
            'window_time_pct_vs_active_requests.pdf',
            'power_vs_active_servers_and_batch_tokens.png',
            'power_vs_active_servers_and_batch_tokens.pdf',
        ]:
            src = f'{best_result.event_file}.{figure_suffix}'
            dst = f'{problems[0].store_prefix}.{figure_suffix}'
            if os.path.exists(src):
                os.system(f'cp {src} {dst}')
    
        for r in run_results:
            os.system(f'rm {r.event_file}*')

    results = list(results.values())

    # Convert results to DataFrame and save source data
    df = pd.DataFrame(results)
    df.to_csv(f'{experiment_dir}/profit_vs_n_device_and_load.csv', index=False)
    print(f"Saved source data to {experiment_dir}/profit_vs_n_device_and_load.csv")

    # 1. Plot: for each (scheduling_policy, routing_policy) pair, show profit vs n_device (for each load_scale)
    import math
    

    # 1. Plot: for each load_scale, create a subfigure showing profit vs n_device for each (scheduling_policy, routing_policy) pair
    os.makedirs(f'{experiment_dir}/figs', exist_ok=True)
    features = ['load_scale', 'n_device', 'ttft_slo_scale', 'slo_tpot']
    for feature in features:
        if len(df[feature].unique()) == 1:
            continue
        other_features = [f for f in features if f != feature]
        n_groups = len(df.groupby(other_features))
        ncols = min(3, n_groups)
        nrows = math.ceil(n_groups / ncols)
        for xlabel, ylabel in [
            (feature, 'energy_est'),
            (feature, 'slo_violation_rate'),
            (feature, 'energy_consumption'),
            (feature, 'energy_consumption_active'),
            (feature, 'energy_consumption_non_idle'),
            ('slo_violation_rate', 'energy_consumption')
        ]:
            if ylabel not in df.columns:
                continue
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
            idx = 0
            for other_feature_values, group in df.groupby(other_features):
                row, col = divmod(idx, ncols)
                ax = axes[row][col]
                idx += 1
                for (sched, route), group in group.groupby(['scheduling_policy', 'routing_policy']):
                    group_sorted = group.sort_values(xlabel)
                    label = f"{sched} / {route}"
                    ax.plot(group_sorted[xlabel], group_sorted[ylabel], marker='o', label=label)
                other_features_dict = {f: v for f, v in zip(other_features, other_feature_values)}
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(f'{ylabel} vs {xlabel}\n({other_features_dict})')
                ax.legend()
            fig.tight_layout()
            fig.savefig(f'{experiment_dir}/figs/{ylabel}_vs_{xlabel}_change_{feature}.png', dpi=300)
            print(f"Saved plot to {experiment_dir}/figs/{ylabel}_vs_{xlabel}_change_{feature}.png")

PROBLEM_GRID = {
    'model_name': [
        'Qwen/Qwen2.5-7B-Instruct',
    ],
    'ttft_slo_scales': [2.0, 5.0, 10.0],
    'slo_tpots': [1.5, 3.0, 5.0],
    'profit': ['constant', 'weighted'],
    'trace': [
        'azure_code_23', 
        'azure_chat_23', 
        'azure_code',
        'azure_chat',
        'deepseek-r1:azure_chat',
        'deepseek-r1:azure_code',
    ],
}

example_command = """
python motivation/bench_api_server.py \
    --run_type devices \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --slo strict \
    --profit constant \
    --trace azure_code_23+azure_chat_23 \
    --window 0:10 \
    --load_scale 1.0 \
    --n_devices 2 4 8
"""


if __name__ == '__main__':
    problem = Problem()
    # execution_results = asyncio.run(main(problem, endpoint))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--ttft_slo_scales', type=float, default=[2.0], nargs='+', help = 'list of relative ttft slo (defined as slowdown to zero-load ttft)')
    parser.add_argument('--slo_tpots', type=float, default=[2.0], nargs='+', help = 'list of relative tpot slo (defined as absolute tpot per token in seconds)')
    parser.add_argument('--profit', type=str, default='constant', choices=['constant', 'weighted'])
    parser.add_argument(
        '--trace',
        type=str,
        nargs='+',
        default=['azure_code_23'],
        help=(
            "list of trace specs to run. Each spec is TRACE or LENGTH:ARRIVAL. "
            "Use '+' to mix multiple sources in one run, for example "
            "'azure_chat_23:azure_chat_23+azure_code_23:azure_code_23'. "
            "A legacy '-' separator is accepted only when dataset names are unambiguous."
        ),
    )
    parser.add_argument('--window', type=str, default='0:1000', help = 'window of trace to run (inclusive)')
    parser.add_argument('--n_devices', type=int, default=[1,2,4,8], nargs='+', help = 'list of logical replicas to run')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='number of GPUs per logical replica')
    parser.add_argument('--load_scales', type=float, default=[0.5,1.0,2.0,3.0,4.0], nargs='+', help = 'list of load scales (we rescale the arrival rate by load scale, higher load scale means higher query per second)')
    parser.add_argument('--router_ports', type=str, default='8001:4', help = 'port of router to run (inclusive)')
    parser.add_argument('--clients', type=str, default=None, help = 'logical replica labels (e.g. r0,r1,r2,r3) or legacy URL pool spec')
    parser.add_argument('--skip_update_clients', action='store_true', default=False,
                        help='assume the router already started with the desired clients and skip POST /update_clients')
    parser.add_argument('--run_all', action = 'store_true')
    # parser.add_argument('--scheduling_policies', type=str, default=SCHEDULING_POLICIES, nargs='+')
    # parser.add_argument('--routing_policies', type=str, default=ROUTING_POLICIES, nargs='+')
    parser.add_argument('--overwrite', action = 'store_true')
    parser.add_argument('--slo_routing_overhead', type=float, default=0.02)
    parser.add_argument('--admission_mode', type=str, default='arrival', choices=['arrival', 'anytime'], help = 'arrival: instant decision at arrival, anytime: admission can be made anytime.')
    parser.add_argument('--policies', type=str, default=[':'.join([a,b]) for a, b in product(ROUTING_POLICIES, SCHEDULING_POLICIES)], nargs='+', help = 'list of policies to run (routing_policy:scheduling_policy [routing_policy:scheduling_policy ...])')
    parser.add_argument('--scheduling_overhead', type=float, default=0.003, help = 'scheduling overhead per token in seconds')
    parser.add_argument('--output_dir', type=str, default='experiments', help = 'output directory')
    parser.add_argument('--perf_model_err', type = float, default = [1.0], nargs = '+', help = 'list of performance model errors')
    parser.add_argument('--routing_overhead', type = float, default = -1.0, help = "mocked overhead from at engine.")
    parser.add_argument('--routing_fallback_policy', type = str, default = 'asap', choices = ['asap', 'reject'])
    parser.add_argument('--kv_xfer_delay', type=float, default=0.05, help='KV transfer delay budget in seconds')
    parser.add_argument('--enable_session_replay', action='store_true', default=False,
                        help='serialize turns within each session and delay later turns until the previous one finishes')
    parser.add_argument('--session_pause_s', type=float, default=0.0,
                        help='fixed think-time delay between turns of the same session when replay is enabled')
    parser.add_argument('--disable_perf_model_error_log', action='store_true',
                        help='disable post-run batch-level estimated-vs-measured perf-model error logging')
    parser.add_argument('--include_perf_model_time_lists', action='store_true',
                        help='include raw estimated_time and measured_time lists in perf-model outputs and results.jsonl')
    parser.add_argument('--disable_perf_model_error_figure', action='store_true',
                        help='disable the default estimated-vs-measured perf-model figure')
    parser.add_argument(
        '--piecewise_perf_model_breakpoints',
        type=int,
        nargs='+',
        default=None,
        help=(
            'enable piecewise perf-model regression for the post-run error log '
            'and use these current-token separation points, for example '
            '--piecewise_perf_model_breakpoints 512 2048'
        ),
    )
    parser.add_argument(
        '--enable_piecewise_perf_model_regression',
        action='store_true',
        default=False,
        help=(
            'enable piecewise perf-model regression for the post-run error log '
            'using the default breakpoints 512 2048 unless '
            '--piecewise_perf_model_breakpoints is provided'
        ),
    )
    parser.add_argument(
        '--baseline_decode_cap',
        '--sarathi_max_decode_batch_size',
        dest='baseline_decode_cap',
        type=int,
        default=None,
        help=(
            'override the capped token budget used by Sarathi-like baselines '
            '(Sarathi and QLM); defaults to the perf-model-derived cap'
        ),
    )
    args = parser.parse_args()
    
    if not args.run_all:
        clients = _normalize_router_clients_arg(args.clients)
        
        for trace in args.trace:
            run (args.model_name,
                args.ttft_slo_scales,
                args.slo_tpots,
                args.profit,
                trace,
                args.window,
                args.load_scales,
                args.n_devices,
                args.tensor_parallel_size,
                endpoint = f'http://localhost:{args.port}',
                clients = clients,
                policies = args.policies,
                overwrite = args.overwrite,
                slo_routing_overhead = args.slo_routing_overhead,
                admission_mode = args.admission_mode,
                scheduling_overhead = args.scheduling_overhead,
                output_dir = args.output_dir,
                perf_model_errs = args.perf_model_err,
                routing_overhead = args.routing_overhead,
                routing_fallback_policy = args.routing_fallback_policy,
                kv_xfer_delay = args.kv_xfer_delay,
                enable_session_replay = args.enable_session_replay,
                session_pause_s = args.session_pause_s,
                log_perf_model_errors = not args.disable_perf_model_error_log,
                include_perf_model_time_lists = args.include_perf_model_time_lists,
                draw_perf_model_error_figure = (
                    not args.disable_perf_model_error_figure),
                enable_piecewise_perf_model_regression = (
                    args.enable_piecewise_perf_model_regression
                    or args.piecewise_perf_model_breakpoints is not None
                ),
                perf_model_piecewise_breakpoints = (
                    args.piecewise_perf_model_breakpoints
                ),
                baseline_decode_cap = args.baseline_decode_cap,
                update_clients = not args.skip_update_clients,
            )
        exit(0)
    
    from itertools import product
    import multiprocessing
    
    problem_grids = product(PROBLEM_GRID['model_name'],
                            PROBLEM_GRID['profit'],
                            PROBLEM_GRID['trace'],
                            args.n_devices)
    running_jobs = []
    router_start, n_router_ports = map(int, args.router_ports.split(':'))
    client_start, n_client_ports = map(int, args.clients.split(':'))
    routers = [f'http://localhost:{router_start + i}' for i in range(n_router_ports)]
    clients = [f'http://localhost:{client_start + i}' for i in range(n_client_ports)]
    for grid in problem_grids:
        model_name, profit, trace, n_device = grid
        while len(routers) == 0 or len(clients) < n_device:
            time.sleep(1)
            for p, clients_str, router in running_jobs[:]:
                if p.exitcode is None:
                    continue
                p.join()
                exit_code = p.exitcode
                if exit_code != 0:
                    print(f"Error: Command on GPUs {router} failed with return code {exit_code}: {p.args}")
                running_jobs.remove((p, clients_str, router))
                routers.append(router)
                clients.extend(clients_str.split(','))
        router = routers.pop(0)
        allocated_clients = clients[:n_device]
        clients = clients[n_device:]
        clients_str = ','.join(allocated_clients)

        p = multiprocessing.Process(
            target=run,
            kwargs = {
                'model_name': model_name,
                'ttft_slo_scales': args.ttft_slo_scales,
                'slo_tpots': args.slo_tpots,
                'profit': profit,
                'policies': args.policies,
                'trace': trace,
                'window': args.window,
                'load_scales': args.load_scales,
                'n_devices': [n_device],
                'tensor_parallel_size': args.tensor_parallel_size,
                'endpoint': router,
                'clients': clients_str,
                'overwrite': args.overwrite,
                'slo_routing_overhead': args.slo_routing_overhead,
                'admission_mode': args.admission_mode,
                'routing_overhead': args.routing_overhead,
                'routing_fallback_policy': args.routing_fallback_policy,
                'scheduling_overhead': args.scheduling_overhead,
                'output_dir': args.output_dir,
                'perf_model_errs': args.perf_model_err,
                'kv_xfer_delay': args.kv_xfer_delay,
                'enable_session_replay': args.enable_session_replay,
                'session_pause_s': args.session_pause_s,
                'log_perf_model_errors': not args.disable_perf_model_error_log,
                'include_perf_model_time_lists': args.include_perf_model_time_lists,
                'draw_perf_model_error_figure': (
                    not args.disable_perf_model_error_figure),
                'enable_piecewise_perf_model_regression': (
                    args.enable_piecewise_perf_model_regression
                    or args.piecewise_perf_model_breakpoints is not None
                ),
                'perf_model_piecewise_breakpoints': (
                    args.piecewise_perf_model_breakpoints
                ),
                'update_clients': not args.skip_update_clients,
            })
        p.start()
        running_jobs.append((p, clients_str, router))

'''
python motivation/bench_api_server.py \
    --run_all \
    --ttft_slo_scales 2.0 5.0 10.0 \
    --slo_tpots 1.5 3.0 5.0 \
    --window 0:1000 \
    --load_scales 1.0 \
    --policies round_robin:vllm-fcfs round_robin:vllm-edf round_robin:slosserve-edf round_robin:slosserve-dp disaggregated:vllm-edf disaggregated:slosserve-edf renaming:slosserve \
    --n_devices 1 2 4 8 16 \
    --router_ports 8001:8 \
    --clients 8501:16
    
python motivation/bench_api_server.py \
    --run_all \
    --ttft_slo_scales 2.0 5.0 10.0 \
    --slo_tpots 1.5 3.0 5.0 \
    --window 0:1000 \
    --load_scales 0.1 0.2 0.4 0.6 0.8 1.0 \
    --policies round_robin:vllm-fcfs round_robin:vllm-edf round_robin:slosserve-edf round_robin:slosserve-dp \
    --n_devices 1 \
    --router_ports 8009:8
    --clients 8517:8
'''
    
    # print('--PROBLEM--')
    # print(problem)
    
    # print('--RESULTS--')
    # print('slo_violation_rate', execution_results.slo_violation_rate)
    # print('profit', execution_results.profit)
    # print('more results', execution_results.results)
    
'''
curl -X POST -s http://0.0.0.0:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "prompt": "San Francisco is a",
    "max_tokens": 10,
    "temperature": 0,
    "stream": true,
    "vllm_xargs": {
        "input_length": 10,
        "output_length": 10,
        "prefill_ddl": 1,
        "profit": 1
    }
}'
'''
