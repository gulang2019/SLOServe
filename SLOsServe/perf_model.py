import argparse
import copy
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from SLOsServe.fitting_utils import (
    fit_linear_perf_model,
    sanitize_filename,
    save_prediction_scatter,
    write_json,
)
from SLOsServe.model_config import get_model_config

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
PERF_MODEL_PATH = ASSETS_DIR / "perf_model.json"
PERF_MODEL_FIG_DIR = ASSETS_DIR / "perf_model_figs"

BATCH_CATEGORY_ORDER = ("prefill", "decode", "mixed", "unknown")
BATCH_CATEGORY_COLORS = {
    "prefill": "#1f77b4",
    "decode": "#ff7f0e",
    "mixed": "#2ca02c",
    "unknown": "#7f7f7f",
}
DEFAULT_CURRENT_TOKEN_BREAKPOINTS = (512, 2048)


def _coerce_non_negative_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _coerce_non_negative_float(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default


def normalize_current_token_breakpoints(
    breakpoints: tuple[int, ...] | list[int] | None,
) -> tuple[int, ...]:
    if breakpoints is None:
        breakpoints = DEFAULT_CURRENT_TOKEN_BREAKPOINTS

    normalized: list[int] = []
    for raw_breakpoint in breakpoints:
        try:
            breakpoint = int(raw_breakpoint)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid current-token breakpoint: {raw_breakpoint!r}"
            ) from exc
        if breakpoint < 0:
            raise ValueError(
                f"Current-token breakpoints must be non-negative: {breakpoint}"
            )
        normalized.append(breakpoint)

    deduped = tuple(sorted(set(normalized)))
    if not deduped:
        raise ValueError("At least one current-token breakpoint is required.")
    return deduped


def iter_current_token_piece_segments(
    breakpoints: tuple[int, ...] | list[int] | None = None,
) -> list[dict[str, Any]]:
    normalized = normalize_current_token_breakpoints(breakpoints)
    segments: list[dict[str, Any]] = []
    lower_bound = 0
    for upper_bound in normalized:
        if lower_bound == 0:
            segment_key = f"le_{upper_bound}"
            label = f"<= {upper_bound}"
        else:
            segment_key = f"{lower_bound}_to_{upper_bound}"
            label = f"{lower_bound}-{upper_bound}"
        segments.append({
            "segment_key": segment_key,
            "label": label,
            "min_current_tokens": lower_bound,
            "max_current_tokens": upper_bound,
        })
        lower_bound = upper_bound + 1

    final_lower = normalized[-1] + 1
    segments.append({
        "segment_key": f"gt_{normalized[-1]}",
        "label": f"> {normalized[-1]}",
        "min_current_tokens": final_lower,
        "max_current_tokens": None,
    })
    return segments


def get_current_token_piece(
    total_current_tokens: int,
    breakpoints: tuple[int, ...] | list[int] | None = None,
) -> dict[str, Any]:
    normalized_total_current_tokens = _coerce_non_negative_int(total_current_tokens)
    for segment in iter_current_token_piece_segments(breakpoints):
        max_current_tokens = segment["max_current_tokens"]
        if max_current_tokens is None:
            return segment
        if normalized_total_current_tokens <= max_current_tokens:
            return segment
    raise RuntimeError("Failed to classify current-token segment.")


def load_batch_trace_events(trace_path: str | Path) -> list[dict[str, Any]]:
    path = Path(trace_path).expanduser()
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    def normalize_payload(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            events = payload.get("events")
            if isinstance(events, list):
                return [item for item in events if isinstance(item, dict)]
            return [payload]
        raise ValueError(f"Unsupported trace payload in {path}")

    try:
        return normalize_payload(json.loads(raw_text))
    except json.JSONDecodeError:
        pass

    events: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse {path} as JSONL at line {line_no}: {exc}"
            ) from exc
        events.extend(normalize_payload(payload))
    return events


def classify_batch_category(batch: list[tuple[int, int]]) -> str:
    scheduled_tokens = [current_tokens for _, current_tokens in batch if current_tokens > 0]
    if not scheduled_tokens:
        return "unknown"
    if max(scheduled_tokens) == 1:
        return "decode"
    if min(scheduled_tokens) > 1:
        return "prefill"
    return "mixed"


def extract_batch_perf_sample(
    event: dict[str, Any],
    *,
    subtract_scheduling_overhead: bool = True,
) -> dict[str, Any] | None:
    if not isinstance(event, dict) or event.get("event_type") != "batch":
        return None

    req_ids = event.get("req_ids")
    computed_tokens = event.get("num_computed_tokens")
    scheduled_tokens = event.get("num_scheduled_tokens")
    if not isinstance(req_ids, list) or not isinstance(computed_tokens, list):
        return None
    if not isinstance(scheduled_tokens, dict):
        return None

    batch: list[tuple[int, int]] = []
    for idx, req_id in enumerate(req_ids):
        raw_past_tokens = computed_tokens[idx] if idx < len(computed_tokens) else 0
        raw_current_tokens = scheduled_tokens.get(req_id)
        if raw_current_tokens is None:
            raw_current_tokens = scheduled_tokens.get(str(req_id), 0)
        past_tokens = _coerce_non_negative_int(raw_past_tokens)
        current_tokens = _coerce_non_negative_int(raw_current_tokens)
        if past_tokens == 0 and current_tokens == 0:
            continue
        batch.append((past_tokens, current_tokens))

    if not batch:
        return None

    elapsed = _coerce_non_negative_float(event.get("elapsed", 0.0))
    scheduling_overhead = _coerce_non_negative_float(
        event.get("scheduling_overhead", 0.0)
    )
    measured_time = (
        max(0.0, elapsed - scheduling_overhead)
        if subtract_scheduling_overhead
        else elapsed
    )

    try:
        device_id = int(event.get("device_id", 0))
    except (TypeError, ValueError):
        device_id = 0
    try:
        batch_id = int(event.get("batch_id", -1))
    except (TypeError, ValueError):
        batch_id = -1

    return {
        "device_id": device_id,
        "batch_id": batch_id,
        "timestamp": _coerce_non_negative_float(event.get("timestamp", 0.0)),
        "elapsed_time": elapsed,
        "scheduling_overhead": scheduling_overhead,
        "measured_time": measured_time,
        "estimated_time": _coerce_non_negative_float(event.get("estimated_time", 0.0)),
        "batch_size": len(batch),
        "total_current_tokens": int(sum(current for _, current in batch)),
        "total_past_tokens": int(sum(past for past, _ in batch)),
        "batch_category": classify_batch_category(batch),
        "batch": batch,
    }


def collect_batch_perf_samples(
    trace_path: str | Path,
    *,
    subtract_scheduling_overhead: bool = True,
    device_id: int | None = None,
) -> dict[str, Any]:
    events = load_batch_trace_events(trace_path)
    rows: list[dict[str, Any]] = []
    batch_times: list[tuple[list[tuple[int, int]], float]] = []
    invalid_event_count = 0

    for event in events:
        sample = extract_batch_perf_sample(
            event,
            subtract_scheduling_overhead=subtract_scheduling_overhead,
        )
        if sample is None:
            invalid_event_count += 1
            continue
        if device_id is not None and sample["device_id"] != device_id:
            continue
        rows.append(sample)
        batch_times.append((sample["batch"], float(sample["measured_time"])))

    return {
        "trace_path": str(Path(trace_path).expanduser().resolve()),
        "loaded_event_count": len(events),
        "invalid_event_count": invalid_event_count,
        "device_id": device_id,
        "rows": rows,
        "batch_times": batch_times,
    }


def _summarize_numeric(values: list[float] | list[int]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _compute_prediction_stats(
    measured_times: list[float],
    predicted_times: list[float],
) -> dict[str, float]:
    measured = np.asarray(measured_times, dtype=float)
    predicted = np.asarray(predicted_times, dtype=float)
    if measured.size == 0 or predicted.size == 0 or measured.size != predicted.size:
        raise ValueError(
            "Measured and predicted times must be non-empty arrays of equal length."
        )

    residuals = predicted - measured
    centered = measured - measured.mean()
    ss_res = float(np.sum((measured - predicted) ** 2))
    ss_tot = float(np.sum(centered ** 2))
    positive_mask = measured > 0.0

    stats = {
        "num_samples": int(measured.size),
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mean_bias": float(np.mean(residuals)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
    }
    if np.any(positive_mask):
        abs_relative_errors = np.abs(residuals[positive_mask]) / measured[positive_mask]
        stats["mape"] = float(np.mean(abs_relative_errors))
        stats["p95_abs_relative_error"] = float(np.percentile(abs_relative_errors, 95))
    return stats


def _derive_plot_path(
    plot_path: str | Path | None,
    *,
    default_name: str,
    suffix: str = "",
) -> Path:
    if plot_path is None:
        return PERF_MODEL_FIG_DIR / f"{default_name}{suffix}.png"
    base_path = Path(plot_path)
    return base_path.with_name(f"{base_path.stem}{suffix}{base_path.suffix or '.png'}")


def save_prediction_scatter_by_category(
    path: str | Path,
    measured_times: list[float],
    predicted_times: list[float],
    categories: list[str],
    *,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    measured = np.asarray(measured_times, dtype=float)
    predicted = np.asarray(predicted_times, dtype=float)
    if measured.size != predicted.size or measured.size != len(categories):
        raise ValueError("Measured times, predicted times, and categories must align.")

    fig, ax = plt.subplots(figsize=(6.5, 6.5), tight_layout=True)
    plotted = False
    for category in BATCH_CATEGORY_ORDER:
        mask = np.asarray([item == category for item in categories], dtype=bool)
        if not np.any(mask):
            continue
        plotted = True
        ax.scatter(
            measured[mask],
            predicted[mask],
            s=14,
            alpha=0.7,
            color=BATCH_CATEGORY_COLORS[category],
            label=f"{category} (n={int(mask.sum())})",
        )
    if not plotted:
        ax.scatter(measured, predicted, s=14, alpha=0.7)

    if measured.size > 0 and predicted.size > 0:
        lo = float(min(measured.min(), predicted.min()))
        hi = float(max(measured.max(), predicted.max()))
        ax.plot([lo, hi], [lo, hi], "--r", linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured Time (s)")
    ax.set_ylabel("Predicted Time (s)")
    ax.set_title(title)
    if plotted:
        ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts = {
        category: int(sum(row["batch_category"] == category for row in rows))
        for category in BATCH_CATEGORY_ORDER
        if any(row["batch_category"] == category for row in rows)
    }
    return {
        "device_ids": sorted({int(row["device_id"]) for row in rows}),
        "batch_categories": category_counts,
        "batch_size": _summarize_numeric(
            [int(row["batch_size"]) for row in rows]
        ),
        "total_current_tokens": _summarize_numeric(
            [int(row["total_current_tokens"]) for row in rows]
        ),
        "total_past_tokens": _summarize_numeric(
            [int(row["total_past_tokens"]) for row in rows]
        ),
        "measured_time": _summarize_numeric(
            [float(row["measured_time"]) for row in rows]
        ),
    }


def fit_piecewise_current_token_model(
    rows: list[dict[str, Any]],
    *,
    breakpoints: tuple[int, ...] | list[int] | None = None,
    min_abs_num_reqs_coef: float = 1e-9,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No batch rows available for piecewise fitting.")

    normalized_breakpoints = normalize_current_token_breakpoints(breakpoints)
    segment_descriptors = iter_current_token_piece_segments(normalized_breakpoints)
    segment_rows: dict[str, list[tuple[int, dict[str, Any]]]] = {
        descriptor["segment_key"]: []
        for descriptor in segment_descriptors
    }
    for idx, row in enumerate(rows):
        descriptor = get_current_token_piece(
            int(row["total_current_tokens"]),
            normalized_breakpoints,
        )
        segment_rows[descriptor["segment_key"]].append((idx, row))

    predicted_times: list[float] = [0.0] * len(rows)
    segment_reports: dict[str, Any] = {}
    for descriptor in segment_descriptors:
        items = segment_rows.get(descriptor["segment_key"], [])
        if not items:
            continue
        piece_batch_times = [
            (list(row["batch"]), float(row["measured_time"]))
            for _, row in items
        ]
        piece_fit_result = fit_linear_perf_model(
            piece_batch_times,
            min_abs_num_reqs_coef=min_abs_num_reqs_coef,
        )
        for (row_idx, _), predicted_time in zip(items, piece_fit_result["predicted_times"]):
            predicted_times[row_idx] = float(predicted_time)
        piece_rows = [row for _, row in items]
        segment_reports[descriptor["segment_key"]] = {
            "label": descriptor["label"],
            "min_current_tokens": descriptor["min_current_tokens"],
            "max_current_tokens": descriptor["max_current_tokens"],
            "used_sample_count": len(items),
            "hardware_params": piece_fit_result["hardware_params"],
            "fit_stats": piece_fit_result["stats"],
            "fitted_estimator_stats": _compute_prediction_stats(
                piece_fit_result["measured_times"],
                piece_fit_result["predicted_times"],
            ),
            "existing_estimator_stats": _compute_prediction_stats(
                piece_fit_result["measured_times"],
                [float(row["estimated_time"]) for row in piece_rows],
            ),
            "sample_summary": _summarize_rows(piece_rows),
        }

    return {
        "breakpoints": list(normalized_breakpoints),
        "segment_order": [
            descriptor["segment_key"] for descriptor in segment_descriptors
        ],
        "segments": segment_reports,
        "predicted_times": predicted_times,
        "aggregate_stats": _compute_prediction_stats(
            [float(row["measured_time"]) for row in rows],
            predicted_times,
        ),
    }


def fit_batch_perf_trace(
    trace_path: str | Path,
    *,
    model_name: str | None = None,
    tag: str | None = None,
    device_id: int | None = None,
    subtract_scheduling_overhead: bool = True,
    min_abs_num_reqs_coef: float = 1e-9,
    fit_by_category: bool = False,
    fit_piecewise_current_tokens: bool = False,
    piecewise_current_token_breakpoints: tuple[int, ...] | list[int] | None = None,
    viz: bool = False,
    plot_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    sample_data = collect_batch_perf_samples(
        trace_path,
        subtract_scheduling_overhead=subtract_scheduling_overhead,
        device_id=device_id,
    )
    rows = sample_data["rows"]
    batch_times = sample_data["batch_times"]
    if not batch_times:
        raise ValueError(f"No valid batch samples found in {trace_path}")

    fit_result = fit_linear_perf_model(
        batch_times,
        min_abs_num_reqs_coef=min_abs_num_reqs_coef,
    )

    trace_name = Path(trace_path).stem
    suffix = f"_device{device_id}" if device_id is not None else ""
    safe_name = sanitize_filename(f"{trace_name}{suffix}")

    report = {
        "trace_path": sample_data["trace_path"],
        "device_id": device_id,
        "loaded_event_count": sample_data["loaded_event_count"],
        "invalid_event_count": sample_data["invalid_event_count"],
        "used_sample_count": len(rows),
        "hardware_params": fit_result["hardware_params"],
        "fit_stats": fit_result["stats"],
        "fitted_estimator_stats": _compute_prediction_stats(
            fit_result["measured_times"],
            fit_result["predicted_times"],
        ),
        "existing_estimator_stats": _compute_prediction_stats(
            fit_result["measured_times"],
            [float(row["estimated_time"]) for row in rows],
        ),
        "sample_summary": _summarize_rows(rows),
    }

    if viz:
        resolved_plot_path = _derive_plot_path(
            plot_path,
            default_name=safe_name,
        )
        report["plot_path"] = str(
            save_prediction_scatter(
                resolved_plot_path,
                fit_result["measured_times"],
                fit_result["predicted_times"],
                title=f"{safe_name} batch fit",
            )
        )
        report["existing_by_category_plot_path"] = str(
            save_prediction_scatter_by_category(
                _derive_plot_path(
                    plot_path,
                    default_name=safe_name,
                    suffix="__existing_by_category",
                ),
                fit_result["measured_times"],
                [float(row["estimated_time"]) for row in rows],
                [str(row["batch_category"]) for row in rows],
                title=f"{safe_name} existing estimator by category",
            )
        )

    if fit_by_category:
        category_rows: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for idx, row in enumerate(rows):
            category_rows.setdefault(str(row["batch_category"]), []).append((idx, row))

        category_predicted_times: list[float] = [0.0] * len(rows)
        category_reports: dict[str, Any] = {}
        for category in BATCH_CATEGORY_ORDER:
            items = category_rows.get(category, [])
            if not items:
                continue
            category_batch_times = [
                (list(row["batch"]), float(row["measured_time"]))
                for _, row in items
            ]
            category_fit_result = fit_linear_perf_model(
                category_batch_times,
                min_abs_num_reqs_coef=min_abs_num_reqs_coef,
            )
            for (row_idx, _), predicted_time in zip(
                items,
                category_fit_result["predicted_times"],
            ):
                category_predicted_times[row_idx] = float(predicted_time)
            category_only_rows = [row for _, row in items]
            category_reports[category] = {
                "used_sample_count": len(items),
                "hardware_params": category_fit_result["hardware_params"],
                "fit_stats": category_fit_result["stats"],
                "fitted_estimator_stats": _compute_prediction_stats(
                    category_fit_result["measured_times"],
                    category_fit_result["predicted_times"],
                ),
                "existing_estimator_stats": _compute_prediction_stats(
                    category_fit_result["measured_times"],
                    [float(row["estimated_time"]) for row in category_only_rows],
                ),
                "sample_summary": _summarize_rows(category_only_rows),
            }

        report["category_fit"] = {
            "aggregate_stats": _compute_prediction_stats(
                fit_result["measured_times"],
                category_predicted_times,
            ),
            "categories": category_reports,
        }
        if viz:
            report["category_fit"]["plot_path"] = str(
                save_prediction_scatter_by_category(
                    _derive_plot_path(
                        plot_path,
                        default_name=safe_name,
                        suffix="__fit_by_category",
                    ),
                    fit_result["measured_times"],
                    category_predicted_times,
                    [str(row["batch_category"]) for row in rows],
                    title=f"{safe_name} category fits",
                )
            )

    if fit_piecewise_current_tokens:
        piecewise_fit = fit_piecewise_current_token_model(
            rows,
            breakpoints=piecewise_current_token_breakpoints,
            min_abs_num_reqs_coef=min_abs_num_reqs_coef,
        )
        report["piecewise_current_token_fit"] = {
            "breakpoints": piecewise_fit["breakpoints"],
            "segment_order": piecewise_fit["segment_order"],
            "aggregate_stats": piecewise_fit["aggregate_stats"],
            "segments": piecewise_fit["segments"],
        }
        if viz:
            report["piecewise_current_token_fit"]["plot_path"] = str(
                save_prediction_scatter_by_category(
                    _derive_plot_path(
                        plot_path,
                        default_name=safe_name,
                        suffix="__fit_piecewise_current_tokens",
                    ),
                    fit_result["measured_times"],
                    piecewise_fit["predicted_times"],
                    [str(row["batch_category"]) for row in rows],
                    title=f"{safe_name} piecewise current-token fits",
                )
            )

    if model_name is not None:
        resolved_tag = tag or safe_name
        upsert_hardware_params(model_name, resolved_tag, fit_result["hardware_params"])
        report["persisted_model_name"] = model_name
        report["persisted_tag"] = resolved_tag
        report["persisted_registry_path"] = str(PERF_MODEL_PATH)

    if report_path is not None:
        report["report_path"] = str(write_json(report_path, report))

    return report

def build_piecewise_current_token_hardware_params(
    segment_params: dict[str, list[float]] | list[list[float]],
    *,
    breakpoints: tuple[int, ...] | list[int] | None = None,
) -> dict[str, Any]:
    normalized_breakpoints = normalize_current_token_breakpoints(breakpoints)
    segment_order = [
        descriptor["segment_key"]
        for descriptor in iter_current_token_piece_segments(normalized_breakpoints)
    ]
    normalized_segment_params: dict[str, list[float]] = {}
    if isinstance(segment_params, dict):
        raw_segment_params = segment_params
    elif isinstance(segment_params, list):
        if len(segment_params) != len(segment_order):
            raise ValueError(
                "piecewise segment params must have one 5-term vector per segment"
            )
        raw_segment_params = {
            segment_key: raw_params
            for segment_key, raw_params in zip(segment_order, segment_params)
        }
    else:
        raise ValueError("piecewise segment params must be a dict or list")

    for segment_key in segment_order:
        normalized = _normalize_hw_param_list(raw_segment_params.get(segment_key))
        if normalized is None:
            raise ValueError(
                f"piecewise segment {segment_key!r} must be a length-5 numeric list"
            )
        normalized_segment_params[segment_key] = normalized

    return {
        "type": "piecewise_current_tokens",
        "breakpoints": list(normalized_breakpoints),
        "segment_params": normalized_segment_params,
    }


def _normalize_piecewise_hw_param_entry(params: Any) -> dict[str, Any] | None:
    if not isinstance(params, dict):
        return None
    if params.get("type") not in (None, "piecewise_current_tokens"):
        return None
    if "breakpoints" not in params or "segment_params" not in params:
        return None
    try:
        return build_piecewise_current_token_hardware_params(
            params["segment_params"],
            breakpoints=params["breakpoints"],
        )
    except ValueError:
        return None


def _normalize_hw_param_entry(params: Any) -> list[float] | dict[str, Any] | None:
    normalized_linear = _normalize_hw_param_list(params)
    if normalized_linear is not None:
        return normalized_linear
    return _normalize_piecewise_hw_param_entry(params)


class PerfModel:
    def __init__(self, model_name, hardware_params: list[float] | dict[str, Any]):
        self.model_name = model_name
        self.model_config = get_model_config(model_name)
        self._online_average_delay = 0.0
        self._online_spike_slack = 0.0
        self._decay_factor = 0.95
        self._update_cnt = 0
        self._model_type = "linear"
        self._linear_hardware_params: list[float] | None = None
        self._piecewise_breakpoints: tuple[int, ...] | None = None
        self._piecewise_segment_order: list[str] = []
        self._piecewise_segment_params: dict[str, list[float]] = {}
        self.hardware_params: list[float] | dict[str, Any]
        self._set_hardware_params(hardware_params)

    @property
    def is_piecewise_current_tokens(self) -> bool:
        return self._model_type == "piecewise_current_tokens"

    def _set_hardware_params(self, hardware_params: Any) -> None:
        normalized = _normalize_hw_param_entry(hardware_params)
        if normalized is None:
            raise ValueError(
                "hardware params must be either a length-5 numeric list or a "
                "piecewise_current_tokens payload"
            )
        if isinstance(normalized, list):
            self._model_type = "linear"
            self._linear_hardware_params = copy.deepcopy(normalized)
            self._piecewise_breakpoints = None
            self._piecewise_segment_order = []
            self._piecewise_segment_params = {}
            self.hardware_params = copy.deepcopy(normalized)
            return

        self._model_type = "piecewise_current_tokens"
        self._linear_hardware_params = None
        self._piecewise_breakpoints = normalize_current_token_breakpoints(
            normalized["breakpoints"]
        )
        self._piecewise_segment_order = [
            descriptor["segment_key"]
            for descriptor in iter_current_token_piece_segments(
                self._piecewise_breakpoints
            )
        ]
        self._piecewise_segment_params = {
            segment_key: copy.deepcopy(normalized["segment_params"][segment_key])
            for segment_key in self._piecewise_segment_order
        }
        self.hardware_params = build_piecewise_current_token_hardware_params(
            self._piecewise_segment_params,
            breakpoints=self._piecewise_breakpoints,
        )

    def _get_piecewise_descriptor(self, total_current_tokens: int) -> dict[str, Any]:
        if not self.is_piecewise_current_tokens:
            raise RuntimeError("piecewise descriptor requested for a linear perf model")
        return get_current_token_piece(
            total_current_tokens,
            self._piecewise_breakpoints,
        )

    def get_active_hardware_params(self, total_current_tokens: int) -> list[float]:
        if not self.is_piecewise_current_tokens:
            assert self._linear_hardware_params is not None
            return copy.deepcopy(self._linear_hardware_params)
        descriptor = self._get_piecewise_descriptor(total_current_tokens)
        return copy.deepcopy(
            self._piecewise_segment_params[descriptor["segment_key"]]
        )

    def describe_hardware_params(self) -> list[float] | dict[str, Any]:
        return copy.deepcopy(self.hardware_params)

    def get_cpp_planner_config(self) -> dict[str, Any]:
        if not self.is_piecewise_current_tokens:
            assert self._linear_hardware_params is not None
            return {
                "type": "linear",
                "hardware_params": copy.deepcopy(self._linear_hardware_params),
            }
        assert self._piecewise_breakpoints is not None
        return {
            "type": "piecewise_current_tokens",
            "breakpoints": list(self._piecewise_breakpoints),
            "segment_hardware_params": [
                copy.deepcopy(self._piecewise_segment_params[segment_key])
                for segment_key in self._piecewise_segment_order
            ],
        }

    def configure_cpp_ar_planner(
        self,
        scheduler: Any,
        *,
        tpots: list[float],
        fixed_bs: bool = False,
        max_bs: int = 16384,
    ) -> None:
        planner_config = self.get_cpp_planner_config()
        if planner_config["type"] == "piecewise_current_tokens":
            scheduler.set_ar_piecewise_planner(
                tpots=tpots,
                current_token_breakpoints=planner_config["breakpoints"],
                segment_hardware_params=planner_config["segment_hardware_params"],
                fixed_bs=fixed_bs,
                max_bs=max_bs,
            )
            return
        scheduler.set_ar_planner(
            tpots=tpots,
            hardware_params=planner_config["hardware_params"],
            fixed_bs=fixed_bs,
            max_bs=max_bs,
        )

    def _eval_params(
        self,
        params: list[float],
        *,
        total_current_tokens: int,
        num_reqs: int,
        num_past_tokens: int,
        num_decode_steps: int,
        include_online_slack: bool,
    ) -> float:
        return (
            params[0] * total_current_tokens
            + params[1] * num_reqs
            + params[2] * num_past_tokens
            + params[3] * num_decode_steps
            + params[4]
            + (self._online_spike_slack if include_online_slack else 0.0)
        )

    def get_batch_time_from_terms(
        self,
        total_current_tokens: int,
        *,
        num_reqs: int,
        num_past_tokens: int = 0,
        num_decode_steps: int = 1,
        include_online_slack: bool = True,
    ) -> float:
        params = self.get_active_hardware_params(total_current_tokens)
        return self._eval_params(
            params,
            total_current_tokens=total_current_tokens,
            num_reqs=num_reqs,
            num_past_tokens=num_past_tokens,
            num_decode_steps=num_decode_steps,
            include_online_slack=include_online_slack,
        )
    
    def get_batch_time(self, num_tokens: list[tuple[int, int]]) -> float:
        num_reqs = len(num_tokens)
        num_tot_tokens = sum([x[1] for x in num_tokens], start = 0)
        num_past_tokens = sum([x[0] for x in num_tokens], start = 0)
        num_decode_steps = 1
        return self.get_batch_time_from_terms(
            num_tot_tokens,
            num_reqs=num_reqs,
            num_past_tokens=num_past_tokens,
            num_decode_steps=num_decode_steps,
        )
    
    def update(self, batch: list[tuple[int, int]], elapsed: float):
        pass 
        return 
        estimated = self.get_batch_time(batch) - self._online_spike_slack
        self._online_average_delay = self._decay_factor * (self._online_average_delay) + \
            (1 - self._decay_factor) * (elapsed - estimated)
        self._online_spike_slack = max(self._online_average_delay, 0.0)
        if self._update_cnt % 100 == 0:
            logger.info(f'[PerfModel::Update]: {self._online_average_delay=}, {self._online_spike_slack=}')
        self._update_cnt += 1
        return 
    
    def get_bs(self, t: float, num_reqs: int, num_past_tokens: int = 0, num_decode_steps: int = 1) -> int:
        budget = float(t)
        if not self.is_piecewise_current_tokens:
            assert self._linear_hardware_params is not None
            return int((budget - self._linear_hardware_params[4] - self._online_spike_slack \
                - self._linear_hardware_params[3] * num_decode_steps - self._linear_hardware_params[2] * num_past_tokens\
                - self._linear_hardware_params[1] * num_reqs) / (self._linear_hardware_params[0]))

        assert self._piecewise_breakpoints is not None
        best_bs = 0
        for descriptor in iter_current_token_piece_segments(self._piecewise_breakpoints):
            params = self._piecewise_segment_params[descriptor["segment_key"]]
            lower_bound = descriptor["min_current_tokens"]
            upper_bound = descriptor["max_current_tokens"]
            residual_budget = (
                budget
                - self._online_spike_slack
                - params[4]
                - params[3] * num_decode_steps
                - params[2] * num_past_tokens
                - params[1] * num_reqs
            )
            if params[0] > 0.0:
                candidate = math.floor(residual_budget / params[0])
            elif residual_budget >= 0.0:
                candidate = upper_bound if upper_bound is not None else lower_bound
            else:
                continue
            if upper_bound is not None:
                candidate = min(candidate, upper_bound)
            if candidate < lower_bound:
                continue
            while candidate >= lower_bound and self.get_batch_time_from_terms(
                candidate,
                num_reqs=num_reqs,
                num_past_tokens=num_past_tokens,
                num_decode_steps=num_decode_steps,
            ) > budget + 1e-12:
                candidate -= 1
            if candidate >= lower_bound:
                best_bs = max(best_bs, candidate)
        return int(best_bs)
    
    def get_max_decode_batch_size(self, t: float, average_context_length: float = 0.0) -> int:
        budget = float(t)
        if not self.is_piecewise_current_tokens:
            assert self._linear_hardware_params is not None
            return int((budget - self._linear_hardware_params[4] - self._linear_hardware_params[3] - self._online_spike_slack) / (self._linear_hardware_params[0] + self._linear_hardware_params[1] + self._linear_hardware_params[2] * average_context_length))

        assert self._piecewise_breakpoints is not None
        best_bs = 0
        for descriptor in iter_current_token_piece_segments(self._piecewise_breakpoints):
            params = self._piecewise_segment_params[descriptor["segment_key"]]
            lower_bound = descriptor["min_current_tokens"]
            upper_bound = descriptor["max_current_tokens"]
            denominator = (
                params[0]
                + params[1]
                + params[2] * average_context_length
            )
            residual_budget = (
                budget
                - params[4]
                - params[3]
                - self._online_spike_slack
            )
            if denominator > 0.0:
                candidate = math.floor(residual_budget / denominator)
            elif residual_budget >= 0.0:
                candidate = upper_bound if upper_bound is not None else lower_bound
            else:
                continue
            if upper_bound is not None:
                candidate = min(candidate, upper_bound)
            if candidate < lower_bound:
                continue
            while candidate >= lower_bound and self.get_batch_time_from_terms(
                candidate,
                num_reqs=candidate,
                num_past_tokens=int(candidate * average_context_length),
                num_decode_steps=1,
            ) > budget + 1e-12:
                candidate -= 1
            if candidate >= lower_bound:
                best_bs = max(best_bs, candidate)
        return int(best_bs)

    def get_zero_load_prefill_affine_params(
        self,
        *,
        current_tokens_hint: int = 1,
    ) -> tuple[float, float]:
        params = self.get_active_hardware_params(current_tokens_hint)
        return params[0], params[1] + params[3] + params[4]

    def copy_with_adjustments(
        self,
        *,
        scale: float = 1.0,
        constant_offset: float = 0.0,
    ) -> 'PerfModel':
        adjusted = copy.deepcopy(self)
        if not adjusted.is_piecewise_current_tokens:
            assert adjusted._linear_hardware_params is not None
            if scale != 1.0:
                adjusted._linear_hardware_params = [
                    param * scale for param in adjusted._linear_hardware_params
                ]
            if constant_offset != 0.0:
                adjusted._linear_hardware_params[4] += constant_offset
            adjusted.hardware_params = copy.deepcopy(adjusted._linear_hardware_params)
            return adjusted

        for segment_key in adjusted._piecewise_segment_order:
            params = adjusted._piecewise_segment_params[segment_key]
            if scale != 1.0:
                params = [param * scale for param in params]
            if constant_offset != 0.0:
                params[4] += constant_offset
            adjusted._piecewise_segment_params[segment_key] = params
        adjusted.hardware_params = build_piecewise_current_token_hardware_params(
            adjusted._piecewise_segment_params,
            breakpoints=adjusted._piecewise_breakpoints,
        )
        return adjusted
    
    @staticmethod
    def get_perf_model(model_name: str, task: str = 'default') -> 'PerfModel':
        return PerfModel(model_name, get_hardware_params(model_name, task))
    
    def get_zero_load_ttft(self, input_length: int, cached_length: int = 0) -> float:
        return self.get_batch_time([(cached_length, input_length - cached_length)])

    def get_kv_mem_per_token(self):
        return self.model_config.get_token_cache_mem()

    def get_max_decode_length(self):
        return get_model_max_tokens(self.model_name)

    def fit(self,
            batch_times: list[tuple[list[tuple[int, int]], float]],
            tag: str, 
            viz=False,
            min_abs_num_reqs_coef: float = 1e-9):
        '''
        fit the linear regression model 
        @param batch_times list of batches: [[(past_len, currentlen)], measured_time]
        @param tag: the store prefix 
        @param viz: visualize or not 
        '''
        fit_result = fit_linear_perf_model(
            batch_times,
            min_abs_num_reqs_coef=min_abs_num_reqs_coef,
        )
        self._set_hardware_params(fit_result["hardware_params"])
        upsert_hardware_params(self.model_name, tag, self.hardware_params)

        if viz:
            safe_name = sanitize_filename(f"{self.model_name}__{tag}")
            plot_path = PERF_MODEL_FIG_DIR / f"{safe_name}.png"
            fit_result["plot_path"] = str(
                save_prediction_scatter(
                    plot_path,
                    fit_result["measured_times"],
                    fit_result["predicted_times"],
                    title=f"{get_easy_name(self.model_name)} [{tag}]",
                )
            )

        logger.info(
            "[PerfModel.fit] model=%s tag=%s params=%s stats=%s",
            self.model_name,
            tag,
            self.describe_hardware_params(),
            fit_result["stats"],
        )
        return fit_result

    def fit_trace(
        self,
        trace_path: str | Path,
        *,
        tag: str,
        device_id: int | None = None,
        subtract_scheduling_overhead: bool = True,
        viz: bool = False,
        min_abs_num_reqs_coef: float = 1e-9,
    ):
        sample_data = collect_batch_perf_samples(
            trace_path,
            subtract_scheduling_overhead=subtract_scheduling_overhead,
            device_id=device_id,
        )
        return self.fit(
            sample_data["batch_times"],
            tag=tag,
            viz=viz,
            min_abs_num_reqs_coef=min_abs_num_reqs_coef,
        )
        
DEFAULT_HW_PARAMS = {
    'default': [4.86e-5, 1.69e-5, 8e-8, 0, 1.4e-2],
    # 'Qwen/Qwen2.5-7B-Instruct': [4.86e-5, 3.7e-5, 5e-8, 0, 1.3e-2],
    'Qwen/Qwen2.5-7B-Instruct': {
        'default': [5.1e-05, 1.69e-5, 8e-8, 0, 1.4e-2],
        'sharegpt_code': [5.1e-05, 0.00, 8e-8, 0, 1.4e-02],
        'azure_chat_23': [5.55e-5, 0.00, 9e-8, 0, 1.4e-2],
        'arxiv_summary': [6.565e-05, 0.00, 8e-8, 0, 1.3e-02]}, # ChatBot
    'google/gemma-3-27b-it': {
        'default': [7.69e-5, 5.82e-5, 4.40e-8, 0, 1.9e-2],
        'azure_chat_23': [7.1e-5, 3.82e-5, 9.380e-8, 0, 1.8e-2],
        'azure_code_23': [3.62e-5, 3.31e-5, 9.7e-8, 0.0, 0.0176],
        'sharegpt_code': [3.62e-5, 3.31e-5, 9.7e-8, 0.0, 0.0176],
    },
    'meta-llama/Llama-3.1-70B': {
        'default': [6.2e-5, 3.7e-5, 5e-8, 0, 1.4e-2]
    },
}

HW_PARAMS = DEFAULT_HW_PARAMS


def _normalize_hw_param_list(params):
    if not isinstance(params, list) or len(params) != 5:
        return None
    try:
        return [float(param) for param in params]
    except (TypeError, ValueError):
        return None


def _load_hw_params() -> dict:
    registry = copy.deepcopy(DEFAULT_HW_PARAMS)
    if not PERF_MODEL_PATH.exists():
        return registry

    try:
        with PERF_MODEL_PATH.open("r", encoding="utf-8") as f:
            persisted = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load perf model file %s: %s", PERF_MODEL_PATH, exc)
        return registry

    if not isinstance(persisted, dict):
        return registry

    persisted_default = _normalize_hw_param_entry(persisted.get("default"))
    if persisted_default is not None:
        registry["default"] = persisted_default

    for model_name, task_params in persisted.items():
        if model_name == "default":
            continue
        normalized_default = _normalize_hw_param_entry(task_params)
        if normalized_default is not None:
            registry[model_name] = {"default": normalized_default}
            continue
        if not isinstance(task_params, dict):
            continue
        model_registry = copy.deepcopy(registry.get(model_name, {}))
        if not isinstance(model_registry, dict):
            model_registry = {}
        for task, params in task_params.items():
            normalized = _normalize_hw_param_entry(params)
            if normalized is not None:
                model_registry[task] = normalized
        if model_registry:
            registry[model_name] = model_registry
    return registry


def upsert_hardware_params(model_name: str, task: str, params: list[float] | dict[str, Any]) -> Path:
    normalized = _normalize_hw_param_entry(params)
    if normalized is None:
        raise ValueError(
            "hardware params must be either a length-5 numeric list or a "
            "piecewise_current_tokens payload"
        )

    registry = _load_hw_params()
    model_registry = copy.deepcopy(registry.get(model_name, {}))
    if not isinstance(model_registry, dict):
        model_registry = {}
    model_registry[task] = normalized
    if "default" not in model_registry and task == "default":
        model_registry["default"] = copy.deepcopy(normalized)
    elif "default" not in model_registry:
        existing_default = _normalize_hw_param_entry(
            DEFAULT_HW_PARAMS.get(model_name, {}).get("default")
            if isinstance(DEFAULT_HW_PARAMS.get(model_name), dict) else None
        )
        model_registry["default"] = existing_default or normalized
    registry[model_name] = model_registry
    return write_json(PERF_MODEL_PATH, registry)


def get_hardware_params(model_name, task):
    hw_params = _load_hw_params()
    if model_name not in hw_params:
        return copy.deepcopy(hw_params['default'])
    if task not in hw_params[model_name]:
        return copy.deepcopy(hw_params[model_name]['default'])
    return copy.deepcopy(hw_params[model_name][task])

get_easy_name = lambda model_name: {
    'Qwen/Qwen2.5-7B-Instruct': 'Qwen-7B', 
    'facebook/opt-125m': 'OPT-125M',
    'google/gemma-7b-it': 'Gemma-7B-IT',
    'google/gemma-3-27b-it': 'Gemma-3-27B-IT',
    'meta-llama/Llama-3.1-70B': 'Llama-70B',
}.get(model_name, model_name)

get_model_max_tokens = lambda model_name: {
    'Qwen/Qwen2.5-7B-Instruct': 24000,
    'facebook/opt-125m': 2048,
    'google/gemma-7b-it': 8192,
    'google/gemma-3-27b-it': 8192,
    'meta-llama/Llama-3.1-70B': 24000,
}.get(model_name, None)

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit the linear batch performance model against a batch event trace."
    )
    parser.add_argument(
        "trace_path",
        type=str,
        help="Path to a batch trace JSON array, JSONL file, or object with an events array.",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=None,
        help="Optional device id filter. Fits all devices by default.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional model registry key to persist fitted coefficients into assets/perf_model.json.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional task tag used when persisting into assets/perf_model.json.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Optional JSON output path for the fit summary.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default=None,
        help="Optional scatter plot path. If omitted, the plot goes under assets/perf_model_figs/.",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Write a measured-vs-predicted scatter plot.",
    )
    parser.add_argument(
        "--keep-scheduling-overhead",
        action="store_true",
        help="Fit against full elapsed time instead of subtracting scheduling_overhead.",
    )
    parser.add_argument(
        "--min-abs-num-reqs-coef",
        type=float,
        default=1e-9,
        help="Clamp the fitted num_reqs coefficient away from zero when its magnitude is too small.",
    )
    parser.add_argument(
        "--fit-by-category",
        action="store_true",
        help="Fit separate models for prefill, decode, and mixed batches.",
    )
    parser.add_argument(
        "--fit-piecewise-current-tokens",
        action="store_true",
        help="Fit separate linear models for current-token segments.",
    )
    parser.add_argument(
        "--piecewise-current-token-breakpoints",
        type=int,
        nargs="*",
        default=list(DEFAULT_CURRENT_TOKEN_BREAKPOINTS),
        help="Inclusive current-token breakpoints for piecewise fitting. Defaults to 512 2048.",
    )
    args = parser.parse_args()

    report = fit_batch_perf_trace(
        args.trace_path,
        model_name=args.model_name,
        tag=args.tag,
        device_id=args.device_id,
        subtract_scheduling_overhead=not args.keep_scheduling_overhead,
        min_abs_num_reqs_coef=args.min_abs_num_reqs_coef,
        fit_by_category=args.fit_by_category,
        fit_piecewise_current_tokens=args.fit_piecewise_current_tokens,
        piecewise_current_token_breakpoints=args.piecewise_current_token_breakpoints,
        viz=args.viz,
        plot_path=args.plot_path,
        report_path=args.report_path,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
