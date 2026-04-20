from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from plots.common import (
        get_method_label as _common_get_method_label,
        get_method_style as _common_get_method_style,
    )
except ModuleNotFoundError:
    from common import (
        get_method_label as _common_get_method_label,
        get_method_style as _common_get_method_style,
    )


DEFAULT_OUTPUT_STEM = Path("figs/energy_per_duty_cycle")
DEFAULT_WINDOW_SIZE_S = 1.0
IDLE_POWER_W = 70.0
REPO_PATH_MARKER = "SLOServe/"

STATE_POWER_COEFFS = {
    "decode_active": 89.341,
    "mixed_active": 58.585,
    "prefill_active": 376.009,
    "decode_present": 130.004,
    "mixed_present": 77.342,
    "prefill_present": 0.0,
}

LOCAL_LABEL_MAP = {
    "atfc / slosserve_disagg_planner_oracle_mem": "SLO Packer (Disagg)",
    "atfc / slosserve_disagg_planner": "SLO Packer (Disagg)",
    "atfc / slosserve_planner_oracle_mem": "SLO Packer (Oracle)",
    "qlm / round_robin-disagg": "vLLM+ (Disagg)",
    "vllm / round_robin": "Colocated",
}


@dataclass(frozen=True)
class RunMetadata:
    event_file: Path
    method: str
    rps: float | None = None
    load_scale: float | None = None
    n_device: int | None = None
    source: Path | None = None


@dataclass(frozen=True)
class RunPoint:
    method: str
    label: str
    event_file: Path
    duty_cycle_pct: float
    duty_cycle_union_pct: float
    energy_j: float
    energy_per_duty_pct: float
    energy_per_duty_fraction: float
    energy_source: str
    energy_scope: str
    rps: float | None
    load_scale: float | None
    n_device: int
    analysis_device_index: int | None
    prefill_devices: tuple[int, ...]
    observation_window: str
    observation_start_s: float
    observation_end_s: float
    run_duration_s: float
    prefill_busy_s: float


def _get_plotting_dependencies():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to render this figure. Install it first, "
            "then rerun the script."
        ) from exc
    return plt


def _apply_paper_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.titlesize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _method_label(method: str) -> str:
    return LOCAL_LABEL_MAP.get(method, _common_get_method_label(method))


def _method_style(method: str) -> dict[str, object]:
    style = dict(_common_get_method_style(method))
    style.setdefault("linestyle", "-")
    style["linewidth"] = max(2.4, float(style.get("linewidth", 0.0)))
    style["markersize"] = max(7.5, float(style.get("markersize", 0.0)))
    return style


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _as_int(value: Any) -> int | None:
    parsed = _as_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _load_json_events(event_file: Path) -> list[dict[str, Any]]:
    text = event_file.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped.startswith("["):
        events = json.loads(text)
        if not isinstance(events, list):
            raise ValueError(f"{event_file} must contain a JSON list of events.")
        return [event for event in events if isinstance(event, dict)]

    events: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"{event_file}:{lineno} is not a JSON object.")
        events.append(parsed)
    return events


def _load_result_rows(results_jsonl: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with results_jsonl.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{results_jsonl}:{lineno} is not a JSON object.")
            rows.append(row)
    return rows


def _truncate_to_repo_relative_path(raw_path: str) -> Path | None:
    normalized = raw_path.replace("\\", "/")
    if REPO_PATH_MARKER not in normalized:
        return None
    return Path(normalized.split(REPO_PATH_MARKER, maxsplit=1)[1])


def _resolve_event_file(raw_path: str, *, results_jsonl: Path | None = None) -> Path:
    repo_relative_path = _truncate_to_repo_relative_path(raw_path)
    event_file = Path(raw_path).expanduser()
    candidates: list[Path] = []
    if repo_relative_path is not None:
        candidates.append(repo_relative_path)
    candidates.append(event_file)
    if results_jsonl is not None and not event_file.is_absolute():
        candidates.append((results_jsonl.parent / event_file).expanduser())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return repo_relative_path if repo_relative_path is not None else event_file


def _method_from_row(row: dict[str, Any]) -> str:
    scheduling_policy = row.get("scheduling_policy") or row.get("base_scheduling_policy")
    routing_policy = row.get("routing_policy") or row.get("base_routing_policy")
    if scheduling_policy and routing_policy:
        return f"{scheduling_policy} / {routing_policy}"
    return ""


def _infer_method_from_event_file(event_file: Path) -> str:
    name = event_file.name
    if name.startswith("atfc_slosserve_disagg") or "slosserve_disagg" in name:
        return "atfc / slosserve_disagg_planner"
    if name.startswith("atfc_slosserve_planner"):
        return "atfc / slosserve_planner"
    if name.startswith("atfc_round_robin"):
        return "atfc / round_robin"
    if name.startswith("qlm_round_robin-disagg") or "round_robin-disagg" in name:
        return "qlm / round_robin-disagg"
    if name.startswith("qlm_round_robin"):
        return "qlm / round_robin"
    if name.startswith("vllm_round_robin"):
        return "vllm / round_robin"
    return event_file.stem


def _infer_rps_from_event_file(event_file: Path) -> float | None:
    match = re.search(r"_arrival_([0-9]+(?:\.[0-9]+)?)_", event_file.name)
    if match:
        return _as_float(match.group(1))
    return None


def _metadata_from_results(
    results_jsonl: Path,
    *,
    include_incomplete: bool,
) -> list[RunMetadata]:
    metadata: list[RunMetadata] = []
    for row in _load_result_rows(results_jsonl):
        if not include_incomplete and row.get("run_status") not in (None, "", "completed"):
            continue
        raw_event_file = row.get("event_file")
        if not raw_event_file:
            continue
        event_file = _resolve_event_file(str(raw_event_file), results_jsonl=results_jsonl)
        method = _method_from_row(row) or _infer_method_from_event_file(event_file)
        metadata.append(
            RunMetadata(
                event_file=event_file,
                method=method,
                rps=_as_float(row.get("rps")),
                load_scale=_as_float(row.get("load_scale")),
                n_device=(
                    _as_int(row.get("effective_n_device"))
                    or _as_int(row.get("n_device"))
                    or _as_int(row.get("total_gpus"))
                ),
                source=results_jsonl,
            )
        )
    return metadata


def _metadata_from_event_file(event_file: Path) -> RunMetadata:
    event_file = event_file.expanduser().resolve()
    return RunMetadata(
        event_file=event_file,
        method=_infer_method_from_event_file(event_file),
        rps=_infer_rps_from_event_file(event_file),
    )


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def _collect_metadata(args: argparse.Namespace) -> list[RunMetadata]:
    metadata_by_event_file: OrderedDict[Path, RunMetadata] = OrderedDict()

    for pattern in args.results_glob:
        for results_jsonl in _expand_paths([pattern]):
            for metadata in _metadata_from_results(
                results_jsonl,
                include_incomplete=args.include_incomplete,
            ):
                metadata_by_event_file[metadata.event_file] = metadata

    for results_jsonl in args.results_jsonl:
        for metadata in _metadata_from_results(
            Path(results_jsonl),
            include_incomplete=args.include_incomplete,
        ):
            metadata_by_event_file[metadata.event_file] = metadata

    for pattern in args.event_glob:
        for event_file in _expand_paths([pattern]):
            metadata = _metadata_from_event_file(event_file)
            metadata_by_event_file.setdefault(metadata.event_file, metadata)

    for event_file in args.events:
        metadata = _metadata_from_event_file(Path(event_file))
        metadata_by_event_file.setdefault(metadata.event_file, metadata)

    return list(metadata_by_event_file.values())


def _scheduled_values(event: dict[str, Any]) -> list[int]:
    raw = event.get("num_scheduled_tokens")
    values: list[Any]
    if isinstance(raw, dict):
        values = list(raw.values())
    elif isinstance(raw, list):
        values = raw
    elif raw is None:
        values = []
    else:
        values = [raw]
    parsed: list[int] = []
    for value in values:
        parsed_value = _as_int(value)
        if parsed_value is not None:
            parsed.append(parsed_value)
    return parsed


def _computed_values(event: dict[str, Any]) -> list[int]:
    raw = event.get("num_computed_tokens")
    if raw is None:
        return []
    values = raw if isinstance(raw, list) else [raw]
    parsed: list[int] = []
    for value in values:
        parsed_value = _as_int(value)
        if parsed_value is not None:
            parsed.append(parsed_value)
    return parsed


def _batch_interval(event: dict[str, Any]) -> tuple[float, float] | None:
    end = _as_float(event.get("timestamp"))
    elapsed = _as_float(event.get("elapsed"))
    if end is None or elapsed is None:
        return None
    start = end - max(0.0, elapsed)
    if end <= start:
        return None
    return start, end


def _batch_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event.get("event_type") == "batch" and _batch_interval(event) is not None
    ]


def _batch_time_bounds(
    batch_events: list[dict[str, Any]],
) -> tuple[float, float] | None:
    starts_ends = [_batch_interval(event) for event in batch_events]
    starts_ends = [interval for interval in starts_ends if interval is not None]
    if not starts_ends:
        return None
    return min(start for start, _ in starts_ends), max(end for _, end in starts_ends)


def _event_time_bounds(
    events: list[dict[str, Any]],
    event_type: str,
) -> tuple[float, float] | None:
    timestamps = [
        timestamp
        for event in events
        if event.get("event_type") == event_type
        for timestamp in [_as_float(event.get("timestamp"))]
        if timestamp is not None
    ]
    if len(timestamps) < 2:
        return None
    start_time = min(timestamps)
    end_time = max(timestamps)
    if end_time <= start_time:
        return None
    return start_time, end_time


def _select_observation_bounds(
    events: list[dict[str, Any]],
    batch_events: list[dict[str, Any]],
    mode: str,
) -> tuple[str, float, float]:
    candidates: list[tuple[str, tuple[float, float] | None]]
    if mode == "auto":
        candidates = [
            ("global_arrival", _event_time_bounds(events, "global_arrival")),
            ("arrival", _event_time_bounds(events, "arrival")),
            ("batch", _batch_time_bounds(batch_events)),
        ]
    elif mode in {"global_arrival", "arrival"}:
        candidates = [
            (mode, _event_time_bounds(events, mode)),
            ("batch", _batch_time_bounds(batch_events)),
        ]
    elif mode == "batch":
        candidates = [("batch", _batch_time_bounds(batch_events))]
    else:
        raise ValueError(f"Unsupported observation window: {mode}")

    for selected_mode, bounds in candidates:
        if bounds is not None:
            start_time, end_time = bounds
            return selected_mode, start_time, end_time
    raise ValueError("Unable to infer an observation window from the event file.")


def _batch_kind_for_power(event: dict[str, Any]) -> str:
    scheduled = _scheduled_values(event)
    if scheduled and max(scheduled) == 1:
        return "decode"
    if scheduled and min(scheduled) > 1:
        return "prefill"
    return "mixed"


def _batch_has_prefill_work(event: dict[str, Any]) -> bool:
    scheduled = _scheduled_values(event)
    computed = _computed_values(event)
    return any(value > 1 for value in scheduled) or any(value == 0 for value in computed)


def _num_requests(event: dict[str, Any]) -> int:
    req_ids = event.get("req_ids")
    if isinstance(req_ids, list):
        return len(req_ids)
    scheduled = _scheduled_values(event)
    return len(scheduled)


def _infer_n_device(
    batch_events: list[dict[str, Any]],
    metadata_n_device: int | None,
) -> int:
    max_device_id = -1
    for event in batch_events:
        device_id = _as_int(event.get("device_id"))
        if device_id is not None:
            max_device_id = max(max_device_id, device_id)
    inferred = max_device_id + 1 if max_device_id >= 0 else 0
    if metadata_n_device is not None and metadata_n_device > 0:
        return max(metadata_n_device, inferred)
    return inferred


def _make_time_bins(start_time: float, end_time: float, window_size: float) -> list[float]:
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if end_time <= start_time:
        end_time = start_time + window_size
    n_windows = max(1, int(math.ceil((end_time - start_time) / window_size)))
    return [start_time + idx * window_size for idx in range(n_windows + 1)]


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _compute_prefill_duty(
    batch_events: list[dict[str, Any]],
    *,
    start_time: float,
    end_time: float,
    device_index: int | None,
) -> tuple[float, float, float, tuple[int, ...], float]:
    run_duration_s = max(0.0, end_time - start_time)
    if run_duration_s <= 0.0:
        return 0.0, 0.0, 0.0, tuple(), 0.0

    intervals_by_device: dict[int, list[tuple[float, float]]] = defaultdict(list)
    all_prefill_intervals: list[tuple[float, float]] = []
    for event in batch_events:
        if not _batch_has_prefill_work(event):
            continue
        device_id = _as_int(event.get("device_id"))
        interval = _batch_interval(event)
        if device_id is None or interval is None:
            continue
        if device_index is not None and device_id != device_index:
            continue
        clipped_start = max(start_time, interval[0])
        clipped_end = min(end_time, interval[1])
        if clipped_end <= clipped_start:
            continue
        clipped_interval = (clipped_start, clipped_end)
        intervals_by_device[device_id].append(clipped_interval)
        all_prefill_intervals.append(clipped_interval)

    prefill_devices = tuple(sorted(intervals_by_device))
    if not prefill_devices:
        return run_duration_s, 0.0, 0.0, tuple(), 0.0

    busy_seconds = 0.0
    for device_id in prefill_devices:
        busy_seconds += sum(
            end - start for start, end in _merge_intervals(intervals_by_device[device_id])
        )

    union_busy_seconds = sum(
        end - start for start, end in _merge_intervals(all_prefill_intervals)
    )
    if device_index is None:
        duty_cycle_pct = 100.0 * busy_seconds / (run_duration_s * len(prefill_devices))
        duty_cycle_union_pct = 100.0 * union_busy_seconds / run_duration_s
    else:
        duty_cycle_pct = 100.0 * busy_seconds / run_duration_s
        duty_cycle_union_pct = duty_cycle_pct
    return (
        run_duration_s,
        min(100.0, duty_cycle_pct),
        min(100.0, duty_cycle_union_pct),
        prefill_devices,
        busy_seconds,
    )


def _compute_measured_energy(
    events: list[dict[str, Any]],
    *,
    start_time: float,
    end_time: float,
) -> tuple[float, dict[int, float], bool]:
    total_energy = 0.0
    per_device_energy: dict[int, float] = defaultdict(float)
    found = False

    for event in events:
        if event.get("event_type") != "energy":
            continue
        timestamp = _as_float(event.get("timestamp"))
        if timestamp is None or timestamp < start_time or timestamp > end_time:
            continue

        gpu_energy = event.get("gpu_energy")
        if isinstance(gpu_energy, list):
            parsed_gpu_energy = [
                float(value)
                for value in gpu_energy
                if _as_float(value) is not None
            ]
            if parsed_gpu_energy:
                found = True
                energy = sum(parsed_gpu_energy)
                total_energy += energy
                for device_id, device_energy in enumerate(parsed_gpu_energy):
                    per_device_energy[device_id] += device_energy
                continue

        energy = _as_float(event.get("energy"))
        if energy is None:
            continue
        found = True
        total_energy += energy
        device_id = _as_int(event.get("device_id"))
        if device_id is not None:
            per_device_energy[device_id] += energy

    return total_energy, dict(per_device_energy), found


def _compute_state_based_energy(
    batch_events: list[dict[str, Any]],
    *,
    n_device: int,
    start_time: float,
    end_time: float,
    window_size: float,
) -> tuple[float, dict[int, float]]:
    if not batch_events or n_device <= 0:
        return 0.0, {}

    device_ids = list(range(n_device))
    bins = _make_time_bins(start_time, end_time, window_size)
    n_windows = len(bins) - 1
    per_device_power = {
        device_id: [IDLE_POWER_W for _ in range(n_windows)]
        for device_id in device_ids
    }
    feature_names = (
        "decode_active",
        "mixed_active",
        "prefill_active",
        "decode_present",
        "mixed_present",
        "prefill_present",
    )
    state_features = {
        device_id: {name: [0.0 for _ in range(n_windows)] for name in feature_names}
        for device_id in device_ids
    }
    active_feature_by_kind = {
        "decode": "decode_active",
        "mixed": "mixed_active",
        "prefill": "prefill_active",
    }

    for event in batch_events:
        device_id = _as_int(event.get("device_id"))
        interval = _batch_interval(event)
        if device_id is None or interval is None or device_id not in state_features:
            continue
        batch_start = max(start_time, interval[0])
        batch_end = min(end_time, interval[1])
        if batch_end <= batch_start:
            continue
        kind = _batch_kind_for_power(event)
        nreq = float(_num_requests(event))
        left = max(0, int(math.floor((batch_start - start_time) / window_size)))
        right = min(n_windows, int(math.ceil((batch_end - start_time) / window_size)))
        for idx in range(left, right):
            win_start = bins[idx]
            win_end = bins[idx + 1]
            overlap = max(0.0, min(batch_end, win_end) - max(batch_start, win_start))
            if overlap <= 0.0:
                continue
            frac = overlap / window_size
            features = state_features[device_id]
            features[active_feature_by_kind[kind]][idx] += nreq * frac
            features[f"{kind}_present"][idx] = 1.0

    for device_id in device_ids:
        for feature_name, coeff in STATE_POWER_COEFFS.items():
            feature_values = state_features[device_id][feature_name]
            for idx, value in enumerate(feature_values):
                per_device_power[device_id][idx] += value * coeff

    per_device_energy = {
        device_id: sum(power_values) * window_size
        for device_id, power_values in per_device_power.items()
    }
    return sum(per_device_energy.values()), per_device_energy


def _compute_run_point(
    metadata: RunMetadata,
    *,
    window_size: float,
    observation_window: str,
    device_index: int | None,
) -> RunPoint | None:
    if not metadata.event_file.exists():
        raise FileNotFoundError(f"Missing event file: {metadata.event_file}")

    events = _load_json_events(metadata.event_file)
    batches = _batch_events(events)
    if not batches:
        return None

    selected_window, start_time, end_time = _select_observation_bounds(
        events,
        batches,
        observation_window,
    )
    run_duration_s, duty_pct, union_duty_pct, prefill_devices, prefill_busy_s = (
        _compute_prefill_duty(
            batches,
            start_time=start_time,
            end_time=end_time,
            device_index=device_index,
        )
    )
    n_device = _infer_n_device(batches, metadata.n_device)
    if device_index is not None and (device_index < 0 or device_index >= max(n_device, 1)):
        return None

    measured_energy_j, measured_per_device_energy, has_measured_energy = _compute_measured_energy(
        events,
        start_time=start_time,
        end_time=end_time,
    )

    if device_index is None:
        energy_j = measured_energy_j
        energy_source = "measured"
        energy_scope = "total"
        if not has_measured_energy or energy_j <= 0.0:
            energy_j, _state_per_device_energy = _compute_state_based_energy(
                batches,
                n_device=n_device,
                start_time=start_time,
                end_time=end_time,
                window_size=window_size,
            )
            energy_source = "state_based"
    else:
        energy_j = measured_per_device_energy.get(device_index, 0.0)
        energy_source = "measured_device"
        energy_scope = f"device-{device_index}"
        if not has_measured_energy or energy_j <= 0.0:
            _state_energy_j, state_per_device_energy = _compute_state_based_energy(
                batches,
                n_device=n_device,
                start_time=start_time,
                end_time=end_time,
                window_size=window_size,
            )
            energy_j = state_per_device_energy.get(device_index, 0.0)
            energy_source = "state_based_device"

    if duty_pct <= 0.0 or energy_j <= 0.0:
        return None

    method = metadata.method
    return RunPoint(
        method=method,
        label=_method_label(method),
        event_file=metadata.event_file,
        duty_cycle_pct=duty_pct,
        duty_cycle_union_pct=union_duty_pct,
        energy_j=energy_j,
        energy_per_duty_pct=energy_j / duty_pct,
        energy_per_duty_fraction=energy_j / (duty_pct / 100.0),
        energy_source=energy_source,
        energy_scope=energy_scope,
        rps=metadata.rps,
        load_scale=metadata.load_scale,
        n_device=n_device,
        analysis_device_index=device_index,
        prefill_devices=prefill_devices,
        observation_window=selected_window,
        observation_start_s=start_time,
        observation_end_s=end_time,
        run_duration_s=run_duration_s,
        prefill_busy_s=prefill_busy_s,
    )


def _write_csv(points: list[RunPoint], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "label",
        "event_file",
        "rps",
        "load_scale",
        "duty_cycle_pct",
        "duty_cycle_union_pct",
        "energy_j",
        "energy_per_duty_pct",
        "energy_per_duty_fraction",
        "energy_source",
        "energy_scope",
        "n_device",
        "analysis_device_index",
        "prefill_devices",
        "observation_window",
        "observation_start_s",
        "observation_end_s",
        "run_duration_s",
        "prefill_busy_s",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "method": point.method,
                    "label": point.label,
                    "event_file": str(point.event_file),
                    "rps": "" if point.rps is None else point.rps,
                    "load_scale": "" if point.load_scale is None else point.load_scale,
                    "duty_cycle_pct": point.duty_cycle_pct,
                    "duty_cycle_union_pct": point.duty_cycle_union_pct,
                    "energy_j": point.energy_j,
                    "energy_per_duty_pct": point.energy_per_duty_pct,
                    "energy_per_duty_fraction": point.energy_per_duty_fraction,
                    "energy_source": point.energy_source,
                    "energy_scope": point.energy_scope,
                    "n_device": point.n_device,
                    "analysis_device_index": (
                        "" if point.analysis_device_index is None else point.analysis_device_index
                    ),
                    "prefill_devices": " ".join(str(device_id) for device_id in point.prefill_devices),
                    "observation_window": point.observation_window,
                    "observation_start_s": point.observation_start_s,
                    "observation_end_s": point.observation_end_s,
                    "run_duration_s": point.run_duration_s,
                    "prefill_busy_s": point.prefill_busy_s,
                }
            )


def _plot(
    points: list[RunPoint],
    output_stem: Path,
    *,
    device_index: int | None,
) -> None:
    plt = _get_plotting_dependencies()
    _apply_paper_style(plt)

    fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    series_by_method: OrderedDict[str, list[RunPoint]] = OrderedDict()
    for point in points:
        series_by_method.setdefault(point.method, []).append(point)

    for method, series in series_by_method.items():
        series.sort(
            key=lambda point: (
                point.duty_cycle_pct,
                -1.0 if point.rps is None else point.rps,
                str(point.event_file),
            )
        )
        style = _method_style(method)
        xs = [point.duty_cycle_pct for point in series]
        ys = [point.energy_per_duty_pct for point in series]
        ax.plot(
            xs,
            ys,
            label=_method_label(method),
            color=style.get("color"),
            marker=style.get("marker"),
            linestyle=style.get("linestyle", "-"),
            linewidth=style.get("linewidth", 2.4),
            markersize=style.get("markersize", 7.5),
        )

    if device_index is None:
        ax.set_xlabel("Prefill Engine Duty Cycle (%)")
        ax.set_ylabel("Energy / Duty Cycle (J / %-point)")
    else:
        ax.set_xlabel(f"Device {device_index} Prefill Duty Cycle (%)")
        ax.set_ylabel(f"Device {device_index} Energy / Duty Cycle (J / %-point)")
    ax.set_facecolor("white")
    ax.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.9,
        linestyle=(0, (2, 2)),
    )
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)
    ax.tick_params(axis="both", colors="black", pad=2)
    ax.legend(frameon=False)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot energy consumed divided by prefill-engine duty cycle. "
            "Duty cycle is computed from batch intervals in event files."
        )
    )
    parser.add_argument(
        "events",
        nargs="*",
        help="Event files to include. JSON arrays and JSONL event files are both accepted.",
    )
    parser.add_argument(
        "--event-glob",
        action="append",
        default=[],
        help="Glob for event files. May be passed multiple times.",
    )
    parser.add_argument(
        "--results-jsonl",
        action="append",
        default=[],
        help="results.jsonl file whose rows contain event_file/rps/method metadata.",
    )
    parser.add_argument(
        "--results-glob",
        action="append",
        default=[],
        help="Glob for results.jsonl files. May be passed multiple times.",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include result rows whose run_status is not completed.",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=DEFAULT_WINDOW_SIZE_S,
        help="Window size in seconds for the state-based energy model.",
    )
    parser.add_argument(
        "--observation-window",
        choices=("auto", "global_arrival", "arrival", "batch"),
        default="auto",
        help=(
            "Time interval used for duty cycle and energy. auto uses global_arrival "
            "when available, then arrival, then batch service time."
        ),
    )
    parser.add_argument(
        "--device-index",
        "--device-id",
        dest="device_index",
        type=int,
        default=None,
        help=(
            "Restrict the analysis to one device index. Duty cycle uses only that "
            "device's prefill intervals, and energy uses that device's measured or "
            "state-modeled energy."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_STEM,
        help="Output stem. The script writes .png, .pdf, and .csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.device_index is not None and args.device_index < 0:
        raise SystemExit("--device-index must be non-negative.")

    metadata = _collect_metadata(args)
    if not metadata:
        raise SystemExit(
            "No inputs found. Pass event files, --event-glob, --results-jsonl, or --results-glob."
        )

    points: list[RunPoint] = []
    skipped = 0
    for item in metadata:
        point = _compute_run_point(
            item,
            window_size=args.window_size,
            observation_window=args.observation_window,
            device_index=args.device_index,
        )
        if point is None:
            skipped += 1
            continue
        points.append(point)

    if not points:
        raise SystemExit("No usable runs found: all inputs lacked batch events or prefill duty.")

    points.sort(key=lambda point: (point.method, point.duty_cycle_pct, str(point.event_file)))
    _write_csv(points, args.output.with_suffix(".csv"))
    _plot(points, args.output, device_index=args.device_index)

    print(f"Wrote {args.output.with_suffix('.png')}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    print(f"Wrote {args.output.with_suffix('.csv')}")
    print(f"Plotted {len(points)} runs; skipped {skipped} inputs without usable duty data.")


if __name__ == "__main__":
    main()
