from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from plots.energy_per_duty_cycle import (
        IDLE_POWER_W,
        REPO_PATH_MARKER,
        STATE_POWER_COEFFS,
        _as_float,
        _as_int,
        _batch_events,
        _batch_interval,
        _batch_kind_for_power,
        _infer_method_from_event_file,
        _infer_n_device,
        _load_json_events,
        _load_result_rows,
        _method_from_row,
        _num_requests,
        _resolve_event_file,
        _select_observation_bounds,
        _truncate_to_repo_relative_path,
    )
except ModuleNotFoundError:
    from energy_per_duty_cycle import (
        IDLE_POWER_W,
        REPO_PATH_MARKER,
        STATE_POWER_COEFFS,
        _as_float,
        _as_int,
        _batch_events,
        _batch_interval,
        _batch_kind_for_power,
        _infer_method_from_event_file,
        _infer_n_device,
        _load_json_events,
        _load_result_rows,
        _method_from_row,
        _num_requests,
        _resolve_event_file,
        _select_observation_bounds,
        _truncate_to_repo_relative_path,
    )


DEFAULT_OUTPUT_STEM = Path("figs/device_active_idle_power")


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
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _repo_relative_path(path: Path) -> str:
    raw = str(path)
    repo_relative = _truncate_to_repo_relative_path(raw)
    if repo_relative is not None:
        return str(repo_relative)
    return raw


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


def _interval_contains(intervals: list[tuple[float, float]], timestamp: float) -> bool:
    return any(start <= timestamp <= end for start, end in intervals)


def _build_active_intervals(
    batch_events: list[dict[str, Any]],
    *,
    n_device: int,
    start_time: float,
    end_time: float,
) -> tuple[dict[int, list[tuple[float, float]]], dict[int, float], dict[int, int]]:
    intervals_by_device: dict[int, list[tuple[float, float]]] = {
        device_id: [] for device_id in range(n_device)
    }
    batch_elapsed_sum = {device_id: 0.0 for device_id in range(n_device)}
    batch_count = {device_id: 0 for device_id in range(n_device)}

    for event in batch_events:
        device_id = _as_int(event.get("device_id"))
        interval = _batch_interval(event)
        if device_id is None or interval is None or device_id not in intervals_by_device:
            continue
        clipped_start = max(start_time, interval[0])
        clipped_end = min(end_time, interval[1])
        if clipped_end <= clipped_start:
            continue
        intervals_by_device[device_id].append((clipped_start, clipped_end))
        batch_elapsed_sum[device_id] += clipped_end - clipped_start
        batch_count[device_id] += 1

    merged_by_device = {
        device_id: _merge_intervals(intervals)
        for device_id, intervals in intervals_by_device.items()
    }
    return merged_by_device, batch_elapsed_sum, batch_count


def _compute_state_interval_energy(
    batch_events: list[dict[str, Any]],
    *,
    n_device: int,
    start_time: float,
    end_time: float,
    active_intervals_by_device: dict[int, list[tuple[float, float]]],
) -> dict[int, dict[str, float | str]]:
    duration = max(0.0, end_time - start_time)
    rows: dict[int, dict[str, float | str]] = {}
    for device_id in range(n_device):
        active_time = sum(
            interval_end - interval_start
            for interval_start, interval_end in active_intervals_by_device[device_id]
        )
        idle_time = max(0.0, duration - active_time)
        rows[device_id] = {
            "active_time_s": active_time,
            "idle_time_s": idle_time,
            "active_energy_j": IDLE_POWER_W * active_time,
            "idle_energy_j": IDLE_POWER_W * idle_time,
            "power_source": "state_based_interval",
        }

    present_coeff_by_kind = {
        "decode": STATE_POWER_COEFFS["decode_present"],
        "mixed": STATE_POWER_COEFFS["mixed_present"],
        "prefill": STATE_POWER_COEFFS["prefill_present"],
    }
    active_coeff_by_kind = {
        "decode": STATE_POWER_COEFFS["decode_active"],
        "mixed": STATE_POWER_COEFFS["mixed_active"],
        "prefill": STATE_POWER_COEFFS["prefill_active"],
    }

    for event in batch_events:
        device_id = _as_int(event.get("device_id"))
        interval = _batch_interval(event)
        if device_id is None or interval is None or device_id not in rows:
            continue
        clipped_start = max(start_time, interval[0])
        clipped_end = min(end_time, interval[1])
        if clipped_end <= clipped_start:
            continue

        elapsed = clipped_end - clipped_start
        kind = _batch_kind_for_power(event)
        nreq = float(_num_requests(event))
        extra_power_w = active_coeff_by_kind[kind] * nreq + present_coeff_by_kind[kind]
        rows[device_id]["active_energy_j"] = (
            float(rows[device_id]["active_energy_j"]) + extra_power_w * elapsed
        )

    for device_id, row in rows.items():
        active_time = float(row["active_time_s"])
        idle_time = float(row["idle_time_s"])
        row["active_power_w"] = (
            float(row["active_energy_j"]) / active_time if active_time > 0.0 else math.nan
        )
        row["idle_power_w"] = (
            float(row["idle_energy_j"]) / idle_time if idle_time > 0.0 else math.nan
        )
    return rows


def _override_with_measured_power_samples(
    rows: dict[int, dict[str, float | str]],
    events: list[dict[str, Any]],
    *,
    active_intervals_by_device: dict[int, list[tuple[float, float]]],
    start_time: float,
    end_time: float,
) -> None:
    samples: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"active": [], "idle": []}
    )

    for event in events:
        if event.get("event_type") != "energy":
            continue
        timestamp = _as_float(event.get("timestamp"))
        if timestamp is None or timestamp < start_time or timestamp > end_time:
            continue

        gpu_power = event.get("gpu_power")
        if isinstance(gpu_power, list):
            for device_id, power in enumerate(gpu_power):
                parsed_power = _as_float(power)
                if parsed_power is None or device_id not in rows:
                    continue
                state = (
                    "active"
                    if _interval_contains(active_intervals_by_device[device_id], timestamp)
                    else "idle"
                )
                samples[device_id][state].append(parsed_power)
            continue

        device_id = _as_int(event.get("device_id"))
        power = _as_float(event.get("power"))
        if device_id is None or power is None or device_id not in rows:
            continue
        state = (
            "active"
            if _interval_contains(active_intervals_by_device[device_id], timestamp)
            else "idle"
        )
        samples[device_id][state].append(power)

    for device_id, state_samples in samples.items():
        row = rows[device_id]
        active_samples = state_samples["active"]
        idle_samples = state_samples["idle"]
        if active_samples:
            active_power = sum(active_samples) / len(active_samples)
            row["active_power_w"] = active_power
            row["active_energy_j"] = active_power * float(row["active_time_s"])
            row["active_power_sample_count"] = len(active_samples)
            row["power_source"] = "measured_samples"
        if idle_samples:
            idle_power = sum(idle_samples) / len(idle_samples)
            row["idle_power_w"] = idle_power
            row["idle_energy_j"] = idle_power * float(row["idle_time_s"])
            row["idle_power_sample_count"] = len(idle_samples)
            row["power_source"] = "measured_samples"


def _select_result_row(
    results_jsonl: Path,
    *,
    row_index: int | None,
) -> tuple[int, dict[str, Any]]:
    rows = _load_result_rows(results_jsonl)
    if row_index is not None:
        if row_index < 0 or row_index >= len(rows):
            raise ValueError(f"{results_jsonl} has {len(rows)} rows; row {row_index} is invalid.")
        row = rows[row_index]
        if not row.get("event_file"):
            raise ValueError(f"{results_jsonl}:{row_index} does not contain event_file.")
        return row_index, row

    for idx, row in enumerate(rows):
        if not row.get("event_file"):
            continue
        if row.get("run_status") in (None, "", "completed"):
            return idx, row
    raise ValueError(f"No usable row with event_file found in {results_jsonl}.")


def _build_rows_from_event_file(
    event_file: Path,
    *,
    results_jsonl: Path | None,
    result_row_index: int | None,
    result_row: dict[str, Any],
    observation_window: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not event_file.exists():
        raise FileNotFoundError(f"Missing event file: {event_file}")

    events = _load_json_events(event_file)
    batches = _batch_events(events)
    if not batches:
        raise ValueError(f"No batch events found in {event_file}.")

    selected_window, start_time, end_time = _select_observation_bounds(
        events,
        batches,
        observation_window,
    )
    n_device = _infer_n_device(
        batches,
        _as_int(result_row.get("effective_n_device"))
        or _as_int(result_row.get("n_device"))
        or _as_int(result_row.get("total_gpus")),
    )
    if n_device <= 0:
        raise ValueError(f"Could not infer n_device from {event_file}.")

    active_intervals_by_device, batch_elapsed_sum, batch_count = _build_active_intervals(
        batches,
        n_device=n_device,
        start_time=start_time,
        end_time=end_time,
    )
    rows_by_device = _compute_state_interval_energy(
        batches,
        n_device=n_device,
        start_time=start_time,
        end_time=end_time,
        active_intervals_by_device=active_intervals_by_device,
    )
    _override_with_measured_power_samples(
        rows_by_device,
        events,
        active_intervals_by_device=active_intervals_by_device,
        start_time=start_time,
        end_time=end_time,
    )

    method = (
        str(result_row.get("method") or "")
        or _method_from_row(result_row)
        or _infer_method_from_event_file(event_file)
    )
    duration = max(0.0, end_time - start_time)
    output_rows: list[dict[str, Any]] = []
    for device_id in range(n_device):
        row = rows_by_device[device_id]
        active_time = float(row["active_time_s"])
        idle_time = float(row["idle_time_s"])
        output_rows.append(
            {
                "results_jsonl": "" if results_jsonl is None else _repo_relative_path(results_jsonl),
                "result_row_index": "" if result_row_index is None else result_row_index,
                "method": method,
                "event_file": _repo_relative_path(event_file),
                "rps": "" if result_row.get("rps") is None else result_row.get("rps"),
                "load_scale": (
                    "" if result_row.get("load_scale") is None else result_row.get("load_scale")
                ),
                "n_device": n_device,
                "device_id": device_id,
                "observation_window": selected_window,
                "observation_start_s": start_time,
                "observation_end_s": end_time,
                "observation_duration_s": duration,
                "batch_count": batch_count[device_id],
                "batch_elapsed_sum_s": batch_elapsed_sum[device_id],
                "active_time_s": active_time,
                "idle_time_s": idle_time,
                "active_fraction": active_time / duration if duration > 0.0 else math.nan,
                "active_power_w": row["active_power_w"],
                "idle_power_w": row["idle_power_w"],
                "active_energy_j": row["active_energy_j"],
                "idle_energy_j": row["idle_energy_j"],
                "active_power_sample_count": row.get("active_power_sample_count", 0),
                "idle_power_sample_count": row.get("idle_power_sample_count", 0),
                "power_source": row["power_source"],
            }
        )

    metadata = {
        "results_jsonl": results_jsonl,
        "result_row_index": result_row_index,
        "event_file": event_file,
        "method": method,
        "observation_window": selected_window,
    }
    return output_rows, metadata


def _build_rows(
    results_jsonl: Path,
    *,
    row_index: int | None,
    observation_window: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_index, result_row = _select_result_row(results_jsonl, row_index=row_index)
    event_file = _resolve_event_file(str(result_row["event_file"]), results_jsonl=results_jsonl)
    return _build_rows_from_event_file(
        event_file,
        results_jsonl=results_jsonl,
        result_row_index=selected_index,
        result_row=result_row,
        observation_window=observation_window,
    )


def _build_rows_for_direct_event_file(
    event_file: Path,
    *,
    n_device: int | None,
    method: str,
    observation_window: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved_event_file = _resolve_event_file(str(event_file), results_jsonl=None)
    result_row: dict[str, Any] = {}
    if n_device is not None:
        result_row["n_device"] = n_device
    if method:
        result_row["method"] = method
    return _build_rows_from_event_file(
        resolved_event_file,
        results_jsonl=None,
        result_row_index=None,
        result_row=result_row,
        observation_window=observation_window,
    )


def _write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "results_jsonl",
        "result_row_index",
        "method",
        "event_file",
        "rps",
        "load_scale",
        "n_device",
        "device_id",
        "observation_window",
        "observation_start_s",
        "observation_end_s",
        "observation_duration_s",
        "batch_count",
        "batch_elapsed_sum_s",
        "active_time_s",
        "idle_time_s",
        "active_fraction",
        "active_power_w",
        "idle_power_w",
        "active_energy_j",
        "idle_energy_j",
        "active_power_sample_count",
        "idle_power_sample_count",
        "power_source",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]], metadata: dict[str, Any], png_path: Path) -> None:
    plt = _get_plotting_dependencies()
    _apply_paper_style(plt)

    device_ids = [int(row["device_id"]) for row in rows]
    active_times = [float(row["active_time_s"]) for row in rows]
    idle_times = [float(row["idle_time_s"]) for row in rows]
    active_energies = [
        0.0 if math.isnan(float(row["active_energy_j"])) else float(row["active_energy_j"])
        for row in rows
    ]
    idle_energies = [
        0.0 if math.isnan(float(row["idle_energy_j"])) else float(row["idle_energy_j"])
        for row in rows
    ]

    fig, (time_ax, energy_ax) = plt.subplots(
        2,
        1,
        figsize=(8.4, 6.0),
        sharex=True,
        constrained_layout=True,
    )
    x_positions = list(range(len(device_ids)))

    time_ax.bar(x_positions, active_times, color="#4C78A8", label="Active")
    time_ax.bar(x_positions, idle_times, bottom=active_times, color="#B8B8B8", label="Idle")
    time_ax.set_ylabel("Time (s)")
    time_ax.legend(frameon=False, ncol=2)

    width = 0.38
    energy_ax.bar(
        [x - width / 2 for x in x_positions],
        active_energies,
        width=width,
        color="#2E7D32",
        label="Active",
    )
    energy_ax.bar(
        [x + width / 2 for x in x_positions],
        idle_energies,
        width=width,
        color="#E76F51",
        label="Idle",
    )
    energy_ax.set_ylabel("Energy (J)")
    energy_ax.set_xlabel("Device Index")
    energy_ax.legend(frameon=False, ncol=2)
    energy_ax.set_xticks(x_positions)
    energy_ax.set_xticklabels([str(device_id) for device_id in device_ids])

    for ax in (time_ax, energy_ax):
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, linestyle=(0, (2, 2)))
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.1)

    title_parts = []
    if metadata.get("method"):
        title_parts.append(str(metadata["method"]))
    if metadata.get("result_row_index") is not None:
        title_parts.append(f"row {metadata['result_row_index']}")
    else:
        title_parts.append(Path(metadata["event_file"]).name)
    title_parts.append(str(metadata["observation_window"]))
    time_ax.set_title(" | ".join(title_parts))

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-device active/idle time from batch events and active/idle "
            "average power for one results.jsonl row or one event file."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        help="Path to a results.jsonl file or a *.events.jsonl event file.",
    )
    parser.add_argument(
        "--event-file",
        type=Path,
        default=None,
        help="Analyze this event file directly instead of reading event_file from results.jsonl.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=None,
        help="Zero-based results.jsonl row to analyze. Defaults to the first usable row.",
    )
    parser.add_argument(
        "--observation-window",
        choices=("auto", "global_arrival", "arrival", "batch"),
        default="batch",
        help=(
            "Window over which idle time is measured. batch uses first batch start "
            "to last batch end."
        ),
    )
    parser.add_argument(
        "--n-device",
        type=int,
        default=None,
        help="Optional device count override for direct --event-file mode.",
    )
    parser.add_argument(
        "--method",
        default="",
        help="Optional method label for direct --event-file mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_STEM,
        help="Output stem. The script writes .csv and .png.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.n_device is not None and args.n_device <= 0:
        raise SystemExit("--n-device must be positive.")

    direct_event_file = args.event_file
    if direct_event_file is None and args.input_path is not None:
        input_name = args.input_path.name
        if input_name.endswith(".events.jsonl") or ".events." in input_name:
            direct_event_file = args.input_path

    if direct_event_file is not None:
        rows, metadata = _build_rows_for_direct_event_file(
            direct_event_file,
            n_device=args.n_device,
            method=args.method,
            observation_window=args.observation_window,
        )
    else:
        if args.input_path is None:
            raise SystemExit("Pass a results.jsonl path or use --event-file path/to/file.events.jsonl.")
        rows, metadata = _build_rows(
            args.input_path,
            row_index=args.row_index,
            observation_window=args.observation_window,
        )

    csv_path = args.output.with_suffix(".csv")
    png_path = args.output.with_suffix(".png")
    _write_csv(rows, csv_path)
    _plot(rows, metadata, png_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    if metadata.get("results_jsonl") is None:
        print(f"Analyzed event file {metadata['event_file']}.")
    else:
        print(
            f"Analyzed row {metadata['result_row_index']} from {metadata['results_jsonl']} "
            f"({metadata['event_file']})."
        )


if __name__ == "__main__":
    main()
