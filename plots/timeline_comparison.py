import json
from collections import defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from plots.common import get_method_color, get_paper_figure_dir
from motivation.events_analysis import (
    build_active_requests_step,
    count_at_times,
    draw_energy_comparison,
)


def timeline_comparison(
    event_files: list[tuple[str, str, int]] = [
        ("Baseline", "traces/7B_code_baseline/qlm_round_robin_1.0_8_tp1_arrival_3.0_0.025_asap_fbsz256.events.jsonl", 8),
        ("Ours", "traces/7B_code_ours/atfc_slosserve_planner_1.0_8_tp1_arrival_3.0_0.025_asap_fbsz256.events.jsonl", 8)
    ],
    prefix: str | None = None,
    surfix = 'coding'
):
    if prefix is None:
        prefix = str(get_paper_figure_dir("timeline_comparison", "timeline_comparison") / "energy_consumption")
    draw_energy_comparison(event_files, output_suffix=surfix, output_prefix= prefix)


def _infer_trace_style_key(event_file: str) -> str:
    stem = Path(event_file).stem
    if "atfc" in stem or "slosserve_planner" in stem:
        return "atfc / slosserve_planner"
    if "sarathi" in stem:
        return "sarathi / round_robin"
    if "vllm" in stem:
        return "vllm / round_robin"
    if "llumnix" in stem:
        return "qlm / llumnix_load"
    return "qlm / round_robin"


def _blend_with_white(color: str, blend: float) -> tuple[float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    return tuple(rgb * (1.0 - blend) + blend)


def _collect_energy_samples(events: list[dict]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    energy_times: dict[int, list[float]] = defaultdict(list)
    energy_powers: dict[int, list[float]] = defaultdict(list)
    for event in events:
        if event.get("event_type") != "energy":
            continue
        device_id = int(event.get("device_id", -1))
        if device_id < 0:
            continue
        energy_times[device_id].append(float(event["timestamp"]))
        energy_powers[device_id].append(float(event["power"]))

    sorted_times: dict[int, np.ndarray] = {}
    sorted_powers: dict[int, np.ndarray] = {}
    for device_id, times in energy_times.items():
        order = np.argsort(times)
        sorted_times[device_id] = np.asarray(times, dtype=float)[order]
        sorted_powers[device_id] = np.asarray(energy_powers[device_id], dtype=float)[order]
    return sorted_times, sorted_powers


def _nearest_power(
    energy_times: np.ndarray,
    energy_powers: np.ndarray,
    timestamp: float,
    max_gap_s: float = 0.2,
) -> float | None:
    if energy_times.size == 0:
        return None

    idx = int(np.searchsorted(energy_times, timestamp, side="left"))
    best_idx = -1
    best_gap = max_gap_s
    for candidate in (idx - 1, idx):
        if 0 <= candidate < energy_times.size:
            gap = abs(float(energy_times[candidate]) - timestamp)
            if gap <= best_gap:
                best_gap = gap
                best_idx = candidate
    if best_idx < 0:
        return None
    return float(energy_powers[best_idx])


def _plot_mean_std_by_groups(
    ax: plt.Axes,
    labels: list[str],
    grouped_values: list[list[float]],
    *,
    color: str,
    title: str,
    xlabel: str,
) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Server Power (W)")
    ax.grid(True, alpha=0.3)

    if not grouped_values:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
        return

    means = np.asarray([np.mean(values) for values in grouped_values], dtype=float)
    stds = np.asarray([np.std(values) for values in grouped_values], dtype=float)
    x = np.arange(len(labels), dtype=float)
    ax.errorbar(
        x,
        means,
        yerr=stds,
        fmt="o",
        color=color,
        markerfacecolor=color,
        markeredgecolor=color,
        markersize=7,
        linewidth=2.0,
        ecolor=color,
        capsize=5,
        elinewidth=1.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if len(labels) > 12:
        ax.tick_params(axis="x", rotation=45)


def _group_by_exact_value(xs: list[int], ys: list[float]) -> tuple[list[str], list[list[float]]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for x, y in zip(xs, ys):
        grouped[int(x)].append(float(y))
    values = sorted(grouped)
    return [str(v) for v in values], [grouped[v] for v in values]


def _group_batch_tokens(batch_tokens: list[int], batch_powers: list[float]) -> tuple[list[str], list[list[float]]]:
    bucket_defs = [
        ("1-2", 1, 2),
        ("3-4", 3, 4),
        ("5-16", 5, 16),
    ]
    grouped: dict[str, list[float]] = {label: [] for label, _, _ in bucket_defs}
    overflow: list[float] = []

    for token_count, power in zip(batch_tokens, batch_powers):
        placed = False
        for label, low, high in bucket_defs:
            if low <= token_count <= high:
                grouped[label].append(float(power))
                placed = True
                break
        if not placed and token_count >= 17:
            overflow.append(float(power))

    labels = ["0"]
    values = [[70.0]]
    for label, _, _ in bucket_defs:
        if grouped[label]:
            labels.append(label)
            values.append(grouped[label])
    if overflow:
        labels.append("17+")
        values.append(overflow)
    return labels, values


def draw_energy_vs_bs_or_concurrency(
    event_file: str =  "traces/7B_code_baseline/qlm_round_robin_1.0_8_tp1_arrival_3.0_0.025_asap_fbsz256.events.jsonl"
):
    with open(event_file, "r", encoding="utf-8") as f:
        events = json.load(f)

    energy_times, energy_powers = _collect_energy_samples(events)

    arrivals: dict[str, tuple[int, float]] = {}
    active_intervals: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for event in events:
        event_type = event.get("event_type")
        request_id = event.get("request_id")
        if request_id is None:
            continue
        if event_type == "arrival":
            device_id = int(event.get("device_id", -1))
            if device_id >= 0:
                arrivals[str(request_id)] = (device_id, float(event["timestamp"]))
        elif event_type == "finish":
            start = arrivals.get(str(request_id))
            if start is None:
                continue
            finish_time = float(event["timestamp"])
            if finish_time >= start[1]:
                active_intervals[start[0]].append((start[1], finish_time))

    active_request_counts: list[int] = []
    active_request_powers: list[float] = []
    for device_id, times in energy_times.items():
        intervals = active_intervals.get(device_id)
        if not intervals:
            continue
        event_times, counts_after = build_active_requests_step(intervals)
        counts = count_at_times(event_times, counts_after, times).astype(int)
        active_request_counts.extend(counts.tolist())
        active_request_powers.extend(energy_powers[device_id].tolist())

    batch_tokens: list[int] = []
    batch_powers: list[float] = []
    for event in events:
        if event.get("event_type") != "batch":
            continue
        device_id = int(event.get("device_id", -1))
        times = energy_times.get(device_id)
        powers = energy_powers.get(device_id)
        if times is None or powers is None:
            continue
        token_count = sum(int(v) for v in event.get("num_scheduled_tokens", {}).values())
        if token_count <= 0:
            continue
        power = _nearest_power(times, powers, float(event["timestamp"]))
        if power is None:
            continue
        batch_tokens.append(token_count)
        batch_powers.append(power)

    output_dir = get_paper_figure_dir("timeline_comparison", "draw_energy_vs_bs_or_concurrency")
    output_prefix = output_dir / f"{Path(event_file).stem.removesuffix('.events')}_power_vs_state"
    base_color = get_method_color(_infer_trace_style_key(event_file))
    accent_color = _blend_with_white(base_color, 0.28)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), tight_layout=True)
    active_labels, active_groups = _group_by_exact_value(
        active_request_counts,
        active_request_powers,
    )
    _plot_mean_std_by_groups(
        axes[0],
        active_labels,
        active_groups,
        color=base_color,
        title="Power vs # Active Requests",
        xlabel="# Active Requests",
    )
    token_labels, token_groups = _group_batch_tokens(batch_tokens, batch_powers)
    _plot_mean_std_by_groups(
        axes[1],
        token_labels,
        token_groups,
        color=accent_color,
        title="Power vs # Current Tokens in Batch",
        xlabel="# Current Tokens in Batch",
    )
    fig.savefig(f"{output_prefix}.png", dpi=250, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_prefix}.png")
    print(f"Saved {output_prefix}.pdf")


def _to_nonnegative_float(value: object) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _extract_overhead_metrics(events: list[dict]) -> dict[str, object]:
    routing_overheads: list[float] = []
    scheduling_overheads: list[float] = []
    execution_times: list[float] = []
    single_admission_overheads: list[float] = []
    compound_routing_overheads: list[float] = []

    scheduling_by_device: dict[int, list[float]] = defaultdict(list)
    execution_by_device: dict[int, list[float]] = defaultdict(list)
    router_arrivals: dict[str, float] = {}
    engine_admissions: dict[str, float] = {}
    admission_rounds: dict[str, int] = defaultdict(int)
    pending_dispatches: dict[str, list[float]] = defaultdict(list)

    for event in events:
        event_type = event.get("event_type")
        timestamp = _to_nonnegative_float(event.get("timestamp", 0.0))

        if event_type == "routing":
            routing_overheads.append(_to_nonnegative_float(event.get("routing_overhead", 0.0)))
        elif event_type == "batch":
            device_id = int(event.get("device_id", -1))
            scheduling_overhead = _to_nonnegative_float(event.get("scheduling_overhead", 0.0))
            elapsed = _to_nonnegative_float(event.get("elapsed", 0.0))
            execution_time = max(0.0, elapsed - scheduling_overhead)
            scheduling_overheads.append(scheduling_overhead)
            execution_times.append(execution_time)
            scheduling_by_device[device_id].append(scheduling_overhead)
            execution_by_device[device_id].append(execution_time)

        request_id = event.get("request_id")
        if request_id is None:
            continue
        request_id = str(request_id)

        if event_type == "arrival-router" and request_id not in router_arrivals:
            router_arrivals[request_id] = timestamp
        elif event_type in {"router_decision", "temporal_rej"}:
            admission_rounds[request_id] += 1
        elif event_type in {"dispatch-both", "dispatch-prefill"}:
            pending_dispatches[request_id].append(timestamp)
        elif event_type == "add_request":
            extra_args = event.get("extra_args") or {}
            if extra_args.get("admitted") and request_id not in engine_admissions:
                engine_admissions[request_id] = timestamp
        elif event_type in {"admitted", "rescheduling"}:
            dispatches = pending_dispatches.get(request_id)
            if dispatches:
                single_admission_overheads.append(max(0.0, timestamp - dispatches.pop(0)))

    for request_id, engine_admit_time in engine_admissions.items():
        router_arrival_time = router_arrivals.get(request_id)
        if router_arrival_time is None:
            continue
        compound_routing_overheads.append(max(0.0, engine_admit_time - router_arrival_time))

    all_request_ids = sorted(set(router_arrivals) | set(admission_rounds))
    return {
        "routing_overheads": routing_overheads,
        "single_admission_overheads": single_admission_overheads,
        "compound_routing_overheads": compound_routing_overheads,
        "scheduling_overheads": scheduling_overheads,
        "execution_times": execution_times,
        "admission_round_counts": [int(admission_rounds.get(request_id, 0)) for request_id in all_request_ids],
        "scheduling_by_device": scheduling_by_device,
        "execution_by_device": execution_by_device,
    }


def _plot_time_distribution(
    ax: plt.Axes,
    values: list[float],
    *,
    title: str,
    xlabel: str,
    color: str,
) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    if not values:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
        return

    arr = np.asarray(values, dtype=float)
    bins = min(60, max(12, int(np.sqrt(arr.size))))
    ax.hist(arr * 1000.0, bins=bins, color=color, alpha=0.82)


def _plot_round_distribution(
    ax: plt.Axes,
    counts: list[int],
    *,
    title: str,
    color: str,
) -> None:
    ax.set_title(title)
    ax.set_xlabel("# Admission Rounds")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    if not counts:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
        return

    xs, ys = np.unique(np.asarray(counts, dtype=int), return_counts=True)
    ax.bar(xs, ys, color=color, alpha=0.82, width=0.8)
    ax.set_xticks(xs)


def _plot_device_overhead_breakdown(
    ax: plt.Axes,
    scheduling_by_device: dict[int, list[float]],
    execution_by_device: dict[int, list[float]],
    *,
    scheduling_color: str,
    execution_color: str,
) -> None:
    ax.set_title("Per-Device Scheduling vs Execution")
    ax.set_xlabel("Device ID")
    ax.set_ylabel("Cumulative Batch Time (s)")
    ax.grid(True, alpha=0.3)

    device_ids = sorted(set(scheduling_by_device) | set(execution_by_device))
    if not device_ids:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.arange(len(device_ids), dtype=float)
    scheduling_totals = np.asarray(
        [sum(scheduling_by_device.get(device_id, ())) for device_id in device_ids],
        dtype=float,
    )
    execution_totals = np.asarray(
        [sum(execution_by_device.get(device_id, ())) for device_id in device_ids],
        dtype=float,
    )
    ax.bar(x, scheduling_totals, color=scheduling_color, alpha=0.82, label="Scheduling")
    ax.bar(
        x,
        execution_totals,
        bottom=scheduling_totals,
        color=execution_color,
        alpha=0.82,
        label="Execution",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(device_id) for device_id in device_ids])
    ax.legend()
    
def draw_overhead_breakdown(
    event_files:list[tuple[str, str]] = [("Baseline", "traces/7B_code_baseline/qlm_round_robin_1.0_8_tp1_arrival_3.0_0.025_asap_fbsz256.events.jsonl"),
    ("Ours", "traces/7B_code_ours/atfc_slosserve_planner_1.0_8_tp1_arrival_3.0_0.025_asap_fbsz256.events.jsonl")]
):
    output_dir = get_paper_figure_dir("timeline_comparison", "draw_overhead_breakdown")

    for label, event_file in event_files:
        with open(event_file, "r", encoding="utf-8") as f:
            events = json.load(f)

        metrics = _extract_overhead_metrics(events)
        output_prefix = output_dir / f"{Path(event_file).stem.removesuffix('.events')}_overhead_breakdown"
        base_color = get_method_color(_infer_trace_style_key(event_file))
        accent_color = _blend_with_white(base_color, 0.28)

        fig, axes = plt.subplots(4, 2, figsize=(15, 18))
        axes = axes.ravel()

        _plot_device_overhead_breakdown(
            axes[0],
            metrics["scheduling_by_device"],
            metrics["execution_by_device"],
            scheduling_color=base_color,
            execution_color=accent_color,
        )
        _plot_time_distribution(
            axes[1],
            metrics["routing_overheads"],
            title="Single-Round Routing Overhead",
            xlabel="Routing Overhead (ms)",
            color=base_color,
        )
        _plot_time_distribution(
            axes[2],
            metrics["single_admission_overheads"],
            title="Single Admission Check Overhead",
            xlabel="Admission Check Overhead (ms)",
            color=base_color,
        )
        _plot_time_distribution(
            axes[3],
            metrics["compound_routing_overheads"],
            title="Compound Routing Overhead",
            xlabel="Arrival-to-Admission Overhead (ms)",
            color=base_color,
        )
        _plot_time_distribution(
            axes[4],
            metrics["scheduling_overheads"],
            title="Scheduling Overhead",
            xlabel="Scheduling Overhead (ms)",
            color=base_color,
        )
        _plot_time_distribution(
            axes[5],
            metrics["execution_times"],
            title="Execution Time",
            xlabel="Execution Time (ms)",
            color=accent_color,
        )
        _plot_round_distribution(
            axes[6],
            metrics["admission_round_counts"],
            title="Admission Rounds Distribution",
            color=base_color,
        )
        axes[7].axis("off")

        fig.suptitle(label)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
        fig.savefig(f"{output_prefix}.png", dpi=250, bbox_inches="tight")
        fig.savefig(f"{output_prefix}.pdf", dpi=250, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_prefix}.png")
        print(f"Saved {output_prefix}.pdf")

timeline_comparison(event_files = [
        ("Baseline", "traces/7B_code_baseline/qlm_round_robin_1.0_8_tp1_arrival_3.0_0.025_asap_fbsz256.events.jsonl", 8),
        ("Ours", "traces/7B_code_ours/atfc_slosserve_planner_1.0_8_tp1_arrival_3.0_0.025_asap_fbsz256.events.jsonl", 8)
    ],
    surfix = 'coding')
draw_energy_vs_bs_or_concurrency("traces/7B_code_baseline/qlm_round_robin_1.0_8_tp1_arrival_3.0_0.025_asap_fbsz256.events.jsonl")
draw_energy_vs_bs_or_concurrency("traces/7B_code_ours/atfc_slosserve_planner_1.0_8_tp1_arrival_3.0_0.025_asap_fbsz256.events.jsonl")
draw_overhead_breakdown()