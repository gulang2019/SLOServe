import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from motivation.events_analysis import (
    _compute_active_device_series,
    _compute_measured_power_series,
    analyze_events,
    build_active_requests_step,
    count_at_times,
)


def load_events(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_request_intervals(events: list[dict]) -> list[tuple[float, float]]:
    arrivals: dict[str, float] = {}
    finishes: dict[str, float] = {}
    for event in events:
        event_type = event.get("event_type")
        req_id = event.get("request_id")
        if req_id is None:
            continue
        if event_type == "arrival":
            arrivals.setdefault(req_id, float(event["timestamp"]))
        elif event_type == "finish":
            finishes[req_id] = float(event["timestamp"])

    intervals: list[tuple[float, float]] = []
    for req_id, arrival_ts in arrivals.items():
        finish_ts = finishes.get(req_id)
        if finish_ts is None:
            continue
        if finish_ts < arrival_ts:
            finish_ts = arrival_ts
        intervals.append((arrival_ts, finish_ts))
    return intervals


def extract_request_intervals_by_device(
    events: list[dict],
) -> dict[int, list[tuple[float, float]]]:
    arrivals: dict[str, tuple[float, int]] = {}
    finishes: dict[str, tuple[float, int]] = {}
    for event in events:
        event_type = event.get("event_type")
        req_id = event.get("request_id")
        if req_id is None:
            continue
        if event_type == "arrival":
            arrivals.setdefault(
                req_id, (float(event["timestamp"]), int(event.get("device_id", -1)))
            )
        elif event_type == "finish":
            finishes[req_id] = (
                float(event["timestamp"]),
                int(event.get("device_id", -1)),
            )

    intervals: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for req_id, (arrival_ts, device_id) in arrivals.items():
        finish_info = finishes.get(req_id)
        if finish_info is None:
            continue
        finish_ts, finish_device_id = finish_info
        if device_id < 0 and finish_device_id >= 0:
            device_id = finish_device_id
        if device_id < 0:
            continue
        if finish_ts < arrival_ts:
            finish_ts = arrival_ts
        intervals[device_id].append((arrival_ts, finish_ts))
    return intervals


def compute_energy_power_bins(
    events: list[dict],
    start_time: float,
    end_time: float,
    window_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy_events = [e for e in events if e.get("event_type") == "energy"]
    if not energy_events:
        return np.array([]), np.array([]), np.array([])

    n_windows = max(1, int(math.ceil((end_time - start_time) / window_size)))
    time_axis = start_time + np.arange(n_windows, dtype=np.float64) * window_size
    energy_bins = np.zeros(n_windows, dtype=np.float64)

    for event in energy_events:
        ts = float(event["timestamp"])
        if ts < start_time or ts >= end_time:
            continue
        idx = int((ts - start_time) // window_size)
        if 0 <= idx < n_windows:
            energy_bins[idx] += float(event.get("energy", 0.0) or 0.0)

    power_bins = energy_bins / window_size
    return time_axis - start_time, energy_bins, power_bins


def compute_energy_power_bins_by_device(
    events: list[dict],
    start_time: float,
    end_time: float,
    window_size: float,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    energy_events = [e for e in events if e.get("event_type") == "energy"]
    n_windows = max(1, int(math.ceil((end_time - start_time) / window_size)))
    time_axis = start_time + np.arange(n_windows, dtype=np.float64) * window_size
    energy_by_device: dict[int, np.ndarray] = {}

    for event in energy_events:
        ts = float(event["timestamp"])
        if ts < start_time or ts >= end_time:
            continue
        device_id = int(event.get("device_id", -1))
        if device_id < 0:
            continue
        idx = int((ts - start_time) // window_size)
        if not 0 <= idx < n_windows:
            continue
        if device_id not in energy_by_device:
            energy_by_device[device_id] = np.zeros(n_windows, dtype=np.float64)
        energy_by_device[device_id][idx] += float(event.get("energy", 0.0) or 0.0)

    power_by_device = {
        device_id: energy / window_size
        for device_id, energy in energy_by_device.items()
    }
    return time_axis - start_time, energy_by_device, power_by_device


def compute_active_request_bins(
    events: list[dict],
    start_time: float,
    end_time: float,
    window_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    intervals = extract_request_intervals(events)
    n_windows = max(1, int(math.ceil((end_time - start_time) / window_size)))
    time_axis = start_time + np.arange(n_windows, dtype=np.float64) * window_size
    centers = time_axis + 0.5 * window_size
    if not intervals:
        return time_axis - start_time, np.zeros(n_windows, dtype=int)
    event_times, counts_after = build_active_requests_step(intervals)
    counts = count_at_times(event_times, counts_after, centers)
    return time_axis - start_time, counts.astype(int)


def compute_active_request_bins_by_device(
    events: list[dict],
    start_time: float,
    end_time: float,
    window_size: float,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    n_windows = max(1, int(math.ceil((end_time - start_time) / window_size)))
    time_axis = start_time + np.arange(n_windows, dtype=np.float64) * window_size
    centers = time_axis + 0.5 * window_size
    intervals_by_device = extract_request_intervals_by_device(events)
    result: dict[int, np.ndarray] = {}
    for device_id, intervals in intervals_by_device.items():
        if not intervals:
            result[device_id] = np.zeros(n_windows, dtype=int)
            continue
        event_times, counts_after = build_active_requests_step(intervals)
        counts = count_at_times(event_times, counts_after, centers)
        result[device_id] = counts.astype(int)
    return time_axis - start_time, result


def summarize_trace(
    path: str,
    start_time: float,
    end_time: float,
    window_size: float,
    *,
    time_offset: float = 0.0,
    recover_idle_gaps: bool = False,
    idle_power_per_device: float = 70.0,
) -> dict:
    events = load_events(path)
    req_time, active_reqs = compute_active_request_bins(
        events, start_time, end_time, window_size
    )
    typed_events, _ = analyze_events(path)
    _, active_servers, _, _ = _compute_active_device_series(
        typed_events,
        window_size=window_size,
        start_time=start_time,
        end_time=end_time,
        n_device=8,
    )
    power_summary = _compute_measured_power_series(
        typed_events,
        n_device=8,
        window_size=window_size,
        start_time=start_time,
        end_time=end_time,
        recover_idle_gaps=recover_idle_gaps,
        idle_power_per_device=idle_power_per_device,
    )
    power_time = np.asarray(power_summary["time"], dtype=float)
    power_bins = np.asarray(power_summary["total_power"], dtype=float)
    energy_bins = power_bins * window_size
    power_bins_by_device = {
        int(device_id): np.asarray(device_power, dtype=float)
        for device_id, device_power in power_summary["per_device_power"].items()
    }
    energy_bins_by_device = {
        device_id: device_power * window_size
        for device_id, device_power in power_bins_by_device.items()
    }
    _, active_reqs_by_device = compute_active_request_bins_by_device(
        events, start_time, end_time, window_size
    )
    return {
        "path": path,
        "events": events,
        "time": req_time + time_offset,
        "active_reqs": active_reqs,
        "active_servers": np.asarray(active_servers, dtype=int),
        "power_time": power_time + time_offset,
        "energy_bins": energy_bins,
        "power_bins": power_bins,
        "energy_bins_by_device": energy_bins_by_device,
        "power_bins_by_device": power_bins_by_device,
        "active_reqs_by_device": active_reqs_by_device,
        "total_energy_joules": float(np.sum(energy_bins)),
        "power_source": power_summary["source"],
    }


def compute_common_window(
    left_events: list[dict],
    right_events: list[dict],
) -> tuple[float, float]:
    left_energy_times = [
        float(e["timestamp"]) for e in left_events if e.get("event_type") == "energy"
    ]
    right_energy_times = [
        float(e["timestamp"]) for e in right_events if e.get("event_type") == "energy"
    ]
    if not left_energy_times or not right_energy_times:
        raise ValueError("Both traces need energy events.")
    start_time = max(min(left_energy_times), min(right_energy_times))
    end_time = min(max(left_energy_times), max(right_energy_times))
    if end_time <= start_time:
        raise ValueError("The traces do not overlap in time.")
    return start_time, end_time


def resolve_view_window(
    common_start_time: float,
    common_end_time: float,
    *,
    view_start: float | None,
    view_end: float | None,
) -> tuple[float, float]:
    start_time = common_start_time
    end_time = common_end_time
    if view_start is not None:
        start_time = common_start_time + float(view_start)
    if view_end is not None:
        end_time = common_start_time + float(view_end)
    if end_time <= start_time:
        raise ValueError(
            f"Invalid view window: start={start_time:.3f}, end={end_time:.3f}"
        )
    start_time = max(common_start_time, start_time)
    end_time = min(common_end_time, end_time)
    if end_time <= start_time:
        raise ValueError(
            f"View window [{view_start}, {view_end}] is outside the common window."
        )
    return start_time, end_time


def plot_comparison(
    rr: dict,
    planner: dict,
    window_size: float,
    output_prefix: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), tight_layout=True)

    ax = axes[0, 0]
    ax.step(rr["time"], rr["active_reqs"], where="post", label="Round Robin", color="#c73e1d")
    ax.step(
        planner["time"],
        planner["active_reqs"],
        where="post",
        label="Packer",
        color="#1f78b4",
    )
    ax.set_title(f"Active Requests Over Time ({window_size:.1f}s bins)")
    ax.set_xlabel("Time Since Common Window Start (s)")
    ax.set_ylabel("Active Requests")
    ax.legend()

    ax = axes[0, 1]
    ax.step(rr["power_time"], rr["power_bins"], where="post", label="Round Robin", color="#c73e1d")
    ax.step(
        planner["power_time"],
        planner["power_bins"],
        where="post",
        label="Packer",
        color="#1f78b4",
    )
    ax.set_title(f"Total Power Over Time ({window_size:.1f}s bins)")
    ax.set_xlabel("Time Since Common Window Start (s)")
    ax.set_ylabel("Power (W)")
    ax.legend()

    ax = axes[1, 0]
    rr_counts = np.bincount(rr["active_reqs"]) if len(rr["active_reqs"]) else np.array([0])
    planner_counts = (
        np.bincount(planner["active_reqs"]) if len(planner["active_reqs"]) else np.array([0])
    )
    max_len = max(len(rr_counts), len(planner_counts))
    rr_counts = np.pad(rr_counts, (0, max_len - len(rr_counts)))
    planner_counts = np.pad(planner_counts, (0, max_len - len(planner_counts)))
    xs = np.arange(max_len)
    width = 0.42
    total_rr_time = max(float(np.sum(rr_counts) * window_size), 1e-12)
    total_planner_time = max(float(np.sum(planner_counts) * window_size), 1e-12)
    rr_pct = rr_counts * window_size * 100.0 / total_rr_time
    planner_pct = planner_counts * window_size * 100.0 / total_planner_time
    ax.bar(xs - width / 2, rr_pct, width=width, label="Round Robin", color="#c73e1d", alpha=0.75)
    ax.bar(xs + width / 2, planner_pct, width=width, label="Packer", color="#1f78b4", alpha=0.75)
    ax.set_title("Time Spent At Each Active-Request Count")
    ax.set_xlabel("Active Requests")
    ax.set_ylabel("Window Time (%)")
    ax.legend()

    ax = axes[1, 1]
    bins = np.linspace(
        0.0,
        max(
            float(np.max(rr["power_bins"])) if len(rr["power_bins"]) else 0.0,
            float(np.max(planner["power_bins"])) if len(planner["power_bins"]) else 0.0,
            1.0,
        ),
        40,
    )
    ax.hist(rr["power_bins"], bins=bins, alpha=0.55, label="Round Robin", density=True, color="#c73e1d")
    ax.hist(planner["power_bins"], bins=bins, alpha=0.55, label="Packer", density=True, color="#1f78b4")
    ax.set_title("Power Distribution")
    ax.set_xlabel("Windowed Power (W)")
    ax.set_ylabel("Density")
    ax.legend()

    fig.savefig(f"{output_prefix}.png", dpi=250, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_power_vs_active_requests(
    rr: dict,
    planner: dict,
    output_prefix: str,
) -> None:
    max_req = int(max(
        np.max(rr["active_reqs"]) if len(rr["active_reqs"]) else 0,
        np.max(planner["active_reqs"]) if len(planner["active_reqs"]) else 0,
    ))
    rr_groups = []
    planner_groups = []
    labels = []
    positions_rr = []
    positions_planner = []
    for nreq in range(max_req + 1):
        rr_vals = rr["power_bins"][rr["active_reqs"] == nreq]
        planner_vals = planner["power_bins"][planner["active_reqs"] == nreq]
        if len(rr_vals) == 0 and len(planner_vals) == 0:
            continue
        base = len(labels) * 2.6
        positions_rr.append(base - 0.45)
        positions_planner.append(base + 0.45)
        rr_groups.append(rr_vals if len(rr_vals) else np.array([np.nan]))
        planner_groups.append(planner_vals if len(planner_vals) else np.array([np.nan]))
        labels.append(str(nreq))

    fig, ax = plt.subplots(figsize=(16, 6), tight_layout=True)
    rr_bp = ax.boxplot(
        rr_groups,
        positions=positions_rr,
        widths=0.7,
        patch_artist=True,
        showfliers=False,
    )
    planner_bp = ax.boxplot(
        planner_groups,
        positions=positions_planner,
        widths=0.7,
        patch_artist=True,
        showfliers=False,
    )

    for patch in rr_bp["boxes"]:
        patch.set(facecolor="#c73e1d", alpha=0.5)
    for patch in planner_bp["boxes"]:
        patch.set(facecolor="#1f78b4", alpha=0.5)
    for key in ("whiskers", "caps", "medians"):
        for artist in rr_bp[key]:
            artist.set(color="#7f2704", linewidth=1.5)
        for artist in planner_bp[key]:
            artist.set(color="#08519c", linewidth=1.5)

    tick_positions = [(a + b) / 2.0 for a, b in zip(positions_rr, positions_planner)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Active Requests Per Window")
    ax.set_ylabel("Windowed Power (W)")
    ax.set_title("Power Distribution By Active-Request Bin")
    ax.legend(
        [
            plt.Line2D([0], [0], color="#c73e1d", lw=8, alpha=0.5),
            plt.Line2D([0], [0], color="#1f78b4", lw=8, alpha=0.5),
        ],
        ["Round Robin", "Packer"],
        loc="upper left",
    )

    fig.savefig(f"{output_prefix}.png", dpi=250, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_active_server_distribution(
    rr: dict,
    planner: dict,
    window_size: float,
    output_prefix: str,
) -> None:
    rr_counts = (
        np.bincount(rr["active_servers"])
        if len(rr["active_servers"]) else np.array([0], dtype=float)
    )
    planner_counts = (
        np.bincount(planner["active_servers"])
        if len(planner["active_servers"]) else np.array([0], dtype=float)
    )
    max_len = max(len(rr_counts), len(planner_counts))
    rr_counts = np.pad(rr_counts, (0, max_len - len(rr_counts)))
    planner_counts = np.pad(planner_counts, (0, max_len - len(planner_counts)))
    xs = np.arange(max_len)
    width = 0.42

    total_rr_time = max(float(np.sum(rr_counts) * window_size), 1e-12)
    total_planner_time = max(float(np.sum(planner_counts) * window_size), 1e-12)
    rr_pct = rr_counts * window_size * 100.0 / total_rr_time
    planner_pct = planner_counts * window_size * 100.0 / total_planner_time

    fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
    ax.bar(
        xs - width / 2,
        rr_pct,
        width=width,
        label="Round Robin",
        color="#c73e1d",
        alpha=0.75,
    )
    ax.bar(
        xs + width / 2,
        planner_pct,
        width=width,
        label="Packer",
        color="#1f78b4",
        alpha=0.75,
    )
    ax.set_title("Active-Machine Distribution")
    ax.set_xlabel("Active Machines")
    ax.set_ylabel("Window Time (%)")
    ax.set_xticks(xs)
    ax.legend()

    fig.savefig(f"{output_prefix}.png", dpi=250, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_active_machines_and_power_over_time(
    rr: dict,
    planner: dict,
    window_size: float,
    output_prefix: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), tight_layout=True, sharex=True)

    ax = axes[0]
    rr_avg_active = float(np.mean(rr["active_servers"])) if len(rr["active_servers"]) else 0.0
    planner_avg_active = (
        float(np.mean(planner["active_servers"])) if len(planner["active_servers"]) else 0.0
    )
    ax.step(
        rr["time"],
        rr["active_servers"],
        where="post",
        label="Round Robin",
        color="#c73e1d",
    )
    ax.step(
        planner["time"],
        planner["active_servers"],
        where="post",
        label="Packer",
        color="#1f78b4",
    )
    ax.axhline(
        rr_avg_active,
        color="#c73e1d",
        linestyle=":",
        linewidth=2.0,
        label=f"Round Robin Avg = {rr_avg_active:.2f}",
    )
    ax.axhline(
        planner_avg_active,
        color="#1f78b4",
        linestyle=":",
        linewidth=2.0,
        label=f"Packer Avg = {planner_avg_active:.2f}",
    )
    ax.set_title(f"Active Machines Over Time ({window_size:.1f}s bins)")
    ax.set_ylabel("Active Machines")
    ax.legend()

    ax = axes[1]
    ax.step(
        rr["power_time"],
        rr["power_bins"],
        where="post",
        label="Round Robin",
        color="#c73e1d",
    )
    ax.step(
        planner["power_time"],
        planner["power_bins"],
        where="post",
        label="Packer",
        color="#1f78b4",
    )
    ax.set_title(f"Total Power Over Time ({window_size:.1f}s bins)")
    ax.set_xlabel("Time Since Common Window Start (s)")
    ax.set_ylabel("Power (W)")
    ax.legend()

    fig.savefig(f"{output_prefix}.png", dpi=250, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_power_vs_active_requests_per_server(
    rr: dict,
    planner: dict,
    output_prefix: str,
) -> None:
    device_ids = sorted(
        set(rr["active_reqs_by_device"]).union(set(planner["active_reqs_by_device"]))
    )
    if not device_ids:
        return

    ncols = 2
    nrows = int(math.ceil(len(device_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows), tight_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax, device_id in zip(axes.flat, device_ids):
        rr_power = rr["power_bins_by_device"].get(
            device_id, np.zeros_like(rr["power_bins"], dtype=float)
        )
        planner_power = planner["power_bins_by_device"].get(
            device_id, np.zeros_like(planner["power_bins"], dtype=float)
        )
        rr_counts = rr["active_reqs_by_device"].get(
            device_id, np.zeros_like(rr["power_bins"], dtype=int)
        )
        planner_counts = planner["active_reqs_by_device"].get(
            device_id, np.zeros_like(planner["power_bins"], dtype=int)
        )
        max_req = int(
            max(
                np.max(rr_counts) if len(rr_counts) else 0,
                np.max(planner_counts) if len(planner_counts) else 0,
            )
        )
        rr_groups = []
        planner_groups = []
        labels = []
        positions_rr = []
        positions_planner = []
        for nreq in range(max_req + 1):
            rr_vals = rr_power[rr_counts == nreq]
            planner_vals = planner_power[planner_counts == nreq]
            if len(rr_vals) == 0 and len(planner_vals) == 0:
                continue
            base = len(labels) * 2.4
            positions_rr.append(base - 0.4)
            positions_planner.append(base + 0.4)
            rr_groups.append(rr_vals if len(rr_vals) else np.array([np.nan]))
            planner_groups.append(planner_vals if len(planner_vals) else np.array([np.nan]))
            labels.append(str(nreq))

        rr_bp = ax.boxplot(
            rr_groups,
            positions=positions_rr,
            widths=0.65,
            patch_artist=True,
            showfliers=False,
        )
        planner_bp = ax.boxplot(
            planner_groups,
            positions=positions_planner,
            widths=0.65,
            patch_artist=True,
            showfliers=False,
        )
        for patch in rr_bp["boxes"]:
            patch.set(facecolor="#c73e1d", alpha=0.5)
        for patch in planner_bp["boxes"]:
            patch.set(facecolor="#1f78b4", alpha=0.5)
        for key in ("whiskers", "caps", "medians"):
            for artist in rr_bp[key]:
                artist.set(color="#7f2704", linewidth=1.2)
            for artist in planner_bp[key]:
                artist.set(color="#08519c", linewidth=1.2)

        tick_positions = [(a + b) / 2.0 for a, b in zip(positions_rr, positions_planner)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(labels)
        ax.set_title(f"Server {device_id}")
        ax.set_xlabel("Active Requests On Server")
        ax.set_ylabel("Server Power (W)")

    for ax in axes.flat[len(device_ids):]:
        ax.axis("off")

    fig.legend(
        [
            plt.Line2D([0], [0], color="#c73e1d", lw=8, alpha=0.5),
            plt.Line2D([0], [0], color="#1f78b4", lw=8, alpha=0.5),
        ],
        ["Round Robin", "Packer"],
        loc="upper center",
        ncol=2,
    )
    fig.suptitle("Per-Server Power Distribution By Active Requests", y=1.02)
    fig.savefig(f"{output_prefix}.png", dpi=250, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_per_server_power_over_time(
    rr: dict,
    planner: dict,
    window_size: float,
    output_prefix: str,
) -> None:
    device_ids = sorted(
        set(rr["power_bins_by_device"]).union(set(planner["power_bins_by_device"]))
    )
    if not device_ids:
        return

    ncols = 2
    nrows = int(math.ceil(len(device_ids) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(16, 4.5 * nrows),
        tight_layout=True,
        sharex=True,
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax, device_id in zip(axes.flat, device_ids):
        rr_power = rr["power_bins_by_device"].get(
            device_id, np.zeros_like(rr["power_bins"], dtype=float)
        )
        planner_power = planner["power_bins_by_device"].get(
            device_id, np.zeros_like(planner["power_bins"], dtype=float)
        )
        ax.step(
            rr["power_time"],
            rr_power,
            where="post",
            color="#c73e1d",
            label="Round Robin",
        )
        ax.step(
            planner["power_time"],
            planner_power,
            where="post",
            color="#1f78b4",
            label="Packer",
        )
        ax.set_title(f"Server {device_id}")
        ax.set_xlabel("Time Since Common Window Start (s)")
        ax.set_ylabel("Server Power (W)")

    for ax in axes.flat[len(device_ids):]:
        ax.axis("off")

    fig.legend(
        [
            plt.Line2D([0], [0], color="#c73e1d", lw=2),
            plt.Line2D([0], [0], color="#1f78b4", lw=2),
        ],
        ["Round Robin", "Packer"],
        loc="upper center",
        ncol=2,
    )
    fig.suptitle(
        f"Per-Server Power Over Time ({window_size:.1f}s bins)",
        y=1.02,
    )
    fig.savefig(f"{output_prefix}.png", dpi=250, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def print_summary(rr: dict, planner: dict, start_time: float, end_time: float, window_size: float) -> None:
    print(f"Common window: [{start_time:.3f}, {end_time:.3f}] ({end_time - start_time:.3f} s)")
    print(f"Window size: {window_size:.3f} s")
    for label, summary in [("round_robin", rr), ("packer", planner)]:
        active = summary["active_reqs"]
        power = summary["power_bins"]
        print(label)
        print(f"  total_energy_joules: {summary['total_energy_joules']:.3f}")
        print(f"  average_power_watts: {float(np.mean(power)):.3f}")
        print(f"  p50_power_watts: {float(np.percentile(power, 50)):.3f}")
        print(f"  p90_power_watts: {float(np.percentile(power, 90)):.3f}")
        print(f"  average_active_reqs: {float(np.mean(active)):.3f}")
        print(f"  p50_active_reqs: {float(np.percentile(active, 50)):.3f}")
        print(f"  p90_active_reqs: {float(np.percentile(active, 90)):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rr", required=True, help="Round-robin events trace")
    parser.add_argument("--packer", required=True, help="Packed planner events trace")
    parser.add_argument("--window-size", type=float, default=1.0)
    parser.add_argument(
        "--view-start",
        type=float,
        default=None,
        help="Optional zoom start in seconds since the common window start.",
    )
    parser.add_argument(
        "--view-end",
        type=float,
        default=None,
        help="Optional zoom end in seconds since the common window start.",
    )
    parser.add_argument("--recover-idle-gaps", action="store_true")
    parser.add_argument("--idle-power-per-device", type=float, default=70.0)
    parser.add_argument(
        "--output-prefix",
        default="compare_rr_packer_window",
        help="Output path prefix without extension",
    )
    args = parser.parse_args()

    rr_events = load_events(args.rr)
    planner_events = load_events(args.packer)
    common_start_time, common_end_time = compute_common_window(rr_events, planner_events)
    start_time, end_time = resolve_view_window(
        common_start_time,
        common_end_time,
        view_start=args.view_start,
        view_end=args.view_end,
    )

    rr = summarize_trace(
        args.rr,
        start_time,
        end_time,
        args.window_size,
        time_offset=start_time - common_start_time,
        recover_idle_gaps=args.recover_idle_gaps,
        idle_power_per_device=args.idle_power_per_device,
    )
    planner = summarize_trace(
        args.packer,
        start_time,
        end_time,
        args.window_size,
        time_offset=start_time - common_start_time,
        recover_idle_gaps=args.recover_idle_gaps,
        idle_power_per_device=args.idle_power_per_device,
    )

    output_prefix = str(Path(args.output_prefix))
    plot_comparison(rr, planner, args.window_size, output_prefix)
    plot_active_server_distribution(
        rr,
        planner,
        args.window_size,
        f"{output_prefix}.active_machine_distribution",
    )
    plot_active_machines_and_power_over_time(
        rr,
        planner,
        args.window_size,
        f"{output_prefix}.active_machine_and_power_over_time",
    )
    plot_power_vs_active_requests(rr, planner, f"{output_prefix}.power_vs_active_reqs")
    plot_power_vs_active_requests_per_server(
        rr, planner, f"{output_prefix}.power_vs_active_reqs_per_server"
    )
    plot_per_server_power_over_time(
        rr,
        planner,
        args.window_size,
        f"{output_prefix}.power_over_time_per_server",
    )
    print_summary(rr, planner, start_time, end_time, args.window_size)
    print(f"Saved {output_prefix}.png")
    print(f"Saved {output_prefix}.pdf")
    print(f"Saved {output_prefix}.active_machine_distribution.png")
    print(f"Saved {output_prefix}.active_machine_distribution.pdf")
    print(f"Saved {output_prefix}.active_machine_and_power_over_time.png")
    print(f"Saved {output_prefix}.active_machine_and_power_over_time.pdf")
    print(f"Saved {output_prefix}.power_vs_active_reqs.png")
    print(f"Saved {output_prefix}.power_vs_active_reqs.pdf")
    print(f"Saved {output_prefix}.power_vs_active_reqs_per_server.png")
    print(f"Saved {output_prefix}.power_vs_active_reqs_per_server.pdf")
    print(f"Saved {output_prefix}.power_over_time_per_server.png")
    print(f"Saved {output_prefix}.power_over_time_per_server.pdf")


if __name__ == "__main__":
    main()
