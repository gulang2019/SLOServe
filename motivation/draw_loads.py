import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from Dataset.dataset import ArrivalTimes, Requests
from SLOsServe.perf_model import PerfModel, get_easy_name

FIGDIR = "figs/loads"
os.makedirs(FIGDIR, exist_ok=True)

PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.labelsize": 24,
    "axes.titlesize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 19,
    "axes.linewidth": 1.4,
    "lines.linewidth": 2.8,
    "grid.linewidth": 0.9,
}
plt.rcParams.update(PAPER_STYLE)


def count_intervals(intervals: list[tuple[float, float, float, str]],
                    window: float,
                    mode: str = "max") -> list[tuple[float, float, float, float, float]]:
    if not intervals:
        return []
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if mode not in {"min", "max"}:
        raise ValueError(f"Unsupported mode={mode}")

    start_time = min(s for s, _, _, _ in intervals)
    end_time = max(e for _, e, _, _ in intervals)
    n_windows = max(1, math.ceil((end_time - start_time) / window))
    windows = [
        [start_time + idx * window, start_time + (idx + 1) * window, 0.0, 0.0, 0.0]
        for idx in range(n_windows)
    ]

    for start, end, service_time, interval_type in intervals:
        if end <= start:
            continue
        first_window_idx = max(0, math.floor((start - start_time) / window))
        end_exclusive = math.nextafter(end, -math.inf)
        last_window_idx = min(
            n_windows - 1,
            math.floor((end_exclusive - start_time) / window),
        )
        if first_window_idx > last_window_idx:
            continue
        if mode == "min" and first_window_idx != last_window_idx:
            continue

        for window_idx in range(first_window_idx, last_window_idx + 1):
            if interval_type == "P":
                windows[window_idx][2] += service_time
            else:
                windows[window_idx][3] += service_time
            windows[window_idx][4] += service_time

    for window_stat in windows:
        for i in range(2, 5):
            window_stat[i] /= window
    return [tuple(window_stat) for window_stat in windows]


def build_required_server_intervals(arrival_times_name: str,
                                    requests_name: str,
                                    model_name: str,
                                    ttft_slo_scale: float,
                                    slo_tpot: float,
                                    window_start: int,
                                    window_end: int,
                                    load_scale: float = 1.0,
                                    slo_constant: float = 0.0,
                                    time_start: float | None = None,
                                    time_end: float | None = None) -> tuple[list[tuple[float, float, float, str]], int]:
    arrival_times = ArrivalTimes.load(arrival_times_name, load_scale=load_scale).arrival_times
    requests = Requests.load(requests_name, max_tokens=32768).requests
    print('trace len', arrival_times[-1])

    arrival_times = arrival_times[window_start:window_end]
    target_len = len(arrival_times)
    if requests and target_len:
        request_offset = window_start % len(requests)
        repeat_count = math.ceil((request_offset + target_len) / len(requests))
        requests = (requests * repeat_count)[request_offset:request_offset + target_len]
    else:
        requests = []

    if not requests:
        raise ValueError("No requests available after slicing")

    perf_model = PerfModel.get_perf_model(model_name)
    mean_input_length = float(np.mean([req.input_length for req in requests]))
    max_decode_batch_size = max(1, int(perf_model.get_max_decode_batch_size(slo_tpot, mean_input_length)))

    slo_ttft_per_token = perf_model.hardware_params[0] * ttft_slo_scale
    slo_ttft_constant = perf_model.hardware_params[4] * ttft_slo_scale + slo_constant

    intervals: list[tuple[float, float, float, str]] = []
    for request, arrival_time in zip(requests, arrival_times):
        cached_length = getattr(request, "cached_length", 0)
        thinking_length = getattr(request, "thinking_length", 0)
        uncached_input_length = request.input_length - cached_length
        ttft_slo = slo_ttft_per_token * uncached_input_length + slo_ttft_constant
        prefill_service_time = perf_model.get_batch_time([(cached_length, uncached_input_length)])
        intervals.append((
            arrival_time,
            arrival_time + ttft_slo,
            prefill_service_time,
            "P",
        ))

        n_decode_steps = max(request.output_length + thinking_length - 1, 0)
        decode_service_time = slo_tpot / max_decode_batch_size
        for step_idx in range(n_decode_steps):
            step_start = arrival_time + ttft_slo + slo_tpot * step_idx
            step_end = step_start + slo_tpot
            intervals.append((step_start, step_end, decode_service_time, "D"))

    if time_start is not None or time_end is not None:
        lower = -math.inf if time_start is None else float(time_start)
        upper = math.inf if time_end is None else float(time_end)
        clipped_intervals = []
        for start, end, service_time, interval_type in intervals:
            clipped_start = max(start, lower)
            clipped_end = min(end, upper)
            if clipped_end <= clipped_start:
                continue
            clipped_intervals.append((clipped_start, clipped_end, service_time, interval_type))
        intervals = clipped_intervals

    return intervals, max_decode_batch_size


def build_required_server_trace_from_intervals(intervals: list[tuple[float, float, float, str]],
                                               trace_window: float,
                                               max_decode_batch_size: int) -> dict:
    max_intervals = count_intervals(intervals, trace_window, mode="max")
    if not max_intervals:
        raise ValueError("No interval statistics produced")

    window_starts, window_ends, _, _, max_tot_servers = zip(*max_intervals)
    required_servers = np.ceil(np.asarray(max_tot_servers)).astype(int)

    return {
        "window_starts": np.asarray(window_starts, dtype=float),
        "window_ends": np.asarray(window_ends, dtype=float),
        "required_servers": required_servers,
        "trace_window": trace_window,
        "max_decode_batch_size": max_decode_batch_size,
    }


def build_required_server_trace(arrival_times_name: str,
                                requests_name: str,
                                model_name: str,
                                ttft_slo_scale: float,
                                slo_tpot: float,
                                window_start: int,
                                window_end: int,
                                trace_window: float,
                                load_scale: float = 1.0,
                                slo_constant: float = 0.0,
                                time_start: float | None = None,
                                time_end: float | None = None) -> dict:
    intervals, max_decode_batch_size = build_required_server_intervals(
        arrival_times_name=arrival_times_name,
        requests_name=requests_name,
        model_name=model_name,
        ttft_slo_scale=ttft_slo_scale,
        slo_tpot=slo_tpot,
        window_start=window_start,
        window_end=window_end,
        load_scale=load_scale,
        slo_constant=slo_constant,
        time_start=time_start,
        time_end=time_end,
    )
    return build_required_server_trace_from_intervals(intervals, trace_window, max_decode_batch_size)


def summarize_provisioning_windows(window_starts: np.ndarray,
                                   window_ends: np.ndarray,
                                   required_servers: np.ndarray,
                                   provisioning_window: float) -> list[dict]:
    if provisioning_window <= 0:
        raise ValueError(f"provisioning_window must be positive, got {provisioning_window}")
    if len(required_servers) == 0:
        return []

    coarse_groups: dict[int, list[int]] = {}
    start_time = float(window_starts[0])
    for idx, window_start in enumerate(window_starts):
        coarse_idx = int(math.floor((float(window_start) - start_time) / provisioning_window + 1e-12))
        coarse_groups.setdefault(coarse_idx, []).append(idx)

    summaries = []
    for coarse_idx in sorted(coarse_groups):
        indices = coarse_groups[coarse_idx]
        values = required_servers[indices].astype(float)
        avg = float(np.mean(values))
        p95 = float(np.percentile(values, 95))
        peak = float(np.max(values))
        if avg == 0:
            peak_to_avg = 1.0 if peak == 0 else math.inf
            p95_to_avg = 1.0 if p95 == 0 else math.inf
        else:
            peak_to_avg = peak / avg
            p95_to_avg = p95 / avg
        summaries.append({
            "window_idx": coarse_idx,
            "start": float(window_starts[indices[0]]),
            "end": float(window_ends[indices[-1]]),
            "avg": avg,
            "p95": p95,
            "peak": peak,
            "peak_to_avg": peak_to_avg,
            "p95_to_avg": p95_to_avg,
            "n_points": len(indices),
        })
    return summaries


def _plot_cdf(ax, values: np.ndarray, label: str, linestyle: str = "-"):
    sorted_values = np.sort(np.asarray(values, dtype=float))
    if len(sorted_values) == 0:
        return
    y = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    ax.step(sorted_values, y, where="post", label=label, color="black", linestyle=linestyle, linewidth=3.2)


def plot_required_server_distribution(ax, values: np.ndarray):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("No required-server values to plot")

    min_servers = int(np.min(values))
    max_servers = int(np.max(values))
    bins = np.arange(min_servers - 0.5, max_servers + 1.5, 1.0)
    weights = np.full(len(values), 1.0 / len(values), dtype=float)

    avg = float(np.mean(values))
    p95 = float(np.percentile(values, 95))
    p99 = float(np.percentile(values, 99))

    ax.hist(
        values,
        bins=bins,
        weights=weights,
        color="0.2",
        alpha=0.88,
        edgecolor="white",
        linewidth=1.0,
    )
    ax.axvline(avg, color="royalblue", linestyle="-", linewidth=2.8, label=f"Avg {avg:.2f}")
    ax.axvline(p95, color="darkorange", linestyle="--", linewidth=2.8, label=f"P95 {p95:.2f}")
    ax.axvline(p99, color="crimson", linestyle=":", linewidth=3.0, label=f"P99 {p99:.2f}")
    ax.set_xlabel("Required Servers (1s Window)")
    ax.set_ylabel("Fraction of Windows")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)


def summarize_highlight_region(window_starts: np.ndarray,
                               window_ends: np.ndarray,
                               required_servers: np.ndarray,
                               highlight_start: float,
                               highlight_end: float) -> dict:
    if highlight_end <= highlight_start:
        raise ValueError("highlight_end must be greater than highlight_start")

    mask = (window_ends > highlight_start) & (window_starts < highlight_end)
    if not np.any(mask):
        raise ValueError("Highlight interval does not overlap the trace horizon")

    values = required_servers[mask].astype(float)
    avg = float(np.mean(values))
    p95 = float(np.percentile(values, 95))
    peak = float(np.max(values))
    peak_to_avg = peak / avg if avg > 0 else math.inf
    p95_to_avg = p95 / avg if avg > 0 else math.inf
    return {
        "start": float(highlight_start),
        "end": float(highlight_end),
        "avg": avg,
        "p95": p95,
        "peak": peak,
        "peak_to_avg": peak_to_avg,
        "p95_to_avg": p95_to_avg,
    }


def plot_headroom_figure(trace: dict,
                         provisioning_summaries: list[dict],
                         distribution_trace: dict,
                         output_path: str,
                         title: str,
                         highlight_start: float | None = None,
                         highlight_end: float | None = None):
    if not provisioning_summaries:
        raise ValueError("No provisioning summaries to plot")

    representative = max(
        provisioning_summaries,
        key=lambda item: (item["peak_to_avg"], item["peak"], item["avg"]),
    )

    window_starts = trace["window_starts"]
    window_ends = trace["window_ends"]
    required_servers = trace["required_servers"]

    highlight_summary = representative
    if highlight_start is not None and highlight_end is not None:
        highlight_summary = summarize_highlight_region(
            window_starts,
            window_ends,
            required_servers,
            highlight_start,
            highlight_end,
        )

    fig, (ax_ts, ax_dist) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [2.2, 1.15]},
        constrained_layout=True,
    )

    step_x = np.append(window_starts, window_ends[-1])
    step_y = np.append(required_servers, required_servers[-1])
    ax_ts.step(step_x, step_y, where="post", color="black", linewidth=2.2, label="Required servers")
    ax_ts.axvspan(highlight_summary["start"], highlight_summary["end"], color="gold", alpha=0.16, zorder=0)
    ax_ts.hlines(
        highlight_summary["peak"],
        highlight_summary["start"],
        highlight_summary["end"],
        color="red",
        linewidth=3.0,
        label="Window peak",
    )
    ax_ts.hlines(
        highlight_summary["avg"],
        highlight_summary["start"],
        highlight_summary["end"],
        color="royalblue",
        linewidth=3.0,
        linestyle="--",
        label="Window average",
    )
    y_span = max(1.0, float(np.max(required_servers)) - float(np.min(required_servers)))
    x_annot = highlight_summary["start"] + 0.05 * (highlight_summary["end"] - highlight_summary["start"])
    peak_y = min(float(np.max(required_servers)) - 0.03 * y_span, highlight_summary["peak"] + 0.08 * y_span)
    avg_y = max(
        float(np.min(required_servers)) + 0.12 * y_span,
        highlight_summary["avg"] + 0.06 * y_span,
    )
    ax_ts.text(
        x_annot,
        peak_y,
        f"Peak {highlight_summary['peak']:.0f}",
        ha="left",
        va="bottom",
        fontsize=22,
        bbox={"facecolor": "white", "edgecolor": "0.35", "linewidth": 1.2, "alpha": 0.96},
    )
    ax_ts.text(
        x_annot,
        avg_y,
        f"Avg {highlight_summary['avg']:.2f}",
        ha="left",
        va="bottom",
        fontsize=22,
        bbox={"facecolor": "white", "edgecolor": "0.35", "linewidth": 1.2, "alpha": 0.96},
    )
    ax_ts.set_ylabel("Required Servers")
    ax_ts.set_xlabel("Time (s)")
    ax_ts.set_xlim(window_starts[0], window_ends[-1])
    ax_ts.yaxis.set_major_locator(ticker.MaxNLocator(nbins=7, integer=True))
    ax_ts.grid(True, alpha=0.25)
    ax_ts.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)

    dist_mask = (
        (distribution_trace["window_ends"] > highlight_summary["start"]) &
        (distribution_trace["window_starts"] < highlight_summary["end"])
    )
    if not np.any(dist_mask):
        raise ValueError("Highlighted interval does not overlap the 1s distribution trace")
    plot_required_server_distribution(ax_dist, distribution_trace["required_servers"][dist_mask])

    ax_ts.text(0.5, -0.20, "(a) Required Servers Over Time", transform=ax_ts.transAxes,
               ha="center", va="top", fontsize=22)
    ax_dist.text(0.5, -0.40, "(b) Distribution of Required Servers in Highlighted 1s Windows", transform=ax_dist.transAxes,
                 ha="center", va="top", fontsize=22)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def generate_motivation_figure(arrival_times_name: str,
                               requests_name: str,
                               model_name: str,
                               ttft_slo_scale: float,
                               slo_tpot: float,
                               window_start: int,
                               window_end: int,
                               trace_window: float,
                               provisioning_window: float,
                               load_scale: float = 1.0,
                               slo_constant: float = 0.0,
                               output_path: str | None = None,
                               time_start: float | None = None,
                               time_end: float | None = None,
                               highlight_start: float | None = None,
                               highlight_end: float | None = None) -> dict:
    intervals, max_decode_batch_size = build_required_server_intervals(
        arrival_times_name=arrival_times_name,
        requests_name=requests_name,
        model_name=model_name,
        ttft_slo_scale=ttft_slo_scale,
        slo_tpot=slo_tpot,
        window_start=window_start,
        window_end=window_end,
        load_scale=load_scale,
        slo_constant=slo_constant,
        time_start=time_start,
        time_end=time_end,
    )
    trace = build_required_server_trace_from_intervals(intervals, trace_window, max_decode_batch_size)
    distribution_trace = build_required_server_trace_from_intervals(intervals, 1.0, max_decode_batch_size)
    summaries = summarize_provisioning_windows(
        trace["window_starts"],
        trace["window_ends"],
        trace["required_servers"],
        provisioning_window=provisioning_window,
    )
    easy_name = get_easy_name(model_name)
    if output_path is None:
        output_path = os.path.join(
            FIGDIR,
            f"{arrival_times_name}-{requests_name}-{easy_name}-headroom.png",
        )

    plot_headroom_figure(
        trace,
        summaries,
        distribution_trace,
        output_path=output_path,
        title="",
        highlight_start=highlight_start,
        highlight_end=highlight_end,
    )

    representative = max(summaries, key=lambda item: (item["peak_to_avg"], item["peak"], item["avg"]))
    print(f"Saved {output_path}")
    print(
        "Representative window:",
        f"start={representative['start']:.1f}s",
        f"end={representative['end']:.1f}s",
        f"peak={representative['peak']:.0f}",
        f"avg={representative['avg']:.2f}",
        f"peak/avg={representative['peak_to_avg']:.2f}x",
        f"p95/avg={representative['p95_to_avg']:.2f}x",
    )
    return {
        "trace": trace,
        "distribution_trace": distribution_trace,
        "provisioning_summaries": summaries,
        "output_path": output_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot overprovisioning headroom from a required-server trace.")
    parser.add_argument("--arrival-times-name", default="azure_code_23")
    parser.add_argument("--requests-name", default="azure_code_23")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--ttft-slo-scale", type=float, default=3.0)
    parser.add_argument("--slo-tpot", type=float, default=0.05)
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument("--window-end", type=int, default=20000)
    parser.add_argument("--trace-window", type=float, default=0.3)
    parser.add_argument("--provisioning-window", type=float, default=600.0)
    parser.add_argument("--time-start", type=float, default=None)
    parser.add_argument("--time-end", type=float, default=None)
    parser.add_argument("--highlight-start", type=float, default=None)
    parser.add_argument("--highlight-end", type=float, default=None)
    parser.add_argument("--load-scale", type=float, default=1.0)
    parser.add_argument("--slo-constant", type=float, default=0.2)
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    generate_motivation_figure(
        arrival_times_name=args.arrival_times_name,
        requests_name=args.requests_name,
        model_name=args.model_name,
        ttft_slo_scale=args.ttft_slo_scale,
        slo_tpot=args.slo_tpot,
        window_start=args.window_start,
        window_end=args.window_end,
        trace_window=args.trace_window,
        provisioning_window=args.provisioning_window,
        load_scale=args.load_scale,
        slo_constant=args.slo_constant,
        output_path=args.output_path,
        time_start=args.time_start,
        time_end=args.time_end,
        highlight_start=args.highlight_start,
        highlight_end=args.highlight_end,
    )


if __name__ == "__main__":
    main()
