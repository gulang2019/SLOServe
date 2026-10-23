import argparse
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from plots.common import get_method_style, get_paper_figure_dir


DEFAULT_METADATA_DIR = Path("Paper/data/timeline")
DEFAULT_DEVICE_IDS = (0, 2, 5, 7)
SINGLE_COLUMN_WIDTH_IN = 3.35
ROW_HEIGHT_IN = 0.98
MODELED_ACTIVE_COLOR = "#6A1B9A"
MODELED_ACTIVE_TRACE_SPECS = {
    "energy_consumption-mixed": {
        "components": (
            ("azure_chat_23", "azure_chat_23"),
            ("azure_code_23", "azure_code_23"),
        ),
        "time_start_s": 1200.0,
        "time_end_s": 1800.0,
        "trace_window_s": 1.0,
        "ttft_slo_scale": 3.0,
        "slo_tpot": 0.020,
        "slo_constant": 0.2,
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
    },
    "energy_consumption-chatting": {
        "components": (
            ("azure_chat_23", "azure_chat_23"),
        ),
        "time_start_s": 1200.0,
        "time_end_s": 1800.0,
        "trace_window_s": 1.0,
        "ttft_slo_scale": 3.0,
        "slo_tpot": 0.020,
        "slo_constant": 0.2,
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
    },
}
DISPLAY_LABELS = {
    "Baseline": "vLLM",
    "Ours": "SLO-Packer",
}


def _apply_asplos_single_column_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _expand_metadata_paths(paths: list[str] | None) -> list[Path]:
    if not paths:
        paths = [str(DEFAULT_METADATA_DIR)]

    expanded_paths: list[Path] = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_dir():
            expanded_paths.extend(sorted(path.glob("*.meta.json")))
        else:
            expanded_paths.append(path)
    return expanded_paths


def _load_metadata(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label)


def _compute_modeled_active_trace(output_stem: str | Path) -> dict[str, object] | None:
    output_stem = Path(output_stem)
    spec = MODELED_ACTIVE_TRACE_SPECS.get(output_stem.name)
    if spec is None:
        return None

    from motivation.draw_loads import (
        build_required_server_intervals,
        build_required_server_trace_from_intervals,
    )

    intervals: list[tuple[float, float, float, str]] = []
    max_decode_batch_sizes: list[int] = []
    for arrival_times_name, requests_name in spec["components"]:
        component_intervals, max_decode_batch_size = build_required_server_intervals(
            arrival_times_name=arrival_times_name,
            requests_name=requests_name,
            model_name=str(spec["model_name"]),
            ttft_slo_scale=float(spec["ttft_slo_scale"]),
            slo_tpot=float(spec["slo_tpot"]),
            window_start=0,
            window_end=-1,
            slo_constant=float(spec["slo_constant"]),
            time_start=float(spec["time_start_s"]),
            time_end=float(spec["time_end_s"]),
        )
        intervals.extend(component_intervals)
        max_decode_batch_sizes.append(int(max_decode_batch_size))

    if not intervals:
        return None

    modeled_trace = build_required_server_trace_from_intervals(
        intervals,
        float(spec["trace_window_s"]),
        min(max_decode_batch_sizes),
    )
    time_start_s = float(spec["time_start_s"])
    time_axis = np.asarray(modeled_trace["window_starts"], dtype=np.float64) - time_start_s
    values = np.asarray(modeled_trace["required_servers"], dtype=np.float64)
    average_value = float(np.mean(values)) if values.size else 0.0
    return {
        "time_s": time_axis,
        "values": values,
        "average": average_value,
        "trace_label": "modeled minimum",
    }


def _plot_step_series(
    ax: plt.Axes,
    time: np.ndarray,
    values: np.ndarray,
    *,
    label: str,
    color: str,
    linewidth: float = 2.0,
    linestyle: str = "-",
    marker: str | None = None,
    markersize: float = 5.0,
) -> None:
    if time.size == 0 or values.size == 0:
        return

    plot_kwargs = {
        "where": "post",
        "label": label,
        "color": color,
        "linewidth": linewidth,
        "linestyle": linestyle,
    }
    if marker:
        plot_kwargs["marker"] = marker
        plot_kwargs["markersize"] = markersize
        plot_kwargs["markevery"] = max(1, int(time.size // 12))
    ax.step(time, values, **plot_kwargs)


def _add_axis_tag(
    ax: plt.Axes,
    text: str,
    *,
    x: float = 0.015,
    fontsize: float = 7.2,
) -> None:
    ax.text(
        x,
        0.88,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="black",
        fontsize=fontsize,
        bbox={
            "boxstyle": "round,pad=0.12",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
        },
    )


def _parse_per_device_power(power_spec: dict) -> dict[int, np.ndarray]:
    per_device = {}
    for device_id, values in power_spec.get("per_device_values_w", {}).items():
        per_device[int(device_id)] = np.asarray(values, dtype=np.float64)
    return per_device


def _derive_active_device_counts(
    per_device_power: dict[int, np.ndarray],
    num_samples: int,
    *,
    idle_power_w: float = 70.0,
) -> np.ndarray:
    counts = np.zeros(num_samples, dtype=np.float64)
    for idx in range(num_samples):
        counts[idx] = sum(
            idx < len(device_power) and float(device_power[idx]) > idle_power_w + 1e-9
            for device_power in per_device_power.values()
        )
    return counts


def _parse_active_devices(
    trace: dict,
    power_time: np.ndarray,
    per_device_power: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    active_spec = trace.get("active_devices", {})
    active_time = np.asarray(active_spec.get("time_s", []), dtype=np.float64)
    active_values = np.asarray(active_spec.get("values", []), dtype=np.float64)
    if active_time.size and active_values.size:
        return active_time, active_values

    idle_power_w = float(active_spec.get("idle_power_w", 70.0))
    return power_time, _derive_active_device_counts(
        per_device_power,
        power_time.size,
        idle_power_w=idle_power_w,
    )


def _compute_cumulative_energy(
    power_time: np.ndarray,
    total_power: np.ndarray,
    *,
    window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    if power_time.size == 0 or total_power.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    cumulative_time = np.concatenate(([0.0], power_time + window_s))
    cumulative_energy = np.concatenate(([0.0], np.cumsum(total_power * window_s)))
    return cumulative_time, cumulative_energy


def render_timeline_comparison_metadata(
    metadata: dict,
    *,
    output_stem: str | Path,
    device_ids: tuple[int, ...] = DEFAULT_DEVICE_IDS,
    xlim: tuple[float, float] = (0.0, 600.0),
    show_arrivals: bool = True,
    show_total_power: bool = True,
    show_device_power: bool = True,
    show_active_devices: bool = True,
    bottom_panel: str | None = "energy",
    row_height_in: float = ROW_HEIGHT_IN,
    min_fig_height_in: float = 4.2,
    device_panel_tags: tuple[str, ...] | None = None,
    shared_ylabel: str | None = None,
    shared_ylabel_x: float = 0.01,
    arrival_ylabel: str | None = None,
    total_power_ylabel: str | None = None,
    legend_y_anchor: float = 1.05,
    font_scale: float = 1.0,
) -> tuple[plt.Figure, np.ndarray]:
    _apply_asplos_single_column_style()
    output_stem = Path(output_stem)
    tick_labelsize = float(mpl.rcParams["xtick.labelsize"]) * font_scale
    axis_label_fontsize = float(mpl.rcParams["axes.labelsize"]) * font_scale
    legend_fontsize = float(mpl.rcParams["legend.fontsize"]) * font_scale
    axis_tag_fontsize = 7.2 * font_scale
    modeled_active_trace = (
        _compute_modeled_active_trace(output_stem)
        if bottom_panel == "modeled_active"
        else None
    )
    traces = metadata.get("traces", [])
    if not traces:
        raise ValueError("Metadata does not contain any traces.")
    if metadata.get("figure") != "timeline_comparison":
        raise ValueError(f"Unsupported metadata figure: {metadata.get('figure')}")

    legend_handles = []
    trace_data = []
    arrival_time_axis = np.array([], dtype=np.float64)
    arrival_counts = np.array([], dtype=np.float64)
    available_device_ids: set[int] = set()

    for trace in traces:
        label = str(trace["label"])
        style = get_method_style(label)
        display_label = _display_label(label)
        color = str(style["color"])
        marker = None
        markersize = 0.0
        linestyle = "-"

        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=color,
                linewidth=1.0,
                linestyle=linestyle,
                label=display_label,
            )
        )

        power_spec = trace.get("power", {})
        power_time = np.asarray(power_spec.get("time_s", []), dtype=np.float64)
        total_power = np.asarray(power_spec.get("values_w", []), dtype=np.float64)
        per_device_power = _parse_per_device_power(power_spec)
        power_window_s = float(power_spec.get("window_s", 1.0))
        active_time, active_values = _parse_active_devices(trace, power_time, per_device_power)
        cumulative_energy_time, cumulative_energy = _compute_cumulative_energy(
            power_time,
            total_power,
            window_s=power_window_s,
        )

        arrivals = trace.get("arrivals", {})
        if arrival_time_axis.size == 0:
            arrival_time_axis = np.asarray(arrivals.get("time_s", []), dtype=np.float64)
            arrival_counts = np.asarray(arrivals.get("values", []), dtype=np.float64)

        violations = trace.get("slo_violations", {})
        violation_time_axis = np.asarray(violations.get("time_s", []), dtype=np.float64)
        violation_counts = np.asarray(violations.get("values", []), dtype=np.float64)
        available_device_ids.update(per_device_power)
        trace_data.append({
            "label": display_label,
            "color": color,
            "marker": marker,
            "markersize": markersize,
            "linestyle": linestyle,
            "power_time": power_time,
            "total_power": total_power,
            "per_device_power": per_device_power,
            "active_time": active_time,
            "active_values": active_values,
            "violation_time_axis": violation_time_axis,
            "violation_counts": violation_counts,
            "cumulative_energy_time": cumulative_energy_time,
            "cumulative_energy": cumulative_energy,
        })

    selected_device_ids = [device_id for device_id in device_ids if device_id in available_device_ids]
    num_axes = 0
    if show_arrivals:
        num_axes += 1
    if show_total_power:
        num_axes += 1
    if show_device_power:
        num_axes += len(selected_device_ids)
    if show_active_devices:
        num_axes += 1
    if bottom_panel is not None:
        num_axes += 1
    if num_axes <= 0:
        raise ValueError("At least one panel must be enabled.")
    fig_height = max(min_fig_height_in, row_height_in * max(num_axes, 1) + 0.25)
    fig, axes = plt.subplots(
        num_axes,
        figsize=(SINGLE_COLUMN_WIDTH_IN, fig_height),
        sharex=True,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)
    axes = np.atleast_1d(axes)
    axis_idx = 0
    arrival_ax = None
    total_power_ax = None
    device_power_axes = []
    active_ax = None
    bottom_ax = None
    if show_arrivals:
        arrival_ax = axes[axis_idx]
        axis_idx += 1
    if show_total_power:
        total_power_ax = axes[axis_idx]
        axis_idx += 1
    if show_device_power:
        device_power_axes = list(axes[axis_idx: axis_idx + len(selected_device_ids)])
        axis_idx += len(selected_device_ids)
    if show_active_devices:
        active_ax = axes[axis_idx]
        axis_idx += 1
    if bottom_panel is not None:
        bottom_ax = axes[axis_idx]

    if arrival_ax is not None and arrival_time_axis.size:
        arrival_ax.step(arrival_time_axis, arrival_counts, where="post", color="black", linewidth=2.0)
    if arrival_ax is not None:
        arrival_ax.set_title("")
        if arrival_ylabel:
            arrival_ax.set_ylabel(arrival_ylabel, fontsize=axis_label_fontsize)
        else:
            arrival_ax.set_ylabel("")
            _add_axis_tag(arrival_ax, "Req / Win", fontsize=axis_tag_fontsize)
        arrival_ax.grid(True, alpha=0.3)

    shared_device_power_ymax = max(
        1.0,
        max(
            (
                float(np.max(device_power))
                for trace in trace_data
                for device_power in trace["per_device_power"].values()
                if len(device_power)
            ),
            default=0.0,
        ) * 1.05,
    )

    for trace in trace_data:
        if total_power_ax is not None:
            _plot_step_series(
                total_power_ax,
                trace["power_time"],
                trace["total_power"],
                label=trace["label"],
                color=trace["color"],
                linewidth=1.0,
                linestyle=trace["linestyle"],
                marker=trace["marker"],
                markersize=trace["markersize"],
            )
        if active_ax is not None:
            _plot_step_series(
                active_ax,
                trace["active_time"],
                trace["active_values"],
                label=trace["label"],
                color=trace["color"],
                linewidth=1.0,
                linestyle=trace["linestyle"],
                marker=trace["marker"],
                markersize=trace["markersize"],
            )
        if device_power_axes:
            for device_id, device_ax in zip(selected_device_ids, device_power_axes):
                device_power = trace["per_device_power"].get(device_id)
                if device_power is None:
                    continue
                _plot_step_series(
                    device_ax,
                    trace["power_time"],
                    device_power,
                    label=trace["label"],
                    color=trace["color"],
                    linewidth=0.95,
                    linestyle=trace["linestyle"],
                    marker=trace["marker"],
                    markersize=trace["markersize"],
                )

    if total_power_ax is not None:
        if total_power_ylabel:
            total_power_ax.set_ylabel(total_power_ylabel, fontsize=axis_label_fontsize)
        else:
            total_power_ax.set_ylabel("")
            _add_axis_tag(total_power_ax, "GPU Power (W)", fontsize=axis_tag_fontsize)
        total_power_ax.set_xlabel("")
        total_power_ax.grid(True, alpha=0.3)

    for panel_idx, (device_id, device_ax) in enumerate(zip(selected_device_ids, device_power_axes)):
        device_ax.set_ylabel("")
        panel_tag = f"GPU {device_id}"
        if device_panel_tags and panel_idx < len(device_panel_tags):
            panel_tag = device_panel_tags[panel_idx]
        _add_axis_tag(device_ax, panel_tag, fontsize=axis_tag_fontsize)
        device_ax.set_ylim(0, shared_device_power_ymax)
        device_ax.grid(True, alpha=0.3)

    if active_ax is not None:
        active_ax.set_ylabel("")
        _add_axis_tag(active_ax, "Measured Active GPU", fontsize=axis_tag_fontsize)
        active_ax.set_xlabel("")
        active_ax.set_ylim(bottom=0.0)
        active_ax.grid(True, alpha=0.3)

    if bottom_ax is not None and modeled_active_trace is not None:
        _plot_step_series(
            bottom_ax,
            np.asarray(modeled_active_trace["time_s"], dtype=np.float64),
            np.asarray(modeled_active_trace["values"], dtype=np.float64),
            label=str(modeled_active_trace["trace_label"]),
            color=MODELED_ACTIVE_COLOR,
            linewidth=1.2,
        )
        modeled_average = float(modeled_active_trace["average"])
        bottom_ax.axhline(
            modeled_average,
            color=MODELED_ACTIVE_COLOR,
            linestyle="--",
            linewidth=1.0,
            alpha=0.85,
        )
        bottom_ax.text(
            0.02,
            0.92,
            f"modeled min avg = {modeled_average:.2f}",
            transform=bottom_ax.transAxes,
            ha="left",
            va="top",
            color=MODELED_ACTIVE_COLOR,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "0.55",
                "linewidth": 0.7,
                "alpha": 0.94,
            },
        )
        bottom_ax.set_ylabel("")
        bottom_ax.set_xlabel("")
        bottom_ax.set_ylim(bottom=0.0)
        bottom_ax.grid(True, alpha=0.3)
    elif bottom_ax is not None and bottom_panel == "energy":
        for trace in trace_data:
            if trace["cumulative_energy_time"].size:
                bottom_ax.plot(
                    trace["cumulative_energy_time"],
                    trace["cumulative_energy"],
                    color=trace["color"],
                    linewidth=1.0,
                    linestyle=trace["linestyle"],
                )
        bottom_ax.set_ylabel("")
        _add_axis_tag(bottom_ax, "Cum. Energy (J)", fontsize=axis_tag_fontsize)
        bottom_ax.set_xlabel("")
        bottom_ax.set_ylim(bottom=0.0)
        bottom_ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlim(*xlim)
        ax.tick_params(axis="both", pad=1.5, labelsize=tick_labelsize)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
    last_ax = axes[-1]
    last_ax.set_xlabel("Time (s)", fontsize=axis_label_fontsize)
    if xlim[0] == 0.0 and xlim[1] == 600.0:
        last_ax.set_xticks([0, 200, 400, 600])

    if shared_ylabel:
        fig.supylabel(shared_ylabel, x=shared_ylabel_x, fontsize=axis_label_fontsize)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y_anchor),
        fontsize=legend_fontsize,
        frameon=False,
        handlelength=1.7,
        handletextpad=0.4,
        columnspacing=0.8,
        ncol=max(1, min(2, len(legend_handles))),
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_stem}.pdf", dpi=300, bbox_inches="tight")
    print(f"saved to {output_stem}.png")
    print(f"saved to {output_stem}.pdf")
    return fig, axes


def replay_timeline_comparisons(
    metadata_paths: list[str] | None = None,
    *,
    output_dir: str | Path | None = None,
    device_ids: tuple[int, ...] = DEFAULT_DEVICE_IDS,
    xlim: tuple[float, float] = (0.0, 600.0),
    layout: str = "full",
) -> list[Path]:
    resolved_paths = _expand_metadata_paths(metadata_paths)
    if not resolved_paths:
        raise FileNotFoundError("No timeline metadata files found.")

    if output_dir is None:
        output_dir = get_paper_figure_dir("timeline_comparison", "replay_metadata")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    show_arrivals = True
    show_total_power = True
    show_device_power = True
    show_active_devices = True
    bottom_panel: str | None = "energy"
    row_height_in = ROW_HEIGHT_IN
    min_fig_height_in = 4.2
    device_panel_tags: tuple[str, ...] | None = None
    shared_ylabel: str | None = None
    shared_ylabel_x = 0.01
    arrival_ylabel: str | None = None
    total_power_ylabel: str | None = None
    legend_y_anchor = 1.05
    font_scale = 1.0
    if layout == "intro":
        show_device_power = False
        show_active_devices = False
        bottom_panel = None
        row_height_in = 0.70
        min_fig_height_in = 2.85
        arrival_ylabel = "Request / Window"
        total_power_ylabel = "Power (W)"
        legend_y_anchor = 1.10
    elif layout == "evaluation":
        show_arrivals = False
        show_total_power = False
        show_active_devices = False
        bottom_panel = None
        row_height_in = 0.86
        min_fig_height_in = 3.55
        device_panel_tags = ("TOP", "MID", "BOTTOM")
        shared_ylabel = "Power (W)"
        shared_ylabel_x = -0.075
        legend_y_anchor = 1.10
        font_scale = 1.20
    elif layout != "full":
        raise ValueError(f"Unsupported layout: {layout}")

    for metadata_path in resolved_paths:
        metadata = _load_metadata(metadata_path)
        output_name = metadata_path.name.removesuffix(".meta.json")
        output_stem = output_dir / output_name
        fig, _ = render_timeline_comparison_metadata(
            metadata,
            output_stem=output_stem,
            device_ids=device_ids,
            xlim=xlim,
            show_arrivals=show_arrivals,
            show_total_power=show_total_power,
            show_device_power=show_device_power,
            show_active_devices=show_active_devices,
            bottom_panel=bottom_panel,
            row_height_in=row_height_in,
            min_fig_height_in=min_fig_height_in,
            device_panel_tags=device_panel_tags,
            shared_ylabel=shared_ylabel,
            shared_ylabel_x=shared_ylabel_x,
            arrival_ylabel=arrival_ylabel,
            total_power_ylabel=total_power_ylabel,
            legend_y_anchor=legend_y_anchor,
            font_scale=font_scale,
        )
        plt.close(fig)
        output_paths.append(output_stem)
    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce timeline comparison figures from exported metadata.",
    )
    parser.add_argument(
        "metadata",
        nargs="*",
        help="Metadata JSON files or directories containing *.meta.json. Defaults to Paper/data/timeline.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for rendered figures. Defaults to Paper/figs/timeline_comparison/replay_metadata.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="*",
        default=list(DEFAULT_DEVICE_IDS),
        help="Device ids to keep as per-device power panels. Missing devices are skipped.",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=0.0,
        help="Left x-limit in seconds.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=600.0,
        help="Right x-limit in seconds.",
    )
    parser.add_argument(
        "--layout",
        choices=("full", "intro", "evaluation"),
        default="full",
        help="Figure layout to render. Default: %(default)s",
    )
    args = parser.parse_args()
    replay_timeline_comparisons(
        args.metadata,
        output_dir=args.output_dir,
        device_ids=tuple(args.devices),
        xlim=(args.x_min, args.x_max),
        layout=args.layout,
    )


if __name__ == "__main__":
    main()
