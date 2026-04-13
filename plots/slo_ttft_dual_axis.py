from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

try:
    from plots.common import get_method_label, get_method_style
except ModuleNotFoundError:
    from common import get_method_label, get_method_style


DEFAULT_SWEEP_FIELD = "ttft_slo_scale"
DEFAULT_N_DEVICE = 4
SLO_AXIS_COLOR = "#8B1E3F"
ENERGY_AXIS_COLOR = "#1F4E79"
SLO_LINESTYLE = "-"
ENERGY_LINESTYLE = "--"
PAPER_FIGSIZE = (5.4, 4.4)
PERF_MODEL_ERR_EXCLUDED_METHODS = {"atfc / round_robin"}


SWEEP_CONFIGS = {
    "ttft_slo_scale": {
        "default_csv": Path("Paper/data/slo_ttft.csv"),
        "default_output_stem": Path("plots/e2e/slo_ttft_dual_axis"),
        "xlabel": "TTFT SLO Scale",
        "constant_fields": ("load_scale", "slo_tpot", "perf_model_err"),
    },
    "slo_tpot": {
        "default_csv": Path("Paper/data/slo_tpot.csv"),
        "default_output_stem": Path("plots/e2e/slo_tpot_dual_axis"),
        "xlabel": "TPOT SLO",
        "constant_fields": ("load_scale", "ttft_slo_scale", "perf_model_err"),
    },
    "perf_model_err": {
        "default_csv": Path("Paper/data/perf_model_err.csv"),
        "default_output_stem": Path("plots/e2e/perf_model_err_dual_axis"),
        "xlabel": "Performance Model Error",
        "constant_fields": ("load_scale", "ttft_slo_scale", "slo_tpot"),
        "reference_lines": {
            "slo_violation_rate": {
                "value": 14.0,
                "label": "Baseline SLO vio = 14%",
                "color": "black",
                "linestyle": "-.",
                "label_position": "below",
            },
            "energy_consumption": {
                "value": 800.0,
                "label": "Baseline energy = 800 kJ",
                "color": "black",
                "linestyle": ":",
                "label_position": "below",
            },
        },
    },
}


def _get_plotting_dependencies():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import ScalarFormatter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to render this figure. "
            "Install it first, then rerun the script."
        ) from exc
    return plt, Line2D, ScalarFormatter


def _apply_paper_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 14,
            "mathtext.fontset": "stix",
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _parse_required_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value == "":
        raise ValueError(f"Missing required field '{field}' in row: {row}")
    return float(value)


def _get_sweep_config(sweep_field: str) -> dict[str, object]:
    try:
        return SWEEP_CONFIGS[sweep_field]
    except KeyError as exc:
        supported = ", ".join(sorted(SWEEP_CONFIGS))
        raise ValueError(
            f"Unsupported sweep field '{sweep_field}'. Expected one of: {supported}"
        ) from exc


def _load_rows(
    csv_path: Path | str,
    sweep_field: str,
) -> tuple[list[dict[str, float | int | str]], int]:
    csv_path = Path(csv_path)
    sweep_config = _get_sweep_config(sweep_field)
    required_fields = {
        "scheduling_policy",
        "routing_policy",
        "n_device",
        sweep_field,
        "slo_violation_rate",
        "energy_consumption",
    }

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = required_fields - fieldnames
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(sorted(missing))}"
            )

        deduped_rows: OrderedDict[tuple[str, int, float], dict[str, float | int | str]] = OrderedDict()
        constant_context: dict[str, set[str]] = {
            field: set() for field in sweep_config["constant_fields"]
        }
        duplicate_count = 0
        for row in reader:
            for field in constant_context:
                value = row.get(field, "")
                if value != "":
                    constant_context[field].add(value)

            method = f"{row['scheduling_policy']} / {row['routing_policy']}"
            parsed_row = {
                "method": method,
                "n_device": int(_parse_required_float(row, "n_device")),
                "sweep_value": _parse_required_float(row, sweep_field),
                "slo_violation_rate": _parse_required_float(row, "slo_violation_rate") * 100.0,
                "energy_consumption": _parse_required_float(row, "energy_consumption") / 1000.0,
            }
            # Keep the latest point when the CSV appends corrected reruns for the same sweep x-value.
            dedupe_key = (
                parsed_row["method"],
                parsed_row["n_device"],
                parsed_row["sweep_value"],
            )
            if dedupe_key in deduped_rows:
                duplicate_count += 1
            deduped_rows[dedupe_key] = parsed_row

    varying_context = {
        field: sorted(values)
        for field, values in constant_context.items()
        if len(values) > 1
    }
    if varying_context:
        details = ", ".join(
            f"{field}={values}" for field, values in varying_context.items()
        )
        raise ValueError(
            f"This plot expects {sweep_field} to be the only swept variable. "
            f"Found additional varying context in {csv_path}: {details}"
        )

    return list(deduped_rows.values()), duplicate_count


def _build_panels(
    rows: list[dict[str, float | int | str]],
) -> tuple[dict[int, OrderedDict[str, list[dict[str, float | int | str]]]], list[str]]:
    panels: dict[int, OrderedDict[str, list[dict[str, float | int | str]]]] = OrderedDict()
    method_order: OrderedDict[str, None] = OrderedDict()

    for row in rows:
        method = str(row["method"])
        n_device = int(row["n_device"])
        method_order.setdefault(method, None)
        panel = panels.setdefault(n_device, OrderedDict())
        panel.setdefault(method, []).append(row)

    for panel in panels.values():
        for points in panel.values():
            points.sort(key=lambda point: float(point["sweep_value"]))

    return panels, list(method_order.keys())


def _filter_rows_by_n_device(
    rows: list[dict[str, float | int | str]],
    n_device: int,
) -> tuple[list[dict[str, float | int | str]], list[int]]:
    available_n_devices = sorted({int(row["n_device"]) for row in rows})
    filtered_rows = [row for row in rows if int(row["n_device"]) == n_device]
    return filtered_rows, available_n_devices


def _filter_rows_for_sweep(
    rows: list[dict[str, float | int | str]],
    sweep_field: str,
) -> list[dict[str, float | int | str]]:
    if sweep_field != "perf_model_err":
        return rows
    return [
        row
        for row in rows
        if str(row["method"]) not in PERF_MODEL_ERR_EXCLUDED_METHODS
    ]


def _get_plot_style(
    method: str,
    sweep_field: str,
) -> dict[str, object]:
    return dict(get_method_style(method))


def _annotate_reference_line(
    ax,
    value: float,
    label: str,
    color: str,
    axis_upper: float,
    place_above: bool,
) -> None:
    if axis_upper <= 0:
        axis_upper = 1.0
    y_offset = 4 if place_above else -4
    vertical_alignment = "bottom" if place_above else "top"
    ax.annotate(
        label,
        xy=(0.955, value),
        xycoords=("axes fraction", "data"),
        xytext=(-6, y_offset),
        textcoords="offset points",
        ha="right",
        va=vertical_alignment,
        color=color,
        fontsize=12,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
        },
        zorder=4,
    )


def _configure_x_axis(
    ax,
    sweep_field: str,
    xticks: list[float],
) -> None:
    if not xticks:
        return

    unique_xticks = sorted(set(xticks))
    if sweep_field == "slo_tpot":
        if any(tick <= 0 for tick in unique_xticks):
            raise ValueError("slo_tpot must be strictly positive to use a log-scaled x-axis.")
        ax.set_xscale("log")
        ax.set_xlim(unique_xticks[0] * 0.92, unique_xticks[-1] * 1.08)
    ax.set_xticks(unique_xticks)
    ax.set_xticklabels([f"{tick:g}" for tick in unique_xticks])
    if sweep_field == "slo_tpot":
        ax.tick_params(axis="x", labelrotation=90)


def draw_slo_dual_axis(
    sweep_field: str = DEFAULT_SWEEP_FIELD,
    csv_path: Path | str | None = None,
    output_stem: Path | str | None = None,
    n_device: int = DEFAULT_N_DEVICE,
) -> list[Path]:
    sweep_config = _get_sweep_config(sweep_field)
    if csv_path is None:
        csv_path = sweep_config["default_csv"]
    if output_stem is None:
        output_stem = Path(f"{sweep_config['default_output_stem']}_n{n_device}")
    csv_path = Path(csv_path)
    output_stem = Path(output_stem)

    plt, Line2D, ScalarFormatter = _get_plotting_dependencies()
    _apply_paper_style(plt)
    rows, duplicate_count = _load_rows(csv_path, sweep_field=sweep_field)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    rows, available_n_devices = _filter_rows_by_n_device(rows, n_device=n_device)
    if not rows:
        raise ValueError(
            f"No rows found in {csv_path} for n_device={n_device}. "
            f"Available values: {available_n_devices}"
        )
    rows = _filter_rows_for_sweep(rows, sweep_field=sweep_field)
    if not rows:
        raise ValueError(
            f"No rows left to plot in {csv_path} for n_device={n_device} "
            f"after applying the {sweep_field} method filter."
        )

    panels, method_order = _build_panels(rows)
    panel = panels[n_device]
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    energy_ax = ax.twinx()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    energy_ax.set_facecolor("none")

    comparator_handles = []
    seen_methods: set[str] = set()
    xticks: list[float] = []
    all_violations: list[float] = []
    all_energies: list[float] = []
    for method in method_order:
        points = panel.get(method)
        if not points:
            continue

        style = _get_plot_style(method, sweep_field=sweep_field)
        label = get_method_label(method)
        xs = [float(point["sweep_value"]) for point in points]
        violations = [float(point["slo_violation_rate"]) for point in points]
        energies = [float(point["energy_consumption"]) for point in points]
        xticks.extend(xs)
        all_violations.extend(violations)
        all_energies.extend(energies)

        ax.plot(
            xs,
            violations,
            color=style["color"],
            marker=style["marker"],
            linewidth=max(2.2, style["linewidth"] + 0.4),
            markersize=style["markersize"] + 0.8,
            linestyle=SLO_LINESTYLE,
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            markeredgewidth=0.8,
            zorder=3,
            label=f"{label} - SLO violation",
        )
        energy_ax.plot(
            xs,
            energies,
            color=style["color"],
            marker=style["marker"],
            linewidth=max(1.9, style["linewidth"]),
            markersize=style["markersize"] + 1.1,
            linestyle=ENERGY_LINESTYLE,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=1.8,
            alpha=0.95,
            zorder=2,
            label=f"{label} - Energy consumption",
        )

        if method not in seen_methods:
            comparator_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=2.2,
                    markersize=6.8,
                    label=label,
                )
            )
            seen_methods.add(method)

    ax.set_xlabel(str(sweep_config["xlabel"]))
    ax.set_ylabel("SLO Violation Rate (%)", color="black")
    energy_ax.set_ylabel("Energy Consumption (kJ)", color="black")
    ax.tick_params(axis="x", colors="black", pad=2)
    ax.tick_params(axis="y", colors="black", pad=2)
    energy_ax.tick_params(axis="y", colors="black", pad=2)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    energy_ax.yaxis.label.set_color("black")
    for spine_name in ("top", "bottom", "left"):
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("black")
        ax.spines[spine_name].set_linewidth(0.9)
    energy_ax.spines["right"].set_visible(True)
    energy_ax.spines["right"].set_color("black")
    energy_ax.spines["right"].set_linewidth(0.9)
    energy_ax.spines["top"].set_visible(False)
    energy_ax.spines["left"].set_visible(False)
    energy_ax.spines["bottom"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, linestyle=(0, (2, 2)))
    energy_formatter = ScalarFormatter(useMathText=True)
    energy_formatter.set_scientific(False)
    energy_ax.yaxis.set_major_formatter(energy_formatter)
    ax.margins(x=0.03)

    reference_lines = sweep_config.get("reference_lines", {})
    slo_reference_values = []
    energy_reference_values = []
    if "slo_violation_rate" in reference_lines:
        slo_ref = reference_lines["slo_violation_rate"]
        slo_reference_values.append(float(slo_ref["value"]))
        ax.axhline(
            float(slo_ref["value"]),
            color=str(slo_ref["color"]),
            linestyle=str(slo_ref["linestyle"]),
            linewidth=1.8,
            alpha=0.95,
            zorder=1,
        )
    if "energy_consumption" in reference_lines:
        energy_ref = reference_lines["energy_consumption"]
        energy_reference_values.append(float(energy_ref["value"]))
        energy_ax.axhline(
            float(energy_ref["value"]),
            color=str(energy_ref["color"]),
            linestyle=str(energy_ref["linestyle"]),
            linewidth=1.8,
            alpha=0.95,
            zorder=1,
        )

    slo_upper = max(all_violations + slo_reference_values) if (all_violations or slo_reference_values) else 1.0
    energy_upper = max(all_energies + energy_reference_values) if (all_energies or energy_reference_values) else 1.0
    ax.set_ylim(0, slo_upper * 1.08)
    energy_ax.set_ylim(0, energy_upper * 1.08)
    if "slo_violation_rate" in reference_lines:
        slo_ref = reference_lines["slo_violation_rate"]
        _annotate_reference_line(
            ax,
            value=float(slo_ref["value"]),
            label=str(slo_ref["label"]),
            color=str(slo_ref["color"]),
            axis_upper=float(ax.get_ylim()[1]),
            place_above=str(slo_ref.get("label_position", "above")) == "above",
        )
    if "energy_consumption" in reference_lines:
        energy_ref = reference_lines["energy_consumption"]
        _annotate_reference_line(
            energy_ax,
            value=float(energy_ref["value"]),
            label=str(energy_ref["label"]),
            color=str(energy_ref["color"]),
            axis_upper=float(energy_ax.get_ylim()[1]),
            place_above=str(energy_ref.get("label_position", "above")) == "above",
        )

    _configure_x_axis(ax, sweep_field=sweep_field, xticks=xticks)

    metric_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=SLO_LINESTYLE,
            marker="o",
            linewidth=2.2,
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=6.5,
            label="SLO violation (solid, filled, left axis)",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=ENERGY_LINESTYLE,
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.6,
            linewidth=2.0,
            markersize=6.8,
            label="Energy (dashed, hollow, right axis)",
        ),
    ]

    if sweep_field != "perf_model_err":
        comparator_legend = fig.legend(
            handles=comparator_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.975),
            ncol=max(1, min(3, len(comparator_handles))),
            frameon=False,
            handlelength=2.0,
            columnspacing=1.1,
            handletextpad=0.6,
        )
        fig.add_artist(comparator_legend)
    fig.legend(
        handles=metric_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=2,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.4,
        handletextpad=0.6,
    )
    top_rect = 0.965 if sweep_field == "perf_model_err" else 0.885
    fig.tight_layout(rect=(0.0, 0.09, 1.0, top_rect))

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for suffix in (".png", ".pdf"):
        output_path = output_stem.with_suffix(suffix)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        saved_paths.append(output_path)
    plt.close(fig)

    if duplicate_count:
        print(
            f"Retained the last CSV row for {duplicate_count} duplicate "
            f"(method, n_device, {sweep_field}) points."
        )
    for output_path in saved_paths:
        print(f"Saved {output_path}")
    return saved_paths


def draw_slo_ttft_dual_axis(
    csv_path: Path | str | None = None,
    output_stem: Path | str | None = None,
    n_device: int = DEFAULT_N_DEVICE,
) -> list[Path]:
    return draw_slo_dual_axis(
        sweep_field="ttft_slo_scale",
        csv_path=csv_path,
        output_stem=output_stem,
        n_device=n_device,
    )


def draw_slo_tpot_dual_axis(
    csv_path: Path | str | None = None,
    output_stem: Path | str | None = None,
    n_device: int = DEFAULT_N_DEVICE,
) -> list[Path]:
    return draw_slo_dual_axis(
        sweep_field="slo_tpot",
        csv_path=csv_path,
        output_stem=output_stem,
        n_device=n_device,
    )


def draw_perf_model_err_dual_axis(
    csv_path: Path | str | None = None,
    output_stem: Path | str | None = None,
    n_device: int = DEFAULT_N_DEVICE,
) -> list[Path]:
    return draw_slo_dual_axis(
        sweep_field="perf_model_err",
        csv_path=csv_path,
        output_stem=output_stem,
        n_device=n_device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot SLO sweeps with the chosen SLO scale on x, "
            "SLO violation on the left axis, and energy consumption on the right axis."
        )
    )
    parser.add_argument(
        "--sweep-field",
        choices=sorted(SWEEP_CONFIGS),
        default=DEFAULT_SWEEP_FIELD,
        help="Sweep column to plot on the x-axis. Default: %(default)s",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Input CSV file. Defaults to the CSV associated with --sweep-field.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        help="Output path without extension. Defaults to a sweep-specific path in plots/e2e/.",
    )
    parser.add_argument(
        "--n-device",
        type=int,
        default=DEFAULT_N_DEVICE,
        help="Plot a single figure for this n_device value. Default: %(default)s",
    )
    args = parser.parse_args()

    draw_slo_dual_axis(
        sweep_field=args.sweep_field,
        csv_path=args.csv,
        output_stem=args.output_stem,
        n_device=args.n_device,
    )


if __name__ == "__main__":
    main()
