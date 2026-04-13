from __future__ import annotations

import argparse
import ast
import csv
import re
from collections import OrderedDict
from pathlib import Path

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


DEFAULT_INPUT_DIR = Path("Paper/data")
DEFAULT_GLOB_PATTERN = "sensitivity_load_scale*.csv"
DEFAULT_CSV_PATH = DEFAULT_INPUT_DIR / "sensitivity_load_scale_chat23.csv"
DEFAULT_DISAGG_COMPONENT_CSV = DEFAULT_INPUT_DIR / "disagg_energy_profiling.csv"
DEFAULT_OUTPUT_DIR = Path("Paper/figs")
DEFAULT_POWER_AGGREGATION = "sum"
FIGSIZE = (7.6, 5.6)
CONSTANT_FIELDS = ("n_device", "ttft_slo_scale", "slo_tpot", "perf_model_err")
IDLE_POWER_W = 70.0
MAX_POWER_W = 400.0
SERIES_LINEWIDTH = 3.2
SERIES_MARKERSIZE = 8.5
REFERENCE_LINEWIDTH = 2.2
INCLUDED_METHODS = {"vllm / round_robin"}
AUTO_DISAGG_COMPONENT_CSVS = {
    "sensitivity_load_scale_chat23": DEFAULT_DISAGG_COMPONENT_CSV,
}
DISAGG_COMPONENT_CURVES = (
    {
        "method": "prefill_only",
        "label": "Prefill Only",
        "component_index": 0,
        "color": "#2A9D8F",
        "marker": "o",
        "linestyle": "--",
    },
    {
        "method": "decode_only",
        "label": "Decode Only",
        "component_index": 1,
        "color": "#F4A261",
        "marker": "D",
        "linestyle": "-.",
    },
)
LOCAL_LABEL_MAP = {
    "vllm / round_robin": "Colocated",
    **{
        str(curve["method"]): str(curve["label"]) for curve in DISAGG_COMPONENT_CURVES
    },
}
LOCAL_STYLE_MAP = {
    str(curve["method"]): {
        "color": str(curve["color"]),
        "marker": str(curve["marker"]),
        "linestyle": str(curve["linestyle"]),
        "linewidth": SERIES_LINEWIDTH,
        "markersize": SERIES_MARKERSIZE,
    }
    for curve in DISAGG_COMPONENT_CURVES
}
DATASET_LABEL_MAP = {
    "sensitivity_load_scale_chat23": "Qwen7B-Chat",
    "sensitivity_load_scale_code23": "Qwen7B-Code",
}


def _get_plotting_dependencies():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import FuncFormatter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to render this figure. "
            "Install it first, then rerun the script."
        ) from exc
    return plt, Line2D, FuncFormatter


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
            "legend.fontsize": 13,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _parse_required_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value == "":
        raise ValueError(f"Missing required field '{field}' in row: {row}")
    return float(value)


def _parse_per_server_power(raw_value: str) -> list[float] | None:
    if raw_value == "":
        return None
    try:
        parsed = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Failed to parse per_server_power={raw_value!r}") from exc
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(
            f"per_server_power must parse to a list or tuple, got {type(parsed).__name__}"
        )
    return [float(value) for value in parsed]


def _aggregate_power(per_server_power: list[float], aggregation: str) -> float | None:
    if not per_server_power:
        return None
    if aggregation == "sum":
        return float(sum(per_server_power))
    if aggregation == "mean":
        return float(sum(per_server_power) / len(per_server_power))
    if aggregation == "max":
        return float(max(per_server_power))
    raise ValueError(f"Unsupported power aggregation: {aggregation}")


def _get_method_label(method: str) -> str:
    return LOCAL_LABEL_MAP.get(method, _common_get_method_label(method))


def _get_method_style(method: str) -> dict[str, object]:
    if method in LOCAL_STYLE_MAP:
        return dict(LOCAL_STYLE_MAP[method])
    style = dict(_common_get_method_style(method))
    style.setdefault("linestyle", "-")
    style["linewidth"] = max(SERIES_LINEWIDTH, float(style.get("linewidth", 0.0)))
    style["markersize"] = max(SERIES_MARKERSIZE, float(style.get("markersize", 0.0)))
    return style


def _dataset_label(csv_path: Path) -> str:
    if csv_path.stem in DATASET_LABEL_MAP:
        return DATASET_LABEL_MAP[csv_path.stem]
    stem = csv_path.stem
    if stem.startswith("sensitivity_load_scale_"):
        stem = stem.removeprefix("sensitivity_load_scale_")
    return stem.replace("_", " ")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return slug or "figure"


def _default_output_stem(
    csv_path: Path,
    power_aggregation: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    label = _slugify(_dataset_label(csv_path))
    base_name = f"power_vs_rps-{label}"
    if power_aggregation != DEFAULT_POWER_AGGREGATION:
        base_name = f"{base_name}_{power_aggregation}"
    return output_dir / base_name


def _power_axis_label(power_aggregation: str) -> str:
    if power_aggregation == "sum":
        return "Power (W)"
    if power_aggregation == "mean":
        return "Mean Per-Server Power (W)"
    if power_aggregation == "max":
        return "Max Per-Server Power (W)"
    raise ValueError(f"Unsupported power aggregation: {power_aggregation}")


def _iter_csv_files(
    input_dir: Path | str,
    glob_pattern: str = DEFAULT_GLOB_PATTERN,
) -> list[Path]:
    input_dir = Path(input_dir)
    csv_paths = sorted(path for path in input_dir.glob(glob_pattern) if path.is_file())
    if not csv_paths:
        raise ValueError(f"No CSV files matching {glob_pattern!r} found in {input_dir}")
    return csv_paths


def _validate_constant_context(
    csv_path: Path,
    constant_context: dict[str, set[str]],
) -> None:
    varying_context = {
        field: sorted(values)
        for field, values in constant_context.items()
        if len(values) > 1
    }
    if varying_context:
        details = ", ".join(f"{field}={values}" for field, values in varying_context.items())
        raise ValueError(
            "This figure expects RPS/load_scale to be the only swept variable. "
            f"Found additional varying context in {csv_path}: {details}"
        )


def _load_rows(
    csv_path: Path | str,
    power_aggregation: str,
) -> tuple[list[dict[str, float | str]], int, int, int]:
    csv_path = Path(csv_path)
    required_fields = {
        "scheduling_policy",
        "routing_policy",
        "rps",
        "per_server_power",
    }

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = required_fields - fieldnames
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(sorted(missing))}"
            )

        constant_context: dict[str, set[str]] = {
            field: set() for field in CONSTANT_FIELDS if field in fieldnames
        }
        deduped_rows: OrderedDict[tuple[str, float], dict[str, float | str]] = OrderedDict()
        duplicate_count = 0
        skipped_missing_power = 0
        skipped_zero_power = 0

        for row in reader:
            for field in constant_context:
                value = row.get(field, "")
                if value != "":
                    constant_context[field].add(value)

            per_server_power = _parse_per_server_power(row.get("per_server_power", ""))
            if per_server_power is None:
                skipped_missing_power += 1
                continue

            total_power = float(sum(per_server_power))
            if total_power <= 0:
                skipped_zero_power += 1
                continue

            aggregated_power = _aggregate_power(per_server_power, power_aggregation)
            if aggregated_power is None:
                skipped_missing_power += 1
                continue

            method = f"{row['scheduling_policy']} / {row['routing_policy']}"
            parsed_row = {
                "method": method,
                "x_value": _parse_required_float(row, "rps"),
                "power": aggregated_power,
            }
            dedupe_key = (method, float(parsed_row["x_value"]))
            if dedupe_key in deduped_rows:
                duplicate_count += 1
            deduped_rows[dedupe_key] = parsed_row

    _validate_constant_context(csv_path, constant_context)

    rows = list(deduped_rows.values())
    if not rows:
        raise ValueError(f"No rows with usable per_server_power found in {csv_path}.")
    return rows, duplicate_count, skipped_missing_power, skipped_zero_power


def _load_disagg_component_rows(
    csv_path: Path | str,
) -> tuple[list[dict[str, float | str]], int, int, int]:
    csv_path = Path(csv_path)
    required_fields = {"rps", "per_server_power"}

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = required_fields - fieldnames
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(sorted(missing))}"
            )

        constant_context: dict[str, set[str]] = {
            field: set() for field in CONSTANT_FIELDS if field in fieldnames
        }
        deduped_rows: OrderedDict[tuple[str, float], dict[str, float | str]] = OrderedDict()
        duplicate_count = 0
        skipped_missing_power = 0
        skipped_zero_power = 0

        for row in reader:
            for field in constant_context:
                value = row.get(field, "")
                if value != "":
                    constant_context[field].add(value)

            per_server_power = _parse_per_server_power(row.get("per_server_power", ""))
            if per_server_power is None:
                skipped_missing_power += len(DISAGG_COMPONENT_CURVES)
                continue

            x_value = _parse_required_float(row, "rps")
            for curve in DISAGG_COMPONENT_CURVES:
                component_index = int(curve["component_index"])
                if component_index >= len(per_server_power):
                    skipped_missing_power += 1
                    continue

                component_power = float(per_server_power[component_index])
                if component_power <= 0:
                    skipped_zero_power += 1
                    continue

                method = str(curve["method"])
                parsed_row = {
                    "method": method,
                    "x_value": x_value,
                    "power": component_power,
                }
                dedupe_key = (method, x_value)
                if dedupe_key in deduped_rows:
                    duplicate_count += 1
                deduped_rows[dedupe_key] = parsed_row

    _validate_constant_context(csv_path, constant_context)

    rows = list(deduped_rows.values())
    if not rows:
        raise ValueError(f"No usable disaggregated component rows found in {csv_path}.")
    return rows, duplicate_count, skipped_missing_power, skipped_zero_power


def _build_method_series(
    rows: list[dict[str, float | str]],
) -> tuple[dict[str, list[dict[str, float | str]]], list[str]]:
    series_by_method: dict[str, list[dict[str, float | str]]] = OrderedDict()
    method_order: OrderedDict[str, None] = OrderedDict()

    for row in rows:
        method = str(row["method"])
        method_order.setdefault(method, None)
        series_by_method.setdefault(method, []).append(row)

    for points in series_by_method.values():
        points.sort(key=lambda point: float(point["x_value"]))

    return series_by_method, list(method_order.keys())


def _filter_methods(
    series_by_method: dict[str, list[dict[str, float | str]]],
    method_order: list[str],
) -> tuple[dict[str, list[dict[str, float | str]]], list[str]]:
    filtered_method_order = [
        method for method in method_order if method in INCLUDED_METHODS
    ]
    filtered_series_by_method = {
        method: series_by_method[method]
        for method in filtered_method_order
        if method in series_by_method
    }
    return filtered_series_by_method, filtered_method_order


def _merge_method_series(
    series_by_method: dict[str, list[dict[str, float | str]]],
    method_order: list[str],
    extra_rows: list[dict[str, float | str]],
) -> tuple[dict[str, list[dict[str, float | str]]], list[str]]:
    extra_series_by_method, extra_method_order = _build_method_series(extra_rows)
    merged_series_by_method = OrderedDict(series_by_method)
    merged_method_order = list(method_order)

    for method in extra_method_order:
        merged_series_by_method[method] = extra_series_by_method[method]
        if method not in merged_method_order:
            merged_method_order.append(method)
    return merged_series_by_method, merged_method_order


def _resolve_disagg_component_csv(
    csv_path: Path,
    disagg_component_csv: Path | str | None,
    *,
    include_disagg_components: bool,
) -> Path | None:
    if not include_disagg_components:
        return None
    if disagg_component_csv is not None:
        return Path(disagg_component_csv)
    auto_csv = AUTO_DISAGG_COMPONENT_CSVS.get(csv_path.stem)
    return Path(auto_csv) if auto_csv is not None else None


def _method_serving_mode(method: str) -> str:
    return "disagg" if "disagg" in method else "normal"


def _split_methods_by_serving_mode(
    series_by_method: dict[str, list[dict[str, float | str]]],
    method_order: list[str],
) -> OrderedDict[str, tuple[dict[str, list[dict[str, float | str]]], list[str]]]:
    grouped: OrderedDict[str, tuple[dict[str, list[dict[str, float | str]]], list[str]]] = OrderedDict()
    for serving_mode in ("normal", "disagg"):
        mode_methods = [method for method in method_order if _method_serving_mode(method) == serving_mode]
        if not mode_methods:
            continue
        grouped[serving_mode] = (
            {method: series_by_method[method] for method in mode_methods},
            mode_methods,
        )
    return grouped


def _style_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.tick_params(axis="both", colors="black", pad=2)
    for spine_name in ("top", "bottom", "left", "right"):
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("black")
        ax.spines[spine_name].set_linewidth(1.2)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.9, linestyle=(0, (2, 2)))
    ax.set_axisbelow(True)


def _annotate_reference_line(
    ax,
    value: float,
    label: str,
    color: str,
    *,
    place_above: bool,
) -> None:
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


def _plot_power_vs_rps(
    ax,
    series_by_method: dict[str, list[dict[str, float | str]]],
    method_order: list[str],
) -> None:
    for method in method_order:
        points = series_by_method.get(method)
        if not points:
            continue

        style = _get_method_style(method)
        xs = [float(point["x_value"]) for point in points]
        ys = [float(point["power"]) for point in points]
        if not xs:
            continue

        ax.plot(
            xs,
            ys,
            color=style["color"],
            marker=style["marker"],
            linestyle=str(style.get("linestyle", "-")),
            linewidth=max(SERIES_LINEWIDTH, float(style["linewidth"])),
            markersize=max(SERIES_MARKERSIZE, float(style["markersize"])),
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            markeredgewidth=1.0,
        )


def _output_stem_for_serving_mode(
    output_stem: Path,
    serving_mode: str,
    split_count: int,
) -> Path:
    if split_count <= 1:
        return output_stem
    return output_stem.with_name(f"{output_stem.name}_{serving_mode}")


def _render_figure(
    *,
    plt,
    line2d_cls,
    func_formatter_cls,
    output_stem: Path,
    series_by_method: dict[str, list[dict[str, float | str]]],
    method_order: list[str],
    title: str,
    y_label: str,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")

    _plot_power_vs_rps(ax, series_by_method, method_order)
    ax.set_xlabel("RPS", color="black")
    ax.set_ylabel(y_label, color="black")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    formatter = func_formatter_cls(lambda value, _pos: f"{value:g}")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    _style_axis(ax)
    ax.axhline(
        IDLE_POWER_W,
        color="black",
        linestyle=":",
        linewidth=REFERENCE_LINEWIDTH,
        alpha=0.95,
        zorder=1,
    )
    _annotate_reference_line(
        ax,
        value=IDLE_POWER_W,
        label="Idle power = 70 W",
        color="black",
        place_above=True,
    )
    ax.axhline(
        MAX_POWER_W,
        color="#555555",
        linestyle="--",
        linewidth=REFERENCE_LINEWIDTH,
        alpha=0.95,
        zorder=1,
    )
    _annotate_reference_line(
        ax,
        value=MAX_POWER_W,
        label="Maximum power = 400 W",
        color="#555555",
        place_above=False,
    )
    ax.set_ylim(top=max(ax.get_ylim()[1], MAX_POWER_W + 20.0))

    legend_handles = []
    for method in method_order:
        style = _get_method_style(method)
        legend_handles.append(
            line2d_cls(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linestyle=str(style.get("linestyle", "-")),
                linewidth=max(SERIES_LINEWIDTH, float(style["linewidth"])),
                markersize=max(SERIES_MARKERSIZE, float(style["markersize"])),
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                label=_get_method_label(method),
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(3, len(legend_handles)),
        frameon=False,
        handlelength=2.4,
        columnspacing=1.4,
        handletextpad=0.7,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for suffix in (".png", ".pdf"):
        output_path = output_stem.with_suffix(suffix)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
        saved_paths.append(output_path)
    plt.close(fig)

    for output_path in saved_paths:
        print(f"Saved {output_path}")
    return saved_paths


def draw_power_vs_rps(
    csv_path: Path | str = DEFAULT_CSV_PATH,
    output_stem: Path | str | None = None,
    power_aggregation: str = DEFAULT_POWER_AGGREGATION,
    disagg_component_csv: Path | str | None = None,
    include_disagg_components: bool = True,
) -> list[Path]:
    csv_path = Path(csv_path)
    if output_stem is None:
        output_stem = _default_output_stem(
            csv_path,
            power_aggregation=power_aggregation,
        )
    output_stem = Path(output_stem)

    plt, Line2D, FuncFormatter = _get_plotting_dependencies()
    _apply_paper_style(plt)

    rows, duplicate_count, skipped_missing_power, skipped_zero_power = _load_rows(
        csv_path,
        power_aggregation=power_aggregation,
    )
    series_by_method, method_order = _build_method_series(rows)
    series_by_method, method_order = _filter_methods(series_by_method, method_order)
    resolved_disagg_component_csv = _resolve_disagg_component_csv(
        csv_path,
        disagg_component_csv,
        include_disagg_components=include_disagg_components,
    )
    if resolved_disagg_component_csv is not None:
        (
            component_rows,
            component_duplicate_count,
            component_skipped_missing_power,
            component_skipped_zero_power,
        ) = _load_disagg_component_rows(resolved_disagg_component_csv)
        series_by_method, method_order = _merge_method_series(
            series_by_method,
            method_order,
            component_rows,
        )
        duplicate_count += component_duplicate_count
        skipped_missing_power += component_skipped_missing_power
        skipped_zero_power += component_skipped_zero_power
    grouped_by_mode = _split_methods_by_serving_mode(series_by_method, method_order)
    if not grouped_by_mode:
        raise ValueError(f"No methods found to plot in {csv_path}.")

    dataset_label = _dataset_label(csv_path)
    y_label = _power_axis_label(power_aggregation)
    saved_paths = []
    for serving_mode, (mode_series_by_method, mode_method_order) in grouped_by_mode.items():
        mode_title = dataset_label
        if len(grouped_by_mode) > 1:
            mode_suffix = "Disagg" if serving_mode == "disagg" else "Normal"
            mode_title = f"{dataset_label} ({mode_suffix})"
        saved_paths.extend(
            _render_figure(
                plt=plt,
                line2d_cls=Line2D,
                func_formatter_cls=FuncFormatter,
                output_stem=_output_stem_for_serving_mode(
                    output_stem=output_stem,
                    serving_mode=serving_mode,
                    split_count=len(grouped_by_mode),
                ),
                series_by_method=mode_series_by_method,
                method_order=mode_method_order,
                title=mode_title,
                y_label=y_label,
            )
        )

    if duplicate_count:
        print(f"Retained the last CSV row for {duplicate_count} duplicate (method, rps) points.")
    if skipped_missing_power:
        print(f"Skipped {skipped_missing_power} row(s) without usable per_server_power.")
    if skipped_zero_power:
        print(f"Skipped {skipped_zero_power} row(s) with zero power values.")
    if resolved_disagg_component_csv is not None:
        print(
            "Included prefill/decode component curves from "
            f"{resolved_disagg_component_csv}."
        )
    return saved_paths


def draw_all_power_vs_rps(
    input_dir: Path | str = DEFAULT_INPUT_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    power_aggregation: str = DEFAULT_POWER_AGGREGATION,
    glob_pattern: str = DEFAULT_GLOB_PATTERN,
    disagg_component_csv: Path | str | None = None,
    include_disagg_components: bool = True,
) -> list[Path]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    saved_paths: list[Path] = []
    failures: list[tuple[Path, Exception]] = []
    for csv_path in _iter_csv_files(input_dir, glob_pattern=glob_pattern):
        output_stem = _default_output_stem(
            csv_path,
            power_aggregation=power_aggregation,
            output_dir=output_dir,
        )
        try:
            saved_paths.extend(
                draw_power_vs_rps(
                    csv_path=csv_path,
                    output_stem=output_stem,
                    power_aggregation=power_aggregation,
                    disagg_component_csv=disagg_component_csv,
                    include_disagg_components=include_disagg_components,
                )
            )
        except Exception as exc:
            failures.append((csv_path, exc))

    print(
        f"Processed {len(saved_paths) // 2} figure set(s) from {input_dir} "
        f"matching {glob_pattern!r}."
    )
    if failures:
        failure_lines = "\n".join(f"- {path}: {exc}" for path, exc in failures)
        raise RuntimeError(f"Failed to render some CSVs:\n{failure_lines}")
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render Power-vs-RPS figures from the sensitivity CSVs using the "
            "per_server_power field and comparator labels/styles from plots/common.py."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Input CSV file. Default: %(default)s",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        help="Output path without extension. Defaults to Paper/figs/power_vs_rps-<dataset>.",
    )
    parser.add_argument(
        "--draw-all",
        action="store_true",
        help="Render figures for every matching CSV under --input-dir. Ignores --csv and --output-stem.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory scanned by --draw-all. Default: %(default)s",
    )
    parser.add_argument(
        "--glob",
        default=DEFAULT_GLOB_PATTERN,
        help="Glob used by --draw-all. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory used by --draw-all. Default: %(default)s",
    )
    parser.add_argument(
        "--power-aggregation",
        choices=("sum", "mean", "max"),
        default=DEFAULT_POWER_AGGREGATION,
        help=(
            "How to reduce the per_server_power list for each CSV row. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--disagg-component-csv",
        type=Path,
        help=(
            "CSV used for the Prefill Only and Decode Only overlay curves. "
            "Defaults to Paper/data/disagg_energy_profiling.csv for chat23."
        ),
    )
    parser.add_argument(
        "--no-disagg-components",
        action="store_true",
        help="Disable the Prefill Only and Decode Only overlay curves.",
    )
    args = parser.parse_args()

    if args.draw_all:
        draw_all_power_vs_rps(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            power_aggregation=args.power_aggregation,
            glob_pattern=args.glob,
            disagg_component_csv=args.disagg_component_csv,
            include_disagg_components=not args.no_disagg_components,
        )
    else:
        draw_power_vs_rps(
            csv_path=args.csv,
            output_stem=args.output_stem,
            power_aggregation=args.power_aggregation,
            disagg_component_csv=args.disagg_component_csv,
            include_disagg_components=not args.no_disagg_components,
        )


if __name__ == "__main__":
    main()
