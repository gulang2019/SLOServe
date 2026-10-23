from __future__ import annotations

import argparse
import csv
import math
from bisect import bisect_left
from collections import OrderedDict
from pathlib import Path

import numpy as np

try:
    from plots.common import get_method_label, get_method_style
except ModuleNotFoundError:
    from common import get_method_label, get_method_style


DEFAULT_INPUT_DIR = Path("Paper/data/e2e")
DEFAULT_CSV_PATH = DEFAULT_INPUT_DIR / "chat_7B.csv"
DEFAULT_OUTPUT_DIR = Path("Paper/figs/e2e_new")
FIGSIZE = (14.8, 4.6)
DUO_FIGSIZE = (7.6, 3.2)
SLO_FOCUS_X_ANCHORS = np.array([0.0, 1.0, 5.0, 10.0], dtype=float)
SLO_FOCUS_Y_ANCHORS = np.array([0.0, 0.2, 0.84, 1.0], dtype=float)
CONSTANT_FIELDS = ("load_scale", "ttft_slo_scale", "slo_tpot", "perf_model_err")
DEFAULT_MAX_SLO_VIOLATION = 10.0
TARGET_SLO_VIOLATIONS = (1.0, 5.0)
ANNOTATION_COLOR = "#808080"
ANNOTATION_TEXT_COLOR = "#C62828"
MODELED_OPTIMAL_LABEL = "modeled optimal"
MODELED_OPTIMAL_MARKER_COLOR = "#C48A00"
MODELED_OPTIMAL_MARKER = "*"
MODELED_OPTIMAL_MARKERSIZE = 12.0
MODELED_OPTIMAL_MARKEREDGEWIDTH = 0.8
ZERO_SLO_LEFT_X_MARGIN = -0.35
MODELED_OPTIMAL_ACTIVE_GPUS_BY_CSV_STEM = {
    "chat_7B": 2.7507233811956753,
    "code_7B": 1.27,
    "mixed_7B": 3.9442960235355233,
    "32B_chat": 4.569926819583333,
    "e2e_mulitiround_chatting": 8.0,
    "chatting_large_scale": 8.82,
}
DEFAULT_PANEL_KEYS = ("energy", "active_gpus", "slo_vs_gpus")
DEFAULT_ASPLOS_PANEL_KEYS = DEFAULT_PANEL_KEYS[:2]
DEFAULT_BASELINE_FAMILY_COLLAPSE = (
    (
        "vllm / round_robin",
        (
            "qlm / round_robin",
            "vllm / round_robin",
        ),
    ),
    (
        "vllm / round_robin_session",
        (
            "qlm / round_robin_session",
            "vllm / round_robin_session",
        ),
    ),
    (
        "sarathi / round_robin",
        (
            "sarathi / round_robin",
            "sarathi+ / round_robin",
        ),
    ),
)
DEFAULT_VLLM_BASELINE_FAMILY_COLLAPSE = (
    (
        "vllm / round_robin",
        (
            "qlm / round_robin",
            "vllm / round_robin",
        ),
    ),
    (
        "vllm / round_robin_session",
        (
            "qlm / round_robin_session",
            "vllm / round_robin_session",
        ),
    ),
)


def _get_plotting_dependencies():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D
        from matplotlib.ticker import MaxNLocator
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to render this figure. "
            "Install it first, then rerun the script."
        ) from exc
    return plt, LineCollection, Line2D, MaxNLocator


def _apply_paper_style(
    plt,
    *,
    panel_keys: tuple[str, ...] = DEFAULT_PANEL_KEYS,
    large_format: bool = False,
    single_column: bool = False,
    content_scale: float = 1.0,
) -> None:
    font_scale = _font_scale_for_layout(
        panel_keys,
        large_format=large_format,
        single_column=single_column,
        content_scale=content_scale,
    )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.5 * font_scale,
            "axes.labelsize": 12 * font_scale,
            "axes.titlesize": 12 * font_scale,
            "xtick.labelsize": 10.5 * font_scale,
            "ytick.labelsize": 10.5 * font_scale,
            "legend.fontsize": 10.5 * font_scale,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 4.0 * font_scale,
            "ytick.major.size": 4.0 * font_scale,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _font_scale_for_layout(
    panel_keys: tuple[str, ...],
    *,
    large_format: bool = False,
    single_column: bool = False,
    content_scale: float = 1.0,
) -> float:
    if content_scale <= 0:
        raise ValueError("content_scale must be positive.")
    if single_column and len(panel_keys) == 2:
        base_scale = 1.35 if large_format else 1.15
    elif len(panel_keys) == 1:
        base_scale = 1.75 if large_format else 1.2
    elif len(panel_keys) == 2:
        base_scale = 1.3 if large_format else 1.0
    else:
        base_scale = 1.55 if large_format else 1.0
    return base_scale * content_scale


def _parse_required_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value == "":
        raise ValueError(f"Missing required field '{field}' in row: {row}")
    return float(value)


def _parse_optional_float(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value == "":
        return None
    return float(value)


def _default_output_stem(
    csv_path: Path,
    max_slo_violation: float,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    panel_keys: tuple[str, ...] = DEFAULT_PANEL_KEYS,
) -> Path:
    suffix = str(int(max_slo_violation)) if float(max_slo_violation).is_integer() else str(max_slo_violation).replace(".", "p")
    if panel_keys == DEFAULT_PANEL_KEYS:
        layout_tag = "triptych"
    elif panel_keys == DEFAULT_ASPLOS_PANEL_KEYS:
        layout_tag = "duo"
    else:
        layout_tag = f"{len(panel_keys)}panel"
    return output_dir / f"{csv_path.stem}_n_device_{layout_tag}_slo_cap_{suffix}"


def _append_output_suffix(output_stem: Path, suffix: str) -> Path:
    if output_stem.name.endswith(suffix):
        return output_stem
    return output_stem.with_name(f"{output_stem.name}{suffix}")


def _normalize_panel_keys(panel_keys: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if panel_keys is None:
        panel_keys = DEFAULT_PANEL_KEYS
    ordered_keys: list[str] = []
    for key in panel_keys:
        if key not in DEFAULT_PANEL_KEYS:
            raise ValueError(
                f"Unsupported panel '{key}'. Expected one of: {', '.join(DEFAULT_PANEL_KEYS)}"
            )
        if key not in ordered_keys:
            ordered_keys.append(key)
    if not ordered_keys:
        raise ValueError("At least one panel must be selected.")
    return tuple(ordered_keys)


def _figure_size_for_panels(
    panel_keys: tuple[str, ...],
    *,
    large_format: bool = False,
    single_column: bool = False,
) -> tuple[float, float]:
    if len(panel_keys) == 2:
        if single_column:
            return (3.35, 5.15) if large_format else (3.35, 4.6)
        return DUO_FIGSIZE if large_format else (10.4, 4.8)
    if len(panel_keys) == 1:
        return (5.0, 4.35) if large_format else (4.7, 4.0)
    return (15.4, 4.9) if large_format else FIGSIZE


def _iter_csv_files(input_dir: Path | str) -> list[Path]:
    input_dir = Path(input_dir)
    csv_paths = sorted(path for path in input_dir.glob("*.csv") if path.is_file())
    if not csv_paths:
        raise ValueError(f"No CSV files found in {input_dir}")
    return csv_paths


def _load_rows(
    csv_path: Path | str,
) -> tuple[list[dict[str, float | int | str | None]], int]:
    csv_path = Path(csv_path)
    required_fields = {
        "scheduling_policy",
        "routing_policy",
        "n_device",
        "slo_violation_rate",
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
        deduped_rows: OrderedDict[tuple[str, int], dict[str, float | int | str | None]] = OrderedDict()
        duplicate_count = 0

        for row in reader:
            for field in constant_context:
                value = row.get(field, "")
                if value != "":
                    constant_context[field].add(value)

            slo_violation_rate = _parse_required_float(row, "slo_violation_rate") * 100.0

            method = f"{row['scheduling_policy']} / {row['routing_policy']}"
            parsed_row = {
                "method": method,
                "n_device": int(_parse_required_float(row, "n_device")),
                "slo_violation_rate": slo_violation_rate,
                "energy_consumption": None,
                "average_n_active_servers": _parse_optional_float(row, "average_n_active_servers"),
                "active_gpu_proportion": None,
            }
            energy_consumption = _parse_optional_float(row, "energy_consumption")
            if energy_consumption is not None:
                parsed_row["energy_consumption"] = energy_consumption / 1000.0
            average_n_active_servers = parsed_row["average_n_active_servers"]
            if average_n_active_servers is not None and int(parsed_row["n_device"]) > 0:
                parsed_row["active_gpu_proportion"] = (
                    float(average_n_active_servers) / float(parsed_row["n_device"])
                )
            dedupe_key = (parsed_row["method"], parsed_row["n_device"])
            if dedupe_key in deduped_rows:
                duplicate_count += 1
            deduped_rows[dedupe_key] = parsed_row

    varying_context = {
        field: sorted(values)
        for field, values in constant_context.items()
        if len(values) > 1
    }
    if varying_context:
        details = ", ".join(f"{field}={values}" for field, values in varying_context.items())
        raise ValueError(
            "This figure expects n_device to be the only swept variable. "
            f"Found additional varying context in {csv_path}: {details}"
        )

    rows = list(deduped_rows.values())
    if not rows:
        raise ValueError(f"No rows found in {csv_path}.")
    return rows, duplicate_count


def _build_method_series(
    rows: list[dict[str, float | int | str | None]],
) -> tuple[dict[str, list[dict[str, float | int | str | None]]], list[str]]:
    series_by_method: dict[str, list[dict[str, float | int | str | None]]] = OrderedDict()
    method_order: OrderedDict[str, None] = OrderedDict()

    for row in rows:
        method = str(row["method"])
        method_order.setdefault(method, None)
        series_by_method.setdefault(method, []).append(row)

    for points in series_by_method.values():
        points.sort(key=lambda point: int(point["n_device"]))

    return series_by_method, list(method_order.keys())


def _method_serving_mode(method: str) -> str:
    return "disagg" if "disagg" in method else "normal"


def _is_slo_packer_candidate(method: str) -> bool:
    return method.startswith("atfc /")


def _split_methods_by_serving_mode(
    series_by_method: dict[str, list[dict[str, float | int | str | None]]],
    method_order: list[str],
) -> OrderedDict[str, tuple[dict[str, list[dict[str, float | int | str | None]]], list[str]]]:
    grouped: OrderedDict[str, tuple[dict[str, list[dict[str, float | int | str | None]]], list[str]]] = OrderedDict()
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
        ax.spines[spine_name].set_linewidth(0.9)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, linestyle=(0, (2, 2)))
    ax.set_axisbelow(True)


def _positive_slo_violation_x_values(
    series_by_method: dict[str, list[dict[str, float | int | str | None]]],
    max_slo_violation: float,
) -> list[float]:
    positive_xs = [
        float(point["slo_violation_rate"])
        for points in series_by_method.values()
        for point in points
        if point.get("slo_violation_rate") is not None
        and float(point["slo_violation_rate"]) > 0.0
        and float(point["slo_violation_rate"]) <= max_slo_violation
    ]
    if positive_xs:
        return positive_xs
    return [
        float(point["slo_violation_rate"])
        for points in series_by_method.values()
        for point in points
        if point.get("slo_violation_rate") is not None
        and float(point["slo_violation_rate"]) > 0.0
    ]


def _slo_violation_x_lower_bound(*, include_left_margin: bool) -> float:
    if include_left_margin:
        return ZERO_SLO_LEFT_X_MARGIN
    return 0.0


def _apply_slo_violation_log_xscale(
    ax,
    *,
    series_by_method: dict[str, list[dict[str, float | int | str | None]]],
    max_slo_violation: float,
    include_left_margin: bool = False,
) -> None:
    positive_xs = _positive_slo_violation_x_values(
        series_by_method,
        max_slo_violation=max_slo_violation,
    )
    if not positive_xs:
        raise ValueError(
            "Cannot apply log scale to the SLO-violation axis because there are no "
            "positive SLO-violation values in the plotted data."
        )

    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    tick_candidates = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
    visible_ticks = [
        tick
        for tick in tick_candidates
        if tick <= max_slo_violation + 1e-9
    ]
    if not visible_ticks:
        visible_ticks = [0.0, 0.5]

    lower_bound = _slo_violation_x_lower_bound(
        include_left_margin=include_left_margin,
    )
    upper_bound = visible_ticks[-1]
    max_positive = max(positive_xs)
    if max_positive > upper_bound:
        upper_bound = max_positive

    # Piecewise focus scale: compress 0-1 and 5-10 while expanding 1-5.
    ax.set_xscale(
        "function",
        functions=(_slo_violation_focus_forward, _slo_violation_focus_inverse),
    )
    ax.set_xlim(lower_bound, upper_bound)
    ax.xaxis.set_major_locator(FixedLocator(visible_ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_locator(NullLocator())


def _apply_slo_violation_symlog_xscale(
    ax,
    *,
    series_by_method: dict[str, list[dict[str, float | int | str | None]]],
    max_slo_violation: float,
    include_left_margin: bool = False,
) -> None:
    positive_xs = _positive_slo_violation_x_values(
        series_by_method,
        max_slo_violation=max_slo_violation,
    )
    if not positive_xs:
        raise ValueError(
            "Cannot apply symlog scale to the SLO-violation axis because there are no "
            "positive SLO-violation values in the plotted data."
        )

    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    tick_candidates = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
    visible_ticks = [
        tick
        for tick in tick_candidates
        if tick <= max_slo_violation + 1e-9
    ]
    if not visible_ticks:
        visible_ticks = [0.0, 0.5]

    upper_bound = max(visible_ticks[-1], max(positive_xs))
    ax.set_xscale("symlog", linthresh=1.0, linscale=1.0, base=10.0)
    ax.set_xlim(
        _slo_violation_x_lower_bound(
            include_left_margin=include_left_margin,
        ),
        upper_bound,
    )
    ax.xaxis.set_major_locator(FixedLocator(visible_ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_locator(NullLocator())


def _add_zero_slo_reference_line(ax) -> None:
    ax.axvline(
        0.0,
        color=ANNOTATION_COLOR,
        linestyle="-",
        linewidth=1.0,
        alpha=0.75,
        zorder=0,
    )


def _piecewise_linear_scale(
    values,
    *,
    input_anchors: np.ndarray,
    output_anchors: np.ndarray,
):
    array_values = np.asarray(values, dtype=float)
    flattened = array_values.reshape(-1)
    scaled = np.interp(flattened, input_anchors, output_anchors)

    if len(input_anchors) >= 2:
        left_slope = (
            (output_anchors[1] - output_anchors[0])
            / (input_anchors[1] - input_anchors[0])
        )
        right_slope = (
            (output_anchors[-1] - output_anchors[-2])
            / (input_anchors[-1] - input_anchors[-2])
        )
        below = flattened < input_anchors[0]
        above = flattened > input_anchors[-1]
        if np.any(below):
            scaled[below] = output_anchors[0] + (
                flattened[below] - input_anchors[0]
            ) * left_slope
        if np.any(above):
            scaled[above] = output_anchors[-1] + (
                flattened[above] - input_anchors[-1]
            ) * right_slope

    scaled = scaled.reshape(array_values.shape)
    if np.isscalar(values):
        return float(scaled)
    return scaled


def _slo_violation_focus_forward(values):
    return _piecewise_linear_scale(
        values,
        input_anchors=SLO_FOCUS_X_ANCHORS,
        output_anchors=SLO_FOCUS_Y_ANCHORS,
    )


def _slo_violation_focus_inverse(values):
    return _piecewise_linear_scale(
        values,
        input_anchors=SLO_FOCUS_Y_ANCHORS,
        output_anchors=SLO_FOCUS_X_ANCHORS,
    )


def _blend_with_white(color: str, blend: float) -> tuple[float, float, float]:
    color = color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got: {color}")
    blend = max(0.0, min(1.0, blend))
    rgb = tuple(int(color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return tuple(channel * (1.0 - blend) + blend for channel in rgb)


def _gradient_point_colors(base_color: str, values: list[float]) -> list[tuple[float, float, float]]:
    if len(values) <= 1:
        return [_blend_with_white(base_color, 0.0) for _ in values]
    low = min(values)
    high = max(values)
    if high <= low:
        return [_blend_with_white(base_color, 0.0) for _ in values]
    colors = []
    for value in values:
        norm = (value - low) / (high - low)
        # Low n_device is lighter; high n_device approaches the method's base color.
        colors.append(_blend_with_white(base_color, blend=0.58 * (1.0 - norm)))
    return colors


def _palette_for_base_color(base_color: str) -> list[str]:
    palettes = {
        "#2E7D32": [
            "#2E7D32",
            "#00897B",
            "#1565C0",
            "#6A1B9A",
            "#EF6C00",
            "#C62828",
        ],
        "#E76F51": [
            "#E76F51",
            "#F4A261",
            "#D1495B",
            "#B56576",
            "#F77F00",
            "#9C6644",
        ],
    }
    return palettes.get(
        base_color,
        [
            base_color,
            "#00897B",
            "#1565C0",
            "#6A1B9A",
            "#EF6C00",
            "#C62828",
        ],
    )


def _build_method_styles(method_order: list[str]) -> dict[str, dict[str, object]]:
    style_by_method = {
        method: dict(get_method_style(method))
        for method in method_order
    }
    methods_by_color: OrderedDict[str, list[str]] = OrderedDict()
    for method in method_order:
        color = str(style_by_method[method]["color"])
        methods_by_color.setdefault(color, []).append(method)

    for base_color, methods in methods_by_color.items():
        if len(methods) <= 1:
            continue
        palette = _palette_for_base_color(base_color)
        for idx, method in enumerate(methods):
            style_by_method[method]["color"] = palette[idx % len(palette)]

    return style_by_method


def _build_legend_handles(
    line2d_cls,
    method_order: list[str],
    style_by_method: dict[str, dict[str, object]],
    *,
    style_scale: float = 1.0,
    include_modeled_optimal: bool = False,
):
    legend_handles = []
    for method in method_order:
        style = style_by_method[method]
        legend_handles.append(
            line2d_cls(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linewidth=max(2.0, float(style["linewidth"]) * style_scale),
                markersize=max(6.5, float(style["markersize"]) * style_scale),
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                label=get_method_label(method),
            )
        )
    if include_modeled_optimal:
        legend_handles.append(
            line2d_cls(
                [0],
                [0],
                color=MODELED_OPTIMAL_MARKER_COLOR,
                marker=MODELED_OPTIMAL_MARKER,
                linewidth=0,
                markersize=MODELED_OPTIMAL_MARKERSIZE * style_scale,
                markerfacecolor=MODELED_OPTIMAL_MARKER_COLOR,
                markeredgecolor="black",
                markeredgewidth=MODELED_OPTIMAL_MARKEREDGEWIDTH,
                label=MODELED_OPTIMAL_LABEL,
            )
        )
    return legend_handles


def _polyline_breakpoints_for_x_key(x_key: str) -> tuple[float, ...]:
    if x_key == "slo_violation_rate":
        return tuple(float(value) for value in SLO_FOCUS_X_ANCHORS[1:-1])
    return ()


def _densify_polyline_for_breakpoints(
    xs: list[float],
    ys: list[float],
    breakpoints: tuple[float, ...],
) -> tuple[list[float], list[float], list[int]]:
    if len(xs) <= 1 or not breakpoints:
        return list(xs), list(ys), list(range(max(0, len(xs) - 1)))

    densified_xs = [xs[0]]
    densified_ys = [ys[0]]
    segment_source_indices: list[int] = []

    for idx, (x0, y0, x1, y1) in enumerate(zip(xs[:-1], ys[:-1], xs[1:], ys[1:])):
        interior_breaks = [
            breakpoint
            for breakpoint in breakpoints
            if min(x0, x1) < breakpoint < max(x0, x1)
        ]
        for breakpoint in interior_breaks:
            interpolation = (breakpoint - x0) / (x1 - x0)
            densified_xs.append(breakpoint)
            densified_ys.append(y0 + interpolation * (y1 - y0))
            segment_source_indices.append(idx)
        densified_xs.append(x1)
        densified_ys.append(y1)
        segment_source_indices.append(idx)

    return densified_xs, densified_ys, segment_source_indices


def _estimate_value_at_target(
    points: list[dict[str, float | int | str | None]],
    *,
    x_key: str,
    y_key: str,
    target: float,
) -> tuple[float | None, tuple[tuple[float, float], tuple[float, float]] | None]:
    points_by_x: dict[float, float] = {}
    for point in points:
        x_value = point.get(x_key)
        y_value = point.get(y_key)
        if x_value is None or y_value is None:
            continue
        points_by_x[float(x_value)] = float(y_value)

    sorted_points = sorted(points_by_x.items())
    if not sorted_points:
        return None, None

    xs = [point[0] for point in sorted_points]
    ys = [point[1] for point in sorted_points]
    idx = bisect_left(xs, target)

    if idx < len(xs) and math.isclose(xs[idx], target):
        return ys[idx], None
    if len(sorted_points) == 1:
        return None, None

    if idx == 0:
        x0, y0 = sorted_points[0]
        x1, y1 = sorted_points[1]
        extension_start = (x0, y0)
    elif idx == len(sorted_points):
        x0, y0 = sorted_points[-2]
        x1, y1 = sorted_points[-1]
        extension_start = (x1, y1)
    else:
        x0, y0 = sorted_points[idx - 1]
        x1, y1 = sorted_points[idx]
        extension_start = None

    if math.isclose(x0, x1):
        return None, None

    slope = (y1 - y0) / (x1 - x0)
    target_y = y0 + slope * (target - x0)
    if y_key in {"energy_consumption", "average_n_active_servers"}:
        target_y = max(0.0, target_y)
    elif y_key == "active_gpu_proportion":
        target_y = min(1.0, max(0.0, target_y))
    extension = None
    if extension_start is not None:
        extension = (extension_start, (target, target_y))
    return target_y, extension


def _primary_ours_method(
    method_order: list[str],
    *,
    serving_mode: str | None = None,
) -> str | None:
    filtered_methods = [
        method
        for method in method_order
        if serving_mode is None or _method_serving_mode(method) == serving_mode
    ]
    if serving_mode == "disagg":
        preferred_methods = (
            "atfc / slosserve_disagg_planner",
            "atfc / slosserve_planner_disagg",
        )
    elif serving_mode == "normal":
        preferred_methods = (
            "atfc / slosserve_4_planner",
            "atfc / slosserve_planner",
        )
    else:
        preferred_methods = (
            "atfc / slosserve_4_planner",
            "atfc / slosserve_planner",
            "atfc / slosserve_disagg_planner",
            "atfc / slosserve_planner_disagg",
        )

    for method in preferred_methods:
        if method in filtered_methods:
            return method

    if serving_mode == "disagg":
        return next(
            (
                method
                for method in filtered_methods
                if method.startswith("atfc /") and "disagg" in method
            ),
            None,
        )

    return next(
        (
            method
            for method in filtered_methods
            if (
                "slosserve_4_planner" in method
                or "slosserve_planner" in method
            )
        ),
        None,
    )


def _filter_default_e2e_methods(
    series_by_method: dict[str, list[dict[str, float | int | str | None]]],
    method_order: list[str],
) -> tuple[
    dict[str, list[dict[str, float | int | str | None]]],
    list[str],
]:
    primary_ours_methods = {
        serving_mode: _primary_ours_method(method_order, serving_mode=serving_mode)
        for serving_mode in ("normal", "disagg")
    }
    filtered_method_order = [
        method
        for method in method_order
        if not method.startswith("atfc /")
        or method == primary_ours_methods.get(_method_serving_mode(method))
    ]
    filtered_series_by_method = {
        method: series_by_method[method]
        for method in filtered_method_order
        if method in series_by_method
    }
    return filtered_series_by_method, filtered_method_order


def _exclude_methods_by_substring(
    series_by_method: dict[str, list[dict[str, float | int | str | None]]],
    method_order: list[str],
    excluded_method_substrings: tuple[str, ...] | list[str] | None,
) -> tuple[
    dict[str, list[dict[str, float | int | str | None]]],
    list[str],
]:
    if not excluded_method_substrings:
        return series_by_method, method_order

    patterns = tuple(
        pattern.strip().lower()
        for pattern in excluded_method_substrings
        if pattern and pattern.strip()
    )
    if not patterns:
        return series_by_method, method_order

    filtered_method_order = [
        method
        for method in method_order
        if not any(
            pattern in method.lower() or pattern in get_method_label(method).lower()
            for pattern in patterns
        )
    ]
    filtered_series_by_method = {
        method: series_by_method[method]
        for method in filtered_method_order
        if method in series_by_method
    }
    return filtered_series_by_method, filtered_method_order


def _format_gap_label(y_key: str, ours_y: float, baseline_y: float) -> str | None:
    if y_key == "energy_consumption":
        if baseline_y <= 0:
            return None
        savings = (baseline_y - ours_y) / baseline_y * 100.0
        if savings >= 0:
            return f"{savings:.1f}% $\\downarrow$"
        return f"{abs(savings):.1f}% $\\uparrow$"
    if y_key == "average_n_active_servers":
        if baseline_y <= 0:
            return None
        gap_pct = (baseline_y - ours_y) / baseline_y * 100.0
        if gap_pct >= 0:
            return f"{gap_pct:.1f}% $\\downarrow$"
        return f"{abs(gap_pct):.1f}% $\\uparrow$"
    if y_key == "active_gpu_proportion":
        if baseline_y <= 0:
            return None
        gap_pct = (baseline_y - ours_y) / baseline_y * 100.0
        if gap_pct >= 0:
            return f"{gap_pct:.1f}% $\\downarrow$"
        return f"{abs(gap_pct):.1f}% $\\uparrow$"
    return None


def _annotation_metric_for_y_key(y_key: str) -> str:
    return y_key


def _select_best_baseline_method(
    series_by_method: dict[str, list[dict[str, float | int | str | None]]],
    method_order: list[str],
) -> str | None:
    return _select_best_method_from_candidates(
        series_by_method,
        method_order,
        candidate_methods=[
            method for method in method_order if not _is_slo_packer_candidate(method)
        ],
    )


def _score_candidate_method(
    series_by_method: dict[str, list[dict[str, float | int | str | None]]],
    method: str,
) -> tuple[float, ...] | None:
    points = series_by_method.get(method, [])
    energy_values: list[float] = []
    active_values: list[float] = []
    for target in TARGET_SLO_VIOLATIONS:
        energy_y, _ = _estimate_value_at_target(
            points,
            x_key="slo_violation_rate",
            y_key="energy_consumption",
            target=target,
        )
        active_y, _ = _estimate_value_at_target(
            points,
            x_key="slo_violation_rate",
            y_key="average_n_active_servers",
            target=target,
        )
        if energy_y is not None:
            energy_values.append(float(energy_y))
        if active_y is not None:
            active_values.append(float(active_y))

    if not energy_values and not active_values:
        return None

    return (
        -float(len(energy_values)),
        (
            math.inf
            if not energy_values
            else sum(energy_values) / len(energy_values)
        ),
        -float(len(active_values)),
        (
            math.inf
            if not active_values
            else sum(active_values) / len(active_values)
        ),
        float(
            min(
                point["slo_violation_rate"]
                for point in points
                if point.get("slo_violation_rate") is not None
            )
        ),
    )


def _select_best_method_from_candidates(
    series_by_method: dict[str, list[dict[str, float | int | str | None]]],
    method_order: list[str],
    candidate_methods: list[str] | tuple[str, ...],
) -> str | None:
    best_method: str | None = None
    best_score: tuple[float, ...] | None = None

    for method in method_order:
        if method not in candidate_methods:
            continue

        score = _score_candidate_method(series_by_method, method)
        if score is None:
            continue
        if best_score is None or score < best_score:
            best_score = score
            best_method = method

    return best_method


def _collapse_baseline_families(
    series_by_method: dict[str, list[dict[str, float | int | str | None]]],
    method_order: list[str],
    family_specs: tuple[tuple[str, tuple[str, ...]], ...] = DEFAULT_BASELINE_FAMILY_COLLAPSE,
) -> tuple[
    dict[str, list[dict[str, float | int | str | None]]],
    list[str],
    dict[str, str],
]:
    family_lookup: dict[str, str] = {}
    family_members: dict[str, tuple[str, ...]] = {}
    best_by_family: dict[str, str] = {}
    selected_details: dict[str, str] = {}

    for canonical_method, members in family_specs:
        family_members[canonical_method] = members
        for member in members:
            family_lookup[member] = canonical_method
        best_method = _select_best_method_from_candidates(
            series_by_method,
            method_order,
            candidate_methods=list(members),
        )
        if best_method is not None:
            best_by_family[canonical_method] = best_method
            selected_details[canonical_method] = best_method

    collapsed_series_by_method: dict[str, list[dict[str, float | int | str | None]]] = OrderedDict()
    collapsed_method_order: list[str] = []
    inserted_families: set[str] = set()

    for method in method_order:
        canonical_method = family_lookup.get(method)
        if canonical_method is None:
            collapsed_method_order.append(method)
            collapsed_series_by_method[method] = series_by_method[method]
            continue
        if canonical_method in inserted_families:
            continue
        best_method = best_by_family.get(canonical_method)
        inserted_families.add(canonical_method)
        if best_method is None:
            continue
        collapsed_method_order.append(canonical_method)
        collapsed_series_by_method[canonical_method] = [
            dict(point, method=canonical_method)
            for point in series_by_method[best_method]
        ]

    return collapsed_series_by_method, collapsed_method_order, selected_details


def _filter_to_slo_packer_candidates_plus_best_baseline(
    series_by_method: dict[str, list[dict[str, float | int | str | None]]],
    method_order: list[str],
) -> tuple[
    dict[str, list[dict[str, float | int | str | None]]],
    list[str],
    list[str],
]:
    candidate_methods = [
        method for method in method_order if _is_slo_packer_candidate(method)
    ]
    best_baseline = _select_best_baseline_method(series_by_method, method_order)
    selected_methods = set(candidate_methods)
    if best_baseline is not None:
        selected_methods.add(best_baseline)

    filtered_method_order = [
        method for method in method_order if method in selected_methods
    ]
    filtered_series_by_method = {
        method: series_by_method[method]
        for method in filtered_method_order
    }
    baseline_methods = [] if best_baseline is None else [best_baseline]
    return filtered_series_by_method, filtered_method_order, baseline_methods


def _add_threshold_gap_annotations(
    ax,
    *,
    series_by_method: dict[str, list[dict[str, float | int | str | None]]],
    method_order: list[str],
    y_key: str,
    max_slo_violation: float,
    baseline_methods: list[str] | None = None,
    annotation_fontsize: float = 10.5,
    annotation_text_color: str = ANNOTATION_TEXT_COLOR,
    annotation_placement: str = "lower_left_of_ours",
    show_gap_connector: bool = True,
    show_gap_label: bool = True,
    show_target_reference_lines: bool = True,
) -> list[dict[str, object]]:
    ours_method = _primary_ours_method(method_order)
    if ours_method is None:
        return []

    comparisons: list[dict[str, object]] = []
    ours_points = series_by_method.get(ours_method, [])
    for target in TARGET_SLO_VIOLATIONS:
        if target > max_slo_violation:
            continue
        if show_target_reference_lines:
            ax.axvline(
                target,
                color=ANNOTATION_COLOR,
                linestyle=":",
                linewidth=1.0,
                zorder=0,
            )

        ours_y, ours_extension = _estimate_value_at_target(
            ours_points,
            x_key="slo_violation_rate",
            y_key=y_key,
            target=target,
        )
        if ours_extension is not None:
            (x0, y0), (x1, y1) = ours_extension
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=ANNOTATION_COLOR,
                linestyle="--",
                linewidth=1.1,
                zorder=3,
            )

        best_baseline: dict[str, object] | None = None
        candidate_baseline_methods = (
            method_order if baseline_methods is None else baseline_methods
        )
        for method in candidate_baseline_methods:
            if method == ours_method:
                continue
            baseline_y, baseline_extension = _estimate_value_at_target(
                series_by_method.get(method, []),
                x_key="slo_violation_rate",
                y_key=y_key,
                target=target,
            )
            if baseline_y is None:
                continue
            if best_baseline is None or float(baseline_y) < float(best_baseline["y"]):
                best_baseline = {
                    "method": method,
                    "y": float(baseline_y),
                    "extension": baseline_extension,
                }

        if ours_y is None or best_baseline is None:
            continue

        baseline_y = float(best_baseline["y"])
        baseline_extension = best_baseline["extension"]
        if baseline_extension is not None:
            (x0, y0), (x1, y1) = baseline_extension
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=ANNOTATION_COLOR,
                linestyle="--",
                linewidth=1.1,
                zorder=3,
            )

        y_low, y_high = sorted((float(ours_y), baseline_y))
        if show_gap_connector:
            ax.plot(
                [target, target],
                [y_low, y_high],
                color=ANNOTATION_COLOR,
                linewidth=1.2,
                zorder=4,
            )
        ax.scatter(
            [target, target],
            [float(ours_y), baseline_y],
            color=ANNOTATION_COLOR,
            s=18,
            zorder=5,
        )

        label_metric = _annotation_metric_for_y_key(y_key)
        ours_label_y, _ = _estimate_value_at_target(
            ours_points,
            x_key="slo_violation_rate",
            y_key=label_metric,
            target=target,
        )
        baseline_label_y, _ = _estimate_value_at_target(
            series_by_method.get(str(best_baseline["method"]), []),
            x_key="slo_violation_rate",
            y_key=label_metric,
            target=target,
        )
        if ours_label_y is not None and baseline_label_y is not None:
            label = _format_gap_label(
                label_metric,
                float(ours_label_y),
                float(baseline_label_y),
            )
        else:
            label = _format_gap_label(y_key, float(ours_y), baseline_y)
        if label is not None and show_gap_label:
            label_bbox = {
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
            }
            if annotation_placement == "lower_left_of_ours":
                if (
                    y_key in {"average_n_active_servers", "active_gpu_proportion"}
                    and target <= TARGET_SLO_VIOLATIONS[0]
                ):
                    xytext = (14, -16)
                    ha = "left"
                else:
                    xytext = (5, -10)
                    ha = "right"
                ax.annotate(
                    label,
                    xy=(target, float(ours_y)),
                    xytext=xytext,
                    textcoords="offset points",
                    color=annotation_text_color,
                    fontsize=annotation_fontsize,
                    ha=ha,
                    va="top",
                    bbox=label_bbox,
                )
            elif annotation_placement == "below_ours":
                ax.annotate(
                    label,
                    xy=(target, float(ours_y)),
                    xytext=(0, -8),
                    textcoords="offset points",
                    color=annotation_text_color,
                    fontsize=annotation_fontsize,
                    ha="center",
                    va="top",
                    bbox=label_bbox,
                )
            elif annotation_placement == "clear_gap":
                y_min, y_max = ax.get_ylim()
                y_span = max(y_max - y_min, 1.0)
                text_x = min(max_slo_violation - 0.65, target + 0.75)
                if target <= TARGET_SLO_VIOLATIONS[0]:
                    unclamped_text_y = y_high + 0.10 * y_span
                    text_y = min(y_max - 0.12 * y_span, unclamped_text_y)
                    va = "top" if text_y < unclamped_text_y else "bottom"
                else:
                    unclamped_text_y = y_low - 0.10 * y_span
                    text_y = max(y_min + 0.10 * y_span, unclamped_text_y)
                    va = "bottom" if text_y > unclamped_text_y else "top"
                ax.text(
                    text_x,
                    text_y,
                    label,
                    color=annotation_text_color,
                    fontsize=annotation_fontsize,
                    ha="left",
                    va=va,
                    bbox=label_bbox,
                    zorder=6,
                )
            else:
                ax.annotate(
                    label,
                    xy=(target, (float(ours_y) + baseline_y) / 2.0),
                    xytext=(6, 10 if target <= TARGET_SLO_VIOLATIONS[0] else -10),
                    textcoords="offset points",
                    color=annotation_text_color,
                    fontsize=annotation_fontsize,
                    ha="left",
                    va="bottom" if target <= TARGET_SLO_VIOLATIONS[0] else "top",
                    bbox=label_bbox,
                )
        comparisons.append(
            {
                "y_key": y_key,
                "target": target,
                "ours_method": ours_method,
                "ours_y": float(ours_y),
                "baseline_method": str(best_baseline["method"]),
                "baseline_y": baseline_y,
                "label": label,
            }
        )
    return comparisons


def _plot_metric_pair(
    ax,
    series_by_method,
    method_order,
    style_by_method,
    x_key: str,
    y_key: str,
    *,
    sort_key: str | None = None,
    gradient_key: str | None = None,
    solid_markers: bool = False,
    line_collection_cls=None,
    style_scale: float = 1.0,
) -> None:
    for method in method_order:
        points = series_by_method.get(method)
        if not points:
            continue

        style = style_by_method[method]
        filtered_points = [
            point
            for point in points
            if point.get(x_key) is not None and point.get(y_key) is not None
        ]
        if not filtered_points:
            continue
        order_key = sort_key or x_key
        filtered_points.sort(key=lambda point: float(point[order_key]))
        xs = [float(point[x_key]) for point in filtered_points]
        ys = [float(point[y_key]) for point in filtered_points]
        line_xs, line_ys, segment_source_indices = _densify_polyline_for_breakpoints(
            xs,
            ys,
            _polyline_breakpoints_for_x_key(x_key),
        )

        if gradient_key is not None:
            gradient_values = [float(point[gradient_key]) for point in filtered_points]
            point_colors = _gradient_point_colors(str(style["color"]), gradient_values)
            if len(line_xs) > 1 and line_collection_cls is not None:
                segments = [
                    [(line_xs[idx], line_ys[idx]), (line_xs[idx + 1], line_ys[idx + 1])]
                    for idx in range(len(line_xs) - 1)
                ]
                segment_colors = [
                    point_colors[min(source_idx + 1, len(point_colors) - 1)]
                    for source_idx in segment_source_indices
                ]
                ax.add_collection(
                    line_collection_cls(
                        segments,
                        colors=segment_colors,
                        linewidths=max(2.0, float(style["linewidth"]) * style_scale),
                        zorder=2,
                    )
                )
            ax.scatter(
                xs,
                ys,
                s=(max(6.5, float(style["markersize"]) * style_scale) + 0.6) ** 2,
                marker=style["marker"],
                c=str(style["color"]) if solid_markers else point_colors,
                edgecolors=str(style["color"]),
                linewidths=0.9,
                zorder=3,
            )
        else:
            ax.plot(
                line_xs,
                line_ys,
                color=style["color"],
                linewidth=max(2.0, float(style["linewidth"]) * style_scale),
                zorder=2,
            )
            ax.plot(
                xs,
                ys,
                color=style["color"],
                marker=style["marker"],
                linestyle="None",
                markersize=max(6.5, float(style["markersize"]) * style_scale),
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                markeredgewidth=0.8,
                zorder=3,
            )


def _output_stem_for_serving_mode(
    output_stem: Path,
    serving_mode: str,
    split_count: int,
) -> Path:
    if split_count <= 1:
        return output_stem
    return output_stem.with_name(f"{output_stem.name}_{serving_mode}")


def _modeled_optimal_active_gpus_for_csv(csv_path: Path) -> float | None:
    return MODELED_OPTIMAL_ACTIVE_GPUS_BY_CSV_STEM.get(csv_path.stem)


def _render_triptych_figure(
    *,
    plt,
    line_collection_cls,
    line2d_cls,
    max_n_locator_cls,
    output_stem: Path,
    series_by_method: dict[str, list[dict[str, float | int | str | None]]],
    method_order: list[str],
    max_slo_violation: float,
    baseline_methods: list[str] | None = None,
    use_active_gpu_proportion: bool = False,
    panel_keys: tuple[str, ...] = DEFAULT_PANEL_KEYS,
    large_format: bool = False,
    show_legend: bool = True,
    single_column: bool = False,
    show_energy_gradient_note: bool = False,
    show_panel_titles: bool = True,
    annotation_text_color: str = ANNOTATION_TEXT_COLOR,
    annotation_placement: str = "lower_left_of_ours",
    show_threshold_gap_connectors: bool = True,
    show_threshold_gap_labels: bool = True,
    show_slo_reference_lines: bool = True,
    energy_y_min: float | None = None,
    energy_y_max: float | None = None,
    modeled_optimal_active_gpus: float | None = None,
    slo_violation_x_symlog_scale: bool = False,
    slo_violation_x_log_scale: bool = False,
    content_scale: float = 1.0,
) -> list[Path]:
    panel_keys = _normalize_panel_keys(panel_keys)
    nrows = 2 if (single_column and len(panel_keys) == 2) else 1
    ncols = 1 if (single_column and len(panel_keys) == 2) else len(panel_keys)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=_figure_size_for_panels(
            panel_keys,
            large_format=large_format,
            single_column=single_column,
        ),
    )
    if hasattr(axes, "flat"):
        axes = list(axes.flat)
    else:
        axes = [axes]
    fig.patch.set_facecolor("white")
    style_by_method = _build_method_styles(method_order)
    threshold_comparisons: list[dict[str, object]] = []
    font_scale = _font_scale_for_layout(
        panel_keys,
        large_format=large_format,
        single_column=single_column,
        content_scale=content_scale,
    )
    style_scale = (1.0 if single_column else 1.2 if large_format else 1.0) * content_scale
    annotation_fontsize = 10.5 * font_scale
    active_gpu_y_key = (
        "active_gpu_proportion"
        if use_active_gpu_proportion
        else "average_n_active_servers"
    )
    # Keep a small negative margin so the x=0 guide is visible inside the panel.
    include_zero_slo_left_margin = True
    axes_by_panel = dict(zip(panel_keys, axes))

    if "energy" in axes_by_panel:
        energy_ax = axes_by_panel["energy"]
        _plot_metric_pair(
            energy_ax,
            series_by_method,
            method_order,
            style_by_method,
            x_key="slo_violation_rate",
            y_key="energy_consumption",
            sort_key="n_device",
            gradient_key="n_device",
            solid_markers=True,
            line_collection_cls=line_collection_cls,
            style_scale=style_scale,
        )
        energy_ax.set_xlabel("SLO Violation Rate (%)", color="black")
        energy_ax.set_ylabel("Energy (kJ)", color="black")
        if show_panel_titles:
            energy_ax.set_title("Energy vs. SLO Violation", color="black", pad=8)
        if slo_violation_x_symlog_scale:
            _apply_slo_violation_symlog_xscale(
                energy_ax,
                series_by_method=series_by_method,
                max_slo_violation=max_slo_violation,
                include_left_margin=include_zero_slo_left_margin,
            )
        elif slo_violation_x_log_scale:
            _apply_slo_violation_log_xscale(
                energy_ax,
                series_by_method=series_by_method,
                max_slo_violation=max_slo_violation,
                include_left_margin=include_zero_slo_left_margin,
            )
        else:
            energy_ax.set_xlim(
                _slo_violation_x_lower_bound(
                    include_left_margin=include_zero_slo_left_margin,
                ),
                max_slo_violation,
            )
        if show_slo_reference_lines:
            _add_zero_slo_reference_line(energy_ax)
        current_energy_y_min, current_energy_y_max = energy_ax.get_ylim()
        if energy_y_min is not None or energy_y_max is not None:
            energy_ax.set_ylim(
                bottom=(
                    energy_y_min
                    if energy_y_min is not None
                    else current_energy_y_min
                ),
                top=(
                    energy_y_max
                    if energy_y_max is not None
                    else current_energy_y_max
                ),
            )
        else:
            energy_ax.set_ylim(bottom=0)
        threshold_comparisons.extend(
            _add_threshold_gap_annotations(
                energy_ax,
                series_by_method=series_by_method,
                method_order=method_order,
                y_key="energy_consumption",
                max_slo_violation=max_slo_violation,
                baseline_methods=baseline_methods,
                annotation_fontsize=annotation_fontsize,
                annotation_text_color=annotation_text_color,
                annotation_placement=annotation_placement,
                show_gap_connector=show_threshold_gap_connectors,
                show_gap_label=show_threshold_gap_labels,
                show_target_reference_lines=show_slo_reference_lines,
            )
        )
        if show_energy_gradient_note:
            energy_ax.text(
                0.03,
                0.97,
                "lighter: fewer GPUs\n darker: more GPUs",
                transform=energy_ax.transAxes,
                ha="left",
                va="top",
                color="black",
                fontsize=9.5 * font_scale,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.9,
                },
            )

    if "active_gpus" in axes_by_panel:
        active_gpu_ax = axes_by_panel["active_gpus"]
        _plot_metric_pair(
            active_gpu_ax,
            series_by_method,
            method_order,
            style_by_method,
            x_key="slo_violation_rate",
            y_key=active_gpu_y_key,
            style_scale=style_scale,
        )
        active_gpu_ax.set_xlabel("SLO Violation Rate (%)", color="black")
        active_gpu_ax.set_ylabel(
            "Avg. Active GPU Proportion" if use_active_gpu_proportion else "Avg. # Active GPUs",
            color="black",
        )
        if show_panel_titles:
            active_gpu_ax.set_title(
                "Active GPU Proportion vs. SLO Violation"
                if use_active_gpu_proportion
                else "Avg. Active GPUs vs. SLO Violation",
                color="black",
                pad=8,
            )
        if slo_violation_x_symlog_scale:
            _apply_slo_violation_symlog_xscale(
                active_gpu_ax,
                series_by_method=series_by_method,
                max_slo_violation=max_slo_violation,
                include_left_margin=include_zero_slo_left_margin,
            )
        elif slo_violation_x_log_scale:
            _apply_slo_violation_log_xscale(
                active_gpu_ax,
                series_by_method=series_by_method,
                max_slo_violation=max_slo_violation,
                include_left_margin=include_zero_slo_left_margin,
            )
        else:
            active_gpu_ax.set_xlim(
                _slo_violation_x_lower_bound(
                    include_left_margin=include_zero_slo_left_margin,
                ),
                max_slo_violation,
            )
        if show_slo_reference_lines:
            _add_zero_slo_reference_line(active_gpu_ax)
        active_gpu_ax.set_ylim(bottom=0)
        if use_active_gpu_proportion:
            active_gpu_ax.set_ylim(top=1.0)
        elif modeled_optimal_active_gpus is not None:
            active_gpu_ax.scatter(
                [0.0],
                [modeled_optimal_active_gpus],
                marker=MODELED_OPTIMAL_MARKER,
                s=(MODELED_OPTIMAL_MARKERSIZE * 1.12 * font_scale) ** 2,
                c=MODELED_OPTIMAL_MARKER_COLOR,
                edgecolors="black",
                linewidths=MODELED_OPTIMAL_MARKEREDGEWIDTH,
                zorder=5,
                clip_on=False,
            )
        threshold_comparisons.extend(
            _add_threshold_gap_annotations(
                active_gpu_ax,
                series_by_method=series_by_method,
                method_order=method_order,
                y_key=active_gpu_y_key,
                max_slo_violation=max_slo_violation,
                baseline_methods=baseline_methods,
                annotation_fontsize=annotation_fontsize,
                annotation_text_color=annotation_text_color,
                annotation_placement=annotation_placement,
                show_gap_connector=show_threshold_gap_connectors,
                show_gap_label=show_threshold_gap_labels,
                show_target_reference_lines=show_slo_reference_lines,
            )
        )

    if "slo_vs_gpus" in axes_by_panel:
        slo_vs_gpu_ax = axes_by_panel["slo_vs_gpus"]
        _plot_metric_pair(
            slo_vs_gpu_ax,
            series_by_method,
            method_order,
            style_by_method,
            x_key="n_device",
            y_key="slo_violation_rate",
            style_scale=style_scale,
        )
        slo_vs_gpu_ax.set_xlabel("# GPUs", color="black")
        slo_vs_gpu_ax.set_ylabel("SLO Violation Rate (%)", color="black")
        slo_vs_gpu_ax.set_xscale('log')
        if show_panel_titles:
            slo_vs_gpu_ax.set_title("SLO Violation vs. # GPUs", color="black", pad=8)
        slo_vs_gpu_ax.set_xlim(left=1)
        slo_vs_gpu_ax.set_ylim(0, max_slo_violation)
        slo_vs_gpu_ax.xaxis.set_major_locator(max_n_locator_cls(integer=True))

    for ax in axes:
        _style_axis(ax)

    legend_handles = []
    if show_legend:
        legend_handles = _build_legend_handles(
            line2d_cls,
            method_order,
            style_by_method,
            style_scale=style_scale,
            include_modeled_optimal=(
                modeled_optimal_active_gpus is not None
                and not use_active_gpu_proportion
            ),
        )
        if len(panel_keys) == 1:
            legend_ncol = min(2, len(legend_handles))
        else:
            legend_ncol = min(2 if single_column else 3, len(legend_handles))
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.998),
            ncol=legend_ncol,
            frameon=False,
            handlelength=2.0 * font_scale,
            columnspacing=1.1,
            handletextpad=0.6,
        )
    compact_two_panel_layout = (
        len(panel_keys) == 2
        and not single_column
        and not show_legend
        and not show_panel_titles
    )
    if compact_two_panel_layout:
        fig.tight_layout(
            rect=(0.0, 0.0, 1.0, 0.995),
            pad=0.15,
            w_pad=0.45,
            h_pad=0.15,
        )
        save_padding = 0.01
    else:
        legend_top = (
            0.98
            if not show_legend
            else 0.76 if len(panel_keys) == 1
            else 0.82 if (single_column and len(panel_keys) == 2 and not show_panel_titles)
            else 0.78 if (single_column and len(panel_keys) == 2)
            else 0.82 if len(panel_keys) == 1 else 0.85 if len(panel_keys) == 2 else 0.90
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, legend_top))
        save_padding = 0.04

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for suffix in (".png", ".pdf"):
        output_path = output_stem.with_suffix(suffix)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=save_padding)
        saved_paths.append(output_path)
    plt.close(fig)

    for output_path in saved_paths:
        print(f"Saved {output_path}")
    if threshold_comparisons:
        print("Threshold gap annotations")
        for comparison in threshold_comparisons:
            label = comparison["label"] or "gap unavailable"
            print(
                "  "
                f"{comparison['y_key']} @ {comparison['target']:.1f}%: "
                f"{comparison['ours_method']}={comparison['ours_y']:.3f}, "
                f"best baseline {comparison['baseline_method']}={comparison['baseline_y']:.3f} "
                f"({label})"
            )
    return saved_paths


def save_shared_legend(
    output_stem: Path | str,
    method_order: list[str] | tuple[str, ...],
    *,
    large_format: bool = True,
    ncol: int | None = None,
    include_modeled_optimal: bool = True,
    content_scale: float = 1.0,
) -> list[Path]:
    output_stem = Path(output_stem)
    deduped_methods: list[str] = []
    seen_labels: set[str] = set()
    for method in method_order:
        label = get_method_label(method)
        if label in seen_labels:
            continue
        deduped_methods.append(method)
        seen_labels.add(label)

    plt, _, Line2D, _ = _get_plotting_dependencies()
    _apply_paper_style(
        plt,
        large_format=large_format,
        content_scale=content_scale,
    )
    font_scale = (1.55 if large_format else 1.0) * content_scale
    style_scale = (1.2 if large_format else 1.0) * content_scale
    style_by_method = _build_method_styles(deduped_methods)
    legend_handles = _build_legend_handles(
        Line2D,
        deduped_methods,
        style_by_method,
        style_scale=style_scale,
        include_modeled_optimal=include_modeled_optimal,
    )
    ncol = ncol or min(4, len(legend_handles))
    nrow = max(1, math.ceil(len(legend_handles) / max(1, ncol)))
    fig_height = 0.42 + 0.36 * nrow * font_scale
    fig_width = 7.6 if ncol <= 4 else 9.2
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("white")
    fig.legend(
        handles=legend_handles,
        loc="center",
        ncol=ncol,
        frameon=False,
        handlelength=2.2 * font_scale,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for suffix in (".png", ".pdf"):
        output_path = output_stem.with_suffix(suffix)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        saved_paths.append(output_path)
    plt.close(fig)

    for output_path in saved_paths:
        print(f"Saved {output_path}")
    return saved_paths


def draw_n_device_triptych(
    csv_path: Path | str = DEFAULT_CSV_PATH,
    output_stem: Path | str | None = None,
    max_slo_violation: float = DEFAULT_MAX_SLO_VIOLATION,
    filter_baselines: bool = True,
    slo_packer_candidates_plus_best_baseline: bool = False,
    use_active_gpu_proportion: bool = False,
    panel_keys: tuple[str, ...] = DEFAULT_PANEL_KEYS,
    large_format: bool = False,
    show_legend: bool = True,
    single_column: bool = False,
    show_energy_gradient_note: bool = False,
    show_panel_titles: bool = True,
    annotation_text_color: str = ANNOTATION_TEXT_COLOR,
    annotation_placement: str = "lower_left_of_ours",
    exclude_method_substrings: tuple[str, ...] | list[str] | None = None,
    collapse_vllm_family: bool = False,
    collapse_baseline_families: bool = True,
    show_threshold_gap_connectors: bool = True,
    show_threshold_gap_labels: bool = True,
    show_slo_reference_lines: bool = True,
    energy_y_min: float | None = None,
    energy_y_max: float | None = None,
    slo_violation_x_symlog_scale: bool = False,
    slo_violation_x_log_scale: bool = False,
    content_scale: float = 1.0,
) -> list[Path]:
    csv_path = Path(csv_path)
    panel_keys = _normalize_panel_keys(panel_keys)
    if output_stem is None:
        output_stem = _default_output_stem(
            csv_path,
            max_slo_violation=max_slo_violation,
            panel_keys=panel_keys,
        )
    output_stem = Path(output_stem)
    if slo_packer_candidates_plus_best_baseline:
        output_stem = _append_output_suffix(output_stem, "_ablation")

    plt, LineCollection, Line2D, MaxNLocator = _get_plotting_dependencies()
    _apply_paper_style(
        plt,
        panel_keys=panel_keys,
        large_format=large_format,
        single_column=single_column,
        content_scale=content_scale,
    )

    rows, duplicate_count = _load_rows(csv_path)
    modeled_optimal_active_gpus = _modeled_optimal_active_gpus_for_csv(csv_path)
    series_by_method, method_order = _build_method_series(rows)
    series_by_method, method_order = _exclude_methods_by_substring(
        series_by_method,
        method_order,
        exclude_method_substrings,
    )
    if not method_order:
        raise ValueError(
            f"No methods left to plot in {csv_path} after applying the method exclusion filter."
        )
    if not slo_packer_candidates_plus_best_baseline and filter_baselines:
        series_by_method, method_order = _filter_default_e2e_methods(
            series_by_method,
            method_order,
        )
        if not method_order:
            raise ValueError(
                f"No methods left to plot in {csv_path} after applying the e2e method filter."
            )
    if collapse_vllm_family:
        (
            series_by_method,
            method_order,
            family_selection_details,
        ) = _collapse_baseline_families(
            series_by_method,
            method_order,
            family_specs=DEFAULT_VLLM_BASELINE_FAMILY_COLLAPSE,
        )
        if family_selection_details:
            for canonical_method, raw_method in family_selection_details.items():
                print(
                    f"Collapsed vLLM family {canonical_method} <- {raw_method}"
                )
    elif collapse_baseline_families:
        (
            series_by_method,
            method_order,
            family_selection_details,
        ) = _collapse_baseline_families(
            series_by_method,
            method_order,
        )
        if family_selection_details:
            for canonical_method, raw_method in family_selection_details.items():
                print(
                    f"Collapsed baseline family {canonical_method} <- {raw_method}"
                )
    grouped_by_mode = _split_methods_by_serving_mode(series_by_method, method_order)
    saved_paths = []
    for serving_mode, (mode_series_by_method, mode_method_order) in grouped_by_mode.items():
        comparison_baseline_methods: list[str] | None = None
        if slo_packer_candidates_plus_best_baseline:
            (
                mode_series_by_method,
                mode_method_order,
                comparison_baseline_methods,
            ) = _filter_to_slo_packer_candidates_plus_best_baseline(
                mode_series_by_method,
                mode_method_order,
            )
            if not mode_method_order:
                continue
            if comparison_baseline_methods:
                print(
                    f"{serving_mode}: selected best baseline "
                    f"{comparison_baseline_methods[0]}"
                )
            else:
                print(f"{serving_mode}: no baseline matched the selection rule")
        saved_paths.extend(
            _render_triptych_figure(
                plt=plt,
                line_collection_cls=LineCollection,
                line2d_cls=Line2D,
                max_n_locator_cls=MaxNLocator,
                output_stem=_output_stem_for_serving_mode(
                    output_stem=output_stem,
                    serving_mode=serving_mode,
                    split_count=len(grouped_by_mode),
                ),
                series_by_method=mode_series_by_method,
                method_order=mode_method_order,
                max_slo_violation=max_slo_violation,
                baseline_methods=comparison_baseline_methods,
                use_active_gpu_proportion=use_active_gpu_proportion,
                panel_keys=panel_keys,
                large_format=large_format,
                show_legend=show_legend,
                single_column=single_column,
                show_energy_gradient_note=show_energy_gradient_note,
                show_panel_titles=show_panel_titles,
                annotation_text_color=annotation_text_color,
                annotation_placement=annotation_placement,
                show_threshold_gap_connectors=show_threshold_gap_connectors,
                show_threshold_gap_labels=show_threshold_gap_labels,
                show_slo_reference_lines=show_slo_reference_lines,
                energy_y_min=energy_y_min,
                energy_y_max=energy_y_max,
                modeled_optimal_active_gpus=modeled_optimal_active_gpus,
                slo_violation_x_symlog_scale=slo_violation_x_symlog_scale,
                slo_violation_x_log_scale=slo_violation_x_log_scale,
                content_scale=content_scale,
            )
        )

    if duplicate_count:
        print(
            f"Retained the last CSV row for {duplicate_count} duplicate "
            "(method, n_device) points."
        )
    return saved_paths


def draw_all_n_device_triptychs(
    input_dir: Path | str = DEFAULT_INPUT_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    max_slo_violation: float = DEFAULT_MAX_SLO_VIOLATION,
    filter_baselines: bool = True,
    slo_packer_candidates_plus_best_baseline: bool = False,
    use_active_gpu_proportion: bool = False,
    panel_keys: tuple[str, ...] = DEFAULT_PANEL_KEYS,
    large_format: bool = False,
    show_legend: bool = True,
    single_column: bool = False,
    show_energy_gradient_note: bool = False,
    show_panel_titles: bool = True,
    annotation_text_color: str = ANNOTATION_TEXT_COLOR,
    annotation_placement: str = "lower_left_of_ours",
    exclude_method_substrings: tuple[str, ...] | list[str] | None = None,
    collapse_vllm_family: bool = False,
    collapse_baseline_families: bool = True,
    show_threshold_gap_connectors: bool = True,
    show_threshold_gap_labels: bool = True,
    show_slo_reference_lines: bool = True,
    energy_y_min: float | None = None,
    energy_y_max: float | None = None,
    slo_violation_x_symlog_scale: bool = False,
    slo_violation_x_log_scale: bool = False,
    content_scale: float = 1.0,
) -> list[Path]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    panel_keys = _normalize_panel_keys(panel_keys)

    saved_paths: list[Path] = []
    failures: list[tuple[Path, Exception]] = []
    for csv_path in _iter_csv_files(input_dir):
        output_stem = _default_output_stem(
            csv_path,
            max_slo_violation=max_slo_violation,
            output_dir=output_dir,
            panel_keys=panel_keys,
        )
        try:
            saved_paths.extend(
                draw_n_device_triptych(
                    csv_path=csv_path,
                    output_stem=output_stem,
                    max_slo_violation=max_slo_violation,
                    filter_baselines=filter_baselines,
                    slo_packer_candidates_plus_best_baseline=(
                        slo_packer_candidates_plus_best_baseline
                    ),
                    use_active_gpu_proportion=use_active_gpu_proportion,
                    panel_keys=panel_keys,
                    large_format=large_format,
                    show_legend=show_legend,
                    single_column=single_column,
                    show_energy_gradient_note=show_energy_gradient_note,
                    show_panel_titles=show_panel_titles,
                    annotation_text_color=annotation_text_color,
                    annotation_placement=annotation_placement,
                    exclude_method_substrings=exclude_method_substrings,
                    collapse_vllm_family=collapse_vllm_family,
                    collapse_baseline_families=collapse_baseline_families,
                    show_threshold_gap_connectors=show_threshold_gap_connectors,
                    show_threshold_gap_labels=show_threshold_gap_labels,
                    show_slo_reference_lines=show_slo_reference_lines,
                    energy_y_min=energy_y_min,
                    energy_y_max=energy_y_max,
                    slo_violation_x_symlog_scale=slo_violation_x_symlog_scale,
                    slo_violation_x_log_scale=slo_violation_x_log_scale,
                    content_scale=content_scale,
                )
            )
        except Exception as exc:
            failures.append((csv_path, exc))

    print(
        f"Processed {len(saved_paths) // 2} figure set(s) from {input_dir} "
        f"with SLO-violation axis capped at {max_slo_violation}%."
    )
    if failures:
        failure_lines = "\n".join(f"- {path}: {exc}" for path, exc in failures)
        raise RuntimeError(f"Failed to render some CSVs:\n{failure_lines}")
    return saved_paths


def draw_chat_7b_n_device_triptych(
    csv_path: Path | str = DEFAULT_CSV_PATH,
    output_stem: Path | str | None = None,
    max_slo_violation: float = DEFAULT_MAX_SLO_VIOLATION,
    filter_baselines: bool = True,
    slo_packer_candidates_plus_best_baseline: bool = False,
    use_active_gpu_proportion: bool = False,
    panel_keys: tuple[str, ...] = DEFAULT_PANEL_KEYS,
    large_format: bool = False,
    show_legend: bool = True,
    single_column: bool = False,
    show_energy_gradient_note: bool = False,
    show_panel_titles: bool = True,
    annotation_text_color: str = ANNOTATION_TEXT_COLOR,
    annotation_placement: str = "lower_left_of_ours",
    exclude_method_substrings: tuple[str, ...] | list[str] | None = None,
    collapse_vllm_family: bool = False,
    collapse_baseline_families: bool = True,
    show_threshold_gap_connectors: bool = True,
    show_threshold_gap_labels: bool = True,
    show_slo_reference_lines: bool = True,
    energy_y_min: float | None = None,
    energy_y_max: float | None = None,
    slo_violation_x_symlog_scale: bool = False,
    slo_violation_x_log_scale: bool = False,
    content_scale: float = 1.0,
) -> list[Path]:
    return draw_n_device_triptych(
        csv_path=csv_path,
        output_stem=output_stem,
        max_slo_violation=max_slo_violation,
        filter_baselines=filter_baselines,
        slo_packer_candidates_plus_best_baseline=(
            slo_packer_candidates_plus_best_baseline
        ),
        use_active_gpu_proportion=use_active_gpu_proportion,
        panel_keys=panel_keys,
        large_format=large_format,
        show_legend=show_legend,
        single_column=single_column,
        show_energy_gradient_note=show_energy_gradient_note,
        show_panel_titles=show_panel_titles,
        annotation_text_color=annotation_text_color,
        annotation_placement=annotation_placement,
        exclude_method_substrings=exclude_method_substrings,
        collapse_vllm_family=collapse_vllm_family,
        collapse_baseline_families=collapse_baseline_families,
        show_threshold_gap_connectors=show_threshold_gap_connectors,
        show_threshold_gap_labels=show_threshold_gap_labels,
        show_slo_reference_lines=show_slo_reference_lines,
        energy_y_min=energy_y_min,
        energy_y_max=energy_y_max,
        slo_violation_x_symlog_scale=slo_violation_x_symlog_scale,
        slo_violation_x_log_scale=slo_violation_x_log_scale,
        content_scale=content_scale,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render an n_device-sweep figure for CSVs in Paper/data/e2e using "
            "comparator labels and styles from plots/common.py."
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
        help="Output path without extension. Defaults to Paper/figs/e2e_new/<csv-stem>_n_device_<layout>_slo_cap_<threshold>.",
    )
    parser.add_argument(
        "--draw-all",
        action="store_true",
        help="Render figures for every CSV under --input-dir. Ignores --csv and --output-stem.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory scanned by --draw-all. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory used by --draw-all. Default: %(default)s",
    )
    parser.add_argument(
        "--max-slo-violation",
        type=float,
        default=DEFAULT_MAX_SLO_VIOLATION,
        help="Cap the displayed SLO-violation axis at this percentage while still loading all points. Default: %(default)s",
    )
    parser.add_argument(
        "--panels",
        nargs="+",
        choices=DEFAULT_PANEL_KEYS,
        default=list(DEFAULT_PANEL_KEYS),
        help=(
            "Panel subset to render, in left-to-right order. "
            "Default: energy active_gpus slo_vs_gpus"
        ),
    )
    parser.add_argument(
        "--large-format",
        action="store_true",
        help=(
            "Use larger typography and a roomier canvas. "
            "Useful for two-panel paper figures."
        ),
    )
    parser.add_argument(
        "--content-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply the internal typography, marker, and legend sizing without "
            "changing the overall panel canvas. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--hide-legend",
        action="store_true",
        help="Suppress the per-figure legend. Useful when the paper provides a shared legend.",
    )
    parser.add_argument(
        "--single-column",
        action="store_true",
        help="Use a compact canvas and larger typography tuned for a single-column figure.",
    )
    parser.add_argument(
        "--hide-energy-gradient-note",
        dest="show_energy_gradient_note",
        action="store_false",
        help="Omit the in-panel 'lighter/darker GPUs' note from the energy panel.",
    )
    parser.add_argument(
        "--show-energy-gradient-note",
        dest="show_energy_gradient_note",
        action="store_true",
        help="Show the in-panel 'lighter/darker GPUs' note in the energy panel.",
    )
    parser.add_argument(
        "--hide-panel-titles",
        action="store_true",
        help="Omit per-panel titles such as 'Energy vs. SLO Violation'.",
    )
    parser.add_argument(
        "--annotation-placement",
        choices=("lower_left_of_ours", "midline", "below_ours", "clear_gap"),
        default="lower_left_of_ours",
        help="Where to place the threshold-gap labels. Default: %(default)s",
    )
    parser.add_argument(
        "--annotation-text-color",
        default=ANNOTATION_TEXT_COLOR,
        help="Text color for the threshold-gap labels. Default: %(default)s",
    )
    parser.add_argument(
        "--hide-threshold-gap-connectors",
        action="store_true",
        help=(
            "Hide the solid vertical connector segment between SLO Packer and "
            "the comparison baseline at each annotated threshold."
        ),
    )
    parser.add_argument(
        "--hide-threshold-gap-labels",
        action="store_true",
        help="Hide the `%% downarrow` threshold-gap labels.",
    )
    parser.add_argument(
        "--energy-y-min",
        type=float,
        help="Optional lower bound for the energy-panel y-axis.",
    )
    parser.add_argument(
        "--energy-y-max",
        type=float,
        help="Optional upper bound for the energy-panel y-axis.",
    )
    parser.add_argument(
        "--exclude-method-substring",
        action="append",
        default=[],
        help=(
            "Exclude any method whose raw id or display label contains this "
            "case-insensitive substring. May be provided multiple times."
        ),
    )
    parser.add_argument(
        "--collapse-vllm-family",
        action="store_true",
        help=(
            "Collapse only the vLLM-family baselines so `qlm / round_robin` "
            "and `vllm / round_robin` render as a single canonical `vLLM` curve. "
            "Session-aware variants collapse separately to `vLLM (Session-Aware)`."
        ),
    )
    parser.add_argument(
        "--collapse-baseline-families",
        dest="collapse_baseline_families",
        action="store_true",
        help=(
            "Collapse the vLLM-family and Sarathi-family baselines to one "
            "best-performing canonical curve per family."
        ),
    )
    parser.add_argument(
        "--no-collapse-baseline-families",
        dest="collapse_baseline_families",
        action="store_false",
        help="Disable the default collapse of the vLLM and Sarathi baseline families.",
    )
    parser.add_argument(
        "--keep-all-baselines",
        action="store_true",
        help=(
            "Disable the default e2e baseline filter and plot every method found in the CSV, "
            "including non-primary ATFC baselines."
        ),
    )
    parser.add_argument(
        "--slo-packer-candidates-plus-best-baseline",
        action="store_true",
        help=(
            "Plot all methods that use `atfc` as the scheduler plus a single best baseline. "
            "The best baseline is selected per serving mode using the lowest average "
            "interpolated energy at the 1%% and 5%% SLO-violation targets."
        ),
    )
    parser.add_argument(
        "--slo-violation-log-x",
        action="store_true",
        help=(
            "Render any panel that uses SLO violation on the x-axis with a "
            "piecewise focus scale: 0 remains visible, while the 1-5%% region "
            "gets most of the horizontal space."
        ),
    )
    parser.add_argument(
        "--slo-violation-symlog-x",
        action="store_true",
        help=(
            "Render any panel that uses SLO violation on the x-axis with a "
            "symlog scale: 0-1%% remains linear while values above 1%% are "
            "log-scaled."
        ),
    )
    parser.add_argument(
        "--hide-slo-reference-lines",
        action="store_true",
        help=(
            "Hide the vertical SLO reference lines, including the 0%% line "
            "and the dotted 1%%/5%% target guides."
        ),
    )
    parser.set_defaults(
        use_active_gpu_proportion=False,
        show_energy_gradient_note=False,
        collapse_baseline_families=True,
    )
    parser.add_argument(
        "--use-active-gpu-proportion",
        action="store_true",
        help=(
            "Use average active GPU proportion (`average_n_active_servers / n_device`) "
            "in the middle panel instead of the default raw average number of active GPUs."
        ),
    )
    parser.add_argument(
        "--use-raw-active-gpus",
        dest="use_active_gpu_proportion",
        action="store_false",
        help=(
            "Use the raw average number of active GPUs in the middle panel. This is the default."
        ),
    )
    args = parser.parse_args()
    panel_keys = _normalize_panel_keys(tuple(args.panels))

    if args.draw_all:
        draw_all_n_device_triptychs(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_slo_violation=args.max_slo_violation,
            filter_baselines=not args.keep_all_baselines,
            slo_packer_candidates_plus_best_baseline=(
                args.slo_packer_candidates_plus_best_baseline
            ),
            use_active_gpu_proportion=args.use_active_gpu_proportion,
            panel_keys=panel_keys,
            large_format=args.large_format,
            show_legend=not args.hide_legend,
            single_column=args.single_column,
            show_energy_gradient_note=args.show_energy_gradient_note,
            show_panel_titles=not args.hide_panel_titles,
            annotation_text_color=args.annotation_text_color,
            annotation_placement=args.annotation_placement,
            exclude_method_substrings=args.exclude_method_substring,
            collapse_vllm_family=args.collapse_vllm_family,
            collapse_baseline_families=args.collapse_baseline_families,
            show_threshold_gap_connectors=not args.hide_threshold_gap_connectors,
            show_threshold_gap_labels=not args.hide_threshold_gap_labels,
            show_slo_reference_lines=not args.hide_slo_reference_lines,
            energy_y_min=args.energy_y_min,
            energy_y_max=args.energy_y_max,
            slo_violation_x_symlog_scale=args.slo_violation_symlog_x,
            slo_violation_x_log_scale=args.slo_violation_log_x,
            content_scale=args.content_scale,
        )
    else:
        draw_n_device_triptych(
            csv_path=args.csv,
            output_stem=args.output_stem,
            max_slo_violation=args.max_slo_violation,
            filter_baselines=not args.keep_all_baselines,
            slo_packer_candidates_plus_best_baseline=(
                args.slo_packer_candidates_plus_best_baseline
            ),
            use_active_gpu_proportion=args.use_active_gpu_proportion,
            panel_keys=panel_keys,
            large_format=args.large_format,
            show_legend=not args.hide_legend,
            single_column=args.single_column,
            show_energy_gradient_note=args.show_energy_gradient_note,
            show_panel_titles=not args.hide_panel_titles,
            annotation_text_color=args.annotation_text_color,
            annotation_placement=args.annotation_placement,
            exclude_method_substrings=args.exclude_method_substring,
            collapse_vllm_family=args.collapse_vllm_family,
            collapse_baseline_families=args.collapse_baseline_families,
            show_threshold_gap_connectors=not args.hide_threshold_gap_connectors,
            show_threshold_gap_labels=not args.hide_threshold_gap_labels,
            show_slo_reference_lines=not args.hide_slo_reference_lines,
            energy_y_min=args.energy_y_min,
            energy_y_max=args.energy_y_max,
            slo_violation_x_symlog_scale=args.slo_violation_symlog_x,
            slo_violation_x_log_scale=args.slo_violation_log_x,
            content_scale=args.content_scale,
        )


if __name__ == "__main__":
    main()
