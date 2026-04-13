from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

try:
    from plots.common import get_method_label, get_method_style
except ModuleNotFoundError:
    from common import get_method_label, get_method_style


DEFAULT_INPUT_DIR = Path("Paper/data/e2e")
DEFAULT_CSV_PATH = DEFAULT_INPUT_DIR / "chat_7B.csv"
DEFAULT_OUTPUT_DIR = Path("Paper/figs/e2e_new")
FIGSIZE = (14.8, 4.6)
CONSTANT_FIELDS = ("load_scale", "ttft_slo_scale", "slo_tpot", "perf_model_err")
DEFAULT_MAX_SLO_VIOLATION = 10.0


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


def _apply_paper_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.5,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


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
) -> Path:
    suffix = str(int(max_slo_violation)) if float(max_slo_violation).is_integer() else str(max_slo_violation).replace(".", "p")
    return output_dir / f"{csv_path.stem}_n_device_triptych_slo_cap_{suffix}"


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
            }
            energy_consumption = _parse_optional_float(row, "energy_consumption")
            if energy_consumption is not None:
                parsed_row["energy_consumption"] = energy_consumption / 1000.0
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


def _is_primary_ours_method(method: str) -> bool:
    return method in {
        "atfc / slosserve_planner",
        "atfc / slosserve_disagg_planner",
    }


def _method_serving_mode(method: str) -> str:
    return "disagg" if "disagg" in method else "normal"


def _keep_e2e_method(method: str) -> bool:
    return _is_primary_ours_method(method) or not method.startswith("atfc /")


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


def _plot_metric_pair(
    ax,
    series_by_method,
    method_order,
    x_key: str,
    y_key: str,
    *,
    sort_key: str | None = None,
    gradient_key: str | None = None,
    line_collection_cls=None,
) -> None:
    for method in method_order:
        points = series_by_method.get(method)
        if not points:
            continue

        style = get_method_style(method)
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

        if gradient_key is not None:
            gradient_values = [float(point[gradient_key]) for point in filtered_points]
            point_colors = _gradient_point_colors(str(style["color"]), gradient_values)
            if len(xs) > 1 and line_collection_cls is not None:
                segments = [
                    [(xs[idx], ys[idx]), (xs[idx + 1], ys[idx + 1])]
                    for idx in range(len(xs) - 1)
                ]
                segment_colors = [
                    point_colors[min(idx + 1, len(point_colors) - 1)]
                    for idx in range(len(segments))
                ]
                ax.add_collection(
                    line_collection_cls(
                        segments,
                        colors=segment_colors,
                        linewidths=max(2.0, float(style["linewidth"])),
                        zorder=2,
                    )
                )
            ax.scatter(
                xs,
                ys,
                s=(max(6.5, float(style["markersize"])) + 0.6) ** 2,
                marker=style["marker"],
                c=point_colors,
                edgecolors=str(style["color"]),
                linewidths=0.9,
                zorder=3,
            )
        else:
            ax.plot(
                xs,
                ys,
                color=style["color"],
                marker=style["marker"],
                linewidth=max(2.0, float(style["linewidth"])),
                markersize=max(6.5, float(style["markersize"])),
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                markeredgewidth=0.8,
            )


def _output_stem_for_serving_mode(
    output_stem: Path,
    serving_mode: str,
    split_count: int,
) -> Path:
    if split_count <= 1:
        return output_stem
    return output_stem.with_name(f"{output_stem.name}_{serving_mode}")


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
) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    fig.patch.set_facecolor("white")

    _plot_metric_pair(
        axes[0],
        series_by_method,
        method_order,
        x_key="slo_violation_rate",
        y_key="energy_consumption",
        sort_key="n_device",
        gradient_key="n_device",
        line_collection_cls=line_collection_cls,
    )
    axes[0].set_xlabel("SLO Violation Rate (%)", color="black")
    axes[0].set_ylabel("Energy Consumption (kJ)", color="black")
    axes[0].set_title("Energy vs. SLO Violation", color="black", pad=8)
    axes[0].set_xlim(0, max_slo_violation)
    axes[0].set_ylim(bottom=0)
    axes[0].text(
        0.03,
        0.97,
        "lighter: fewer GPUs\n darker: more GPUs",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        color="black",
        fontsize=9.5,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )

    _plot_metric_pair(
        axes[1],
        series_by_method,
        method_order,
        x_key="slo_violation_rate",
        y_key="average_n_active_servers",
    )
    axes[1].set_xlabel("SLO Violation Rate (%)", color="black")
    axes[1].set_ylabel("Avg. # Active GPUs", color="black")
    axes[1].set_title("Avg. Active GPUs vs. SLO Violation", color="black", pad=8)
    axes[1].set_xlim(0, max_slo_violation)
    axes[1].set_ylim(bottom=0)

    _plot_metric_pair(
        axes[2],
        series_by_method,
        method_order,
        x_key="n_device",
        y_key="slo_violation_rate",
    )
    axes[2].set_xlabel("# GPUs", color="black")
    axes[2].set_ylabel("SLO Violation Rate (%)", color="black")
    axes[2].set_title("SLO Violation vs. # GPUs", color="black", pad=8)
    axes[2].set_xlim(left=1)
    axes[2].set_ylim(0, max_slo_violation)
    axes[2].xaxis.set_major_locator(max_n_locator_cls(integer=True))

    for ax in axes:
        _style_axis(ax)

    legend_handles = []
    for method in method_order:
        style = get_method_style(method)
        legend_handles.append(
            line2d_cls(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linewidth=max(2.0, float(style["linewidth"])),
                markersize=max(6.5, float(style["markersize"])),
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                label=get_method_label(method),
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(3, len(legend_handles)),
        frameon=False,
        handlelength=2.0,
        columnspacing=1.1,
        handletextpad=0.6,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

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


def draw_n_device_triptych(
    csv_path: Path | str = DEFAULT_CSV_PATH,
    output_stem: Path | str | None = None,
    max_slo_violation: float = DEFAULT_MAX_SLO_VIOLATION,
) -> list[Path]:
    csv_path = Path(csv_path)
    if output_stem is None:
        output_stem = _default_output_stem(csv_path, max_slo_violation=max_slo_violation)
    output_stem = Path(output_stem)

    plt, LineCollection, Line2D, MaxNLocator = _get_plotting_dependencies()
    _apply_paper_style(plt)

    rows, duplicate_count = _load_rows(csv_path)
    series_by_method, method_order = _build_method_series(rows)
    method_order = [method for method in method_order if _keep_e2e_method(method)]
    series_by_method = {
        method: series_by_method[method]
        for method in method_order
        if method in series_by_method
    }
    if not method_order:
        raise ValueError(
            f"No methods left to plot in {csv_path} after applying the e2e method filter."
        )
    grouped_by_mode = _split_methods_by_serving_mode(series_by_method, method_order)
    saved_paths = []
    for serving_mode, (mode_series_by_method, mode_method_order) in grouped_by_mode.items():
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
) -> list[Path]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    saved_paths: list[Path] = []
    failures: list[tuple[Path, Exception]] = []
    for csv_path in _iter_csv_files(input_dir):
        output_stem = _default_output_stem(
            csv_path,
            max_slo_violation=max_slo_violation,
            output_dir=output_dir,
        )
        try:
            saved_paths.extend(
                draw_n_device_triptych(
                    csv_path=csv_path,
                    output_stem=output_stem,
                    max_slo_violation=max_slo_violation,
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
) -> list[Path]:
    return draw_n_device_triptych(
        csv_path=csv_path,
        output_stem=output_stem,
        max_slo_violation=max_slo_violation,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render a 3-panel n_device-sweep figure for CSVs in Paper/data/e2e using "
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
        help="Output path without extension. Defaults to Paper/figs/e2e_new/<csv-stem>_n_device_triptych_slo_cap_<threshold>.",
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
    args = parser.parse_args()

    if args.draw_all:
        draw_all_n_device_triptychs(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_slo_violation=args.max_slo_violation,
        )
    else:
        draw_n_device_triptych(
            csv_path=args.csv,
            output_stem=args.output_stem,
            max_slo_violation=args.max_slo_violation,
        )


if __name__ == "__main__":
    main()
