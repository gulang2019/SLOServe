from __future__ import annotations

import argparse
import csv
import re
from collections import OrderedDict
from pathlib import Path

try:
    from plots.common import get_method_label, get_method_style
except ModuleNotFoundError:
    from common import get_method_label, get_method_style


DEFAULT_INPUT_DIR = Path("Paper/data")
DEFAULT_GLOB_PATTERN = "sensitivity_load_scale*.csv"
DEFAULT_CSV_PATH = DEFAULT_INPUT_DIR / "sensitivity_load_scale_chat23.csv"
DEFAULT_OUTPUT_DIR = Path("Paper/figs")
FIGSIZE = (6.4, 4.6)
CONSTANT_FIELDS = ("n_device", "ttft_slo_scale", "slo_tpot", "perf_model_err")
DEFAULT_MAX_SLO_VIOLATION = 100.0
OURS_METHOD_PREFERENCE = (
    "atfc / slosserve_planner",
    "atfc / slosserve_disagg_planner",
    "atfc / round_robin",
)
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


def _slo_cap_suffix(max_slo_violation: float) -> str:
    if float(max_slo_violation).is_integer():
        return str(int(max_slo_violation))
    return str(max_slo_violation).replace(".", "p")


def _default_output_stem(
    csv_path: Path,
    max_slo_violation: float,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    label = _slugify(_dataset_label(csv_path))
    base_name = f"slo_violation_rate_vs_load_scale-{label}"
    if max_slo_violation != DEFAULT_MAX_SLO_VIOLATION:
        base_name = f"{base_name}_slo_cap_{_slo_cap_suffix(max_slo_violation)}"
    return output_dir / base_name


def _iter_csv_files(
    input_dir: Path | str,
    glob_pattern: str = DEFAULT_GLOB_PATTERN,
) -> list[Path]:
    input_dir = Path(input_dir)
    csv_paths = sorted(path for path in input_dir.glob(glob_pattern) if path.is_file())
    if not csv_paths:
        raise ValueError(f"No CSV files matching {glob_pattern!r} found in {input_dir}")
    return csv_paths


def _load_rows(
    csv_path: Path | str,
) -> tuple[list[dict[str, float | str | None]], int, str]:
    csv_path = Path(csv_path)
    required_fields = {
        "scheduling_policy",
        "routing_policy",
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
        x_field = "rps" if "rps" in fieldnames else "load_scale"
        if x_field not in fieldnames:
            raise ValueError(
                f"{csv_path} must include either an 'rps' or 'load_scale' column."
            )

        constant_context: dict[str, set[str]] = {
            field: set() for field in CONSTANT_FIELDS if field in fieldnames
        }
        deduped_rows: OrderedDict[tuple[str, float], dict[str, float | str | None]] = OrderedDict()
        duplicate_count = 0

        for row in reader:
            for field in constant_context:
                value = row.get(field, "")
                if value != "":
                    constant_context[field].add(value)

            method = f"{row['scheduling_policy']} / {row['routing_policy']}"
            parsed_row = {
                "method": method,
                "x_value": _parse_required_float(row, x_field),
                "slo_violation_rate": _parse_required_float(row, "slo_violation_rate") * 100.0,
                "rejection_rate": None,
            }
            rejection_rate = _parse_optional_float(row, "rejection_rate")
            if rejection_rate is not None:
                parsed_row["rejection_rate"] = rejection_rate * 100.0
            dedupe_key = (method, float(parsed_row["x_value"]))
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
            "This figure expects load_scale to be the only swept variable. "
            f"Found additional varying context in {csv_path}: {details}"
        )

    rows = list(deduped_rows.values())
    if not rows:
        raise ValueError(f"No rows found in {csv_path}.")
    return rows, duplicate_count, x_field


def _build_method_series(
    rows: list[dict[str, float | str | None]],
) -> tuple[dict[str, list[dict[str, float | str | None]]], list[str]]:
    series_by_method: dict[str, list[dict[str, float | str | None]]] = OrderedDict()
    method_order: OrderedDict[str, None] = OrderedDict()

    for row in rows:
        method = str(row["method"])
        method_order.setdefault(method, None)
        series_by_method.setdefault(method, []).append(row)

    for points in series_by_method.values():
        points.sort(key=lambda point: float(point["x_value"]))

    return series_by_method, list(method_order.keys())


def _method_serving_mode(method: str) -> str:
    return "disagg" if "disagg" in method else "normal"


def _split_methods_by_serving_mode(
    series_by_method: dict[str, list[dict[str, float | str | None]]],
    method_order: list[str],
) -> OrderedDict[str, tuple[dict[str, list[dict[str, float | str | None]]], list[str]]]:
    grouped: OrderedDict[str, tuple[dict[str, list[dict[str, float | str | None]]], list[str]]] = OrderedDict()
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


def _plot_slo_violation_vs_load_scale(
    ax,
    series_by_method: dict[str, list[dict[str, float | str | None]]],
    method_order: list[str],
) -> None:
    for method in method_order:
        points = series_by_method.get(method)
        if not points:
            continue

        style = get_method_style(method)
        xs = [float(point["x_value"]) for point in points]
        ys = [float(point["slo_violation_rate"]) for point in points]
        if not xs:
            continue

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


def _find_ours_method(method_order: list[str]) -> str | None:
    for candidate in OURS_METHOD_PREFERENCE:
        if candidate in method_order:
            return candidate
    for method in method_order:
        if method.startswith("atfc /"):
            return method
    return None


def _plot_ours_rejection_rate(
    ax,
    series_by_method: dict[str, list[dict[str, float | str | None]]],
    ours_method: str | None,
) -> bool:
    if ours_method is None:
        return False

    points = series_by_method.get(ours_method, [])
    filtered_points = [
        point for point in points if point.get("rejection_rate") is not None
    ]
    if not filtered_points:
        return False

    style = get_method_style(ours_method)
    xs = [float(point["x_value"]) for point in filtered_points]
    ys = [float(point["rejection_rate"]) for point in filtered_points]
    ax.plot(
        xs,
        ys,
        color=style["color"],
        linestyle="--",
        marker=style["marker"],
        linewidth=max(1.8, float(style["linewidth"]) - 0.1),
        markersize=max(6.0, float(style["markersize"])),
        markerfacecolor="white",
        markeredgecolor=style["color"],
        markeredgewidth=1.0,
        zorder=4,
    )
    return True


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
    series_by_method: dict[str, list[dict[str, float | str | None]]],
    method_order: list[str],
    max_slo_violation: float,
    title: str,
    x_label: str,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")

    _plot_slo_violation_vs_load_scale(ax, series_by_method, method_order)
    ours_method = _find_ours_method(method_order)
    has_ours_rejection = _plot_ours_rejection_rate(ax, series_by_method, ours_method)
    ax.set_xlabel(x_label, color="black")
    ax.set_ylabel("SLO Violation / Rejection Rate (%)", color="black")
    ax.set_title(title, color="black", pad=8)
    ax.set_xlim(left=0)
    ax.set_ylim(0, max_slo_violation)
    ax.xaxis.set_major_formatter(func_formatter_cls(lambda value, _pos: f"{value:g}"))
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
        if has_ours_rejection and method == ours_method:
            legend_handles.append(
                line2d_cls(
                    [0],
                    [0],
                    color=style["color"],
                    linestyle="--",
                    marker=style["marker"],
                    linewidth=max(1.8, float(style["linewidth"]) - 0.1),
                    markersize=max(6.0, float(style["markersize"])),
                    markerfacecolor="white",
                    markeredgecolor=style["color"],
                    markeredgewidth=1.0,
                    label=f"{get_method_label(method)} Rejection Rate",
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


def draw_slo_violation_vs_load_scale(
    csv_path: Path | str = DEFAULT_CSV_PATH,
    output_stem: Path | str | None = None,
    max_slo_violation: float = DEFAULT_MAX_SLO_VIOLATION,
) -> list[Path]:
    csv_path = Path(csv_path)
    if output_stem is None:
        output_stem = _default_output_stem(csv_path, max_slo_violation=max_slo_violation)
    output_stem = Path(output_stem)

    plt, Line2D, FuncFormatter = _get_plotting_dependencies()
    _apply_paper_style(plt)

    rows, duplicate_count, x_field = _load_rows(csv_path)
    series_by_method, method_order = _build_method_series(rows)
    grouped_by_mode = _split_methods_by_serving_mode(series_by_method, method_order)
    if not grouped_by_mode:
        raise ValueError(f"No methods found to plot in {csv_path}.")
    x_label = "RPS" if x_field == "rps" else "Load Scale"

    dataset_label = _dataset_label(csv_path)
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
                max_slo_violation=max_slo_violation,
                title=mode_title,
                x_label=x_label,
            )
        )

    if duplicate_count:
        print(
            f"Retained the last CSV row for {duplicate_count} duplicate "
            f"(method, {x_field}) points."
        )
    return saved_paths


def draw_all_slo_violation_vs_load_scale(
    input_dir: Path | str = DEFAULT_INPUT_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    max_slo_violation: float = DEFAULT_MAX_SLO_VIOLATION,
    glob_pattern: str = DEFAULT_GLOB_PATTERN,
) -> list[Path]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    saved_paths: list[Path] = []
    failures: list[tuple[Path, Exception]] = []
    for csv_path in _iter_csv_files(input_dir, glob_pattern=glob_pattern):
        output_stem = _default_output_stem(
            csv_path,
            max_slo_violation=max_slo_violation,
            output_dir=output_dir,
        )
        try:
            saved_paths.extend(
                draw_slo_violation_vs_load_scale(
                    csv_path=csv_path,
                    output_stem=output_stem,
                    max_slo_violation=max_slo_violation,
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
            "Render SLO-violation-vs-load-scale figures from the sensitivity CSVs "
            "using comparator labels and styles from plots/common.py."
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
        help="Output path without extension. Defaults to Paper/figs/slo_violation_rate_vs_load_scale-<dataset>.",
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
        "--max-slo-violation",
        type=float,
        default=DEFAULT_MAX_SLO_VIOLATION,
        help="Cap the displayed SLO-violation axis at this percentage. Default: %(default)s",
    )
    args = parser.parse_args()

    if args.draw_all:
        draw_all_slo_violation_vs_load_scale(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_slo_violation=args.max_slo_violation,
            glob_pattern=args.glob,
        )
    else:
        draw_slo_violation_vs_load_scale(
            csv_path=args.csv,
            output_stem=args.output_stem,
            max_slo_violation=args.max_slo_violation,
        )


if __name__ == "__main__":
    main()
