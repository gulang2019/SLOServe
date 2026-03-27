import argparse
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SLOsServe.perf_model import (
    BATCH_CATEGORY_COLORS,
    BATCH_CATEGORY_ORDER,
    extract_batch_perf_sample,
    fit_piecewise_current_token_model,
    load_batch_trace_events,
)

# Edit these geometry values to retune the figure layout.
FIGSIZE = (14.0, 9.4)
DPI = 240
DECODE_INSET_BOUNDS = [0.04, 0.20, 0.39, 0.36]
MIXED_PREFILL_INSET_BOUNDS = [0.57, 0.20, 0.39, 0.36]
DECODE_ANNOTATION_POS = (145, 62)
DECODE_ROI = (0.5, 7.5, 13.0, 12.5)
MIXED_PREFILL_ROI = (0.0, 12.0, 512.0, 38.0)


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.titlesize": 20,
        "axes.linewidth": 1.2,
    })


def median_piecewise_curve(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["total_current_tokens"])].append(
            float(row["piecewise_predicted_time_ms"])
        )
    xs = np.array(sorted(grouped), dtype=float)
    ys = np.array([np.median(grouped[int(x)]) for x in xs], dtype=float)
    return xs, ys


def add_highlight_box(ax, roi: tuple[float, float, float, float], edgecolor: str) -> None:
    x0, y0, width, height = roi
    # White halo keeps the ROI box visible over the red fit line.
    ax.add_patch(Rectangle(
        (x0, y0),
        width,
        height,
        fill=False,
        edgecolor="white",
        linewidth=6.0,
        zorder=8,
        joinstyle="round",
    ))
    ax.add_patch(Rectangle(
        (x0, y0),
        width,
        height,
        fill=False,
        edgecolor=edgecolor,
        linewidth=2.6,
        zorder=8.1,
        joinstyle="round",
    ))


def load_rows(trace_path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for event in load_batch_trace_events(trace_path):
        sample = extract_batch_perf_sample(event, subtract_scheduling_overhead=True)
        if sample is not None:
            rows.append(sample)
    return rows


def apply_piecewise_predictions(rows: list[dict]) -> None:
    piecewise = fit_piecewise_current_token_model(rows, breakpoints=(512, 2048))
    for row, predicted_time in zip(rows, piecewise["predicted_times"]):
        row["piecewise_predicted_time_ms"] = float(predicted_time) * 1000.0


def build_figure(trace_path: str | Path, output_base: str | Path) -> tuple[Path, Path]:
    configure_plot_style()

    rows = load_rows(trace_path)
    if not rows:
        raise ValueError(f"No valid rows found in {trace_path}")
    apply_piecewise_predictions(rows)

    decode_rows = [row for row in rows if row["batch_category"] == "decode"]
    mixed_prefill_rows = [
        row for row in rows
        if row["batch_category"] in ("prefill", "mixed")
        and row["total_current_tokens"] <= 512
    ]

    fig, ax_main = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)

    for category in BATCH_CATEGORY_ORDER:
        xs = [row["total_current_tokens"] for row in rows if row["batch_category"] == category]
        ys = [row["measured_time"] * 1000.0 for row in rows if row["batch_category"] == category]
        if not xs:
            continue
        ax_main.scatter(
            xs,
            ys,
            s=10,
            alpha=0.18,
            c=BATCH_CATEGORY_COLORS[category],
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )

    main_x, main_y = median_piecewise_curve(rows)
    ax_main.plot(main_x, main_y, color="red", linewidth=2.7, zorder=3)
    for breakpoint in (512, 2048):
        ax_main.axvline(
            breakpoint,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.75,
            zorder=1,
        )

    ax_main.set_title("Execution Time vs Current Tokens")
    ax_main.set_xlabel("Total current tokens in batch")
    ax_main.set_ylabel("Measured execution time (ms)")
    ax_main.set_xlim(0, 4096)
    ax_main.set_ylim(0, 520)
    ax_main.grid(True, alpha=0.22)

    ax_decode = ax_main.inset_axes(DECODE_INSET_BOUNDS, zorder=6)
    ax_mixed_prefill = ax_main.inset_axes(MIXED_PREFILL_INSET_BOUNDS, zorder=6)
    for inset in (ax_decode, ax_mixed_prefill):
        inset.patch.set_facecolor("white")
        inset.patch.set_alpha(1.0)
        inset.patch.set_zorder(6)

    ax_decode.scatter(
        [row["total_current_tokens"] for row in decode_rows],
        [row["measured_time"] * 1000.0 for row in decode_rows],
        s=12,
        alpha=0.34,
        c=BATCH_CATEGORY_COLORS["decode"],
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )
    decode_x, decode_y = median_piecewise_curve(decode_rows)
    ax_decode.plot(decode_x, decode_y, color="red", linewidth=2.5, zorder=3)
    ax_decode.set_title("(a) Decode-only Zoom", fontsize=16)
    ax_decode.set_xlabel("Total current tokens")
    ax_decode.set_ylabel("Measured time (ms)")
    ax_decode.set_xlim(0.5, 13.5)
    ax_decode.set_ylim(7.5, 20.0)
    ax_decode.grid(True, alpha=0.20)
    for spine in ax_decode.spines.values():
        spine.set_edgecolor(BATCH_CATEGORY_COLORS["decode"])
        spine.set_linewidth(2.1)

    for category in ("prefill", "mixed"):
        subset = [row for row in mixed_prefill_rows if row["batch_category"] == category]
        if not subset:
            continue
        ax_mixed_prefill.scatter(
            [row["total_current_tokens"] for row in subset],
            [row["measured_time"] * 1000.0 for row in subset],
            s=12,
            alpha=0.32,
            c=BATCH_CATEGORY_COLORS[category],
            edgecolors="none",
            rasterized=True,
            label=f"{category} (n={len(subset):,})",
            zorder=2,
        )
    mixed_prefill_x, mixed_prefill_y = median_piecewise_curve(mixed_prefill_rows)
    ax_mixed_prefill.plot(mixed_prefill_x, mixed_prefill_y, color="red", linewidth=2.5, zorder=3)
    ax_mixed_prefill.set_title("(b) Mixed/Prefill Zoom, <=512 Tokens", fontsize=16)
    ax_mixed_prefill.set_xlabel("Total current tokens")
    ax_mixed_prefill.set_ylabel("Measured time (ms)")
    ax_mixed_prefill.set_xlim(0, 512)
    ax_mixed_prefill.set_ylim(12, 50)
    ax_mixed_prefill.grid(True, alpha=0.20)
    for spine in ax_mixed_prefill.spines.values():
        spine.set_edgecolor(BATCH_CATEGORY_COLORS["mixed"])
        spine.set_linewidth(2.1)
    ax_mixed_prefill.legend(loc="upper left", frameon=False, fontsize=11)

    add_highlight_box(ax_main, DECODE_ROI, BATCH_CATEGORY_COLORS["decode"])
    add_highlight_box(ax_main, MIXED_PREFILL_ROI, BATCH_CATEGORY_COLORS["mixed"])

    ax_main.annotate(
        "decode zoom",
        xy=(DECODE_ROI[0] + DECODE_ROI[2], DECODE_ROI[1] + DECODE_ROI[3]),
        xytext=DECODE_ANNOTATION_POS,
        textcoords="data",
        color=BATCH_CATEGORY_COLORS["decode"],
        fontsize=13,
        arrowprops=dict(
            arrowstyle="->",
            color=BATCH_CATEGORY_COLORS["decode"],
            lw=1.6,
        ),
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
        ),
        zorder=9,
    )

    connections = [
        ConnectionPatch(
            xyA=(DECODE_ROI[0] + DECODE_ROI[2], DECODE_ROI[1] + DECODE_ROI[3]),
            coordsA=ax_main.transData,
            xyB=(1.0, 0.0),
            coordsB=ax_decode.transAxes,
            color=BATCH_CATEGORY_COLORS["decode"],
            linewidth=1.6,
            zorder=1,
            clip_on=False,
        ),
        ConnectionPatch(
            xyA=(DECODE_ROI[0], DECODE_ROI[1]),
            coordsA=ax_main.transData,
            xyB=(0.0, 0.0),
            coordsB=ax_decode.transAxes,
            color=BATCH_CATEGORY_COLORS["decode"],
            linewidth=1.6,
            zorder=1,
            clip_on=False,
        ),
        ConnectionPatch(
            xyA=(MIXED_PREFILL_ROI[0], MIXED_PREFILL_ROI[1] + MIXED_PREFILL_ROI[3]),
            coordsA=ax_main.transData,
            xyB=(0.0, 0.0),
            coordsB=ax_mixed_prefill.transAxes,
            color=BATCH_CATEGORY_COLORS["mixed"],
            linewidth=1.6,
            zorder=1,
            clip_on=False,
        ),
        ConnectionPatch(
            xyA=(
                MIXED_PREFILL_ROI[0] + MIXED_PREFILL_ROI[2],
                MIXED_PREFILL_ROI[1] + MIXED_PREFILL_ROI[3],
            ),
            coordsA=ax_main.transData,
            xyB=(1.0, 0.0),
            coordsB=ax_mixed_prefill.transAxes,
            color=BATCH_CATEGORY_COLORS["mixed"],
            linewidth=1.6,
            zorder=1,
            clip_on=False,
        ),
    ]
    for connection in connections:
        fig.add_artist(connection)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="prefill",
            markerfacecolor=BATCH_CATEGORY_COLORS["prefill"],
            markersize=9,
            alpha=0.9,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="decode",
            markerfacecolor=BATCH_CATEGORY_COLORS["decode"],
            markersize=9,
            alpha=0.9,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="mixed",
            markerfacecolor=BATCH_CATEGORY_COLORS["mixed"],
            markersize=9,
            alpha=0.9,
        ),
        Line2D([0], [0], color="red", linewidth=2.7, label="Median piecewise fit"),
        Line2D(
            [0],
            [0],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Piecewise breakpoints",
        ),
    ]
    ax_main.legend(handles=legend_handles, loc="upper left", frameon=False, ncol=2)

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot execution time vs current tokens with insets for decode and mixed/prefill batches."
    )
    parser.add_argument("trace_path", type=str, help="Path to the batch trace JSON/JSONL.")
    parser.add_argument(
        "--output-base",
        type=str,
        default="assets/perf_model_figs/batches__measured_vs_current_tokens__piecewise_inset_asplos_v2",
        help="Output base path without extension.",
    )
    args = parser.parse_args()

    png_path, pdf_path = build_figure(args.trace_path, args.output_base)
    print(png_path)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
