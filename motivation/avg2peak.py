from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path


HEADROOM_METRICS = ["avg2peak", "peak2min", "n_total", "il_mean", "ol_mean"]
HEADROOM_WINDOWS = [5, 10, 30]
ARRIVAL_WINDOWS = [1, 2, 3, 4, 5]


class _CompatArrivalTimes:
    pass


class _ArrivalUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "Dataset.dataset" and name == "ArrivalTimes":
            return _CompatArrivalTimes
        return super().find_class(module, name)


def _normalize_trace_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    alias = {
        "azure_code23": "azure_code_23",
        "azure_chat23": "azure_chat_23",
        "azure_code_23": "azure_code_23",
        "azure_chat_23": "azure_chat_23",
    }
    return alias.get(key, key)


def _load_arrival_times(trace_name: str, dataset_dir: Path) -> list[float]:
    trace = _normalize_trace_name(trace_name)
    path = dataset_dir / f"{trace}.arrival.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Arrival trace not found: {path}")

    with path.open("rb") as f:
        obj = _ArrivalUnpickler(f).load()

    times = getattr(obj, "arrival_times", None)
    if not isinstance(times, list) or not times:
        raise ValueError(f"Invalid arrival trace payload: {path}")
    return times


def _peak_to_avg_for_window(arrival_times: list[float], window_sec: int) -> tuple[float, float, float, float]:
    import numpy as np

    times = np.asarray(arrival_times, dtype=float)
    if times.size == 0:
        return 0.0, 0.0, 0.0, float("nan")

    t0 = times.min()
    idx = np.floor((times - t0) / float(window_sec)).astype(np.int64)
    counts = np.bincount(idx)

    peak = float(counts.max())
    avg = float(counts.mean())
    nonzero = counts[counts > 0]
    avg_nonzero = float(nonzero.mean()) if nonzero.size > 0 else 0.0
    ratio = peak / avg if avg > 0.0 else float("nan")
    return peak, avg, avg_nonzero, ratio


def run_arrival_mode(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)

    traces = [_normalize_trace_name(t) for t in args.traces]
    windows = sorted(set(int(w) for w in args.window_seconds if 1 <= int(w) <= 5))
    if not windows:
        windows = ARRIVAL_WINDOWS

    rows: list[dict[str, float | str | int]] = []
    for trace in traces:
        arrivals = _load_arrival_times(trace, dataset_dir)
        for w in windows:
            peak, avg, avg_nonzero, ratio = _peak_to_avg_for_window(arrivals, w)
            rows.append(
                {
                    "dataset": trace,
                    "window_seconds": w,
                    "peak_requests": peak,
                    "avg_requests": avg,
                    "avg_requests_nonzero_windows": avg_nonzero,
                    "peak_to_avg": ratio,
                }
            )

    csv_path = outdir / "arrival_peak_to_avg_1_5s.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "window_seconds",
                "peak_requests",
                "avg_requests",
                "avg_requests_nonzero_windows",
                "peak_to_avg",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    for trace in traces:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        xs = [r["window_seconds"] for r in rows if r["dataset"] == trace]
        ys = [r["peak_to_avg"] for r in rows if r["dataset"] == trace]
        ax.plot(xs, ys, marker="o", linewidth=2, label=trace)
        ax.set_title(trace)
        ax.set_xlabel("Window size (s)")
        ax.set_ylabel("Peak arrivals / Average arrivals")
        ax.set_xticks(windows)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(frameon=False)
        fig.tight_layout()

        fig_path = outdir / f"arrival_peak_to_avg_1_5s_{trace}.png"
        fig.savefig(fig_path, dpi=220)


def _to_bool(series):
    if str(getattr(series, "dtype", "")) == "bool":
        return series
    return series.astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes", "y"])


def _prepare_headroom(df):
    import numpy as np

    required = {"dataset", "is_pd_disagg", "window_minutes", *HEADROOM_METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.loc[:, ["dataset", "is_pd_disagg", "window_minutes", *HEADROOM_METRICS]].copy()
    out["is_pd_disagg"] = _to_bool(out["is_pd_disagg"])
    out["setting"] = np.where(out["is_pd_disagg"], "disaggregated", "aggregated")
    out["window_minutes"] = out["window_minutes"].astype(str)
    out["window_minutes"] = out["window_minutes"].str.replace(".0", "", regex=False)
    out["window_minutes"] = out["window_minutes"].astype("Int64")
    out = out[out["window_minutes"].isin(HEADROOM_WINDOWS)]
    for c in HEADROOM_METRICS:
        out[c] = out[c].astype(float)
    return out.dropna(subset=["window_minutes", *HEADROOM_METRICS])


def _plot_headroom_metric(stats, metric: str, outdir: Path) -> None:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    metric_stats = stats[stats["metric"] == metric]
    datasets = sorted(metric_stats["dataset"].unique())
    if not datasets:
        return

    n_d = len(datasets)
    n_w = len(HEADROOM_WINDOWS)
    fig, axes = plt.subplots(1, n_w, figsize=(max(12, 2.6 * n_d * n_w), 5), sharey=True)
    if n_w == 1:
        axes = [axes]

    x = np.arange(n_d)
    width = 0.38

    for ax, w in zip(axes, HEADROOM_WINDOWS):
        sub = metric_stats[metric_stats["window_minutes"] == w]
        pivot_mean = sub.pivot(index="dataset", columns="setting", values="mean").reindex(datasets)
        pivot_std = sub.pivot(index="dataset", columns="setting", values="std").reindex(datasets).fillna(0.0)

        agg_mean = pivot_mean.get("aggregated", pd.Series(index=datasets, dtype=float)).fillna(0.0).to_numpy()
        dis_mean = pivot_mean.get("disaggregated", pd.Series(index=datasets, dtype=float)).fillna(0.0).to_numpy()
        agg_std = pivot_std.get("aggregated", pd.Series(index=datasets, dtype=float)).to_numpy()
        dis_std = pivot_std.get("disaggregated", pd.Series(index=datasets, dtype=float)).to_numpy()

        ax.bar(x - width / 2, agg_mean, width, yerr=agg_std, capsize=3, label="aggregated", color="#4C78A8")
        ax.bar(x + width / 2, dis_mean, width, yerr=dis_std, capsize=3, label="disaggregated", color="#F58518")
        ax.set_title(f"{w} min")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha="right")
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    axes[0].set_ylabel(metric)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = outdir / f"{metric}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_avg2peak_lines_by_setting(stats, outdir: Path) -> None:
    import matplotlib.pyplot as plt

    metric_stats = stats[stats["metric"] == "avg2peak"]
    datasets = sorted(metric_stats["dataset"].unique())
    if not datasets:
        return

    for setting in ("aggregated", "disaggregated"):
        sub = metric_stats[metric_stats["setting"] == setting]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for dataset in datasets:
            ds = sub[sub["dataset"] == dataset].sort_values("window_minutes")
            if ds.empty:
                continue
            xs = ds["window_minutes"].to_numpy()
            ys = ds["mean"].to_numpy()
            es = ds["std"].fillna(0.0).to_numpy()
            ax.errorbar(xs, ys, yerr=es, marker="o", linewidth=2, capsize=3, label=dataset)

        ax.set_title(f"avg2peak ({setting})")
        ax.set_xlabel("Auto-scaling window (minutes)")
        ax.set_ylabel("avg2peak")
        ax.set_xticks(HEADROOM_WINDOWS)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / f"avg2peak_{setting}_lines.png", dpi=220)
        plt.close(fig)


def run_headroom_mode(args: argparse.Namespace) -> None:
    import pandas as pd

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df = _prepare_headroom(df)

    stats = (
        df.melt(
            id_vars=["dataset", "setting", "window_minutes"],
            value_vars=HEADROOM_METRICS,
            var_name="metric",
            value_name="value",
        )
        .groupby(["metric", "dataset", "window_minutes", "setting"], sort=False, as_index=False)
        .agg(mean=("value", "mean"), std=("value", "std"))
    )

    _plot_avg2peak_lines_by_setting(stats, outdir)
    for metric in HEADROOM_METRICS:
        _plot_headroom_metric(stats, metric, outdir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headroom and arrival-pattern plotting.")
    parser.add_argument("--mode", choices=["headroom", "arrival"], default="headroom")

    parser.add_argument("--input", default="headroom_outputs/headroom.csv", help="Path to headroom CSV")
    parser.add_argument("--outdir", default="headroom_outputs/avg2peak_figs", help="Output directory for figures")

    parser.add_argument(
        "--traces",
        nargs="+",
        default=["azure_code23", "azure_chat23"],
        help="Arrival traces for arrival mode (e.g., azure_code23 azure_chat23)",
    )
    parser.add_argument(
        "--window-seconds",
        nargs="+",
        type=int,
        default=ARRIVAL_WINDOWS,
        help="Arrival window sizes in seconds (supported range: 1..5)",
    )
    parser.add_argument(
        "--dataset-dir",
        default="assets/datasets",
        help="Directory holding *.arrival.pkl files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "arrival":
        run_arrival_mode(args)
    else:
        run_headroom_mode(args)


if __name__ == "__main__":
    main()
