import argparse
import bisect
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


IDLE_POWER_W = 70.0
BATCH_TYPES = ["decode", "mixed", "prefill"]


def load_events(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def matched_rows(events: list[dict]) -> dict[str, list[tuple[int, float]]]:
    energy_ts: dict[int, list[float]] = defaultdict(list)
    energy_pw: dict[int, list[float]] = defaultdict(list)
    for event in events:
        if event.get("event_type") != "energy":
            continue
        device_id = int(event.get("device_id", -1))
        if device_id < 0:
            continue
        energy_ts[device_id].append(float(event["timestamp"]))
        energy_pw[device_id].append(float(event["power"]))

    rows: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for event in events:
        if event.get("event_type") != "batch":
            continue
        scheduled_tokens = list(event["num_scheduled_tokens"].values())
        if not scheduled_tokens:
            continue
        batch_type = (
            "decode"
            if max(scheduled_tokens) == 1
            else "prefill"
            if min(scheduled_tokens) > 1
            else "mixed"
        )
        device_id = int(event.get("device_id", -1))
        if device_id < 0 or device_id not in energy_ts:
            continue
        ts = float(event["timestamp"])
        idx = bisect.bisect_left(energy_ts[device_id], ts)
        best = None
        for j in (idx - 1, idx):
            if 0 <= j < len(energy_ts[device_id]):
                candidate = (
                    abs(energy_ts[device_id][j] - ts),
                    energy_pw[device_id][j],
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is None or best[0] > 0.2:
            continue
        rows[batch_type].append((len(event["req_ids"]), best[1]))
    return rows


def aggregate_means(rows: list[tuple[int, float]]) -> tuple[np.ndarray, np.ndarray]:
    xs = np.asarray([r[0] for r in rows], dtype=float)
    ys = np.asarray([r[1] for r in rows], dtype=float)
    uniq = np.asarray(sorted(set(int(v) for v in xs)), dtype=float)
    means = np.asarray([ys[xs == u].mean() for u in uniq], dtype=float)
    x = np.concatenate([np.array([0.0]), uniq])
    y = np.concatenate([np.array([IDLE_POWER_W]), means])
    return x, y


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def fit_exponential_saturation(x: np.ndarray, y: np.ndarray):
    best = None
    for b in np.linspace(0.01, 2.0, 600):
        z = 1.0 - np.exp(-b * x)
        A = z[:, None]
        coef = np.linalg.lstsq(A, y - IDLE_POWER_W, rcond=None)[0]
        amp = float(max(0.0, coef[0]))
        pred = IDLE_POWER_W + amp * z
        score = r2(y, pred)
        if best is None or score > best["r2"]:
            best = {"r2": score, "b": float(b), "p0": IDLE_POWER_W, "A": amp}
    return best


def fit_michaelis_menten(x: np.ndarray, y: np.ndarray):
    best = None
    for K in np.linspace(0.1, 50.0, 800):
        z = x / (K + x)
        A = z[:, None]
        coef = np.linalg.lstsq(A, y - IDLE_POWER_W, rcond=None)[0]
        amp = float(max(0.0, coef[0]))
        pred = IDLE_POWER_W + amp * z
        score = r2(y, pred)
        if best is None or score > best["r2"]:
            best = {"r2": score, "K": float(K), "p0": IDLE_POWER_W, "A": amp}
    return best


def eval_exp(model: dict, x: np.ndarray) -> np.ndarray:
    return model["p0"] + model["A"] * (1.0 - np.exp(-model["b"] * x))


def eval_mm(model: dict, x: np.ndarray) -> np.ndarray:
    return model["p0"] + model["A"] * x / (model["K"] + x)


def plot_panel(ax: plt.Axes, label: str, rows: list[tuple[int, float]]) -> None:
    if not rows:
        ax.set_title(label)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    x, y = aggregate_means(rows)
    exp_model = fit_exponential_saturation(x, y)
    mm_model = fit_michaelis_menten(x, y)
    x_dense = np.linspace(0.0, x.max(), 300)

    ax.scatter(x, y, color="#222222", s=35, label="Measured mean")
    ax.plot(
        x_dense,
        eval_exp(exp_model, x_dense),
        color="#c73e1d",
        linewidth=2.0,
        label=f"Exp sat (R^2={exp_model['r2']:.3f})",
    )
    ax.plot(
        x_dense,
        eval_mm(mm_model, x_dense),
        color="#1f78b4",
        linewidth=2.0,
        label=f"MM (R^2={mm_model['r2']:.3f})",
    )
    ax.axhline(IDLE_POWER_W, color="#777777", linestyle="--", linewidth=1.0)
    ax.set_title(label)
    ax.set_xlabel("# Active Requests")
    ax.set_ylabel("Server Power (W)")
    ax.legend(fontsize=9)

    print(label)
    print(
        "  exp_sat:",
        f"P(n) = {IDLE_POWER_W:.1f} + {exp_model['A']:.3f} * (1 - exp(-{exp_model['b']:.3f} * n))",
        f"R^2={exp_model['r2']:.3f}",
    )
    print(
        "  michaelis_menten:",
        f"P(n) = {IDLE_POWER_W:.1f} + {mm_model['A']:.3f} * n / ({mm_model['K']:.3f} + n)",
        f"R^2={mm_model['r2']:.3f}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rr", required=True)
    parser.add_argument("--packer", required=True)
    parser.add_argument("--output-prefix", default="figs/power_curve_fits")
    args = parser.parse_args()

    rr_rows = matched_rows(load_events(args.rr))
    packer_rows = matched_rows(load_events(args.packer))

    fig, axes = plt.subplots(len(BATCH_TYPES), 2, figsize=(13, 13), tight_layout=True)
    for row_idx, batch_type in enumerate(BATCH_TYPES):
        plot_panel(
            axes[row_idx, 0],
            f"Round Robin {batch_type.capitalize()}",
            rr_rows.get(batch_type, []),
        )
        plot_panel(
            axes[row_idx, 1],
            f"Packer {batch_type.capitalize()}",
            packer_rows.get(batch_type, []),
        )

    output_prefix = str(Path(args.output_prefix))
    fig.savefig(f"{output_prefix}.png", dpi=250, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_prefix}.png")
    print(f"Saved {output_prefix}.pdf")


if __name__ == "__main__":
    main()
