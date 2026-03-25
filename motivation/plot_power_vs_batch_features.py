import argparse
import bisect
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_events(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def match_batch_power_rows(events: list[dict]) -> list[dict]:
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

    rows: list[dict] = []
    for event in events:
        if event.get("event_type") != "batch":
            continue
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
        rows.append(
            {
                "device_id": device_id,
                "nreq": len(event["req_ids"]),
                "tokens": sum(scheduled_tokens),
                "power": best[1],
                "batch_type": batch_type,
            }
        )
    return rows


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(x) < 2 or np.allclose(x, x[0]):
        return None
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    coeff = np.polyfit(xs, ys, deg=1)
    x_line = np.linspace(xs[0], xs[-1], 100)
    y_line = coeff[0] * x_line + coeff[1]
    return x_line, y_line


def maybe_downsample(rows: list[dict], max_points: int = 12000) -> list[dict]:
    if len(rows) <= max_points:
        return rows
    step = max(1, len(rows) // max_points)
    return rows[::step]


def plot(rows_by_label: dict[str, list[dict]], output_prefix: str) -> None:
    colors = {
        "Round Robin": "#c73e1d",
        "Packer": "#1f78b4",
    }
    batch_types = ["all", "decode", "mixed", "prefill"]
    fig, axes = plt.subplots(4, 2, figsize=(14, 18), tight_layout=True)

    for row_idx, batch_type in enumerate(batch_types):
        for col_idx, x_key in enumerate(["nreq", "tokens"]):
            ax = axes[row_idx, col_idx]
            for label, rows in rows_by_label.items():
                subset = rows if batch_type == "all" else [r for r in rows if r["batch_type"] == batch_type]
                subset = maybe_downsample(subset)
                if not subset:
                    continue
                x = np.asarray([r[x_key] for r in subset], dtype=float)
                y = np.asarray([r["power"] for r in subset], dtype=float)
                ax.scatter(
                    x,
                    y,
                    s=8,
                    alpha=0.16,
                    color=colors[label],
                    label=label if row_idx == 0 and col_idx == 0 else None,
                )
                fitted = fit_line(x, y)
                if fitted is not None:
                    x_line, y_line = fitted
                    ax.plot(x_line, y_line, color=colors[label], linewidth=2.0)

            title_left = "All Batches" if batch_type == "all" else batch_type.capitalize()
            title_right = "# Active Requests" if x_key == "nreq" else "# Scheduled Tokens"
            ax.set_title(f"{title_left}: Power vs {title_right}")
            ax.set_xlabel(title_right)
            ax.set_ylabel("Server Power (W)")

    handles = [
        plt.Line2D([0], [0], color=colors["Round Robin"], lw=2),
        plt.Line2D([0], [0], color=colors["Packer"], lw=2),
    ]
    fig.legend(handles, ["Round Robin", "Packer"], loc="upper center", ncol=2)
    fig.savefig(f"{output_prefix}.png", dpi=250, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rr", required=True)
    parser.add_argument("--packer", required=True)
    parser.add_argument(
        "--output-prefix",
        default="figs/power_vs_batch_features",
    )
    args = parser.parse_args()

    rr_rows = match_batch_power_rows(load_events(args.rr))
    packer_rows = match_batch_power_rows(load_events(args.packer))
    plot(
        {
            "Round Robin": rr_rows,
            "Packer": packer_rows,
        },
        str(Path(args.output_prefix)),
    )
    print(f"round_robin matched rows: {len(rr_rows)}")
    print(f"packer matched rows: {len(packer_rows)}")
    print(f"Saved {args.output_prefix}.png")
    print(f"Saved {args.output_prefix}.pdf")


if __name__ == "__main__":
    main()
