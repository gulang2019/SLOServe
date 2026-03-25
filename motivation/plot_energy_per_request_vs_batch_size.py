import argparse
import bisect
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: str) -> list[tuple[int, float, str]]:
    with open(path, "r", encoding="utf-8") as f:
        events = json.load(f)

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

    rows: list[tuple[int, float, str]] = []
    for event in events:
        if event.get("event_type") != "batch":
            continue
        device_id = int(event.get("device_id", -1))
        if device_id < 0:
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
        batch_size = len(event["req_ids"])
        if batch_size <= 0:
            continue
        power = float(best[1])
        elapsed = float(event.get("elapsed", 0.0) or 0.0)
        energy_per_req = power * elapsed / batch_size
        rows.append((batch_size, energy_per_req, batch_type))
    return rows


def aggregate(rows: list[tuple[int, float, str]], batch_type: str | None) -> tuple[np.ndarray, np.ndarray]:
    filtered = rows if batch_type is None else [r for r in rows if r[2] == batch_type]
    by_n: dict[int, list[float]] = defaultdict(list)
    for n, epr, _ in filtered:
        by_n[n].append(epr)
    xs = np.asarray(sorted(by_n), dtype=float)
    ys = np.asarray([np.mean(by_n[int(x)]) for x in xs], dtype=float)
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rr", required=True)
    parser.add_argument("--packer", required=True)
    parser.add_argument(
        "--output-prefix",
        default="figs/energy_per_request_vs_batch_size",
    )
    args = parser.parse_args()

    rr_rows = load_rows(args.rr)
    pk_rows = load_rows(args.packer)
    batch_types = [None, "decode", "mixed", "prefill"]
    titles = {
        None: "All Batches",
        "decode": "Decode Only",
        "mixed": "Mixed",
        "prefill": "Prefill Only",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), tight_layout=True)
    for ax, batch_type in zip(axes.flat, batch_types):
        rr_x, rr_y = aggregate(rr_rows, batch_type)
        pk_x, pk_y = aggregate(pk_rows, batch_type)
        if len(rr_x):
            ax.plot(rr_x, rr_y, "o-", color="#c73e1d", label="Round Robin")
        if len(pk_x):
            ax.plot(pk_x, pk_y, "o-", color="#1f78b4", label="Packer")
        ax.set_title(titles[batch_type])
        ax.set_xlabel("Batch Size (# Requests)")
        ax.set_ylabel("Energy Per Request (J)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    output_prefix = str(Path(args.output_prefix))
    fig.savefig(f"{output_prefix}.png", dpi=250, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_prefix}.png")
    print(f"Saved {output_prefix}.pdf")


if __name__ == "__main__":
    main()
