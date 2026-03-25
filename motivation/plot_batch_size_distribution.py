import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_batch_sizes(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        events = json.load(f)
    return [len(event["req_ids"]) for event in events if event.get("event_type") == "batch"]


def summarize(label: str, sizes: list[int]) -> None:
    arr = np.asarray(sizes, dtype=float)
    print(label)
    print(f"  batches: {len(sizes)}")
    print(f"  mean_batch_size: {arr.mean():.3f}")
    print(f"  p50_batch_size: {np.percentile(arr, 50):.3f}")
    print(f"  p90_batch_size: {np.percentile(arr, 90):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rr", required=True)
    parser.add_argument("--packer", required=True)
    parser.add_argument("--output-prefix", default="figs/batch_size_distribution")
    args = parser.parse_args()

    rr_sizes = load_batch_sizes(args.rr)
    packer_sizes = load_batch_sizes(args.packer)

    max_size = max(max(rr_sizes, default=0), max(packer_sizes, default=0))
    xs = np.arange(max_size + 1)
    rr_counts = Counter(rr_sizes)
    packer_counts = Counter(packer_sizes)
    rr_pct = np.asarray([100.0 * rr_counts.get(x, 0) / max(1, len(rr_sizes)) for x in xs])
    packer_pct = np.asarray([100.0 * packer_counts.get(x, 0) / max(1, len(packer_sizes)) for x in xs])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), tight_layout=True)

    width = 0.42
    axes[0].bar(xs - width / 2, rr_pct, width=width, color="#c73e1d", alpha=0.75, label="Round Robin")
    axes[0].bar(xs + width / 2, packer_pct, width=width, color="#1f78b4", alpha=0.75, label="Packer")
    axes[0].set_title("Batch Size Distribution")
    axes[0].set_xlabel("# Requests In Batch")
    axes[0].set_ylabel("Batches (%)")
    axes[0].legend()

    axes[1].boxplot(
        [rr_sizes, packer_sizes],
        labels=["Round Robin", "Packer"],
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="#dddddd"),
        medianprops=dict(color="black"),
    )
    axes[1].set_title("Batch Size Summary")
    axes[1].set_ylabel("# Requests In Batch")

    output_prefix = str(Path(args.output_prefix))
    fig.savefig(f"{output_prefix}.png", dpi=250, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)

    summarize("round_robin", rr_sizes)
    summarize("packer", packer_sizes)
    print(f"Saved {output_prefix}.png")
    print(f"Saved {output_prefix}.pdf")


if __name__ == "__main__":
    main()
