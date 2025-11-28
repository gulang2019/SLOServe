#!/usr/bin/env python3
"""
Plot TTFT and TPOT vs. decode batch size from sweep experiments.

Usage:
    python process_utils/plot_decode_bsz_sweep.py experiments_results1 experiments_results2 ...
    python process_utils/plot_decode_bsz_sweep.py experiments_results  # single directory
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from extract_data import extract_ttft_tpots


def extract_bsz_from_dirname(dirname: str) -> int | None:
    """Extract decode batch size from directory name like 'sarathi_rr_bszd64'."""
    match = re.search(r'bszd(\d+)', dirname)
    if match:
        return int(match.group(1))
    return None


def compute_mean_metrics(data: Dict[str, Dict[str, List[float] | float]]) -> Tuple[float, float]:
    """
    Compute mean TTFT and mean TPOT across all requests.

    Returns:
        (mean_ttft, mean_tpot)
    """
    ttfts = []
    all_tpots = []

    for req_id, metrics in data.items():
        ttfts.append(metrics["ttft"])
        all_tpots.extend(metrics["tpots"])

    mean_ttft = np.mean(ttfts) if ttfts else 0.0
    mean_tpot = np.mean(all_tpots) if all_tpots else 0.0

    return mean_ttft, mean_tpot


def load_experiments_from_dir(results_path: Path) -> List[Tuple[int, float, float]]:
    """Load experiment data from a single results directory."""
    experiments: List[Tuple[int, float, float]] = []  # (bsz, mean_ttft, mean_tpot)

    for exp_dir in sorted(results_path.iterdir()):
        if not exp_dir.is_dir():
            continue

        bsz = extract_bsz_from_dirname(exp_dir.name)
        if bsz is None:
            print(f"Skipping {exp_dir.name} (could not extract batch size)")
            continue

        try:
            # Look for nested subdirectory containing the actual data
            subdirs = [d for d in exp_dir.iterdir() if d.is_dir()]
            if subdirs:
                # Use the first subdirectory (should be the experiment results dir)
                actual_dir = subdirs[0]
            else:
                actual_dir = exp_dir

            data = extract_ttft_tpots(actual_dir)
            mean_ttft, mean_tpot = compute_mean_metrics(data)
            experiments.append((bsz, mean_ttft, mean_tpot))
            print(f"Processed {results_path.name}/{exp_dir.name}: BSZ={bsz}, TTFT={mean_ttft:.4f}s, TPOT={mean_tpot:.4f}s")
        except Exception as e:
            print(f"Error processing {exp_dir.name}: {e}")
            continue

    return experiments


def main():
    parser = argparse.ArgumentParser(
        description="Plot TTFT and TPOT vs. decode batch size from multiple experiment directories"
    )
    parser.add_argument(
        "results_dirs",
        type=str,
        nargs='+',
        help="One or more directories containing experiment subdirectories (e.g., experiments_results1 experiments_results2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="decode_bsz_sweep.png",
        help="Output figure filename (default: decode_bsz_sweep.png)"
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs='+',
        default=None,
        help="Labels for each results directory (default: directory names)"
    )
    args = parser.parse_args()

    # Collect data from all result directories
    all_experiments = []  # List of (label, experiments_data)

    labels = args.labels if args.labels else [Path(d).name for d in args.results_dirs]

    if len(labels) != len(args.results_dirs):
        raise ValueError(f"Number of labels ({len(labels)}) must match number of directories ({len(args.results_dirs)})")

    for results_dir, label in zip(args.results_dirs, labels):
        results_path = Path(results_dir)
        if not results_path.exists():
            print(f"Warning: Results directory does not exist: {results_path}, skipping...")
            continue

        print(f"\n=== Processing {label} ===")
        experiments = load_experiments_from_dir(results_path)

        if not experiments:
            print(f"Warning: No valid experiment data found in {results_path}")
            continue

        # Sort by batch size
        experiments.sort(key=lambda x: x[0])
        all_experiments.append((label, experiments))

    if not all_experiments:
        raise ValueError("No valid experiment data found in any directory")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Define colors and markers for different series
    colors = plt.cm.tab10(range(len(all_experiments)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    # Plot TTFT
    for idx, (label, experiments) in enumerate(all_experiments):
        bszs, ttfts, tpots = zip(*experiments)
        ax1.plot(bszs, ttfts,
                marker=markers[idx % len(markers)],
                linewidth=2,
                markersize=8,
                color=colors[idx],
                label=label)

    ax1.set_xlabel('Decode Batch Size', fontsize=12)
    ax1.set_ylabel('Mean TTFT (s)', fontsize=12)
    ax1.set_title('Time to First Token vs. Decode Batch Size', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.legend()

    # Plot TPOT
    for idx, (label, experiments) in enumerate(all_experiments):
        bszs, ttfts, tpots = zip(*experiments)
        ax2.plot(bszs, tpots,
                marker=markers[idx % len(markers)],
                linewidth=2,
                markersize=8,
                color=colors[idx],
                label=label)

    ax2.set_xlabel('Decode Batch Size', fontsize=12)
    ax2.set_ylabel('Mean TPOT (s)', fontsize=12)
    ax2.set_title('Time Per Output Token vs. Decode Batch Size', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
