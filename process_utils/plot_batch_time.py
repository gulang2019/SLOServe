from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from motivation.common import PerfModel


def plot_batch_time_vs_batch_size(
    output_path: str | Path = "batch_time_vs_batch_size.png",
    *,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    task: str = "default",
    max_batch_size: int = 128,
    scheduling_overhead: float = 0.005,
) -> Path:
    """
    Plot predicted throughput vs. batch size using the simplified perf model.

    Uses only k1 (per new token) and b (constant), plus scheduling_overhead.
    Throughput(bs) = bs / (k1*bs + b + overhead)
    """
    perf_model = PerfModel.get_perf_model(model_name, task)
    k1, _, _, _, b = [float(x) for x in perf_model.hardware_params]
    scheduling_overhead = float(scheduling_overhead)

    batch_sizes = list(range(1, max_batch_size + 1))
    times = [k1 * bs + b + scheduling_overhead for bs in batch_sizes]
    throughputs = [bs / t if t > 0 else 0 for bs, t in zip(batch_sizes, times)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(batch_sizes, throughputs, marker="o", linewidth=1.5)
    ax.set_xlabel("Batch size (number of new tokens)")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title(f"Throughput vs batch size - {model_name}")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Clean, readable batch-time formulas printed on the plot.
    k2 = float(perf_model.hardware_params[1])
    k3 = float(perf_model.hardware_params[2])
    k4 = float(perf_model.hardware_params[3])
    formula_text = (
        "Batch time (perf):\n"
        "  k1*bs + k2*num_reqs + k3*num_past_tokens + k4*num_decode_steps + b + overhead\n"
        "Batch time (simplified):\n"
        "  k1*bs + b + overhead\n"
        f"k1={k1:.6g}, k2={k2:.6g}, k3={k3:.6g}, k4={k4:.6g}, "
        f"b={b:.6g}, overhead={scheduling_overhead:.6g}"
    )
    ax.text(
        0.98,
        0.02,
        formula_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"),
    )

    output_path = Path(output_path)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
