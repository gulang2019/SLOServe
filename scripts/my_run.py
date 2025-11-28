from __future__ import annotations

from pathlib import Path
import sys

# Ensure repo root is on PYTHONPATH when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_utils.plot_batch_time import plot_batch_time_vs_batch_size


def my_run(
    output_path: str | Path = "Xput_vs_batch_size.png",
    *,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    task: str = "default",
    max_batch_size: int = 2048,
    scheduling_overhead: float = 0.005,
) -> Path:
    """
    Convenience entrypoint: generate the throughput plot for the given model/task.
    """
    output_path = plot_batch_time_vs_batch_size(
        output_path=output_path,
        model_name=model_name,
        task=task,
        max_batch_size=max_batch_size,
        scheduling_overhead=scheduling_overhead,
    )
    return Path(output_path)


if __name__ == "__main__":
    path = my_run()
    print(f"Saved plot to {path}")
