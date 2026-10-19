#!/usr/bin/env python3
"""Convert BEAM's NVML cumulative-energy samples into a GPU power timeline."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    energy_key = f"gpu_{args.gpu}_joules"
    frequency_key = f"gpu_{args.gpu}_freq_mhz"
    with args.input.open(newline="", encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))
    if not samples or energy_key not in samples[0]:
        raise ValueError(f"{args.input} does not contain {energy_key}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "timestamp", "elapsed_s", "cumulative_energy_mj",
            "graphics_clock_mhz", "power_w",
        ))
        writer.writeheader()
        start = previous_time = parse_time(samples[0]["timestamp"])
        previous_energy = float(samples[0][energy_key])
        for index, sample in enumerate(samples):
            timestamp = parse_time(sample["timestamp"])
            energy_mj = float(sample[energy_key])
            if index == 0:
                power = ""
            else:
                interval_s = (timestamp - previous_time).total_seconds()
                # nvmlDeviceGetTotalEnergyConsumption returns millijoules.
                power = (energy_mj - previous_energy) / 1000.0 / interval_s if interval_s > 0 else ""
            writer.writerow({
                "timestamp": sample["timestamp"],
                "elapsed_s": f"{(timestamp - start).total_seconds():.6f}",
                "cumulative_energy_mj": f"{energy_mj:.3f}",
                "graphics_clock_mhz": sample.get(frequency_key, ""),
                "power_w": power if power == "" else f"{power:.6f}",
            })
            previous_time, previous_energy = timestamp, energy_mj


if __name__ == "__main__":
    main()
