from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from SLOsServe.batch_config import load_batch_config, normalize_batch_config


def count_clients_spec(spec: str) -> int:
    if "," in spec:
        return len([item for item in spec.split(",") if item])
    if "-" in spec and ":" not in spec:
        start, end = spec.split("-", 1)
        return int(end) - int(start) + 1
    if ":" in spec and "," not in spec:
        return int(spec.split(":", 1)[1])
    return 1


def policy_supports_partial_rr(policy: str) -> bool:
    routing = policy.split(":", 1)[0]
    return routing in {"round_robin", "round_robin_retry", "round_robin_session"}


def parse_window_seconds(window: str) -> float:
    start, end = window.split(":", 1)

    def parse_one(token: str) -> float:
        token = token.strip()
        while token and not (token[0].isdigit() or token[0] in "+-."):
            token = token[1:]
        return float(token)

    return max(0.0, parse_one(end) - parse_one(start))


@dataclass
class ConfigReport:
    config_name: str
    available_clients: int
    experiments: int
    serial_wall_hours: float
    useful_gpu_hours: float
    ideal_parallel_wall_hours: float
    sequential_node_gpu_hours: float
    max_effective_gpus: int


def build_config_report(config_path: str | Path) -> ConfigReport:
    normalized = normalize_batch_config(load_batch_config(config_path))
    available_clients = count_clients_spec(normalized.get("server_clients", "0-7"))

    experiments = 0
    serial_wall_hours = 0.0
    useful_gpu_hours = 0.0
    max_effective_gpus = 0

    for trace in normalized["traces"]:
        spec = normalized["trace_specs"][trace]
        duration_hours = parse_window_seconds(spec["window"]) / 3600.0
        load_scales = spec["load_scales"]
        ttft_slo_scales = spec["ttft_slo_scales"]
        slo_tpots = spec["slo_tpots"]
        perf_model_errs = spec["perf_model_errs"]
        decode_length_offsets = spec.get("decode_length_offsets", ["0"])
        tensor_parallel_size = int(spec["tensor_parallel_size"])
        combinations_per_device = (
            len(load_scales)
            * len(ttft_slo_scales)
            * len(slo_tpots)
            * len(perf_model_errs)
            * len(decode_length_offsets)
        )

        for policy in spec["policies"]:
            run_devices = [int(item) for item in spec["n_devices"]]
            if not policy_supports_partial_rr(policy):
                run_devices = [item for item in run_devices if item <= available_clients]
                if not run_devices:
                    continue
            for requested_n_device in run_devices:
                effective_n_device = (
                    min(requested_n_device, available_clients)
                    if policy_supports_partial_rr(policy)
                    else requested_n_device
                )
                total_gpus = effective_n_device * tensor_parallel_size
                experiments += combinations_per_device
                serial_wall_hours += combinations_per_device * duration_hours
                useful_gpu_hours += combinations_per_device * duration_hours * total_gpus
                max_effective_gpus = max(max_effective_gpus, total_gpus)

    ideal_parallel_wall_hours = (
        useful_gpu_hours / available_clients if available_clients > 0 else 0.0
    )
    sequential_node_gpu_hours = serial_wall_hours * available_clients

    return ConfigReport(
        config_name=Path(config_path).name,
        available_clients=available_clients,
        experiments=experiments,
        serial_wall_hours=serial_wall_hours,
        useful_gpu_hours=useful_gpu_hours,
        ideal_parallel_wall_hours=ideal_parallel_wall_hours,
        sequential_node_gpu_hours=sequential_node_gpu_hours,
        max_effective_gpus=max_effective_gpus,
    )


def build_directory_report(config_dir: str | Path) -> dict[str, object]:
    root = Path(config_dir).expanduser().resolve()
    reports = []
    for path in sorted(root.glob("*.json")) + sorted(root.glob("*.jsonl")):
        report = build_config_report(path)
        if report.experiments == 0:
            continue
        reports.append(report)

    total = ConfigReport(
        config_name="TOTAL",
        available_clients=max((report.available_clients for report in reports), default=0),
        experiments=sum(report.experiments for report in reports),
        serial_wall_hours=sum(report.serial_wall_hours for report in reports),
        useful_gpu_hours=sum(report.useful_gpu_hours for report in reports),
        ideal_parallel_wall_hours=0.0,
        sequential_node_gpu_hours=sum(
            report.sequential_node_gpu_hours for report in reports
        ),
        max_effective_gpus=max((report.max_effective_gpus for report in reports), default=0),
    )
    if total.available_clients > 0:
        total.ideal_parallel_wall_hours = total.useful_gpu_hours / total.available_clients

    return {
        "configs": [asdict(report) for report in reports],
        "total": asdict(total),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize batch config experiment counts and GPU-hour estimates."
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/batch",
        help="Directory containing batch config JSON or JSONL files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a tabular text summary.",
    )
    args = parser.parse_args()

    report = build_directory_report(args.config_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    for item in report["configs"]:
        print(
            "\t".join(
                [
                    item["config_name"],
                    f"experiments={item['experiments']}",
                    f"serial_wall_hours={item['serial_wall_hours']:.3f}",
                    f"useful_gpu_hours={item['useful_gpu_hours']:.3f}",
                    f"ideal_parallel_wall_hours={item['ideal_parallel_wall_hours']:.3f}",
                    f"sequential_node_gpu_hours={item['sequential_node_gpu_hours']:.3f}",
                    f"max_effective_gpus={item['max_effective_gpus']}",
                ]
            )
        )

    total = report["total"]
    print("TOTAL")
    print(
        "\t".join(
            [
                f"experiments={total['experiments']}",
                f"serial_wall_hours={total['serial_wall_hours']:.3f}",
                f"useful_gpu_hours={total['useful_gpu_hours']:.3f}",
                f"ideal_parallel_wall_hours={total['ideal_parallel_wall_hours']:.3f}",
                f"sequential_node_gpu_hours={total['sequential_node_gpu_hours']:.3f}",
                f"max_effective_gpus={total['max_effective_gpus']}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
