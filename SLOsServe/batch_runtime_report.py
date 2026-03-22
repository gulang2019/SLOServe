from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from SLOsServe.batch_config import load_batch_config, normalize_batch_config
from SLOsServe.batch_report import (
    count_clients_spec,
    parse_window_seconds,
    policy_supports_partial_rr,
)


@dataclass
class TraceRuntimeReport:
    trace: str
    window: str
    window_hours: float
    policy_count: int
    sweep_count: int
    requested_device_count: int
    runnable_policy_invocations: int
    policy_device_runs: int
    experiment_runs: int
    serial_wall_hours: float


@dataclass
class ConfigRuntimeReport:
    config_name: str
    available_clients: int
    trace_count: int
    runnable_policy_invocations: int
    experiment_runs: int
    serial_wall_hours: float
    traces: list[TraceRuntimeReport]
    error: str | None = None


def build_trace_runtime_report(
    trace: str,
    spec: dict[str, Any],
    *,
    available_clients: int,
) -> TraceRuntimeReport:
    window_hours = parse_window_seconds(spec["window"]) / 3600.0
    sweep_count = (
        len(spec["load_scales"])
        * len(spec["ttft_slo_scales"])
        * len(spec["slo_tpots"])
        * len(spec["perf_model_errs"])
    )
    requested_device_count = len(spec["n_devices"])

    runnable_policy_invocations = 0
    policy_device_runs = 0
    experiment_runs = 0

    for policy in spec["policies"]:
        runnable_devices = [int(item) for item in spec["n_devices"]]
        if not policy_supports_partial_rr(policy):
            runnable_devices = [
                item for item in runnable_devices if item <= available_clients
            ]
        if not runnable_devices:
            continue
        runnable_policy_invocations += 1
        policy_device_runs += len(runnable_devices)
        experiment_runs += sweep_count * len(runnable_devices)

    return TraceRuntimeReport(
        trace=trace,
        window=spec["window"],
        window_hours=window_hours,
        policy_count=len(spec["policies"]),
        sweep_count=sweep_count,
        requested_device_count=requested_device_count,
        runnable_policy_invocations=runnable_policy_invocations,
        policy_device_runs=policy_device_runs,
        experiment_runs=experiment_runs,
        serial_wall_hours=window_hours * experiment_runs,
    )


def build_config_runtime_report(config_path: str | Path) -> ConfigRuntimeReport:
    config_name = Path(config_path).name
    try:
        normalized = normalize_batch_config(load_batch_config(config_path))
    except Exception as exc:
        return ConfigRuntimeReport(
            config_name=config_name,
            available_clients=0,
            trace_count=0,
            runnable_policy_invocations=0,
            experiment_runs=0,
            serial_wall_hours=0.0,
            traces=[],
            error=str(exc),
        )

    available_clients = count_clients_spec(normalized.get("server_clients", "0-7"))
    trace_reports = [
        build_trace_runtime_report(
            trace,
            normalized["trace_specs"][trace],
            available_clients=available_clients,
        )
        for trace in normalized["traces"]
    ]

    return ConfigRuntimeReport(
        config_name=config_name,
        available_clients=available_clients,
        trace_count=len(trace_reports),
        runnable_policy_invocations=sum(
            trace.runnable_policy_invocations for trace in trace_reports
        ),
        experiment_runs=sum(trace.experiment_runs for trace in trace_reports),
        serial_wall_hours=sum(trace.serial_wall_hours for trace in trace_reports),
        traces=trace_reports,
    )


def _iter_config_paths(config_dir: str | Path) -> list[Path]:
    root = Path(config_dir).expanduser().resolve()
    paths = sorted(root.glob("*.json")) + sorted(root.glob("*.jsonl"))
    return [
        path
        for path in paths
        if path.name not in {"base.json", "base.jsonl"}
    ]


def build_directory_runtime_report(config_dir: str | Path) -> dict[str, Any]:
    reports = [
        build_config_runtime_report(path)
        for path in _iter_config_paths(config_dir)
    ]
    valid_reports = [report for report in reports if report.error is None]
    invalid_reports = [report for report in reports if report.error is not None]

    return {
        "configs": [asdict(report) for report in reports],
        "total": {
            "config_count": len(valid_reports),
            "invalid_config_count": len(invalid_reports),
            "runnable_policy_invocations": sum(
                report.runnable_policy_invocations for report in valid_reports
            ),
            "experiment_runs": sum(report.experiment_runs for report in valid_reports),
            "serial_wall_hours": sum(
                report.serial_wall_hours for report in valid_reports
            ),
        },
    }


def _format_hours(hours: float) -> str:
    return f"{hours:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Count serial runtime implied by batch configs. "
            "Runtime is computed as window_hours multiplied by the number of "
            "experiment combinations each config will execute."
        )
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
        help="Emit JSON instead of a text summary.",
    )
    args = parser.parse_args()

    report = build_directory_runtime_report(args.config_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    for config in report["configs"]:
        if config["error"] is not None:
            print(f"{config['config_name']}\tERROR={config['error']}")
            continue

        print(
            "\t".join(
                [
                    config["config_name"],
                    f"traces={config['trace_count']}",
                    f"policy_invocations={config['runnable_policy_invocations']}",
                    f"experiment_runs={config['experiment_runs']}",
                    f"serial_wall_hours={_format_hours(config['serial_wall_hours'])}",
                ]
            )
        )
        for trace in config["traces"]:
            print(
                "  "
                + "\t".join(
                    [
                        trace["trace"],
                        f"window={trace['window']}",
                        f"window_hours={_format_hours(trace['window_hours'])}",
                        f"policies={trace['policy_count']}",
                        f"sweeps={trace['sweep_count']}",
                        f"requested_n_devices={trace['requested_device_count']}",
                        f"policy_device_runs={trace['policy_device_runs']}",
                        f"experiment_runs={trace['experiment_runs']}",
                        f"serial_wall_hours={_format_hours(trace['serial_wall_hours'])}",
                    ]
                )
            )

    total = report["total"]
    print("TOTAL")
    print(
        "\t".join(
            [
                f"config_count={total['config_count']}",
                f"invalid_config_count={total['invalid_config_count']}",
                f"policy_invocations={total['runnable_policy_invocations']}",
                f"experiment_runs={total['experiment_runs']}",
                f"serial_wall_hours={_format_hours(total['serial_wall_hours'])}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
