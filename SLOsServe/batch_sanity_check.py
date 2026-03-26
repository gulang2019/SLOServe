from __future__ import annotations

import argparse
import json
import shlex
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
class Finding:
    severity: str
    message: str


@dataclass
class ConfigSanityReport:
    config_name: str
    status: str
    error_count: int
    warning_count: int
    findings: list[Finding]


def _iter_config_paths(config_dir: str | Path) -> list[Path]:
    root = Path(config_dir).expanduser().resolve()
    paths = sorted(root.glob("*.json")) + sorted(root.glob("*.jsonl"))
    return [
        path
        for path in paths
        if path.name not in {"base.json", "base.jsonl"}
    ]


def _append_finding(findings: list[Finding], severity: str, message: str) -> None:
    findings.append(Finding(severity=severity, message=message))


def _find_duplicates(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    duplicates: list[Any] = []
    duplicate_set: set[Any] = set()
    for value in values:
        if value in seen and value not in duplicate_set:
            duplicates.append(value)
            duplicate_set.add(value)
        seen.add(value)
    return duplicates


def _split_shell_args(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _extract_last_flag_value(args: list[str], flag: str) -> tuple[str | None, int]:
    value: str | None = None
    occurrences = 0
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token == flag:
            occurrences += 1
            if idx + 1 >= len(args):
                raise ValueError(f"{flag} is missing a value")
            value = args[idx + 1]
            idx += 2
            continue
        if token.startswith(f"{flag}="):
            occurrences += 1
            value = token.split("=", 1)[1]
        idx += 1
    return value, occurrences


def _parse_positive_int(
    raw_value: str,
    *,
    field_name: str,
    findings: list[Finding],
    trace: str | None = None,
) -> int | None:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        prefix = f"trace {trace}: " if trace else ""
        _append_finding(
            findings,
            "error",
            f"{prefix}{field_name} must be an integer, got {raw_value!r}",
        )
        return None
    if value <= 0:
        prefix = f"trace {trace}: " if trace else ""
        _append_finding(
            findings,
            "error",
            f"{prefix}{field_name} must be positive, got {value}",
        )
        return None
    return value


def _resolve_trace_tensor_parallel_size(
    args: list[str],
    *,
    field_name: str,
    findings: list[Finding],
    trace: str,
    required: bool,
) -> int | None:
    try:
        tp_raw, tp_occurrences = _extract_last_flag_value(
            args,
            "--tensor_parallel_size",
        )
    except ValueError as exc:
        _append_finding(findings, "error", f"trace {trace}: {field_name}: {exc}")
        return None

    if tp_occurrences > 1:
        _append_finding(
            findings,
            "warning",
            f"trace {trace}: {field_name} sets --tensor_parallel_size multiple times; the last value wins",
        )

    if tp_raw is None:
        if not required:
            return None
        _append_finding(
            findings,
            "error",
            f"trace {trace}: {field_name} does not set --tensor_parallel_size",
        )
        return None

    return _parse_positive_int(
        tp_raw,
        field_name=f"{field_name} tensor_parallel_size",
        findings=findings,
        trace=trace,
    )


def build_config_sanity_report(config_path: str | Path) -> ConfigSanityReport:
    config_name = Path(config_path).name
    findings: list[Finding] = []

    try:
        loaded = load_batch_config(config_path)
    except Exception as exc:
        _append_finding(findings, "error", str(exc))
        return ConfigSanityReport(
            config_name=config_name,
            status="ERROR",
            error_count=1,
            warning_count=0,
            findings=findings,
        )

    raw_traces = loaded.get("traces")
    raw_configs = loaded.get("configs", {})
    raw_policies = loaded.get("policies")

    if isinstance(raw_traces, list):
        duplicates = _find_duplicates(raw_traces)
        if duplicates:
            _append_finding(
                findings,
                "warning",
                f"duplicate traces declared: {', '.join(str(item) for item in duplicates)}",
            )
        if isinstance(raw_configs, dict):
            unused_configs = sorted(set(raw_configs) - set(raw_traces))
            if unused_configs:
                _append_finding(
                    findings,
                    "warning",
                    f"config entries not referenced by traces: {', '.join(unused_configs)}",
                )

    if isinstance(raw_policies, list):
        duplicates = _find_duplicates(raw_policies)
        if duplicates:
            _append_finding(
                findings,
                "warning",
                f"duplicate policies declared: {', '.join(str(item) for item in duplicates)}",
            )

    try:
        normalized = normalize_batch_config(loaded)
    except Exception as exc:
        _append_finding(findings, "error", str(exc))
        error_count = sum(1 for finding in findings if finding.severity == "error")
        warning_count = sum(1 for finding in findings if finding.severity == "warning")
        return ConfigSanityReport(
            config_name=config_name,
            status="ERROR",
            error_count=error_count,
            warning_count=warning_count,
            findings=findings,
        )

    available_clients = count_clients_spec(normalized.get("server_clients", "0-7"))
    trace_specs: dict[str, dict[str, Any]] = normalized["trace_specs"]

    trace_model_names = sorted({spec["model_name"] for spec in trace_specs.values()})
    if len(trace_model_names) > 1:
        _append_finding(
            findings,
            "error",
            "multiple model_name values across traces are not supported by run_batch.sh: "
            + ", ".join(trace_model_names),
        )

    for trace, spec in trace_specs.items():
        window_hours = parse_window_seconds(spec["window"]) / 3600.0
        if window_hours <= 0.0:
            _append_finding(
                findings,
                "error",
                f"trace {trace}: window {spec['window']} has non-positive duration",
            )

        n_devices = [
            _parse_positive_int(item, field_name="n_devices", findings=findings, trace=trace)
            for item in spec["n_devices"]
        ]
        n_devices = [item for item in n_devices if item is not None]
        if len(n_devices) != len(spec["n_devices"]):
            continue

        duplicate_n_devices = _find_duplicates(n_devices)
        if duplicate_n_devices:
            _append_finding(
                findings,
                "warning",
                f"trace {trace}: duplicate n_devices values: {', '.join(str(item) for item in duplicate_n_devices)}",
            )

        non_partial_policies = [
            policy for policy in spec["policies"] if not policy_supports_partial_rr(policy)
        ]
        skipped_n_devices = [item for item in n_devices if item > available_clients]
        if non_partial_policies and skipped_n_devices:
            _append_finding(
                findings,
                "warning",
                f"trace {trace}: non-partial policies will skip n_devices > available clients "
                f"({available_clients}): {', '.join(str(item) for item in skipped_n_devices)}",
            )
        if non_partial_policies and skipped_n_devices and len(skipped_n_devices) == len(n_devices):
            _append_finding(
                findings,
                "error",
                f"trace {trace}: all n_devices exceed available clients ({available_clients}) "
                f"for non-partial policies",
            )

        declared_tp = _parse_positive_int(
            spec["tensor_parallel_size"],
            field_name="tensor_parallel_size",
            findings=findings,
            trace=trace,
        )
        if declared_tp is None:
            continue

        effective_bench_tp = declared_tp
        trace_extra_args = _split_shell_args(spec.get("extra_args"))
        parsed_trace_extra_tp = _resolve_trace_tensor_parallel_size(
            trace_extra_args,
            field_name="extra_args",
            findings=findings,
            trace=trace,
            required=False,
        ) if trace_extra_args else None
        if parsed_trace_extra_tp is not None:
            effective_bench_tp = parsed_trace_extra_tp
            if parsed_trace_extra_tp != declared_tp:
                _append_finding(
                    findings,
                    "warning",
                    f"trace {trace}: extra_args overrides tensor_parallel_size from "
                    f"{declared_tp} to {parsed_trace_extra_tp}",
                )

        trace_server_args_map = normalized.get("trace_server_args", {})
        trace_server_args = _split_shell_args(trace_server_args_map.get(trace))
        server_tp = _resolve_trace_tensor_parallel_size(
            trace_server_args,
            field_name="trace_server_args",
            findings=findings,
            trace=trace,
            required=True,
        )
        if server_tp is None:
            continue

        if server_tp is not None and server_tp != effective_bench_tp:
            _append_finding(
                findings,
                "error",
                f"trace {trace}: bench tensor_parallel_size resolves to {effective_bench_tp}, "
                f"but api_server_ray will launch with {server_tp}",
            )

    error_count = sum(1 for finding in findings if finding.severity == "error")
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    status = "ERROR" if error_count else ("WARN" if warning_count else "OK")
    return ConfigSanityReport(
        config_name=config_name,
        status=status,
        error_count=error_count,
        warning_count=warning_count,
        findings=findings,
    )


def build_directory_sanity_report(config_dir: str | Path) -> dict[str, Any]:
    reports = [build_config_sanity_report(path) for path in _iter_config_paths(config_dir)]
    return {
        "configs": [asdict(report) for report in reports],
        "total": {
            "config_count": len(reports),
            "ok_count": sum(1 for report in reports if report.status == "OK"),
            "warn_count": sum(1 for report in reports if report.status == "WARN"),
            "error_count": sum(1 for report in reports if report.status == "ERROR"),
            "finding_count": sum(
                report.error_count + report.warning_count for report in reports
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run sanity checks against batch config JSON or JSONL files."
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

    report = build_directory_sanity_report(args.config_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    for config in report["configs"]:
        print(
            "\t".join(
                [
                    config["config_name"],
                    f"status={config['status']}",
                    f"errors={config['error_count']}",
                    f"warnings={config['warning_count']}",
                ]
            )
        )
        for finding in config["findings"]:
            print(f"  {finding['severity'].upper()}: {finding['message']}")

    total = report["total"]
    print("TOTAL")
    print(
        "\t".join(
            [
                f"configs={total['config_count']}",
                f"ok={total['ok_count']}",
                f"warn={total['warn_count']}",
                f"error={total['error_count']}",
                f"findings={total['finding_count']}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
