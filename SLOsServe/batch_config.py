from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


LIST_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "policies": ("policy",),
    "extra_args": ("bench_extra_args", "cli_args"),
    "n_devices": ("n_device",),
    "load_scales": ("load_scale",),
    "ttft_slo_scales": ("ttft_slo_scale", "slo_ttft", "slo_ttfts"),
    "slo_tpots": ("slo_tpot",),
    "perf_model_errs": (
        "perf_model_err",
        "per_model_err",
        "per_model_errs",
    ),
}

TRACE_APPEND_LIST_ALIASES: dict[str, tuple[str, ...]] = {
    "extra_policies": ("extra_policy",),
}

SCALAR_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "window": (),
    "model_name": (),
    "profit": (),
    "admission_mode": (),
    "slo_routing_overhead": (),
    "scheduling_overhead": (),
    "scheduling_safety_margin": (),
    "router_safety_margin": (),
    "routing_overhead": (),
    "routing_fallback_policy": (),
    "tensor_parallel_size": (),
    "output_dir": (),
}

DEFAULT_TRACE_SPEC: dict[str, Any] = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "profit": "constant",
    "admission_mode": "arrival",
    "slo_routing_overhead": "0.05",
    "scheduling_overhead": "0.003",
    "scheduling_safety_margin": "0.002",
    "routing_overhead": "-1.0",
    "routing_fallback_policy": "reject",
    "tensor_parallel_size": "1",
    "load_scales": ["1.0"],
    "ttft_slo_scales": ["5.0"],
    "slo_tpots": ["0.05"],
    "perf_model_errs": ["1.0"],
}

TOP_LEVEL_SCALAR_ALIASES: dict[str, tuple[str, ...]] = {
    "server_clients": ("clients",),
}

RESERVED_BATCH_CLI_FLAGS: dict[str, str] = {
    "--model_name": "model_name",
    "--tensor_parallel_size": "tensor_parallel_size",
}


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = {key: value for key, value in base.items()}
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def load_batch_config(path: str | Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if config_path in _stack:
        cycle = " -> ".join(str(item) for item in (*_stack, config_path))
        raise ValueError(f"Detected batch config cycle: {cycle}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise TypeError(f"Batch config {config_path} must be a JSON object.")

    merged: dict[str, Any] = {}
    for parent in raw.get("extends", []):
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        merged = _deep_merge(
            merged,
            load_batch_config(parent_path, (*_stack, config_path)),
        )

    current = {key: value for key, value in raw.items() if key != "extends"}
    return _deep_merge(merged, current)


def combine_batch_configs(config_dir: str | Path) -> dict[str, Any]:
    root = Path(config_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Batch config directory not found: {root}")

    combined: dict[str, Any] = {}
    for path in sorted(root.glob("*.json")) + sorted(root.glob("*.jsonl")):
        combined[path.name] = normalize_batch_config(load_batch_config(path))
    return combined


def _quote_shell(value: str) -> str:
    return shlex.quote(value)


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    raise TypeError(f"Unsupported scalar value: {value!r}")


def _normalize_server_router_kwargs(value: Any) -> str:
    return json.dumps(
        _normalize_server_router_kwargs_dict(value),
        separators=(",", ":"),
        sort_keys=True,
    )


def _normalize_server_router_kwargs_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(
        "server_router_kwargs must be a JSON string or object, "
        f"got {type(value).__name__}."
    )


def _normalize_shell_args(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return [_format_scalar(item) for item in value]
    return [_format_scalar(value)]


def _field_names(canonical: str, aliases: tuple[str, ...]) -> tuple[str, ...]:
    return (canonical, *aliases)


def _pick_field_value(spec: dict[str, Any], canonical: str, aliases: tuple[str, ...]) -> Any:
    found = [name for name in _field_names(canonical, aliases) if name in spec]
    if len(found) > 1:
        raise ValueError(
            f"Conflicting aliases for {canonical}: {', '.join(found)}"
        )
    if not found:
        return None
    return spec[found[0]]


def _normalize_list_value(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if field_name == "extra_args" and isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{field_name} must be a non-empty list.")
        return [_format_scalar(item) for item in value]
    return [_format_scalar(value)]


def _append_unique_list(base: list[str], extra: list[str]) -> list[str]:
    merged = list(base)
    for item in extra:
        if item not in merged:
            merged.append(item)
    return merged


def _parse_legacy_trace_config(value: str) -> dict[str, Any]:
    parts = value.split()
    if len(parts) < 3:
        raise ValueError(
            "Trace config strings must look like "
            '"<window> <load_scale> <n_device> [n_device...]"'
        )
    return {
        "window": parts[0],
        "load_scales": [parts[1]],
        "n_devices": parts[2:],
    }


def _consume_reserved_cli_overrides(
    args: list[str],
    *,
    field_name: str,
    target_spec: dict[str, Any],
    locked_fields: set[str] | None = None,
) -> list[str]:
    remaining: list[str] = []
    seen: dict[str, str] = {}
    locked_fields = locked_fields or set()
    idx = 0

    while idx < len(args):
        arg = args[idx]
        canonical: str | None = None
        value: str | None = None
        consume_count = 1

        if arg in RESERVED_BATCH_CLI_FLAGS:
            canonical = RESERVED_BATCH_CLI_FLAGS[arg]
            if idx + 1 >= len(args):
                raise ValueError(f"{field_name} is missing a value for {arg}.")
            value = _format_scalar(args[idx + 1])
            consume_count = 2
        else:
            for flag, field in RESERVED_BATCH_CLI_FLAGS.items():
                prefix = f"{flag}="
                if arg.startswith(prefix):
                    canonical = field
                    value = _format_scalar(arg[len(prefix):])
                    break

        if canonical is None:
            remaining.append(arg)
            idx += 1
            continue

        existing = target_spec.get(canonical)
        if existing is not None and _format_scalar(existing) != value:
            if canonical in locked_fields:
                raise ValueError(
                    f"Conflicting {canonical} values between {field_name} and "
                    f"canonical config fields: {_format_scalar(existing)!r} vs {value!r}"
                )
        if canonical in seen and seen[canonical] != value:
            raise ValueError(
                f"Conflicting {canonical} values inside {field_name}: "
                f"{seen[canonical]!r} vs {value!r}"
            )

        seen[canonical] = value
        target_spec[canonical] = value
        idx += consume_count

    return remaining


def _build_trace_server_router_kwargs(
    server_router_kwargs: dict[str, Any],
    trace_spec: dict[str, Any],
) -> str:
    merged = dict(server_router_kwargs)
    merged["model_name"] = trace_spec["model_name"]
    if "scheduling_overhead" in trace_spec:
        merged["scheduling_overhead"] = float(trace_spec["scheduling_overhead"])
    if "scheduling_safety_margin" in trace_spec:
        merged["scheduling_safety_margin"] = float(
            trace_spec["scheduling_safety_margin"]
        )
    if "router_safety_margin" in trace_spec:
        merged["router_safety_margin"] = float(
            trace_spec["router_safety_margin"]
        )

    slo_tpots = trace_spec.get("slo_tpots", [])
    if len(slo_tpots) == 1:
        merged["tpot"] = float(slo_tpots[0])
    else:
        merged.pop("tpot", None)

    perf_model_errs = trace_spec.get("perf_model_errs", [])
    if len(perf_model_errs) == 1:
        merged["perf_model_err"] = float(perf_model_errs[0])
    else:
        merged.pop("perf_model_err", None)

    return json.dumps(merged, separators=(",", ":"), sort_keys=True)


def _build_trace_server_args(
    trace_spec: dict[str, Any],
    extra_server_args: list[str],
) -> list[str]:
    return [
        *extra_server_args,
        "--model_name",
        trace_spec["model_name"],
        "--tensor_parallel_size",
        trace_spec["tensor_parallel_size"],
    ]


def _normalize_trace_spec(spec: Any, base: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(spec, str):
        raw_spec = _parse_legacy_trace_config(spec)
    elif isinstance(spec, dict):
        raw_spec = spec
    else:
        raise TypeError(
            "Trace config entries must be strings or objects, "
            f"got {type(spec).__name__}."
        )

    normalized = dict(base or {})
    explicit_scalar_fields = {
        field_name
        for field_name, aliases in SCALAR_FIELD_ALIASES.items()
        if _pick_field_value(raw_spec, field_name, aliases) is not None
    }

    for field_name, aliases in LIST_FIELD_ALIASES.items():
        if field_name == "extra_args":
            continue
        value = _pick_field_value(raw_spec, field_name, aliases)
        if value is not None:
            normalized[field_name] = _normalize_list_value(
                value,
                field_name=field_name,
            )

    for field_name, aliases in SCALAR_FIELD_ALIASES.items():
        value = _pick_field_value(raw_spec, field_name, aliases)
        if value is not None:
            normalized[field_name] = _format_scalar(value)

    extra_args = _pick_field_value(raw_spec, "extra_args", LIST_FIELD_ALIASES["extra_args"])
    if extra_args is not None:
        cleaned_extra_args = _consume_reserved_cli_overrides(
            _normalize_list_value(extra_args, field_name="extra_args"),
            field_name="extra_args",
            target_spec=normalized,
            locked_fields=explicit_scalar_fields,
        )
        if cleaned_extra_args:
            normalized["extra_args"] = [
                *normalized.get("extra_args", []),
                *cleaned_extra_args,
            ]

    extra_policies = _pick_field_value(
        raw_spec,
        "extra_policies",
        TRACE_APPEND_LIST_ALIASES["extra_policies"],
    )
    if extra_policies is not None:
        normalized["policies"] = _append_unique_list(
            normalized.get("policies", []),
            _normalize_list_value(extra_policies, field_name="extra_policies"),
        )

    return normalized


def normalize_batch_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = {key: value for key, value in config.items()}

    for field_name, aliases in TOP_LEVEL_SCALAR_ALIASES.items():
        value = _pick_field_value(normalized, field_name, aliases)
        if value is not None:
            normalized[field_name] = _format_scalar(value)
            for alias in aliases:
                normalized.pop(alias, None)

    server_router_kwargs = normalized.get("server_router_kwargs")
    server_router_kwargs_dict: dict[str, Any] = {}
    if server_router_kwargs is not None:
        server_router_kwargs_dict = _normalize_server_router_kwargs_dict(
            server_router_kwargs
        )
        normalized["server_router_kwargs"] = _normalize_server_router_kwargs(
            server_router_kwargs
        )
    if "extra_server_args" in normalized:
        normalized["extra_server_args"] = _normalize_shell_args(
            normalized["extra_server_args"],
            field_name="extra_server_args",
        )

    default_trace_source: dict[str, Any] = {}
    raw_defaults = normalized.get("defaults", {})
    if raw_defaults is not None:
        if not isinstance(raw_defaults, dict):
            raise TypeError("defaults must be an object when provided.")
        default_trace_source = _deep_merge(default_trace_source, raw_defaults)

    for field_name, aliases in LIST_FIELD_ALIASES.items():
        for name in _field_names(field_name, aliases):
            if name in normalized:
                default_trace_source[name] = normalized[name]
    for field_name, aliases in SCALAR_FIELD_ALIASES.items():
        for name in _field_names(field_name, aliases):
            if name in normalized:
                default_trace_source[name] = normalized[name]

    if "extra_server_args" in normalized:
        cleaned_extra_server_args = _consume_reserved_cli_overrides(
            normalized["extra_server_args"],
            field_name="extra_server_args",
            target_spec=default_trace_source,
        )
        if cleaned_extra_server_args:
            normalized["extra_server_args"] = cleaned_extra_server_args
        else:
            normalized.pop("extra_server_args", None)

    default_trace_spec = _normalize_trace_spec(
        default_trace_source,
        base=DEFAULT_TRACE_SPEC,
    )

    raw_configs = normalized.get("configs", {})
    if not isinstance(raw_configs, dict):
        raise TypeError("configs must be an object keyed by trace pair.")

    raw_traces = normalized.get("traces")
    if raw_traces is None:
        trace_order = list(raw_configs.keys())
    else:
        if not isinstance(raw_traces, list) or not all(
            isinstance(item, str) for item in raw_traces
        ):
            raise TypeError("traces must be a list of strings.")
        trace_order = raw_traces

    normalized_traces: dict[str, dict[str, Any]] = {}
    for trace in trace_order:
        trace_spec_raw = raw_configs.get(trace)
        if trace_spec_raw is None:
            raise ValueError(f"Missing config entry for trace {trace}.")
        trace_spec = _normalize_trace_spec(
            trace_spec_raw,
            base=default_trace_spec,
        )
        if "window" not in trace_spec:
            raise ValueError(f"Trace {trace} is missing required field window.")
        if "n_devices" not in trace_spec:
            raise ValueError(f"Trace {trace} is missing required field n_devices.")
        if "policies" not in trace_spec:
            raise ValueError(
                f"Trace {trace} is missing policies. "
                "Set top-level policies/defaults or per-trace policies."
            )
        normalized_traces[trace] = trace_spec

    trace_server_args = {
        trace: _build_trace_server_args(
            spec,
            normalized.get("extra_server_args", []),
        )
        for trace, spec in normalized_traces.items()
    }
    trace_server_router_kwargs = {
        trace: _build_trace_server_router_kwargs(server_router_kwargs_dict, spec)
        for trace, spec in normalized_traces.items()
        if server_router_kwargs_dict
    }

    normalized["traces"] = trace_order
    normalized["trace_specs"] = normalized_traces
    normalized["trace_server_args"] = trace_server_args
    if trace_server_router_kwargs:
        normalized["trace_server_router_kwargs"] = trace_server_router_kwargs
    return {
        key: value
        for key, value in normalized.items()
        if key not in LIST_FIELD_ALIASES
        and key not in SCALAR_FIELD_ALIASES
        and key != "defaults"
        and key != "configs"
    }


def _render_shell_list(name: str, values: list[str]) -> list[str]:
    lines = [f"{name}=("]
    lines.extend(f"  {_quote_shell(item)}" for item in values)
    lines.append(")")
    return lines


def _render_shell_assoc_array(name: str, values: dict[str, str]) -> list[str]:
    lines = [f"declare -gA {name}=("]
    for key, value in values.items():
        lines.append(f"  [{_quote_shell(key)}]={_quote_shell(value)}")
    lines.append(")")
    return lines


def render_bash_assignments(config: dict[str, Any]) -> str:
    normalized = normalize_batch_config(config)
    trace_specs: dict[str, dict[str, Any]] = normalized["trace_specs"]
    lines = ["BATCH_CONFIG_LOADED=1"]

    if "experiment_name" in normalized:
        lines.append(
            f"EXPERIMENT_NAME={_quote_shell(str(normalized['experiment_name']))}"
        )
    if "server_clients" in normalized:
        lines.append(
            f"SERVER_CLIENTS={_quote_shell(str(normalized['server_clients']))}"
        )
    if "server_router_kwargs" in normalized:
        lines.append(
            "SERVER_ROUTER_KWARGS="
            f"{_quote_shell(normalized['server_router_kwargs'])}"
        )
    if "extra_server_args" in normalized:
        lines.append(
            "SERVER_EXTRA_ARGS_SHELL="
            f"{_quote_shell(' '.join(_quote_shell(arg) for arg in normalized['extra_server_args']))}"
        )

    lines.extend(_render_shell_list("TRACES", normalized["traces"]))
    lines.extend(_render_shell_assoc_array(
        "TRACE_SERVER_ARGS_SHELL",
        {
            trace: " ".join(_quote_shell(arg) for arg in args)
            for trace, args in normalized["trace_server_args"].items()
        },
    ))
    if "trace_server_router_kwargs" in normalized:
        lines.extend(_render_shell_assoc_array(
            "TRACE_SERVER_ROUTER_KWARGS",
            normalized["trace_server_router_kwargs"],
        ))

    for field_name in LIST_FIELD_ALIASES:
        if field_name == "extra_args":
            assoc_name = "TRACE_EXTRA_ARGS_SHELL"
            assoc_values = {
                trace: " ".join(_quote_shell(arg) for arg in spec[field_name])
                for trace, spec in trace_specs.items()
                if field_name in spec
            }
        else:
            assoc_name = f"TRACE_{field_name.upper()}"
            assoc_values = {
                trace: " ".join(spec[field_name])
                for trace, spec in trace_specs.items()
                if field_name in spec
            }
        lines.extend(_render_shell_assoc_array(assoc_name, assoc_values))

    for field_name in SCALAR_FIELD_ALIASES:
        assoc_name = f"TRACE_{field_name.upper()}"
        assoc_values = {
            trace: spec[field_name]
            for trace, spec in trace_specs.items()
            if field_name in spec
        }
        lines.extend(_render_shell_assoc_array(assoc_name, assoc_values))

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render composed batch experiment configs for run_batch.sh."
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=str,
        help="Path to a JSON batch config file.",
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Render bash assignments for eval/source style loading.",
    )
    parser.add_argument(
        "--combine-dir",
        type=str,
        default=None,
        help="Combine and normalize every *.json config in a directory.",
    )
    args = parser.parse_args()

    if args.combine_dir:
        if args.shell:
            raise ValueError("--shell is only supported for a single config.")
        print(json.dumps(combine_batch_configs(args.combine_dir), indent=2, sort_keys=True))
        return 0

    if not args.config:
        parser.error("config is required unless --combine-dir is used.")

    config = load_batch_config(args.config)
    if args.shell:
        print(render_bash_assignments(config))
    else:
        print(json.dumps(normalize_batch_config(config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
