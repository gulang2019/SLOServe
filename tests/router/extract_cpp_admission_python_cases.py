#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SLOsServe.router import adm_ctrl


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    num_prompt_tokens: int
    num_computed_tokens: int
    prefill_ddl: float
    slo_tpot: float = 0.05
    prefill_only: bool = False
    output_length: int = 8
    kv_ready_time: float | None = None
    service_tier: str = "default"


@dataclass(frozen=True)
class AdmissionMismatchCase:
    index: int
    device_id: int
    request_id: str
    add_request_timestamp: float
    cpp_result: bool
    cpp_reject_reason: str | None
    python_result: bool
    python_failure: tuple | None
    existing_specs: list[RequestSpec]
    new_spec: RequestSpec

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "device_id": self.device_id,
            "request_id": self.request_id,
            "add_request_timestamp": self.add_request_timestamp,
            "cpp_result": self.cpp_result,
            "cpp_reject_reason": self.cpp_reject_reason,
            "python_result": self.python_result,
            "python_failure": list(self.python_failure)
            if isinstance(self.python_failure, tuple)
            else self.python_failure,
            "existing_specs": [asdict(spec) for spec in self.existing_specs],
            "new_spec": asdict(self.new_spec),
        }


@dataclass(frozen=True)
class _TraceRequestState:
    request_id: str
    num_prompt_tokens: int
    num_computed_tokens: int
    prefill_ddl_abs: float
    slo_tpot: float
    prefill_only: bool
    output_length: int
    kv_ready_time_abs: float | None
    service_tier: str


class _LinearPerfModel:

    def __init__(self, hardware_params: list[float]):
        self.hardware_params = list(hardware_params)

    def get_batch_time(self, batch: list[tuple[int, int]]) -> float:
        num_reqs = len(batch)
        num_tot_tokens = sum(n_tokens for _n_past, n_tokens in batch)
        num_past_tokens = sum(n_past for n_past, _n_tokens in batch)
        num_decode_steps = 1
        return (
            self.hardware_params[0] * num_tot_tokens
            + self.hardware_params[1] * num_reqs
            + self.hardware_params[2] * num_past_tokens
            + self.hardware_params[3] * num_decode_steps
            + self.hardware_params[4]
        )

    def get_bs(
        self,
        t: float,
        num_reqs: int = 1,
        num_past_tokens: int = 0,
        num_decode_steps: int = 1,
    ) -> int:
        return int(
            (
                t
                - self.hardware_params[4]
                - self.hardware_params[3] * num_decode_steps
                - self.hardware_params[2] * num_past_tokens
                - self.hardware_params[1] * num_reqs
            )
            / self.hardware_params[0]
        )


@contextmanager
def _suppress_native_output(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _load_events(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        first_non_ws = ""
        while True:
            ch = f.read(1)
            if ch == "":
                return []
            if not ch.isspace():
                first_non_ws = ch
                break
        f.seek(0)
        if first_non_ws == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def _make_planner(
    request_specs: list[RequestSpec],
    *,
    now: float = 0.0,
    hardware_params: list[float] | None = None,
    max_batch_size: int = 4096,
    max_decode_length: int = 64,
    is_oracle: bool = True,
    quiet_native: bool = True,
) -> tuple[adm_ctrl.BatchPlanner, dict[str, float]]:
    if hardware_params is None:
        # This mirrors the current control-side constants after the
        # scheduling-overhead adjustment.
        hardware_params = [6.1e-5, 1.6e-5, 3.5e-7, 0.0, 0.018]

    with _suppress_native_output(quiet_native):
        planner = adm_ctrl.BatchPlanner(
            _perf_model=_LinearPerfModel(hardware_params),
            _block_size=16,
            _max_decode_length=max_decode_length,
            _num_free_blocks=100000,
            _max_batch_size=max_batch_size,
            _is_oracle=is_oracle,
        )
    now_box = {"t": now}
    planner._now = lambda: now_box["t"]
    planner.batch_id = -1
    for spec in request_specs:
        planner._requests[spec.request_id] = adm_ctrl.Request(
            request_id=spec.request_id,
            num_prompt_tokens=spec.num_prompt_tokens,
            num_computed_tokens=spec.num_computed_tokens,
            prefill_ddl=spec.prefill_ddl,
            slo_tpot=spec.slo_tpot,
            prefill_only=spec.prefill_only,
            kv_ready_time=spec.kv_ready_time,
            output_length=spec.output_length,
            service_tier=spec.service_tier,
        )
    return planner, now_box


def _make_request(spec: RequestSpec) -> adm_ctrl.Request:
    return adm_ctrl.Request(
        request_id=spec.request_id,
        num_prompt_tokens=spec.num_prompt_tokens,
        num_computed_tokens=spec.num_computed_tokens,
        prefill_ddl=spec.prefill_ddl,
        slo_tpot=spec.slo_tpot,
        prefill_only=spec.prefill_only,
        kv_ready_time=spec.kv_ready_time,
        output_length=spec.output_length,
        service_tier=spec.service_tier,
    )


def _cpp_admission_with_new(
    existing_specs: list[RequestSpec],
    new_spec: RequestSpec,
    *,
    now: float = 0.0,
    hardware_params: list[float] | None = None,
    quiet_native: bool = True,
) -> tuple[bool, str | None]:
    planner, _ = _make_planner(
        copy.deepcopy(existing_specs),
        now=now,
        hardware_params=hardware_params,
        is_oracle=True,
        quiet_native=quiet_native,
    )
    with _suppress_native_output(quiet_native):
        return planner._cpp_feasible_with_new(_make_request(new_spec), now)


def _shadow_refresh_fast_feasible(
    existing_specs: list[RequestSpec],
    new_spec: RequestSpec,
    *,
    now: float = 0.0,
    hardware_params: list[float] | None = None,
    quiet_native: bool = True,
) -> tuple[bool, tuple | None]:
    planner, now_box = _make_planner(
        copy.deepcopy(existing_specs) + [copy.deepcopy(new_spec)],
        now=now,
        hardware_params=hardware_params,
        is_oracle=True,
        quiet_native=quiet_native,
    )
    epsilon = 1e-12

    while True:
        unfinished = [
            req
            for req in planner._requests.values()
            if req.arrived and not req.finished(planner._is_oracle)
        ]
        if not unfinished:
            return True, None

        planner._next_batch_time = now_box["t"]
        py_feasible, py_batches, _ = planner._refresh_fast()
        if not py_feasible or not py_batches:
            return False, ("empty_or_infeasible", now_box["t"])

        batch = py_batches[0]
        batch_end = now_box["t"] + planner._get_batch_time(batch)
        scheduled_tokens = batch.n_scheduled_tokens

        for req in unfinished:
            next_load = req.get_next_load()
            next_ddl = req.get_next_ddl()
            if next_load is None or next_ddl is None:
                continue
            n_scheduled = scheduled_tokens.get(req.request_id, 0)
            if n_scheduled >= next_load:
                if batch_end > next_ddl + epsilon:
                    return False, (
                        "scheduled_but_late",
                        req.request_id,
                        batch_end,
                        next_ddl,
                        next_load,
                        n_scheduled,
                        dict(scheduled_tokens),
                    )
            elif batch_end > next_ddl + epsilon:
                return False, (
                    "missed_before_service",
                    req.request_id,
                    batch_end,
                    next_ddl,
                    next_load,
                    n_scheduled,
                    dict(scheduled_tokens),
                )

        for req_id, n_scheduled in scheduled_tokens.items():
            planner._requests[req_id].commit(n_scheduled)
        finished_request_ids = [
            req_id
            for req_id, req in list(planner._requests.items())
            if req.finished(planner._is_oracle)
        ]
        for req_id in finished_request_ids:
            planner.finish_request(req_id)
        now_box["t"] = batch_end


def _request_spec_from_events(
    arrival_event: dict[str, Any],
    add_request_event: dict[str, Any],
    add_request_sch_event: dict[str, Any] | None,
    *,
    default_slo_tpot: float,
) -> RequestSpec:
    sch_extra = (add_request_sch_event or {}).get("extra_args", {})
    kv_transfer_params = sch_extra.get("kv_transfer_params") or {}
    add_request_timestamp = float(add_request_event["timestamp"])
    kv_ready_time = None
    if sch_extra.get("load_kv_async") and kv_transfer_params.get("arrival_time") is not None:
        kv_ready_time = float(kv_transfer_params["arrival_time"]) - add_request_timestamp
    num_computed_tokens = int(
        sch_extra.get("num_external_computed_tokens", 0)
        + sch_extra.get("num_new_local_computed_tokens", 0)
    )
    prefill_only = bool(
        arrival_event.get("prefill_only", False)
        or kv_transfer_params.get("do_remote_decode", False)
    )
    return RequestSpec(
        request_id=str(add_request_event["request_id"]),
        num_prompt_tokens=int(arrival_event.get("prompt_tokens", 0)),
        num_computed_tokens=num_computed_tokens,
        prefill_ddl=float(arrival_event.get("prefill_ddl", 0.0)) - add_request_timestamp,
        slo_tpot=default_slo_tpot,
        prefill_only=prefill_only,
        output_length=int(arrival_event.get("max_tokens", 0)),
        kv_ready_time=kv_ready_time,
    )


def _live_state_to_request_spec(
    state: _TraceRequestState,
    *,
    now: float,
) -> RequestSpec:
    kv_ready_time = None
    if state.kv_ready_time_abs is not None:
        kv_ready_time = state.kv_ready_time_abs - now
    return RequestSpec(
        request_id=state.request_id,
        num_prompt_tokens=state.num_prompt_tokens,
        num_computed_tokens=state.num_computed_tokens,
        prefill_ddl=state.prefill_ddl_abs - now,
        slo_tpot=state.slo_tpot,
        prefill_only=state.prefill_only,
        output_length=state.output_length,
        kv_ready_time=kv_ready_time,
        service_tier=state.service_tier,
    )


def _request_spec_to_live_state(
    spec: RequestSpec,
    *,
    now: float,
) -> _TraceRequestState:
    kv_ready_time_abs = None
    if spec.kv_ready_time is not None:
        kv_ready_time_abs = now + spec.kv_ready_time
    return _TraceRequestState(
        request_id=spec.request_id,
        num_prompt_tokens=spec.num_prompt_tokens,
        num_computed_tokens=spec.num_computed_tokens,
        prefill_ddl_abs=now + spec.prefill_ddl,
        slo_tpot=spec.slo_tpot,
        prefill_only=spec.prefill_only,
        output_length=spec.output_length,
        kv_ready_time_abs=kv_ready_time_abs,
        service_tier=spec.service_tier,
    )


def extract_cpp_admission_python_mismatches(
    events: list[dict[str, Any]],
    *,
    default_slo_tpot: float = 0.05,
    limit: int | None = None,
    device_id: int | None = None,
    quiet_native: bool = True,
) -> list[AdmissionMismatchCase]:
    arrival_meta: dict[str, dict[str, Any]] = {}
    live_specs_by_device: dict[int, dict[str, _TraceRequestState]] = {}
    cases: list[AdmissionMismatchCase] = []

    for event in events:
        event_type = event.get("event_type")
        if event_type == "arrival":
            arrival_meta[str(event["request_id"])] = dict(event)
            continue

        if event_type == "add_request_sch":
            request_id = str(event["request_id"])
            arrival_meta.setdefault(request_id, {})["_add_request_sch"] = dict(event)
            continue

        if event_type == "add_request":
            if not event.get("extra_args", {}).get("admitted"):
                continue
            event_device_id = int(event.get("device_id", -1))
            if event_device_id < 0:
                continue
            if device_id is not None and event_device_id != device_id:
                continue

            request_id = str(event["request_id"])
            arrival_event = arrival_meta.get(request_id)
            if arrival_event is None:
                continue
            add_request_sch_event = arrival_event.get("_add_request_sch")
            new_spec = _request_spec_from_events(
                arrival_event,
                event,
                add_request_sch_event,
                default_slo_tpot=default_slo_tpot,
            )

            live_specs = live_specs_by_device.setdefault(event_device_id, {})
            current_time = float(event["timestamp"])
            existing_specs = [
                _live_state_to_request_spec(spec, now=current_time)
                for spec in live_specs.values()
            ]
            cpp_result, cpp_reject_reason = _cpp_admission_with_new(
                existing_specs,
                new_spec,
                quiet_native=quiet_native,
            )
            python_result, python_failure = _shadow_refresh_fast_feasible(
                existing_specs,
                new_spec,
                quiet_native=quiet_native,
            )
            if cpp_result and not python_result:
                cases.append(
                    AdmissionMismatchCase(
                        index=len(cases),
                        device_id=event_device_id,
                        request_id=request_id,
                        add_request_timestamp=float(event["timestamp"]),
                        cpp_result=cpp_result,
                        cpp_reject_reason=cpp_reject_reason,
                        python_result=python_result,
                        python_failure=python_failure,
                        existing_specs=existing_specs,
                        new_spec=new_spec,
                    )
                )
                if limit is not None and len(cases) >= limit:
                    return cases
            live_specs[request_id] = _request_spec_to_live_state(
                new_spec,
                now=current_time,
            )
            continue

        if event_type == "batch":
            event_device_id = int(event.get("device_id", -1))
            live_specs = live_specs_by_device.setdefault(event_device_id, {})
            for request_id, n_scheduled in (event.get("num_scheduled_tokens") or {}).items():
                spec = live_specs.get(str(request_id))
                if spec is None:
                    continue
                live_specs[str(request_id)] = _TraceRequestState(
                    request_id=spec.request_id,
                    num_prompt_tokens=spec.num_prompt_tokens,
                    num_computed_tokens=spec.num_computed_tokens + int(n_scheduled),
                    prefill_ddl_abs=spec.prefill_ddl_abs,
                    slo_tpot=spec.slo_tpot,
                    prefill_only=spec.prefill_only,
                    output_length=spec.output_length,
                    kv_ready_time_abs=spec.kv_ready_time_abs,
                    service_tier=spec.service_tier,
                )
            continue

        if event_type == "finish":
            event_device_id = int(event.get("device_id", -1))
            live_specs = live_specs_by_device.get(event_device_id)
            if live_specs is not None:
                live_specs.pop(str(event["request_id"]), None)

    return cases


def _render_request_spec(spec: RequestSpec, *, indent: str) -> str:
    lines = [
        f"{indent}RequestSpec(",
        f"{indent}    request_id={spec.request_id!r},",
        f"{indent}    num_prompt_tokens={spec.num_prompt_tokens},",
        f"{indent}    num_computed_tokens={spec.num_computed_tokens},",
        f"{indent}    prefill_ddl={spec.prefill_ddl!r},",
    ]
    if spec.slo_tpot != 0.05:
        lines.append(f"{indent}    slo_tpot={spec.slo_tpot!r},")
    if spec.prefill_only:
        lines.append(f"{indent}    prefill_only=True,")
    if spec.output_length != 8:
        lines.append(f"{indent}    output_length={spec.output_length},")
    if spec.kv_ready_time is not None:
        lines.append(f"{indent}    kv_ready_time={spec.kv_ready_time!r},")
    if spec.service_tier != "default":
        lines.append(f"{indent}    service_tier={spec.service_tier!r},")
    lines.append(f"{indent})")
    return "\n".join(lines)


def render_pytest_stub(case: AdmissionMismatchCase) -> str:
    suffix = str(case.request_id).replace("-", "_")
    rendered_new_spec = _render_request_spec(case.new_spec, indent="    ")
    rendered_new_spec = rendered_new_spec.replace("    RequestSpec(", "RequestSpec(", 1)
    lines = [
        "@pytest.mark.xfail(",
        "    strict=True,",
        "    reason=(",
        f"        \"Trace-extracted optimistic C-admission mismatch from device {case.device_id}, \"",
        f"        \"request {case.request_id}. Python failure: {case.python_failure!r}\"",
        "    ),",
        ")",
        f"def test_cpp_admission_python_shadow_trace_case_{case.index:03d}_{suffix}():",
        "    existing_specs = [",
    ]
    for spec in case.existing_specs:
        lines.append(_render_request_spec(spec, indent="        ") + ",")
    lines.extend(
        [
            "    ]",
            f"    new_spec = {rendered_new_spec}",
            "    _assert_admission_parity(existing_specs, new_spec)",
        ]
    )
    return "\n".join(lines)


def _render_summary(case: AdmissionMismatchCase) -> str:
    failure = case.python_failure
    violating_request = failure[1] if isinstance(failure, tuple) and len(failure) > 1 else "?"
    failure_kind = failure[0] if isinstance(failure, tuple) and failure else "unknown"
    return (
        f"[{case.index}] device={case.device_id} request={case.request_id} "
        f"python_failure={failure_kind}:{violating_request} "
        f"existing={len(case.existing_specs)} add_request_ts={case.add_request_timestamp:.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract trace-derived cases where C local admission accepts a request, "
            "but a Python shadow replay of _refresh_fast() is infeasible."
        )
    )
    parser.add_argument("events_file", type=Path)
    parser.add_argument(
        "--default-slo-tpot",
        type=float,
        default=0.05,
        help="Fallback TPOT SLO to use when the trace does not record one.",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=None,
        help="Only inspect admissions on this device.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of mismatch cases to emit.",
    )
    parser.add_argument(
        "--format",
        choices=("summary", "json", "pytest"),
        default="summary",
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file. Defaults to stdout.",
    )
    parser.add_argument(
        "--show-existing",
        action="store_true",
        help="Include the existing request specs in summary mode.",
    )
    parser.add_argument(
        "--no-quiet-native",
        action="store_true",
        help="Do not suppress noisy stdout/stderr from the native C++ planner.",
    )
    args = parser.parse_args()

    events = _load_events(args.events_file)
    cases = extract_cpp_admission_python_mismatches(
        events,
        default_slo_tpot=args.default_slo_tpot,
        limit=args.limit,
        device_id=args.device_id,
        quiet_native=not args.no_quiet_native,
    )

    if args.format == "json":
        payload = json.dumps([case.to_json() for case in cases], indent=2)
    elif args.format == "pytest":
        body = [
            "# Paste into tests/router/test_batch_planner_admission_parity.py",
            "# and ensure pytest is imported in that file.",
            "",
        ]
        body.extend(render_pytest_stub(case) + "\n" for case in cases)
        payload = "\n".join(body).rstrip() + "\n"
    else:
        summary_lines = [f"Found {len(cases)} mismatch case(s)."]
        for case in cases:
            summary_lines.append(_render_summary(case))
            if args.show_existing:
                summary_lines.append(
                    "  existing="
                    + json.dumps([asdict(spec) for spec in case.existing_specs], indent=2)
                )
                summary_lines.append("  new=" + json.dumps(asdict(case.new_spec), indent=2))
                summary_lines.append(
                    "  python_failure=" + json.dumps(case.python_failure, default=str)
                )
        payload = "\n".join(summary_lines) + "\n"

    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
