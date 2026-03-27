import copy
import os
from dataclasses import dataclass

import pytest

pytest.importorskip("SLOsServe_C")

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


def _make_planner(
    request_specs: list[RequestSpec],
    *,
    now: float = 0.0,
    hardware_params: list[float] | None = None,
    max_batch_size: int = 4096,
    max_decode_length: int = 64,
    is_oracle: bool = True,
) -> tuple[adm_ctrl.BatchPlanner, dict[str, float]]:
    if hardware_params is None:
        # This mirrors the current control-side constants after the
        # scheduling-overhead adjustment.
        hardware_params = [6.1e-5, 1.6e-5, 3.5e-7, 0.0, 0.018]

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


def _format_specs(existing_specs: list[RequestSpec], new_spec: RequestSpec) -> dict:
    def _fmt(spec: RequestSpec) -> dict:
        next_load = (
            max(spec.num_prompt_tokens - spec.num_computed_tokens, 1)
            if not spec.prefill_only
            else (spec.num_prompt_tokens - spec.num_computed_tokens)
        )
        return {
            "request_id": spec.request_id,
            "prompt": spec.num_prompt_tokens,
            "computed": spec.num_computed_tokens,
            "prefill_ddl": spec.prefill_ddl,
            "next_load": next_load,
            "output_length": spec.output_length,
            "service_tier": spec.service_tier,
        }

    return {
        "existing": [_fmt(spec) for spec in existing_specs],
        "new": _fmt(new_spec),
    }


def _cpp_admission_with_new(
    existing_specs: list[RequestSpec],
    new_spec: RequestSpec,
    *,
    now: float = 0.0,
    hardware_params: list[float] | None = None,
) -> tuple[bool, str | None]:
    planner, _ = _make_planner(
        copy.deepcopy(existing_specs),
        now=now,
        hardware_params=hardware_params,
        is_oracle=True,
    )
    return planner._cpp_feasible_with_new(_make_request(new_spec), now)


def _shadow_refresh_fast_feasible(
    existing_specs: list[RequestSpec],
    new_spec: RequestSpec,
    *,
    now: float = 0.0,
    hardware_params: list[float] | None = None,
) -> tuple[bool, tuple | None]:
    planner, now_box = _make_planner(
        copy.deepcopy(existing_specs) + [copy.deepcopy(new_spec)],
        now=now,
        hardware_params=hardware_params,
        is_oracle=True,
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


def _assert_admission_parity(
    existing_specs: list[RequestSpec],
    new_spec: RequestSpec,
) -> None:
    cpp_result = _cpp_admission_with_new(existing_specs, new_spec)
    py_result = _shadow_refresh_fast_feasible(existing_specs, new_spec)
    assert cpp_result[0] == py_result[0], (
        f"cpp={cpp_result}, python={py_result}, "
        f"specs={_format_specs(existing_specs, new_spec)}"
    )


def _find_first_admission_mismatch(
    *,
    num_trials: int = 500,
    seed: int = 0,
) -> tuple[
    list[RequestSpec],
    RequestSpec,
    tuple[bool, str | None],
    tuple[bool, tuple | None],
] | None:
    import random

    prompt_choices = [64, 128, 256, 512, 768, 1024, 1536]
    output_choices = [4, 8, 16]
    random.seed(seed)
    for _trial in range(num_trials):
        existing_specs: list[RequestSpec] = []
        for idx in range(random.randint(0, 5)):
            prompt = random.choice(prompt_choices)
            computed = random.choice(
                [0, 0, random.randint(1, min(prompt, 64)), prompt]
            )
            ddl = random.uniform(0.03, 0.4 if computed >= prompt else 0.8)
            existing_specs.append(
                RequestSpec(
                    request_id=f"e{idx}",
                    num_prompt_tokens=prompt,
                    num_computed_tokens=computed,
                    prefill_ddl=ddl,
                    output_length=random.choice(output_choices),
                )
            )
        prompt = random.choice(prompt_choices)
        computed = random.choice([0, 0, random.randint(1, min(prompt, 64)), prompt])
        ddl = random.uniform(0.03, 0.4 if computed >= prompt else 0.8)
        new_spec = RequestSpec(
            request_id="new",
            num_prompt_tokens=prompt,
            num_computed_tokens=computed,
            prefill_ddl=ddl,
            output_length=random.choice(output_choices),
        )
        cpp_result = _cpp_admission_with_new(existing_specs, new_spec)
        py_result = _shadow_refresh_fast_feasible(existing_specs, new_spec)
        if cpp_result[0] != py_result[0]:
            return existing_specs, new_spec, cpp_result, py_result
    return None


@pytest.mark.parametrize(
    ("name", "existing_specs", "new_spec"),
    [
        (
            "single_prefill_accept",
            [],
            RequestSpec(
                request_id="new",
                num_prompt_tokens=512,
                num_computed_tokens=0,
                prefill_ddl=0.25,
                prefill_only=True,
                output_length=0,
            ),
        ),
        (
            "short_decode_mix_accept",
            [
                RequestSpec(
                    request_id="e0",
                    num_prompt_tokens=128,
                    num_computed_tokens=128,
                    prefill_ddl=0.05,
                    output_length=4,
                )
            ],
            RequestSpec(
                request_id="new",
                num_prompt_tokens=256,
                num_computed_tokens=0,
                prefill_ddl=0.18,
                output_length=8,
            ),
        ),
    ],
)
def test_cpp_admission_python_shadow_sanity_cases(
    name: str,
    existing_specs: list[RequestSpec],
    new_spec: RequestSpec,
):
    del name
    _assert_admission_parity(existing_specs, new_spec)


def test_cpp_admission_python_shadow_known_optimistic_mismatch():
    existing_specs = [
        RequestSpec(
            request_id="e0",
            num_prompt_tokens=512,
            num_computed_tokens=0,
            prefill_ddl=0.05522339660602958,
            output_length=8,
        )
    ]
    new_spec = RequestSpec(
        request_id="new",
        num_prompt_tokens=512,
        num_computed_tokens=0,
        prefill_ddl=0.13160199899446032,
        output_length=8,
    )
    _assert_admission_parity(existing_specs, new_spec)


@pytest.mark.skipif(
    os.getenv("RUN_CPP_ADMISSION_PARITY_FUZZ") != "1",
    reason="Set RUN_CPP_ADMISSION_PARITY_FUZZ=1 to run the admission fuzz stub.",
)
def test_cpp_admission_python_shadow_seeded_fuzz_stub():
    mismatch = _find_first_admission_mismatch(num_trials=200, seed=0)
    assert mismatch is None, (
        "C admission / Python shadow mismatch found: "
        f"specs={_format_specs(mismatch[0], mismatch[1])}, "
        f"cpp={mismatch[2]}, python={mismatch[3]}"
    )
