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
    output_length: int = 64
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
) -> adm_ctrl.BatchPlanner:
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
        _is_oracle=False,
    )
    planner._now = lambda: now
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
    return planner


def _run_python_and_cpp(
    request_specs: list[RequestSpec],
    *,
    now: float = 0.0,
    hardware_params: list[float] | None = None,
) -> tuple[
    tuple[bool, dict[str, int]],
    tuple[bool, dict[str, int]],
]:
    planner = _make_planner(
        request_specs,
        now=now,
        hardware_params=hardware_params,
    )
    py_feasible, py_batches, _ = planner._refresh_fast()
    cpp_feasible, cpp_batches, _ = planner._refresh_c()
    py_first = py_batches[0].n_scheduled_tokens if py_batches else {}
    cpp_first = cpp_batches[0].n_scheduled_tokens if cpp_batches else {}
    return (py_feasible, py_first), (cpp_feasible, cpp_first)


def _format_specs(request_specs: list[RequestSpec]) -> list[dict]:
    return [
        {
            "request_id": spec.request_id,
            "prompt": spec.num_prompt_tokens,
            "computed": spec.num_computed_tokens,
            "prefill_ddl": spec.prefill_ddl,
            "next_load": max(spec.num_prompt_tokens - spec.num_computed_tokens, 1)
            if not spec.prefill_only
            else (spec.num_prompt_tokens - spec.num_computed_tokens),
            "service_tier": spec.service_tier,
        }
        for spec in request_specs
    ]


def _assert_parity(request_specs: list[RequestSpec]) -> None:
    py_result, cpp_result = _run_python_and_cpp(request_specs)
    assert py_result == cpp_result, (
        f"python={py_result}, cpp={cpp_result}, specs={_format_specs(request_specs)}"
    )


@pytest.mark.parametrize(
    ("name", "request_specs"),
    [
        (
            "single_prefill_request",
            [
                RequestSpec(
                    request_id="req-0",
                    num_prompt_tokens=512,
                    num_computed_tokens=0,
                    prefill_ddl=0.25,
                    prefill_only=True,
                    output_length=0,
                )
            ],
        ),
        (
            "single_decode_request",
            [
                RequestSpec(
                    request_id="req-0",
                    num_prompt_tokens=256,
                    num_computed_tokens=256,
                    prefill_ddl=0.1,
                    prefill_only=False,
                    output_length=64,
                )
            ],
        ),
    ],
)
def test_python_cpp_parity_sanity_cases(name: str, request_specs: list[RequestSpec]):
    del name
    _assert_parity(request_specs)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Regular-only state where Python _refresh_fast and C _refresh_c "
        "produce different first batches."
    ),
)
def test_python_cpp_parity_known_first_batch_mismatch():
    request_specs = [
        RequestSpec("r0", 512, 5, 0.703766),
        RequestSpec("r1", 768, 770, 0.281769),
        RequestSpec("r2", 64, 22, 0.097915),
        RequestSpec("r3", 1024, 21, 0.445680),
        RequestSpec("r4", 512, 0, 0.420937),
        RequestSpec("r5", 256, 0, 0.656784),
    ]
    _assert_parity(request_specs)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Regular-only state where Python _refresh_fast and C _refresh_c "
        "disagree on feasibility."
    ),
)
def test_python_cpp_parity_known_feasibility_mismatch():
    request_specs = [
        RequestSpec("r0", 1536, 58, 0.433399),
        RequestSpec("r1", 64, 65, 0.142170),
        RequestSpec("r2", 512, 512, 0.051366),
        RequestSpec("r3", 128, 128, 0.042024),
        RequestSpec("r4", 64, 64, 0.156453),
        RequestSpec("r5", 512, 0, 0.036530),
        RequestSpec("r6", 1536, 5, 0.770895),
    ]
    _assert_parity(request_specs)


def _find_first_regular_only_mismatch(
    *,
    num_trials: int = 500,
    seed: int = 0,
) -> tuple[list[RequestSpec], tuple[bool, dict[str, int]], tuple[bool, dict[str, int]]] | None:
    import random

    random.seed(seed)
    prompt_choices = [64, 128, 256, 512, 768, 1024, 1536]
    for _trial in range(num_trials):
        request_specs: list[RequestSpec] = []
        for idx in range(random.randint(2, 7)):
            prompt = random.choice(prompt_choices)
            computed = random.choice(
                [0, 0, 0, random.randint(1, min(prompt, 64)), prompt, prompt + 1]
            )
            if computed < prompt:
                ddl = random.uniform(0.03, 0.8)
            else:
                ddl = random.uniform(0.03, 0.2)
            request_specs.append(
                RequestSpec(
                    request_id=f"r{idx}",
                    num_prompt_tokens=prompt,
                    num_computed_tokens=computed,
                    prefill_ddl=ddl,
                    output_length=64,
                )
            )
        py_result, cpp_result = _run_python_and_cpp(request_specs)
        if py_result != cpp_result:
            return request_specs, py_result, cpp_result
    return None


@pytest.mark.skipif(
    os.getenv("RUN_CPP_PYTHON_PARITY_FUZZ") != "1",
    reason="Set RUN_CPP_PYTHON_PARITY_FUZZ=1 to run the parity fuzz stub.",
)
def test_python_cpp_parity_seeded_fuzz_stub():
    mismatch = _find_first_regular_only_mismatch(num_trials=200, seed=0)
    assert mismatch is None, (
        "Python/C parity mismatch found: "
        f"specs={_format_specs(mismatch[0])}, python={mismatch[1]}, cpp={mismatch[2]}"
    )
