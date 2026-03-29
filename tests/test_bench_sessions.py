import json
from types import SimpleNamespace

import numpy as np

from Dataset.dataset import Request
from motivation import bench_api_server


def test_compute_ttft_slo_uses_only_new_tokens():
    slo_ttft = bench_api_server.compute_ttft_slo(
        prompt_tokens=12,
        cached_tokens=5,
        slo_ttft_per_token=2.0,
        slo_ttft_constant=1.5,
        slo_routing_overhead=0.25,
    )

    assert slo_ttft == 15.75


def test_build_request_payload_includes_session_metadata():
    payload = bench_api_server._build_request_payload(
        model_name="test-model",
        prompt=[1, 2, 3],
        input_length=12,
        output_length=4,
        zero_load_ttft=0.8,
        cached_tokens=5,
        session_id="session-1",
        ttft_slo=2.1,
        slo_tpot=0.05,
        expected_profit=3.0,
        request_id="req-1",
    )

    assert payload["vllm_xargs"]["cached_tokens"] == 5
    assert payload["vllm_xargs"]["session_id"] == "session-1"
    assert payload["vllm_xargs"]["slo_ttft"] == 2.1


def test_split_ready_request_indices_respects_session_gate():
    requests = [
        Request(input_length=4, output_length=2, session_id="s1"),
        Request(input_length=4, output_length=2, session_id="s1"),
        Request(input_length=4, output_length=2, session_id="s2"),
    ]

    ready, blocked = bench_api_server._split_ready_request_indices(
        pending_indices=[0, 1, 2],
        requests=requests,
        elapsed_time=5.0,
        session_ready_at={"s1": float("inf"), "s2": 0.0},
        enable_session_replay=True,
    )

    assert ready == [2]
    assert blocked == [0, 1]


def test_split_ready_request_indices_is_noop_when_replay_disabled():
    requests = [
        Request(input_length=4, output_length=2, session_id="s1"),
        Request(input_length=4, output_length=2, session_id="s1"),
    ]

    ready, blocked = bench_api_server._split_ready_request_indices(
        pending_indices=[0, 1],
        requests=requests,
        elapsed_time=0.0,
        session_ready_at={"s1": float("inf")},
        enable_session_replay=False,
    )

    assert ready == [0, 1]
    assert blocked == []


def test_ensure_prompts_present_clears_mixed_prompt_sources():
    requests = [
        Request(input_length=4, output_length=2, prompt="real prompt"),
        Request(input_length=4, output_length=2, prompt=None),
        Request(input_length=2, output_length=1, prompt="other prompt"),
    ]

    bench_api_server.ensure_prompts_present(requests, model_name="test-model")

    for request in requests:
        assert isinstance(request.prompt, list)
        assert len(request.prompt) == request.input_length
        assert all(isinstance(token, int) for token in request.prompt)


def test_ensure_prompts_present_normalizes_existing_strings_to_token_ids():
    requests = [
        Request(input_length=4, output_length=2, prompt="prompt a"),
        Request(input_length=2, output_length=1, prompt="prompt b"),
    ]

    bench_api_server.ensure_prompts_present(requests, model_name="test-model")

    for request in requests:
        assert isinstance(request.prompt, list)
        assert len(request.prompt) == request.input_length
        assert all(isinstance(token, int) for token in request.prompt)


def test_generate_session_replay_prompts_preserves_cached_prefix():
    requests = [
        Request(
            input_length=5,
            output_length=2,
            cached_length=0,
            session_id="session-1",
            prompt="first prompt",
        ),
        Request(
            input_length=8,
            output_length=2,
            cached_length=5,
            session_id="session-1",
            prompt="second prompt",
        ),
    ]

    bench_api_server.generate_session_replay_prompts(requests, seed=123)

    assert isinstance(requests[0].prompt, list)
    assert isinstance(requests[1].prompt, list)
    assert len(requests[0].prompt) == 5
    assert len(requests[1].prompt) == 8
    assert requests[1].prompt[:5] == requests[0].prompt[:5]


def test_normalize_rejection_reason_maps_scheduler_codes():
    assert bench_api_server._normalize_rejection_reason("CMP") == "compute"
    assert bench_api_server._normalize_rejection_reason("MEM") == "memory"
    assert bench_api_server._normalize_rejection_reason("OOM") == "oom"
    assert bench_api_server._normalize_rejection_reason("router") == "router"
    assert bench_api_server._normalize_rejection_reason("UNKNOWN") == "unknown"
    assert bench_api_server._normalize_rejection_reason(None) is None


def test_make_overload_run_result_records_terminal_overload(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bench_api_server.ArrivalTimes,
        "load",
        lambda *args, **kwargs: SimpleNamespace(arrival_times=[0.0, 2.0, 4.0]),
    )

    problem = bench_api_server.Problem(
        arrival_pattern="arrival-trace",
        length_pattern="length-trace",
        window="0:3",
        n_devices=2,
        store_prefix=str(tmp_path / "bench"),
    )

    result = bench_api_server._make_overload_run_result(
        problem,
        requested_n_devices=4,
        error=RuntimeError("engine died"),
    )

    assert result.results["run_status"] == "overloaded"
    assert result.results["overloaded"] is True
    assert result.results["requested_n_device"] == 4
    assert result.results["effective_n_device"] == 2
    assert result.results["rr_sliced"] is True
    assert result.results["rps"] == 0.75

    event_path = tmp_path / "bench.overloaded.events.jsonl"
    assert event_path.exists()
    payload = json.loads(event_path.read_text())
    assert payload[0]["status"] == "overloaded"


def test_select_admission_output_length_supports_percentiles():
    output_lengths = np.asarray([8, 16, 32, 64], dtype=np.float64)

    assert bench_api_server._select_admission_output_length(
        output_lengths,
        "max",
    ) == 64
    assert bench_api_server._select_admission_output_length(
        output_lengths,
        "p80",
    ) == 45
    assert bench_api_server._select_admission_output_length(
        output_lengths,
        "p95",
    ) == 60


def test_build_problems_uses_configured_admission_output_length(
    monkeypatch,
    tmp_path,
):
    requests = [
        Request(input_length=16, output_length=8),
        Request(input_length=24, output_length=16),
        Request(input_length=32, output_length=32),
        Request(input_length=40, output_length=64),
    ]

    monkeypatch.setattr(
        bench_api_server,
        "_load_trace_inputs",
        lambda *args, **kwargs: (requests, [0.0, 1.0, 2.0, 3.0], [("trace", "trace")]),
    )

    class _FakePerfModel:
        is_piecewise_current_tokens = False

        def describe_hardware_params(self):
            return {}

        def get_max_decode_batch_size(self, slo_tpot, average_input_length=0.0):
            return 16

        def get_batch_time(self, batch):
            return 0.01

        def get_zero_load_prefill_affine_params(self):
            return 0.001, 0.01

    monkeypatch.setattr(
        bench_api_server.PerfModel,
        "get_perf_model",
        staticmethod(lambda *args, **kwargs: _FakePerfModel()),
    )

    problems = bench_api_server.build_problems(
        model_name="test-model",
        trace="trace",
        ttft_slo_scale=1.0,
        slo_tpot=0.05,
        profit="constant",
        scheduling_policy="atfc",
        routing_policy="slosserve",
        n_device=4,
        tensor_parallel_size=1,
        window="0:4",
        load_scale=1.0,
        experiment_dir=str(tmp_path),
        admission_output_length_mode="p80",
    )

    assert len(problems) == 1
    problem = problems[0]
    expected_cap = bench_api_server._select_admission_output_length(
        np.asarray([request.output_length for request in requests], dtype=np.float64),
        "p80",
    )

    assert problem.admission_output_length_mode == "p80"
    assert problem.admission_output_length == expected_cap
    assert problem.scheduling_kwargs["max_decoding_length"] == 64
    assert (
        problem.scheduling_kwargs["admission_max_decoding_length"] == expected_cap
    )
    assert problem.routing_kwargs["max_decode_length"] == 64
    assert (
        problem.routing_kwargs["admission_max_decode_length"] == expected_cap
    )
    assert "/atfc_p80_slosserve_p80_" in problem.store_prefix
