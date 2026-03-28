import json
from types import SimpleNamespace

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
