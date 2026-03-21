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
