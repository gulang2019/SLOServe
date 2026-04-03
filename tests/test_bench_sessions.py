import json
from types import SimpleNamespace

import numpy as np
import pytest

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


def test_build_frontend_httpx_limits_normalizes_values():
    limits = bench_api_server._build_frontend_httpx_limits(
        max_connections=0,
        max_keepalive_connections=8,
    )

    assert limits is not None
    assert limits.max_connections == 1
    assert limits.max_keepalive_connections == 1


def test_frontend_httpx_suffix_reflects_selected_debug_mode():
    suffix = bench_api_server._frontend_httpx_suffix(
        dedicated_client_per_request=True,
        max_connections=256,
        max_keepalive_connections=128,
    )

    assert suffix == "_fhx_dedicated_mc256_mk128"


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
    assert payload["vllm_xargs"]["service_tier"] == "default"
    assert payload["vllm_xargs"]["initial_service_tier"] == "default"


def test_build_request_payload_marks_best_effort_when_forced():
    payload = bench_api_server._build_request_payload(
        model_name="test-model",
        prompt=[1, 2, 3],
        input_length=12,
        output_length=4,
        zero_load_ttft=0.8,
        cached_tokens=0,
        session_id=None,
        ttft_slo=2.1,
        slo_tpot=0.05,
        expected_profit=0.0,
        request_id="req-best-effort",
        must_admit=True,
        service_tier="best_effort",
    )

    assert payload["vllm_xargs"]["must_admit"] is True
    assert payload["vllm_xargs"]["service_tier"] == "best_effort"
    assert payload["vllm_xargs"]["initial_service_tier"] == "best_effort"


def test_make_best_effort_request_drops_cache_and_session():
    template = Request(
        input_length=8,
        output_length=4,
        cached_length=3,
        session_id="session-1",
        prompt="ignored",
    )

    request = bench_api_server._make_best_effort_request(template, seed=7)

    assert request.cached_length == 0
    assert request.session_id is None
    assert isinstance(request.prompt, list)
    assert len(request.prompt) == 8


def test_summarize_service_tier_metrics_splits_normal_and_best_effort():
    class _FakeReq:
        def __init__(
            self,
            *,
            tier,
            violation,
            arrival_time,
            finish_time,
            schedules,
        ):
            self.initial_service_tier = tier
            self.service_tier = tier
            self.finish_reason = "length"
            self.arrival_time = arrival_time
            self.schedules = schedules
            self.events = [SimpleNamespace(event_type="finish", timestamp=finish_time)]
            self._violation = violation

        def violate_slo(self):
            return self._violation

    reqs = {
        "regular-ok": _FakeReq(
            tier="default",
            violation="none",
            arrival_time=0.0,
            finish_time=2.0,
            schedules=[SimpleNamespace(timestamp=1.0, elapsed=0.2, num_scheduled_tokens=6)],
        ),
        "regular-bad": _FakeReq(
            tier="default",
            violation="tpot",
            arrival_time=1.0,
            finish_time=3.0,
            schedules=[SimpleNamespace(timestamp=2.0, elapsed=0.2, num_scheduled_tokens=5)],
        ),
        "best-effort": _FakeReq(
            tier="best_effort",
            violation="none",
            arrival_time=0.5,
            finish_time=2.5,
            schedules=[
                SimpleNamespace(timestamp=1.5, elapsed=0.1, num_scheduled_tokens=4),
                SimpleNamespace(timestamp=2.5, elapsed=0.1, num_scheduled_tokens=6),
            ],
        ),
    }

    summary = bench_api_server._summarize_service_tier_metrics(reqs, [])

    assert summary["normal_total_requests"] == 2
    assert summary["normal_slo_attainment_rate"] == 0.5
    assert summary["best_effort_total_requests"] == 1
    assert summary["best_effort_completed_requests"] == 1
    assert summary["best_effort_scheduled_tokens_in_normal_window"] == 10
    assert summary["best_effort_scheduled_token_throughput_tps"] == 10.0 / 3.0


def test_apply_service_tier_reporting_rewrites_top_level_slo_metrics():
    results = {
        "slo_attainment_rate": 0.25,
        "slo_violation_rate": 0.75,
        "violations": {"none": 0.25, "tpot": 0.75},
    }

    updated = bench_api_server._apply_service_tier_reporting(
        results,
        {
            "best_effort_total_requests": 3,
            "normal_slo_attainment_rate": 0.8,
            "normal_slo_violation_rate": 0.2,
            "normal_violation_reason_rates": {"none": 0.8, "tpot": 0.2},
        },
    )

    assert updated["all_request_slo_attainment_rate"] == 0.25
    assert updated["all_request_slo_violation_rate"] == 0.75
    assert updated["slo_attainment_rate"] == 0.8
    assert updated["slo_violation_rate"] == 0.2
    assert updated["violations"] == {"none": 0.8, "tpot": 0.2}


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


def test_summarize_rejection_breakdown_reports_counts_rates_and_shares():
    summary = bench_api_server._summarize_rejection_breakdown(
        bench_api_server.Counter({"compute": 3, "memory": 1, "router": 2}),
        total_requests=12,
    )

    assert summary["counts"] == {"compute": 3, "memory": 1, "router": 2}
    assert summary["rates"]["compute"] == pytest.approx(3 / 12)
    assert summary["rates"]["memory"] == pytest.approx(1 / 12)
    assert summary["shares"]["compute"] == pytest.approx(3 / 6)
    assert summary["shares"]["memory"] == pytest.approx(1 / 6)
    assert summary["compute_rejection_count"] == 3
    assert summary["memory_rejection_count"] == 1
    assert summary["compute_rejection_share"] == pytest.approx(3 / 6)
    assert summary["memory_rejection_share"] == pytest.approx(1 / 6)


def test_summarize_per_server_memory_metrics_uses_batch_snapshots():
    summary = bench_api_server._summarize_per_server_memory_metrics(
        [
            {
                "event_type": "batch",
                "device_id": 0,
                "used_kv_memory_bytes": 40.0,
                "effective_used_kv_memory_bytes": 30.0,
                "total_kv_memory_bytes": 100,
                "bytes_per_block": 10,
                "total_blocks": 10,
            },
            {
                "event_type": "batch",
                "device_id": 0,
                "used_kv_memory_bytes": 60.0,
                "effective_used_kv_memory_bytes": 50.0,
                "total_kv_memory_bytes": 100,
                "bytes_per_block": 10,
                "total_blocks": 10,
            },
            {
                "event_type": "batch",
                "device_id": 1,
                "used_kv_memory_bytes": 20.0,
                "effective_used_kv_memory_bytes": 10.0,
                "total_kv_memory_bytes": 80,
                "bytes_per_block": 8,
                "total_blocks": 10,
            },
        ],
        n_devices=2,
    )

    assert summary["per_server_average_used_memory_bytes"] == [50.0, 20.0]
    assert summary["per_server_peak_used_memory_bytes"] == [60.0, 20.0]
    assert summary["per_server_average_memory_utilization"] == [0.5, 0.25]
    assert summary["per_server_peak_memory_utilization"] == [0.6, 0.25]
    assert summary["per_server_average_effective_used_memory_bytes"] == [40.0, 10.0]
    assert summary["per_server_peak_effective_used_memory_bytes"] == [50.0, 10.0]
    assert summary["per_server_memory"][0]["total_memory_bytes"] == 100
    assert summary["per_server_memory"][1]["bytes_per_block"] == 8


def test_summarize_memory_occupancy_replay_uses_token_snapshots():
    summary = bench_api_server._summarize_memory_occupancy_replay(
        [
            {
                "event_type": "batch",
                "device_id": 0,
                "timestamp": 11.0,
                "used_kv_tokens": 32,
                "effective_used_kv_tokens": 16,
                "total_kv_tokens": 128,
            },
            {
                "event_type": "batch",
                "device_id": 0,
                "timestamp": 13.0,
                "used_kv_tokens": 64,
                "effective_used_kv_tokens": 48,
                "total_kv_tokens": 128,
            },
            {
                "event_type": "batch",
                "device_id": 1,
                "timestamp": 12.0,
                "used_kv_tokens": 24,
                "effective_used_kv_tokens": 24,
                "total_kv_tokens": 96,
            },
        ],
        n_devices=2,
        start_time=10.0,
        end_time=20.0,
    )

    assert summary["has_effective"] is True
    assert summary["devices"][0]["max_tokens"] == 128
    assert summary["devices"][1]["max_tokens"] == 96
    assert summary["devices"][0]["samples"][0] == {
        "time": 1.0,
        "used_tokens": 32.0,
        "effective_used_tokens": 16.0,
    }


def test_save_benchmark_figures_from_result_row_includes_memory_plot(tmp_path):
    output_prefix = tmp_path / "bench"

    figure_paths = bench_api_server.save_benchmark_figures_from_result_row(
        {
            "scheduling_policy": "atfc",
            "routing_policy": "slosserve_planner",
            "benchmark_figure_replay": {
                "window_time_pct_vs_active_requests": {"points": []},
                "power_vs_active_servers": {"stats": []},
                "power_vs_batch_tokens": {"stats": []},
                "memory_occupancy_over_time": {
                    "devices": [
                        {
                            "device_id": 0,
                            "max_tokens": 128,
                            "samples": [
                                {
                                    "time": 0.0,
                                    "used_tokens": 16.0,
                                    "effective_used_tokens": 8.0,
                                },
                                {
                                    "time": 1.0,
                                    "used_tokens": 32.0,
                                    "effective_used_tokens": 24.0,
                                },
                            ],
                        }
                    ],
                    "has_effective": True,
                },
            },
        },
        output_prefix,
    )

    assert figure_paths["memory_occupancy_over_time_figure"].endswith(
        ".memory_occupancy_over_time.png"
    )
    assert (tmp_path / "bench.memory_occupancy_over_time.png").exists()
    assert (tmp_path / "bench.memory_occupancy_over_time.pdf").exists()


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


def test_resolve_policy_admission_output_length_mode_from_suffix():
    scheduling_policy, routing_policy, mode = (
        bench_api_server._resolve_policy_admission_output_length_mode(
            "atfc_mem_p90",
            "slosserve",
            "max",
        )
    )

    assert scheduling_policy == "atfc"
    assert routing_policy == "slosserve"
    assert mode == "p90"


def test_resolve_policy_admission_output_length_mode_rejects_conflicts():
    with pytest.raises(ValueError):
        bench_api_server._resolve_policy_admission_output_length_mode(
            "atfc_mem_p80",
            "slosserve_mem_p90",
            "max",
        )


def test_build_problems_parses_mem_suffixes_from_policy_names(
    monkeypatch,
    tmp_path,
):
    requests = [
        Request(input_length=16, output_length=8),
        Request(input_length=24, output_length=16),
        Request(input_length=32, output_length=64),
    ]

    monkeypatch.setattr(
        bench_api_server,
        "_load_trace_inputs",
        lambda *args, **kwargs: (requests, [0.0, 1.0, 2.0], [("trace", "trace")]),
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
        scheduling_policy="atfc_mem_p90",
        routing_policy="slosserve_mem_p90",
        n_device=4,
        tensor_parallel_size=1,
        window="0:3",
        load_scale=1.0,
        experiment_dir=str(tmp_path),
        admission_output_length_mode="max",
    )

    assert len(problems) == 1
    problem = problems[0]
    expected_cap = bench_api_server._select_admission_output_length(
        np.asarray([request.output_length for request in requests], dtype=np.float64),
        "p90",
    )

    assert problem.scheduling_policy == "atfc"
    assert problem.routing_policy == "slosserve"
    assert problem.admission_output_length_mode == "p90"
    assert problem.admission_output_length == expected_cap
    assert "/atfc_p90_slosserve_p90_" in problem.store_prefix


def test_build_problems_propagates_frontend_httpx_settings(
    monkeypatch,
    tmp_path,
):
    requests = [
        Request(input_length=16, output_length=8),
        Request(input_length=24, output_length=16),
    ]

    monkeypatch.setattr(
        bench_api_server,
        "_load_trace_inputs",
        lambda *args, **kwargs: (requests, [0.0, 1.0], [("trace", "trace")]),
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
        window="0:2",
        load_scale=1.0,
        experiment_dir=str(tmp_path),
        frontend_httpx_max_connections=512,
        frontend_httpx_max_keepalive_connections=256,
        frontend_dedicated_client_per_request=True,
    )

    assert len(problems) == 1
    problem = problems[0]
    assert problem.frontend_httpx_max_connections == 512
    assert problem.frontend_httpx_max_keepalive_connections == 256
    assert problem.frontend_dedicated_client_per_request is True
    assert problem.store_prefix.endswith("_fhx_dedicated_mc512_mk256")
