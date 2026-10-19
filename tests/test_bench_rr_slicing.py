from types import SimpleNamespace

from motivation import bench_api_server


def test_resolve_rr_effective_n_devices_clamps_to_available_clients():
    effective, sliced = bench_api_server._resolve_rr_effective_n_devices(
        requested_n_devices=8,
        routing_policy="round_robin",
        routing_kwargs={},
        clients_arg="0-3",
    )

    assert effective == 4
    assert sliced is True


def test_resolve_rr_effective_n_devices_skips_pd_disagg():
    effective, sliced = bench_api_server._resolve_rr_effective_n_devices(
        requested_n_devices=8,
        routing_policy="round_robin",
        routing_kwargs={"is_pd_disagg": True},
        clients_arg="r0,r1,r2,r3",
    )

    assert effective == 8
    assert sliced is False


def test_slice_rr_workload_keeps_prefix_shards_with_original_timestamps():
    requests = list(range(12))
    arrival_times = [i * 0.5 for i in range(12)]

    kept_requests, kept_arrivals, kept_indices = bench_api_server._slice_rr_workload(
        requests,
        arrival_times,
        requested_n_devices=6,
        effective_n_devices=2,
    )

    assert kept_indices == [0, 1, 6, 7]
    assert kept_requests == [0, 1, 6, 7]
    assert kept_arrivals == [0.0, 0.5, 3.0, 3.5]


def test_apply_trace_arrival_limit_keeps_prefix_within_relative_cutoff():
    requests = ["r0", "r1", "r2", "r3"]
    arrival_times = [10.0, 10.5, 12.0, 12.1]

    kept_requests, kept_arrivals = bench_api_server._apply_trace_arrival_limit(
        requests,
        arrival_times,
        trace_arrival_limit_s=2.0,
    )

    assert kept_requests == ["r0", "r1", "r2"]
    assert kept_arrivals == [10.0, 10.5, 12.0]


def test_apply_trace_arrival_limit_is_noop_when_unset():
    requests = ["r0", "r1"]
    arrival_times = [3.0, 7.0]

    kept_requests, kept_arrivals = bench_api_server._apply_trace_arrival_limit(
        requests,
        arrival_times,
        trace_arrival_limit_s=None,
    )

    assert kept_requests == requests
    assert kept_arrivals == arrival_times


def test_rr_slice_energy_multiplier_scales_requested_over_effective():
    multiplier = bench_api_server._rr_slice_energy_multiplier(
        requested_n_devices=8,
        effective_n_devices=4,
        rr_sliced=True,
    )

    assert multiplier == 2.0


def test_scale_rr_energy_results_updates_total_and_estimate():
    results = {
        "extra_metrics": {
            "energy_est": 12.5,
            "energy_all": 14.0,
            "average_power_active_w": 5.0,
            "average_n_active_servers": 1.5,
        }
    }

    scaled_results, scaled_energy = bench_api_server._scale_rr_energy_results(
        results=results,
        energy_consumption=20.0,
        requested_n_devices=6,
        effective_n_devices=2,
        rr_sliced=True,
    )

    assert scaled_energy == 60.0
    assert scaled_results["extra_metrics"]["energy_est"] == 37.5
    assert scaled_results["extra_metrics"]["energy_all"] == 42.0
    assert scaled_results["extra_metrics"]["average_power_active_w"] == 15.0
    assert scaled_results["extra_metrics"]["average_n_active_servers"] == 1.5
    assert results["extra_metrics"]["energy_est"] == 12.5


def test_summarize_per_server_energy_metrics_aggregates_energy_and_power():
    metrics = bench_api_server._summarize_per_server_energy_metrics(
        [
            {"event_type": "energy", "device_id": 0, "energy": 10.0, "power": 100.0},
            {"event_type": "energy", "device_id": 0, "energy": 20.0, "power": 200.0},
            SimpleNamespace(event_type="energy", device_id=1, energy=5.0, power=50.0),
            {"event_type": "batch", "device_id": 0, "energy": 999.0, "power": 999.0},
        ],
        n_devices=3,
    )

    assert metrics["per_server_energy_consumption"] == [30.0, 5.0, 0.0]
    assert metrics["per_server_power"] == [150.0, 50.0, 0.0]


def test_summarize_per_device_active_idle_power_splits_batch_active_windows():
    active_windows = bench_api_server._compute_per_device_active_windows(
        [
            SimpleNamespace(
                event_type="batch",
                timestamp=30.5,
                elapsed=0.1,
                device_id=0,
            ),
            SimpleNamespace(
                event_type="batch",
                timestamp=31.5,
                elapsed=0.1,
                device_id=1,
            ),
        ],
        n_devices=2,
        start_time=30.0,
        end_time=32.0,
        window_size=1.0,
    )
    power_summary = {
        "per_device_power": {
            0: [100.0, 20.0],
            1: [30.0, 200.0],
        },
    }

    metrics = bench_api_server._summarize_per_device_active_idle_power(
        power_summary,
        active_windows,
        n_devices=2,
        window_size=1.0,
        configured_idle_power_w=70.0,
    )

    assert metrics["per_device_active_power_w"] == [100.0, 200.0]
    assert metrics["per_device_idle_power_w"] == [20.0, 30.0]
    assert metrics["per_device_active_minus_idle_power_w"] == [80.0, 170.0]
    assert metrics["per_device_active_duration_s"] == [1.0, 1.0]
    assert metrics["per_device_idle_duration_s"] == [1.0, 1.0]
    assert metrics["per_device_active_idle_power"] == [
        {
            "device_id": 0,
            "active_power_w": 100.0,
            "idle_power_w": 20.0,
            "active_minus_idle_power_w": 80.0,
            "configured_idle_power_w": 70.0,
            "active_energy_j": 100.0,
            "idle_energy_j": 20.0,
            "active_duration_s": 1.0,
            "idle_duration_s": 1.0,
            "active_window_count": 1,
            "idle_window_count": 1,
            "active_sample_count": 1,
            "idle_sample_count": 1,
            "idle_power_source": "measured",
        },
        {
            "device_id": 1,
            "active_power_w": 200.0,
            "idle_power_w": 30.0,
            "active_minus_idle_power_w": 170.0,
            "configured_idle_power_w": 70.0,
            "active_energy_j": 200.0,
            "idle_energy_j": 30.0,
            "active_duration_s": 1.0,
            "idle_duration_s": 1.0,
            "active_window_count": 1,
            "idle_window_count": 1,
            "active_sample_count": 1,
            "idle_sample_count": 1,
            "idle_power_source": "measured",
        },
    ]


def test_summarize_energy_events_respects_stable_region_window():
    per_gpu, total = bench_api_server.summarize_energy_events(
        [
            {"event_type": "energy", "timestamp": 29.0, "device_id": 0, "energy": 1.0},
            {"event_type": "energy", "timestamp": 30.0, "device_id": 0, "energy": 2.0},
            {"event_type": "energy", "timestamp": 45.0, "device_id": 1, "energy": 3.0},
            {"event_type": "energy", "timestamp": 60.0, "device_id": 1, "energy": 4.0},
            {"event_type": "batch", "timestamp": 45.0, "device_id": 0, "energy": 99.0},
        ],
        n_devices=2,
        start_time=30.0,
        end_time=60.0,
    )

    assert per_gpu == [2.0, 3.0]
    assert total == 5.0


def test_summarize_per_server_energy_metrics_respects_stable_region_window():
    metrics = bench_api_server._summarize_per_server_energy_metrics(
        [
            {"event_type": "energy", "timestamp": 25.0, "device_id": 0, "energy": 9.0, "power": 90.0},
            {"event_type": "energy", "timestamp": 30.0, "device_id": 0, "energy": 10.0, "power": 100.0},
            SimpleNamespace(
                event_type="energy",
                timestamp=45.0,
                device_id=1,
                energy=20.0,
                power=200.0,
            ),
            {"event_type": "energy", "timestamp": 60.0, "device_id": 1, "energy": 30.0, "power": 300.0},
        ],
        n_devices=2,
        start_time=30.0,
        end_time=60.0,
    )

    assert metrics["per_server_energy_consumption"] == [10.0, 20.0]
    assert metrics["per_server_power"] == [100.0, 200.0]


def test_resolve_benchmark_window_bounds_uses_session_arrival_heads_when_given():
    start_time, end_time = bench_api_server._resolve_benchmark_window_bounds(
        events=[],
        reqs={
            "r0": SimpleNamespace(arrival_time=5.0, engine_arrival_time=8.0),
            "r1": SimpleNamespace(arrival_time=40.0, engine_arrival_time=75.0),
            "r2": SimpleNamespace(arrival_time=120.0, engine_arrival_time=120.0),
        },
        window_size=bench_api_server.BENCHMARK_METRIC_WINDOW_S,
        arrival_times=[8.0, 75.0],
    )

    assert start_time == bench_api_server.BENCHMARK_ENERGY_RUN_START_DELAY_S
    assert end_time == 105.0


def test_summarize_benchmark_energy_and_figures_reports_active_average_power(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        bench_api_server,
        "_write_json_artifact",
        lambda filename, payload: str(tmp_path / "memory.json"),
    )
    monkeypatch.setattr(
        bench_api_server,
        "save_benchmark_figures_from_result_row",
        lambda row, prefix: {},
    )

    reqs = {
        "r0": SimpleNamespace(
            arrival_time=30.0,
            engine_arrival_time=30.0,
            events=[SimpleNamespace(event_type="finish", timestamp=32.0)],
        )
    }
    events = [
        SimpleNamespace(
            event_type="energy",
            timestamp=30.0,
            device_id=0,
            energy=100.0,
            power=100.0,
            mhz=1200.0,
        ),
        SimpleNamespace(
            event_type="energy",
            timestamp=31.0,
            device_id=0,
            energy=120.0,
            power=120.0,
            mhz=1200.0,
        ),
        SimpleNamespace(
            event_type="energy",
            timestamp=32.0,
            device_id=0,
            energy=140.0,
            power=140.0,
            mhz=1200.0,
        ),
        SimpleNamespace(
            event_type="batch",
            timestamp=30.0,
            elapsed=0.5,
            device_id=0,
            num_scheduled_tokens={"r0": 8},
        ),
        SimpleNamespace(
            event_type="batch",
            timestamp=31.0,
            elapsed=0.5,
            device_id=0,
            num_scheduled_tokens={"r0": 8},
        ),
        SimpleNamespace(
            event_type="batch",
            timestamp=32.0,
            elapsed=0.5,
            device_id=0,
            num_scheduled_tokens={"r0": 8},
        ),
    ]

    metrics = bench_api_server._summarize_benchmark_energy_and_figures(
        events,
        reqs,
        n_devices=1,
        tensor_parallel_size=1,
        output_prefix=str(tmp_path / "bench"),
        scheduling_policy="sched",
        routing_policy="route",
        start_time=30.0,
        end_time=33.0,
        arrival_times=[30.0],
        window_size=1.0,
    )

    assert metrics["energy_consumption_active"] == 220.0
    assert metrics["benchmark_active_duration_seconds"] == 2.0
    assert metrics["average_power_active_w"] == 110.0
    assert metrics["per_device_active_power_w"] == [110.0]
    assert metrics["per_device_idle_power_w"] == [140.0]
    assert metrics["per_device_active_minus_idle_power_w"] == [-30.0]
    assert metrics["per_device_active_energy_j"] == [220.0]
    assert metrics["per_device_idle_energy_j"] == [140.0]
    assert metrics["per_device_active_idle_power_file"] == str(
        tmp_path / "bench.per_device_active_idle_power.csv"
    )

    csv_lines = (tmp_path / "bench.per_device_active_idle_power.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert csv_lines[0] == (
        "device_id,active_power_w,idle_power_w,active_minus_idle_power_w,"
        "configured_idle_power_w,active_energy_j,idle_energy_j,"
        "active_duration_s,idle_duration_s,active_window_count,"
        "idle_window_count,active_sample_count,idle_sample_count,"
        "idle_power_source"
    )
    assert csv_lines[1] == (
        "0,110.0,140.0,-30.0,70.0,220.0,140.0,2.0,1.0,2,1,2,1,"
        "measured"
    )


def test_summarize_per_server_rps_counts_dispatches_by_stage():
    per_server_rps = bench_api_server._summarize_per_server_rps(
        [
            {"event_type": "dispatch-both", "prefill_device_id": 0},
            {"event_type": "dispatch-prefill", "prefill_device_id": 1},
            SimpleNamespace(event_type="dispatch-decode", decode_device_id=2),
            {"event_type": "dispatch-decode", "decode_device_id": -1},
            {"event_type": "arrival-router", "prefill_device_id": 0},
        ],
        n_devices=3,
        duration_s=2.0,
    )

    assert per_server_rps == [0.5, 0.5, 0.5]


def test_summarize_rr_disagg_power_metrics_splits_prefill_and_decode_servers():
    metrics = bench_api_server._summarize_rr_disagg_power_metrics(
        n_devices=4,
        routing_policy="round_robin",
        routing_kwargs={
            "is_pd_disagg": True,
            "group_size": 2,
            "n_prefill_per_group": 1,
        },
        per_server_power=[10.0, 20.0, 30.0, 40.0],
    )

    assert metrics["per_prefill_power"] == 20.0
    assert metrics["per_decode_power"] == 30.0
    assert metrics["per_prefill_server_power"] == [10.0, 30.0]
    assert metrics["per_decode_server_power"] == [20.0, 40.0]
