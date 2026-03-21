from motivation import bench_api_server


def test_resolve_rr_effective_n_devices_clamps_to_available_clients():
    effective, sliced = bench_api_server._resolve_rr_effective_n_devices(
        requested_n_devices=8,
        routing_policy="round_robin",
        routing_kwargs={},
        clients_arg="r0,r1,r2,r3",
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
    assert scaled_results["extra_metrics"]["average_n_active_servers"] == 1.5
    assert results["extra_metrics"]["energy_est"] == 12.5
