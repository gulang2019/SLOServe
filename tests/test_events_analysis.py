from motivation import events_analysis


def test_compute_window_series_counts_only_non_none_violations():
    time, num_reqs, num_violations, *_ = events_analysis._compute_window_series(
        [
            {"arrival_time": 0.1, "violation": "none", "ttft_normalized_laxity": 0.0, "max_tpot_laxity": 0.0, "prompt_tokens": 10, "output_tokens": 5, "total_tokens": 15, "kv_xfer_delay": 0.0},
            {"arrival_time": 0.2, "violation": "ttft", "ttft_normalized_laxity": 1.0, "max_tpot_laxity": 0.0, "prompt_tokens": 12, "output_tokens": 4, "total_tokens": 16, "kv_xfer_delay": 0.0},
            {"arrival_time": 1.2, "violation": "none", "ttft_normalized_laxity": 0.0, "max_tpot_laxity": 0.0, "prompt_tokens": 8, "output_tokens": 3, "total_tokens": 11, "kv_xfer_delay": 0.0},
        ],
        window_size=1.0,
    )

    assert time.tolist() == [0.0]
    assert num_reqs.tolist() == [2]
    assert num_violations.tolist() == [1]


def test_compute_measured_power_series_bins_real_energy_events():
    summary = events_analysis._compute_measured_power_series(
        [
            events_analysis.Energy(event_type="energy", timestamp=0.2, device_id=0, energy=5.0, power=50.0, mhz=1000.0),
            events_analysis.Energy(event_type="energy", timestamp=0.7, device_id=1, energy=3.0, power=30.0, mhz=1000.0),
            events_analysis.Energy(event_type="energy", timestamp=1.2, device_id=0, energy=7.0, power=70.0, mhz=1000.0),
        ],
        n_device=2,
        window_size=1.0,
        start_time=0.0,
        end_time=2.0,
    )

    assert summary["source"] == "measured"
    assert summary["time"].tolist() == [0.0, 1.0]
    assert summary["total_power"].tolist() == [8.0, 7.0]
    assert summary["per_device_power"][0].tolist() == [5.0, 7.0]
    assert summary["per_device_power"][1].tolist() == [3.0, 0.0]
    assert summary["total_energy_joules"] == 15.0


def test_compute_avg_batch_tokens_series_averages_batches_per_window():
    time, avg_tokens = events_analysis._compute_avg_batch_tokens_series(
        [
            events_analysis.Batch(
                event_type="batch",
                timestamp=0.5,
                device_id=0,
                batch_id=1,
                req_ids=["r0"],
                num_computed_tokens=[0],
                num_scheduled_tokens={"r0": 10},
                elapsed=0.2,
                scheduling_overhead=0.0,
            ),
            events_analysis.Batch(
                event_type="batch",
                timestamp=1.2,
                device_id=0,
                batch_id=2,
                req_ids=["r1"],
                num_computed_tokens=[0],
                num_scheduled_tokens={"r1": 20},
                elapsed=0.2,
                scheduling_overhead=0.0,
            ),
            events_analysis.Batch(
                event_type="batch",
                timestamp=1.8,
                device_id=1,
                batch_id=3,
                req_ids=["r2"],
                num_computed_tokens=[0],
                num_scheduled_tokens={"r2": 40},
                elapsed=0.2,
                scheduling_overhead=0.0,
            ),
        ],
        window_size=1.0,
        start_time=0.0,
        end_time=3.0,
    )

    assert time.tolist() == [0.0, 1.0, 2.0]
    assert avg_tokens.tolist() == [10.0, 30.0, 0.0]
