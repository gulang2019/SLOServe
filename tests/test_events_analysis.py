import json

from motivation import events_analysis
from plots.timeline_comparison import export_timeline_comparison_metadata


def _write_timeline_events_trace(tmp_path):
    events = [
        {
            "event_type": "global_arrival",
            "request_id": "r0",
            "timestamp": 0.2,
        },
        {
            "event_type": "arrival-router",
            "request_id": "r0",
            "timestamp": 0.2,
            "device_id": -1,
            "prefill_ddl": 3.25,
            "profit": 1.0,
            "prompt_tokens": 2,
            "max_tokens": 1,
            "num_cached_tokens": 0,
            "zero_load_ttft": 1.0,
        },
        {
            "event_type": "arrival",
            "request_id": "r0",
            "timestamp": 0.25,
            "device_id": 0,
            "prompt_tokens": 2,
            "max_tokens": 1,
            "num_cached_tokens": 0,
            "zero_load_ttft": 1.0,
            "prefill_ddl": 3.25,
            "profit": 1.0,
            "prefill_only": False,
            "decode_only": False,
            "add_req_time": 0.23,
        },
        {
            "event_type": "batch",
            "timestamp": 1.0,
            "device_id": 0,
            "batch_id": 1,
            "req_ids": ["r0"],
            "num_computed_tokens": [0],
            "num_scheduled_tokens": {"r0": 2},
            "elapsed": 0.1,
            "scheduling_overhead": 0.0,
        },
        {
            "event_type": "finish",
            "request_id": "r0",
            "timestamp": 1.05,
            "device_id": 0,
            "finish_reason": "length",
        },
        {
            "event_type": "global_arrival",
            "request_id": "r1",
            "timestamp": 5.2,
        },
        {
            "event_type": "arrival-router",
            "request_id": "r1",
            "timestamp": 5.2,
            "device_id": -1,
            "prefill_ddl": 8.25,
            "profit": 1.0,
            "prompt_tokens": 2,
            "max_tokens": 1,
            "num_cached_tokens": 0,
            "zero_load_ttft": 1.0,
        },
        {
            "event_type": "arrival",
            "request_id": "r1",
            "timestamp": 5.25,
            "device_id": 0,
            "prompt_tokens": 2,
            "max_tokens": 1,
            "num_cached_tokens": 0,
            "zero_load_ttft": 1.0,
            "prefill_ddl": 8.25,
            "profit": 1.0,
            "prefill_only": False,
            "decode_only": False,
            "add_req_time": 5.23,
        },
        {
            "event_type": "batch",
            "timestamp": 8.6,
            "device_id": 0,
            "batch_id": 2,
            "req_ids": ["r1"],
            "num_computed_tokens": [0],
            "num_scheduled_tokens": {"r1": 2},
            "elapsed": 0.1,
            "scheduling_overhead": 0.0,
        },
        {
            "event_type": "finish",
            "request_id": "r1",
            "timestamp": 8.65,
            "device_id": 0,
            "finish_reason": "length",
        },
        {
            "event_type": "energy",
            "timestamp": 0.2,
            "device_id": 0,
            "energy": 5.0,
            "power": 50.0,
            "mhz": 1000.0,
        },
        {
            "event_type": "energy",
            "timestamp": 0.7,
            "device_id": 1,
            "energy": 3.0,
            "power": 30.0,
            "mhz": 1000.0,
        },
        {
            "event_type": "energy",
            "timestamp": 1.2,
            "device_id": 0,
            "energy": 7.0,
            "power": 70.0,
            "mhz": 1000.0,
        },
        {
            "event_type": "energy",
            "timestamp": 1.7,
            "device_id": 1,
            "energy": 3.0,
            "power": 30.0,
            "mhz": 1000.0,
        },
    ]
    trace_path = tmp_path / "test_trace_3.0_0.025.events.jsonl"
    trace_path.write_text(json.dumps(events), encoding="utf-8")
    return trace_path


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


def test_build_energy_comparison_metadata_exports_timeline_series(tmp_path):
    trace_path = _write_timeline_events_trace(tmp_path)

    metadata = events_analysis.build_energy_comparison_metadata(
        [("Synthetic", str(trace_path), 2)],
    )

    assert metadata["schema_version"] == events_analysis.ENERGY_COMPARISON_METADATA_VERSION
    assert metadata["figure"] == "timeline_comparison"
    assert len(metadata["traces"]) == 1

    trace = metadata["traces"][0]
    assert trace["label"] == "Synthetic"
    assert trace["event_file"] == str(trace_path)
    assert trace["n_device"] == 2

    assert trace["power"] == {
        "time_s": [0.0, 1.0],
        "values_w": [8.0, 10.0],
        "per_device_values_w": {
            "0": [5.0, 7.0],
            "1": [3.0, 3.0],
        },
        "window_s": 1.0,
        "source": "measured",
    }
    assert trace["active_devices"] == {
        "time_s": [0.0, 1.0],
        "values": [0, 0],
        "window_s": 1.0,
        "idle_power_w": 70.0,
        "derived_from": "power.per_device_values_w",
    }
    assert trace["arrivals"] == {
        "time_s": [0.0, 5.0],
        "values": [1, 1],
        "window_s": 5.0,
    }
    assert trace["slo_violations"] == {
        "time_s": [0.0, 5.0],
        "values": [0, 1],
        "window_s": 5.0,
    }

    assert json.loads(json.dumps(metadata)) == metadata


def test_export_timeline_comparison_metadata_writes_round_trip_json(tmp_path):
    trace_path = _write_timeline_events_trace(tmp_path)
    output_prefix = tmp_path / "timeline_export"

    metadata, output_path = export_timeline_comparison_metadata(
        [("Synthetic", str(trace_path), 2)],
        prefix=str(output_prefix),
        suffix="unit",
    )

    assert output_path == tmp_path / "timeline_export-unit.meta.json"
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == metadata
