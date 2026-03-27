import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from SLOsServe import perf_model
from SLOsServe.fitting_utils import fit_linear_perf_model
from motivation import bench_api_server


TRUE_HW_PARAMS = [0.1, 0.2, 0.01, 0.0, 0.5]
PIECEWISE_TRUE_HW_PARAMS = {
    "le_512": [0.1, 0.2, 0.01, 0.0, 0.5],
    "513_to_2048": [0.04, 0.35, 0.02, 0.0, 1.25],
    "gt_2048": [0.015, 0.45, 0.03, 0.0, 3.0],
}


def _expected_time(batch, params=TRUE_HW_PARAMS):
    total_current = sum(current for _, current in batch)
    total_past = sum(past for past, _ in batch)
    num_reqs = len(batch)
    return (
        params[0] * total_current
        + params[1] * num_reqs
        + params[2] * total_past
        + params[3] * 1
        + params[4]
    )


def _make_batch_events():
    batches = [
        [(0, 1)],
        [(2, 1)],
        [(0, 2), (0, 3)],
        [(1, 4), (3, 2)],
        [(5, 1), (7, 1), (9, 1)],
    ]
    events = []
    for batch_id, batch in enumerate(batches, start=1):
        scheduling_overhead = 0.05
        measured_time = _expected_time(batch)
        req_ids = [f"req-{batch_id}-{idx}" for idx in range(len(batch))]
        events.append({
            "event_type": "batch",
            "device_id": 0,
            "batch_id": batch_id,
            "timestamp": float(batch_id),
            "elapsed": measured_time + scheduling_overhead,
            "scheduling_overhead": scheduling_overhead,
            "estimated_time": measured_time * 1.1,
            "req_ids": req_ids,
            "num_computed_tokens": [past for past, _ in batch],
            "num_scheduled_tokens": {
                req_id: current for req_id, (_, current) in zip(req_ids, batch)
            },
        })
    return batches, events


def test_fit_linear_perf_model_recovers_known_params():
    batches, _ = _make_batch_events()

    fit_result = fit_linear_perf_model([
        (batch, _expected_time(batch))
        for batch in batches
    ])

    assert fit_result["hardware_params"] == pytest.approx(TRUE_HW_PARAMS)
    assert fit_result["stats"]["num_samples"] == len(batches)
    assert fit_result["predicted_times"] == pytest.approx(
        [_expected_time(batch) for batch in batches]
    )
    assert all("predicted_time" in record for record in fit_result["records"])
    assert fit_result["stats"]["fit_method"] == "nnls"
    assert fit_result["stats"]["non_negative_constraints_applied"] is True


def test_fit_linear_perf_model_enforces_non_negative_terms():
    fit_result = fit_linear_perf_model([
        ([(0, 1)], 3.0),
        ([(0, 2)], 2.0),
        ([(0, 3)], 1.0),
    ], min_abs_num_reqs_coef=0.0)

    assert all(param >= 0.0 for param in fit_result["hardware_params"])
    assert fit_result["hardware_params"][0] == pytest.approx(0.0)
    assert fit_result["hardware_params"][4] >= 0.0


def test_perf_model_fit_persists_params_and_plot(tmp_path, monkeypatch):
    perf_model_path = tmp_path / "assets" / "perf_model.json"
    perf_model_fig_dir = tmp_path / "assets" / "perf_model_figs"
    monkeypatch.setattr(perf_model, "PERF_MODEL_PATH", perf_model_path)
    monkeypatch.setattr(perf_model, "PERF_MODEL_FIG_DIR", perf_model_fig_dir)

    batches, _ = _make_batch_events()
    model = perf_model.PerfModel.get_perf_model(
        "Qwen/Qwen2.5-7B-Instruct",
        "default",
    )

    fit_result = model.fit([
        (batch, _expected_time(batch))
        for batch in batches
    ], tag="unit_test_trace", viz=True)

    assert model.hardware_params == pytest.approx(TRUE_HW_PARAMS)
    assert Path(fit_result["plot_path"]).exists()
    assert perf_model.get_hardware_params(
        "Qwen/Qwen2.5-7B-Instruct",
        "unit_test_trace",
    ) == pytest.approx(TRUE_HW_PARAMS)

    persisted = json.loads(perf_model_path.read_text())
    assert persisted["Qwen/Qwen2.5-7B-Instruct"]["unit_test_trace"] == pytest.approx(
        TRUE_HW_PARAMS
    )


def test_perf_model_copy_with_adjustments_keeps_source_model_unchanged():
    model = perf_model.PerfModel(
        "Qwen/Qwen2.5-7B-Instruct",
        [1.0, 2.0, 3.0, 4.0, 5.0],
    )

    adjusted = model.copy_with_adjustments(scale=1.2, constant_offset=0.7)

    assert adjusted.hardware_params == pytest.approx([1.2, 2.4, 3.6, 4.8, 6.7])
    assert model.hardware_params == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])


def test_piecewise_perf_model_runtime_selects_segments_and_copies_adjustments():
    piecewise_params = perf_model.build_piecewise_current_token_hardware_params(
        PIECEWISE_TRUE_HW_PARAMS,
        breakpoints=[512, 2048],
    )
    model = perf_model.PerfModel(
        "Qwen/Qwen2.5-7B-Instruct",
        piecewise_params,
    )

    assert model.is_piecewise_current_tokens is True
    assert model.get_batch_time([(5, 256)]) == pytest.approx(
        _expected_time([(5, 256)], PIECEWISE_TRUE_HW_PARAMS["le_512"])
    )
    assert model.get_batch_time([(5, 700), (10, 100)]) == pytest.approx(
        _expected_time([(5, 700), (10, 100)], PIECEWISE_TRUE_HW_PARAMS["513_to_2048"])
    )
    assert model.get_batch_time([(5, 2200)]) == pytest.approx(
        _expected_time([(5, 2200)], PIECEWISE_TRUE_HW_PARAMS["gt_2048"])
    )

    target_t = _expected_time([(12, 640)], PIECEWISE_TRUE_HW_PARAMS["513_to_2048"])
    assert model.get_bs(target_t, num_reqs=1, num_past_tokens=12) == 640

    adjusted = model.copy_with_adjustments(scale=1.1, constant_offset=0.3)
    assert adjusted.is_piecewise_current_tokens is True
    assert adjusted.describe_hardware_params()["segment_params"]["le_512"] == pytest.approx(
        [0.11, 0.22, 0.011, 0.0, 0.85]
    )
    assert model.describe_hardware_params()["segment_params"]["le_512"] == pytest.approx(
        PIECEWISE_TRUE_HW_PARAMS["le_512"]
    )


def test_build_problems_propagates_perf_model_err_and_log_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bench_api_server.ArrivalTimes,
        "load",
        staticmethod(lambda *args, **kwargs: SimpleNamespace(arrival_times=[0.0, 1.0])),
    )
    monkeypatch.setattr(
        bench_api_server.Requests,
        "load",
        staticmethod(
            lambda *args, **kwargs: SimpleNamespace(requests=[
                SimpleNamespace(input_length=16, output_length=32),
                SimpleNamespace(input_length=8, output_length=24),
            ])),
    )

    problems = bench_api_server.build_problems(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        trace="azure_chat_23",
        ttft_slo_scale=5.0,
        slo_tpot=0.5,
        profit="constant",
        scheduling_policy="atfc",
        routing_policy="slosserve",
        n_device=2,
        tensor_parallel_size=1,
        window="0:2",
        load_scale=1.0,
        experiment_dir=str(tmp_path),
        perf_model_err=1.2,
        log_perf_model_errors=False,
        enable_piecewise_perf_model_regression=True,
        perf_model_piecewise_breakpoints=[128, 256],
    )

    assert problems
    assert problems[0].routing_kwargs["perf_model_err"] == pytest.approx(1.2)
    assert problems[0].log_perf_model_errors is False
    assert problems[0].enable_piecewise_perf_model_regression is True
    assert problems[0].perf_model_piecewise_breakpoints == [128, 256]
    assert "_pmreg_piecewise_128-256" in problems[0].store_prefix


def test_extract_batch_perf_error_row_derives_compact_batch_fields():
    event = {
        "event_type": "batch",
        "device_id": "2",
        "batch_id": "7",
        "timestamp": "3.5",
        "elapsed": 1.25,
        "scheduling_overhead": 0.15,
        "estimated_time": 1.4,
        "req_ids": ["req-0", "req-1"],
        "num_computed_tokens": [10, 20],
        "num_scheduled_tokens": {
            "req-0": 4,
            "req-1": 1,
        },
    }

    row = bench_api_server._extract_batch_perf_error_row(event)

    assert row["device_id"] == 2
    assert row["batch_id"] == 7
    assert row["timestamp"] == pytest.approx(3.5)
    assert row["batch_size"] == 2
    assert row["total_current_tokens"] == 5
    assert row["total_past_tokens"] == 30
    assert row["estimated_time"] == pytest.approx(1.4)
    assert row["estimated_full_time"] == pytest.approx(1.55)
    assert row["measured_time"] == pytest.approx(1.1)
    assert row["elapsed_time"] == pytest.approx(1.25)
    assert row["scheduling_overhead"] == pytest.approx(0.15)


def test_collect_batch_perf_error_rows_computes_signed_and_relative_errors():
    batches, events = _make_batch_events()

    rows = bench_api_server._collect_batch_perf_error_rows(events)

    assert len(rows) == len(events)
    for row, batch in zip(rows, batches):
        measured_time = _expected_time(batch)
        expected_error = measured_time * 0.1
        assert row["measured_time"] == pytest.approx(measured_time)
        assert row["error_s"] == pytest.approx(expected_error)
        assert row["abs_error_s"] == pytest.approx(expected_error)
        assert row["relative_error"] == pytest.approx(0.1)
        assert row["abs_relative_error"] == pytest.approx(0.1)
        assert row["estimated_over_measured"] == pytest.approx(1.1)
        assert row["estimated_full_time"] == pytest.approx(measured_time * 1.1 + 0.05)
        assert row["full_error_s"] == pytest.approx(expected_error)
        assert row["full_relative_error"] == pytest.approx(
            expected_error / (measured_time + 0.05)
        )
        assert "batch" not in row


def test_collect_batch_perf_error_rows_handles_zero_measured_time():
    rows = bench_api_server._collect_batch_perf_error_rows([{
        "event_type": "batch",
        "device_id": 0,
        "batch_id": 1,
        "timestamp": 0.0,
        "elapsed": 0.2,
        "scheduling_overhead": 0.2,
        "estimated_time": 0.3,
        "req_ids": ["req-0"],
        "num_computed_tokens": [0],
        "num_scheduled_tokens": {"req-0": 1},
    }])

    assert len(rows) == 1
    assert rows[0]["measured_time"] == 0.0
    assert rows[0]["error_s"] == pytest.approx(0.3)
    assert rows[0]["relative_error"] is None
    assert rows[0]["abs_relative_error"] is None
    assert rows[0]["estimated_over_measured"] is None
    assert rows[0]["estimated_full_time"] == pytest.approx(0.5)
    assert rows[0]["full_error_s"] == pytest.approx(0.3)
    assert rows[0]["full_relative_error"] == pytest.approx(1.5)
    assert rows[0]["estimated_full_over_elapsed"] == pytest.approx(2.5)


def test_log_perf_model_errors_from_batch_events_writes_jsonl_summary_and_rows(tmp_path):
    batches, events = _make_batch_events()
    problem = bench_api_server.Problem(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        length_pattern="unit_task",
        store_prefix=str(tmp_path / "runs" / "case_a"),
        perf_model_err=1.25,
        scheduling_overhead=0.05,
        enable_piecewise_perf_model_regression=True,
        perf_model_piecewise_breakpoints=[512, 2048],
    )
    output_path = tmp_path / "case_a.perf_model_errors.jsonl"

    artifacts = bench_api_server._log_perf_model_errors_from_batch_events(
        problem,
        events,
        output_path,
        event_file="case_a.0.events.jsonl",
    )

    assert artifacts is not None
    assert Path(artifacts["path"]) == output_path
    assert output_path.exists()
    assert Path(artifacts["figure_path"]).exists()
    assert Path(artifacts["full_elapsed_figure_path"]).exists()
    assert Path(artifacts["regression_figure_path"]).exists()

    rows = [
        json.loads(line)
        for line in output_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == len(events) + 1

    summary = rows[0]
    batch_rows = rows[1:]
    expected_signed_errors = [_expected_time(batch) * 0.1 for batch in batches]

    assert summary["record_type"] == "summary"
    assert summary["model_name"] == problem.model_name
    assert summary["length_pattern"] == problem.length_pattern
    assert summary["event_file"] == "case_a.0.events.jsonl"
    assert summary["perf_model_err"] == pytest.approx(1.25)
    assert summary["configured_scheduling_overhead_s"] == pytest.approx(0.05)
    assert summary["regression_model_type"] == "piecewise_current_tokens"
    assert summary["regression_breakpoints"] == [512, 2048]
    assert summary["regressed_hardware_params"]["type"] == "piecewise_current_tokens"
    assert summary["regressed_hardware_params"]["segment_params"]["le_512"] == pytest.approx(
        TRUE_HW_PARAMS
    )
    assert summary["regressed_hardware_params_fitted_segments"] == ["le_512"]
    assert summary["regressed_hardware_params_fallback_segments"] == [
        "513_to_2048",
        "gt_2048",
    ]
    assert summary["relative_error_denominator"] == "measured_time"
    assert summary["estimated_minus_measured_s"]["count"] == len(events)
    assert summary["estimated_minus_measured_s"]["mean"] == pytest.approx(
        sum(expected_signed_errors) / len(expected_signed_errors)
    )
    assert summary["estimated_minus_measured_relative"]["mean"] == pytest.approx(0.1)
    assert summary["estimated_minus_measured_relative"]["p50"] == pytest.approx(0.1)
    assert summary["abs_estimated_minus_measured_relative"]["p95"] == pytest.approx(0.1)
    assert summary["estimated_with_overhead_minus_elapsed_s"]["mean"] == pytest.approx(
        sum(expected_signed_errors) / len(expected_signed_errors)
    )
    assert summary["regression_stats"]["aggregate"]["r2"] == pytest.approx(1.0)
    assert summary["empirical_scheduling_overhead"]["overhead_s"]["mean"] == pytest.approx(0.05)
    assert summary["empirical_scheduling_overhead"]["overhead_s"]["p50"] == pytest.approx(0.05)
    assert summary["empirical_scheduling_overhead"]["relative_to_measured_time"]["count"] == len(events)
    assert Path(summary["estimated_vs_measured_figure_path"]).exists()
    assert Path(summary["estimated_with_overhead_vs_elapsed_figure_path"]).exists()
    assert Path(summary["regression_figure_path"]).exists()

    assert all(row["record_type"] == "batch_error" for row in batch_rows)
    assert [row["batch_id"] for row in batch_rows] == list(range(1, len(events) + 1))
    assert all("batch" not in row for row in batch_rows)
    assert all(row["relative_error"] == pytest.approx(0.1) for row in batch_rows)
    assert all(
        row["estimated_over_measured"] == pytest.approx(1.1)
        for row in batch_rows
    )


def test_load_batch_trace_events_accepts_jsonl_lines(tmp_path):
    _, events = _make_batch_events()
    trace_path = tmp_path / "batches.jsonl"
    trace_path.write_text(
        "\n".join(json.dumps(event) for event in events),
        encoding="utf-8",
    )

    loaded = perf_model.load_batch_trace_events(trace_path)

    assert loaded == events


def test_collect_batch_perf_samples_filters_by_device_and_subtracts_overhead(tmp_path):
    batches, events = _make_batch_events()
    for idx, event in enumerate(events):
        event["device_id"] = idx % 2

    trace_path = tmp_path / "batches.jsonl"
    trace_path.write_text(json.dumps(events), encoding="utf-8")

    sample_data = perf_model.collect_batch_perf_samples(trace_path, device_id=1)

    expected_batches = [batch for idx, batch in enumerate(batches) if idx % 2 == 1]
    expected_times = [_expected_time(batch) for batch in expected_batches]

    assert sample_data["loaded_event_count"] == len(events)
    assert sample_data["invalid_event_count"] == 0
    assert [batch for batch, _ in sample_data["batch_times"]] == expected_batches
    assert [measured for _, measured in sample_data["batch_times"]] == pytest.approx(
        expected_times
    )


def test_fit_batch_perf_trace_writes_report_plot_and_registry(tmp_path, monkeypatch):
    perf_model_path = tmp_path / "assets" / "perf_model.json"
    perf_model_fig_dir = tmp_path / "assets" / "perf_model_figs"
    monkeypatch.setattr(perf_model, "PERF_MODEL_PATH", perf_model_path)
    monkeypatch.setattr(perf_model, "PERF_MODEL_FIG_DIR", perf_model_fig_dir)

    _, events = _make_batch_events()
    trace_path = tmp_path / "batches.jsonl"
    trace_path.write_text(json.dumps(events), encoding="utf-8")
    report_path = tmp_path / "batch_fit_report.json"

    report = perf_model.fit_batch_perf_trace(
        trace_path,
        model_name="unit/model",
        tag="unit_trace",
        viz=True,
        report_path=report_path,
    )

    assert report["hardware_params"] == pytest.approx(TRUE_HW_PARAMS)
    assert report["fitted_estimator_stats"]["r2"] == pytest.approx(1.0)
    assert Path(report["plot_path"]).exists()
    assert Path(report["report_path"]).exists()
    assert perf_model.get_hardware_params("unit/model", "unit_trace") == pytest.approx(
        TRUE_HW_PARAMS
    )

    persisted = json.loads(perf_model_path.read_text())
    assert persisted["unit/model"]["unit_trace"] == pytest.approx(TRUE_HW_PARAMS)

    written_report = json.loads(report_path.read_text())
    assert written_report["used_sample_count"] == len(events)
    assert written_report["existing_estimator_stats"]["mae"] > 0.0


def test_piecewise_perf_model_registry_round_trip(tmp_path, monkeypatch):
    perf_model_path = tmp_path / "assets" / "perf_model.json"
    monkeypatch.setattr(perf_model, "PERF_MODEL_PATH", perf_model_path)
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    piecewise_params = perf_model.build_piecewise_current_token_hardware_params(
        PIECEWISE_TRUE_HW_PARAMS,
        breakpoints=[512, 2048],
    )
    perf_model.upsert_hardware_params(model_name, "piecewise_trace", piecewise_params)

    loaded = perf_model.get_hardware_params(model_name, "piecewise_trace")
    assert loaded == piecewise_params

    loaded_model = perf_model.PerfModel.get_perf_model(model_name, "piecewise_trace")
    assert loaded_model.is_piecewise_current_tokens is True
    assert loaded_model.describe_hardware_params() == piecewise_params


def _make_multicategory_batch_events():
    batches = {
        "decode": [
            [(0, 1)],
            [(2, 1)],
            [(5, 1), (7, 1), (9, 1)],
        ],
        "prefill": [
            [(0, 2)],
            [(0, 2), (0, 3)],
            [(1, 4), (3, 2)],
        ],
        "mixed": [
            [(0, 5), (3, 1)],
            [(0, 2), (5, 1), (7, 1)],
            [(2, 4), (8, 1)],
        ],
    }
    events = []
    batch_id = 1
    for category_batches in batches.values():
        for batch in category_batches:
            scheduling_overhead = 0.05
            measured_time = _expected_time(batch)
            req_ids = [f"req-{batch_id}-{idx}" for idx in range(len(batch))]
            events.append({
                "event_type": "batch",
                "device_id": 0,
                "batch_id": batch_id,
                "timestamp": float(batch_id),
                "elapsed": measured_time + scheduling_overhead,
                "scheduling_overhead": scheduling_overhead,
                "estimated_time": measured_time * 1.1,
                "req_ids": req_ids,
                "num_computed_tokens": [past for past, _ in batch],
                "num_scheduled_tokens": {
                    req_id: current for req_id, (_, current) in zip(req_ids, batch)
                },
            })
            batch_id += 1
    return batches, events


def _make_piecewise_batch_events():
    segment_batches = {
        "le_512": [
            [(0, 128)],
            [(5, 256)],
            [(10, 200), (20, 100)],
            [(30, 150), (40, 200)],
            [(15, 100), (25, 120), (35, 140)],
        ],
        "513_to_2048": [
            [(0, 600)],
            [(100, 700), (200, 150)],
            [(50, 512), (60, 300)],
            [(10, 1024)],
            [(30, 600), (40, 650), (50, 200)],
            [(80, 800), (90, 900)],
        ],
        "gt_2048": [
            [(0, 2200)],
            [(100, 1500), (200, 800)],
            [(50, 1024), (60, 1200)],
            [(10, 3000)],
            [(30, 1200), (40, 1100), (50, 900)],
            [(80, 2048), (90, 1500)],
        ],
    }
    events = []
    batch_id = 1
    for segment_key, batches in segment_batches.items():
        params = PIECEWISE_TRUE_HW_PARAMS[segment_key]
        for batch in batches:
            scheduling_overhead = 0.05
            measured_time = _expected_time(batch, params=params)
            req_ids = [f"req-{batch_id}-{idx}" for idx in range(len(batch))]
            events.append({
                "event_type": "batch",
                "device_id": 0,
                "batch_id": batch_id,
                "timestamp": float(batch_id),
                "elapsed": measured_time + scheduling_overhead,
                "scheduling_overhead": scheduling_overhead,
                "estimated_time": measured_time * 1.1,
                "req_ids": req_ids,
                "num_computed_tokens": [past for past, _ in batch],
                "num_scheduled_tokens": {
                    req_id: current for req_id, (_, current) in zip(req_ids, batch)
                },
            })
            batch_id += 1
    return segment_batches, events


def test_extract_batch_perf_sample_classifies_prefill_decode_and_mixed():
    decode = perf_model.extract_batch_perf_sample({
        "event_type": "batch",
        "device_id": 0,
        "batch_id": 1,
        "timestamp": 0.0,
        "elapsed": 0.2,
        "scheduling_overhead": 0.01,
        "estimated_time": 0.2,
        "req_ids": ["a", "b"],
        "num_computed_tokens": [3, 4],
        "num_scheduled_tokens": {"a": 1, "b": 1},
    })
    prefill = perf_model.extract_batch_perf_sample({
        "event_type": "batch",
        "device_id": 0,
        "batch_id": 2,
        "timestamp": 0.0,
        "elapsed": 0.2,
        "scheduling_overhead": 0.01,
        "estimated_time": 0.2,
        "req_ids": ["a", "b"],
        "num_computed_tokens": [0, 0],
        "num_scheduled_tokens": {"a": 2, "b": 3},
    })
    mixed = perf_model.extract_batch_perf_sample({
        "event_type": "batch",
        "device_id": 0,
        "batch_id": 3,
        "timestamp": 0.0,
        "elapsed": 0.2,
        "scheduling_overhead": 0.01,
        "estimated_time": 0.2,
        "req_ids": ["a", "b"],
        "num_computed_tokens": [0, 4],
        "num_scheduled_tokens": {"a": 4, "b": 1},
    })

    assert decode["batch_category"] == "decode"
    assert prefill["batch_category"] == "prefill"
    assert mixed["batch_category"] == "mixed"


def test_fit_batch_perf_trace_by_category_writes_category_reports_and_plots(tmp_path):
    batches, events = _make_multicategory_batch_events()
    trace_path = tmp_path / "batches.jsonl"
    trace_path.write_text(json.dumps(events), encoding="utf-8")

    report = perf_model.fit_batch_perf_trace(
        trace_path,
        fit_by_category=True,
        viz=True,
    )

    assert report["sample_summary"]["batch_categories"] == {
        "prefill": len(batches["prefill"]),
        "decode": len(batches["decode"]),
        "mixed": len(batches["mixed"]),
    }
    assert Path(report["plot_path"]).exists()
    assert Path(report["existing_by_category_plot_path"]).exists()
    assert Path(report["category_fit"]["plot_path"]).exists()
    assert report["category_fit"]["aggregate_stats"]["r2"] == pytest.approx(1.0)
    assert report["category_fit"]["categories"]["decode"]["used_sample_count"] == len(
        batches["decode"]
    )
    assert report["category_fit"]["categories"]["prefill"]["used_sample_count"] == len(
        batches["prefill"]
    )
    assert report["category_fit"]["categories"]["mixed"]["used_sample_count"] == len(
        batches["mixed"]
    )


def test_fit_batch_perf_trace_piecewise_current_tokens_recovers_segment_models(tmp_path):
    segment_batches, events = _make_piecewise_batch_events()
    trace_path = tmp_path / "batches.jsonl"
    trace_path.write_text(json.dumps(events), encoding="utf-8")

    report = perf_model.fit_batch_perf_trace(
        trace_path,
        fit_piecewise_current_tokens=True,
        viz=True,
    )

    piecewise = report["piecewise_current_token_fit"]
    assert piecewise["breakpoints"] == [512, 2048]
    assert piecewise["segment_order"] == ["le_512", "513_to_2048", "gt_2048"]
    assert Path(piecewise["plot_path"]).exists()
    assert piecewise["aggregate_stats"]["r2"] == pytest.approx(1.0)
    assert piecewise["aggregate_stats"]["mae"] == pytest.approx(0.0)
    assert piecewise["aggregate_stats"]["mae"] < report["fitted_estimator_stats"]["mae"]

    for segment_key, expected_batches in segment_batches.items():
        segment_report = piecewise["segments"][segment_key]
        assert segment_report["used_sample_count"] == len(expected_batches)
        assert segment_report["hardware_params"] == pytest.approx(
            PIECEWISE_TRUE_HW_PARAMS[segment_key]
        )
