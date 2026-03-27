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
    )

    assert problems
    assert problems[0].routing_kwargs["perf_model_err"] == pytest.approx(1.2)
    assert problems[0].log_perf_model_errors is False


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
