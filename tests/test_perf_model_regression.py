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


def test_build_problems_propagates_perf_model_err_to_router_kwargs(monkeypatch, tmp_path):
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
    )

    assert problems
    assert problems[0].routing_kwargs["perf_model_err"] == pytest.approx(1.2)


def test_extract_batch_regression_row_derives_batch_fields():
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

    row = bench_api_server._extract_batch_regression_row(event)

    assert row == {
        "device_id": 2,
        "batch_id": 7,
        "timestamp": 3.5,
        "batch_size": 2,
        "total_current_tokens": 5,
        "total_past_tokens": 30,
        "estimated_time": 1.4,
        "measured_time": 1.1,
        "elapsed_time": 1.25,
        "scheduling_overhead": 0.15,
        "batch": [
            {"past_tokens": 10, "scheduled_tokens": 4},
            {"past_tokens": 20, "scheduled_tokens": 1},
        ],
    }


def test_regress_perf_model_from_batch_events_writes_dataset_and_figure(tmp_path, monkeypatch):
    perf_model_path = tmp_path / "assets" / "perf_model.json"
    perf_model_fig_dir = tmp_path / "assets" / "perf_model_figs"
    regression_dir = tmp_path / "assets" / "perf_model_regressions"
    monkeypatch.setattr(perf_model, "PERF_MODEL_PATH", perf_model_path)
    monkeypatch.setattr(perf_model, "PERF_MODEL_FIG_DIR", perf_model_fig_dir)
    monkeypatch.setattr(bench_api_server, "PERF_MODEL_REGRESSION_DIR", regression_dir)

    _, events = _make_batch_events()
    problem = bench_api_server.Problem(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        length_pattern="unit_task",
        store_prefix=str(tmp_path / "runs" / "case_a"),
    )

    artifacts = bench_api_server._regress_perf_model_from_batch_events(problem, events)

    assert artifacts is not None
    regression_path = Path(artifacts["regression_path"])
    figure_path = Path(artifacts["figure_path"])
    assert regression_path.exists()
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0

    payload = json.loads(regression_path.read_text())
    assert payload["model_name"] == problem.model_name
    assert payload["task"] == problem.length_pattern
    assert payload["hardware_params"] == pytest.approx(TRUE_HW_PARAMS)
    assert len(payload["new_regressed"]) == len(events)
    assert all("predicted_time" in row for row in payload["new_regressed"])
    assert [row["predicted_time"] for row in payload["new_regressed"]] == pytest.approx(
        [row["measured_time"] for row in payload["new_regressed"]]
    )

    assert perf_model.get_hardware_params(
        problem.model_name,
        problem.length_pattern,
    ) == pytest.approx(TRUE_HW_PARAMS)
