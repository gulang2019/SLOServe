import csv
import json

import pytest

from scripts import beam_figure7_metrics as metrics


def test_selection_prefers_latest_sources_without_pooling_old_mixed(tmp_path):
    names = [
        (metrics.OLD_ARCHIVE, "chat_7b/split_1_0"),
        (metrics.OLD_ARCHIVE, "code_7b/split_5_0"),
        (metrics.OLD_ARCHIVE, "code_7b/split_6_0"),
        (metrics.OLD_ARCHIVE, "mixed_7b/split_8_0"),
        (metrics.OLD_ARCHIVE, "mixed_7b/split_10_0"),
        (metrics.NEW_ARCHIVE, "code/split_6_1"),
        (metrics.NEW_ARCHIVE, "mixed/split_8_3"),
        (metrics.NEW_ARCHIVE, "mixed/split_8_7"),
        (metrics.NEW_ARCHIVE, "large_scale_chat/split_8_0"),
        (metrics.NEW_ARCHIVE, "multiround_chat/split_12_0"),
    ]
    for archive, name in names:
        (tmp_path / archive / name).mkdir(parents=True)
    selected = metrics._select_sources(tmp_path)
    assert list(selected) == [
        ("chat_7b", 1), ("code_7b", 5), ("code_7b", 6),
        ("mixed_7b", 8), ("mixed_7b", 10),
        ("large_scale_chat", 8), ("multiround_chat", 12),
    ]
    assert selected["code_7b", 5] == [tmp_path / metrics.OLD_ARCHIVE / "code_7b/split_5_0"]
    assert selected["code_7b", 6] == [tmp_path / metrics.NEW_ARCHIVE / "code/split_6_1"]
    assert selected["mixed_7b", 8] == [
        tmp_path / metrics.NEW_ARCHIVE / "mixed/split_8_3",
        tmp_path / metrics.NEW_ARCHIVE / "mixed/split_8_7",
    ]


def test_selection_rejects_partial_updated_mixed_pair(tmp_path):
    (tmp_path / metrics.NEW_ARCHIVE / "mixed/split_8_3").mkdir(parents=True)
    with pytest.raises(ValueError, match="both shards 3 and 7"):
        metrics._select_sources(tmp_path)


def test_old_coding_option_changes_only_code_and_keeps_all_old_points(tmp_path):
    paths = [
        (metrics.OLD_ARCHIVE, "chat_7b/split_8_0"),
        (metrics.OLD_ARCHIVE, "code_7b/split_5_0"),
        (metrics.OLD_ARCHIVE, "code_7b/split_6_0"),
        (metrics.OLD_ARCHIVE, "code_7b/split_8_0"),
        (metrics.OLD_ARCHIVE, "mixed_7b/split_8_0"),
        (metrics.NEW_ARCHIVE, "code/split_6_1"),
        (metrics.NEW_ARCHIVE, "code/split_8_1"),
        (metrics.NEW_ARCHIVE, "mixed/split_8_3"),
        (metrics.NEW_ARCHIVE, "mixed/split_8_7"),
        (metrics.NEW_ARCHIVE, "large_scale_chat/split_16_0"),
        (metrics.NEW_ARCHIVE, "multiround_chat/split_12_0"),
    ]
    for archive, name in paths:
        (tmp_path / archive / name).mkdir(parents=True)
    latest = metrics._select_sources(tmp_path)
    assert latest == metrics._select_sources(tmp_path, coding_source="latest")
    old = metrics._select_sources(tmp_path, coding_source="old")
    assert old.keys() == latest.keys()
    for (workload, n_device), selected in old.items():
        if workload == "code_7b":
            assert selected == [
                tmp_path / metrics.OLD_ARCHIVE / "code_7b" / f"split_{n_device}_0",
            ]
        else:
            assert selected == latest[workload, n_device]
    assert old["code_7b", 6] != latest["code_7b", 6]


@pytest.mark.parametrize("coding_source", ["new", "", None, 0])
@pytest.mark.parametrize("collector", [metrics._select_sources, metrics.collect_beam_points])
def test_coding_source_option_is_validated(tmp_path, coding_source, collector):
    with pytest.raises(ValueError, match="coding_source must be 'latest' or 'old'"):
        collector(tmp_path, coding_source=coding_source)


def test_collector_forwards_old_coding_option(tmp_path, monkeypatch):
    seen = []

    def select(root, *, coding_source):
        seen.append((root, coding_source))
        return {}

    monkeypatch.setattr(metrics, "_select_sources", select)
    assert metrics.collect_beam_points(tmp_path, coding_source="old") == []
    assert seen == [(tmp_path, "old")]


def _run_record(device, requests, violations, energy):
    return {
        "source": f"mixed/split_8_{device}", "device_index": device,
        "requests": requests,
        "counts": {"lower": violations, "upper": violations + 1,
                   "ttft": 0, "errors": 0, "ambiguous": 1},
        "scaled_gpu_energy_j": energy, "raw_gpu_energy_j": energy / 2,
        "control_targets": {"device_index": device, "tbt_s": .025},
        "monitor_duration_s": 2.0 + device, "benchmark_duration_s": 2.0,
        "source_sha256": {f"source-{device}": f"hash-{device}"},
    }


def test_pooling_weights_slo_by_requests_but_energy_by_shards():
    point = metrics._combine_runs("mixed_7b", 8, [
        _run_record(3, 10, 1, 100), _run_record(7, 30, 9, 300),
    ])
    assert point["requests"] == 40
    assert point["violation_pct_lower"] == 25.0
    assert point["violation_pct_upper"] == 30.0
    assert point["energy_kj"] == 1.6  # 8 * mean(100, 300) J / 1000.
    assert point["raw_energy_kj"] == .8
    assert point["monitor_durations_s"] == [5.0, 9.0]
    assert point["device_indices"] == [3, 7]
    assert point["eval_tpot_s"] == .025
    assert point["original_csv_tpot_s"] == .020
    assert point["allowance_s"] == .200
    assert point["evaluation_matches_original_tpot"] is False
    assert point["energy_is_extrapolated"] is True
    json.dumps(point)  # The plotting wrapper writes the entire point as JSON.


def _write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def session_run(tmp_path):
    path = tmp_path / metrics.NEW_ARCHIVE / "multiround_chat/split_8_0"
    path.mkdir(parents=True)
    summary = {
        "requests": 1, "ttft_scale": 5, "zero_load_ttft_per_token_s": .000061,
        "zero_load_ttft_constant_s": .01, "routing_overhead_s": .05,
        "slo_tpot_s": .05,
    }
    config = dict(summary, beam_global_ttft_s=.7, beam_tbt_s=.05, beam_tpot_s=.05)
    (path / "slo_config.txt").write_text("\n".join(f"{k}={v}" for k, v in config.items()))
    _write_json(path / "per_request_slo.summary.json", summary)
    _write_json(path / "beam_detailed.json", {
        "ttfts": [.12], "duration": 1.9, "request_ids": ["r1"], "cached_tokens": [0],
    })
    _write_json(path / "per_request_slo.jsonl", {
        "request_index": 0, "prompt_tokens": 100, "target_output_tokens": 2,
        "actual_output_tokens": 2, "ttft_budget_s": .05 + 5 * (.01 + .000061 * 100),
        "ttft_s": .12, "itls_s": [.02], "success": True,
    })
    (path / "power_timeline_gpu_0.csv").write_text(
        "elapsed_s,cumulative_energy_mj\n0,1000000\n1,1052000\n2,1104000\n",
    )
    trace = tmp_path / metrics.SESSION_TRACES / "8/0.csv"
    trace.parent.mkdir(parents=True)
    with trace.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "RequestId", "ContextTokens", "GeneratedTokens", "CachedTokens",
        ])
        writer.writeheader()
        writer.writerow(dict(RequestId="r1", ContextTokens=100, GeneratedTokens=2, CachedTokens=80))
    return tmp_path, path


def test_session_budget_uses_expected_cache_not_observed_hits(session_run):
    root, path = session_run
    first = metrics._read_run(root, path, "multiround_chat")
    # Cold budget=.1305 s would pass; expected-cache budget=.1061 s fails.
    assert first["counts"] == {"lower": 1, "upper": 1, "ttft": 1, "errors": 0, "ambiguous": 0}
    assert first["expected_cache_adjusted"] is True
    detail = json.loads((path / "beam_detailed.json").read_text())
    detail["cached_tokens"] = [99]
    _write_json(path / "beam_detailed.json", detail)
    second = metrics._read_run(root, path, "multiround_chat")
    assert second["counts"] == first["counts"]


def test_energy_uses_full_monitor_counter_delta_and_per_gpu_scaling(session_run):
    root, path = session_run
    run = metrics._read_run(root, path, "multiround_chat")
    assert run["raw_gpu_energy_j"] == pytest.approx(104.0)
    assert run["scaled_gpu_energy_j"] == pytest.approx(140.0)
    assert run["monitor_duration_s"] == 2.0
    assert run["benchmark_duration_s"] == 1.9
    assert run["control_targets"]["tbt_s"] == .05
    assert len(run["source_sha256"]) == 6


def test_multiround_25ms_changes_decode_only_not_energy_cache_or_control(session_run):
    root, path = session_run
    row = json.loads((path / "per_request_slo.jsonl").read_text())
    row.update(ttft_s=.10, itls_s=[.04])
    _write_json(path / "per_request_slo.jsonl", row)
    detail = json.loads((path / "beam_detailed.json").read_text())
    detail["ttfts"] = [.10]
    _write_json(path / "beam_detailed.json", detail)
    default = metrics._read_run(root, path, "multiround_chat")
    updated = metrics._read_run(root, path, "multiround_chat", multiround_tpot_s=.025)
    # Cached-aware TTFT budget stays .1061 s. Second token at .14 s passes
    # its .1561 s deadline at 50 ms but misses the .1311 s deadline at 25 ms.
    assert default["counts"]["lower"] == default["counts"]["upper"] == 0
    assert updated["counts"]["lower"] == updated["counts"]["upper"] == 1
    assert default["counts"]["ttft"] == updated["counts"]["ttft"] == 0
    assert default["eval_tpot_s"] == .05
    assert updated["eval_tpot_s"] == .025
    for key in (
        "raw_gpu_energy_j", "scaled_gpu_energy_j", "monitor_duration_s",
        "benchmark_duration_s", "control_targets", "saved_eval_tpot_s",
        "ttft_calibration", "expected_cache_adjusted", "allowance_s",
        "config", "source_sha256",
    ):
        assert updated[key] == default[key]
    assert updated["saved_eval_tpot_s"] == updated["control_targets"]["tpot_s"] == .05
    assert updated["expected_cache_adjusted"] is True
    point = metrics._combine_runs(
        "multiround_chat", 8, [updated], multiround_tpot_s=.025,
    )
    assert point["eval_tpot_s"] == .025
    assert point["original_csv_tpot_s"] == .05
    assert point["evaluation_matches_original_tpot"] is False
    assert any("25 ms" in text and "50 ms" in text for text in point["limitations"])


def test_default_and_explicit_none_multiround_scoring_are_identical(session_run):
    root, path = session_run
    default = metrics._read_run(root, path, "multiround_chat")
    assert default == metrics._read_run(root, path, "multiround_chat", multiround_tpot_s=None)
    assert metrics._combine_runs("multiround_chat", 8, [default]) == metrics._combine_runs(
        "multiround_chat", 8, [default], multiround_tpot_s=None,
    )
    assert metrics.collect_beam_points(root) == metrics.collect_beam_points(root, multiround_tpot_s=None)


@pytest.mark.parametrize("workload", ["chat_7b", "code_7b", "mixed_7b", "large_scale_chat"])
def test_multiround_override_leaves_other_workload_scoring_unchanged(session_run, workload):
    root, path = session_run
    summary = json.loads((path / "per_request_slo.summary.json").read_text())
    summary.update(ttft_scale=3, slo_tpot_s=.025)
    _write_json(path / "per_request_slo.summary.json", summary)
    config = dict(summary, beam_global_ttft_s=.7, beam_tbt_s=.025, beam_tpot_s=.025)
    (path / "slo_config.txt").write_text("\n".join(f"{k}={v}" for k, v in config.items()))
    row = json.loads((path / "per_request_slo.jsonl").read_text())
    row["ttft_budget_s"] = .05 + 3 * (.01 + .000061 * 100)
    _write_json(path / "per_request_slo.jsonl", row)
    default = metrics._read_run(root, path, workload)
    updated = metrics._read_run(root, path, workload, multiround_tpot_s=.010)
    assert updated == default
    assert metrics._combine_runs(workload, 8, [default]) == metrics._combine_runs(
        workload, 8, [updated], multiround_tpot_s=.010,
    )


def test_collector_forwards_multiround_override_to_run_and_point(session_run):
    root, _ = session_run
    points = metrics.collect_beam_points(root, multiround_tpot_s=.025)
    assert len(points) == 1
    assert points[0]["workload"] == "multiround_chat"
    assert points[0]["eval_tpot_s"] == points[0]["per_shard"][0]["eval_tpot_s"] == .025
    assert points[0]["original_csv_tpot_s"] == .05
    assert points[0]["control_targets"][0]["tpot_s"] == .05


@pytest.mark.parametrize("override", [0, -.025, float("nan"), float("inf"), -float("inf"), True, "bad"])
def test_invalid_multiround_tpot_is_rejected_before_reading_files(tmp_path, override):
    with pytest.raises(ValueError, match="multiround_tpot_s must be finite and positive"):
        metrics.collect_beam_points(tmp_path, multiround_tpot_s=override)
    with pytest.raises(ValueError, match="multiround_tpot_s must be finite and positive"):
        metrics._read_run(tmp_path, tmp_path / "split_8_0", "multiround_chat", multiround_tpot_s=override)
    with pytest.raises(ValueError, match="multiround_tpot_s must be finite and positive"):
        metrics._combine_runs("multiround_chat", 8, [], multiround_tpot_s=override)


def test_session_request_ids_must_match_trace(session_run):
    root, path = session_run
    detail = json.loads((path / "beam_detailed.json").read_text())
    detail["request_ids"] = ["wrong"]
    _write_json(path / "beam_detailed.json", detail)
    with pytest.raises(ValueError, match="RequestId population"):
        metrics._read_run(root, path, "multiround_chat")


def test_archived_cold_budget_must_match_recorded_calibration(session_run):
    root, path = session_run
    row = json.loads((path / "per_request_slo.jsonl").read_text())
    row["ttft_budget_s"] = .2
    _write_json(path / "per_request_slo.jsonl", row)
    with pytest.raises(ValueError, match="TTFT budget does not match"):
        metrics._read_run(root, path, "multiround_chat")


@pytest.mark.parametrize("samples", [
    "0,1000\n0,2000\n", "0,2000\n1,1000\n", "0,1000\n1,nan\n", "0,1000\n",
])
def test_invalid_power_measurements_are_rejected(session_run, samples):
    root, path = session_run
    (path / "power_timeline_gpu_0.csv").write_text("elapsed_s,cumulative_energy_mj\n" + samples)
    with pytest.raises(ValueError, match="Invalid power timeline"):
        metrics._read_run(root, path, "multiround_chat")
