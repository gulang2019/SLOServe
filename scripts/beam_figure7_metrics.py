"""Auditable BEAM inputs for the Figure 7 sensitivity preview.

Evaluation uses 25 ms TPOT for request workloads and, by default, 50 ms for sessions,
not necessarily the recorded scheduler target or the original-system SLO.
In particular, the original Mixed/Large CSVs retain their 20 ms TPOT scores.
Energy is N times mean scaled single-GPU full-monitor energy, not a measured
deployment or an equal-duration comparison with the original systems.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.plot_beam_figure7_comparisons import scale_energy
from scripts.plot_beam_slo_timeline import evaluate_request


OLD_ARCHIVE = "beam-figure7-7b-20261020-215014"
NEW_ARCHIVE = "beam-jobs-mlsys"
SESSION_TRACES = (
    "3rdparty/BEAM/benchmarks/energy/datasets/"
    "sloserve_fig7_session_round_robin/multiround_chat"
)
# Total evaluation allowance, evaluation TPOT, original CSV TPOT, TTFT scale.
SETTINGS = {
    "chat_7b": (.100, .025, .025, 3),
    "code_7b": (.200, .025, .025, 3),
    "mixed_7b": (.200, .025, .020, 3),
    "large_scale_chat": (.050, .025, .020, 3),
    "multiround_chat": (.050, .050, .050, 5),
}


def _validate_multiround_tpot(value):
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("multiround_tpot_s must be finite and positive") from exc
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(converted) or converted <= 0:
        raise ValueError("multiround_tpot_s must be finite and positive")
    return converted


def _evaluation_settings(workload, multiround_tpot_s):
    override = _validate_multiround_tpot(multiround_tpot_s)
    allowance, tpot, original_tpot, scale = SETTINGS[workload]
    if workload == "multiround_chat" and override is not None:
        tpot = override
    return allowance, tpot, original_tpot, scale


def _identity(path):
    _, n_device, device = path.name.split("_")
    n_device, device = int(n_device), int(device)
    if n_device < 1 or not 0 <= device < n_device:
        raise ValueError(f"Invalid shard identity: {path}")
    return n_device, device


def _select_sources(root, *, coding_source="latest"):
    if coding_source not in ("latest", "old"):
        raise ValueError("coding_source must be 'latest' or 'old'")
    selected = {}
    for workload in ("chat_7b", "code_7b", "mixed_7b"):
        for path in (root / OLD_ARCHIVE / workload).glob("split_*_0"):
            selected[workload, _identity(path)[0]] = [path]
    # Default: prefer the new Code shard, not whichever run scores best.
    # The explicit old option changes only Coding and retains every old shard 0.
    if coding_source == "latest":
        for path in (root / NEW_ARCHIVE / "code").glob("split_*_1"):
            selected["code_7b", _identity(path)[0]] = [path]
    mixed = [root / NEW_ARCHIVE / "mixed" / f"split_8_{i}" for i in (3, 7)]
    if any(path.exists() for path in mixed):
        if not all(path.is_dir() for path in mixed):
            raise ValueError("Updated Mixed N=8 requires both shards 3 and 7")
        selected["mixed_7b", 8] = mixed
    for workload in ("large_scale_chat", "multiround_chat"):
        for path in (root / NEW_ARCHIVE / workload).glob("split_*_0"):
            selected[workload, _identity(path)[0]] = [path]
    order = {workload: i for i, workload in enumerate(SETTINGS)}
    return dict(sorted(selected.items(), key=lambda item: (order[item[0][0]], item[0][1])))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_run(root, path, workload, *, multiround_tpot_s=None):
    allowance, tpot, _, expected_scale = _evaluation_settings(workload, multiround_tpot_s)
    n_device, device = _identity(path)
    paths = {name: path / name for name in (
        "beam_detailed.json", "per_request_slo.jsonl", "per_request_slo.summary.json",
        "slo_config.txt", "power_timeline_gpu_0.csv",
    )}
    detail = json.loads(paths["beam_detailed.json"].read_text())
    summary = json.loads(paths["per_request_slo.summary.json"].read_text())
    rows = [json.loads(line) for line in paths["per_request_slo.jsonl"].read_text().splitlines()
            if line.strip()]
    config = dict(line.split("=", 1) for line in paths["slo_config.txt"].read_text().splitlines()
                  if "=" in line)
    if not rows or len(rows) != summary["requests"] or len(rows) != len(detail["ttfts"]):
        raise ValueError(f"Inconsistent or empty request population: {path}")
    keys = ("ttft_scale", "zero_load_ttft_per_token_s", "zero_load_ttft_constant_s",
            "routing_overhead_s", "slo_tpot_s")
    for key in keys:
        value = float(summary[key])
        if not np.isfinite(value) or value < 0 or not np.isclose(
                value, float(config[key]), rtol=0, atol=1e-10):
            raise ValueError(f"Invalid or inconsistent {key}: {path}")
    if float(summary["ttft_scale"]) != expected_scale:
        raise ValueError(f"Unexpected TTFT scale: {path}")
    slope = float(summary["zero_load_ttft_per_token_s"])
    intercept = float(summary["zero_load_ttft_constant_s"])
    scale = float(summary["ttft_scale"])
    saved_allowance = float(summary["routing_overhead_s"])
    session_rows = None
    if workload == "multiround_chat":
        trace_path = root / SESSION_TRACES / str(n_device) / f"{device}.csv"
        with trace_path.open(newline="") as handle:
            trace_rows = list(csv.DictReader(handle))
        session_rows = {row["RequestId"]: row for row in trace_rows}
        ids = [str(value) for value in detail["request_ids"]]
        if (len(ids) != len(rows) or len(set(ids)) != len(ids)
                or len(session_rows) != len(trace_rows) or set(ids) != set(session_rows)):
            raise ValueError(f"Session RequestId population does not match trace: {path}")
        paths["expected_session_trace"] = trace_path
    scores = []
    for index, row in enumerate(rows):
        if row["request_index"] != index:
            raise ValueError(f"Request order mismatch: {path}")
        prompt_tokens = int(row["prompt_tokens"])
        cold_budget = saved_allowance + scale * (intercept + slope * prompt_tokens)
        if not np.isclose(float(row["ttft_budget_s"]), cold_budget, rtol=0, atol=1e-8):
            raise ValueError(f"Archived TTFT budget does not match calibration: {path}, row {index}")
        scored_row = row
        if session_rows is not None:
            source = session_rows[str(detail["request_ids"][index])]
            cached = int(source["CachedTokens"])
            if (int(source["ContextTokens"]) != prompt_tokens
                    or int(source["GeneratedTokens"]) != row["target_output_tokens"]
                    or not 0 <= cached <= prompt_tokens):
                raise ValueError(f"Session length/cache metadata mismatch: {path}, row {index}")
            # Fixed expected prefix metadata, never observed cache hits. A miss
            # must not silently grant the request a more lenient deadline.
            budget = saved_allowance + scale * (intercept + slope * (prompt_tokens - cached))
            scored_row = dict(row, ttft_budget_s=budget)
        scores.append(evaluate_request(scored_row, saved_allowance, allowance, tpot))
    with paths["power_timeline_gpu_0.csv"].open(newline="") as handle:
        samples = list(csv.DictReader(handle))
    times = np.asarray([float(row["elapsed_s"]) for row in samples])
    energy = np.asarray([float(row["cumulative_energy_mj"]) / 1000 for row in samples])
    if (len(times) < 2 or not np.isfinite(times).all() or not np.isfinite(energy).all()
            or np.any(np.diff(times) <= 0) or np.any(np.diff(energy) < 0)):
        raise ValueError(f"Invalid power timeline: {path}")
    duration = float(times[-1] - times[0])
    raw_energy = float(energy[-1] - energy[0])
    counts = {
        "lower": sum(int(score["violation_lower"]) for score in scores),
        "upper": sum(int(score["violation_upper"]) for score in scores),
        "ttft": sum(int(score["ttft_violation"]) for score in scores),
        "errors": sum(int(score["error"]) for score in scores),
        "ambiguous": sum(score["violation_lower"] != score["violation_upper"] for score in scores),
    }
    return {
        "source": str(path.relative_to(root)), "device_index": device,
        "requests": len(rows), "counts": counts,
        "raw_gpu_energy_j": raw_energy,
        "scaled_gpu_energy_j": float(scale_energy(raw_energy, duration)),
        "monitor_duration_s": duration, "benchmark_duration_s": float(detail["duration"]),
        "max_power_sample_gap_s": float(np.max(np.diff(times))),
        "requests_with_missing_token_timestamps": sum(
            score["missing_token_timestamps"] > 0 for score in scores),
        "control_targets": {
            "device_index": device,
            "ttft_s": float(config["beam_global_ttft_s"]),
            "tbt_s": float(config["beam_tbt_s"]),
            "tpot_s": float(config["beam_tpot_s"]) if "beam_tpot_s" in config else None,
        },
        "saved_allowance_s": saved_allowance,
        "saved_eval_tpot_s": float(summary["slo_tpot_s"]),
        "eval_tpot_s": tpot,
        "allowance_s": allowance,
        "ttft_calibration": {key: float(summary[key]) for key in keys[:3]},
        "expected_cache_adjusted": session_rows is not None,
        "config": config,
        "source_sha256": {str(p.relative_to(root)): _sha256(p) for p in paths.values()},
    }


def _combine_runs(workload, n_device, runs, *, multiround_tpot_s=None):
    allowance, tpot, original_tpot, _ = _evaluation_settings(workload, multiround_tpot_s)
    requests = sum(run["requests"] for run in runs)
    counts = {key: sum(run["counts"][key] for run in runs) for key in runs[0]["counts"]}
    return {
        "workload": workload, "n_device": n_device,
        "violation_pct_lower": 100 * counts["lower"] / requests,
        "violation_pct_upper": 100 * counts["upper"] / requests,
        "energy_kj": n_device * float(np.mean([run["scaled_gpu_energy_j"] for run in runs])) / 1000,
        "raw_energy_kj": n_device * float(np.mean([run["raw_gpu_energy_j"] for run in runs])) / 1000,
        "sources": [run["source"] for run in runs],
        "device_indices": [run["device_index"] for run in runs],
        "requests": requests, "counts": counts,
        "eval_tpot_s": tpot, "allowance_s": allowance,
        "original_csv_tpot_s": original_tpot,
        "evaluation_matches_original_tpot": tpot == original_tpot,
        "control_targets": [run["control_targets"] for run in runs],
        "monitor_durations_s": [run["monitor_duration_s"] for run in runs],
        "benchmark_durations_s": [run["benchmark_duration_s"] for run in runs],
        "per_shard": runs,
        "source_sha256": {key: value for run in runs for key, value in run["source_sha256"].items()},
        "energy_is_extrapolated": True,
        "energy_estimator": "N * mean(scaled single-GPU full-monitor energy); unequal windows retained",
        "scoring": "Prior BEAM sensitivity convention; cumulative-deadline chunk bounds, raw client timings",
        "limitations": [
            "Not measured simultaneous deployment energy; selected shards may not represent the whole workload.",
            "Full monitor intervals include overhead and differ across runs; no 600-second normalization or idle addback.",
            "PCIe-to-SXM scaling is heuristic and does not predict SXM latency or scheduling.",
            (f"Original-system scores are unchanged; BEAM evaluation TPOT is {tpot * 1000:g} ms "
             f"versus original CSV TPOT {original_tpot * 1000:g} ms. Recorded scheduler targets are unchanged."),
            "BEAM-only TTFT allowances are sensitivity settings, not proven matched historical calibration.",
            "Multi-round uses current session traces and expected-cache budgets, not verified historical Figure 7 replay.",
        ],
    }


def collect_beam_points(root, *, coding_source="latest", multiround_tpot_s=None) -> list[dict]:
    """Read archives; optionally restore Coding or rescore only session TPOT."""
    multiround_tpot_s = _validate_multiround_tpot(multiround_tpot_s)
    root = Path(root)
    points = []
    for (workload, n_device), paths in _select_sources(root, coding_source=coding_source).items():
        runs = [_read_run(root, path, workload, multiround_tpot_s=multiround_tpot_s) for path in paths]
        points.append(_combine_runs(workload, n_device, runs, multiround_tpot_s=multiround_tpot_s))
    return points
