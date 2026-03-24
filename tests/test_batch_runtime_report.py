import json

from SLOsServe.batch_runtime_report import (
    build_config_runtime_report,
    build_directory_runtime_report,
)


def test_build_config_runtime_report_counts_runtime_components(tmp_path):
    config_path = tmp_path / "exp.json"
    config_path.write_text(
        json.dumps(
            {
                "server_clients": "0-3",
                "policies": ["round_robin:atfc", "slosserve_planner:atfc"],
                "configs": {
                    "azure_chat:azure_chat": {
                        "window": "t0:3600",
                        "load_scales": [1.0, 2.0],
                        "ttft_slo_scales": [3.0],
                        "slo_tpots": [0.05],
                        "perf_model_errs": [1.0],
                        "n_devices": [2, 8],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_config_runtime_report(config_path)
    assert report.error is None
    assert report.trace_count == 1
    assert report.runnable_policy_invocations == 2
    assert report.experiment_runs == 6
    assert report.serial_wall_hours == 6.0

    trace = report.traces[0]
    assert trace.trace == "azure_chat:azure_chat"
    assert trace.window_hours == 1.0
    assert trace.policy_count == 2
    assert trace.sweep_count == 2
    assert trace.requested_device_count == 2
    assert trace.policy_device_runs == 3
    assert trace.experiment_runs == 6
    assert trace.serial_wall_hours == 6.0


def test_build_config_runtime_report_surfaces_invalid_config(tmp_path):
    config_path = tmp_path / "broken.jsonl"
    config_path.write_text(
        json.dumps(
            {
                "policies": ["round_robin:atfc"],
                "traces": ["azure_chat:azure_chat"],
                "configs": {},
            }
        ),
        encoding="utf-8",
    )

    report = build_config_runtime_report(config_path)
    assert report.error == "Missing config entry for trace azure_chat:azure_chat."
    assert report.experiment_runs == 0
    assert report.serial_wall_hours == 0.0


def test_build_directory_runtime_report_supports_jsonl_and_skips_base(tmp_path):
    (tmp_path / "base.json").write_text(
        json.dumps({"defaults": {"policies": ["round_robin:atfc"]}}),
        encoding="utf-8",
    )
    (tmp_path / "a.json").write_text(
        json.dumps(
            {
                "extends": ["base.json"],
                "configs": {
                    "azure_chat:azure_chat": {
                        "window": "t0:1800",
                        "n_devices": [1],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "b.jsonl").write_text(
        json.dumps(
            {
                "extends": ["base.json"],
                "configs": {
                    "azure_code:azure_code": {
                        "window": "t0:3600",
                        "n_devices": [2],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_directory_runtime_report(tmp_path)
    assert [item["config_name"] for item in report["configs"]] == ["a.json", "b.jsonl"]
    assert report["total"]["config_count"] == 2
    assert report["total"]["invalid_config_count"] == 0
    assert report["total"]["runnable_policy_invocations"] == 2
    assert report["total"]["experiment_runs"] == 2
    assert report["total"]["serial_wall_hours"] == 1.5
