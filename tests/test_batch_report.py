import json

from SLOsServe.batch_report import build_config_report, build_directory_report


def test_build_config_report_counts_experiments_and_gpu_hours(tmp_path):
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
                        "n_devices": [2, 8]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_config_report(config_path)
    assert report.available_clients == 4
    assert report.experiments == 6
    assert report.serial_wall_hours == 6.0
    assert report.useful_gpu_hours == 16.0
    assert report.ideal_parallel_wall_hours == 4.0
    assert report.sequential_node_gpu_hours == 24.0
    assert report.max_effective_gpus == 4


def test_build_directory_report_sums_config_reports(tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps(
            {
                "server_clients": "0-1",
                "policies": ["round_robin:atfc"],
                "configs": {
                    "azure_chat:azure_chat": {
                        "window": "t0:3600",
                        "n_devices": [1]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "b.json").write_text(
        json.dumps(
            {
                "server_clients": "0-1",
                "policies": ["round_robin:atfc"],
                "configs": {
                    "azure_code:azure_code": {
                        "window": "t0:1800",
                        "n_devices": [2]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_directory_report(tmp_path)
    assert len(report["configs"]) == 2
    assert report["total"]["experiments"] == 2
    assert report["total"]["serial_wall_hours"] == 1.5
    assert report["total"]["useful_gpu_hours"] == 2.0
