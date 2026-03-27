import json

from SLOsServe.batch_sanity_check import (
    build_config_sanity_report,
    build_directory_sanity_report,
)


def test_build_config_sanity_report_ok_for_simple_config(tmp_path):
    config_path = tmp_path / "ok.json"
    config_path.write_text(
        json.dumps(
            {
                "server_clients": "0-3",
                "policies": ["round_robin:atfc"],
                "extra_server_args": "--tensor_parallel_size 2",
                "configs": {
                    "azure_chat:azure_chat": {
                        "window": "t0:3600",
                        "n_devices": [2],
                        "tensor_parallel_size": 2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_config_sanity_report(config_path)
    assert report.status == "OK"
    assert report.error_count == 0
    assert report.warning_count == 0


def test_build_config_sanity_report_reports_missing_trace_config(tmp_path):
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

    report = build_config_sanity_report(config_path)
    assert report.status == "ERROR"
    assert any(
        "Missing config entry for trace azure_chat:azure_chat." in finding.message
        for finding in report.findings
    )


def test_build_config_sanity_report_warns_on_unused_config_entries(tmp_path):
    config_path = tmp_path / "unused.json"
    config_path.write_text(
        json.dumps(
            {
                "policies": ["round_robin:atfc"],
                "traces": ["azure_chat:azure_chat"],
                "configs": {
                    "azure_chat:azure_chat": {
                        "window": "t0:3600",
                        "n_devices": [1],
                    },
                    "azure_code:azure_code": {
                        "window": "t0:3600",
                        "n_devices": [1],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_config_sanity_report(config_path)
    assert report.status == "WARN"
    assert any(
        "config entries not referenced by traces: azure_code:azure_code"
        in finding.message
        for finding in report.findings
    )


def test_build_config_sanity_report_warns_when_non_partial_policies_skip_devices(tmp_path):
    config_path = tmp_path / "skip.json"
    config_path.write_text(
        json.dumps(
            {
                "server_clients": "0-3",
                "policies": ["llumnix_load:atfc", "round_robin:atfc"],
                "configs": {
                    "azure_chat:azure_chat": {
                        "window": "t0:3600",
                        "n_devices": [2, 8],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_config_sanity_report(config_path)
    assert report.status == "WARN"
    assert any(
        "non-partial policies will skip n_devices > available clients (4): 8"
        in finding.message
        for finding in report.findings
    )


def test_build_config_sanity_report_allows_canonicalized_tensor_parallel_override(tmp_path):
    config_path = tmp_path / "tp.json"
    config_path.write_text(
        json.dumps(
            {
                "server_clients": "0-3",
                "policies": ["round_robin:atfc"],
                "configs": {
                    "azure_chat:azure_chat": {
                        "window": "t0:3600",
                        "n_devices": [2],
                        "extra_args": "--tensor_parallel_size 2",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_config_sanity_report(config_path)
    assert report.status == "OK"
    assert report.error_count == 0
    assert report.warning_count == 0


def test_build_directory_sanity_report_supports_jsonl_and_skips_base(tmp_path):
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
                        "window": "t0:3600",
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
                "traces": ["azure_code:azure_code"],
                "configs": {},
            }
        ),
        encoding="utf-8",
    )

    report = build_directory_sanity_report(tmp_path)
    assert [item["config_name"] for item in report["configs"]] == ["a.json", "b.jsonl"]
    assert report["total"]["config_count"] == 2
    assert report["total"]["ok_count"] == 1
    assert report["total"]["error_count"] == 1
