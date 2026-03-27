import json

import pytest

from SLOsServe.batch_config import (
    combine_batch_configs,
    load_batch_config,
    normalize_batch_config,
    render_bash_assignments,
)


def test_load_batch_config_merges_extends(tmp_path):
    base_path = tmp_path / "base.json"
    child_path = tmp_path / "child.json"

    base_path.write_text(
        json.dumps(
            {
                "experiment_name": "base",
                "server_clients": "0-7",
                "defaults": {
                    "policies": ["round_robin:atfc"],
                    "model_name": "base/model",
                },
                "server_router_kwargs": {
                    "device_mem": 1024,
                    "block_size": 16,
                    "nested": {"keep": 1, "override": 1},
                },
                "configs": {
                    "azure_chat:azure_chat": {
                        "window": "t0:10",
                        "load_scale": 1.0,
                        "n_devices": [2, 4],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    child_path.write_text(
        json.dumps(
            {
                "extends": ["base.json"],
                "experiment_name": "child",
                "server_router_kwargs": {
                    "nested": {"override": 2},
                    "max_decode_length": 500,
                },
                "policies": ["slosserve_planner:atfc"],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_batch_config(child_path)
    assert loaded["experiment_name"] == "child"
    assert loaded["server_clients"] == "0-7"
    assert loaded["policies"] == ["slosserve_planner:atfc"]
    assert loaded["server_router_kwargs"]["device_mem"] == 1024
    assert loaded["server_router_kwargs"]["nested"] == {"keep": 1, "override": 2}
    assert loaded["server_router_kwargs"]["max_decode_length"] == 500


def test_normalize_batch_config_merges_defaults_and_aliases():
    normalized = normalize_batch_config(
        {
            "experiment_name": "exp_a",
            "policies": ["round_robin:atfc"],
            "defaults": {
                "model_name": "Qwen/Test",
                "load_scales": [1.0],
                "perf_model_errs": [1.0],
            },
            "configs": {
                "sharegpt_code:azure_code_23": {
                    "window": "t1200:1800",
                    "n_devices": [2, 4, 8],
                    "slo_ttft": [2, 3, 5],
                    "slo_tpot": [0.025, 0.05],
                    "per_model_errs": [0.8, 1.2],
                }
            },
        }
    )

    spec = normalized["trace_specs"]["sharegpt_code:azure_code_23"]
    assert spec["window"] == "t1200:1800"
    assert spec["n_devices"] == ["2", "4", "8"]
    assert spec["load_scales"] == ["1.0"]
    assert spec["ttft_slo_scales"] == ["2", "3", "5"]
    assert spec["slo_tpots"] == ["0.025", "0.05"]
    assert spec["perf_model_errs"] == ["0.8", "1.2"]
    assert spec["model_name"] == "Qwen/Test"
    assert spec["policies"] == ["round_robin:atfc"]


def test_render_bash_assignments_emits_trace_arrays():
    rendered = render_bash_assignments(
        {
            "experiment_name": "exp_a",
            "server_clients": "0-3",
            "server_router_kwargs": {"device_mem": 1024, "block_size": 16},
            "policies": ["round_robin:atfc", "slosserve_planner:atfc"],
            "traces": ["azure_chat:azure_chat"],
            "configs": {
                "azure_chat:azure_chat": {
                    "window": "t10:20",
                    "load_scales": [2.0, 3.0],
                    "n_devices": [2, 4],
                    "slo_tpot": [0.05, 0.1],
                }
            },
        }
    )

    assert "BATCH_CONFIG_LOADED=1" in rendered
    assert "EXPERIMENT_NAME=exp_a" in rendered
    assert "SERVER_CLIENTS=0-3" in rendered
    assert "SERVER_ROUTER_KWARGS=" in rendered
    assert "TRACES=(" in rendered
    assert "declare -gA TRACE_SERVER_ARGS_SHELL=(" in rendered
    assert "declare -gA TRACE_SERVER_ROUTER_KWARGS=(" in rendered
    assert "declare -gA TRACE_POLICIES=(" in rendered
    assert "declare -gA TRACE_EXTRA_ARGS_SHELL=(" in rendered
    assert "declare -gA TRACE_LOAD_SCALES=(" in rendered
    assert "declare -gA TRACE_SLO_TPOTS=(" in rendered
    assert "declare -gA TRACE_WINDOW=(" in rendered
    assert "azure_chat:azure_chat" in rendered
    assert "2.0 3.0" in rendered
    assert "0.05 0.1" in rendered


def test_normalize_batch_config_supports_extra_args_string_alias():
    normalized = normalize_batch_config(
        {
            "policies": ["round_robin:atfc"],
            "extra_args": "--enable_session_replay --session_pause_s 5.0",
            "configs": {
                "azure_chat:azure_chat": {
                    "window": "t10:20",
                    "n_devices": [2],
                }
            },
        }
    )

    spec = normalized["trace_specs"]["azure_chat:azure_chat"]
    assert spec["extra_args"] == [
        "--enable_session_replay",
        "--session_pause_s",
        "5.0",
    ]


def test_normalize_batch_config_appends_trace_extra_args_and_extra_policies():
    normalized = normalize_batch_config(
        {
            "policies": ["round_robin:atfc", "slosserve_planner:atfc"],
            "extra_args": "--enable_prefix_cache",
            "configs": {
                "azure_chat:azure_chat": {
                    "window": "t10:20",
                    "n_devices": [2],
                    "extra_args": "--enable_session_replay --session_pause_s 5.0",
                    "extra_policies": [
                        "round_robin_session:atfc",
                        "round_robin:atfc",
                    ],
                }
            },
        }
    )

    spec = normalized["trace_specs"]["azure_chat:azure_chat"]
    assert spec["extra_args"] == [
        "--enable_prefix_cache",
        "--enable_session_replay",
        "--session_pause_s",
        "5.0",
    ]
    assert spec["policies"] == [
        "round_robin:atfc",
        "slosserve_planner:atfc",
        "round_robin_session:atfc",
    ]


def test_normalize_batch_config_extracts_server_launch_overrides():
    normalized = normalize_batch_config(
        {
            "server_router_kwargs": {
                "device_mem": 1024,
                "block_size": 16,
                "model_name": "stale/model",
            },
            "extra_server_args": (
                "--tensor_parallel_size 2 "
                "--worker_env NCCL_IB_DISABLE=1"
            ),
            "extra_args": "--model_name Qwen/Test-30B",
            "policies": ["round_robin:atfc"],
            "configs": {
                "azure_chat:azure_chat": {
                    "window": "t10:20",
                    "n_devices": [2],
                }
            },
        }
    )

    spec = normalized["trace_specs"]["azure_chat:azure_chat"]
    assert spec["model_name"] == "Qwen/Test-30B"
    assert spec["tensor_parallel_size"] == "2"
    assert "extra_args" not in spec
    assert normalized["extra_server_args"] == [
        "--worker_env",
        "NCCL_IB_DISABLE=1",
    ]
    assert normalized["trace_server_args"]["azure_chat:azure_chat"] == [
        "--worker_env",
        "NCCL_IB_DISABLE=1",
        "--model_name",
        "Qwen/Test-30B",
        "--tensor_parallel_size",
        "2",
    ]
    assert json.loads(
        normalized["trace_server_router_kwargs"]["azure_chat:azure_chat"]
    ) == {
        "block_size": 16,
        "device_mem": 1024,
        "model_name": "Qwen/Test-30B",
    }


def test_normalize_batch_config_promotes_reserved_server_args_into_trace_defaults():
    normalized = normalize_batch_config(
        {
            "tensor_parallel_size": 1,
            "extra_server_args": "--tensor_parallel_size 2",
            "policies": ["round_robin:atfc"],
            "configs": {
                "azure_chat:azure_chat": {
                    "window": "t10:20",
                    "n_devices": [2],
                }
            },
        }
    )

    spec = normalized["trace_specs"]["azure_chat:azure_chat"]
    assert spec["tensor_parallel_size"] == "2"
    assert "extra_server_args" not in normalized
    assert normalized["trace_server_args"]["azure_chat:azure_chat"] == [
        "--model_name",
        "Qwen/Qwen2.5-7B-Instruct",
        "--tensor_parallel_size",
        "2",
    ]


def test_normalize_batch_config_rejects_missing_required_fields():
    with pytest.raises(ValueError):
        normalize_batch_config(
            {
                "policies": ["round_robin:atfc"],
                "configs": {
                    "azure_chat:azure_chat": {
                        "window": "t10:20",
                        "load_scale": 2.0,
                    }
                },
            }
        )


def test_combine_batch_configs_normalizes_directory(tmp_path):
    (tmp_path / "base.json").write_text(
        json.dumps(
            {
                "defaults": {
                    "policies": ["round_robin:atfc"],
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "exp.json").write_text(
        json.dumps(
            {
                "extends": ["base.json"],
                "configs": {
                    "azure_chat:azure_chat": {
                        "window": "t10:20",
                        "n_devices": [2],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    combined = combine_batch_configs(tmp_path)
    assert "base.json" in combined
    assert "exp.json" in combined
    assert combined["exp.json"]["trace_specs"]["azure_chat:azure_chat"]["policies"] == [
        "round_robin:atfc"
    ]
