from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_run_batch_load_config_supports_compound_trace_key():
    result = _run_bash(
        """
        set -euo pipefail
        export RUN_BATCH_SOURCE_ONLY=1
        source ./run_batch.sh >/dev/null
        eval "$(./.venv/bin/python -m SLOsServe.batch_config --shell configs/batch/e2e_30B.jsonl)"
        trace='sharegpt_code:azure_code_23+azure_chat_23:azure_chat_23'
        load_config "$trace"
        printf 'window=%s\\n' "$window"
        printf 'policy_count=%s\\n' "${#trace_policies[@]}"
        printf 'trace=%s\\n' "$trace"
        """
    )

    assert "window=t1200:1800" in result.stdout
    assert "policy_count=5" in result.stdout
    assert "trace=sharegpt_code:azure_code_23+azure_chat_23:azure_chat_23" in result.stdout


def test_run_batch_make_run_key_preserves_simple_and_compound_traces():
    result = _run_bash(
        """
        set -euo pipefail
        export RUN_BATCH_SOURCE_ONLY=1
        source ./run_batch.sh >/dev/null
        window='t0:10'
        load_scales=('1.0')
        ttft_slo_scales=('5.0')
        slo_tpots=('0.05')
        perf_model_errs=('1.0')
        model_name='test/model'
        tensor_parallel_size='2'
        profit='constant'
        admission_mode='arrival'
        slo_routing_overhead='0.05'
        scheduling_overhead='0.003'
        routing_overhead='-1.0'
        routing_fallback_policy='asap'
        output_dir='experiments_test'
        extra_bench_args=()
        make_run_key 'azure_chat:azure_chat' 'round_robin:atfc' 2
        make_run_key 'sharegpt_code:azure_code_23+azure_chat_23:azure_chat_23' 'round_robin:atfc' 2
        """
    )

    lines = result.stdout.strip().splitlines()
    assert lines[0].startswith("emulation_0316|azure_chat|azure_chat|round_robin:atfc|")
    assert lines[1].startswith(
        "emulation_0316|sharegpt_code:azure_code_23+azure_chat_23:azure_chat_23|"
        "sharegpt_code:azure_code_23+azure_chat_23:azure_chat_23|round_robin:atfc|"
    )
