#!/usr/bin/env bash
set -euo pipefail

# Run a fixed round_robin:sarathi benchmark on azure_chat_23.
export PYTHONPATH="${PYTHONPATH:-.}:."
python motivation/bench_api_server.py --overwrite \
  --n_devices 1 \
  --policies round_robin:sarathi \
  --load_scales 1.0 \
  --slo_tpots 0.025 \
  --ttft_slo_scales 3.0 \
  --window 3979:4580 \
  --trace azure_chat_23:azure_chat_23 \
  --profit constant --admission_mode anytime \
  --slo_routing_overhead 0.05 --scheduling_overhead 0.005 \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --clients 0 \
  --output_dir experiments_sarathi_rr
