#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to run round_robin:sarathi on azure_chat_23.
# Override env vars to tweak defaults.
PORT="${PORT:-8000}"
CLIENTS="${CLIENTS:-0}"
LOAD_SCALE="${LOAD_SCALE:-1.0}"
SLO_TPOT="${SLO_TPOT:-0.025}"
TTFT_SCALE="${TTFT_SCALE:-3.0}"
WINDOW="${WINDOW:-3979:4580}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments_sarathi_rr}"

python motivation/bench_api_server.py --overwrite \
  --n_devices 1 \
  --policies round_robin:sarathi \
  --load_scales "$LOAD_SCALE" \
  --slo_tpots "$SLO_TPOT" \
  --ttft_slo_scales "$TTFT_SCALE" \
  --window "$WINDOW" \
  --trace azure_chat_23:azure_chat_23 \
  --profit constant --admission_mode anytime \
  --slo_routing_overhead 0.05 --scheduling_overhead 0.005 \
  --model_name "$MODEL" \
  --port "$PORT" \
  --clients "$CLIENTS" \
  --output_dir "$OUTPUT_DIR"
