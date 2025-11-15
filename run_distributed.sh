#!/bin/bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  cat <<'USAGE'
Usage: run_distributed.sh <gpu_csv> <router_port> <n_device> <baseline> <load_scale>                            [slo_tpots] [ttft_slo_scales] [window] [trace] [model_name]
USAGE
  exit 1
fi

GPU_CSV="$1"
ROUTER_PORT="$2"
N_DEVICE="$3"
BASELINE="$4"
LOAD_SCALE="$5"
SLO_TPOTS="${6:-0.025}"
TTFT_SLO="${7:-3.0}"
WINDOW="${8:-3979:4580}"
TRACE="${9:-sharegpt_code:azure_code_23}"
MODEL_NAME="${10:-Qwen/Qwen2.5-7B-Instruct}"

export CUDA_VISIBLE_DEVICES="$GPU_CSV"

ROOT="${PID_DIR:-/tmp/slosserve_routers}"
mkdir -p "$ROOT"
PID_FILE="$ROOT/router_${ROUTER_PORT}.pid"
LOG_FILE="router_${ROUTER_PORT}.log"

stop_router() {
  if [[ -f "$PID_FILE" ]]; then
    read -r pid _ < "$PID_FILE"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "Stopping router pid $pid (port $ROUTER_PORT)"
      kill "$pid" || true
      wait "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  fi
}

cleanup() {
  stop_router
}
trap cleanup EXIT INT TERM

stop_router

first_gpu=$(echo "$GPU_CSV" | cut -d',' -f1)
echo "first_gpu: $first_gpu"
export UCX_NET_DEVICES=all
export VLLM_NIXL_SIDE_CHANNEL_PORT=$((5600 + (first_gpu * 10)))
python -m SLOsServe.router.api_server_ray \
  --host 0.0.0.0 \
  --port "$ROUTER_PORT" \
  --window_size 0.001 \
  --router round_robin \
  --router_kwargs '{"max_decode_batch_size":300}' \
  --clients "0" \
  --admission_mode anytime \
  --model_name "$MODEL_NAME" \
  --mock_connector  --stat_window 2 \
  >"$LOG_FILE" 2>&1 &
ROUTER_PID=$!
echo "$ROUTER_PID $GPU_CSV" > "$PID_FILE"

for i in {1..600}; do
  if curl --silent --fail --max-time 5 "http://127.0.0.1:${ROUTER_PORT}/docs" >/dev/null 2>&1; then
    echo "Router is ready on port $ROUTER_PORT after $i seconds"
    break
  fi
  sleep 1
done

python motivation/bench_api_server.py \
  --port "$ROUTER_PORT" \
  --model_name "$MODEL_NAME" \
  --n_devices "$N_DEVICE" \
  --policies "$BASELINE" \
  --load_scales "$LOAD_SCALE" \
  --slo_tpots "$SLO_TPOTS" \
  --ttft_slo_scales "$TTFT_SLO" \
  --window "$WINDOW" \
  --trace "$TRACE" \
  --profit constant \
  --admission_mode anytime \
  --overwrite --slo_routing_overhead 0.02\
  --clients "$GPU_CSV"
