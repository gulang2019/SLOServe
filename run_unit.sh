#!/bin/bash

# Accept arguments:
# GPU, PORT, BASELINE, LOAD_SCALE
# (Optional): SLO_TPOTS, TTFT_SLO_SCALES, WINDOW, TRACE
# Defaults:
#   SLO_TPOTS=0.025
#   TTFT_SLO_SCALES=3.0
#   WINDOW=3979:4580
#   TRACE=sharegpt_code:azure_code_23

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <gpu> <port> <baseline> <load_scale> [slo_tpots] [ttft_slo_scales] [window] [trace] [model_name]"
  exit 1
fi

gpu="$1"
port="$2"
baseline="$3"
load_scale="$4"
slo_tpots="${5:-0.025}"
ttft_slo_scales="${6:-3.0}"
window="${7:-3979:4580}"
trace="${8:-sharegpt_code:azure_code_23}"
model_name="${9:-Qwen/Qwen2.5-7B-Instruct}"
export CUDA_VISIBLE_DEVICES="$gpu"

cd /u/gulang/workspace/SLOsServe
source /sw/external/python/anaconda3_cpu/etc/profile.d/conda.sh
conda activate slosserve

is_port_listening() {
  lsof -Pn -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

# Check if the port is already open
if is_port_listening "$port"; then
  echo "Port $port is already in use. Skipping vllm launch."
  VLLM_PID=""
else
  echo "Port $port is not in use. Starting vllm serve."
  # Start vllm serve in the background
  vllm serve $model_name \
      --port "$port" \
      --enable-chunked-prefill \
      --max_num_batched_tokens 16384 \
      --max-num-seqs 1024 \
      --long-prefill-token-threshold 16384 \
      --scheduler-cls "vllm.v1.core.sched.scheduler_adm_ctrl.SchedulerAdmCtrl" &
  VLLM_PID=$!

  # Wait for vllm to be ready (check for open port, up to 240s)
  for _ in {1..240}; do
    if is_port_listening "$port"; then
      echo "vllm is ready on port $port."
      break
    fi
    sleep 1
  done

  if ! is_port_listening "$port"; then
    echo "vllm did not become ready on port $port."
    kill $VLLM_PID
    exit 1
  fi
fi


python motivation/bench_api_server.py \
  --n_devices 1 \
  --policies "$baseline" \
  --load_scales "$load_scale" \
  --slo_tpots "$slo_tpots" \
  --ttft_slo_scales "$ttft_slo_scales" \
  --window "$window" \
  --trace "$trace" \
  --profit constant --admission_mode anytime \
  --port "$port" \
  --model_name "$model_name"

# Optionally: kill vllm after use
# kill $VLLM_PID