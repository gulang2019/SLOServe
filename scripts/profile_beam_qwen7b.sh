#!/usr/bin/env bash
set -euo pipefail

# Generate the profile required by BEAM's S1/S2 scheduler for the local 7B
# service. This is intentionally a separate, long-running step;
# serve_beam_qwen7b.sh runs it automatically when the profile is missing.
# BEAM_PROFILE_NUM_CLOCKS=N profiles only N evenly spaced clocks (smoke run).

source "$(dirname -- "${BASH_SOURCE[0]}")/beam_common.sh"
beam_memory_utilization="${BEAM_GPU_MEMORY_UTILIZATION:-0.90}"

require_docker
require_gpu
# BEAM's profiler locks clocks and reads energy on NVML device 0 regardless
# of CUDA_VISIBLE_DEVICES, so profiling another GPU would measure the wrong one.
if [[ "$beam_gpu" != "0" ]]; then
  echo "Profiling supports only BEAM_GPU=0 (BEAM's profiler measures NVML device 0)." >&2
  exit 1
fi
start_container

if docker exec "$container_name" pgrep -f 'vllm serve' >/dev/null 2>&1; then
  echo "A BEAM vLLM server is running in $container_name and holds GPU memory;" >&2
  echo "stop it with scripts/stop_beam_qwen7b.sh before profiling." >&2
  exit 1
fi

docker exec \
  -e "CUDA_VISIBLE_DEVICES=$beam_gpu" \
  -e "HF_TOKEN=${HF_TOKEN:-}" \
  -e "BEAM_PROFILE_NUM_CLOCKS=${BEAM_PROFILE_NUM_CLOCKS:-0}" \
  "$container_name" \
  bash -lc "cd /workspace/benchmarks/energy && ./run_offline_profile.sh --model '$beam_model' --tp 1 --pp 1 --gpu-memory-utilization '$beam_memory_utilization'"

profile="$(profile_path)"
if [[ ! -s "$profile" ]]; then
  echo "BEAM profiling did not produce $profile" >&2
  exit 1
fi
echo "Profile written: $profile"
