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
require_idle_gpu
start_container

# The patched profiler locks clocks and reads energy on the NVML device that
# CUDA_VISIBLE_DEVICES selects; PCI order keeps CUDA and NVML indices equal.
docker exec \
  -e "CUDA_DEVICE_ORDER=PCI_BUS_ID" \
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
