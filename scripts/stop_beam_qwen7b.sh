#!/usr/bin/env bash
set -euo pipefail

# Stop the BEAM vLLM service and release the GPU clock it may have locked.

source "$(dirname -- "${BASH_SOURCE[0]}")/beam_common.sh"

if docker ps --format '{{.Names}}' | grep -qx "$container_name"; then
  docker exec "$container_name" pkill -f 'vllm serve' || true
  for _ in $(seq 1 30); do
    docker exec "$container_name" pgrep -f 'vllm serve' >/dev/null 2>&1 || break
    sleep 2
  done
  docker exec "$container_name" nvidia-smi -i "$beam_gpu" -rgc >/dev/null || true
  echo "Stopped the BEAM vLLM service and reset GPU $beam_gpu clocks."
else
  echo "BEAM evaluator container is not running."
fi
