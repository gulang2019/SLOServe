#!/usr/bin/env bash
set -euo pipefail

# Stop the BEAM vLLM service on BEAM_PORT and release the clock it may have
# locked on BEAM_GPU. Other BEAM servers in the container keep running.

source "$(dirname -- "${BASH_SOURCE[0]}")/beam_common.sh"

if docker ps --format '{{.Names}}' | grep -qx "$container_name"; then
  docker exec "$container_name" pkill -f "$beam_server_pattern" || true
  for _ in $(seq 1 30); do
    docker exec "$container_name" pgrep -f "$beam_server_pattern" >/dev/null 2>&1 || break
    sleep 2
  done
  docker exec "$container_name" nvidia-smi -i "$beam_gpu" -rgc >/dev/null || true
  echo "Stopped the BEAM vLLM service on port $beam_port and reset GPU $beam_gpu clocks."
else
  echo "BEAM evaluator container is not running."
fi
