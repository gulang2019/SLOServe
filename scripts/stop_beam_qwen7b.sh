#!/usr/bin/env bash
set -euo pipefail

container_name="vllm_evaluator_env"

if docker ps --format '{{.Names}}' | grep -qx "$container_name"; then
  docker exec "$container_name" pkill -f 'vllm serve' || true
  echo "Stopped the BEAM vLLM service."
else
  echo "BEAM evaluator container is not running."
fi
