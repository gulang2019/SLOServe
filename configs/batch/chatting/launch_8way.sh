#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PORT_BASE="${PORT_BASE:-8100}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/batch_logs/single_server}"

mkdir -p "$LOG_ROOT"

pids=()
for gpu in {0..7}; do
  config_path="$SCRIPT_DIR/gpu${gpu}.json"
  log_dir="$LOG_ROOT/gpu${gpu}"
  mkdir -p "$log_dir"

  echo "Launching gpu${gpu}: config=$config_path port=$((PORT_BASE + gpu)) log_dir=$log_dir"

  CUDA_VISIBLE_DEVICES="$gpu" \
  RUN_BATCH_PORT="$((PORT_BASE + gpu))" \
  SUCCEEDED_LOG="$log_dir/succeeded_runs.log" \
  FAILED_LOG="$log_dir/failed_runs.log" \
  RUN_LOG="$log_dir/all_runs.log" \
  "$REPO_ROOT/run_batch.sh" "$config_path" \
    >"$log_dir/stdout.log" 2>&1 &

  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
