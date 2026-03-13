#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
  echo "ERROR: Missing virtualenv activate script at $SCRIPT_DIR/.venv/bin/activate" >&2
  exit 1
fi

# Load the repo virtualenv so tmux/nohup runs use the expected interpreter.
source "$SCRIPT_DIR/.venv/bin/activate"

export PYTHONPATH="$SCRIPT_DIR"

SUCCEEDED_LOG="${SUCCEEDED_LOG:-$SCRIPT_DIR/succeeded_runs.log}"
FAILED_LOG="${FAILED_LOG:-$SCRIPT_DIR/failed_runs.log}"
RUN_LOG="${RUN_LOG:-$SCRIPT_DIR/all_runs.log}"
SERVER_ROUTER_KWARGS='{"device_mem":1248576, "block_size":16, "model_name":"Qwen/Qwen2.5-7B-Instruct", "tpot":0.05, "scheduling_overhead":0.005, "max_decode_length":500, "is_pd_disagg":0, "n_prefill_per_group":1, "max_decode_bs":16, "enable_rerouting":0}'

touch "$SUCCEEDED_LOG" "$FAILED_LOG" "$RUN_LOG"

load_config() {
  local length_trace="$1"
  local arrival_trace="$2"
  local key="${length_trace}:${arrival_trace}"
  local cfg
  local -a cfg_parts=()

  if [[ -z "${configs[$key]:-}" ]]; then
    echo "ERROR: Missing config for ${key}" >&2
    exit 1
  fi

  cfg="${configs[$key]}"
  read -r -a cfg_parts <<< "$cfg"

  if ((${#cfg_parts[@]} < 3)); then
    echo "ERROR: Invalid config for ${key}: ${cfg}" >&2
    exit 1
  fi

  window="${cfg_parts[0]}"
  load_scale="${cfg_parts[1]}"
  n_devices=("${cfg_parts[@]:2}")
}

make_run_key() {
  local length_trace="$1"
  local arrival_trace="$2"
  local policy="$3"
  shift 3
  local devices=("$@")

  printf '%s|%s|%s|%s|%s\n' \
    "$length_trace" \
    "$arrival_trace" \
    "$policy" \
    "$window" \
    "${devices[*]}"
}

already_succeeded() {
  local run_key="$1"
  grep -Fqx "$run_key" "$SUCCEEDED_LOG"
}

log_run_cmd() {
  local run_status="$1"
  local run_key="$2"
  local cmd="$3"
  local now

  now="$(date '+%Y-%m-%d %H:%M:%S')"

  {
    echo "[$now] STATUS=$run_status"
    echo "KEY=$run_key"
    echo "CMD=$cmd"
    echo
  } >> "$RUN_LOG"
}

cleanup_server() {
  local server_pid="$1"
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}

run_suite() {
  local trace
  local length_trace
  local arrival_trace
  local policy
  local run_key
  local server_pid
  local bench_status
  local cmd_str
  local -a server_cmd=()
  local -a cmd=()

  for trace in "${TRACES[@]}"; do
    length_trace="${trace%%:*}"
    arrival_trace="${trace##*:}"

    load_config "$length_trace" "$arrival_trace"

    for policy in "${POLICIES[@]}"; do
      run_key="$(make_run_key "$length_trace" "$arrival_trace" "$policy" "${n_devices[@]}")"

      if already_succeeded "$run_key"; then
        echo "SKIP (already succeeded): $run_key"
        log_run_cmd "SKIPPED" "$run_key" "<already succeeded>"
        continue
      fi

      echo "RUNNING: ${length_trace}:${arrival_trace}"
      echo "  policy=$policy"
      echo "  window=$window"
      echo "  load_scale=$load_scale"
      echo "  n_devices=${n_devices[*]}"

      server_cmd=(
        python -m SLOsServe.router.api_server_ray
        --stat_window 2
        --host 0.0.0.0
        --port 8000
        --window_size 0.005
        --router slosserve
        --router_kwargs "$SERVER_ROUTER_KWARGS"
        --clients 0,1
        --admission_mode anytime
        --mock_connector
        --mock_engine
      )

      "${server_cmd[@]}" &
      server_pid=$!
      trap 'cleanup_server "$server_pid"' EXIT

      sleep 30

      cmd=(
        python motivation/bench_api_server.py
        --overwrite
        --n_devices "${n_devices[@]}"
        --policies "$policy"
        --load_scales "$load_scale"
        --slo_tpots 0.05
        --ttft_slo_scales 5.0
        --window "$window"
        --trace "${length_trace}:${arrival_trace}"
        --profit constant
        --admission_mode arrival
        --slo_routing_overhead 0.05
        --scheduling_overhead 0.005
        --model_name Qwen/Qwen2.5-7B-Instruct
        --port 8000
        --clients 0-31
        --output_dir experiments_emulation_0313
        --routing_overhead -1.0
      )

      printf -v cmd_str '%q ' "${cmd[@]}"

      set +e
      "${cmd[@]}"
      bench_status=$?
      set -e

      trap - EXIT
      cleanup_server "$server_pid"

      if ((bench_status == 0)); then
        echo "SUCCESS: $run_key"
        echo "$run_key" >> "$SUCCEEDED_LOG"
        log_run_cmd "SUCCESS" "$run_key" "$cmd_str"
      else
        echo "FAILED: $run_key"
        echo "$run_key" >> "$FAILED_LOG"
        log_run_cmd "FAILED" "$run_key" "$cmd_str"
      fi
    done
  done
}

POLICIES=(
  "slosserve_planner:atfc"
  "round_robin:atfc"
  "round_robin:sarathi+"
)

TRACES=(
  sharegpt_code:azure_code_23
  sharegpt_chat:azure_chat_23
  azure_chat_23:azure_chat_23
  sharegpt_code:azure_code
  azure_code:azure_code
  sharegpt_chat:azure_chat
  azure_chat:azure_chat
)

declare -A configs=(
  # ["azure_code_23:azure_code_23"]="t0:600 1.0 4 8 12 16 32"
  ["sharegpt_code:azure_code_23"]="t0:600 1.0 2 4 6 8"
  ["azure_chat_23:azure_chat_23"]="t0:600 1.0 2 4 6 8"
  ["sharegpt_chat:azure_chat_23"]="t0:600 1.0 2 4 6 8"
  ["azure_code:azure_code"]="t0:600 1.0 4 8 12 16 32"
  ["sharegpt_code:azure_code"]="t0:600 1.0 4 8 12 16 32"
  ["azure_chat:azure_chat"]="t0:600 1.0 4 8 12 16 32"
  ["sharegpt_chat:azure_chat"]="t0:600 1.0 4 8 12 16 32"
)

run_suite
