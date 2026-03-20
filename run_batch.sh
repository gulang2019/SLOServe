#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_NAME="emulation_0316"

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
SERVER_CLIENTS="${SERVER_CLIENTS:-0-7}"

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

  printf '%s|%s|%s|%s|%s|%s\n' \
    "$EXPERIMENT_NAME" \
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

wait_for_server() {
  local deadline=$((SECONDS + 60))
  while ((SECONDS < deadline)); do
    if curl -fsS "http://localhost:8000/health_check" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "ERROR: server did not become ready within 60s" >&2
  return 1
}

count_clients_spec() {
  local spec="$1"
  local start
  local end
  local count

  if [[ "$spec" == *","* ]]; then
    awk -F',' 'NF { print NF }' <<< "$spec"
    return
  fi

  if [[ "$spec" == *"-"* && "$spec" != *":"* ]]; then
    start="${spec%-*}"
    end="${spec#*-}"
    echo $((end - start + 1))
    return
  fi

  if [[ "$spec" == *":"* && "$spec" != *","* ]]; then
    count="${spec#*:}"
    echo "$count"
    return
  fi

  echo 1
}

policy_supports_partial_rr() {
  local policy="$1"
  local routing="${policy%%:*}"
  [[ "$routing" == "round_robin" || "$routing" == "round_robin_retry" ]]
}

max_requested_devices() {
  local max=0
  local value
  for value in "$@"; do
    if ((value > max)); then
      max="$value"
    fi
  done
  echo "$max"
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
  local available_clients
  local max_devices
  local -a server_cmd=()
  local -a cmd=()

  available_clients="$(count_clients_spec "$SERVER_CLIENTS")"

  server_cmd=(
    python -m SLOsServe.router.api_server_ray
    --stat_window 2
    --host 0.0.0.0
    --port 8000
    --window_size 0.005
    --router slosserve
    --router_kwargs "$SERVER_ROUTER_KWARGS"
    --clients "$SERVER_CLIENTS"
    --admission_mode anytime
    --mock_connector
    --mock_engine
  )

  "${server_cmd[@]}" &
  server_pid=$!
  trap 'cleanup_server "$server_pid"' EXIT
  wait_for_server

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

      max_devices="$(max_requested_devices "${n_devices[@]}")"
      if ((max_devices > available_clients)) && ! policy_supports_partial_rr "$policy"; then
        echo "SKIP (insufficient clients for non-RR policy): $run_key"
        echo "  available_clients=$available_clients"
        echo "  max_requested_devices=$max_devices"
        log_run_cmd "SKIPPED_INCOMPATIBLE" "$run_key" "available_clients=$available_clients max_requested_devices=$max_devices"
        continue
      fi

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
        --scheduling_overhead 0.003
        --model_name Qwen/Qwen2.5-7B-Instruct
        --port 8000
        --clients "$SERVER_CLIENTS"
        --output_dir experiments_"$EXPERIMENT_NAME"
        --routing_overhead -1.0
        --routing_fallback_policy reject
      )

      printf -v cmd_str '%q ' "${cmd[@]}"

      set +e
      "${cmd[@]}"
      bench_status=$?
      set -e

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

  trap - EXIT
  cleanup_server "$server_pid"
}

POLICIES=(
  "round_robin:sarathi+"
  "llumnix_load:sarathi+"
  "round_robin:atfc"
  "slosserve_planner:atfc"
  "round_robin-disagg:sarathi+"
  "llumnix_load-disagg:sarathi+"
  "round_robin-disagg:atfc"
  "slosserve_disagg_planner:atfc"
  # "round_robin-disagg:atfc"
)

TRACES=(
  azure_chat:azure_chat
  azure_code_23:azure_code_23
  sharegpt_code:azure_code_23
  azure_chat_23:azure_chat_23
  sharegpt_chat:azure_chat_23
  sharegpt_code:azure_code
  azure_code:azure_code
  sharegpt_chat:azure_chat
  reasoning:azure_chat_23
)

declare -A configs=(
  ["azure_code_23:azure_code_23"]="t1200:1800 1.0 2 4 8 12 16 32"
  ["sharegpt_code:azure_code_23"]="t1200:1800 1.0 2 4 6 8"
  ["azure_chat_23:azure_chat_23"]="t1200:1800 1.0 2 4 6 8"
  ["sharegpt_chat:azure_chat_23"]="t1200:1800 1.0 2 4 6 8"
  ["azure_code:azure_code"]="t1200:1800 1.0 4 8 12 16 32"
  ["sharegpt_code:azure_code"]="t1200:1800 1.0 4 8 12 16 32"
  ["azure_chat:azure_chat"]="t1200:1800 1.0 4 8 12 16 32"
  ["sharegpt_chat:azure_chat"]="t1200:1800 1.0 4 8 12 16 32"
  ["reasoning:azure_chat_23"]="t1200:1800 1.0 4 8 12 16 32"
)

run_suite
