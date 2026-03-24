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

BATCH_CONFIG_PATH="${BATCH_CONFIG_PATH:-${1:-}}"
RUN_BATCH_PORT="${RUN_BATCH_PORT:-8000}"

SUCCEEDED_LOG="${SUCCEEDED_LOG:-$SCRIPT_DIR/succeeded_runs.log}"
FAILED_LOG="${FAILED_LOG:-$SCRIPT_DIR/failed_runs.log}"
RUN_LOG="${RUN_LOG:-$SCRIPT_DIR/all_runs.log}"
SERVER_ROUTER_KWARGS='{"device_mem":1248576, "block_size":16, "model_name":"Qwen/Qwen2.5-7B-Instruct", "tpot":0.05, "scheduling_overhead":0.005, "max_decode_length":500, "is_pd_disagg":0, "n_prefill_per_group":1, "max_decode_bs":16, "enable_rerouting":0}'
SERVER_EXTRA_ARGS_SHELL="${SERVER_EXTRA_ARGS_SHELL:-}"
SERVER_CLIENTS="${SERVER_CLIENTS:-0-7}"

touch "$SUCCEEDED_LOG" "$FAILED_LOG" "$RUN_LOG"

BATCH_CONFIG_LOADED=0

DEFAULT_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
DEFAULT_PROFIT="constant"
DEFAULT_ADMISSION_MODE="arrival"
DEFAULT_SLO_ROUTING_OVERHEAD="0.05"
DEFAULT_SCHEDULING_OVERHEAD="0.003"
DEFAULT_ROUTING_OVERHEAD="-1.0"
DEFAULT_ROUTING_FALLBACK_POLICY="asap"
DEFAULT_TENSOR_PARALLEL_SIZE="1"
DEFAULT_LOAD_SCALES=("1.0")
DEFAULT_TTFT_SLO_SCALES=("5.0")
DEFAULT_SLO_TPOTS=("0.05")
DEFAULT_PERF_MODEL_ERRS=("1.0")

load_config() {
  local length_trace="$1"
  local arrival_trace="$2"
  local key="${length_trace}:${arrival_trace}"
  local cfg
  local -a cfg_parts=()

  server_router_kwargs="$SERVER_ROUTER_KWARGS"

  if [[ "$BATCH_CONFIG_LOADED" == "1" ]]; then
    if [[ -z "${TRACE_WINDOW[$key]:-}" ]]; then
      echo "ERROR: Missing normalized trace config for ${key}" >&2
      exit 1
    fi

    window="${TRACE_WINDOW[$key]}"
    read -r -a n_devices <<< "${TRACE_N_DEVICES[$key]}"
    read -r -a load_scales <<< "${TRACE_LOAD_SCALES[$key]}"
    read -r -a ttft_slo_scales <<< "${TRACE_TTFT_SLO_SCALES[$key]}"
    read -r -a slo_tpots <<< "${TRACE_SLO_TPOTS[$key]}"
    read -r -a perf_model_errs <<< "${TRACE_PERF_MODEL_ERRS[$key]}"
    read -r -a trace_policies <<< "${TRACE_POLICIES[$key]}"
    extra_bench_args=()
    if [[ -n "${TRACE_EXTRA_ARGS_SHELL[$key]:-}" ]]; then
      eval "extra_bench_args=(${TRACE_EXTRA_ARGS_SHELL[$key]})"
    fi

    model_name="${TRACE_MODEL_NAME[$key]}"
    profit="${TRACE_PROFIT[$key]}"
    admission_mode="${TRACE_ADMISSION_MODE[$key]}"
    slo_routing_overhead="${TRACE_SLO_ROUTING_OVERHEAD[$key]}"
    scheduling_overhead="${TRACE_SCHEDULING_OVERHEAD[$key]}"
    routing_overhead="${TRACE_ROUTING_OVERHEAD[$key]}"
    routing_fallback_policy="${TRACE_ROUTING_FALLBACK_POLICY[$key]}"
    tensor_parallel_size="${TRACE_TENSOR_PARALLEL_SIZE[$key]}"
    output_dir="${TRACE_OUTPUT_DIR[$key]:-experiments_${EXPERIMENT_NAME}}"
    server_router_kwargs="${TRACE_SERVER_ROUTER_KWARGS[$key]:-$SERVER_ROUTER_KWARGS}"
    server_extra_args=()
    if [[ -n "${TRACE_SERVER_ARGS_SHELL[$key]:-}" ]]; then
      eval "server_extra_args=(${TRACE_SERVER_ARGS_SHELL[$key]})"
    else
      if [[ -n "${SERVER_EXTRA_ARGS_SHELL:-}" ]]; then
        eval "server_extra_args=(${SERVER_EXTRA_ARGS_SHELL})"
      fi
      server_extra_args+=(--model_name "$model_name" --tensor_parallel_size "$tensor_parallel_size")
    fi
    return
  fi

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
  load_scales=("${cfg_parts[1]}")
  n_devices=("${cfg_parts[@]:2}")
  ttft_slo_scales=("${DEFAULT_TTFT_SLO_SCALES[@]}")
  slo_tpots=("${DEFAULT_SLO_TPOTS[@]}")
  perf_model_errs=("${DEFAULT_PERF_MODEL_ERRS[@]}")
  trace_policies=("${POLICIES[@]}")
  extra_bench_args=()
  model_name="$DEFAULT_MODEL_NAME"
  profit="$DEFAULT_PROFIT"
  admission_mode="$DEFAULT_ADMISSION_MODE"
  slo_routing_overhead="$DEFAULT_SLO_ROUTING_OVERHEAD"
  scheduling_overhead="$DEFAULT_SCHEDULING_OVERHEAD"
  routing_overhead="$DEFAULT_ROUTING_OVERHEAD"
  routing_fallback_policy="$DEFAULT_ROUTING_FALLBACK_POLICY"
  tensor_parallel_size="$DEFAULT_TENSOR_PARALLEL_SIZE"
  output_dir="experiments_${EXPERIMENT_NAME}"
  server_extra_args=()
  if [[ -n "${SERVER_EXTRA_ARGS_SHELL:-}" ]]; then
    eval "server_extra_args=(${SERVER_EXTRA_ARGS_SHELL})"
  fi
  server_extra_args+=(--model_name "$model_name" --tensor_parallel_size "$tensor_parallel_size")
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
    "$window|load_scales=${load_scales[*]}|ttft_slo_scales=${ttft_slo_scales[*]}|slo_tpots=${slo_tpots[*]}|perf_model_errs=${perf_model_errs[*]}|model_name=${model_name}|tensor_parallel_size=${tensor_parallel_size}|profit=${profit}|admission_mode=${admission_mode}|slo_routing_overhead=${slo_routing_overhead}|scheduling_overhead=${scheduling_overhead}|routing_overhead=${routing_overhead}|routing_fallback_policy=${routing_fallback_policy}|output_dir=${output_dir}|extra_args=${extra_bench_args[*]}" \
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
  local server_pid="${1:-}"
  if [[ -z "$server_pid" ]]; then
    return 0
  fi
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}

wait_for_server() {
  local deadline=$((SECONDS + 3000))
  while ((SECONDS < deadline)); do
    if curl -fsS "http://localhost:${RUN_BATCH_PORT}/health_check" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "ERROR: server did not become ready within 3000s" >&2
  return 1
}

ensure_server_ready() {
  if [[ -z "${server_pid:-}" ]]; then
    start_server
    return 0
  fi

  if curl -fsS "http://localhost:${RUN_BATCH_PORT}/health_check" >/dev/null 2>&1; then
    return 0
  fi

  echo "RESTARTING UNHEALTHY SERVER on port ${RUN_BATCH_PORT}"
  cleanup_server "$server_pid"
  server_pid=""
  start_server
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

filter_compatible_devices() {
  local available_clients="$1"
  shift
  local value
  for value in "$@"; do
    if ((value <= available_clients)); then
      printf '%s\n' "$value"
    fi
  done
}

compute_server_signature() {
  local server_args_key
  printf -v server_args_key '%q ' "${server_extra_args[@]}"
  printf '%s|%s|%s|%s\n' \
    "$RUN_BATCH_PORT" \
    "$SERVER_CLIENTS" \
    "$server_router_kwargs" \
    "$server_args_key"
}

start_server() {
  local -a server_cmd=()
  local server_cmd_str

  server_cmd=(
    python -m SLOsServe.router.api_server_ray
    --stat_window 2
    --host 0.0.0.0
    --port "$RUN_BATCH_PORT"
    --window_size 0.005
    --router slosserve
    --router_kwargs "$server_router_kwargs"
    --clients "$SERVER_CLIENTS"
    --admission_mode anytime
    --mock_connector
    "${server_extra_args[@]}"
  )

  printf -v server_cmd_str '%q ' "${server_cmd[@]}"
  echo "STARTING SERVER: $server_cmd_str"
  log_run_cmd "SERVER_START" "$current_server_signature" "$server_cmd_str"

  "${server_cmd[@]}" &
  server_pid=$!
  wait_for_server
}

run_suite() {
  local trace
  local length_trace
  local arrival_trace
  local policy
  local run_key
  local server_pid=""
  local current_server_signature=""
  local next_server_signature
  local bench_status
  local cmd_str
  local available_clients
  local -a cmd=()
  local -a run_devices=()
  local -a trace_policies=()
  local -a extra_bench_args=()
  local -a server_extra_args=()

  available_clients="$(count_clients_spec "$SERVER_CLIENTS")"
  trap 'cleanup_server "$server_pid"' EXIT

  for trace in "${TRACES[@]}"; do
    length_trace="${trace%%:*}"
    arrival_trace="${trace##*:}"

    load_config "$length_trace" "$arrival_trace"
    next_server_signature="$(compute_server_signature)"
    if [[ -z "$server_pid" || "$next_server_signature" != "$current_server_signature" ]]; then
      if [[ -n "$server_pid" ]]; then
        echo "RESTARTING SERVER for ${length_trace}:${arrival_trace}"
        cleanup_server "$server_pid"
        server_pid=""
      fi
      current_server_signature="$next_server_signature"
      start_server
    fi

    for policy in "${trace_policies[@]}"; do
      run_devices=("${n_devices[@]}")
      if ! policy_supports_partial_rr "$policy"; then
        mapfile -t run_devices < <(filter_compatible_devices "$available_clients" "${n_devices[@]}")
        if ((${#run_devices[@]} == 0)); then
          run_key="$(make_run_key "$length_trace" "$arrival_trace" "$policy" "${n_devices[@]}")"
          echo "SKIP (no compatible client counts for non-RR policy): $run_key"
          echo "  available_clients=$available_clients"
          echo "  requested_n_devices=${n_devices[*]}"
          log_run_cmd "SKIPPED_INCOMPATIBLE" "$run_key" "available_clients=$available_clients requested_n_devices=${n_devices[*]}"
          continue
        fi
      fi

      run_key="$(make_run_key "$length_trace" "$arrival_trace" "$policy" "${run_devices[@]}")"

      if already_succeeded "$run_key"; then
        echo "SKIP (already succeeded): $run_key"
        log_run_cmd "SKIPPED" "$run_key" "<already succeeded>"
        continue
      fi

      echo "RUNNING: ${length_trace}:${arrival_trace}"
      echo "  policy=$policy"
      echo "  window=$window"
      echo "  load_scales=${load_scales[*]}"
      echo "  n_devices=${run_devices[*]}"
      echo "  ttft_slo_scales=${ttft_slo_scales[*]}"
      echo "  slo_tpots=${slo_tpots[*]}"
      echo "  perf_model_errs=${perf_model_errs[*]}"
      echo "  model_name=$model_name"
      echo "  tensor_parallel_size=$tensor_parallel_size"
      echo "  profit=$profit"
      echo "  admission_mode=$admission_mode"
      echo "  slo_routing_overhead=$slo_routing_overhead"
      echo "  scheduling_overhead=$scheduling_overhead"
      echo "  routing_overhead=$routing_overhead"
      echo "  routing_fallback_policy=$routing_fallback_policy"
      echo "  output_dir=$output_dir"
      echo "  extra_args=${extra_bench_args[*]}"
      if ((${#run_devices[@]} != ${#n_devices[@]})); then
        echo "  requested_n_devices=${n_devices[*]}"
        echo "  available_clients=$available_clients"
      fi

      ensure_server_ready

      cmd=(
        python motivation/bench_api_server.py
        --overwrite
        --n_devices "${run_devices[@]}"
        --policies "$policy"
        --load_scales "${load_scales[@]}"
        --slo_tpots "${slo_tpots[@]}"
        --ttft_slo_scales "${ttft_slo_scales[@]}"
        --perf_model_err "${perf_model_errs[@]}"
        --window "$window"
        --trace "${length_trace}:${arrival_trace}"
        --profit "$profit"
        --admission_mode "$admission_mode"
        --slo_routing_overhead "$slo_routing_overhead"
        --scheduling_overhead "$scheduling_overhead"
        --model_name "$model_name"
        --tensor_parallel_size "$tensor_parallel_size"
        --port "$RUN_BATCH_PORT"
        --clients "$SERVER_CLIENTS"
        --output_dir "$output_dir"
        --routing_overhead "$routing_overhead"
        --routing_fallback_policy "$routing_fallback_policy"
        "${extra_bench_args[@]}"
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
        cleanup_server "$server_pid"
        server_pid=""
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
  "slosserve_planner_ablation-no_global:atfc"
  "slosserve_planner_ablation-no_local:atfc"
  "round_robin-disagg:sarathi+"
  "llumnix_load-disagg:sarathi+"
  "round_robin-disagg:atfc"
  "slosserve_disagg_planner:atfc"
  "slosserve_disagg_planner_oracle_mem:atfc"
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
  ["azure_chat_23:azure_chat_23"]="t1200:1800 1.0 2 4 6 8"
  ["azure_code:azure_code"]="t1200:1800 1.0 2 4 8 12 16 32"
  ["azure_chat:azure_chat"]="t1200:1800 1.0 2 4 8 12 16 32"
  ["sharegpt_code:azure_code_23"]="t1200:1800 1.0 2 4 6 8"
  ["sharegpt_chat:azure_chat_23"]="t1200:1800 1.0 2 4 6 8"
  ["sharegpt_code:azure_code"]="t1200:1800 1.0 4 8 12 16 32"
  ["sharegpt_chat:azure_chat"]="t1200:1800 1.0 4 8 12 16 32"
  ["reasoning:azure_chat_23"]="t1200:1800 1.0 4 8 12 16 32"
)

if [[ -n "$BATCH_CONFIG_PATH" ]]; then
  if [[ ! -f "$BATCH_CONFIG_PATH" ]]; then
    echo "ERROR: Missing batch config at $BATCH_CONFIG_PATH" >&2
    exit 1
  fi

  echo "Loading batch config: $BATCH_CONFIG_PATH"
  eval "$(
    python -m SLOsServe.batch_config --shell "$BATCH_CONFIG_PATH"
  )"
fi

run_suite
