#!/usr/bin/env bash
# Run BEAM for the Figure 7 7B ChatBot, Coder, and Mixed traces.
# Requires an already-running BEAM Qwen2.5-7B server (normally s1s2 mode).
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
beam_root="$repo_root/3rdparty/BEAM"
container_name="${BEAM_CONTAINER:-vllm_evaluator_env}"
beam_host="${BEAM_HOST:-127.0.0.1}"
beam_port="${BEAM_PORT:-8008}"
beam_gpu="${BEAM_GPU:-0}"
output_root="results/beam-figure7-7b-$(date +%Y%m%d-%H%M%S)"
first_only=0
ttft_quantile="50"

usage() {
  cat <<'EOF'
Usage: scripts/run_beam_figure7_7b.sh [--first-trace] [--ttft-quantile 0-100] [--output-root RELATIVE_PATH]

Runs the Figure 7 7B ChatBot, Coder, and Mixed traces. --first-trace runs
only ChatBot. The global BEAM TTFT control value defaults to the median
prompt-dependent SLOServe budget. Output is relative to
3rdparty/BEAM/benchmarks/energy.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --first-trace) first_only=1 ;;
    --ttft-quantile) ttft_quantile="$2"; shift ;;
    --output-root) output_root="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if ! [[ "$ttft_quantile" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
   ! awk -v q="$ttft_quantile" 'BEGIN { exit !(q >= 0 && q <= 100) }'; then
  echo "--ttft-quantile must be in [0, 100]." >&2
  exit 2
fi

if [[ "$output_root" == /* || "$output_root" == *".."* || ! "$output_root" =~ ^[A-Za-z0-9_./-]+$ ]]; then
  echo "--output-root must be a safe relative path." >&2
  exit 2
fi
if ! docker ps --format '{{.Names}}' | grep -qx "$container_name"; then
  echo "BEAM container $container_name is not running." >&2
  exit 1
fi
if ! curl -fsS "http://$beam_host:$beam_port/v1/models" >/dev/null; then
  echo "BEAM server is not reachable at http://$beam_host:$beam_port/v1." >&2
  exit 1
fi
if ! curl -fsS "http://$beam_host:$beam_port/openapi.json" | grep -q '"/start_profile"'; then
  echo "BEAM profiling endpoints are unavailable. Restart the server with scripts/serve_beam_qwen7b.sh from this checkout." >&2
  exit 1
fi

# name|trace|unscaled TTFT slope|unscaled TTFT intercept|routing allowance|TPOT/TBT
configs=(
  "chat_7b|sloserve_fig7_chat_7b_t1200_1800.csv|5.55e-05|0.014|0.05|0.025"
  "code_7b|sloserve_fig7_code_7b_t1200_1800.csv|6.1390787826193327e-05|0.010899481145765222|0.05|0.025"
  "mixed_7b|sloserve_fig7_mixed_7b_t1200_1800.csv|6.0702667844154913e-05|0.011045273917089907|0.02|0.020"
)

stop_power_monitor() {
  local container_dir="$1"
  docker exec "$container_name" bash -lc \
    "test -f '$container_dir/power_monitor.pid' && kill -TERM \$(cat '$container_dir/power_monitor.pid') 2>/dev/null || true" || true
}

update_server_slos() {
  local tpot="$1" ttft="$2"
  local base_url="http://$beam_host:$beam_port"
  # BEAM's S1/S2 controller consumes TTFT and TBT; set TPOT as well because
  # the server exposes both decode-SLO variants. Fail before sending traffic
  # if any update is rejected.
  curl -fsS -X POST "$base_url/update_tpot_slo" \
    -H 'Content-Type: application/json' --data "{\"tpot_slo\":$tpot}" >/dev/null
  curl -fsS -X POST "$base_url/update_tbt_slo" \
    -H 'Content-Type: application/json' --data "{\"tbt_slo\":$tpot}" >/dev/null
  curl -fsS -X POST "$base_url/update_ttft_slo" \
    -H 'Content-Type: application/json' --data "{\"ttft_slo\":$ttft}" >/dev/null
}

run_one() {
  local name="$1" trace_name="$2" slope="$3" intercept="$4" routing="$5" tpot="$6"
  local trace_host="$beam_root/benchmarks/energy/datasets/$trace_name"
  local run_rel="$output_root/$name"
  local run_host="$beam_root/benchmarks/energy/$run_rel"
  local run_container="/workspace/benchmarks/energy/$run_rel"
  local scalar_ttft
  if [[ ! -f "$trace_host" ]]; then
    echo "Missing trace: $trace_host. Run scripts/export_beam_figure7_traces.py first." >&2
    exit 1
  fi

  # BEAM accepts only one global TTFT control target. Use a configurable
  # quantile of the prompt-dependent SLOServe budget (median by default),
  # while the postprocessor remains the authoritative exact score.
  scalar_ttft="$(python3 -c '
import csv, math, sys
path, slope, intercept, routing, quantile = sys.argv[1:]
budgets = sorted(float(routing) + 3 * (float(intercept) + float(slope) * int(row["ContextTokens"])) for row in csv.DictReader(open(path, encoding="utf-8")))
if not budgets: raise SystemExit("trace is empty")
position = float(quantile) / 100 * (len(budgets) - 1)
lower, upper = math.floor(position), math.ceil(position)
value = budgets[lower] + (budgets[upper] - budgets[lower]) * (position - lower)
print(f"{value:.9f}")
' "$trace_host" "$slope" "$intercept" "$routing" "$ttft_quantile")"

  mkdir -p "$run_host"
  printf '%s\n' \
    "trace=$trace_name" "ttft_scale=3" "zero_load_ttft_per_token_s=$slope" \
    "zero_load_ttft_constant_s=$intercept" "routing_overhead_s=$routing" \
    "slo_tpot_s=$tpot" "beam_ttft_quantile=$ttft_quantile" \
    "beam_global_ttft_s=$scalar_ttft" > "$run_host/slo_config.txt"

  echo "==> $name: TPOT/TBT=${tpot}s, BEAM p${ttft_quantile} TTFT=${scalar_ttft}s"
  echo "    Updating BEAM server SLOs through its API."
  update_server_slos "$tpot" "$scalar_ttft"
  docker exec "$container_name" bash -lc \
    "mkdir -p '$run_container'; nohup python3 /workspace/benchmarks/energy/gpu_energy_monitor.py '$run_container' > '$run_container/power-monitor.log' 2>&1 & echo \$! > '$run_container/power_monitor.pid'"

  set +e
  docker exec "$container_name" bash -lc \
    "cd /workspace/benchmarks/energy && PYTHONPATH=/workspace/benchmarks:\$PYTHONPATH python3 benchmark_script.py \
      --backend vllm --host '$beam_host' --port '$beam_port' --endpoint /v1/completions \
      --model Qwen/Qwen2.5-7B-Instruct --dataset-name traced \
      --dataset-path '/workspace/benchmarks/energy/datasets/$trace_name' \
      --profile \
      --save-result --save-detailed --result-dir '$run_rel' \
      --result-filename beam_detailed.json --disable-tqdm" \
    | tee "$run_host/benchmark.log"
  local benchmark_status=${PIPESTATUS[0]}
  set -e
  stop_power_monitor "$run_container"
  sleep 1
  if [[ $benchmark_status -ne 0 ]]; then
    echo "BEAM benchmark failed for $name; preserving logs in $run_host." >&2
    return "$benchmark_status"
  fi

  local raw_power="$run_host/gpu_energy_and_frequency_huggyllama_llama-13b.csv"
  if [[ -f "$raw_power" ]]; then
    python3 "$repo_root/scripts/derive_beam_power_timeline.py" \
      --input "$raw_power" --output "$run_host/power_timeline_gpu_${beam_gpu}.csv" --gpu "$beam_gpu"
  else
    echo "Warning: BEAM power monitor produced no CSV; see $run_host/power-monitor.log" >&2
  fi
  python3 "$repo_root/scripts/score_beam_detailed_results.py" \
    --result "$run_host/beam_detailed.json" --trace "$trace_host" \
    --output "$run_host/per_request_slo.jsonl" \
    --ttft-scale 3 --zero-load-ttft-per-token "$slope" \
    --zero-load-ttft-constant "$intercept" --routing-overhead "$routing" --slo-tpot "$tpot"
}

for config in "${configs[@]}"; do
  IFS='|' read -r name trace slope intercept routing tpot <<< "$config"
  run_one "$name" "$trace" "$slope" "$intercept" "$routing" "$tpot"
  [[ $first_only -eq 1 ]] && break
done

echo "Completed. Results: $beam_root/benchmarks/energy/$output_root"
