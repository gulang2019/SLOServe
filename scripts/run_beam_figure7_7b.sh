#!/usr/bin/env bash
# Benchmark a running BEAM Qwen2.5-7B server (normally s1s2 mode, started by
# scripts/serve_beam_qwen7b.sh) on the Figure 7 ChatBot, Coder and Mixed
# traces, then record power and score every request with the SLOServe rule.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/beam_common.sh"

beam_host="${BEAM_HOST:-127.0.0.1}"
beam_port="${BEAM_PORT:-8008}"
output_root="results/beam-figure7-7b-$(date +%Y%m%d-%H%M%S)-gpu${beam_gpu}"
first_only=0
only_trace=""
split=""
duration=""
ttft_quantile="50"
tbt_scale="1"

usage() {
  cat <<'EOF'
Usage: scripts/run_beam_figure7_7b.sh [--first-trace | --trace NAME] [--split N/I]
         [--duration SECONDS] [--ttft-quantile 0-100] [--tbt-scale X] [--output-root RELATIVE_PATH]

Runs the Figure 7 7B ChatBot, Coder, and Mixed traces. --first-trace runs
only ChatBot; --trace runs one of chat_7b, code_7b, mixed_7b. --split N/I
replays replica I of the N-way round-robin split instead of the full trace,
and --duration keeps only its first SECONDS.

BEAM takes one global TTFT and one TBT control target. --ttft-quantile sets
the TTFT target to that quantile of the trace's per-request SLOServe TTFT
budgets (default 50, the median); --tbt-scale sets the TBT target to X times
the SLOServe TPOT (default 1). Scoring always uses the SLOServe definition.
Output is relative to 3rdparty/BEAM/benchmarks/energy.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --first-trace) first_only=1 ;;
    --trace) only_trace="$2"; shift ;;
    --split) split="$2"; shift ;;
    --duration) duration="$2"; shift ;;
    --ttft-quantile) ttft_quantile="$2"; shift ;;
    --tbt-scale) tbt_scale="$2"; shift ;;
    --output-root) output_root="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ "$output_root" == /* || "$output_root" == *".."* || ! "$output_root" =~ ^[A-Za-z0-9_./-]+$ ]]; then
  echo "--output-root must be a safe relative path." >&2
  exit 2
fi
if [[ -n "$split" && ! "$split" =~ ^[0-9]+/[0-9]+$ ]]; then
  echo "--split must look like N/I, e.g. 8/0." >&2
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
  [[ -n "$split" ]] && trace_name="sloserve_fig7_round_robin/$name/$split.csv"
  local trace_host="$beam_root/benchmarks/energy/datasets/$trace_name"
  local trace_container="/workspace/benchmarks/energy/datasets/$trace_name"
  local run_rel="$output_root/$name"
  local run_host="$beam_root/benchmarks/energy/$run_rel"
  local run_container="/workspace/benchmarks/energy/$run_rel"
  local scalar_ttft beam_tbt
  if [[ ! -f "$trace_host" ]]; then
    echo "Missing trace: $trace_host. Run scripts/export_beam_figure7_traces.py first." >&2
    exit 1
  fi
  mkdir -p "$run_host"

  # Replay only the first $duration seconds; benchmark and scorer both use
  # the truncated copy so request indices stay aligned.
  if [[ -n "$duration" ]]; then
    awk -F, -v d="$duration" 'NR == 1 {print; next}
      {split($1, dt, " "); split(substr(dt[2], 1, 12), t, ":");
       s = t[1] * 3600 + t[2] * 60 + t[3]; if (NR == 2) s0 = s; if (s - s0 < d) print}' \
      "$trace_host" > "$run_host/trace.csv"
    trace_host="$run_host/trace.csv"
    trace_container="$run_container/trace.csv"
  fi

  # BEAM accepts only one global TTFT control target. Use a quantile of the
  # per-request SLOServe budgets (the median by default). The postprocessor
  # remains the authoritative exact score.
  scalar_ttft="$(python3 -c '
import csv, math, sys
path, slope, intercept, routing, quantile = sys.argv[1:]
budgets = sorted(float(routing) + 3 * (float(intercept) + float(slope) * int(row["ContextTokens"])) for row in csv.DictReader(open(path, encoding="utf-8")))
if not budgets: raise SystemExit("trace is empty")
position = float(quantile) / 100 * (len(budgets) - 1)
lower, upper = math.floor(position), math.ceil(position)
print(f"{budgets[lower] + (budgets[upper] - budgets[lower]) * (position - lower):.9f}")
' "$trace_host" "$slope" "$intercept" "$routing" "$ttft_quantile")"
  beam_tbt="$(awk -v t="$tpot" -v s="$tbt_scale" 'BEGIN {printf "%.6f", t * s}')"

  printf '%s\n' \
    "trace=$trace_name" "duration_s=${duration:-full}" "ttft_scale=3" \
    "zero_load_ttft_per_token_s=$slope" \
    "zero_load_ttft_constant_s=$intercept" "routing_overhead_s=$routing" \
    "slo_tpot_s=$tpot" "beam_ttft_quantile=$ttft_quantile" \
    "beam_global_ttft_s=$scalar_ttft" "beam_tbt_scale=$tbt_scale" \
    "beam_tbt_s=$beam_tbt" > "$run_host/slo_config.txt"

  echo "==> $name: SLOServe TPOT=${tpot}s; BEAM TBT=${beam_tbt}s, global TTFT=${scalar_ttft}s"
  echo "    Updating BEAM server SLOs through its API."
  update_server_slos "$beam_tbt" "$scalar_ttft"
  docker exec "$container_name" bash -lc \
    "mkdir -p '$run_container'; nohup python3 /workspace/benchmarks/energy/gpu_energy_monitor.py '$run_container' > '$run_container/power-monitor.log' 2>&1 & echo \$! > '$run_container/power_monitor.pid'"

  set +e
  docker exec "$container_name" bash -lc \
    "cd /workspace/benchmarks/energy && PYTHONPATH=/workspace/benchmarks:\$PYTHONPATH python3 benchmark_script.py \
      --backend vllm --host '$beam_host' --port '$beam_port' --endpoint /v1/completions \
      --model '$beam_model' --dataset-name traced \
      --dataset-path '$trace_container' \
      --profile --ignore-eos \
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
  [[ -n "$only_trace" && "$name" != "$only_trace" ]] && continue
  run_one "$name" "$trace" "$slope" "$intercept" "$routing" "$tpot"
  [[ $first_only -eq 1 ]] && break
done

echo
python3 "$repo_root/scripts/summarize_beam_sweep.py" "$beam_energy/$output_root"
echo "Completed. Results: $beam_energy/$output_root"
