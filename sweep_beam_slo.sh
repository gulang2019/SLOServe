#!/usr/bin/env bash
# Grid-search BEAM's TTFT/TBT control targets for SLOServe attainment.
# Each grid point is one scripts/run_beam_figure7_7b.sh run against the
# already-running server.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
trace="chat_7b"
split=""
duration=""
quantiles="0 25 50 90"
scales="0.6 0.8 1.0"
output_root="results/beam-slo-sweep-$(date +%Y%m%d-%H%M%S)"

usage() {
  cat <<'EOF'
Usage: sweep_beam_slo.sh [--trace NAME] [--split N/I] [--duration SECONDS]
                         [--quantiles "Q ..."] [--scales "X ..."] [--output-root RELATIVE_PATH]

Runs scripts/run_beam_figure7_7b.sh for every (--ttft-quantile, --tbt-scale) pair and prints
SLOServe attainment and energy per pair. Re-using --output-root adds points
to an existing sweep; the summary covers every run under it.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --trace) trace="$2"; shift ;;
    --split) split="$2"; shift ;;
    --duration) duration="$2"; shift ;;
    --quantiles) quantiles="$2"; shift ;;
    --scales) scales="$2"; shift ;;
    --output-root) output_root="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

extra=()
[[ -n "$split" ]] && extra+=(--split "$split")
[[ -n "$duration" ]] && extra+=(--duration "$duration")

for scale in $scales; do
  for quantile in $quantiles; do
    bash "$repo_root/scripts/run_beam_figure7_7b.sh" --trace "$trace" "${extra[@]}" \
      --ttft-quantile "$quantile" --tbt-scale "$scale" \
      --output-root "$output_root/q${quantile}_s${scale}"
  done
done

sweep_host="$repo_root/3rdparty/BEAM/benchmarks/energy/$output_root"
python3 "$repo_root/scripts/summarize_beam_sweep.py" "$sweep_host" --csv "$sweep_host/summary.csv"
