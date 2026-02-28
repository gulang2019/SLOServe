#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   # Option A: specify pairs (preferred)
#   PAIRS="azure_chat_23:azure_chat_23 other_arrival:other_length" \
#   # Option B: separate lists (cross-product)
#   ARRIVAL_PATTERNS="azure_chat_23 another_arrival" \
#   LENGTH_PATTERNS="azure_chat_23 another_length" \
#   WINDOW_MINUTES="5 10 30" \
#   NO_CONFIRM=1 \
#   python_cmd="python" \
#   ./run_headroom_analysis.sh
#
# If PAIRS is set, ARRIVAL_PATTERNS/LENGTH_PATTERNS are ignored.
# If LENGTH_PATTERNS is empty, it defaults to ARRIVAL_PATTERNS.

python_cmd="${python_cmd:-python}"

PAIRS="${PAIRS:-}"
ARRIVAL_PATTERNS="${ARRIVAL_PATTERNS:-azure_chat_23}"
LENGTH_PATTERNS="${LENGTH_PATTERNS:-$ARRIVAL_PATTERNS}"
WINDOW_MINUTES="${WINDOW_MINUTES:-5 10 30}"
NO_CONFIRM="${NO_CONFIRM:-1}"

run_pair() {
  local arrival="$1"
  local length="$2"
  for is_oracle in 0 1; do
    for is_pd_disagg in 0 1; do
      args=(
        "--arrival-pattern" "$arrival"
        "--length-pattern" "$length"
        "--window-minutes" $WINDOW_MINUTES
      )
      if [[ "$NO_CONFIRM" == "1" ]]; then
        args+=("--no-confirm")
      fi
      if [[ "$is_oracle" == "1" ]]; then
        args+=("--is-oracle")
      fi
      if [[ "$is_pd_disagg" == "1" ]]; then
        args+=("--is-pd-disagg")
      fi
      echo "Running: $python_cmd SLOsServe/analysis/headroom_analysis.py ${args[*]}"
      $python_cmd SLOsServe/analysis/headroom_analysis.py "${args[@]}"
    done
  done
}

if [[ -n "$PAIRS" ]]; then
  for pair in $PAIRS; do
    arrival="${pair%%:*}"
    length="${pair#*:}"
    if [[ "$arrival" == "$length" && "$pair" != *:* ]]; then
      length="$arrival"
    fi
    run_pair "$arrival" "$length"
  done
else
  for arrival in ${ARRIVAL_PATTERNS}; do
    for length in ${LENGTH_PATTERNS}; do
      run_pair "$arrival" "$length"
    done
  done
fi
