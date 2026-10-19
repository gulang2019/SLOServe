#!/usr/bin/env bash
set -euo pipefail

# Generate the profile required by BEAM's S1/S2 scheduler for the local 7B
# service. This is intentionally a separate, long-running step.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
beam_root="$repo_root/3rdparty/BEAM"
container_name="vllm_evaluator_env"
beam_gpu="${BEAM_GPU:-0}"
beam_model="${BEAM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

if ! nvidia-smi -i "$beam_gpu" >/dev/null 2>&1; then
  echo "BEAM_GPU=$beam_gpu is not an available NVIDIA GPU." >&2
  exit 1
fi

compose=(docker compose -f docker-compose.yml -f "$repo_root/scripts/beam-compose-7b.yml")
(
  cd "$beam_root"
  "${compose[@]}" up -d evaluator
)

docker exec \
  -e "CUDA_VISIBLE_DEVICES=$beam_gpu" \
  -e "HF_TOKEN=${HF_TOKEN:-}" \
  "$container_name" \
  bash -lc "cd /workspace/benchmarks/energy && ./run_offline_profile.sh --model '$beam_model' --tp 1 --pp 1"

gpu_name="$(nvidia-smi -i "$beam_gpu" --query-gpu=name --format=csv,noheader | head -n 1)"
model_name_clean="${beam_model//\//_}"
profile_path="$beam_root/benchmarks/energy/offline_profile_results/dvfs_profile_${gpu_name}_${model_name_clean}_tp1_pp1_one.csv"
if [[ ! -s "$profile_path" ]]; then
  echo "BEAM profiling did not produce $profile_path" >&2
  exit 1
fi
