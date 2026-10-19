#!/usr/bin/env bash
set -euo pipefail

# Start a dedicated, single-GPU BEAM/vLLM OpenAI-compatible service.
# The default intentionally omits --use-s1/--use-s2: those policies require a
# fresh DVFS profile for the selected GPU, model, and parallelism configuration.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
beam_root="$repo_root/3rdparty/BEAM"
container_name="vllm_evaluator_env"

beam_gpu="${BEAM_GPU:-0}"
beam_port="${BEAM_PORT:-8008}"
beam_model="${BEAM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
beam_max_model_len="${BEAM_MAX_MODEL_LEN:-8192}"
beam_memory_utilization="${BEAM_GPU_MEMORY_UTILIZATION:-0.90}"
beam_mode="${BEAM_MODE:-vanilla}"

if [[ ! -f "$beam_root/docker-compose.yml" ]]; then
  echo "BEAM checkout not found at $beam_root" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker is unavailable to the current user." >&2
  exit 1
fi

if ! nvidia-smi -i "$beam_gpu" >/dev/null 2>&1; then
  echo "BEAM_GPU=$beam_gpu is not an available NVIDIA GPU." >&2
  exit 1
fi

beam_extra_args=()
case "$beam_mode" in
  vanilla)
    ;;
  s1s2)
    gpu_name="$(nvidia-smi -i "$beam_gpu" --query-gpu=name --format=csv,noheader | head -n 1)"
    model_name_clean="${beam_model//\//_}"
    profile_path="$beam_root/benchmarks/energy/offline_profile_results/dvfs_profile_${gpu_name}_${model_name_clean}_tp1_pp1_one.csv"
    if [[ ! -f "$profile_path" ]]; then
      echo "Missing BEAM S1/S2 DVFS profile: $profile_path" >&2
      echo "Run ./scripts/profile_beam_qwen7b.sh before launching BEAM_MODE=s1s2." >&2
      exit 1
    fi
    beam_extra_args=(--use-s1 --use-s2)
    ;;
  *)
    echo "Unsupported BEAM_MODE=$beam_mode (supported: vanilla, s1s2)." >&2
    exit 1
    ;;
esac

compose=(docker compose -f docker-compose.yml -f "$repo_root/scripts/beam-compose-7b.yml")
(
  cd "$beam_root"
  "${compose[@]}" up -d evaluator
)

if docker exec "$container_name" pgrep -f 'vllm serve' >/dev/null 2>&1; then
  echo "A BEAM vLLM server is already running in $container_name; refusing to replace it." >&2
  exit 1
fi

docker exec -d \
  -e "CUDA_VISIBLE_DEVICES=$beam_gpu" \
  -e "HF_TOKEN=${HF_TOKEN:-}" \
  "$container_name" \
  bash -lc "cd /workspace/benchmarks/energy && exec vllm serve '$beam_model' --host 0.0.0.0 --port '$beam_port' --tensor-parallel-size 1 --pipeline-parallel-size 1 --max-model-len '$beam_max_model_len' --gpu-memory-utilization '$beam_memory_utilization' ${beam_extra_args[*]} > /tmp/beam-qwen7b.log 2>&1"

echo "BEAM ($beam_mode) is starting on GPU $beam_gpu at http://127.0.0.1:$beam_port/v1"
echo "Follow startup logs with: docker exec -it $container_name tail -f /tmp/beam-qwen7b.log"
