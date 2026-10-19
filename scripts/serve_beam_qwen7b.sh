#!/usr/bin/env bash
set -euo pipefail

# Profile (if needed) and start a dedicated, single-GPU BEAM/vLLM
# OpenAI-compatible service, then wait until it answers.
#
# BEAM_MODE=vanilla (default) serves at default clocks and ignores SLO targets.
# BEAM_MODE=s1s2 enables BEAM's DVFS scheduler, which needs the offline profile
# for this GPU and model. BEAM_PROFILE controls that profile in s1s2 mode:
#   auto   (default) profile first only when the profile CSV is missing
#   always re-profile before every launch
#   never  fail when the profile is missing

source "$(dirname -- "${BASH_SOURCE[0]}")/beam_common.sh"
beam_port="${BEAM_PORT:-8008}"
beam_max_model_len="${BEAM_MAX_MODEL_LEN:-8192}"
beam_memory_utilization="${BEAM_GPU_MEMORY_UTILIZATION:-0.90}"
beam_mode="${BEAM_MODE:-vanilla}"
beam_profile="${BEAM_PROFILE:-auto}"
ready_timeout="${BEAM_READY_TIMEOUT:-900}"

if [[ ! -f "$beam_root/docker-compose.yml" ]]; then
  echo "BEAM checkout not found at $beam_root; run scripts/setup_beam.sh first." >&2
  exit 1
fi
require_docker
require_gpu

beam_extra_args=()
case "$beam_mode" in
  vanilla)
    ;;
  s1s2)
    profile="$(profile_path)"
    case "$beam_profile" in
      always) "$repo_root/scripts/profile_beam_qwen7b.sh" ;;
      auto) [[ -f "$profile" ]] || "$repo_root/scripts/profile_beam_qwen7b.sh" ;;
      never) ;;
      *) echo "Unsupported BEAM_PROFILE=$beam_profile (supported: auto, always, never)." >&2; exit 1 ;;
    esac
    if [[ ! -f "$profile" ]]; then
      echo "Missing BEAM S1/S2 DVFS profile: $profile" >&2
      exit 1
    fi
    beam_extra_args=(--use-s1 --use-s2)
    ;;
  *)
    echo "Unsupported BEAM_MODE=$beam_mode (supported: vanilla, s1s2)." >&2
    exit 1
    ;;
esac

start_container

if docker exec "$container_name" pgrep -f 'vllm serve' >/dev/null 2>&1; then
  echo "A BEAM vLLM server is already running in $container_name; refusing to replace it." >&2
  echo "Stop it first with scripts/stop_beam_qwen7b.sh." >&2
  exit 1
fi

# VLLM_TORCH_PROFILER_DIR enables /start_profile, /stop_profile and the SLO
# update endpoints; --custom-profiler makes /start_profile run BEAM's energy
# logger instead of the torch profiler. Per-run benchmarks override the
# output directory.
docker exec -d \
  -e "CUDA_VISIBLE_DEVICES=$beam_gpu" \
  -e "HF_TOKEN=${HF_TOKEN:-}" \
  -e "VLLM_TORCH_PROFILER_DIR=/workspace/benchmarks/energy/results" \
  "$container_name" \
  bash -lc "cd /workspace/benchmarks/energy && exec vllm serve '$beam_model' --host 0.0.0.0 --port '$beam_port' --tensor-parallel-size 1 --pipeline-parallel-size 1 --max-model-len '$beam_max_model_len' --gpu-memory-utilization '$beam_memory_utilization' --custom-profiler ${beam_extra_args[*]} > /tmp/beam-qwen7b.log 2>&1"

echo "BEAM ($beam_mode) is starting on GPU $beam_gpu; waiting for http://127.0.0.1:$beam_port/v1 ..."
waited=0
until curl -fsS "http://127.0.0.1:$beam_port/v1/models" >/dev/null 2>&1; do
  if ! docker exec "$container_name" pgrep -f 'vllm serve' >/dev/null 2>&1; then
    echo "The BEAM server exited during startup. Last log lines:" >&2
    docker exec "$container_name" tail -n 30 /tmp/beam-qwen7b.log >&2 || true
    exit 1
  fi
  if (( waited >= ready_timeout )); then
    echo "The BEAM server did not answer within ${ready_timeout} s; see /tmp/beam-qwen7b.log in $container_name." >&2
    exit 1
  fi
  sleep 5
  waited=$((waited + 5))
done

echo "BEAM ($beam_mode) is ready at http://127.0.0.1:$beam_port/v1"
echo "Follow logs with: docker exec -it $container_name tail -f /tmp/beam-qwen7b.log"
