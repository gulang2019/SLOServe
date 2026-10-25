# Shared settings and helpers for the BEAM scripts. Source this file; do not run it.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
beam_root="$repo_root/3rdparty/BEAM"
beam_energy="$beam_root/benchmarks/energy"
container_name="${BEAM_CONTAINER:-vllm_evaluator_env}"
beam_image="jlee335/beam-volume-260307:latest"
beam_gpu="${BEAM_GPU:-0}"
beam_model="${BEAM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
beam_port="${BEAM_PORT:-8008}"
# One BEAM server per GPU, each on its own port, can share the container.
beam_server_pattern="vllm serve .* --port $beam_port "
beam_server_log="/tmp/beam-qwen7b-$beam_port.log"

require_docker() {
  if ! docker info >/dev/null 2>&1; then
    echo "Docker is unavailable to the current user. If you were just added to the" >&2
    echo "docker group, log in again or run this script through: sg docker -c '...'" >&2
    exit 1
  fi
}

require_gpu() {
  if ! nvidia-smi -i "$beam_gpu" >/dev/null 2>&1; then
    echo "BEAM_GPU=$beam_gpu is not an available NVIDIA GPU." >&2
    exit 1
  fi
}

# Timing and energy are measured per GPU, so nothing else may run on it.
require_idle_gpu() {
  local pids
  pids="$(nvidia-smi -i "$beam_gpu" --query-compute-apps=pid --format=csv,noheader)"
  if [[ -n "$pids" ]]; then
    echo "GPU $beam_gpu is in use (PIDs: $(echo $pids)); BEAM needs it exclusively." >&2
    exit 1
  fi
}

# A config naming a credential helper that is not installed (e.g. Docker
# Desktop's "desktop") breaks every pull; fall back to an empty config.
skip_missing_credential_helper() {
  local config="${DOCKER_CONFIG:-$HOME/.docker}/config.json" helper
  [[ -f "$config" ]] || return 0
  helper="$(sed -n 's/.*"credsStore"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$config")"
  if [[ -n "$helper" ]] && ! command -v "docker-credential-$helper" >/dev/null 2>&1; then
    export DOCKER_CONFIG="${XDG_CACHE_HOME:-$HOME/.cache}/beam-docker-config"
    mkdir -p "$DOCKER_CONFIG"
    [[ -f "$DOCKER_CONFIG/config.json" ]] || echo '{}' > "$DOCKER_CONFIG/config.json"
    echo "docker-credential-$helper is not installed; using an empty DOCKER_CONFIG at $DOCKER_CONFIG."
  fi
}

# Start the evaluator container and wait until its startup command has
# installed the evaluation packages (pandas, pynvml, ...).
start_container() {
  (
    cd "$beam_root"
    docker compose -f docker-compose.yml -f "$repo_root/scripts/beam-compose-7b.yml" up -d evaluator
  )
  local waited=0
  until docker logs "$container_name" 2>&1 | grep -q 'Evaluator environment ready'; do
    if (( waited >= 600 )); then
      echo "Container $container_name did not finish setup in 600 s; see: docker logs $container_name" >&2
      exit 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

profile_path() {
  local gpu_name
  gpu_name="$(nvidia-smi -i "$beam_gpu" --query-gpu=name --format=csv,noheader | head -n 1)"
  echo "$beam_energy/offline_profile_results/dvfs_profile_${gpu_name}_${beam_model//\//_}_tp1_pp1_one.csv"
}
