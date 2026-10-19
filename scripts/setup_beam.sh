#!/usr/bin/env bash
# Prepare everything the BEAM baseline needs before profiling:
#   1. the BEAM submodule, with scripts/patches/beam-baseline.patch applied,
#   2. the evaluator image (pulled, or built from source with --build),
#   3. a running evaluator container with its packages installed,
#   4. BEAM vLLM files patched in this checkout, copied into the container
#      (the server runs the vLLM installed in the image, not the checkout),
#   5. the Figure 7 traces, when SLOServe's datasets are downloaded.
#
# Usage: scripts/setup_beam.sh [--build]
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/beam_common.sh"

build=0
case "${1:-}" in
  --build) build=1 ;;
  "") ;;
  -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

require_docker
require_gpu

# 1. Submodule. Only initialize a missing checkout: updating an existing one
# would move it to the recorded commit and could discard local changes.
if [[ ! -f "$beam_root/docker-compose.yml" ]]; then
  git -C "$repo_root" submodule update --init 3rdparty/BEAM
fi

# BEAM fixes for this baseline live as a patch, since the submodule tracks
# upstream BEAM. Apply it once; skip when it is already in the checkout.
patch="$repo_root/scripts/patches/beam-baseline.patch"
if git -C "$beam_root" apply --reverse --check "$patch" 2>/dev/null; then
  echo "BEAM baseline patch already applied."
elif git -C "$beam_root" apply --check "$patch" 2>/dev/null; then
  git -C "$beam_root" apply "$patch"
  echo "Applied $patch to 3rdparty/BEAM."
else
  echo "Cannot apply $patch: 3rdparty/BEAM has conflicting local changes." >&2
  exit 1
fi

# 2. Image.
skip_missing_credential_helper
if [[ $build -eq 1 ]]; then
  (
    cd "$beam_root"
    docker compose -f docker-compose.yml -f "$repo_root/scripts/beam-compose-7b.yml" build evaluator
  )
else
  docker pull "$beam_image"
fi

# 3. Container.
start_container
echo "Container $container_name is ready."

# 4. Patched vLLM files: every file under vllm/ that differs from the
# submodule's commit is copied over the image's installed copy.
site="$(docker exec "$container_name" python3 -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')"
while IFS= read -r path; do
  [[ -f "$beam_root/$path" ]] || continue
  docker cp "$beam_root/$path" "$container_name:$site/${path#vllm/}"
  echo "Copied patched $path into the container."
done < <(git -C "$beam_root" diff --name-only HEAD -- vllm/)

# 5. Traces.
datasets="$beam_energy/datasets"
if ls "$datasets"/sloserve_fig7_{chat,code,mixed}_7b_t1200_1800.csv >/dev/null 2>&1; then
  echo "Figure 7 traces already exported."
elif [[ -d "$repo_root/assets/datasets" ]]; then
  (cd "$repo_root" && PYTHONPATH="$repo_root" "${PYTHON:-$repo_root/.venv/bin/python}" scripts/export_beam_figure7_traces.py)
else
  echo "Skipping trace export: assets/datasets is missing. Run 'PYTHONPATH=. python Dataset/download_dataset.py'"
  echo "and rerun, or benchmark the round-robin splits with --split N/I."
fi

echo
echo "Setup complete. Next: scripts/serve_beam_qwen7b.sh (profiles on first s1s2 launch)."
