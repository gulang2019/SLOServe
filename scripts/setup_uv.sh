#!/usr/bin/env bash
# Bootstrap the pinned SLOServe checkout using uv (no Conda required).
#
# Usage from the repository root:
#   ./scripts/setup_uv.sh
#   VENV_PATH=.venv-sloserve ./scripts/setup_uv.sh

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it first: https://docs.astral.sh/uv/"
    exit 1
fi

VENV_PATH="${VENV_PATH:-.venv}"

# Check out the exact modified-vLLM revision recorded by this SLOServe commit.
# Do not update 3rdparty/vllm to its remote main branch here.
git submodule update --init --recursive

uv venv --python 3.10 "$VENV_PATH"
PYTHON="$ROOT/$VENV_PATH/bin/python"

# vLLM's setup.py imports torch while its editable-build metadata is being
# generated. Install it in the environment first because this install uses
# --no-build-isolation. Keep setuptools within vLLM's supported range too.
uv pip install --python "$PYTHON" --torch-backend=auto \
    "setuptools>=77,<80" "torch==2.7.1"
uv pip install --python "$PYTHON" -r requirements.txt
VLLM_USE_PRECOMPILED=1 uv pip install --python "$PYTHON" \
    --no-build-isolation --editable 3rdparty/vllm
uv pip install --python "$PYTHON" --no-build-isolation --editable csrc

echo
echo "Setup complete. Activate with: source $VENV_PATH/bin/activate"
echo "Then run: export PYTHONPATH=$ROOT"
