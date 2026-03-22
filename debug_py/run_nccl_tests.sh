#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git not found" >&2
  exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found" >&2
  exit 1
fi

HAVE_MPIRUN=0
HAVE_MPICC=0
if command -v mpirun >/dev/null 2>&1; then
  HAVE_MPIRUN=1
fi
if command -v mpicc >/dev/null 2>&1; then
  HAVE_MPICC=1
fi

CUDA_HOME="${CUDA_HOME:-$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)}"
MPI_HOME="${MPI_HOME:-/usr}"
NCCL_TESTS_DIR="${NCCL_TESTS_DIR:-${ROOT_DIR}/.cache/nccl-tests}"
NCCL_TESTS_REF="${NCCL_TESTS_REF:-master}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NP="${NP:-2}"
MIN_BYTES="${MIN_BYTES:-8}"
MAX_BYTES="${MAX_BYTES:-128M}"
STEP_FACTOR="${STEP_FACTOR:-2}"
USE_MPI="${USE_MPI:-auto}"
NGPUS="${NGPUS:-}"
NAME_SUFFIX="${NAME_SUFFIX:-_mpi}"

NCCL_HOME="${NCCL_HOME:-$("$PYTHON_BIN" - <<'PY'
import os
try:
    import nvidia.nccl
except Exception as exc:
    raise SystemExit(f"failed to import nvidia.nccl: {exc}")
print(os.path.dirname(nvidia.nccl.__file__))
PY
)}"

mkdir -p "$(dirname "${NCCL_TESTS_DIR}")"

if [[ ! -d "${NCCL_TESTS_DIR}/.git" ]]; then
  git clone https://github.com/NVIDIA/nccl-tests.git "${NCCL_TESTS_DIR}"
fi

git -C "${NCCL_TESTS_DIR}" fetch --tags origin
git -C "${NCCL_TESTS_DIR}" checkout "${NCCL_TESTS_REF}"

export CUDA_HOME
export MPI_HOME
export NCCL_HOME
export CUDA_VISIBLE_DEVICES
export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"

if [[ "${USE_MPI}" == "auto" ]]; then
  if [[ "${HAVE_MPIRUN}" == "1" && "${HAVE_MPICC}" == "1" ]]; then
    USE_MPI=1
  else
    USE_MPI=0
  fi
fi

if [[ "${USE_MPI}" == "1" ]]; then
  if [[ "${HAVE_MPIRUN}" != "1" || "${HAVE_MPICC}" != "1" ]]; then
    echo "USE_MPI=1 requested but mpirun/mpicc not found" >&2
    exit 1
  fi
  if [[ -z "${NGPUS}" ]]; then
    NGPUS=1
  fi
else
  NAME_SUFFIX=""
  if [[ -z "${NGPUS}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
      IFS=',' read -r -a _gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
      NGPUS="${#_gpu_list[@]}"
    else
      NGPUS=2
    fi
  fi
fi

echo "Using PYTHON_BIN=${PYTHON_BIN}"
echo "Using CUDA_HOME=${CUDA_HOME}"
echo "Using MPI_HOME=${MPI_HOME}"
echo "Using NCCL_HOME=${NCCL_HOME}"
echo "Using NCCL_TESTS_DIR=${NCCL_TESTS_DIR}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Using USE_MPI=${USE_MPI}"
echo "Using NGPUS=${NGPUS}"
echo "Building nccl-tests..."

if [[ "${USE_MPI}" == "1" ]]; then
  make -C "${NCCL_TESTS_DIR}" -j \
    MPI=1 \
    MPI_HOME="${MPI_HOME}" \
    CUDA_HOME="${CUDA_HOME}" \
    NCCL_HOME="${NCCL_HOME}" \
    NAME_SUFFIX="${NAME_SUFFIX}"
else
  make -C "${NCCL_TESTS_DIR}" -j \
    CUDA_HOME="${CUDA_HOME}" \
    NCCL_HOME="${NCCL_HOME}"
fi

BIN="${NCCL_TESTS_DIR}/build/all_reduce_perf${NAME_SUFFIX}"
if [[ ! -x "${BIN}" ]]; then
  echo "Expected binary not found: ${BIN}" >&2
  exit 1
fi

echo "Running ${BIN}"
set -x
if [[ "${USE_MPI}" == "1" ]]; then
  mpirun -np "${NP}" -bind-to none -map-by slot \
    -x CUDA_VISIBLE_DEVICES \
    -x LD_LIBRARY_PATH \
    -x NCCL_DEBUG \
    -x NCCL_DEBUG_SUBSYS \
    -x NCCL_IB_DISABLE \
    -x NCCL_P2P_DISABLE \
    -x NCCL_P2P_LEVEL \
    -x NCCL_CUMEM_ENABLE \
    -x NCCL_CUMEM_HOST_ENABLE \
    "${BIN}" \
    -b "${MIN_BYTES}" \
    -e "${MAX_BYTES}" \
    -f "${STEP_FACTOR}" \
    -g "${NGPUS}"
else
  "${BIN}" \
    -b "${MIN_BYTES}" \
    -e "${MAX_BYTES}" \
    -f "${STEP_FACTOR}" \
    -g "${NGPUS}"
fi
