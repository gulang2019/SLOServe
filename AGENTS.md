# Repository Guidelines

## Project Structure & Module Organization
- Core engine, routing, schedulers, and operators live in `SLOsServe/` (e.g., `router/api_server_ray.py`, `scheduler/`, `ops/`).
- Benchmark and SLO evaluation scripts are in `motivation/` (`bench_api_server.py`, `auto_scaling.py`); datasets download via `Dataset/`.
- Third-party dependencies sit under `3rdparty/` and `3rdpartys/` (vLLM, TensorRT-LLM, sglang, Mooncake). Treat these as vendored.
- C++/CUDA extensions are in `csrc/` with accompanying `test.cc`/`test.py`; assets and configs such as `auto_scaling_model.json` live at the repo root.

## Setup, Build, and Development Commands
- Initialize submodules and environment:
  - `git submodule update --init`
  - `conda create --name slosserve python=3.10 -y && conda activate slosserve`
  - `cd 3rdparty/vllm && VLLM_USE_PRECOMPILED=1 pip install --editable . && cd -`
  - `cd csrc && pip install -e . --no-build-isolation && cd -`
- Download traces: `PYTHONPATH=. python Dataset/download_dataset.py`
- Launch router/API server (mock mode for local dev): `python -m SLOsServe.router.api_server_ray --mock_connector --mock_engine --host 0.0.0.0 --port 8000`
- Run a benchmark sweep example: see `README.md` or `python motivation/bench_api_server.py --help`.
- E2E smoke (single GPU): `source ./run_unit.sh 0 8000 slosserve-edf 0.5`.

## Coding Style & Naming Conventions
- Python: follow PEP 8 with 4-space indents; prefer type hints and dataclasses where helpful. Keep module-level constants upper snake case, functions lower snake, classes CapWords.
- C++/CUDA: mirror existing style in `csrc/` (`snake_case` variables, `CamelCase` types). Keep headers minimal and include guards consistent.
- Use docstrings for public APIs; keep logging lightweight and actionable.
- Academic codebase: keep style simple and uniform; do not add alternative input styles or branching ergonomics. Avoid `try/except` wrappers that hide real errors—fail loudly so issues surface.

## Testing Guidelines
- Prefer targeted tests near the code under test (Python: colocate `*_test.py`; C++: keep small test drivers like `csrc/test.cc`).
- For Python benches, validate via `python motivation/bench_api_server.py ...` with a representative trace; capture outputs under `experiments_*`.
- Before pushing, ensure router/server starts with `--help` and a short mock run to catch import issues.

## Commit & Pull Request Guidelines
- Commit messages should be short, imperative summaries (e.g., `add auto_scaler_params`). Group related changes together; avoid noisy vendor diffs.
- PRs: include a concise description, key commands run (benchmarks/tests), and linked issue/trace where applicable. Add screenshots or logs for benchmark results if they affect performance.
- Keep changes scoped; flag any edits under `3rdparty*`/`3rdpartys*` so reviewers can focus on first-party code.

## Security & Configuration Tips
- Avoid committing downloaded traces or experiment outputs; use `.gitignore` for local artifacts.
- GPU/port usage: check availability before launching services (`lsof -Pn -iTCP:<port>`). Set `CUDA_VISIBLE_DEVICES` explicitly in scripts when sharing machines.
- If you are unsure about a change, pause and ask the maintainer before making invasive edits.
