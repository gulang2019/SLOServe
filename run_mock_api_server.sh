#!/usr/bin/env bash
set -euo pipefail

# Minimal mock API server for local testing (no GPUs, no model load).
python -m SLOsServe.router.api_server_ray \
  --host 0.0.0.0 \
  --port 8000 \
  --router round_robin \
  --router_kwargs '{"max_decode_batch_size":32}' \
  --clients mock0 \
  --mock_connector \
  --mock_engine
