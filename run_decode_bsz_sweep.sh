#!/usr/bin/env bash
set -euo pipefail

# Sweep over different DECODE_MAX_BATCH_TOKENS values
# PREFILL_MAX_BATCH_TOKENS is kept constant at 2048

DECODE_BSZ_VALUES=(64 128 256 512)
PORT=8000
WAIT_TIME=20
TRACE=ExpD_azure_chat_23:ExpD_azure_chat_23
DIRECTORY=experiments_results/ExpD_azure_chat_23

for DECODE_BSZ in "${DECODE_BSZ_VALUES[@]}"; do
    echo "=========================================="
    echo "Running experiment with DECODE_MAX_BATCH_TOKENS=${DECODE_BSZ}"
    echo "=========================================="

    # Update emulator_config.py - only change the DECODE_MAX_BATCH_TOKENS line
    sed -i.bak "s/^DECODE_MAX_BATCH_TOKENS = .*/DECODE_MAX_BATCH_TOKENS = ${DECODE_BSZ}/" emulator_config.py
    echo "Updated emulator_config.py with DECODE_MAX_BATCH_TOKENS=${DECODE_BSZ}"

    # Create output directory
    OUTPUT_DIR="${DIRECTORY}/sarathi_rr_bszd${DECODE_BSZ}"
    mkdir -p "${OUTPUT_DIR}"

    # Start API server in background, redirect output to log file
    echo "Starting API server on port ${PORT}..."
    python -m SLOsServe.router.api_server_ray \
        --host 0.0.0.0 \
        --port ${PORT} \
        --router round_robin \
        --router_kwargs '{"max_decode_batch_size":32}' \
        --clients mock0 \
        --mock_connector \
        --mock_engine > "${OUTPUT_DIR}/server_log.txt" 2>&1 &

    SERVER_PID=$!
    echo "API server started with PID ${SERVER_PID} (log: ${OUTPUT_DIR}/server_log.txt)"

    # Wait for server to be ready
    echo "Waiting ${WAIT_TIME} seconds for server to initialize..."
    sleep ${WAIT_TIME}

    # Run benchmark
    echo "Running benchmark..."
    export PYTHONPATH="${PYTHONPATH:-.}:."
    python motivation/bench_api_server.py --overwrite \
        --n_devices 1 \
        --policies round_robin:sarathi \
        --load_scales 1.0 \
        --slo_tpots 0.025 \
        --ttft_slo_scales 3.0 \
        --window 3979:4580 \
        --trace ${TRACE} \
        --profit constant --admission_mode anytime \
        --slo_routing_overhead 0.05 --scheduling_overhead 0.005 \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --port ${PORT} \
        --clients 0 \
        --output_dir "${OUTPUT_DIR}"

    echo "Benchmark completed for DECODE_BSZ=${DECODE_BSZ}"

    # Kill the API server and all child processes
    echo "Stopping API server (PID ${SERVER_PID})..."
    # Kill the process group to ensure all child processes are terminated
    pkill -P ${SERVER_PID} 2>/dev/null || true
    kill ${SERVER_PID} 2>/dev/null || true
    sleep 2
    # Force kill if still running
    kill -9 ${SERVER_PID} 2>/dev/null || true

    # Also clean up any remaining Ray processes
    pkill -f "ray::" 2>/dev/null || true

    # Wait a bit before next iteration
    sleep 5

    echo "Experiment with DECODE_BSZ=${DECODE_BSZ} completed"
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "Results stored in experiments_results/sarathi_rr_bszd*/"
echo "=========================================="
