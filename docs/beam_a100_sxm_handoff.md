# BEAM on A100 SXM: Figure 7 handoff

Goal: regenerate the BEAM curves in the top row of Figure 7 (ChatBot, Coder
and Mixed, Qwen2.5-7B) on `NVIDIA A100-SXM4-80GB`, with BEAM running on the
same SLOServe backend as the other systems. Only one GPU is available, so each
N-GPU point is measured by replaying round-robin shards of the N-way split one
at a time.

Branch: `beam-a100-sxm` (main repo and the `3rdparty/vllm` submodule).

## How a point is measured

With round-robin routing, request *i* goes to GPU *i* mod N and GPUs never
interact, so one shard replayed alone is exactly one GPU of the full run.
The benchmark does this when `--n_devices N` exceeds the GPUs given in
`--clients`: it keeps shard `SLOSSERVE_RR_SHARD_OFFSET` (default 0), keeps the
original timestamps, and reports energy x N. SLO violation of the point is the
shard's violation rate. With more than one shard of a point, average their
violation rates (weighted by request count) and their per-shard energies.

## 0. Setup (once)

```sh
git fetch && git switch beam-a100-sxm
git submodule update --init 3rdparty/vllm 3rdparty/BEAM
# Build/install as for any SLOServe checkout (.venv, vllm editable install).
git -C 3rdparty/BEAM checkout -- .     # only if an older baseline patch is applied
./scripts/setup_beam.sh                # BEAM container + patched BEAM vLLM
export PYTHONPATH=$PWD CUDA_DEVICE_ORDER=PCI_BUS_ID
GPU=<free GPU index as nvidia-smi lists it>
```

Checks before anything else:

1. `nvidia-smi -i $GPU --query-gpu=name --format=csv,noheader` prints
   `NVIDIA A100-SXM4-80GB`, and `nvidia-smi -i $GPU` shows no processes.
2. Clock control: `nvidia-smi -i $GPU -lgc 1410 && nvidia-smi -i $GPU -rgc`.
   If it reports insufficient permissions, every SLOServe server below must
   be started as root (`sudo -E env "PATH=$PATH" .venv/bin/python ...`).
   The engine now refuses to start BEAM without clock permission.
3. TTFT SLOs must equal those of the existing Figure 7 baselines. They come
   from this machine's `assets/perf_model.json` (untracked) through the
   benchmark's `perf_model_task`. Print what the benchmark will use:

   ```sh
   .venv/bin/python - <<'EOF'
   import motivation.bench_api_server as b, SLOsServe.perf_model as pm
   for s in ["azure_chat_23:azure_chat_23", "azure_code_23:azure_code_23",
             "azure_code_23:azure_code_23+azure_chat_23:azure_chat_23"]:
       t = b._perf_model_task_for_trace_spec(s)
       print(s, t, pm.get_hardware_params("Qwen/Qwen2.5-7B-Instruct", t))
   EOF
   ```

   Zero-load TTFT is `p[0]*prompt + p[4]` (+ tiny terms). The native-BEAM
   Figure 7 runner scored with chat `5.55e-5, 0.014`, code
   `6.139e-5, 0.0109`, mixed `6.070e-5, 0.01105`. If this machine prints
   something else for a trace, find out which the paper's baseline runs used
   before running BEAM; do not edit `perf_model.json` silently.

## 1. Profile the GPU (BEAM's own profiler)

```sh
BEAM_GPU=$GPU BEAM_PROFILE_NUM_CLOCKS=3 ./scripts/profile_beam_qwen7b.sh   # smoke, ~10 min
mv "3rdparty/BEAM/benchmarks/energy/offline_profile_results/dvfs_profile_NVIDIA A100-SXM4-80GB_Qwen_Qwen2.5-7B-Instruct_tp1_pp1_one.csv"{,.smoke}
BEAM_GPU=$GPU ./scripts/profile_beam_qwen7b.sh                             # full, hours
export SLOSSERVE_BEAM_PROFILE="$PWD/3rdparty/BEAM/benchmarks/energy/offline_profile_results/dvfs_profile_NVIDIA A100-SXM4-80GB_Qwen_Qwen2.5-7B-Instruct_tp1_pp1_one.csv"
```

During the smoke run, `watch nvidia-smi -i $GPU -q -d CLOCK` should show the
clock stepping on `$GPU` only; the profiler log names the GPU it profiles.
Also record the idle power with the model loaded (any `nvidia-smi` reading
while a server is up and idle); simulation needs it (`mock_idle_power_w`).

## 2. Start the SLOServe server (one engine on $GPU)

```sh
export CUDA_VISIBLE_DEVICES=$GPU SLOSSERVE_BEAM_BACKEND=real
.venv/bin/python -m SLOsServe.router.api_server_ray --stat_window 2 \
  --host 0.0.0.0 --port 8000 --window_size 0.005 --router slosserve \
  --router_kwargs '{"device_mem":1248576, "block_size":16, "model_name":"Qwen/Qwen2.5-7B-Instruct", "tpot":0.05, "scheduling_overhead":0.003, "scheduling_safety_margin":0.002, "max_decode_length":500, "is_pd_disagg":0, "n_prefill_per_group":1, "max_decode_bs":16, "enable_rerouting":0}' \
  --clients 0 --admission_mode anytime --mock_connector 2>&1 | tee server.log
```

(Prefix with `sudo -E env "PATH=$PATH"` if check 2 needed root. Keep
`SLOSSERVE_BEAM_PROFILE` exported. Do not set
`SLOSSERVE_MOCK_PROFILE_CLOCK_MHZ` on this machine: it would redefine the SLOs
from the profile.)

## 3. Validate before the real runs

a. Latency check, about 15 min. Lock the clock, run a non-BEAM policy on one
   shard, unlock, and compare batch times with the profile. Sarathi chunks
   prefills, so its batches stay inside the profile's 512-token range:

```sh
nvidia-smi -i $GPU -lgc 1410
.venv/bin/python motivation/bench_api_server.py --model_name Qwen/Qwen2.5-7B-Instruct \
  --policies round_robin:sarathi --n_devices 8 --clients 0 --port 8000 \
  --trace azure_chat_23:azure_chat_23 --window t1200:1800 --load_scales 1.0 \
  --ttft_slo_scales 3 --slo_tpots 0.025 --profit constant --admission_mode arrival \
  --output_dir runs/a100sxm_beam/latency_check
nvidia-smi -i $GPU -rgc
.venv/bin/python scripts/compare_beam_profile_latency.py \
  runs/a100sxm_beam/latency_check/*/*.events.jsonl \
  --profile "$SLOSSERVE_BEAM_PROFILE" --clock 1410
```

   Expect `measured / profile` p50 within ~10% of 1.0 for both batch kinds
   (sampling is included in "measured", so slightly above 1 is normal). If it
   is far off, stop: BEAM's decisions would use the wrong latency model.

b. BEAM smoke test, about 3 min: the first command of section 4 with
   `--window t1200:1300 --n_devices 8`. In `server.log` there must be no
   NVML errors; while it runs, `nvidia-smi -i $GPU -q -d CLOCK` should move
   between clocks; the events file must contain `beam_decision` events with
   phases `s1`, `s2`, `decode_clock`.

## 4. Runs for Figure 7

In the benchmark's shell, export the same `PYTHONPATH`,
`SLOSSERVE_BEAM_PROFILE` and `SLOSSERVE_BEAM_BACKEND=real` as the server's
(the benchmark refuses `beam` otherwise). These specs replay exactly the
requests of the native-BEAM Figure 7 exports (4123 / 2112 / 6235 requests).

Fixed for every run: `round_robin:beam`, `--window t1200:1800`,
`--load_scales 1.0`, `--ttft_slo_scales 3`, `--profit constant`,
`--admission_mode arrival`, `--scheduling_overhead 0.003`,
`--scheduling_safety_margin 0.002`, `--clients 0`. Each run replays a
600 s window; budget ~12 min per (trace, N, shard).

| Trace | `--trace` | `--slo_tpots` | `--slo_routing_overhead` |
|---|---|---|---|
| ChatBot | `azure_chat_23:azure_chat_23` | 0.025 | 0.05 |
| Coder | `azure_code_23:azure_code_23` | 0.025 | 0.05 |
| Mixed | `azure_code_23:azure_code_23+azure_chat_23:azure_chat_23` | 0.02 | 0.02 |
| Large-Scale Chatting | `azure_chat:azure_chat` | 0.02 | 0.05 |

Trace specs and allowances follow the native-BEAM Figure 7 export
(`scripts/export_beam_figure7_traces.py`, `scripts/run_beam_figure7_7b.sh`)
and, for Large-Scale Chatting, `configs/batch/sim_fig7/large_scale_chat.json`
(13627 requests, ~23 req/s; its zero-load TTFT uses the model's `default`
perf-model entry). Confirm they match the baselines in
`Paper/data/e2e/{chat,code,mixed}_7B.csv` and `chatting_large_scale.csv`
(Coder in particular: some newer configs use `sharegpt_code` lengths).

The figure shows points with at most 10% SLO violation. Earlier BEAM runs
were visible at ChatBot N = 4-8, Coder N = 6-16, Mixed N = 10-16 and
Large-Scale Chatting N = 16 only (N = 8-14 were above 10%; the baselines
there are visible from N = 10). The lists below cover those plus the point
just below each threshold, so the curve's left end is found even if A100
SXM shifts it.

| Tier | What | Runs | Time |
|---|---|---|---|
| 1 | shard 0 of: ChatBot N = 3-8; Coder N = 5, 6, 8, 10, 12, 14, 16; Mixed N = 8, 10, 12, 14, 16; Large-Scale N = 8, 10, 12, 14, 16 | 23 | ~4.6 h |
| 2 | shard N-1 of every Tier-1 point with <= 10% violation | <= 23 | <= 4.6 h |
| 3 | native BEAM on ChatBot 8/0, Coder 10/0, Mixed 12/0 (fidelity, section 5) | 3 | ~40 min |
| 4 | Coder N = 7 (shard 0), only if the curve looks odd between 6 and 8 | 1 | ~12 min |

If time is short, run Tier 1 in this order: Mixed, Large-Scale, Coder,
ChatBot (heaviest load first, where BEAM's points are least certain).

Tier 1 commands (one per trace; the benchmark runs the listed N in order):

```sh
run() {  # trace tpot allowance offset outdir n...
  local trace=$1 tpot=$2 allow=$3 off=$4 out=$5; shift 5
  SLOSSERVE_RR_SHARD_OFFSET=$off .venv/bin/python motivation/bench_api_server.py \
    --model_name Qwen/Qwen2.5-7B-Instruct --policies round_robin:beam \
    --n_devices "$@" --clients 0 --port 8000 --trace "$trace" \
    --window t1200:1800 --load_scales 1.0 --ttft_slo_scales 3 --slo_tpots "$tpot" \
    --profit constant --admission_mode arrival --slo_routing_overhead "$allow" \
    --scheduling_overhead 0.003 --scheduling_safety_margin 0.002 \
    --output_dir "$out" --disable_perf_model_error_figure
}
run azure_chat_23:azure_chat_23 0.025 0.05 0 runs/a100sxm_beam/shard0 8 7 6 5 4 3
run azure_code_23:azure_code_23 0.025 0.05 0 runs/a100sxm_beam/shard0 16 14 12 10 8 6 5
run azure_code_23:azure_code_23+azure_chat_23:azure_chat_23 0.02 0.02 0 runs/a100sxm_beam/shard0 16 14 12 10 8
run azure_chat:azure_chat 0.02 0.05 0 runs/a100sxm_beam/shard0 16 14 12 10 8
```

Not in this list (bottom row of Figure 7):

- Multi-Round Chatting (`sharegpt_chat:azure_chat`, `round_robin_session`,
  session replay). The benchmark's shard slicing splits by request index,
  but session routing assigns whole sessions, so a sliced shard is not one
  GPU's workload. It needs session-aware slicing (or the native session
  shards in `datasets/sloserve_fig7_session_round_robin`) before it can be
  run on one GPU.
- ChatBot (32B). It needs its own BEAM profile (Qwen2.5-32B, hours), and
  earlier BEAM runs had no point under 10%. Skip unless asked.

Tier 2: the last shard of N is offset N-1, so run each N on its own,
e.g. `run azure_chat_23:azure_chat_23 0.025 0.05 7 runs/a100sxm_beam/shard_last 8`.
Use a separate `--output_dir` per offset (results are cached per setting).

Each run prints an SLO grid; the per-run row (violation rate, energy scaled
x N, `rr_shard_offset`) is in `<output_dir>/*/results.jsonl`.

## 5. Fidelity check against native BEAM (Tier 3)

Stop the SLOServe server (the GPU must be free), then on the same GPU:

```sh
python scripts/split_beam_figure7_rr_load.py
BEAM_GPU=$GPU BEAM_PORT=8008 BEAM_MODE=s1s2 ./scripts/serve_beam_qwen7b.sh
BEAM_GPU=$GPU BEAM_PORT=8008 ./scripts/run_beam_figure7_7b.sh --trace chat_7b --split 8/0 \
  --output-root results/a100sxm/native/chat_7b_8_0
#   ... code_7b --split 10/0, mixed_7b --split 12/0
BEAM_GPU=$GPU BEAM_PORT=8008 ./scripts/stop_beam_qwen7b.sh
```

Compare with the same (trace, N, shard 0) from Tier 1: energy per shard and
p50/p90 TTFT and TPOT. Native BEAM takes one global TTFT target (the median
budget) and the runner scores with its own zero-load parameters (section 0,
check 3); for violation rates, rescore `beam_detailed.json` with
`scripts/score_beam_detailed_results.py` using the parameters the benchmark
printed. Agreement within GPU-to-GPU noise supports using the shared-backend
numbers in the paper.

## What to bring back

- The full profile CSV (and the `.smoke` one), plus the idle power reading.
- `runs/a100sxm_beam/**` (`results.jsonl`, `*.report_slo_grid.*`,
  `*.events.jsonl`) and `results/a100sxm/native/**` under
  `3rdparty/BEAM/benchmarks/energy/`.
- `compare_beam_profile_latency.py` output, `server.log`, and a note of
  anything that deviated from this document.

## Known limitations

- BEAM in SLOServe supports TP=1, PP=1 only, and does not port BEAM's
  prefill/decode-disaggregated mode.
- Differences from upstream BEAM (per-request TTFT budgets, 512-token first
  batch, preempted requests resume after the waiting queue) are listed in
  the README section "Integrated BEAM policies".
- Shard energies are per-shard measurements scaled by N, not a simultaneous
  N-GPU measurement.
