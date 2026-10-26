# BEAM on 8x A100 SXM: Figure 7 bottom row

Goal: BEAM curves for the bottom row of Figure 7 on `NVIDIA A100-SXM4-80GB`,
on the same SLOServe backend as the baselines:

| Panel | Model / server | Trace | Routing | TTFT x | TPOT | N (servers) |
|---|---|---|---|---|---|---|
| Large-Scale Chatting | Qwen2.5-7B, 1 GPU | `azure_chat:azure_chat` | `round_robin` | 3 | 0.02 | 8, 10, 12, 14, 16 |
| Multi-Round Chatting | Qwen2.5-7B, 1 GPU | `sharegpt_chat:azure_chat`, session replay, 5 s pause | `round_robin_session` | 5 | 0.05 | 8, 12, 16 |
| ChatBot (32B) | Qwen3-32B, 2 GPUs (TP=2) | `azure_chat_23:azure_chat_23` | `round_robin` | 5 | 0.05 | 2-8 |

All runs use `--window t1200:1800`, load 1.0, `--routing_fallback_policy asap`,
`--slo_routing_overhead 0.05`, the same settings as the baselines in
`figure_data/beam_paper/original_e2e/{chatting_large_scale,e2e_mulitiround_chatting,32B_chat}.csv`
and `configs/batch/sim_fig7/*.json`.

Branch: `beam-a100-sxm` (main repo and `3rdparty/vllm`).

## Read first: the earlier Tier-1 BEAM runs are affected by a bug

Every request reaches an engine twice: a prefill stage, then a decode stage
whose KV arrives through the connector. BEAM counted the prompt at both
arrivals but drained it once, so its pending-prefill counter grew without
bound (about 1.5M tokens after a 60 s multi-round replay). S1 then planned
for a huge backlog: mostly 1410 MHz instead of 900-990 MHz. This affects
**every** BEAM run so far, including the Tier-1 top-row runs from
`docs/beam_a100_sxm_handoff.md`. It is fixed now: only the prefill stage
is a BEAM arrival, as in upstream, where each request arrives once. The
top-row runs should be redone (Phase 3).

## How a point is measured on 8 GPUs

The round-robin shard slicing from the first handoff still applies:
`--n_devices N` with fewer engines in `--clients` replays shards
`SLOSSERVE_RR_SHARD_OFFSET .. offset+k-1` of the N-way split, with original
timestamps, and reports energy x N/k.

- Large-Scale and 32B (request round robin): shards are statistically alike,
  so one run that fills all engines (k of N shards) is the point.
- Multi-Round: whole sessions are dealt out in turn (session *s* goes to
  GPU *s* mod N), and session lengths vary a lot. Shards of the 8-way split
  hold 361-647 requests, so one shard x N is not a good estimate. **Every
  shard of N is replayed**, over several runs at different offsets, and
  `scripts/merge_beam_shard_runs.py` sums their raw energies.

`results.jsonl` rows now record `rr_shard_offset`. Run every
`results.jsonl` through the merge script, even single runs (it passes them
through).

## 0. Setup

```sh
git -c fetch.recurseSubmodules=no pull     # branch beam-a100-sxm
git submodule update --init 3rdparty/vllm 3rdparty/BEAM
git -C 3rdparty/vllm log -1 --oneline      # must be the commit pinned by HEAD
export PYTHONPATH=$PWD CUDA_DEVICE_ORDER=PCI_BUS_ID SLOSSERVE_BEAM_BACKEND=real
export BEAM7B="$PWD/assets/beam_profiles/dvfs_profile_NVIDIA A100-SXM4-80GB_Qwen_Qwen2.5-7B-Instruct_tp1_pp1_one.csv"
```

The 7B profile from the Tier-1 machine is committed (31 clocks, 510-1410 MHz).

### Required data

Datasets are gitignored. Copy them from the main workstation
(`/ssd1/siyuanch/workspace/SLOServe/assets/datasets/`) into
`assets/datasets/`, then compare checksums:

```
3867f743c942f72b071d7926b997c5676bf1b99a07edf736ba4d7311b3ba19e2  azure_chat.arrival.pkl
a4397fab0af30568a0429949e579e575ecc550e89effd7fc345e8310b703d23d  azure_chat.requests.pkl
2a36bf0fb081e9af6699cbe7758d30dbf4642623b439e88aaad73bf4c566ae35  azure_chat_23.arrival.pkl
0515d4c3aa8052685643c67b650ee77d5d36b91b8e761f09d2fbc4dda1b8e5f4  azure_chat_23.requests.pkl
50f7e2ed227c5c2ac785bbec3289513591daf87fbe7b9a90cf09f10a40eb30dd  sharegpt_chat.requests.pkl   (1.5 GB)
```

Copy `sharegpt_chat.requests.pkl` rather than regenerating it.
Multi-round prompts and sessions are built from it, and a rebuilt file
could differ.

Models: `Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen3-32B` (about 65 GB), in
the HF cache that both the SLOServe venv and the BEAM container use.

### Checks

1. `nvidia-smi --query-gpu=index,name --format=csv` shows eight
   `NVIDIA A100-SXM4-80GB`, with no processes running.
2. Clock control on all GPUs: `sudo nvidia-smi -lgc 1410 && sudo nvidia-smi -rgc`.
   If clock locking needs root, start every SLOServe server below as root,
   for example `sudo -E env "PATH=$PATH" .venv/bin/python ...`.
3. TTFT SLOs must match the baselines. Print what the benchmark will use:

   ```sh
   .venv/bin/python - <<'EOF'
   import motivation.bench_api_server as b, SLOsServe.perf_model as pm
   for m, s in [("Qwen/Qwen2.5-7B-Instruct", "azure_chat:azure_chat"),
                ("Qwen/Qwen2.5-7B-Instruct", "sharegpt_chat:azure_chat"),
                ("Qwen/Qwen3-32B", "azure_chat_23:azure_chat_23")]:
       t = b._perf_model_task_for_trace_spec(s)
       print(m, s, t, pm.get_hardware_params(m, t))
   EOF
   ```

   Expected for both 7B traces: `[6.1e-05, 1.6e-09, 3.5e-08, 0.0, 0.01]`.
   **32B is a blocker.** The main workstation's `assets/perf_model.json`
   has no Qwen3-32B entry, so it falls back to `default`
   (`[4.86e-05, 1.69e-05, 8e-08, 0.0, 0.014]`). That gives about a 60 ms
   zero-load TTFT for 1000 tokens, while the paper states about 500 ms /
   3 = 170 ms for 32B. Get the Qwen3-32B entry that the 32B baselines were
   scored with, and add it to `assets/perf_model.json` before Phase 2. Do
   not fit a new one: that would change the SLOs and leave the baselines
   incomparable.

## Phase 1, in parallel (about 4.5 h)

GPUs 6-7 profile Qwen3-32B for BEAM. GPUs 0-5 run both 7B panels.
Profiling meters only its own GPUs, so the two jobs do not affect each
other's energy.

### 1a. Profile Qwen3-32B, TP=2, on GPUs 6,7 (hours)

```sh
BEAM_MODEL=Qwen/Qwen3-32B BEAM_TP=2 BEAM_GPU=6,7 BEAM_PROFILE_NUM_CLOCKS=3 \
  ./scripts/profile_beam_qwen7b.sh                 # smoke, check it runs
mv "3rdparty/BEAM/benchmarks/energy/offline_profile_results/dvfs_profile_NVIDIA A100-SXM4-80GB_Qwen_Qwen3-32B_tp2_pp1_one.csv"{,.smoke}
BEAM_MODEL=Qwen/Qwen3-32B BEAM_TP=2 BEAM_GPU=6,7 ./scripts/profile_beam_qwen7b.sh
```

The profiler locks clocks on both GPUs and sums their energy; one row is
one TP=2 server. During the smoke run, `nvidia-smi -q -d CLOCK -i 6,7`
should show both GPUs stepping together.

### 1b. 7B server on GPUs 0-5

```sh
export SLOSSERVE_BEAM_PROFILE="$BEAM7B"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 .venv/bin/python -m SLOsServe.router.api_server_ray \
  --stat_window 2 --host 0.0.0.0 --port 8000 --window_size 0.005 --router slosserve \
  --router_kwargs '{"device_mem":1248576, "block_size":16, "model_name":"Qwen/Qwen2.5-7B-Instruct", "tpot":0.05, "scheduling_overhead":0.003, "scheduling_safety_margin":0.002, "max_decode_length":500, "is_pd_disagg":0, "n_prefill_per_group":1, "max_decode_bs":16, "enable_rerouting":0}' \
  --clients 0-5 --admission_mode anytime --mock_connector 2>&1 | tee server7b.log
```

Benchmark helper (in another shell, with the same exports):

```sh
bench() {  # outdir offset clients n_devices...   uses MODEL POLICY TRACE TTFT TPOT EXTRA [WINDOW]
  local out=$1 off=$2 clients=$3; shift 3
  SLOSSERVE_RR_SHARD_OFFSET=$off .venv/bin/python motivation/bench_api_server.py \
    --model_name "$MODEL" --policies "$POLICY" --n_devices "$@" --clients "$clients" \
    --port 8000 --trace "$TRACE" --window "${WINDOW:-t1200:1800}" --load_scales 1.0 \
    --ttft_slo_scales "$TTFT" --slo_tpots "$TPOT" --profit constant \
    --admission_mode arrival --routing_fallback_policy asap \
    --slo_routing_overhead 0.05 --scheduling_overhead 0.003 \
    --scheduling_safety_margin 0.002 $EXTRA \
    --output_dir "$out" --disable_perf_model_error_figure
}
```

**Smoke test first** (about 5 min). Replay 2 minutes, then confirm that
BEAM's pending-prefill counter stays bounded. This is the check that fails
with the old code:

```sh
MODEL=Qwen/Qwen2.5-7B-Instruct POLICY=round_robin_session:beam TRACE=sharegpt_chat:azure_chat \
TTFT=5 TPOT=0.05 EXTRA="--enable_session_replay --session_pause_s 5 --overwrite" \
WINDOW=t1200:1320 bench runs/fig7b/smoke 0 0-5 12
.venv/bin/python - <<'EOF'
import glob, json, collections
for f in glob.glob("runs/fig7b/smoke/*/*beam*.events.jsonl"):
    ev = json.load(open(f))
    w = [e["extra_args"]["num_waiting_prefill_tokens"] for e in ev
         if e.get("event_type") == "beam_decision" and e["extra_args"]["phase"] == "s1"]
    clocks = collections.Counter(e["extra_args"]["clock_mhz"] for e in ev
         if e.get("event_type") == "beam_decision" and e["extra_args"].get("applied"))
    print(f, "s1:", len(w), "max pending:", max(w), "last:", w[-3:], clocks.most_common(5))
EOF
```

Expect a maximum pending count in the thousands, ending near 0 (it can go
slightly negative). Millions means the old code is running. Also check:
there are no NVML errors in `server7b.log`, and `nvidia-smi -q -d CLOCK`
moves on GPUs 0-5 only.

**Large-Scale Chatting**: 6 of N shards per point, 5 runs, about 1 h:

```sh
MODEL=Qwen/Qwen2.5-7B-Instruct POLICY=round_robin:beam TRACE=azure_chat:azure_chat \
TTFT=3 TPOT=0.02 EXTRA= \
  bench runs/fig7b/large/off0 0 0-5 16 14 12 10 8
```

**Multi-Round Chatting**: all shards of every N, 8 runs. Each run lasts
longer than the 600 s window because every session drains; budget about
25 min per run, about 3.3 h total:

```sh
export MODEL=Qwen/Qwen2.5-7B-Instruct POLICY=round_robin_session:beam \
  TRACE=sharegpt_chat:azure_chat TTFT=5 TPOT=0.05 \
  EXTRA="--enable_session_replay --session_pause_s 5"
bench runs/fig7b/multiround/off0_k6  0  0-5 16 12 8   # shards 0-5 of 16, 12, 8
bench runs/fig7b/multiround/off6_k6  6  0-5 16 12     # shards 6-11 of 16 and 12
bench runs/fig7b/multiround/off12_k4 12 0-3 16        # shards 12-15 of 16
bench runs/fig7b/multiround/off6_k2  6  0-1 8         # shards 6-7 of 8
```

Use one `--output_dir` per (offset, clients) pair, because results are
cached per setting. If time is short, run N=16 and N=12 before N=8: the
baselines are visible at 12-16, and at 8 only for SLO-Packer.

## Phase 2: ChatBot (32B), all 8 GPUs (about 1.5 h)

Before starting, check 3 (the Qwen3-32B perf-model entry) must be resolved.
Stop the 7B server, then start the 32B server with four TP=2 engines:

```sh
export SLOSSERVE_BEAM_PROFILE="$PWD/3rdparty/BEAM/benchmarks/energy/offline_profile_results/dvfs_profile_NVIDIA A100-SXM4-80GB_Qwen_Qwen3-32B_tp2_pp1_one.csv"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/python -m SLOsServe.router.api_server_ray \
  --stat_window 2 --host 0.0.0.0 --port 8000 --window_size 0.005 --router slosserve \
  --model_name Qwen/Qwen3-32B --tensor_parallel_size 2 \
  --router_kwargs '{"device_mem":1248576, "block_size":16, "model_name":"Qwen/Qwen3-32B", "tpot":0.05, "scheduling_overhead":0.003, "scheduling_safety_margin":0.002, "max_decode_length":500, "is_pd_disagg":0, "n_prefill_per_group":1, "max_decode_bs":16, "enable_rerouting":0}' \
  --clients 0-3 --admission_mode anytime --mock_connector 2>&1 | tee server32b.log
```

The engine refuses a profile whose file name says a different TP. Each
engine locks the clocks of both of its GPUs.

```sh
MODEL=Qwen/Qwen3-32B POLICY=round_robin:beam TRACE=azure_chat_23:azure_chat_23 \
TTFT=5 TPOT=0.05 EXTRA= \
  bench runs/fig7b/chat32b/off0 0 0-3 8 7 6 5 4 3 2
```

N <= 4 are full runs; N = 5-8 replay 4 of N shards. Smoke-test first, as
in Phase 1, with `WINDOW=t1200:1320` and N=4. Also check that
`per_gpu_energy_consumption` in `results.jsonl` has 8 entries (two per
engine).

The earlier 32B BEAM runs on another GPU had no point under 10% violation.
If N=2-4 are all far above 10%, stop there; the panel will not show them.

## Phase 3 (recommended): redo the top row with the fix, 8 GPUs

Same commands as in section 4 of `docs/beam_a100_sxm_handoff.md`, but with
`--clients 0-7` on an 8-engine 7B server (`CUDA_VISIBLE_DEVICES=0-7`,
`--clients 0-7`). N <= 8 are then full runs, and N > 8 replay 8 shards. This
is the same 23 runs and about the same 4.6 h as Tier 1, with up to 8x the
shards per point.

## Merge and bring back

```sh
.venv/bin/python scripts/merge_beam_shard_runs.py runs/fig7b/*/*/*/results.jsonl \
  --output runs/fig7b/points.csv
```

The merge errors out if two runs cover the same shard. Multi-Round rows
must show `complete=True`. Large-Scale and 32B rows show `complete=False`
by design (sampled shards).

Bring back:
- `runs/fig7b/**`
- `points.csv`
- the 32B profile CSV and its `.smoke`
- `server*.log`
- the check-3 printout
- a note of anything that deviated from this document

## Changes since the first handoff

- **Decode-stage double count fixed.** See the bug section at the top.
- **Prefix-cache hits are discounted.** Upstream BEAM counts the whole
  prompt as pending prefill, and BEAM's own scripts always run with
  `--no-enable-prefix-caching`. SLOServe runs with prefix caching, and
  about 60% of multi-round prompt tokens hit the cache. Without the
  discount the counter would never drain. To restore upstream accounting,
  set `beam_discount_cached_prefill=False`; that is not recommended.
- **TP > 1 supported.** BEAM now locks every GPU of a TP group, like
  upstream `apply_dvfs`, and checks that the profile's TP matches the
  engine's. PP must still be 1.
- **Session-aware slicing.** `round_robin_session` runs slice by session,
  so a replayed shard is exactly what one GPU of the full run would see.
  The routing test proves this against the real router.
- **Shard offset in results.** `rr_shard_offset` is now written to
  `results.jsonl`, and `scripts/merge_beam_shard_runs.py` combines runs.
- **Profiling script accepts other models.** `BEAM_MODEL`, `BEAM_TP` and a
  GPU list in `BEAM_GPU` now work with `scripts/profile_beam_qwen7b.sh`.

## Known limitations

- Multi-Round has no native-BEAM fidelity check: upstream's client cannot
  replay dependent session turns.
- Shard energies come from separate replays, not one simultaneous N-GPU
  measurement. For Large-Scale and 32B they are additionally sampled.
- In the mock engine, `energy_consumption` is 0. Use `energy_est`, which
  the merge script does not read. Real-GPU runs fill `energy_consumption`.
