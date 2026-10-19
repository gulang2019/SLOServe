# TODOs

30B model experiment:

7B:
Coding: 25% at 5%, ?? at 1% SLO violation, need to rerun Sarathi+, baseline with more than 8 servers; 

Mixed: 35% at 5%, 37% at 1%, need to rerun Sarathi+; 

Chatting: 5% at 5%; 50% at 1%; need to rerun Sarathi+; 1-4 devices is okay;

Multi-Node 16 server experiment:
AzureChat: good result;

Multi-round Chatting: AzureChat + Multi-round Chatting 

Disaggregation: 
AzureChat24: No visible energy benefit at 8 machines; The SLO violation is smaller. The ideal # device is larger than 8 devices;


Reasoning:
Ran for load scale 0.3 on disagg servers; it is able to serve with only 2 devices; 

30B:

AzureChat23: baselines saturate at 4 devices; need to check ours at 2 devices to see the gain; 

AzureCode23: Not a huge win for the setup;
Mixed: not ran yet;

Mixed seems promising;

3.21 
Finish the expeirment section: coding + writing + evaluation;
Multi-node experiment;
Tensor Parallel experiment (need a MoE model if possible);  
ShareGPT experiment; KV Cache;
Memory experiment w/ Reasoning;
Writing from the experiment section; Finalize the figures;
Routing Overhead experiment;


For Yi
- []  run through the router/benchmark detailed below; 
- [] open a new branch, add memory feasibility check to the feasibility check in csrc/adm_ctrl_scheduler.cc; (csrc/adm_ctrl_scheduler.cc:1254)
- [] do experiment on the S1K trace; wire the logic to add oom as another source of SLO violation (we do not do migration for now). (3rdparty/vllm/vllm/v1/core/sched/scheduler_adm_ctrl.py:1852, motivation/bench_api_server.py:550)

# Setup

## Step 1: Setup

```sh
git submodule update --init

conda create --name sloserve python=3.10  -y
conda activate sloserve

cd 3rdparty/vllm
git pull origin main
VLLM_USE_PRECOMPILED=1 python3 -m pip install --editable .
cd -

cd csrc
python3 -m pip install -e . --no-build-isolation
cd -
```
> Trouble shooting 1 ImportError:..._C.abi3.so: undefined symbol... : Comment out import vllm._C in SLOServe/3rdparty/vllm/vllm/platforms/cuda.py:18

### uv alternative

If `uv` is available, the bootstrap script creates a Python 3.10 virtual
environment and installs the pinned modified-vLLM submodule plus the native
SLOServe extension. It intentionally does not update `3rdparty/vllm` beyond
the revision recorded by the SLOServe checkout.

```sh
./scripts/setup_uv.sh
source .venv/bin/activate
export PYTHONPATH=$PWD
```

```sh
# download traces
PYTHONPATH=. python Dataset/download_dataset.py # this takes minutes
```

### BEAM baseline (Qwen2.5-7B)

BEAM is provided as `3rdparty/BEAM` and runs in its evaluator container. Its
S1+S2 DVFS mode needs an offline profile for the exact GPU, model, and
parallelism configuration. Profile and serve with the same memory-utilization
setting; lower it when other processes already occupy GPU memory.

```sh
BEAM_GPU=0 BEAM_GPU_MEMORY_UTILIZATION=0.85 ./scripts/profile_beam_qwen7b.sh
BEAM_GPU=0 BEAM_PORT=8008 BEAM_GPU_MEMORY_UTILIZATION=0.85 \
  BEAM_MODE=s1s2 ./scripts/serve_beam_qwen7b.sh
curl -fsS http://127.0.0.1:8008/v1/models
```

BEAM's endpoint is OpenAI-compatible. Its native traced benchmark accepts a
CSV with `TIMESTAMP`, `ContextTokens`, and `GeneratedTokens` columns. The
current Figure 7 driver cannot target BEAM directly: it requires SLOServe
control-plane endpoints and its curves sweep independent replicas. An exact
Figure 7 BEAM baseline therefore needs a multi-replica adapter and trace
converter. Stop the service with `./scripts/stop_beam_qwen7b.sh`.

## Step 2: Launch distributed api server

Launch a distributed api_server (router + engine).
```sh
python -m SLOsServe.router.api_server_ray --help
```

Example: 
```sh
export PYTHONPATH=$PWD
# launch the router & server, the parameter here is not important.
# the mock_engine and mock_connector flags are important for emulation.
python -m SLOsServe.router.api_server_ray --stat_window 2 \
    --host 0.0.0.0 \
    --port 8000 \
    --window_size 0.005 \
    --router slosserve \
    --router_kwargs "{\"device_mem\":1248576, \"block_size\": 16, \"model_name\": \"Qwen/Qwen2.5-7B-Instruct\", \"tpot\":0.05, \"scheduling_overhead\": 0.005, \"max_decode_length\": 500, \"is_pd_disagg\": 0, \"n_prefill_per_group\": 1, \"max_decode_bs\": 16, \"enable_rerouting\": 0}" \
    --clients 0,1 --admission_mode anytime --mock_connector \
    --mock_engine 2>&1 | tee out.txt
```

### Multi-node experiment (no shared filesystem)

The Ray-based router can place engine actors across multiple machines via
`--ray_address`. A shared filesystem is **not** required, but every node
must have:

- the same code checkout (same path is the safest option),
- the same Python environment / dependencies,
- access to the same model (either via Hugging Face download or a copied local path).

If you only need one-way code sync from the head node to a worker node, `rsync`
is enough:

```sh
rsync -ah --delete \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude 'experiments*' \
  /path/to/SLOServe/ \
  user@worker:/path/to/SLOServe/
```

On **every node**:

```sh
cd /path/to/SLOServe
conda activate sloserve
export PYTHONPATH=$PWD
```

Start the Ray head on node 0:

```sh
HEAD_IP=<node0-ip>
ray stop -f || true
ray start --head --node-ip-address "$HEAD_IP" --port 6379
```

Join node 1 (and every other worker node):

```sh
HEAD_IP=<node0-ip>
MY_IP=<node1-ip>
ray stop -f || true
ray start --address "$HEAD_IP:6379" --node-ip-address "$MY_IP"
```

Then launch the router on the head node. In the example below, we run
8 logical replicas (`r0`-`r7`) with tensor parallel size 1:

```sh
python -m SLOsServe.router.api_server_ray \
    --host 0.0.0.0 \
    --port 8000 \
    --ray_address "$HEAD_IP:6379" \
    --tensor_parallel_size 1 \
    --stat_window 2 \
    --window_size 0.005 \
    --router slosserve \
    --router_kwargs "{\"device_mem\":1248576, \"block_size\": 16, \"model_name\": \"Qwen/Qwen2.5-7B-Instruct\", \"tpot\":0.05, \"scheduling_overhead\": 0.005, \"max_decode_length\": 500, \"is_pd_disagg\": 0, \"n_prefill_per_group\": 1, \"max_decode_bs\": 16, \"enable_rerouting\": 0}" \
    --clients r0,r1,r2,r3,r4,r5,r6,r7 \
    --admission_mode anytime
```

`--tensor_parallel_size` is the number of GPUs per logical replica. It must
fit within a single node. For example, with 2 nodes x 4 GPUs each:

- `--n_devices 8 --tensor_parallel_size 1` uses 8 replicas over 8 GPUs.
- `--n_devices 4 --tensor_parallel_size 2` uses 4 replicas over 8 GPUs.

Run the benchmark from the head node:

```sh
python motivation/bench_api_server.py --overwrite \
      --n_devices 8 \
      --tensor_parallel_size 1 \
      --policies slosserve_planner:atfc \
      --load_scales 1.0 \
      --slo_tpots 0.05 \
      --ttft_slo_scales 5.0 \
      --window "t800:1000" \
      --trace "azure_code_23:azure_code_23" \
      --profit constant --admission_mode arrival \
      --slo_routing_overhead 0.05 --scheduling_overhead 0.005 \
      --model_name Qwen/Qwen2.5-7B-Instruct \
      --port 8000 --clients 0-7 \
      --output_dir experiments_emulation_new \
      --routing_overhead -1.0 --routing_fallback_policy reject --kv_xfer_delay 0.05
```

The benchmark driver writes merged results on the head node. It also copies
per-worker energy CSVs back to the head when dumping profile events, so raw
energy profiling does not require a shared filesystem either.

## Step 3: Run benchmark.
```sh
python motivation/bench_api_server.py --help
```

```bash
curl -X POST http://0.0.0.0:8000/v1/completions \
:  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "prompt": "how are you?",
    "max_tokens": 10,
    "stream": true,
    "ignore_eos": true,
    "vllm_xargs": {
      "input_length": 100,
      "output_length": 11, "slo_ttft": 1, "slo_tpot": 0.05,
      "profit": 1.0,
      "request_id": "5c4e5932-50f4-4546-8325-2a767f403z"
    }
  }'
```

```sh
# run on 16 devices, clients 0-15 (the value of the number is not important, but the total counts should be greator than 16), slosserve_planner router, atfc scheduler, 
# azure_code_23 as length pattern, azure_code_23 as arrival pattern, request 3978 to request 4100 
# tpot slo 50ms, ttft slo 5 x zero_load, routing slo 50 ms (for the overhead).
# output directory experiments_mock 
python motivation/bench_api_server.py --overwrite \
      --n_devices 16 \
      --policies slosserve_planner:atfc \
      --load_scales 1.0 \
      --slo_tpots 0.05 \
      --ttft_slo_scales 5.0 \
      --window "t800:1000" \
      --trace "azure_code_23:azure_code_23" \
      --profit constant --admission_mode arrival \
      --overwrite --slo_routing_overhead 0.05 --scheduling_overhead 0.005 \
      --model_name Qwen/Qwen2.5-7B-Instruct \
      --port 8000 --clients 0-15 \
      --output_dir experiments_emulation_new --routing_overhead -1.0 --routing_fallback_policy reject --kv_xfer_delay 0.05
```

## Step 4. end to end evaluation
End-to-end scripts
```sh
source run_batch.sh
```

### Recover Main Multi-Replica Figure

To quickly regenerate the paper's main multi-replica figure set
(`fig:distributed` in `Paper/04-evaluation.tex`) and its shared legend:

```sh
python3 scripts/recover_main_multi_replica_figure.py
```

This rewrites the assets in `Paper/figs/e2e_new/`, including:

- `multi_replica_shared_legend.pdf`
- `chat_7B_two_panel_slo_cap_10_logx.pdf`
- `code_7B_two_panel_slo_cap_10_logx_normal.pdf`
- `mixed_7B_two_panel_slo_cap_10_logx.pdf`
- `32B_chat_two_panel_slo_cap_10_logx.pdf`
- `e2e_mulitiround_chatting_two_panel_slo_cap_10_logx.pdf`
- `chatting_large_scale_two_panel_slo_cap_10_logx.pdf`

To write the recovered assets somewhere else for inspection:

```sh
python3 scripts/recover_main_multi_replica_figure.py --output-dir /tmp/e2e_new
```

# Instructions

## Developing C code
```bash 
# reproduce admission_ctrl + scheduling
# admission control: fast edf simulation to decide feasibility 
# scheduling: construct batch based on admission result; 
make -C csrc repro_long_schedule_dump
./csrc/repro_long_schedule_dump adm_ctrl_dumps/schedule_1_84_adm_ctrl_44.txt
```

## About Auto-Scaler.

We build the indexPacking auto-scaler (we are now abusing the auto-scaler with load-concentrator) to save average number of servers used by concentrating load dynamically. The auto-scaler works by indexing replica (engine) by 1-n number and route new request from 1-n. Predictors decide whether the SLO is attainable on a server.

AutoScaling is enabled by specifying the routing policy in `--policies` in format `ASPOLICY-PREDICTOR-THRESHOLD`.
`ASPOLICY` chooses from `auto_scaling_resch` and `auto_scaling`, where the later enables the server-side `early rerouting` mechanism.

`PREDICTOR` binary predicts whether a request's SLO is not attainable on a server (positive prediction leads to rejection) using logistic regression. The parameters are stored in `auto_scaling_model.json`. Use `all` as default.

`THRESHOLD` is a 0-1 real number controlling the optimism of the predictor. The predictor works by predicting the probability the request's SLO is not attainable on a certain server. Request with unattainable probability higher than the threshold is rejected and routed to the next server. Thus, higher `THRESHOLD` means we are more optimistic and route the request to smaller 

```sh
# run multi-server benchmark 
python motivation/bench_api_server.py --overwrite \
  --n_devices 4 \
  --policies auto_scaling_resch-all_chat-0.2:slosserve-edf \
  --load_scales 1.0 \
  --slo_tpots 0.025 \
  --ttft_slo_scales 3.0 \
  --window 3978:4100 \
  --trace sharegpt_code:azure_chat_23 \
  --profit constant --admission_mode anytime \
  --overwrite --slo_routing_overhead 0.05 --scheduling_overhead 0.005\
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --port 8000 --clients 0,1,2,3 --output_dir experiments_mock
```

### Train the predictor
`motivation/auto_scaling.py` is used to train and test the predictor in auto_scaler. Add the `*admission_history.jsonl` trace from benchmark to the `TRAIN_PATHS` in the `fit_ours` function and specify the `TEST_PATH`, and `PREDICTOR_NAME`. Then run 
```sh
python motivation/auto_scaling.py
```
to get the predictor. A tradeoff plot is shown in `auto_scaling_fpr_fnr.png`.


## Implementing ur own Router

Implement a child class of the Router class in `SLOsServe/router/api_server_ray.py` and register the router in the `create_router` function. Every router is initialized with n_devices and router_kwargs. To benchmark the router, one needs to customize the router_kwargs passed to the router in `motivation/bench_api_server.py:build_problems`. 

```py
class Router(ABC):
    def __init__(self, n_devices: int, router_kwargs: str):
        pass 

    def set_load_stat(self, load_stat: LoadStat):
        self.load_stat = load_stat

    def run(self, waiting_requests: List[RequestInstance],
            running_requests: List[RequestInstance]):
        '''
        Run the router to decide the admission and device assignment for the waiting requests.
        for every request in the waiting_requests, the router decide: 
        request.admitted: bool
        request.prefill_device_id: int
        request.decode_device_id: int
        the router is not required to make decisions, it can keep the request.admitted as None
        '''
        raise NotImplementedError

    def update(self, request: RequestInstance, new_state: RequestState):
        '''
        Optional update for the router to keep track of the request state.
        '''
        pass

    def update_json(self, request_json: dict, i: int):
        '''
        Optional update for the config passed to the i-th engine.
        For example, in KV disagg router, the config is updated for P devices to allow large batch size limit.
        '''
        return request_json
```
