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

```sh
# download traces
PYTHONPATH=. python Dataset/download_dataset.py # this takes minutes
```

## Step 2: Launch distributed api server

Launch a distributed api_server (router + engine).
```sh
python -m SLOsServe.router.api_server_ray --help
```

Example: 
```sh
# launch the router & server, the parameter here is not important.
python -m SLOsServe.router.api_server_ray --stat_window 2 \
    --host 0.0.0.0 \
    --port 8000 \
    --window_size 0.001 \
    --router round_robin \
    --router_kwargs "{\"max_decode_batch_size\":300}" \
    --clients 0 --admission_mode anytime --mock_connector \
    --mock_engine 
```


## Step 3: Run benchmark.
# run benchmark
# single server 
```sh
python motivation/bench_api_server.py --help
```

```sh
python motivation/bench_api_server.py --overwrite \
  --n_devices 4 \
  --policies round_robin:slosserve-edf \
  --load_scales 1.0 \
  --slo_tpots 0.025 \
  --ttft_slo_scales 3.0 \
  --window 3978:4100 \
  --trace sharegpt_code:azure_chat_23 \
  --profit constant --admission_mode anytime \
  --overwrite --slo_routing_overhead 0.05 --scheduling_overhead 0.005\
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --port 8000 --clients 0,1,2,3 --output_dir experiments_mock

 python motivation/bench_api_server.py --overwrite \
  --n_devices 4 \
  --policies round_robin:slosserve-edf \
  --load_scales 1.0 \
  --slo_tpots 0.025 \
  --ttft_slo_scales 3.0 \
  --window 3978:4100 \
  --trace sharegpt_code:azure_chat_23 \
  --profit constant --admission_mode anytime \
  --overwrite --slo_routing_overhead 0.05 --scheduling_overhead 0.005\
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --port 8000 --clients 0,1,2,3 --output_dir experiments_mock


python motivation/bench_api_server.py --overwrite \
  --n_devices 1 \
  --policies round_robin:slosserve-edf \
  --load_scales 1.0 \
  --slo_tpots 0.025 \
  --ttft_slo_scales 3.0 \
  --window 3978:4579 \
  --trace sharegpt_code:azure_chat_23 \
  --profit constant --admission_mode anytime \
  --overwrite --slo_routing_overhead 0.05 --scheduling_overhead 0.002\
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --port 8000 --clients 0


```

## Step 4. end to end evaluation
End-to-end scripts
```sh
# 1. Single Server
# ./run_unit.sh <gpu> <port> <baseline> <load_scale> [slo_tpots] [ttft_slo_scales] [window] [trace] [model_name]
source ./run_unit.sh 0 8000 slosserve-edf 0.5 0.1 5.0 600:1201 azure_chat_23:azure_chat_23 Qwen/Qwen2.5-7B-Instruct 2>&1 | tee out.txt
# run a grid of experiment 
# usage: run_new.py [-h]
#                   [--job {Coder-Qwen7B-bustiness,ChatBot-Qwen7B-bustiness,Coder-Qwen7B,Coder-Qwen7B-tpot-ablation,Coder-Qwen7B-ttft-ablation,ChatBot-Qwen7B,Arxiv-Qwen7B,Coder-Gemma27B,ChatBot-Gemma27B,Arxiv-Gemma27B}]
python run_new.py --job Coder-Qwen8B
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