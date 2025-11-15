Setup

```sh
conda create --name slosserve python=3.10  -y
conda activate slosserve

cd 3rdparty/vllm
pip install -e .
cd -

cd csrc
pip install -e .
```

```sh
# download traces
python Dataset/download_dataset.py
```

```sh
# launch the router & server, the parameter here is not important.
python -m SLOsServe.router.api_server_ray --stat_window 2 \
    --host 0.0.0.0 \
    --port 8000 \
    --window_size 0.001 \
    --router round_robin \
    --router_kwargs "{\"max_decode_batch_size\":300}" \
    --clients 0 --admission_mode anytime --mock_connector 2>&1

# run benchmark
# single server 
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

# run multi-server benchmark 
python motivation/bench_api_server.py --overwrite \
  --n_devices 4 \        
  --policies auto_scaling-all_chat-0.02:slosserve-edf  \
  --load_scales 0.9 \
  --slo_tpots 0.1 \      
  --ttft_slo_scales 5.0 \
  --window 3978:4579 \                 
  --trace azure_chat_23:azure_code_23 \       
  --profit constant --admission_mode anytime \                        
  --overwrite --slo_routing_overhead 0.20 --scheduling_overhead 0.005\
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --port 8000 --clients 0,1,2,3
```


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