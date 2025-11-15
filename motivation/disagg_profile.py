"""
We want to compare the P99 TPOT and P99 TTFAT.
For (Prefill) (Thinking Decode), we set a max batch size.
For (Prefill Thinking) (Decode), we set the same max batch size.
NOTE: Server-level dummy prefill has been implemented in vLLM. See 3rdparty/vllm/DUMMY_PREFILL_README.md for details.
"""
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from Dataset.dataset import Request, Requests, ArrivalTimes
from collections import defaultdict
import time
import matplotlib.pyplot as plt
from events_analysis import analyze_slo_violation, analyze_events
    
@dataclass
class ExecutionResult:
    request: Request
    timestamps: List[float] # timestamps[0] is the arrival time of the request, timestamps[i] is the time to output the i-th token
    ttft: float  = field(init=False)
    ttfat: float = field(init=False)
    tpots: List[float] = field(init = False)
    normalized_ttft: float = field(init=False)
    normalized_ttfat: float = field(init=False)
    
    def __post_init__(self):
        # assert len(self.timestamps) == 1 + self.request.output_length + self.request.thinking_length
        # print(f'timestamps: {len(self.timestamps)}, thinking_length: {self.request.thinking_length}, output_length: {self.request.output_length}, input_length: {self.request.input_length}')
        # Add safeguards to prevent IndexError and handle edge cases
        if len(self.timestamps) < 2:
            self.ttft = 0.0
            self.normalized_ttft = 0.0
        else:
            self.ttft = self.timestamps[1] - self.timestamps[0]
            self.normalized_ttft = self.ttft / self.request.input_length

        thinking_idx = self.request.thinking_length
        if len(self.timestamps) > thinking_idx:
            self.ttfat = self.timestamps[thinking_idx] - self.timestamps[0]
            self.normalized_ttfat = 0.0
        else:
            self.ttfat = self.timestamps[-1] - self.timestamps[0]
            self.normalized_ttfat = self.ttfat / (self.request.thinking_length + self.request.input_length)
        
        self.tpots = []
        base_idx = self.request.thinking_length + 1
        for i in range(self.request.output_length - 1):
            idx1 = base_idx + i + 1
            idx0 = base_idx + i
            if idx1 < len(self.timestamps) and idx0 < len(self.timestamps):
                self.tpots.append(self.timestamps[idx1] - self.timestamps[idx0])
total_load = 0

@dataclass
class Router:
    endpoints: Dict[str, List[str]]
    load_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    async def _submit_request(self, 
                              end_point: str,
                              model_name: str,
                              prompt: str, 
                              input_length: int,
                              output_length: int) -> Tuple[str, List[float]]:
        # submit streaming request to vllm's endpoint 
        # return the response and the timestamps of tokens, the first timestamp is the arrival time of the request
        self.load_counts[end_point] += 1
        global total_load
        total_load += 1
        import time
        import aiohttp

        timestamps = []
        response_text = ""
        timeout = aiohttp.ClientTimeout(total=3600, connect=3600, sock_read=3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Record the arrival time of the request
            timestamps.append(time.time())
            headers = {
                "Content-Type": "application/json",
            }
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": output_length,
                "stream": True,
                "ignore_eos": True
            }
            # print('Using payload:', payload)

            async with session.post(end_point, json=payload, headers=headers) as resp:
                async for line in resp.content:
                    if not line:
                        continue
                    try:
                        # SSE lines start with "data: "
                        if line.startswith(b"data: "):
                            data = line[6:].decode("utf-8").strip()
                            if data == "[DONE]":
                                break
                            # Parse the streamed token and timestamp
                            # Assume the server returns JSON with at least 'choices' and 'created'
                            import json
                            obj = json.loads(data)
                            if "choices" in obj and obj["choices"]:
                                delta = obj["choices"][0].get("delta", {})
                                token = delta.get("content", "")
                                if token:
                                    response_text += token
                                    timestamps.append(time.time())
                    except Exception:
                        continue
        self.load_counts[end_point] -= 1
        total_load -= 1
        return response_text, timestamps
    
    async def process_request(self, model_name: str, request: Request) -> ExecutionResult:
        raise NotImplementedError
    
    def _find_endpoint(self, tag: str) -> str:
        endpoints = self.endpoints[tag]
        # find the endpoint with the least load
        return min(endpoints, key=lambda x: self.load_counts[x])

class RR_Router(Router):
    def __init__(self, endpoints: Dict[str, List[str]]):
        super().__init__(endpoints)
        self.rr_idx = 0
    
    async def process_request(self, model_name: str, request: Request) -> ExecutionResult:
        endpoint = self._find_endpoint('PD')
        response_text, timestamps = await self._submit_request(
            end_point = endpoint, 
            model_name = model_name,
            prompt = request.prompt,
            input_length = request.input_length,
            output_length = request.output_length
        )
        return ExecutionResult(request, timestamps)

class P_TD_DisaggRouter(Router):
    def __init__(self, endpoints: Dict[str, List[str]]):
        super().__init__(endpoints)
    
    async def process_request(self, model_name: str, request: Request) -> str:
        prefill_endpoint = self._find_endpoint('P')
        response_text, prefill_timestamps = await self._submit_request(
            end_point = prefill_endpoint, 
            model_name = model_name,
            prompt = request.prompt,
            input_length = request.input_length,
            output_length = 1
        )
        decode_endpoint = self._find_endpoint('DT')
        response_text, decode_timestamps = await self._submit_request(
            end_point = decode_endpoint, 
            model_name = model_name,
            prompt = request.thinking, 
            input_length = request.input_length, 
            output_length = request.thinking_length + request.output_length
        )
        # print('prefill timestamps', prefill_timestamps)
        # print('decode timestamps', decode_timestamps)
        result = ExecutionResult(
            request,
            prefill_timestamps + decode_timestamps[1:]
        )
        return result

class PT_D_DisaggRouter(Router):
    def __init__(self, endpoints: Dict[str, List[str]]):
        super().__init__(endpoints)
    
    async def process_request(self, model_name: str, request: Request) -> str:
        prefill_endpoint = self._find_endpoint('PT')
        response_text, prefill_timestamps = await self._submit_request(end_point = prefill_endpoint, 
                                                        model_name = model_name, 
                                                        prompt = request.prompt, 
                                                        input_length = request.input_length, 
                                                        output_length = request.thinking_length)
        decode_endpoint = self._find_endpoint('D')
        response_text, decode_timestamps = await self._submit_request(
            end_point = decode_endpoint, 
            model_name = model_name,
            prompt = request.thinking, 
            input_length = request.input_length + request.thinking_length, 
            output_length = request.output_length)
        result = ExecutionResult(
            request, 
            prefill_timestamps + decode_timestamps[1:]
        )
        return result

async def run_serving(
    router: Router,
    model_name: str,
    requests_name: str,
    arrival_times_name: str,   
    window: str,
    load_scale: float,
    vis_only: bool,
):
    global total_load
    from tqdm import tqdm
    requests = Requests.load(requests_name, 16384)
    # draw the distribution of input_length, output_length, thinking_length 
    requests.visualize()

    arrival_times = ArrivalTimes.load(arrival_times_name, load_scale)
    arrival_times.visualize()
    
    if vis_only:
        exit(0)
    
    wait_times = [arrival_times.arrival_times[i] - arrival_times.arrival_times[i - 1] for i in range(1, len(arrival_times.arrival_times))]
    window_start, window_end = map(int, window.split(':'))
    wait_time_idx = 0

    # Add tqdm for arrived and completed requests
    num_requests = len(requests.requests[window_start:window_end])
    events = []
    start_time = time.time()
    arrived_bar = tqdm(total=num_requests, desc="Requests Arrived")
    completed_bar = tqdm(total=num_requests, desc="Requests Completed")

    jobs = []
    async def wrapped_process_request(*args, **kwargs):
        try:
            await asyncio.sleep(0)
            result = await router.process_request(*args, **kwargs)
            return result
        finally:
            completed_bar.update(1)
            events.append((time.time() - start_time, -1))
    for request in requests.get_requests(window_start, window_end, model_name):
        task = asyncio.create_task(wrapped_process_request(model_name, request))
        jobs.append(task)
        arrived_bar.update(1)
        wait_time = wait_times[wait_time_idx % len(wait_times)]
        while wait_time > 0:
            await asyncio.sleep(min(wait_time, 1))
            if total_load == 0:
                break
            wait_time -= 1
        wait_time_idx += 1
        events.append((time.time() - start_time, 1))
    results = await asyncio.gather(*jobs)
    arrived_bar.close()
    completed_bar.close()
    return results, events

def main(
    model_name: str,
    router_type: str, 
    requests_name: str,
    arrival_times_name: str,
    window: str,
    load_scale: float,
    vis_only: bool,
):
    import json
    with open(f'motivation/endpoints_{router_type}.json', 'r') as f:
        endpoints = json.load(f)
    endpoints = {tag: [f'http://localhost:{port}/v1/chat/completions' for port in ports] for tag, (ports) in endpoints.items()}
    print('Using endpoints:')
    print(endpoints)
    if router_type == 'p_td':
        router = P_TD_DisaggRouter(endpoints)
    elif router_type == 'pt_d':
        router = PT_D_DisaggRouter(endpoints)
    elif router_type == 'pd':
        router = RR_Router(endpoints)
    else:
        raise ValueError(f'Invalid router type: {router_type}')
    
    results = asyncio.run(run_serving(router, model_name, requests_name, arrival_times_name, window, load_scale, vis_only))
    return results

def plot_cdf(results, prefix: str):
    
    import numpy as np
    def _plot_cdf(ax, data: List[float], label: str, log_scale: bool):
        data = sorted(data)
        yvals = [i/len(data) for i in range(len(data))]
        ax.plot(data, yvals, label = label)
        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlabel(label)
        ax.set_ylabel('CDF')
        ax.legend()
    
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, figsize = (4, 20), tight_layout = True)
    _plot_cdf(ax0, [res.ttft for res in results], 'TTFT', log_scale = False)
    _plot_cdf(ax1, [res.ttfat for res in results], 'TTFAT', log_scale = False)
    _plot_cdf(ax2, sum([res.tpots for res in results], start = []), 'TPOT', log_scale = False)
    _plot_cdf(ax3, [res.normalized_ttft for res in results], 'Normalized TTFT', log_scale = False)
    _plot_cdf(ax4, [res.normalized_ttfat for res in results], 'Normalized TTFAT', log_scale = False)
    fig.savefig(f'figs/{prefix}_cdf.png', dpi = 300, bbox_inches = 'tight')
    print(f'Saved figs/{prefix}_cdf.png')

def plot_events(events, prefix: str):
    times = []
    loads = []
    load = 0
    for timestamp, event in events:
        times.append(timestamp)
        load += event
        loads.append(load)
    fig, ax = plt.subplots(figsize = (4, 4), tight_layout = True)
    ax.plot(times, loads)
    ax.set_xlabel('Time')
    ax.set_ylabel('Load')
    fig.savefig(f'figs/{prefix}_events.png', dpi = 300, bbox_inches = 'tight')
    print(f'Saved figs/{prefix}_events.png')

if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', type=str, default = 'Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--router_type', '-t', type=str, required=True, choices=['p_td', 'pt_d', 'pd'])
    parser.add_argument('--requests_name', '-r', type=str, required=True, choices=['deepseek-r1', 
                                                                                   'azure_code', 
                                                                                   'azure_chat',
                                                                                   'sharegpt_chat',
                                                                                   'azure_code_23',
                                                                                   'azure_chat_23'])
    parser.add_argument('--arrival_times_name', '-a', type=str, required=True, choices=['azure_code',
                                                                                        'azure_chat',
                                                                                        'azure_code_23',
                                                                                        'azure_chat_23'])
    parser.add_argument('--load-scale', '-ls', type=float, default = 1, help = 'load scale')
    parser.add_argument('--window', '-w', type=str, required=True, default = '0:-1', help = 'start:end')
    parser.add_argument('--vis_only', '-vo', action='store_true', help = 'only visualize the results')
    args = parser.parse_args()
    results, events = main(args.model_name, args.router_type, args.requests_name, args.arrival_times_name, args.window, args.load_scale, args.vis_only)
    
    ## calc p50, p90, p99, mean TPOT, TTFAT, TTFT
    import numpy as np
    
    # import os
    # if os.path.exists('profile_events.jsonl'):
    #     with open('profile_events.jsonl', 'w') as f:
    #         pass

    # If results is a list of ExecutionResult, extract the relevant metrics
    ttfts = []
    ttfats = []
    all_tpots = []
    normalized_ttfats = []
    normalized_ttfts = []
    for res in results:
        # If result is a dataclass, access fields directly
        if hasattr(res, "ttft") and hasattr(res, "ttfat") and hasattr(res, "tpots"):
            ttfts.append(res.ttft)
            ttfats.append(res.ttfat)
            all_tpots.extend(res.tpots)
            normalized_ttfats.append(res.normalized_ttfat)
            normalized_ttfts.append(res.normalized_ttft)
        # If result is a dict, fallback to dict access
        elif isinstance(res, dict):
            ttfts.append(res.get("ttft", 0))
            ttfats.append(res.get("ttfat", 0))
            all_tpots.extend(res.get("tpots", []))
            normalized_ttfats.append(res.get("normalized_ttfat", 0))
            normalized_ttfts.append(res.get("normalized_ttft", 0))

    def safe_percentile(arr, q):
        if len(arr) == 0:
            return float('nan')
        return float(np.percentile(arr, q))

    def safe_mean(arr):
        if len(arr) == 0:
            return float('nan')
        return float(np.mean(arr))

    stats = {
        "TPOT": {
            "p50": safe_percentile(all_tpots, 50),
            "p90": safe_percentile(all_tpots, 90),
            "p99": safe_percentile(all_tpots, 99),
            "mean": safe_mean(all_tpots),
        },
        "TTFAT": {
            "p50": safe_percentile(ttfats, 50),
            "p90": safe_percentile(ttfats, 90),
            "p99": safe_percentile(ttfats, 99),
            "mean": safe_mean(ttfats),
        },
        "TTFT": {
            "p50": safe_percentile(ttfts, 50),
            "p90": safe_percentile(ttfts, 90),
            "p99": safe_percentile(ttfts, 99),
            "mean": safe_mean(ttfts),
        },
        'Normalized TTFAT': {
            "p50": safe_percentile(normalized_ttfats, 50),
            "p90": safe_percentile(normalized_ttfats, 90),
            "p99": safe_percentile(normalized_ttfats, 99),
            "mean": safe_mean(normalized_ttfats),
        },
        'Normalized TTFT': {
            "p50": safe_percentile(normalized_ttfts, 50),
            "p90": safe_percentile(normalized_ttfts, 90),
            "p99": safe_percentile(normalized_ttfts, 99),
            "mean": safe_mean(normalized_ttfts),
        }
    }
    
    

    print("==== Metrics ====")
    for metric, vals in stats.items():
        print(f"{metric}:")
        for k, v in vals.items():
            print(f"  {k}: {v:.4f}")
    import os
    model_name = args.model_name.replace('/', '_')
    args.window = args.window.replace(':', '-')
    name = f'jsons/{model_name}_{args.router_type}_{args.requests_name}_{args.arrival_times_name}_{args.window}_{args.load_scale}.json'
    print(f'Saving {name}')
    os.makedirs('jsons', exist_ok=True)
    os.makedirs(os.path.dirname(name), exist_ok=True)
    with open(name, 'w') as f:
        json.dump(stats, f)
    
    # plot the CDF of ttft, ttfat, tpots
    prefix = f'{model_name}_{args.router_type}_{args.requests_name}_{args.arrival_times_name}_{args.window}_{args.load_scale}'
    plot_cdf(results, prefix)
    plot_events(events, prefix)
    # store the results in a csv file
    import pandas as pd
    import pickle
    
    name = f'csvs/{model_name}_{args.router_type}_{args.requests_name}_{args.arrival_times_name}_{args.window}_{args.load_scale}.pkl'
    os.makedirs('csvs', exist_ok=True)
    os.makedirs(os.path.dirname(name), exist_ok=True)
    with open(name, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved {name}')
    
    import subprocess

    # Call the endpoint to dump profile events
    try:
        response = subprocess.run(
            ["curl", "-X", "GET", "http://localhost:8000/dump_profile_events"],
            capture_output=True, text=True, check=True
        )
        print("Called /dump_profile_events endpoint. Response:")
        print(response.stdout)
        events_file = 'profile_events.jsonl'
        os.system(f'mv {events_file} events/{prefix}.jsonl')
        print(f'Moved {events_file} to events/{prefix}.jsonl')
    except subprocess.CalledProcessError as e:
        print("Failed to call /dump_profile_events endpoint.")
        print(e)
        print(e.stdout)
        print(e.stderr)
        
   
    events, reqs = analyze_events(f'events/{prefix}.jsonl')
    analyze_slo_violation(reqs, prefix = prefix)
    
    
    # The file events/{prefix}.jsonl is created by moving 'profile_events.jsonl' to that location.
    # If you cannot find events/{prefix}.jsonl, it may be because 'profile_events.jsonl' was not generated,
    # or the move command failed. Please check if 'profile_events.jsonl' exists in your working directory
    # before this step, and ensure that the 'events' directory exists and is writable.
    # events_file = 'profile_events.jsonl'
    # os.system(f'mv {events_file} events/{prefix}.jsonl')
    
