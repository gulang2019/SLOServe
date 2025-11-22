import tqdm
import time
import asyncio
from typing import Tuple, List, Dict, Any
import subprocess
from dataclasses import dataclass, field, asdict
import numpy as np
import pprint
import json
from itertools import product
import logging
import uuid
import random
import httpx
import pandas as pd

from motivation.events_analysis import analyze_events, analyze_slo_violation
from motivation.auto_scaling import eval_auto_scaling
from Dataset.dataset import ArrivalTimes, Requests, Request


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

FRONTEND_DELAY=0.03

from motivation.common import get_model_max_tokens, get_easy_name
from motivation.energy_measure import EnergyMeter, EnergyHistoryRecorder
@dataclass
class Problem:
    
    # problem
    model_name: str = 'Qwen/Qwen2.5-7B-Instruct'
    arrival_pattern: str = 'azure_code_23'
    length_pattern: str = 'azure_code_23'
    window: str = '0:10'
    
    # runtime config
    load_scale: float = 1.0 
    n_devices: int = 2
    
    # slo model
    ttft_slo_scale: float = 1.0
    slo_ttft_per_token: float = 2e-4
    slo_ttft_constant: float = 0.1
    slo_tpot: float = 0.05
    slo_routing_overhead: float = 0.16
     
    # profit model 
    profit_per_input_token: float = 0.0
    profit_per_output_token: float = 0.0
    profit_base: float = 1.0
    
    # scheduling mode
    admission_mode: str = 'arrival'
    
    # policies
    routing_policy: str = 'slo'
    routing_kwargs: dict = field(default_factory=lambda: {'hardware_params': [4.1e-5, 0, 1.3e-2], 'tpot': 0.05, 'device_mem': 16384, 'block_size': 16})
    
    scheduling_policy: str = 'vllm'
    scheduling_kwargs: dict = field(default_factory=lambda: {'max_num_seqs': 128, 'max_num_batched_tokens': 512, 'long_prefill_token_threshold': 256, 'enable_chunked_prefill': False, 'enable_admission': True, 'allow_rejection': True})
    
    # store_prefix
    store_prefix: str = 'problem'
    record_events: bool = False

    def get_expected_profit(self, input_length: int):
        return float(self.profit_per_input_token * input_length + self.profit_per_output_token * average_output_length + self.profit_base)
    
@dataclass
class ExecutionResult:
    request: Request
    timestamps: List[float]
    request_id: str
    slo_result: str | None = None
    laxities: List[float] = field(default_factory=list)
    expected_finish_time: List[float] = field(default_factory=list)
    
    
@dataclass
class ExecutionResults:
    problem: Problem
    execution_results: List[ExecutionResult]
    results: Dict[str, Any]
    event_file: str
    energy_consumption: float = field(default=0.0)
    per_gpu_energy_consumption: List[float] = field(default_factory=list)
    
    # stats
    slo_violation_rate: float = field(init=False)
    profit: float = field(init=False)
    
    def get_slo_result(self, exec_result: ExecutionResult):
        from motivation.common import PerfModel
        perf_model = PerfModel.get_perf_model(self.problem.model_name, self.problem.length_pattern)
        slo_ttft = perf_model.get_zero_load_ttft(exec_result.request.input_length, 
                                                 exec_result.request.cached_length) * \
                    self.problem.ttft_slo_scale + self.problem.slo_routing_overhead
        # print('slo_ttft', slo_ttft, 'input_length', exec_result.request.input_length)
        expected_finish_time = [exec_result.timestamps[0], exec_result.timestamps[0] + slo_ttft]
        
        for _ in range(exec_result.request.output_length - 1):
            expected_finish_time.append(self.problem.slo_tpot + expected_finish_time[-1])
        
        exec_result.expected_finish_time = expected_finish_time
        
        if len(exec_result.timestamps) < len(expected_finish_time):
            return 'unfinished'
        
        if not len(exec_result.timestamps) == len(expected_finish_time):
            logger.warning(f"Request {exec_result.request_id} has {len(exec_result.timestamps)} timestamps but {len(expected_finish_time)} expected finish times")
            exec_result.timestamps = exec_result.timestamps[:len(expected_finish_time)]
        
        laxities = np.array(exec_result.timestamps) - np.array(expected_finish_time)
        exec_result.laxities = laxities.tolist()
        
        for i in range(len(expected_finish_time)):
            if exec_result.timestamps[i] > expected_finish_time[i]:
                return 'slo_violation'
        return 'slo_attained'
        
    def __post_init__(self):
        from collections import Counter
        for exec_result in self.execution_results:
            exec_result.slo_result = self.get_slo_result(exec_result)
        slo_results = [exec_result.slo_result for exec_result in self.execution_results]
        print(f"SLO results histogram: {Counter(slo_results)}")
        is_slo_violation = [slo_result != 'slo_attained' for slo_result in slo_results]
        self.slo_violation_rate = sum(is_slo_violation) / len(is_slo_violation)
        
        profits = [self.problem.profit_per_input_token * exec_result.request.input_length +\
            self.problem.profit_per_output_token * exec_result.request.output_length +\
                self.problem.profit_base for exec_result in self.execution_results]
    
        self.profit = (np.array(profits) * 1 - np.array(is_slo_violation)).sum() / len(self.execution_results)
    

async def run_request(end_point: str,
                    request_id: str,
                    model_name: str,
                    prompt: str | list[int], 
                    input_length: int,
                    output_length: int,
                    zero_load_ttft: float,
                    ttft_slo_scale: float,
                    slo_routing_overhead: float,
                    expected_profit: float) -> Tuple[str, List[float]]:
        timestamps = []
        response_text = ""
        
        headers = {
            "Content-Type": "application/json",
            "X-Request-Id": request_id
        }
        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": output_length,
            "stream": True,
            "ignore_eos": True,
            # "priority": - prefill_ddl, # higher priority means earlier handling
            'vllm_xargs': {
                'input_length': input_length,
                'output_length': output_length,
                # 'prefill_ddl': arrival_time + ttft_slo,
                'zero_load_ttft': zero_load_ttft,
                'slo_ttft': zero_load_ttft * ttft_slo_scale + slo_routing_overhead,
                'profit': expected_profit,
                'request_id': request_id
            }
        }
        
        client = httpx.AsyncClient(timeout = 3600, base_url = end_point)
        chunks = []
        is_rejected = False
        async with client.stream("POST",
                            '/v1/completions',
                            json=payload,
                            headers=headers) as response:
            logger.info(f"Streaming response opened: request_id={request_id}, status_code={response.status_code}")
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line is None:
                    continue
                logger.debug(f"Streaming line for request_id={request_id}, line_len={len(line)}")
                if not line:
                    # SSE message separator (blank line)
                    continue
                if line.strip() == "[done]":
                    break
                # Support both 'data: {...}' and raw '{...}' payloads
                payload_text = line[6:] if line.startswith('data: ') else line
                try:
                    obj = json.loads(payload_text)
                    if 'finish_reason' in obj:
                        is_rejected = obj['finish_reason'] and 'rejected' in obj['finish_reason']
                except Exception as e:
                    if not 'done' in line.lower():
                        logger.error(f"Error parsing SSE line for request_id={request_id}: {e}, line: {line}")
                    timestamps.append(time.time())
                    continue
                n_tokens = 1
                if 'token_ids' in obj:
                    n_tokens = len(obj['token_ids'])
                for _ in range(n_tokens):
                    timestamps.append(time.time())
                chunks.append(obj)
            logger.info(f"Streaming response finished: request_id={request_id}")
        # print(f'Request {request_id} finished with {len(timestamps)} timestamps IL: {input_length} OL: {output_length} .')
        return is_rejected, response_text, timestamps

def get_energy_events(csv_path: str) -> List[Dict[str, Any]]:
    import pandas as pd
    import os 
    our_device_id = os.getenv('CUDA_VISIBLE_DEVICES', "0").split(',')
    our_device_id = [int(_) for _ in our_device_id]
    # Identify per-GPU columns
    j_cols = [f"J_gpu{i}" for i in our_device_id]
    w_cols = [f"W_gpu{i}" for i in our_device_id]
    mhz_cols = [f"MHz_gpu{i}" for i in our_device_id]

    energy_events = []
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        for device_id, (j_col, w_col, mhz_col) in enumerate(zip(j_cols, w_cols, mhz_cols)):
            energy_events.append({
                "event_type": "energy",
                "device_id": device_id,
                "timestamp": float(row["ts"]),
                "energy": float(row.get(j_col, 0)),
                "power": float(row.get(w_col, 0)),
                "mhz": float(row.get(mhz_col, 0)),
            })
    return energy_events

def ensure_prompts_present(requests: List[Request], model_name: str) -> None:
    """Ensure prompts exist for all requests.

    Assumes homogeneity: either all requests already have prompts or none do.
    If none have prompts, synthesize random token sequences per input length
    bucket and batch-decode for efficiency.
    """
    import tiktoken
    if not requests:
        return
    first_has_prompt = requests[0].prompt is not None
    if first_has_prompt:
        # Verify assumption: all must have prompt
        for req in requests:
            assert req.prompt is not None, "Mixed prompt presence encountered; expected homogeneity."
        return
    # Verify assumption: none have prompt
    for req in requests:
        assert req.prompt is None, "Mixed prompt presence encountered; expected homogeneity."
    tokenizer = tiktoken.get_encoding("cl100k_base")
    rng = np.random.default_rng()
    indices_by_length: Dict[int, List[int]] = {}
    for idx, req in enumerate(requests):
        indices_by_length.setdefault(req.input_length, []).append(idx)
    for length, indices in indices_by_length.items():
        num = len(indices)
        tokens_batch = rng.integers(1000, 2001, size=(num, length), dtype=np.int64)
        decoded_batch = tokenizer.batch_decode(tokens_batch.tolist(), skip_special_tokens=True)
        for i, req_idx in enumerate(indices):
            requests[req_idx].prompt = decoded_batch[i]

async def main(problem: Problem, endpoint: str, clients: str):
    window_start, window_end = map(int, problem.window.split(':'))
    
    requests = Requests.load(problem.length_pattern, 
                             window_start = window_start,
                             window_end = window_end,
                             max_tokens = get_model_max_tokens(problem.model_name))
    arrival_times = ArrivalTimes.load(problem.arrival_pattern, 
                                      problem.load_scale, 
                                      window_start = window_start, 
                                      window_end = window_end)
    
    requests = requests.requests
    arrival_times = arrival_times.arrival_times
    arrival_times = [t - arrival_times[0] for t in arrival_times]
    
    from motivation.common import PerfModel
    perf_model = PerfModel.get_perf_model(problem.model_name, problem.length_pattern)
    
    # ensure_prompts_present(requests, problem.model_name)
    for request in requests:
        request.prompt = [random.randint(1000, 2000) for _ in range(request.input_length)]
    
    # requests = requests.requests[window_start:window_end]
    # arrival_times = arrival_times.arrival_times[window_start:window_end]
    
    import numpy as np
    global average_input_length, average_output_length
    average_input_length = np.mean([request.input_length for request in requests])
    average_output_length = np.mean([request.output_length + request.thinking_length for request in requests])
    print(f'#Requests: {len(requests)}')
    print(f'average_input_length: {average_input_length}')
    print(f'average_output_length: {average_output_length}')
    
    
    arrival_idx = 0
    window_size = 0.05
    
    execution_results: List[ExecutionResult] = []
    
    # Post the endpoint with the problem before starting the main loop
    import aiohttp

    # Optionally, you can log or print the endpoint and problem for debugging
    print(f"Posting problem to endpoint: {endpoint}")
    print(f"Problem: {problem}")

    async with aiohttp.ClientSession() as session:
        if clients is not None:
            # Set a very long timeout for this request
            timeout = aiohttp.ClientTimeout(total=600.0)  # 10 minutes
            async with session.post(endpoint + "/update_clients", json={'clients': clients}, timeout=timeout) as response:
                response.raise_for_status()
        # await session.post(endpoint + "/warmup", json={'model': problem.model_name})
        response = await session.post(endpoint + "/update_config", json=asdict(problem))
        response.raise_for_status()
        # exit(0)
    time.sleep(10)

    arrival_bar = tqdm.tqdm(total = len(requests), desc = 'Arrival')
    finished_bar = tqdm.tqdm(total = len(requests), desc = 'Finished')
    
    global_start_time = time.time()
    print(f'global_start_time: {global_start_time}')
    
    tasks = []
    time_offset = 0
    time_offsets = [(global_start_time, 0)]
    timeout = 60
    
    real_arrival_times = {}
    
    bid_to_id = {}
    timed_out_requests: set[str] = set()
    n_rejected = 0
    n_timed_out = 0
    
    with EnergyMeter() as energy_meter:
        rec = EnergyHistoryRecorder(energy_meter, interval_s=0.1, csv_path=f"{problem.store_prefix}.energy.csv")
        rec.start()
        while finished_bar.n < len(requests):
            elapsed_time = time.time() - global_start_time + time_offset
            
            while arrival_idx < len(requests) and arrival_times[arrival_idx] <= elapsed_time:
                request = requests[arrival_idx]
                # if request.prompt is None:
                #     prompt = [random.randint(1000, 2000) for _ in range(request.input_length)]
                # else:
                assert request.prompt is not None
                prompt = request.prompt
                assert prompt is not None
                task_start_time = time.time()
                request_id_backend = str(uuid.uuid4())
                request_id = str(arrival_idx)
                bid_to_id[request_id_backend] = request_id
                real_arrival_times[request_id] = task_start_time
                task = asyncio.create_task(run_request(endpoint,
                                                    request_id_backend,
                                                    problem.model_name, 
                                                    prompt,
                                                    request.input_length,
                                                    request.output_length + request.thinking_length,
                                                    zero_load_ttft = perf_model.get_zero_load_ttft(request.input_length, request.cached_length),
                                                    ttft_slo_scale = problem.ttft_slo_scale,
                                                    slo_routing_overhead = problem.slo_routing_overhead,
                                                    expected_profit = problem.get_expected_profit(request.input_length)))

                tasks.append((task, request, task_start_time, request_id))
                arrival_bar.update(1)
                arrival_bar.set_description(f'Arrival Time: {arrival_times[arrival_idx]:.2f}, Elapsed Time: {elapsed_time:.2f}')
                arrival_idx += 1
                
            real_time = time.time()
            elapsed_time = real_time - global_start_time + time_offset
            if finished_bar.n == arrival_bar.n and arrival_idx < len(requests) and (arrival_times[arrival_idx] - elapsed_time > 10):
                time_offset += arrival_times[arrival_idx] - elapsed_time -1
                time_offsets.append((real_time, time_offset))
                continue
            # Only check finished requests without waiting, and keep unfinished tasks in the list
            new_tasks = []
            current_time = time.time()

            for task, request, task_start_time, request_id in tasks:
                if task.done():
                    timed_out_before = request_id in timed_out_requests
                    finished_bar.update(1)
                    finished_bar.set_description(f'Finished: {finished_bar.n}, Rejected: {n_rejected}, Timed Out: {n_timed_out}')

                    # ---- explicit status checks ----
                    if task.cancelled():
                        logger.info(f"Request {request_id} cancelled before completion")
                        execution_results.append(ExecutionResult(request, [task_start_time], request_id))
                        timed_out_requests.discard(request_id)
                        continue

                    exc = task.exception()  # safe here because task.done() is True and not cancelled
                    if exc is not None:
                        # ===== task FAILED =====
                        logger.error(f"Task for request {request_id} failed: {exc!r}")
                        import traceback
                        logger.error("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
                        # Record something for failures if you track them
                        # failed_requests.append((request_id, exc))
                        timed_out_requests.discard(request_id)
                        continue

                    # ===== task SUCCEEDED =====
                    try:
                        is_rejected, response_text, timestamps = task.result()  # no exception expected now
                        if is_rejected:
                            n_rejected += 1
                        if not len(timestamps):
                            timestamps = [task_start_time]
                            # logger.warning(f"Request {request_id} finished with 0 timestamps")
                        execution_results.append(ExecutionResult(request, timestamps, request_id))
                        if timed_out_before:
                            logger.info(f"Request {request_id} returned after timeout")
                    finally:
                        timed_out_requests.discard(request_id)

                elif request_id in timed_out_requests:
                    new_tasks.append((task, request, task_start_time, request_id))

                elif current_time - task_start_time > timeout:
                    # logger.warning(f"Request {request_id} timed out")
                    n_timed_out += 1
                    task.cancel()
                    timed_out_requests.add(request_id)
                    new_tasks.append((task, request, task_start_time, request_id))

                else:
                    new_tasks.append((task, request, task_start_time, request_id))
            tasks = new_tasks
            await asyncio.sleep(window_size)
        
        rec.stop()
        energy_events = get_energy_events(f"{problem.store_prefix}.energy.csv")

    arrival_bar.close()
    finished_bar.close()
    
    def apply_time_offsets(t: float):
        idx = len(time_offsets) - 1
        while idx >= 0 and time_offsets[idx][0] > t:
            idx -= 1
        if idx < 0:
            idx = 0
        return time_offsets[idx][1] + t - global_start_time

    for result in execution_results:
        result.timestamps = [apply_time_offsets(t) for t in result.timestamps]
        
    import json
    import os
    i = 0
    filename = f'{problem.store_prefix}.{i}.events.jsonl'
    while os.path.exists(filename):
        i += 1
        filename = f'{problem.store_prefix}.{i}.events.jsonl'
    
    admission_filename = f'{problem.store_prefix}.{i}.admission_history.jsonl'

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{endpoint}/dump_profile_events",
            json={"filename": filename, "admission_filename": admission_filename, "timeout": 30.0},
            timeout=30.0
        )
        response.raise_for_status()
        
    
    
    with open(filename, 'r') as f:
        events = json.load(f)
        events.extend(energy_events)
        events = sorted(events, key=lambda x: x['timestamp'])
        # print(bid_to_id.keys())
        backend_id_2_id = lambda id: bid_to_id.get(('-'.join(id.split('-')[1:-1]) if id.startswith('cmpl-') else id), '-1')
        for event in events:
            if 'request_id' in event:
                event['request_id'] = backend_id_2_id(event['request_id'])  
            if event['event_type'] == 'batch':
                event['req_ids'] = [backend_id_2_id(req_id) for req_id in event['req_ids']]
                event['num_scheduled_tokens'] = {backend_id_2_id(req_id): num_scheduled_tokens for req_id, num_scheduled_tokens in event['num_scheduled_tokens'].items()}
            if event['event_type'] == 'req_state':
                event['ddl'] = apply_time_offsets(event['ddl'])
            if event['event_type'] == 'arrival':
                event['add_req_time'] = apply_time_offsets(event['add_req_time'])
            if event['event_type'] == 'schedule_problem':
                for req in event['reqs']:
                    req['id'] = backend_id_2_id(req['id'])
                event['accepted_ids'] = [backend_id_2_id(req_id) for req_id in event['accepted_ids']]
                for batch in event['batch_schedule']:
                    batch['id'] = backend_id_2_id(batch['id'])
        estimated_batch_times = {}
        for event in events:
            if event['event_type'] == 'schedule_problem':
                estimated_batch_times[(event.get('device_id', 0), event['batch_id'])] = event['estimated_time']
        for event in events:
            if event['event_type'] == 'batch':
                event['estimated_time'] = estimated_batch_times.get((event.get('device_id', 0), event['batch_id']), 0)
        with open(filename, 'w') as f:
            json.dump(events, f, indent=4)
    
    with open(filename, 'r') as f:
        events = json.load(f)
        for event in events:
            event['timestamp'] = apply_time_offsets(event['timestamp'])
        for req_id, arrival_time in real_arrival_times.items():
            events.append({
                'event_type': 'global_arrival',
                'request_id': req_id,
                'timestamp': apply_time_offsets(arrival_time),
            })
        for event in events:
            if 'prefill_ddl' in event:
                event['prefill_ddl'] = apply_time_offsets(event['prefill_ddl'])
        events = sorted(events, key=lambda x: x['timestamp'])
    with open(filename, 'w') as f:
        json.dump(events, f, indent = 4)
    print(f'Saved {filename}')
    
    
    
    events, reqs = analyze_events(filename, verbose = True)
    results = analyze_slo_violation(reqs, events, 
                                    model_name = problem.model_name, 
                                    length_pattern = problem.length_pattern,
                                    ttft_slo_scale = problem.ttft_slo_scale, 
                                    slo_tpot = problem.slo_tpot, 
                                    slo_ttft_overhead = problem.slo_routing_overhead,
                                    prefix = problem.store_prefix, 
                                    draw = True)
    if os.path.exists(admission_filename):
        with open(admission_filename, 'r') as f:
            admission_history = json.load(f)
        for event in admission_history:
            event['request_id'] = backend_id_2_id(event['request_id'])
            event['slo_violation'] = reqs[event['request_id']].is_violate_slo()
        with open(admission_filename, 'w') as f:
            json.dump(admission_history, f, indent=4)
        if 'auto_scaling' in problem.routing_policy:
            threshold = problem.routing_kwargs.get('threshold', 0.5)
            model_key = problem.routing_kwargs.get('model_key', 'all')
            auto_scaling_analysis = eval_auto_scaling(model_key, admission_filename,
                                                    threshold = threshold)
        else: 
            auto_scaling_analysis = {}
        results['auto_scaling_analysis'] = auto_scaling_analysis
        
    execution_results = sorted(execution_results, key=lambda x: int(x.request_id))
    results =  ExecutionResults(problem, execution_results, results, f'{problem.store_prefix}.{i}')
    with open(f'{problem.store_prefix}.execution_results.jsonl', 'w') as f:
        json.dump([asdict(execution_result) for execution_result in execution_results], f, indent=4)
    reqs = sorted(list(reqs.values()), key=lambda x: int(x.req_id))
    with open(f'{problem.store_prefix}.reqs.jsonl', 'w') as f:
        json.dump([asdict(req) for req in reqs], f, indent=4)
    print(f'Saved {problem.store_prefix}.{i}.execution_results.jsonl')
    print(f'Saved {problem.store_prefix}.{i}.reqs.jsonl')
    return results


def build_problems(
    model_name: str,
    trace: str,
    ttft_slo_scale: float,
    slo_tpot: float,
    profit: str,
    scheduling_policy: str,
    routing_policy: str,
    n_device: int,  
    window: str,
    load_scale: float,
    experiment_dir: str,
    slo_routing_overhead: float = 0.08,
    admission_mode: str = 'arrival',
):
    store_prefix = f'{experiment_dir}/{scheduling_policy}_{routing_policy}_{load_scale}_{n_device}_{admission_mode}_{ttft_slo_scale}_{slo_tpot}'
    
    window_start, window_end = map(int, window.split(':'))
    if ":" in trace: 
        requests_trace, arrival_times_trace = trace.split(":")
    else:
        requests_trace = trace
        arrival_times_trace = trace
    requests = Requests.load(requests_trace, window_start = window_start, window_end = window_end, 
                             max_tokens = get_model_max_tokens(model_name)).requests
    
    average_input_length = np.mean([request.input_length for request in requests])
    average_output_length = np.mean([request.output_length for request in requests])
    max_output_length = max([request.output_length for request in requests])
    
    print(f'average_input_length: {average_input_length}')
    print(f'average_output_length: {average_output_length}')
    print(f'max_output_length: {max_output_length}')
    from motivation.common import PerfModel
    perf_model = PerfModel.get_perf_model(model_name, requests_trace)
    max_decode_batch_size = perf_model.get_max_decode_batch_size(slo_tpot, average_input_length)
    decode_zero_load = perf_model.get_batch_time([(0, 1)])
    
    assert max_decode_batch_size > 0
    
    print(f'max_decode_batch_size: {max_decode_batch_size}')
    
    slo_ttft_per_token = perf_model.hardware_params[0] * ttft_slo_scale
    slo_ttft_constant = (perf_model.hardware_params[4] + perf_model.hardware_params[1]) * ttft_slo_scale
    assert slo_tpot >= decode_zero_load
    
    average_prefill_time = perf_model.get_batch_time([(0, average_input_length)])
    average_decode_time = slo_tpot * average_output_length / max_decode_batch_size
    optimal_prefill_ratio = average_prefill_time / (average_prefill_time + average_decode_time)
        
    if profit == 'constant': 
        profit_per_input_token = 0.0
        profit_per_output_token = 0.0
        profit_base = 1.0
    elif profit == 'weighted':
        profit_per_input_token = 1.25e-6
        profit_per_output_token = 10.0e-6
        profit_base = 0
    
    max_num_batched_tokens_vllm = min(max_decode_batch_size - 10, 16384)
    scheduling_kwargss = []
    if scheduling_policy == 'vllm':
        scheduling_kwargss.append({
            'scheduling_policy': 'vllm',
            'max_num_batched_tokens': 16384,
            'long_prefill_token_threshold': 16384,
            'max_num_seqs': 512,
            'enable_chunked_prefill': False,
            'enable_admission': False,
            'allow_rejection': False
        })
    elif scheduling_policy == 'vllm+': 
        scheduling_kwargss.append({
            'scheduling_policy': 'vllm+',
            'max_num_batched_tokens': 16384,
            'long_prefill_token_threshold': 16384,
            'max_num_seqs': 512,
            'enable_chunked_prefill': False,
            'enable_admission': True,
            'allow_rejection': True
        })
    elif scheduling_policy == 'sarathi':
        scheduling_kwargss.append({
            'scheduling_policy': 'vllm-sarathi',
            'max_num_batched_tokens': max_num_batched_tokens_vllm,
            'long_prefill_token_threshold': max_num_batched_tokens_vllm,
            'max_num_seqs': 512,
            'enable_chunked_prefill': True,
            'enable_admission': False,
            'allow_rejection': False,
        })
        
    elif scheduling_policy == 'sarathi+':
        scheduling_kwargss.append({
            'max_num_batched_tokens': max_num_batched_tokens_vllm,
            'long_prefill_token_threshold': max_num_batched_tokens_vllm,
            'max_num_seqs': 512,
            'enable_chunked_prefill': True,
            'enable_admission': True,
            'allow_rejection': True,
            'scheduling_policy': 'vllm-sarathi+'
        })
    
    elif scheduling_policy == 'qlm':
        # for maximum_queue_length in [10, 20, 50]:
        scheduling_kwargss.append({
            'enable_chunked_prefill': True,
            'max_num_batched_tokens': max_num_batched_tokens_vllm,
            'max_num_seqs': 512,
            'long_prefill_token_threshold': max_num_batched_tokens_vllm,
            'enable_admission': True,
            'allow_rejection': True,
            'scheduling_policy': 'vllm-edf',
        })
    elif scheduling_policy == 'qlm+':
        scheduling_kwargss.append({
            'enable_chunked_prefill': False,
            'max_num_batched_tokens': 16384,
            'max_num_seqs': 512,
            'long_prefill_token_threshold': 16384,
            'enable_admission': True,
            'allow_rejection': True,
            'scheduling_policy': 'vllm-edf',
        })

    elif scheduling_policy == 'slosserve-edf':
        scheduling_kwargss.append({
            'scheduling_policy': 'edf',
            'enable_admission': True,
            "allow_rejection": True,
            "scheduling_overhead": 0.000,
            "slosserve_token_headroom": 1
        })

    elif scheduling_policy == 'slosserve-dp':
        scheduling_kwargss.append({
            'scheduling_policy': 'dp',
            'enable_admission': True,
            "allow_rejection": True,
            "scheduling_overhead": 0.003,
            "slosserve_token_headroom": 1
        })

    for sch_kwargs in scheduling_kwargss:
        sch_kwargs['max_decoding_length'] = max_output_length
    
    routing_kwargss = []
    if routing_policy == 'round_robin':
        routing_policy = 'round_robin'
        routing_kwargss = [{"enable_rerouting": False,
                            "enable_rescheduling": False}]
    elif routing_policy == 'lightest_first':
        routing_policy = 'lightest_first'
        routing_kwargss = [{"enable_rerouting": False,
                            "enable_rescheduling": False}]
    elif routing_policy == 'lightest_first_retry':
        routing_policy = 'lightest_first'
        routing_kwargss = [{"enable_rerouting": False,
                            "enable_rescheduling": True}]
    elif 'disagg_auto_scaling' in routing_policy:
        _args = routing_policy.split('-')
        if len(_args) == 2:
            _, feature = _args
            threshold = None
        elif len(_args) == 3:
            _, feature, threshold = _args
            threshold = float(threshold)
        routing_kwargss = [{"enable_rescheduling": 'resch' in routing_policy,
                            "enable_rerouting": True,
                            "model_path": "auto_scaling_model.json",
                            "model_key": feature,
                            "threshold": threshold,
                            "max_decode_batch_size": max_decode_batch_size}]
    elif 'auto_scaling' in routing_policy:
        _args = routing_policy.split('-')
        if len(_args) == 2:
            _, feature = _args
            threshold = None
        elif len(_args) == 3:
            _, feature, threshold = _args
            threshold = float(threshold)
        routing_kwargss = [{"enable_rescheduling": 'resch' in routing_policy,
                            "enable_rerouting": False,
                            "model_path": "auto_scaling_model.json",
                            "model_key": feature,
                            "threshold": threshold}]
    elif routing_policy == 'round_robin_retry':
        routing_policy = 'round_robin_retry'
        routing_kwargss = [{
            "enable_rescheduling": True,
            "enable_rerouting": False,
            "round_robin_init": True
        }]
    elif routing_policy in ['disaggregated', 'disaggregated-edf']:
        opt_n_prefill_devices = int(optimal_prefill_ratio * n_device)
        print(f'opt_n_prefill_devices: {opt_n_prefill_devices}')
        tx = lambda n: max(min(n, n_device - 1), 1)
        for num_prefill_devices in {tx(opt_n_prefill_devices),
                                    tx(opt_n_prefill_devices) - 1,
                                    tx(opt_n_prefill_devices) + 1}:
            num_decode_devices = n_device - num_prefill_devices
            if num_decode_devices > 0 and num_prefill_devices > 0:
                routing_kwargss.append(f"{num_prefill_devices}P{num_decode_devices}D")
        
    # elif routing_policy == 'slosserve':
    #     routing_policy = 'slosserve'
    #     routing_kwargss.append({
    #         "hardware_params": hardware_params,
    #         "device_mem": 23949,
    #         "tpot": slo_tpot,
    #         "block_size": 16,
    #         "routing_overhead": slo_routing_overhead
    #     })
    elif routing_policy == 'renaming':
        routing_policy = 'renaming'
        routing_kwargss.append({
            "max_decode_batch_size": max_decode_batch_size,
            "enable_rerouting": True
        })
    
    
    return [Problem(
        model_name = model_name,
        arrival_pattern = arrival_times_trace,
        length_pattern = requests_trace,
        window = window,
        load_scale = load_scale,
        n_devices = n_device,
        ttft_slo_scale = ttft_slo_scale,
        slo_ttft_per_token = slo_ttft_per_token,
        slo_ttft_constant = slo_ttft_constant,
        slo_tpot = slo_tpot,
        slo_routing_overhead = slo_routing_overhead,
        profit_per_input_token = profit_per_input_token,
        profit_per_output_token = profit_per_output_token,
        profit_base = profit_base,
        routing_policy = routing_policy,
        routing_kwargs = routing_kwargs,
        scheduling_policy = scheduling_policy,
        scheduling_kwargs = scheduling_kwargs,
        store_prefix = store_prefix,
        admission_mode = admission_mode,
    ) for (scheduling_kwargs, routing_kwargs) in product(
        scheduling_kwargss, routing_kwargss)]



SCHEDULING_POLICIES = ['vllm-no_rejection', 'vllm-fcfs', 'vllm-edf', 'slosserve-edf', 'slosserve-dp']
ROUTING_POLICIES = ['round_robin', 'disaggregated', 'disaggregated-edf', 'slosserve', 'renaming']

def run(
    model_name: str,
    ttft_slo_scales: list[float],
    slo_tpots: list[float],
    profit: str,
    trace: str,
    window: str,
    load_scales: list[float],
    n_devices: list[int],
    endpoint: str,
    clients: str | None,
    policies: list[str],
    overwrite: bool,
    slo_routing_overhead: float,
    admission_mode: str,
    scheduling_overhead: float,
    output_dir: str,
):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    model_name_easy = get_easy_name(model_name)
    global experiment_dir
    if 'bursty' in trace:
        # cutoff what after bursty
        _trace_name, burstiness_level = trace.split('bursty_')
        _trace_name += 'bursty'
        burstiness_level = float(burstiness_level)
    else:
        _trace_name = trace
        burstiness_level = 0.0
    experiment_dir = f"{output_dir}/{model_name_easy}_{profit}_{_trace_name}_{window}_{admission_mode}_{burstiness_level}"
    import os
    os.makedirs(experiment_dir, exist_ok=True)
    
    print('--Problem Grid--')
    print(f"model_name: {model_name_easy}")
    print(f"ttft_slo_scales: {ttft_slo_scales}")
    print(f"slo_tpots: {slo_tpots}")
    print(f"profit: {profit}")
    print(f"trace: {trace}")
    print(f"window: {window}")
    print(f"load_scales: {load_scales}")
    print(f"n_devices: {n_devices}")
    print(f"policies: {policies}")
    print(f"Experiment directory: {experiment_dir}")    
    print(f"admission_mode: {admission_mode}")
    print(f"scheduling_overhead: {scheduling_overhead}")
    print('--End of Problem Grid--')
    import os
    results = {}
    if os.path.exists(f'{experiment_dir}/results.jsonl'):
        print(f'Loading cached results from {experiment_dir}/results.jsonl')
        with open(f'{experiment_dir}/results.jsonl', 'r') as f:
            results = [json.loads(line) for line in f]
            results = {(r['load_scale'], r['n_device'], r['scheduling_policy'], r['routing_policy'], r['ttft_slo_scale'], r['slo_tpot']): r for r in results}
    else:
        results = {}
    
    for ttft_slo_scale, slo_tpot, load_scale, n_device, policy in product(\
        ttft_slo_scales, slo_tpots, load_scales, n_devices, policies):
        if ':' in policy:
            routing_policy, scheduling_policy = policy.split(':')
        else:
            scheduling_policy = policy
            routing_policy = 'round_robin'
        if n_device == 1 and 'disaggregated' in routing_policy:
            print(f'Skipping {load_scale}, {n_device}, {scheduling_policy}, {routing_policy}, {ttft_slo_scale}, {slo_tpot} because n_device is 1 and routing policy is disaggregated')
            continue
        if not overwrite and (load_scale, n_device, scheduling_policy, routing_policy, ttft_slo_scale, slo_tpot) in results:
            print(f'Skipping {load_scale}, {n_device}, {scheduling_policy}, {routing_policy}, {ttft_slo_scale}, {slo_tpot} because it already exists')
            continue
        problems = build_problems(
            model_name,
            trace,
            ttft_slo_scale,
            slo_tpot,
            profit,
            scheduling_policy,
            routing_policy,
            n_device,
            window,
            load_scale,
            experiment_dir,
            slo_routing_overhead,
            admission_mode,
        )
        if not len(problems):
            print(f'No problems found for {load_scale}, {n_device}, {scheduling_policy}, {routing_policy}, {ttft_slo_scale}, {slo_tpot}')
            continue
        run_results = []
        for problem in problems:
            # print(f"Running problem: {problem}")
            problem.scheduling_kwargs['scheduling_overhead'] = scheduling_overhead
            with EnergyMeter() as energy_meter:
                rec = EnergyHistoryRecorder(energy_meter, interval_s=0.5, csv_path=f"{problem.store_prefix}.energy.csv")
                rec.start()
                exec_result = asyncio.run(main(problem, endpoint, clients))
                rec.stop()
            
            per_gpu, total = energy_meter.read()
            if clients and (int(clients.split(',')[0]) < 10) and len(per_gpu) == problem.n_devices:
                gpu_ids = [int(_) for _ in clients.split(',')][:problem.n_devices]
                per_gpu = [per_gpu[i] for i in gpu_ids]
                total = sum(per_gpu)
            exec_result.energy_consumption = total
            exec_result.per_gpu_energy_consumption = per_gpu
            run_results.append(exec_result)
        best_result = max(run_results, key = lambda x: x.profit)
        result = {
            'load_scale': load_scale,
            'n_device': n_device,
            'scheduling_policy': scheduling_policy,
            'routing_policy': routing_policy,
            'profit': best_result.profit,
            'ttft_slo_scale': ttft_slo_scale,
            'slo_tpot': slo_tpot,
            'slo_violation_rate': 1 - best_result.results['slo_attainment_rate'],
            'event_file': f'{problems[0].store_prefix}.events.jsonl',
            'energy_consumption': best_result.energy_consumption,
            'per_gpu_energy_consumption': best_result.per_gpu_energy_consumption,
            'scheduling_overhead': slo_routing_overhead,
            'burstiness_level': burstiness_level,
        }
        if 'auto_scaling_analysis' in best_result.results:
            result.update(best_result.results['auto_scaling_analysis'])
        if 'extra_metrics' in best_result.results:
            result.update(best_result.results['extra_metrics'])
        print('--Result--')
        pprint.pprint(result)
        print('--End of Result--')
        
        results[(load_scale, n_device, scheduling_policy, routing_policy, ttft_slo_scale, slo_tpot)] = result
        with open(f'{experiment_dir}/results.jsonl', 'a') as f:
            f.write(json.dumps(result) + '\n')            
            
        for surfix in ['events', 'reqs', 'execution_results', 'admission_history']:
            os.system(f'cp {best_result.event_file}.{surfix}.jsonl {problems[0].store_prefix}.{surfix}.jsonl')
    
        for r in run_results:
            os.system(f'rm {r.event_file}*')

    results = list(results.values())

    import pandas as pd
    import matplotlib.pyplot as plt

    # Convert results to DataFrame and save source data
    df = pd.DataFrame(results)
    df.to_csv(f'{experiment_dir}/profit_vs_n_device_and_load.csv', index=False)
    print(f"Saved source data to {experiment_dir}/profit_vs_n_device_and_load.csv")

    # 1. Plot: for each (scheduling_policy, routing_policy) pair, show profit vs n_device (for each load_scale)
    import math
    

    # 1. Plot: for each load_scale, create a subfigure showing profit vs n_device for each (scheduling_policy, routing_policy) pair
    os.makedirs(f'{experiment_dir}/figs', exist_ok=True)
    features = ['load_scale', 'n_device', 'ttft_slo_scale', 'slo_tpot']
    for feature in features:
        if len(df[feature].unique()) == 1:
            continue
        other_features = [f for f in features if f != feature]
        n_groups = len(df.groupby(other_features))
        ncols = min(3, n_groups)
        nrows = math.ceil(n_groups / ncols)
        for ylabel in ['profit', 'slo_violation_rate']:
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
            idx = 0
            for other_feature_values, group in df.groupby(other_features):
                row, col = divmod(idx, ncols)
                ax = axes[row][col]
                idx += 1
                for (sched, route), group in group.groupby(['scheduling_policy', 'routing_policy']):
                    group_sorted = group.sort_values(feature)
                    label = f"{sched} / {route}"
                    ax.plot(group_sorted[feature], group_sorted[ylabel], marker='o', label=label)
                other_features_dict = {f: v for f, v in zip(other_features, other_feature_values)}
                ax.set_xlabel(feature)
                ax.set_ylabel(ylabel)
                ax.set_title(f'{ylabel} vs {feature}\n({other_features_dict})')
                ax.legend()
            fig.tight_layout()
            fig.savefig(f'{experiment_dir}/figs/{ylabel}_vs_{feature}.png', dpi=300)
            print(f"Saved plot to {experiment_dir}/figs/{ylabel}_vs_{feature}.png")

PROBLEM_GRID = {
    'model_name': [
        'Qwen/Qwen2.5-7B-Instruct',
    ],
    'ttft_slo_scales': [2.0, 5.0, 10.0],
    'slo_tpots': [1.5, 3.0, 5.0],
    'profit': ['constant', 'weighted'],
    'trace': [
        'azure_code_23', 
        'azure_chat_23', 
        'azure_code',
        'azure_chat',
        'deepseek-r1:azure_chat',
        'deepseek-r1:azure_code',
    ],
}

example_command = """
python motivation/bench_api_server.py \
    --run_type devices \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --slo strict \
    --profit constant \
    --trace azure_code_23 \
    --window 0:10 \
    --load_scale 1.0 \
    --n_devices 2 4 8
"""


if __name__ == '__main__':
    problem = Problem()
    # execution_results = asyncio.run(main(problem, endpoint))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--ttft_slo_scales', type=float, default=[2.0], nargs='+', help = 'list of relative ttft slo (defined as slowdown to zero-load ttft)')
    parser.add_argument('--slo_tpots', type=float, default=[2.0], nargs='+', help = 'list of relative tpot slo (defined as absolute tpot per token in seconds)')
    parser.add_argument('--profit', type=str, default='constant', choices=['constant', 'weighted'])
    parser.add_argument('--trace', type=str, nargs = '+', default=['azure_code_23'], help = 'list of traces to run LENGTH:ARRIVAL [ARRIVAL:LENGTH ...]')
    parser.add_argument('--window', type=str, default='0:1000', help = 'window of trace to run (inclusive)')
    parser.add_argument('--n_devices', type=int, default=[1,2,4,8], nargs='+', help = 'list of number of devices to run')
    parser.add_argument('--load_scales', type=float, default=[0.5,1.0,2.0,3.0,4.0], nargs='+', help = 'list of load scales (we rescale the arrival rate by load scale, higher load scale means higher query per second)')
    parser.add_argument('--router_ports', type=str, default='8001:4', help = 'port of router to run (inclusive)')
    parser.add_argument('--clients', type=str, default=None, help = 'clients to run (inclusive)')
    parser.add_argument('--run_all', action = 'store_true')
    # parser.add_argument('--scheduling_policies', type=str, default=SCHEDULING_POLICIES, nargs='+')
    # parser.add_argument('--routing_policies', type=str, default=ROUTING_POLICIES, nargs='+')
    parser.add_argument('--overwrite', action = 'store_true')
    parser.add_argument('--slo_routing_overhead', type=float, default=0.02)
    parser.add_argument('--admission_mode', type=str, default='arrival', choices=['arrival', 'anytime'], help = 'arrival: instant decision at arrival, anytime: admission can be made anytime.')
    parser.add_argument('--policies', type=str, default=[':'.join([a,b]) for a, b in product(ROUTING_POLICIES, SCHEDULING_POLICIES)], nargs='+', help = 'list of policies to run (routing_policy:scheduling_policy [routing_policy:scheduling_policy ...])')
    parser.add_argument('--scheduling_overhead', type=float, default=0.003, help = 'scheduling overhead per token in seconds')
    parser.add_argument('--output_dir', type=str, default='experiments', help = 'output directory')
    args = parser.parse_args()
    
    if not args.run_all:
        clients = None
        if args.clients is not None:
            if int(args.clients.split(',')[0]) < 10:
                clients = args.clients
            elif ':' in args.clients:
                client_start, n_clients = map(int, args.clients.split(':'))
                clients = [f'http://localhost:{client_start + i}' for i in range(n_clients)]
            elif ',' in args.clients:
                clients = [f'http://localhost:{client}' for client in args.clients.split(',')]
            else:
                clients = [f'http://localhost:{args.clients}']
        
        for trace in args.trace:
            run (args.model_name,
                args.ttft_slo_scales,
                args.slo_tpots,
                args.profit,
                trace,
                args.window,
                args.load_scales,
                args.n_devices,
                endpoint = f'http://localhost:{args.port}',
                clients = clients,
                policies = args.policies,
                overwrite = args.overwrite,
                slo_routing_overhead = args.slo_routing_overhead,
                admission_mode = args.admission_mode,
                scheduling_overhead = args.scheduling_overhead,
                output_dir = args.output_dir,
            )
        exit(0)
    
    from itertools import product
    import multiprocessing
    
    problem_grids = product(PROBLEM_GRID['model_name'],
                            PROBLEM_GRID['profit'],
                            PROBLEM_GRID['trace'],
                            args.n_devices)
    running_jobs = []
    router_start, n_router_ports = map(int, args.router_ports.split(':'))
    client_start, n_client_ports = map(int, args.clients.split(':'))
    routers = [f'http://localhost:{router_start + i}' for i in range(n_router_ports)]
    clients = [f'http://localhost:{client_start + i}' for i in range(n_client_ports)]
    for grid in problem_grids:
        model_name, profit, trace, n_device = grid
        while len(routers) == 0 or len(clients) < n_device:
            time.sleep(1)
            for p, clients_str, router in running_jobs[:]:
                if p.exitcode is None:
                    continue
                p.join()
                exit_code = p.exitcode
                if exit_code != 0:
                    print(f"Error: Command on GPUs {router} failed with return code {exit_code}: {p.args}")
                running_jobs.remove((p, clients_str, router))
                routers.append(router)
                clients.extend(clients_str.split(','))
        router = routers.pop(0)
        allocated_clients = clients[:n_device]
        clients = clients[n_device:]
        clients_str = ','.join(allocated_clients)

        p = multiprocessing.Process(
            target=run,
            kwargs = {
                'model_name': model_name,
                'ttft_slo_scales': args.ttft_slo_scales,
                'slo_tpots': args.slo_tpots,
                'profit': profit,
                'policies': args.policies,
                'trace': trace,
                'window': args.window,
                'load_scales': args.load_scales,
                'n_devices': [n_device],
                'endpoint': router,
                'clients': clients_str,
                'overwrite': args.overwrite,
                'slo_routing_overhead': args.slo_routing_overhead,
                'admission_mode': args.admission_mode,
            })
        p.start()
        running_jobs.append((p, clients_str, router))

'''
python motivation/bench_api_server.py \
    --run_all \
    --ttft_slo_scales 2.0 5.0 10.0 \
    --slo_tpots 1.5 3.0 5.0 \
    --window 0:1000 \
    --load_scales 1.0 \
    --policies round_robin:vllm-fcfs round_robin:vllm-edf round_robin:slosserve-edf round_robin:slosserve-dp disaggregated:vllm-edf disaggregated:slosserve-edf renaming:slosserve \
    --n_devices 1 2 4 8 16 \
    --router_ports 8001:8 \
    --clients 8501:16
    
python motivation/bench_api_server.py \
    --run_all \
    --ttft_slo_scales 2.0 5.0 10.0 \
    --slo_tpots 1.5 3.0 5.0 \
    --window 0:1000 \
    --load_scales 0.1 0.2 0.4 0.6 0.8 1.0 \
    --policies round_robin:vllm-fcfs round_robin:vllm-edf round_robin:slosserve-edf round_robin:slosserve-dp \
    --n_devices 1 \
    --router_ports 8009:8
    --clients 8517:8
'''
    
    # print('--PROBLEM--')
    # print(problem)
    
    # print('--RESULTS--')
    # print('slo_violation_rate', execution_results.slo_violation_rate)
    # print('profit', execution_results.profit)
    # print('more results', execution_results.results)
    
'''
curl -X POST -s http://0.0.0.0:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "prompt": "San Francisco is a",
    "max_tokens": 10,
    "temperature": 0,
    "stream": true,
    "vllm_xargs": {
        "input_length": 10,
        "output_length": 10,
        "prefill_ddl": 1,
        "profit": 1
    }
}'
'''