'''
An event driven simulator to figure out the energy saving headroom
'''
import math
import argparse
import heapq
import bisect
import json
import csv
import os
import tqdm 
import matplotlib.pyplot as plt
import concurrent.futures
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict


from SLOsServe.router.adm_ctrl import BatchPlanner
from SLOsServe.perf_model import PerfModel

from Dataset.dataset import ArrivalTimes, Requests, Request

class EventQueue:
    def __init__(self):
        self._events = []
        self._idx = 0
    
    def push(self, *,
             t,
             event_type, 
             device_id,
             obj):
        heapq.heappush(self._events, (t, self._idx, event_type, device_id, obj))
        self._idx += 1
    
    def pop(self):
        t, _, event_type, device_id, obj = heapq.heappop(self._events)
        return t, event_type, device_id, obj
    
    def __len__(self):
        return len(self._events)

@dataclass 
class RequestInstance:
    req: Request
    req_id: str
    mode: str # prefill_only, decode_only, normal
    device_id: int = -1
    n_computed_tokens: int = 0
    
    def commit(self, n):
        self.n_computed_tokens += n
        
    def is_finished(self):
        if self.mode == 'prefill_only':
            return self.n_computed_tokens >= self.req.input_length
        return self.n_computed_tokens >= (self.req.input_length + self.req.output_length)
    
@dataclass
class MemManager:
    block_size: int
    total_free_blocks: int
    num_free_blocks: int = field(init = False)
    
    def __post_init__(self):
        self.num_free_blocks = self.total_free_blocks
    
    def _get_n_block(self, n_tokens: int):
        return math.ceil(n_tokens / self.block_size)
    
    def alloc(self, req: RequestInstance, n_scheduled_tokens: int):
        n_current_allocated = self._get_n_block(req.n_computed_tokens)
        n_next_allocated = self._get_n_block(req.n_computed_tokens + n_scheduled_tokens)
        to_alloc = n_next_allocated - n_current_allocated
        if to_alloc > 0:
            if self.num_free_blocks >= to_alloc:
                self.num_free_blocks -= to_alloc
                return True
            return False 
        return True 
        
    def free(self, req: RequestInstance):
        self.num_free_blocks += self._get_n_block(n_tokens = req.n_computed_tokens)
        
class Instance:
    _printed_kv_cache_info = False

    def __init__(self, 
                 device_id: int,
                 event_queue: EventQueue,
                 slo_ttft_scale: float, 
                 slo_tpot: float, 
                 model_name = 'Qwen/Qwen2.5-7B-Instruct',
                 block_size = 16,
                 kv_cache_mem = 20e9,
                 max_decode_length = None,
                 is_oracle: bool = False):
        self.device_id = device_id
        self.event_queue = event_queue
        self.perf_model = PerfModel.get_perf_model(model_name)
        self.slo_ttft_scale = slo_ttft_scale
        self.slo_tpot = slo_tpot
        self.mem_manager = MemManager(
            block_size = block_size,
            total_free_blocks = math.floor(kv_cache_mem / (block_size * self.perf_model.get_kv_mem_per_token()))
        )
        if not Instance._printed_kv_cache_info:
            kv_mem_per_token = self.perf_model.get_kv_mem_per_token()
            print(
                f"[Instance {self.device_id}] kv_mem_per_token={kv_mem_per_token} "
                f"num_free_blocks={self.mem_manager.total_free_blocks}"
            )
            Instance._printed_kv_cache_info = True
        self.batch_planner = BatchPlanner(
            _perf_model = self.perf_model,
            _block_size = block_size,
            _max_decode_length = self.perf_model.get_max_decode_length() if max_decode_length is None else max_decode_length,
            _is_oracle = is_oracle,
            _num_free_blocks = self.mem_manager.total_free_blocks,
        )
        
        self.active_requests: dict[str, RequestInstance] = {}
        self.active_times: tuple[(float, float)] = []
        self.failure_reasons: dict = defaultdict(int)

    def _begin_next_batch(self, now):
        if not len(self.active_requests): return
        n_scheduled_tokens, admitted_requests, exec_plan = self.batch_planner.get_next_batch_and_admitted_reqs()
        failed_req_ids = []
        for req_id, n_tokens in n_scheduled_tokens.items():
            suc = self.mem_manager.alloc(self.active_requests[req_id], n_tokens)
            if not suc:
                print('[Error] Memory allocation failed')
                self.failure_reasons['oom'] += 1
                req = self.active_requests.pop(req_id)
                self.batch_planner.finish_request(req_id)
                req.n_computed_tokens = 0
                # Avoid a tight retry loop when memory conditions don't change.
                self.event_queue.push(t = now + 1e-6, event_type = 'arrival', device_id = -1, obj = req)
                failed_req_ids.append(req_id)

        for req_id in failed_req_ids:
            n_scheduled_tokens.pop(req_id, None)

        if not n_scheduled_tokens:
            return
        
        batch_time = self.perf_model.get_batch_time([(
            self.active_requests[req_id].n_computed_tokens, n_query_tokens
        ) for req_id, n_query_tokens in n_scheduled_tokens.items()])
        self.event_queue.push(
            t = now + batch_time, event_type = 'batch_finish', 
            device_id = self.device_id, obj = n_scheduled_tokens
        )
        self.active_times.append((now, now + batch_time))

    def add_request(self, now: float, req: RequestInstance) -> bool:
        self.batch_planner._now = lambda : now
        zero_load_time = self.perf_model.get_batch_time([(0, req.req.input_length)])
        suc = self.batch_planner.add_request(request_id = req.req_id,
                                            num_prompt_tokens = req.req.input_length,
                                            num_computed_tokens = req.req.input_length if req.mode == 'decode_only' else 0,
                                            prefill_ddl = now + zero_load_time * self.slo_ttft_scale,
                                            slo_tpot = self.slo_tpot,
                                            prefill_only = req.mode == 'prefill_only',
                                            output_length=req.req.output_length)
        if suc:
            self.active_requests[req.req_id] = req
            if len(self.active_requests) == 1:
                self._begin_next_batch(now)
        else:
            if 'MEM' in self.batch_planner._last_infeasible_reason:
                self.failure_reasons['mem'] += 1
            else: self.failure_reasons['comp'] += 1
        return suc
    
    def on_batch_finish(self, now: float, n_scheduled_tokens: dict):
        self.batch_planner._now = lambda : now
        finished_reqs = set()
        for req_id, n_scheduled_token in n_scheduled_tokens.items():
            req = self.active_requests[req_id]
            req.commit(n_scheduled_token)
            if req.is_finished():
                self.active_requests.pop(req_id)
                self.mem_manager.free(req)
                finished_reqs.add(req_id)
                if req.mode == 'prefill_only':
                    req.mode = 'decode_only'
                    self.event_queue.push(
                        t = now + 1e-6, event_type = 'arrival_decode',
                        device_id = -1, obj = req
                    )
                else:
                    self.event_queue.push(
                        t = now + 1e-6, event_type = 'request_finish', 
                        device_id = self.device_id, obj = req
                    )
        self.batch_planner.commit_batch(n_scheduled_tokens, finished_reqs, self.mem_manager.num_free_blocks)
        if len(self.active_requests):
            self._begin_next_batch(now)

_DATASET_CACHE: dict[tuple[str, str], tuple[list[float], list[Request]]] = {}

def _get_dataset_data(arrival_pattern: str, length_pattern: str) -> tuple[list[float], list[Request]]:
    cache_key = (arrival_pattern, length_pattern)
    cached = _DATASET_CACHE.get(cache_key)
    if cached is not None:
        return cached
    arrivals = ArrivalTimes.load(arrival_pattern)
    requests = Requests.load(length_pattern).requests
    for req in requests:
        req.input_length -= req.cached_length
        
    times = arrivals.arrival_times
    if not times or not requests:
        data = ([], [])
        _DATASET_CACHE[cache_key] = data
        return data
    max_len = min(len(times), len(requests))
    times = times[:max_len]
    requests = requests[:max_len]
    data = (times, requests)
    _DATASET_CACHE[cache_key] = data
    return data


def calc_avg_num_servers(
    arrival_pattern: str,
    length_pattern: str,
    model_name: str,
    slo_ttft_scale: float,
    slo_tpot: float,
    n_server: int,
    is_oracle: bool = False,
    n_req: int = -1,
    arrival_window_start: float | None = None,
    arrival_window_end: float | None = None,
    arrival_times_list: list[float] | None = None,
    requests_list: list[Request] | None = None,
    is_pd_disagg: bool = False,
    verbose: bool = True,
):
    if arrival_times_list is not None and requests_list is not None:
        arrival_times = ArrivalTimes(arrival_pattern, arrival_times_list)
        lengths = Requests(length_pattern, requests_list)
    else:
        if arrival_window_start is not None or arrival_window_end is not None:
            if arrival_window_start is None:
                arrival_window_start = 0
            arrival_times = ArrivalTimes.load(
                arrival_pattern,
                window_start = f"t{arrival_window_start}",
                window_end = arrival_window_end,
            )
        else:
            window_end = None if n_req is None or n_req < 0 else n_req
            arrival_times = ArrivalTimes.load(arrival_pattern, window_end = window_end)
        lengths = Requests.load(length_pattern, window_start = 0, window_end = len(arrival_times.arrival_times))
    event_queue: EventQueue = EventQueue()
    instances = [Instance(model_name = model_name, 
                          device_id = device_id,
                          event_queue = event_queue,
                          slo_ttft_scale=slo_ttft_scale,
                          slo_tpot = slo_tpot,
                          is_oracle = is_oracle,
                          max_decode_length = int(np.percentile([r.output_length for r in lengths.requests], 80))) for device_id in range(n_server)]
    for i, (t, req) in enumerate(zip(arrival_times.arrival_times, lengths.requests)):
        event_queue.push(t = t, event_type = "arrival", device_id = -1, obj = RequestInstance(req, f'req-{i}',mode = 'prefill_only' if is_pd_disagg else 'normal'))
    
    n_total = len(event_queue)
    if verbose:
        p_arrival = tqdm.tqdm(total=n_total, desc="arrival")
        p_finish = tqdm.tqdm(total=n_total, desc="finish")
    else:
        p_arrival = None
        p_finish = None
    reject_reasons: dict[str, int] = {}
    while len(event_queue):
        now, event_type, device_id, obj = event_queue.pop()
        if event_type == 'arrival':
            assert isinstance(obj, RequestInstance)
            if p_arrival is not None:
                p_arrival.update(1)
            acc = False
            for device_id, instance in enumerate(instances):
                acc = instance.add_request(now, obj)
                if acc:
                    break
            if not acc:
                reason = instance.batch_planner._last_infeasible_reason or "UNKNOWN"
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
        elif event_type == 'arrival_decode':
            assert isinstance(obj, RequestInstance)
            assert obj.mode == 'decode_only'
            acc = False
            for device_id, instance in enumerate(instances[::-1]):
                device_id = len(instances) - device_id - 1
                acc = instance.add_request(now, obj)
                if acc:
                    break
            if not acc:
                reason = instance.batch_planner._last_infeasible_reason or "UNKNOWN"
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
        elif event_type == 'batch_finish':
            instance = instances[device_id]
            instance.on_batch_finish(now, obj)
        elif event_type == 'request_finish':
            if p_finish is not None:
                p_finish.update(1)
    if p_arrival is not None:
        p_arrival.close()
    if p_finish is not None:
        p_finish.close()
    if reject_reasons:
        print("Rejected requests by reason:")
        for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
            print(f"  {count}: {reason}")
    
    all_idle_time = 0.0
    idle_time_ratio = 0
    avg_active_servers_during_at_least_one_active_time = 0.0
    peak2min_active = 0.0
    total_active_time = 0.0
    all_active_intervals = []
    for instance in instances:
        if not instance.active_times:
            continue
        total_active_time += sum(end - start for start, end in instance.active_times)
        all_active_intervals.extend(instance.active_times)

    if all_active_intervals:
        events = []
        for start, end in all_active_intervals:
            events.append((start, 1))
            events.append((end, -1))
        events.sort(key=lambda x: x[0])
        cur_active = 0
        min_active = None
        max_active = 0
        i = 0
        while i < len(events):
            t = events[i][0]
            while i < len(events) and events[i][0] == t:
                cur_active += events[i][1]
                i += 1
            if i < len(events):
                next_t = events[i][0]
                if next_t > t and cur_active > 0:
                    if min_active is None or cur_active < min_active:
                        min_active = cur_active
                    if cur_active > max_active:
                        max_active = cur_active
        if min_active:
            peak2min_active = max_active / min_active

        all_active_intervals.sort(key=lambda x: x[0])
        merged_total = 0.0
        cur_start, cur_end = all_active_intervals[0]
        for start, end in all_active_intervals[1:]:
            if start <= cur_end:
                if end > cur_end:
                    cur_end = end
            else:
                merged_total += cur_end - cur_start
                cur_start, cur_end = start, end
        merged_total += cur_end - cur_start
        span_total = all_active_intervals[-1][1] - all_active_intervals[0][0]
        all_idle_time = max(0.0, span_total - merged_total)
        idle_time_ratio = all_idle_time / span_total
        if merged_total > 0.0:
            avg_active_servers_during_at_least_one_active_time = total_active_time / merged_total
    
    avg2peak = avg_active_servers_during_at_least_one_active_time / (max_active)

    total_comp_fail = 0
    total_mem_fail = 0
    total_oom_fail = 0
    per_instance_failures = []
    for instance in instances:
        comp_fail = instance.failure_reasons.get('comp', 0)
        mem_fail = instance.failure_reasons.get('mem', 0)
        oom_fail = instance.failure_reasons.get('oom', 0)
        total_comp_fail += comp_fail
        total_mem_fail += mem_fail
        total_oom_fail += oom_fail
        per_instance_failures.append({
            "device_id": instance.device_id,
            "comp": comp_fail,
            "mem": mem_fail,
            "oom": oom_fail,
            "total": comp_fail + mem_fail + oom_fail,
        })

    aggregated_failures = {"comp": total_comp_fail, "mem": total_mem_fail, "oom": total_oom_fail}
    print(
        "Failure summary: "
        f"comp={aggregated_failures['comp']} "
        f"mem={aggregated_failures['mem']} "
        f"oom={aggregated_failures['oom']}"
    )
    for entry in per_instance_failures:
        if entry['total'] > 0:
            print(
                "  "
                f"device={entry['device_id']} "
                f"comp={entry['comp']} "
                f"mem={entry['mem']} "
                f"oom={entry['oom']} "
                f"total={entry['total']}"
            )

    return (
        avg2peak,
        peak2min_active,
        avg_active_servers_during_at_least_one_active_time,
        max_active,
        idle_time_ratio,
        aggregated_failures,
        per_instance_failures,
    )


def _summarize_dataset(arrival_pattern: str, length_pattern: str, dataset_label: str):
    arrivals = ArrivalTimes.load(arrival_pattern)
    requests = Requests.load(length_pattern).requests
    times = arrivals.arrival_times
    if not times or not requests:
        return {
            "dataset": dataset_label,
            "n_req": 0,
            "trace_start": None,
            "trace_end": None,
            "duration": 0.0,
        }
    max_len = min(len(times), len(requests))
    times = np.asarray(times[:max_len], dtype=np.float64)
    reqs = requests[:max_len]
    trace_start = float(times[0])
    trace_end = float(times[-1])
    duration = max(0.0, trace_end - trace_start)
    inter_arrival = np.diff(times) if len(times) > 1 else np.array([0.0], dtype=np.float64)
    in_lens = np.asarray([r.input_length for r in reqs], dtype=np.float64)
    out_lens = np.asarray([r.output_length for r in reqs], dtype=np.float64)
    total_lens = in_lens + out_lens
    return {
        "dataset": dataset_label,
        "n_req": int(max_len),
        "trace_start": trace_start,
        "trace_end": trace_end,
        "duration": duration,
        "mean_rps": float(max_len / duration) if duration > 0 else 0.0,
        "ia_mean": float(inter_arrival.mean()),
        "ia_p50": float(np.percentile(inter_arrival, 50)),
        "ia_p95": float(np.percentile(inter_arrival, 95)),
        "in_mean": float(in_lens.mean()),
        "out_mean": float(out_lens.mean()),
        "out_p80": float(np.percentile(out_lens, 80)),
        "out_p95": float(np.percentile(out_lens, 95)),
        "out_max": float(out_lens.max()),
        "total_p50": float(np.percentile(total_lens, 50)),
        "total_p95": float(np.percentile(total_lens, 95)),
    }


def _compute_dataset_windows(
    dataset_label: str,
    arrival_pattern: str,
    length_pattern: str,
    window_minutes: list[int],
    model_name: str,
    slo_ttft_scale: float,
    slo_tpot: float,
    n_server: int,
    is_oracle: bool = False,
    is_pd_disagg: bool = False,
    parallel: bool = False,
    max_workers: int | None = None,
):
    rows = []
    arrivals = ArrivalTimes.load(arrival_pattern)
    requests = Requests.load(length_pattern).requests
    times = arrivals.arrival_times
    if not times:
        return [], {}
    trace_start = times[0]
    trace_end = times[-1]
    max_len = min(len(times), len(requests))
    times = times[:max_len]
    requests = requests[:max_len]

    results_by_dataset_window: dict[tuple[str, int], list[float]] = {}
    jobs = []
    for minutes in window_minutes:
        window_size = minutes * 60
        window_start = trace_start
        while window_start < trace_end:
            window_end = window_start + window_size
            idx_start = bisect.bisect_left(times, window_start)
            idx_end = bisect.bisect_left(times, window_end)
            if idx_end > idx_start:
                window_times = times[idx_start:idx_end]
                window_requests = requests[idx_start:idx_end]
                jobs.append((
                    dataset_label,
                    arrival_pattern,
                    length_pattern,
                    minutes,
                    window_start,
                    window_end,
                    model_name,
                    slo_ttft_scale,
                    slo_tpot,
                    n_server,
                    is_oracle,
                    is_pd_disagg,
                    window_times,
                    window_requests,
                ))
            window_start = window_end

    if parallel and jobs:
        if max_workers is None:
            max_workers = min(len(jobs), os.cpu_count() or 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_compute_window_from_lists, *job): job for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                row, key, avg2peak, msg = result
                rows.append(row)
                results_by_dataset_window.setdefault(key, []).append(avg2peak)
                if msg:
                    print(msg)
    else:
        for job in jobs:
            result = _compute_window_from_lists(*job)
            if result is None:
                continue
            row, key, avg2peak, msg = result
            rows.append(row)
            results_by_dataset_window.setdefault(key, []).append(avg2peak)
            if msg:
                print(msg)
        
    return rows, results_by_dataset_window

def _compute_window_from_lists(
    dataset_label: str,
    arrival_pattern: str,
    length_pattern: str,
    minutes: int,
    window_start: float,
    window_end: float,
    model_name: str,
    slo_ttft_scale: float,
    slo_tpot: float,
    n_server: int,
    is_oracle: bool,
    is_pd_disagg: bool,
    window_times: list[float],
    window_requests: list[Request],
):
    if not window_times:
        return None
    in_lens = np.fromiter((r.input_length for r in window_requests), dtype=np.float64, count=len(window_requests))
    out_lens = np.fromiter((r.output_length for r in window_requests), dtype=np.float64, count=len(window_requests))
    il_mean = float(in_lens.mean())
    il_max = float(in_lens.max())
    il_p95 = float(np.percentile(in_lens, 95))
    il_median = float(np.percentile(in_lens, 50))
    ol_mean = float(out_lens.mean())
    ol_max = float(out_lens.max())
    ol_p95 = float(np.percentile(out_lens, 95))
    ol_median = float(np.percentile(out_lens, 50))
    avg2peak, peak2min, n_active, n_total, idle_ratio, aggregated_failures, _ = calc_avg_num_servers(
        arrival_pattern = arrival_pattern,
        length_pattern = length_pattern,
        model_name = model_name,
        slo_ttft_scale = slo_ttft_scale,
        slo_tpot = slo_tpot,
        n_server = n_server,
        is_oracle = is_oracle,
        is_pd_disagg = is_pd_disagg,
        arrival_times_list = window_times,
        requests_list = window_requests,
        verbose = True,
    )
    row = [
        dataset_label,
        is_oracle,
        is_pd_disagg,
        minutes,
        window_start,
        window_end,
        avg2peak,
        peak2min,
        n_active,
        n_total,
        idle_ratio, 
        il_mean,
        il_max,
        il_p95,
        il_median,
        ol_mean,
        ol_max,
        ol_p95,
        ol_median,
        aggregated_failures["comp"],
        aggregated_failures["mem"],
        aggregated_failures["oom"],
        aggregated_failures["comp"] + aggregated_failures["mem"] + aggregated_failures["oom"],
    ]
    msg = (
        f'{dataset_label} {minutes}min '
        f'window[{window_start:.3f},{window_end:.3f}) n_req {len(window_times)} '
        f'avg2peak {avg2peak:.6f} peak2min {peak2min:.6f} n_avg {n_active:.6f} n_total {n_total} idle {idle_ratio} '
        f'il(mean {il_mean:.2f} max {il_max:.2f} p95 {il_p95:.2f} p50 {il_median:.2f}) '
        f'ol(mean {ol_mean:.2f} max {ol_max:.2f} p95 {ol_p95:.2f} p50 {ol_median:.2f})'
    )
    return row, (dataset_label, minutes), avg2peak, msg

def _compute_window(
    arrival_pattern: str,
    length_pattern: str,
    dataset_label: str,
    minutes: int,
    window_start: float,
    window_end: float,
    model_name: str,
    slo_ttft_scale: float,
    slo_tpot: float,
    n_server: int,
    is_oracle: bool = False,
    is_pd_disagg: bool = False,
):
    times, requests = _get_dataset_data(arrival_pattern, length_pattern)
    if not times:
        return None
    idx_start = bisect.bisect_left(times, window_start)
    idx_end = bisect.bisect_left(times, window_end)
    if idx_end <= idx_start:
        return None
    window_times = times[idx_start:idx_end]
    window_requests = requests[idx_start:idx_end]
    in_lens = np.fromiter((r.input_length for r in window_requests), dtype=np.float64, count=len(window_requests))
    out_lens = np.fromiter((r.output_length for r in window_requests), dtype=np.float64, count=len(window_requests))
    il_mean = float(in_lens.mean())
    il_max = float(in_lens.max())
    il_p95 = float(np.percentile(in_lens, 95))
    il_median = float(np.percentile(in_lens, 50))
    ol_mean = float(out_lens.mean())
    ol_max = float(out_lens.max())
    ol_p95 = float(np.percentile(out_lens, 95))
    ol_median = float(np.percentile(out_lens, 50))
    avg2peak, peak2min, n_active, n_total, idle_ratio, aggregated_failures, _ = calc_avg_num_servers(
        arrival_pattern = arrival_pattern,
        length_pattern = length_pattern,
        model_name = model_name,
        slo_ttft_scale = slo_ttft_scale,
        slo_tpot = slo_tpot,
        n_server = n_server,
        is_oracle = is_oracle,
        is_pd_disagg = is_pd_disagg,
        arrival_times_list = window_times,
        requests_list = window_requests,
        verbose = False,
    )
    row = [
        dataset_label,
        is_oracle,
        is_pd_disagg,
        minutes,
        window_start,
        window_end,
        avg2peak,
        peak2min,
        n_active,
        n_total,
        idle_ratio,
        il_mean,
        il_max,
        il_p95,
        il_median,
        ol_mean,
        ol_max,
        ol_p95,
        ol_median,
        aggregated_failures["comp"],
        aggregated_failures["mem"],
        aggregated_failures["oom"],
        aggregated_failures["comp"] + aggregated_failures["mem"] + aggregated_failures["oom"],
    ]
    print(
        f'{dataset_label} {minutes}min '
        f'window[{window_start:.3f},{window_end:.3f}) n_req {idx_end - idx_start} '
        f'avg2peak {avg2peak:.6f} peak2min {peak2min:.6f} n_avg {n_active:.6f} n_total {n_total} idle {idle_ratio} '
        f'il(mean {il_mean:.2f} max {il_max:.2f} p95 {il_p95:.2f} p50 {il_median:.2f}) '
        f'ol(mean {ol_mean:.2f} max {ol_max:.2f} p95 {ol_p95:.2f} p50 {ol_median:.2f})'
    )
    return row, (dataset_label, minutes), avg2peak


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Headroom analysis")
    parser.add_argument("--arrival-pattern", default="azure_chat_23")
    parser.add_argument("--length-pattern", default=None)
    parser.add_argument("--dataset-label", default=None)
    parser.add_argument("--is-oracle", action="store_true")
    parser.add_argument("--is-pd-disagg", action="store_true")
    parser.add_argument("--window-minutes", type=int, nargs="+", default=[5, 10, 30])
    parser.add_argument("--no-confirm", action="store_true")
    parser.add_argument("--output_name", default = 'headroom')
    args = parser.parse_args()

    arrival_pattern = args.arrival_pattern
    length_pattern = args.length_pattern or arrival_pattern
    dataset_label = args.dataset_label or f"{arrival_pattern}:{length_pattern}"
    datasets = [dataset_label]
    window_minutes = args.window_minutes
    is_oracle = args.is_oracle
    is_pd_disagg = args.is_pd_disagg
    out_dir = "headroom_outputs"
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{args.output_name}.csv")
    failed_jobs_path = os.path.join(out_dir, "failed_window_jobs.jsonl")
    print("Dataset summary (entire trace):")
    dataset_stats = {}
    for dataset in datasets:
        stats = _summarize_dataset(arrival_pattern, length_pattern, dataset)
        if stats["n_req"] == 0:
            print(f"  {dataset}: empty")
            continue
        dataset_stats[dataset] = stats
        print(
            f"  {dataset}: n_req={stats['n_req']} "
            f"t=[{stats['trace_start']:.3f},{stats['trace_end']:.3f}] "
            f"dur={stats['duration']:.3f}s "
            f"mean_rps={stats['mean_rps']:.3f} "
            f"ia_mean={stats['ia_mean']:.3f}s "
            f"ia_p50={stats['ia_p50']:.3f}s "
            f"ia_p95={stats['ia_p95']:.3f}s "
            f"in_mean={stats['in_mean']:.2f} "
            f"out_mean={stats['out_mean']:.2f} "
            f"out_p95={stats['out_p95']:.2f} "
            f"out_p80={stats['out_p80']:.2f} "
            f"out_max={stats['out_max']:.2f} "
            f"total_p50={stats['total_p50']:.2f} "
            f"total_p95={stats['total_p95']:.2f}"
        )
    if not args.no_confirm:
        input("Press Enter to continue with headroom analysis... ")

    results_by_dataset_window: dict[tuple[str, int], list[float]] = {}
    rows = []
    jobs = []
    for dataset, stats in dataset_stats.items():
        trace_start = stats["trace_start"]
        trace_end = stats["trace_end"]
        for minutes in window_minutes:
            window_size = minutes * 60
            window_start = trace_start
            while window_start < trace_end:
                window_end = window_start + window_size
                jobs.append((arrival_pattern, length_pattern, dataset, minutes, window_start, window_end))
                window_start = window_end

    if not jobs:
        print("No window jobs to run.")
    else:
        max_workers = min(len(jobs), os.cpu_count() or 1)
    header = [
        "dataset",
        "is_oracle",
        "is_pd_disagg",
        "window_minutes",
        "window_start",
        "window_end",
        "avg2peak",
        "peak2min",
        "n_avg_active",
        "n_total",
        "idle_ratio",
        "il_mean",
        "il_max",
        "il_p95",
        "il_median",
        "ol_mean",
        "ol_max",
        "ol_p95",
        "ol_median",
        "fail_comp",
        "fail_mem",
        "fail_oom",
        "fail_total",
    ]
    failed_jobs = []
    if jobs:
        csv_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(header)
                f.flush()

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _compute_window,
                        job_arrival_pattern,
                        job_length_pattern,
                        dataset,
                        minutes,
                        window_start,
                        window_end,
                        'Qwen/Qwen2.5-7B-Instruct',
                        5,
                        0.05,
                        100,
                        is_oracle,
                        is_pd_disagg,
                    ): (job_arrival_pattern, job_length_pattern, dataset, minutes, window_start, window_end)
                    for job_arrival_pattern, job_length_pattern, dataset, minutes, window_start, window_end in jobs
                }
                p_datasets = tqdm.tqdm(
                    total=len(futures),
                    desc="windows",
                    unit="win",
                    dynamic_ncols=True,
                    smoothing=0.1,
                )
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as exc:
                        job = futures[future]
                        print(f"Window job {job} failed: {exc}")
                        failed_jobs.append({
                            "dataset": job[2],
                            "window_minutes": job[3],
                            "window_start": job[4],
                            "window_end": job[5],
                            "error": repr(exc),
                        })
                        p_datasets.update(1)
                        continue
                    if result is not None:
                        row, key, avg2peak = result
                        writer.writerow(row)
                        f.flush()
                        results_by_dataset_window.setdefault(key, []).append(avg2peak)
                    p_datasets.update(1)
                p_datasets.close()

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            file_header = next(reader, None)
            file_rows = [row for row in reader if row]

        def _row_sort_key(row: list[str]) -> tuple[object, ...]:
            def _parse_bool(value: str) -> int:
                if value in {"1", "true", "True", "yes", "Yes"}:
                    return 1
                if value in {"0", "false", "False", "no", "No"}:
                    return 0
                raise ValueError(f"Invalid bool value: {value}")

            try:
                return (
                    row[0],
                    _parse_bool(row[1]),
                    _parse_bool(row[2]),
                    int(row[3]),
                    float(row[4]),
                    float(row[5]),
                )
            except (ValueError, IndexError):
                return tuple(row)

        file_rows.sort(key=_row_sort_key)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(file_header or header)
            writer.writerows(file_rows)

    if failed_jobs:
        with open(failed_jobs_path, "a", newline="") as f:
            for entry in failed_jobs:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved {len(failed_jobs)} failed jobs to {failed_jobs_path}")

    for dataset in datasets:
        fig, axes = plt.subplots(len(window_minutes), 1, figsize=(6, 3 * len(window_minutes)), squeeze=False)
        for i, minutes in enumerate(window_minutes):
            values = results_by_dataset_window.get((dataset, minutes), [])
            ax = axes[i][0]
            if values:
                ax.hist(values, bins=30, color="steelblue", alpha=0.85)
            ax.set_title(f"{dataset} {minutes}min avg2peak distribution")
            ax.set_xlabel("avg2peak")
            ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"avg2peak_dist_{dataset}.png"), dpi=200)
        plt.close(fig)
