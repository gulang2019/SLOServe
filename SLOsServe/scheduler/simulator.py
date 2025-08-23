'''
We want to mimic the request and response of the scheduler.
'''
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Callable, Any
import numpy as np
import random
import tqdm
import argparse
import matplotlib.pyplot as plt
import os 
import json
from pprint import pprint
import time
import math
import heapq

from .struct import (
    Problem,
    SpecDecode,
    SLA,
    ReqBatchSchedule,
    Schedule,
    RequestInstanceBase,
    RequestState,
    MemoryInstance,
    BatchSchedule,
    SchedulerConfig
)

from .schedule_algs import ScheduleALG, get_schedule_alg
from .utils import ExponentialBackoff

@dataclass 
class RequestInstance(RequestInstanceBase):
    def __repr__(self):
        return super(RequestInstance, self).__repr__()
    
    def progress(self, 
                req_sch: ReqBatchSchedule, 
                spec_decode: SpecDecode,
                current_time: float):
        assert req_sch.id == self.id 
        if req_sch.is_prefill:
            if self.state in (RequestState.Prefill, RequestState.Arrived):
                # assert self.state in (RequestState.Prefill, RequestState.Arrived)
                self.state = RequestState.Prefill
                self._n_prefill_tokens += req_sch.n 
                if self._n_prefill_tokens > self.input_length:
                    raise RuntimeError(f'{self._n_prefill_tokens} > {self.input_length} for {self}')
                # if self._n_prefill_tokens > self.input_length:
                #     print(self._n_prefill_tokens, '>', self.input_length, self)
                if self._n_prefill_tokens >= self.input_length: 
                    self.state = RequestState.Decode
                    self._n_decode_tokens = 1
                    self._n_decode_scheduled = 1
                    self._timestamps.append((self._n_decode_tokens, current_time))
        else:
            if self.state != RequestState.Decode: 
                print('wrong state', self.state) 
            assert self.state == RequestState.Decode 
            n_yield = spec_decode.sample(req_sch.n)
            self._n_decode_tokens += n_yield
            self._spec_history.append((req_sch.n, n_yield))
            self._timestamps.append((self._n_decode_tokens, current_time))
        if self._n_decode_tokens >= self.output_length:
            self.state = RequestState.Finished

class RequestQueue:
    def __init__(self, reqs: List[RequestInstance]):
        self.heap = [(req.arrive_time, req) for req in reqs]
        heapq.heapify(self.heap)
    def push(self, req: RequestInstance):
        heapq.heappush(self.heap, (req.arrive_time, req))
    
    def pop(self) -> RequestInstance:
        return heapq.heappop(self.heap)[-1]
    
    def __len__(self)-> int:
        return len(self.heap)
    
    def __getitem__(self, idx)->RequestInstance:
        return self.heap[idx][1]
    

@dataclass
class Simulator:
    config: SchedulerConfig
    
    def simulate(
        self,
        prob: Problem,  
        alg: ScheduleALG,
        mem: MemoryInstance, 
    )->Schedule:
        current_time = 0
        n_finished_reqs = 0
        running_reqs: List[RequestInstance] = []
        new_reqs: List[RequestInstance] = []
        best_effort_reqs: List[RequestInstance] = []
        req_instances = [RequestInstance(i, req, req.arrive_time) for i, req in enumerate(prob.reqs)]
        pending_queue = RequestQueue(req_instances)
        scheduled_batches = []
        overhead = 0
        self.failure_cnt = 0
        self.n_schedules = 0
        batches = []
        bid = 0
        n_preempt = 0
        declined_by_failure = 0
        running_best_effort_reqs: List[RequestInstance] = []
        last_arrive_time = 0
        last_finish_time = 0
        
        assert prob.batch_timer is not None
        
        idx = 0
        n_batches_per_schs = []
        with tqdm.tqdm(total = len(req_instances)) as pbar:
            bid = 0    
            while n_finished_reqs < len(req_instances):
                idx += 1
                for req in running_reqs:
                    assert not req.is_finished()
                                    
                start = time.perf_counter()
                
                if len(new_reqs) == 0 and self.config.sch_strategy == 'promax-dp':
                    # batches = batches[bid:] if bid < len(batches) else [BatchSchedule([])]
                    if len(batches) == 0:
                        batches.append(BatchSchedule([], 0))
                    admitted_reqs, declined_reqs = [], []
                    pass
                else: 
                    sch = alg.schedule(
                        running_reqs, 
                        new_reqs,
                        current_time)
                
                    if sch is None:
                        self.failure_cnt += 1
                        admitted_reqs = []
                        declined_by_failure += len(new_reqs)
                        declined_reqs = new_reqs
                    else:
                        batches, admitted_reqs, declined_reqs = sch
                        bid = 0
                        n_prefill_tokens = {}
                        if len(batches):
                            loop_start = len(batches)-1 + batches[-1].next
                            for i, batch in enumerate(batches):
                                has_prefill = False
                                for req_sch in batch.reqs:
                                    if req_sch.is_prefill:
                                        has_prefill = True
                                        n_prefill_tokens[req_sch.id] = n_prefill_tokens.get(req_sch.id, 0) + req_sch.n
                                assert not has_prefill or i < loop_start 
                            for k, v in n_prefill_tokens.items():
                                req_instance = req_instances[k]
                                assert (req_instance.input_length >= (v + req_instance._n_prefill_tokens))
                        
                overhead += time.perf_counter() - start
                self.n_schedules += 1
                
                if self.config.enable_best_effort:
                    best_effort_reqs.extend(declined_reqs)
                    n_block = sum(req.req.num_blocks for req in admitted_reqs)
                    while n_block > mem.get_n_avail() \
                        and len(running_best_effort_reqs):
                        req = running_best_effort_reqs.pop()
                        mem.free(req)
                        req.preempt()
                        n_preempt += 1
                        best_effort_reqs.append(req)
                    assert n_block <= mem.get_n_avail()
                
                for req in admitted_reqs:
                    req.admitted = True 
                    assert mem.optional_alloc(req)
                
                running_reqs.extend(admitted_reqs)
                
                # if len(batches):
                #     decode_req_ids = set([req_sch.id for req_sch in batches[-1].reqs])
                #     running_req_ids = set([req.id for req in running_reqs])
                #     if (not running_req_ids.issubset(decode_req_ids)):
                #         raise RuntimeError(f"Decode Reqs {decode_req_ids} not contain Run Reqs: {running_req_ids}")
                
                # print('all_declined_reqs', all_declined_reqs, 'running reqs', running_reqs, 'running_best_effort', running_best_effort_reqs)
                if self.config.enable_best_effort:
                    while len(best_effort_reqs) and \
                        mem.get_n_avail() >= best_effort_reqs[0].req.num_blocks:
                        req = best_effort_reqs.pop(0)
                        req.best_effort = True
                        assert mem.optional_alloc(req)
                        running_best_effort_reqs.append(req)
                        admitted_reqs.append(req)
                    
                    # batch = batches[-1]
                    # token_budget = 2048 if len(running_reqs) == 0 else (batches[-1].remain_budget)
                    # for req in running_best_effort_reqs:
                    #     if req.is_prefill() and token_budget >= req.input_length:
                    #         batch.add_req(ReqBatchSchedule(req.id, True, req.input_length))
                    #         token_budget -= req.input_length
                    #     elif not req.is_prefill() and token_budget >= 1:
                    #         batch.add_req(ReqBatchSchedule(req.id, False, 1))
                    #         token_budget -= 1
                    #     if token_budget <= 0: break
                else: 
                    n_finished_reqs += len(declined_reqs)
                    pbar.update(len(declined_reqs))
                
                if not len(running_reqs) or not len(batches): 
                    batches = [BatchSchedule([], next = 0, remain_budget=prob.max_seq_len)]
                    bid = 0
                # print('declined reqs', declined_reqs, 'running_reqs', running_reqs, 'new_reqs', new_reqs)
                
                s = f'{idx:<5} t {int(current_time):<5} #running: {len(running_reqs):<5} #new: {len(new_reqs)}, #pending {len(pending_queue)}, # declined running {len(running_best_effort_reqs)}, # best effort {len(best_effort_reqs)} #preempt {n_preempt} #batch {len(batches)} bid {bid}'
                # s = ''
                # if len(running_best_effort_reqs): 
                #     s += str(running_best_effort_reqs[0])
                #     s += str(batches[bid])
                pbar.set_description(s)
        
                n_newly_arrived_reqs: int = 0
                n_newly_finished_reqs: int = 0
                old_time = current_time
                new_reqs = []
                n_batches_per_sch = 0
                while True:
                    n_batches_per_sch += 1
                    batch = batches[bid]
                    
                    for req in running_best_effort_reqs:
                        if batch.remain_budget <= 0:
                            break
                        assert not req.is_finished()
                        if req.is_prefill() and batch.remain_budget >= req.input_length:
                            batch.add_req(ReqBatchSchedule(req.id, True, req.input_length))
                        elif req.is_decode() and batch.remain_budget >= 1:
                            batch.add_req(ReqBatchSchedule(req.id, False, 1))
                

                    assert (not len(running_reqs)) or (batch.get_effective_bs() > 0) 
                    scheduled_batches.append(batch)
                    
                    t = prob.batch_timer(batch.batch_size, batch.decode_steps)
                    
                    # print("EXEC batch", batch)
                    
                    current_time += t
                    if (not len(running_reqs)) and len(pending_queue):
                        current_time = max(current_time, pending_queue[0].arrive_time)
                    # print(f'exec [{batch.batch_size}]{batch} in {t} to {current_time}')
                    while len(pending_queue) \
                        and pending_queue[0].arrive_time <= current_time:
                        req = pending_queue.pop()
                        if req.req.sla.is_best_effort: 
                            best_effort_reqs.append(req)
                        else:
                            new_reqs.append(req)
                        last_arrive_time = req.arrive_time

                    bid += batch.next
                    
                    
                    
                    for req_sch in batch.reqs:
                        req: RequestInstance = req_instances[req_sch.id]
                        if req_instances[req_sch.id].is_finished():
                            continue
                        req.progress(
                            req_sch, prob.spec_decode, current_time)
                        if req.is_finished():
                            if req.admitted:
                                running_reqs.remove(req)
                            else:
                                assert req.best_effort
                                running_best_effort_reqs.remove(req)
                            mem.free(req)
                            n_finished_reqs += 1
                            n_newly_finished_reqs += 1
                            last_finish_time = current_time
                            pbar.update(1)
                            if req.req.follow_up_req is not None:
                                req_instances.append(RequestInstance(len(req_instances), 
                                                                     req.req.follow_up_req[1], 
                                                                     current_time + req.req.follow_up_req[0]))
                                pending_queue.push(req_instances[-1])
                                pbar.total += 1
                                pbar.refresh()
                
                    if n_newly_finished_reqs > self.config.sch_thresh_finish or\
                        n_newly_arrived_reqs > self.config.sch_thresh_arrive or\
                        (current_time - old_time) > self.config.sch_thresh_timeout:
                        break
                n_batches_per_schs.append(n_batches_per_sch)
                        
        print('#acc:', sum([req.is_finished() for req in req_instances]))
        print('declined by failure', declined_by_failure)
        print('elasped time', current_time)
        print('config', self.config)
        print('#batch', np.mean(n_batches_per_schs), '+-', np.std(n_batches_per_schs))
        print('#schedules', self.n_schedules)
        return Schedule(
            name = self.config.sch_strategy + ('-best-effort' if self.config.enable_best_effort else '') + '-simulated',
            reqs = req_instances,
            overhead = overhead,
            failure_rate=self.failure_cnt / self.n_schedules,
            tail_time = last_finish_time - last_arrive_time
        )

if __name__ == '__main__':
    random.seed(123)
    np.random.seed(123)
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type = str, default = 'run')
    parser.add_argument('--trace', type = str, default= 'humaneval')
    parser.add_argument('--arrival_pattern', type = str, default = 'synthetic')
    parser.add_argument('--req_rate', type = int, default=None)
    parser.add_argument('--burstiness', type = int, default = 5)
    parser.add_argument('--sch', type=str, nargs = '+', choices = ['vllm', 'sarathi', 'slo', 'oracle'])
    parser.add_argument('--slo-ttft', default = 3, type = int)
    parser.add_argument('--slo-tpot', default = 1, type = int)
    parser.add_argument('--n_req', type = int, default = 1000)
    parser.add_argument('-o', '--output', type = str, default = 'simulation')
    parser.add_argument('--num-block', type = int, default = 3e4)
    args = parser.parse_args()

    problem = Problem(
        n_param = 7e9,
        mem_bw = 2e12,
        compute_speed = 312e12,
        
        request_trace=args.trace,
        n_req = args.n_req,
        arrival_pattern=args.arrival_pattern,
        req_rate = args.req_rate,
        n_req_at_once = args.burstiness,
        
        alpha = 0.6,
        
        sla_distribution = [(1., SLA(args.slo_ttft, args.slo_tpot, 10, 1, 8))]
    )
    
    import openbox as ob
    memory = MemoryInstance(args.num_block)
    
    for sch in args.sch:
        schedule_alg: ScheduleALG = get_schedule_alg(sch, problem, memory)

        space = ob.sp.Space()
        space.add_variable(ob.sp.Int('timeout', 1, 5))
        space.add_variable(ob.sp.Int('arrive_thresh', 1, 5))
        space.add_variable(ob.sp.Int('finish_thresh', 1, 5))
        
        def _simulate(config):
            simulator = Simulator(config['timeout'] * 5e-3, config['arrive_thresh'], config['finish_thresh'])
            
            _sch = simulator.simulate(problem, schedule_alg, memory)
            return {'objectives': [-_sch.profit]}
        
        opt = ob.Optimizer(_simulate, space, max_runs=2, task_id = sch)
        history = opt.run()
        configs = history.get_config_dicts()
        objs = history.get_objectives()
        idx = min(range(len(objs)), key = lambda i: objs[i])
        config = configs[idx]
        
        simulator = Simulator(config['timeout'] * 5e-3, config['arrive_thresh'], config['finish_thresh'])
        sch = simulator.simulate(problem, schedule_alg, memory)
        # sch.config = SchedulerConfig(**config)
        problem.add_schedule(sch)
    
    print(problem.schedules)
    
    
    os.makedirs(args.output, exist_ok=True)
    filepath = f'{args.output}/{args.tag}.json'
    
    with open(filepath, 'w') as f:
        json.dump(problem.asdict(), f, indent = 4)
    print(f'result saved to {filepath}')