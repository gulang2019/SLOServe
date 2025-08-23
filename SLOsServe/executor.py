import networkx as nx 
import asyncio 
import ray 
from typing import List, Dict, Tuple, Iterable
import logging 
from dataclasses import dataclass
from pprint import pprint
import matplotlib.pyplot as plt
import os 
import numpy as np
import time

import ray.exceptions
from ray.util.queue import Queue

from .ops import Node, OpCode, OP_CLASSES
from .engine_wrapper import EngineWrapper 
from .object import (
    ObjectStatus, TensorRef, ObjectRef, KVCacheRef, OperationID
)
from .device import ClusterStatus
from .comm import Communicator
from .device import DeviceGroup

logger = logging.getLogger(__name__)

# TODO: Change the class name to Scheduler
class Executor:
    engines: List[EngineWrapper]
    dependency_graph: nx.DiGraph

    @dataclass
    class LogEntry:
        op_name: str

    def __init__(self, 
                engines: List[EngineWrapper],
                window_size: float = 0.01,
                timeout: float = 0.005, 
                enable_adaws: bool = False,
                debug: bool = False,
                sch_tot_budget: float = 10,
                report_dir: str = 'report'
        ):
        self.engines = engines
        self.n_engine = len(self.engines)
        self.output_queues = [Queue() for _ in range(self.n_engine)]
        ray.get([engine.execute_method.remote('set_output_queue', queue) 
                 for queue, engine in zip(self.output_queues, self.engines)])
        self.cluster_status = ClusterStatus(self.n_engine)
        self.timeout = timeout
        self.dependency_graph = nx.DiGraph()
        self.logs = []
        self.remote_jobs = []
        self.job_ids = [0 for _ in range(self.n_engine)]
        self.debug = debug
        self.window_size = window_size
        self.enable_adaws = enable_adaws
        self.estimated_times = []
        self.schedule_times = []
        self.sch_tot_budget = sch_tot_budget
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)
        self.round = 0
        schedule_priority = '<'.join([x.name for x in OpCode])
        logger.info(f'Executor initilaized window: {window_size}'
                    f'scheduler schedule budget: {sch_tot_budget}'
                    f'enable adaptive window size: {enable_adaws}'
                    f'schedule priority: {schedule_priority}')
        self.batch_logs = []
        # self.get_times = []

    def set_request_context(self,
                            request_contexts: Dict[int, 'Context']):
        self.request_contexts = request_contexts

    def report_bk(self):
        print('#Engine: %d' % len(self.engines))
        engines_reports = ray.get([engine.execute_method.remote('report') for engine in self.engines])
        rets = {}
        import pandas as pd
        
        for i, (cache_status, objs, profile) in enumerate(engines_reports):
            print(f'--------ENGINE {i}---------')
            print('objs:', objs)
            dict_status = cache_status
            rets[f'Engine {i}'] = {'status': dict_status}   
            # print('len(profile)', len(profile), len(self.estimated_times), len(self.schedule_times), len(self.batch_logs))
            # print('synchronized:')
            # for i in range(min(len(profile), len(self.batch_logs))):
            #     print(profile[i][2], self.batch_logs[i][0])
            # print('batch logs left over')
            # for i in range(min(len(profile), len(self.batch_logs)), len(self.batch_logs)):
            #     print(self.batch_logs[i][0])
            # print('profile left over')
            # for i in range(min(len(profile), len(self.batch_logs)), len(profile)):
            #     print(profile[i][2])
            if len(profile):
                req_times = {}
                req_start_times = {}
                req_end_times = {}
                assert len(profile) == len(self.estimated_times)
                assert len(profile) == len(self.schedule_times)
                assert len(profile) == len(self.batch_logs)
                idle_time = 0
                last_end_time = self.schedule_times[0]
                real_times = []
                job_names = []
                scheduler_preceeds_times = []
                scheduler_behind_times = []
                active_ops = set()
                data = {}
                for schedule_time, predicted_time, (batch_name, req_ids), (start_time, end_time, job_name, bs) in\
                                        zip(self.schedule_times,
                                            self.estimated_times,
                                            self.batch_logs,
                                            profile):
                    elapsed = (end_time - start_time)
                    
                    
                    scheduler_preceeds_times.append(last_end_time - schedule_time)
                    scheduler_behind_times.append(max(schedule_time - last_end_time, 0))
                    
                    # time_count[job_name] = time_count.get(job_name, 0) + elapsed
                    idle_time += start_time - last_end_time
                    last_end_time = end_time
                    real_times.append(elapsed * 1000)
                    job_names.append(job_name)
                    if batch_name != 'DELETE':
                        active_ops.update([req_id for req_id, _ in req_ids])
                        for req_id, req_arrive_time in req_ids:
                            req_start_times[req_id] = req_start_times.get(req_id, req_arrive_time)
                            req_times[req_id] = req_times.get(req_id, 0) + elapsed
                            req_end_times[req_id] = end_time
                    else: 
                        for req_id, _ in req_ids:
                            active_ops.remove(req_id)
                    
                    if batch_name not in data:
                        data[batch_name] = {
                            '#op': 0, 'delay (s)': 0, 'times': [], 'estimated_times': []
                        }
                    data[batch_name]['estimated_times'].append(predicted_time)
                    data[batch_name]['#op'] += len(req_ids)
                    data[batch_name]['times'].append(elapsed)
                    data[batch_name]['delay (s)'] += (len(active_ops) - len(req_ids)) * elapsed
                for batch_name in data:
                    data[batch_name]['#batch'] = len(data[batch_name]['times'])
                    data[batch_name]['per_batch_time (ms)'] = np.mean(data[batch_name]['times']) * 1000
                    data[batch_name]['per_batch_std (std)'] = np.std(data[batch_name]['times'])
                    data[batch_name]['time (s)'] = np.sum(data[batch_name]['times'])
                    data[batch_name].pop('times')
                    data[batch_name]['per_batch_est_time (ms)'] = np.mean(data[batch_name]['estimated_times'])
                    data[batch_name]['per_batch_est_time_std'] = np.std(data[batch_name]['estimated_times'])
                    data[batch_name].pop('estimated_times')
                    
                df = pd.DataFrame.from_dict(data, orient = 'index')
                df['time %'] = df['time (s)'] / df['time (s)'].sum()
                # df['delay %'] = df['delay (s)'] / df['delay (s)'].sum()
                df.sort_values(by='time %', ascending=False, inplace=True)

                req_latencies = [round((req_end_times[req_id] - req_start_times[req_id]), 2) for req_id in req_times]

                dict_status['req_latency'] = np.mean(req_latencies)
                dict_status['req_latency_std'] = np.std(req_latencies)

                req_occupancies = [round(req_times[req_id] / (req_end_times[req_id] - req_start_times[req_id]), 2) for req_id in req_times]
                dict_status['req_occupancy'] = np.mean(req_occupancies)
                dict_status['req_occupancy_std'] = np.std(req_occupancies)
                dict_status['scheduler_preceeds'] = np.mean(scheduler_preceeds_times)
                dict_status['scheduler_preceeds_std'] = np.std(scheduler_preceeds_times)
                dict_status['scheduler_behind'] = np.mean(scheduler_behind_times)
                dict_status['scheduler_behind_std'] = np.std(scheduler_behind_times)

                from sklearn.metrics import r2_score
                score = r2_score(real_times[10:], self.estimated_times[10:])
                print('Cost Model r2:', score)
                
                from sklearn.linear_model import LinearRegression 
                model = LinearRegression()
                estimated_np = np.array(self.estimated_times).reshape(-1,1)
                model.fit(estimated_np, real_times)
                r2_after_fit = r2_score(model.predict(estimated_np), real_times)
                print(f'After fitted r2: {r2_after_fit} (real = est * {model.coef_} + {model.intercept_})')
                dict_status['r2'] = score
                dict_status['r2 after fitted'] = r2_after_fit
                dict_status['k'] = float(model.coef_[0])
                dict_status['b'] = float(model.intercept_)
                
                fig, ax = plt.subplots()
                ax.scatter(real_times, self.estimated_times)
                import random
                for _ in range(int(len(profile) ** 0.8)):
                    idx = random.randint(0, len(profile) - 1)
                    ax.annotate(job_names[idx], (real_times[idx], self.estimated_times[idx]), textcoords="offset points", xytext=(0,10), ha='center')
                # Determine the limit range based on data
                limit_range = (min(min(real_times), min(self.estimated_times)), max(max(real_times), max(self.estimated_times)))

                # Set the same limits for x and y axes
                ax.set_xlim(limit_range)
                ax.set_ylim(limit_range)
                # Set the same scale for x and y axes
                ax.set_aspect('equal', adjustable='box')
                line = np.linspace(limit_range[0], limit_range[1], 100)
                # Draw the line x=y
                ax.plot(line, line, color='green', linestyle='--', linewidth=1, label='x=y')
                ax.set_xlabel('real_times')
                ax.set_ylabel('estimated_time')
                ax.set_title(f'#batch {len(profile)}, r2 {score}')
                ax.grid(True)
                fig_path = os.path.join(self.report_dir, f'cost_model-{self.round}.png')
                print(f'visualization of cost model saved to {fig_path}')
                fig.savefig(fig_path)
                rets[f'Engine {i}']['profile'] = df.to_dict()
            print('status', dict_status)
        return rets

    def report_ft(self):
        print('-------------SCHEDULED--------------')
        for i, log in enumerate(self.logs):
            print(i, ':', log)
        print('-------------SCHEDULED--------------')
        unscheduled_count = {}
        for node in self.dependency_graph.nodes: 
            op = self.dependency_graph.nodes[node]['op']
            unscheduled_count[op.op_code] = unscheduled_count.get(op.op_code, 0) + 1 
        print('------------UNSCHEDULED------------')
        for op_code, num in unscheduled_count.items():
            print(f'{op_code}: {num}')
        print('------------UNSCHEDULED------------')

    def report(self):
        # self.report_ft()
        rets = self.report_bk()
        req_to_ids = {}
        schedule_delays = {}
        for i, ((name, req_ids), schedule_time) in enumerate(zip(self.batch_logs, self.schedule_times)):
            for req_id, arrive_time in req_ids:
                schedule_delays[req_id] = schedule_delays.get(req_id, schedule_time - arrive_time)
                if req_id not in req_to_ids:
                    req_to_ids[req_id] = []
                req_to_ids[req_id].append(i)
        proportions = [len(x) / (max(x) - min(x)) for x in req_to_ids.values()]
        schedule_delays = list(schedule_delays.values())
        schedule_delays_mean = np.mean(schedule_delays)
        schedule_delays_std = np.std(schedule_delays)
        proportions_mean = np.mean(proportions)
        proportions_std = np.std(proportions)
        # get_mean = np.mean(self.get_times)
        # get_std = np.std(self.get_times)
        proportions = {
            'batch occupation': proportions_mean,
            'batch occupation (std)': proportions_std,
            'schedule delay': schedule_delays_mean,
            'schedule delay (std)': schedule_delays_std,
            # 'get delay': get_mean,
            # 'get delay (std)': get_std
        }
        rets.update(proportions)
        print(f'Batch Occupation: {proportions_mean} +- {proportions_std}')
        print(f'Schedule Delay: {schedule_delays_mean} +- {schedule_delays_std}')
        # print(f'get Delay: {get_mean} +- {get_std}')
        return rets

    def set_config(self,
        window_size: float,
        sch_budget: float, 
        profile: bool):
        if window_size is not None:
            if window_size == 0:
                self.enable_adaws = True
            else: 
                self.window_size = window_size
                self.enable_adaws = False
        if sch_budget is not None: 
            self.sch_tot_budget = sch_budget
        if profile is not None:
            ray.get([engine.execute_method.remote('set_profile', profile) for engine in self.engines])

    def visualize(self, file_path):
        with open(file_path, 'w') as file:
            file.write('batches:\n')
            for i, (name, op_ids) in enumerate(self.batch_logs):
                file.write(f'{i} {name}'.ljust(30))
                op_ids = sorted(op_ids, key=lambda x: x[0])
                last_op_id = -1
                for op_id, _ in op_ids:
                    file.write('.' * (op_id - last_op_id - 1))
                    file.write('*')
                    last_op_id = op_id
                file.write('\n')
        print(f'visualization saved to {file_path}')

    async def reset(self):
        self.logs = []
        self.estimated_times = []
        self.schedule_times = []
        self.batch_logs = []
        for device_id in range(len(self.engines)):
            await self.execute_method(device_id, 'reset')

    def communicate(self, src: int, dst: int, refs: List[ObjectRef], keep_olds: List[bool]):
        self.logs.append(f'COMM {src} -> {dst}, BS: {len(refs)},'
                        f'KVCaches: {sum(map(lambda x: isinstance(x, KVCacheRef), refs))},'
                        f'Tensors: {sum(map(lambda x: isinstance(x, TensorRef), refs))}')
        device_refs = [ref.device_ref for ref in refs]
        self.execute_method(src, 'send_batched', dst = dst, refs = device_refs, keep_olds = keep_olds)
        self.execute_method(dst, 'recv_batched', src = src, refs = device_refs)
    
    async def communicate_async(self, src: int, dst: int, refs: List[ObjectRef], keep_olds: List[bool]):
        self.logs.append(f'COMM {src} -> {dst}, BS: {len(refs)},'
                        f'KVCaches: {sum(map(lambda x: isinstance(x, KVCacheRef), refs))},'
                        f'Tensors: {sum(map(lambda x: isinstance(x, TensorRef), refs))}')
        device_refs = [ref.device_ref for ref in refs]
        jobs = [self.execute_method(src, 'send_batched', dst = dst, refs = device_refs, keep_olds = keep_olds),
        self.execute_method(dst, 'recv_batched', src = src, refs = device_refs)]
        await asyncio.wait(jobs)

    def execute_method(self, engine_id:int, method:str, *args, **kwargs):
        job = self.engines[engine_id].execute_method_with_id.remote(self.job_ids[engine_id], method, *args, **kwargs)
        self.remote_jobs.append(job)
        self.job_ids[engine_id] += 1
        return job
    
    async def sync(self):
        print('doing sync...', len(self.dependency_graph.nodes))
        while self.dependency_graph.number_of_nodes():
            await asyncio.sleep(self.timeout)
        print('sync waiting...')
        ready, self.remote_jobs = ray.wait(self.remote_jobs, num_returns = len(self.remote_jobs)) 

        # Process completed jobs
        for obj_ref in ready:
            try:
                result = ray.get(obj_ref)  # This will raise an exception if the job failed
            except Exception as e:
                print(f"Job failed with exception: {e}")
        print('finished sync...')

    async def schedule(self):
        window_size = self.window_size 
        communicator = Communicator()
        # frontiers: Dict[OperationID, Node] = {}
        
        while True:
            '''
            The graph can be changed during the waiting. 
            '''
            await asyncio.sleep(window_size) # The executor do periodical schedule

            '''
            The Schedule body. The dependency graph will not change. 
            '''
            frontiers: Dict[OperationID, Node] = {
                n: self.dependency_graph.nodes[n]['op']
                for n, d in self.dependency_graph.in_degree()
                if d == 0
            }
            
            for op in frontiers.values(): 
                # do placement and communication 
                if not op.placed: # the operation can 
                    op.place(communicator, self.cluster_status)
                    assert op.device_group is not None
                    for ref in op.output_refs:
                        ref.placement.device_groups[0] = op.device_group 
            if self.debug:
                await communicator.batched_commit_async(self.communicate_async)
            else: 
                communicator.batched_commit(self.communicate)

            loads = [0 for _ in range(len(self.engines))]
            
            n_batch = 0
            while len(frontiers) and max(loads) < self.sch_tot_budget:
                n_batch += 1
                batch_loads = [0 for _ in range(len(self.engines))]
                # 1. We classify all operations into <device_group, op_tag, []>
                device_batches: Dict[DeviceGroup, Dict[Tuple, Dict[OperationID, Node]]] = {}
                for node_id, op in frontiers.items(): 
                    if op.device_group not in device_batches:
                        device_batches[op.device_group] = {}
                    if op.op_tag not in device_batches[op.device_group]:
                        device_batches[op.device_group][op.op_tag] = {}
                    device_batches[op.device_group][op.op_tag][node_id] = op

                # 2. For every device, we select one batch
                new_frontiers: Dict[OperationID, Node] = {}
                for device_group, batches in device_batches.items():
                    best_key, batch = self._select_batch(batches)
                    op_code = next(iter(batch.values())).op_code
                    self.batch_logs.append(
                        (best_key[-1] \
                        #   + ('-p' if best_key[3] else '-d'))\
                         if op_code == OpCode.CausalLMInference else str(op_code), 
                         [(op.request_meta.request_id, op.request_meta.arrive_time) for op in batch.values()]))
                    # print(f'schedule batch {op_code.name} {len(batch)}')
                    assert len(batch)
                    operator = OP_CLASSES[op_code] 

                    if op_code == OpCode.DELETE:
                        per_engine_refs = [[] for _ in range(len(self.engines))]
                        all_inputs = sum((op.input_refs for op in batch.values()), start = [])
                        for ref in all_inputs:
                            if not (len(ref.placement.device_groups) <= 1):
                                print('ref device_group > 1', ref)
                                raise RuntimeError('> 1 device_groups')
                            if ref.status == ObjectStatus.SCHEDULED:
                                for device_group in ref.placement.device_groups:
                                    assert len(device_group.devices) == 1
                                    for device in device_group.devices:
                                        per_engine_refs[device].append(ref.device_ref)
                        for i, refs in enumerate(per_engine_refs):
                            # if len(refs):
                            if self.debug: 
                                await self.execute_method(i, 'execute', op_code, refs)
                            else: self.execute_method(i, 'execute', op_code, refs)
                            load = operator.estimate_load(refs)
                            batch_loads[i] += load
                    else:
                        for args, device_id in zip(zip(*[op.args for op in batch.values()]), device_group):
                            if self.debug: 
                                # get_start = time.perf_counter()
                                await self.execute_method(device_id, 'execute', op_code, args)
                                # self.get_times.append(time.perf_counter() - get_start)
                            else: self.execute_method(device_id, 'execute', op_code, args)
                
                        load = operator.estimate_load([op.load_meta for op in batch.values()])
                        for device_id in device_group: 
                            batch_loads[device_id] += load
                    
                    self.schedule_times.append(time.perf_counter())
                    self.estimated_times.append(load)
                    
                    #update the dependency graph 
                    # Step 2 and Step 3: Remove nodes and check their successors
                    for node in batch:
                        successors = list(self.dependency_graph.successors(node))
                        self.dependency_graph.remove_node(node)
                        for successor in successors:
                            if self.dependency_graph.in_degree(successor) == 0:
                                assert successor not in frontiers
                                new_frontiers[successor] = self.dependency_graph.nodes[successor]['op']
                        op = frontiers.pop(node)
                        for ref in op.output_refs:
                            ref.status = ObjectStatus.SCHEDULED
                            # Update the views if necessary
                            for sub_ref in ref.sub_refs:
                                sub_ref.status = ObjectStatus.SCHEDULED
                    
                # 3. We place and communicate the ops on the new frontier,
                # which is a super set of batchable operations
                for op in new_frontiers.values(): 
                    # do placement and communication 
                    op.place(communicator, self.cluster_status)
                    assert op.device_group is not None
                    for ref in op.output_refs:
                        ref.placement.device_groups[0] = op.device_group 
                if self.debug:
                    await communicator.batched_commit_async(self.communicate_async)
                else: 
                    communicator.batched_commit(self.communicate)
                
                frontiers.update(new_frontiers)

                # 4. update the load
                for device_id, load in enumerate(batch_loads):
                    self.cluster_status.add_load(device_id, load)
                for i, load in enumerate(batch_loads):
                    loads[i] += load
            

            if self.enable_adaws:
                # wait until one of the devices is idle
                window_size = min(loads) / 1000
            self.round += 1

            # We need to fetch the result and distribute them back to the requests
            for queue in self.output_queues:
                queue_size = queue.size()
                if queue_size > 0:
                    results = queue.get_nowait_batch(queue_size)
                    for request_id, value, got_time in results:
                        if request_id in self.request_contexts:
                            # self.request_contexts[request_id].request_output_queue.put_nowait((value, got_time, time.perf_counter()))
                            self.request_contexts[request_id].request_output_queue.put_nowait(value)


    # OP_PRIORITIES = {
    #     OpCode.CausalLMInference:0,
    #     OpCode.DECODE:3,
    #     OpCode.ENCODE:2,
    #     OpCode.CONCAT:5,
    #     OpCode.DELETE:4,
    #     OpCode.VERIFY:1,
    #     OpCode.GET:6,
    # }

    # @classmethod
    # def is_greator_delay_slow_op(cls, k1: Tuple, k2: Tuple):
    #     op_code1 = k1[0]
    #     op_code2 = k2[0]
    #     if cls.OP_PRIORITIES[op_code1] != cls.OP_PRIORITIES[op_code2]:
    #         return cls.OP_PRIORITIES[op_code1] > cls.OP_PRIORITIES[op_code2]
    #     if op_code1 == OpCode.CausalLMInference:
    #         if k1[3] != k2[3]:
    #             return k1[3] > k2[3] # We prioritize the prefill
    #         return k1[2] < k2[2] # We prioritize smaller model
    #     return False
    
    @classmethod
    def _select_batch(
            cls, 
            batches: Dict[Tuple, Dict[OperationID, Node]])\
            -> Tuple[OpCode, Dict[OperationID, Node]]:
        best_key = max(batches.keys())
        return best_key, batches[best_key]