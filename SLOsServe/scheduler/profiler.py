from dataclasses import dataclass, asdict, field, is_dataclass, fields
import numpy as np
from typing import List
import ray
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import random
import json
import os

from ..engine_wrapper import EngineWrapper
from .struct import (BatchSchedule,
        ReqBatchSchedule,
        Schedule,
        Request,
        ExecutionResult,
        RequestInitMsg,
        RequestInitResult)
from .utils import from_json

from .struct import (
    Problem
)

@dataclass
class TestCase:
    past_seq_lens: List[int]
    sch: BatchSchedule
    n_tokens: int = field(init = False)
    
    def __post_init__(self):
        self.n_tokens = 0
        for req_sch in self.sch.reqs:
            self.n_tokens += self.past_seq_lens[req_sch.id] + req_sch.n + 16
        
@dataclass
class ProfileDatapoint: 
    bs: int    
    e2e_time: float
    draft_time: float
    verifier_time: float
    sch: BatchSchedule
    compute_time: float = 0.




def save_profile_data(data: List[ProfileDatapoint], file_path: str) -> None:
    """
    Save a list of ProfileDatapoint objects to a JSON file.

    Args:
        data (List[ProfileDatapoint]): List of ProfileDatapoint objects.
        file_path (str): File path to save the data.
    """
    try:
        # Convert list of dataclass objects to list of dictionaries
        data_dicts = [asdict(datapoint) for datapoint in data]
        # Write to JSON file
        with open(file_path, 'w') as f:
            json.dump(data_dicts, f, indent=4)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

def load_profile_data(file_path: str) -> List[ProfileDatapoint]:
    """
    Load a list of ProfileDatapoint objects from a JSON file.

    Args:
        file_path (str): File path to load the data from.

    Returns:
        List[ProfileDatapoint]: List of loaded ProfileDatapoint objects.
    """
    # try:
        # Read JSON file
    with open(file_path, 'r') as f:
        data_dicts = json.load(f)
    # Convert list of dictionaries to list of dataclass objects
    data = [from_json(ProfileDatapoint, datapoint) for datapoint in data_dicts]
    print(f"Data successfully loaded from {file_path}")
    return data
    # except Exception as e:
    #     print(f"Error loading data: {e}")
    #     return []

@dataclass
class Profiler:
    engine: EngineWrapper | None = None
    working_dir: str = 'profile'
    
    def _exec_test_case(self, test_case: TestCase):
        ray.get(self.engine.execute_method.remote('reset'))
        init_msgs: List[RequestInitMsg] = []
        init_schs: List[ReqBatchSchedule] = []
        for req_sch in test_case.sch.reqs:
            if req_sch.is_prefill:
                init_msg = RequestInitMsg(req_sch.id, 
                                          None, 
                                          test_case.past_seq_lens[req_sch.id] + req_sch.n,
                                          1)
                
            else:
                init_msg = RequestInitMsg(req_sch.id, 
                                     None, 
                                     input_length = test_case.past_seq_lens[req_sch.id], 
                                     output_length = req_sch.n + 1)
            init_msgs.append(init_msg)
            init_sch = ReqBatchSchedule(req_sch.id, 
                                        is_prefill = True, 
                                        n = test_case.past_seq_lens[req_sch.id])
            init_schs.append(init_sch)
        ray.get(self.engine.execute_method.remote('init_requests', msgs = init_msgs))
        ray.get(self.engine.execute_method.remote('execute_batch', batch_sch = BatchSchedule(init_schs)))
        start = time.perf_counter()
        res: ExecutionResult = ray.get(self.engine.execute_method.remote('execute_batch', batch_sch = test_case.sch))
        elasped = time.perf_counter() - start
        return elasped, res.draft_time, res.verifier_time
    
    def _run_and_plot(self, test_cases: List[TestCase], label, ax) -> List[ProfileDatapoint]:
        data: List[ProfileDatapoint] = []
        for test_case in tqdm(test_cases, desc = 'profiling'):
            if test_case.n_tokens >= 3e4: continue
            elasped_time, draft_time, verifier_time = self._exec_test_case(test_case)
            data.append(ProfileDatapoint(test_case.sch.get_effective_bs(), elasped_time, draft_time, verifier_time, test_case.sch))
        ax.scatter([x.bs for x in data], [x.e2e_time for x in data], label = label)
        ax.scatter([x.bs for x in data], [x.draft_time for x in data], label = label + '-drafter')
        ax.scatter([x.bs for x in data], [x.verifier_time for x in data], label = label + '-verifier')
        
        return data
        
    def profile(self):
        '''
        1. the previous tokens;
        2. the current batch;
        '''
        # batch_sizes = list(range(64, 1024, 64))
        batch_sizes = [1, 2, 4] + [
            16 * i for i in range(1, 33)
        ]
        
        # from benchmark.scripts.structs import Dataset
        # ds = Dataset.load(f'dataset/{trace_name}.ds')
        
        # input_lengths = [req.prompt_len for req in ds.reqs]
        # output_lengths = [req.output_len for req in ds.reqs]
        
        # input_len_ave = np.mean(input_lengths)
        # output_len_ave = np.mean(output_lengths)
        
        prefill_test_cases: List[TestCase] = []
        spec_decode_sizes = [1,3,5]
        decode_test_casess: List[List[TestCase]] = [[] for _ in range(len(spec_decode_sizes))]
        
        # prefill_test_cases.append(TestCase([128], 
        #                             BatchSchedule([ReqBatchSchedule(0, True, 128)])))
        
        past_seq_len = 128
        for cur_seq_len in batch_sizes:
            prefill_test_cases.append(TestCase([past_seq_len], 
                                    BatchSchedule([ReqBatchSchedule(0, True, cur_seq_len)])))
            for i, spec_decode_size in enumerate(spec_decode_sizes):
                if (cur_seq_len // spec_decode_size) * (cur_seq_len + past_seq_len) > 30000: continue
                decode_test_casess[i].append(TestCase([past_seq_len for _ in range(cur_seq_len)], 
                    BatchSchedule([ReqBatchSchedule(i, False, spec_decode_size) \
                                   for i in range(cur_seq_len // spec_decode_size)])))
            
        fig, ax = plt.subplots()
        self._run_and_plot(prefill_test_cases, 'prefill', ax)
        for spec_decode_size, decode_test_cases in zip(spec_decode_sizes, decode_test_casess):
            self._run_and_plot(decode_test_cases, f'decode-{spec_decode_size}', ax)
        plt.legend()
        
        fig.savefig(f'profile.png')

    def profile_prob(self, prob: Problem, profile_path: str | None = None):
        if profile_path is None or not os.path.exists(profile_path): 
            profile_path = f'{self.working_dir}/{prob.request_trace}.json'

        if os.path.exists(profile_path):
            data = load_profile_data(profile_path)
            assert len(data)
            return data
        
        mean_i_len = np.mean([req.input_length for req in prob.reqs])
        mean_o_len = np.mean([req.output_length for req in prob.reqs])
        ratio = mean_i_len / (mean_i_len + mean_o_len)
        print(f'trace {prob.request_trace} mean_i_len: {mean_i_len} mean_o_len {mean_o_len}')
        
        batch_sizes = [
            16 * i for i in range(1, 33)
        ] + [640, 768, 896, 1024]
        ratio_lb = max(ratio - 0.2, 0)
        ratio_ub = min(ratio + 0.2, 1)
        test_cases: List[TestCase] = []
        for bs in batch_sizes: 
            for _ in range(5 if prob.spec_model is not None else 1):
                prompt_bs =  int(bs*(random.random() * (ratio_ub - ratio_lb) + ratio_lb)) 
                decode_bs = bs - prompt_bs
                req_schs: List[ReqBatchSchedule] = []
                past_seq_lens = [0]
                req_schs.append(ReqBatchSchedule(0, True, prompt_bs))
                n_dcd_token = 0
                idx = 1
                while n_dcd_token < decode_bs:
                    if prob.spec_model is not None:
                        choices = range(1, min(decode_bs - n_dcd_token, 6)+1)
                        cur_seq_len = random.choice(choices)
                    else: 
                        cur_seq_len = 1
                    n_dcd_token += cur_seq_len
                    req_schs.append(ReqBatchSchedule(idx, False, cur_seq_len))
                    idx += 1
                    past_seq_lens.append(int(mean_i_len))
                test_cases.append(TestCase(past_seq_lens, BatchSchedule(req_schs)))
        
        
        fig, ax = plt.subplots()
        data = self._run_and_plot(test_cases, 'trace', ax)
        plt.legend()
        fig_path = f'{self.working_dir}/{prob.request_trace}.png'
        fig.savefig(fig_path)
        print(f'Profiling saved to ', fig_path, profile_path)
        save_profile_data(data, profile_path)
        
        return data

    def profile_spec_decode(self, prob: Problem, prefix: str | None = None) -> List[int]:
        if prefix is not None and os.path.exists(f'{prefix}-spec-n_accs.json'):
            import json
            with open(f'{prefix}-spec-n_accs.json', 'r') as f:
                return json.load(f)
  
        all_reqs = random.sample(prob.reqs, 100)
        pbar = tqdm(total=len(all_reqs), desc = 'profile_spec')
        n_accs = []
        
        def run_batch(reqs: List[Request]):
            ray.get(self.engine.execute_method.remote('reset'))
            init_msgs = [RequestInitMsg(i, req.prompt, None, req.output_length) for i, req in enumerate(reqs)]
            generated_texts = [req.prompt for req in reqs]
            
            init_ress: List[RequestInitResult] = ray.get(
                self.engine.execute_method.remote('init_requests', msgs = init_msgs))
            
            prefill_sch = [ReqBatchSchedule(res.id, True, res.input_length) for res in init_ress]            
            decode_sch = [ReqBatchSchedule(res.id, False, 20) for res in init_ress]
        
            ray.get(
                self.engine.execute_method.remote('execute_batch', 
                            batch_sch = BatchSchedule(prefill_sch)))
            
            while len(decode_sch):
                res: ExecutionResult = ray.get(
                    self.engine.execute_method.remote('execute_batch', 
                            batch_sch = BatchSchedule(decode_sch)))
                
                n_accs.extend([req_res.n_generated for req_res in res.results])
                
                
                for req_res in res.results:
                    assert req_res.n_generated > 0
                    if req_res.is_finished:
                        idx = 0
                        while idx < len(decode_sch) and decode_sch[idx].id != req_res.id:
                            idx += 1
                        assert idx < len(decode_sch)
                        decode_sch.pop(idx)
                        pbar.update(1)
                    else: 
                        generated_texts[req_res.id] += req_res.generated_text
                 
        for i in range(0, len(all_reqs), 10):
            run_batch(all_reqs[i: i + 10])
        pbar.close()         
            
        if prefix is not None:
            import json 
            with open(f'{prefix}-spec-n_accs.json', 'w') as f:
                json.dump(n_accs, f)
        
        return n_accs

    def profile_sch(self, reqs: List[Request], sch: Schedule, prefix):
        scheduled_reqs = set()
        
        data: List[ProfileDatapoint] = []
        for batch in tqdm(sch.batches, desc = f'profile schedule {sch.name}'):
            init_msgs = []
            for req_sch in batch.reqs:
                if req_sch.id not in scheduled_reqs:
                    req = reqs[req_sch.id]
                    init_msgs.append(
                        RequestInitMsg(
                            req_sch.id, 
                            None, 
                            req.input_length,
                            req.output_length))
                    scheduled_reqs.add(req_sch.id)
                    
            ray.get(self.engine.execute_method.remote('init_requests', msgs = init_msgs))
            
            if batch.get_effective_bs() > 0:
                start = time.perf_counter()
                res: ExecutionResult = ray.get(self.engine.execute_method.remote('execute_batch', batch_sch = batch))
                elapsed = time.perf_counter() - start
                data.append(ProfileDatapoint(batch.get_effective_bs(), 
                                             elapsed, 
                                             res.draft_time,
                                             res.verifier_time))
            
        fig, ax = plt.subplots()
        ax.scatter([x.bs for x in data], [x.draft_time for x in data], label='draft time')
        ax.scatter([x.bs for x in data], [x.verifier_time for x in data], label='verify time')
        ax.scatter([x.bs for x in data], [x.e2e_time for x in data], label='elasped time')

        fig.savefig(f'{self.working_dir}/{prefix}.png')

        save_profile_data(data, f'{self.working_dir}/{prefix}.json')
        print(f'sch profile saved to {self.working_dir}/{prefix}.json')
        
        return data