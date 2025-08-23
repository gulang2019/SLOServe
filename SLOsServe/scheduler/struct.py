from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Any
import numpy as np
import enum 
import math
import os 
import json
import random
import tqdm

from .batch_timer import BatchTimer
from .utils import poisson_geo_process, from_json
from .spec_decode import SpecDecode
from benchmark.scripts.structs import Dataset

'''
https://openai.com/api/pricing/
$1.25 / 1M cached** input tokens
$10.00 / 1M output tokens
'''

EOS='EOS'
REJ='REJ'

EASY_SLOS = {
    'HumanDcd': '5.0:0.1:0.0:0.0:1.0',  
    'FastDcd': '5.0:0.05:0.0:0.0:1.0',
    'FastPF': '3.0:0.1:0.0:0.0:1.0',
    'FastThink': '3.0:0.1|0.05:0.0:0.0:1.0',
    'FastTool': '3.0:0.05:0.0:0.0:1.0#3.0:0.1:0.0:0.0:1.0',
    'HumanDcd-S': '5.0:0.2:0.0:0.0:1.0',
    'FastDcd-S': '5.0:0.10:0.0:0.0:1.0',
    'FastPF-S': '3.0:0.2:0.0:0.0:1.0',
    'FastThink-S': '3.0:0.2|0.10:0.0:0.0:1.0',
    'FastTool-S': '3.0:0.10:0.0:0.0:1.0#3.0:0.2:0.0:0.0:1.0',
    'Loose': '5.0:0.05:0.0:0.0:1.0'
}

# EASY_SLOS_13B = {
    
# }

EASY_TRACES = {
    'Coding': 'splitwise_code-humaneval-1786.0:300.0-1.0',
    'Test': 'splitwise_conv-sharegpt-900.0:20.0-1.0',
    'Chatting': 'splitwise_conv-sharegpt-900.0:300.0-1.0',
    'ChattingShort': 'splitwise_conv-sharegpt-1050.0:150.0-1.0',
    'Chatting_Half': 'splitwise_conv-sharegpt-900.0:300.0-0.5',
    'Summary': 'arxiv_summary-sharegpt-0.0:300.0-1.0',
    'Arxiv': 'arxiv_summary-sharegpt-0:{LOAD}-1.0',
    'Reasoning': 'reasoning-reasoning-0:100.0-1.0',
    'ReasoningTest': 'reasoning-reasoning-0:10.0-1.0',
    'Tools': 'tools-tools-1936.0:100.0-1.0',
    'ToolsTest': 'tools-tools-0:10.0-1.0'
}

@dataclass
class SLA:
    name: str = "SLA"
    is_best_effort: bool = False
    ttft_slown_rate: float | None = None 
    tpot_slown_rate: float | None = None 
    ttft_per_token:  float | None = None
    tpot_per_token:  float | None = None
    decode_check_freq: int = 10 # check conformence every X time slots
    ttft_per_token_profit: float = 1
    tpot_per_token_profit: float = 8
    base_profit: float = 0
    tpot_per_token_thinking: float | None = 0
    
    def __post_init__(self):
        if self.tpot_per_token_thinking is None: 
            self.tpot_per_token_thinking = self.tpot_per_token
        if not self.is_best_effort:
            assert self.ttft_per_token is not None\
                or self.ttft_slown_rate is not None
            assert self.tpot_per_token is not None\
                or self.tpot_slown_rate is not None
        self._ttft_slown_rate = self.ttft_slown_rate
        self._ttft_per_token = self.ttft_per_token
        self._tpot_per_token = self.tpot_per_token
        self._tpot_slown_rate = self.tpot_slown_rate 

    
    @staticmethod
    def from_name(name: str):
        assert name in EASY_SLOS
        s = EASY_SLOS[name]
        return [
            SLA.from_str(_, name) for _ in s.split('#')
        ]
    
    @staticmethod
    def from_str(s: str, name: str | None = None):
        if not name: name = s
        if s == 'best_effort':
            return SLA(name, True)
        
        ttft_slow_rate, tpot_per_token, ttft_per_token_profit, tpot_per_token_profit, base_profit  = s.split(':')
        tpots = tpot_per_token.split('|')
        if len(tpots) == 1:
            tpot_per_token = float(tpots[0])
            tpot_per_token_thinking = None
        else: 
            assert(len(tpots)) == 2
            tpot_per_token = float(tpots[0])
            tpot_per_token_thinking = float(tpots[1])
        return SLA(
            name = name,
            ttft_slown_rate = float(ttft_slow_rate),
            tpot_per_token=float(tpot_per_token),
            ttft_per_token_profit = float(ttft_per_token_profit),
            tpot_per_token_profit = float(tpot_per_token_profit),
            base_profit=float(base_profit),
            tpot_per_token_thinking=tpot_per_token_thinking
        )

    def __str__(self):
        return self.name

    def easy_str(self):
        if self.is_best_effort: return 'best_effort'
        return f'TTFT-{self.ttft_slown_rate}-TPOT-{self.tpot_per_token}'

    def update(self, *, ttft_scale: float = 1.0, tpot_scale: float = 1.0):
        assert self._ttft_slown_rate is not None
        self.ttft_slow_rate = self._ttft_slown_rate * ttft_scale
        
        assert self._tpot_per_token is not None
        self.tpot_per_token = self._tpot_per_token * tpot_scale
        
        
    
@dataclass
class Trace:
    name: str
    datasets: List[str]
    prompts: List[str]
    regions: List[Tuple[float, float]]
    slas: List[List[SLA]]
    stretches: List[float]
    
    def to_str(self) -> str:
        """
        Serialize the Trace object into a string.
        """
        return self.name


    @staticmethod
    def from_str(name: str | None) -> 'Trace':
        """
        Deserialize a string into a Trace object.
        """
        if not name:
            return Trace(
                "EmptyTrace",
                [], [], [], [], []
            )
        
        datasets = []
        prompts = []
        regions = []
        slas = []
        stretches = []
        
        for x in name.split(','):
            ds_name, sla_name = x.split(':')
            assert ds_name in EASY_TRACES
            # assert sla_name in EASY_SLOS
            ds, prompt, region, stretch = EASY_TRACES[ds_name].split('-')
            # sla = EASY_SLOS[sla_name]
            start, length = region.split(':')
            datasets.append(ds)
            prompts.append(prompt)
            regions.append((float(start), float(length))) 
            slas.append(SLA.from_name(sla_name))
            stretches.append(float(stretch))
            
        return Trace(
            name,
            datasets,
            prompts, 
            regions, 
            slas,
            stretches
        )
    
    
    def sample(self, 
               max_seq_len: int,
               batch_timer: BatchTimer,
               block_size: int,
               model: str,
               update_prompt = True):
        from benchmark.scripts.structs import Dataset, TestRequest
        rets = []
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)
        for ds_tag, sla, prompt, (s, length), stretch in zip(self.datasets, self.slas, self.prompts, self.regions, self.stretches):
            ds = Dataset.load(f'dataset/{ds_tag}.ds')
            reqs = []
            loads = []
            t = 0

            if not ds.has_arrive_time:
                print(f'WARNING: Use poisson for {ds_tag}')
                from .utils import poisson_process 
                for req, arrive_time in zip(ds.reqs, poisson_process(4, len(ds.reqs))):
                    req.arrive_time = arrive_time

            raw_reqs: List[TestRequest] = []
            for raw_req in ds.reqs:
                assert raw_req.arrive_time is not None
                if raw_req.arrive_time < s:
                    continue
                if raw_req.arrive_time > s + length * stretch: 
                    break
                raw_reqs.append(raw_req)

            if ds.has_prompt:
                all_raw_reqs: List[TestRequest] = []
                for req in raw_reqs:
                    all_raw_reqs.append(req)
                    while req.follow_up_req is not None:
                        req = req.follow_up_req[1]
                        all_raw_reqs.append(req)
                prompts: List[str] = [req.prompt for req in all_raw_reqs]
                prompt_lens = [len(x) for x in tokenizer(prompts).input_ids]
                # print('prompt lens', [len(_) for _ in prompts])
                # print('prompt_lens', prompt_lens)

                for req, prompt_len in zip(all_raw_reqs, prompt_lens):
                    req.prompt_len = prompt_len

            print('max_seq_len', max_seq_len)

            n_dropped = 0
            for raw_req in raw_reqs:
                # In case this request is tooooo long, we randomly replace this with another request
                # while req.prompt_len + req.output_len > max_seq_len: 
                #     _req = random.choice(ds.reqs)
                #     req.prompt_len = _req.prompt_len 
                #     req.output_len = _req.output_len
                assert raw_req.arrive_time
                if raw_req.prompt_len + raw_req.output_len > max_seq_len:
                    print(f'Drop prompt:{raw_req.prompt_len}, output: {raw_req.output_len}')
                    n_dropped += 1
                    continue

                    
                req = Request(
                    raw_req.prompt_len, 
                    raw_req.output_len,
                    (raw_req.arrive_time -s) / stretch,
                    sla[0],
                    batch_timer,
                    prompt = raw_req.prompt , 
                    generation_start = raw_req.generation_start
                )

                # if req.prompt:
                #     assert req.input_length == len(tokenizer(req.prompt))

                reqs.append(req)
                
                while raw_req.arrive_time > t:
                    loads.append(0)
                    t += 1
                loads[-1] += 1
                
                while raw_req.follow_up_req is not None:
                    next_req = raw_req.follow_up_req[1]
                    if next_req.prompt_len + next_req.output_len > max_seq_len:
                        break
                    req.follow_up_req = (
                        raw_req.follow_up_req[0],
                        Request(
                            next_req.prompt_len, 
                            next_req.output_len,
                            -1.0,
                            sla[0] if next_req.follow_up_req else sla[-1], # for the last request, we use the second SLO
                            batch_timer,
                            prompt = next_req.prompt,
                            generation_start = next_req.generation_start
                        )
                    )
                    req = req.follow_up_req[1]
                    raw_req = next_req
                
            # else: 
            #     s = int(s)
            #     length = int(length)
            #     assert sla.is_best_effort
            #     for i in range(s, s + length):
            #         req = ds.reqs[i % len(ds.reqs)]
            #         # while req.prompt_len + req.output_len > max_seq_len: 
            #         #     _req = random.choice(ds.reqs)
            #         #     req.prompt_len = _req.prompt_len 
            #         #     req.output_len = _req.output_len
            #         reqs.append(Request(
            #             req.prompt_len, 
            #             req.output_len,
            #             0.0,
            #             sla,
            #             batch_timer,
            #             prompt = req.prompt,
            #             generation_start=req.generation_start
            #         ))
            
            # print(loads)
            
            if prompt != ds_tag and update_prompt:
                prompt_ds = Dataset.load(f'dataset/{prompt}.ds')
                assert prompt_ds.has_prompt
                prompts = [req.prompt for req in prompt_ds.reqs]
                if len(prompts) < len(reqs):
                    prompts = sum([prompts for _ in range(int(math.ceil(len(reqs) / len(prompts))))], [])
                
                prompts = random.sample(prompts, k=len(reqs))
                from transformers import AutoTokenizer 
                tokenizer = AutoTokenizer.from_pretrained(model)
                input_idss = tokenizer(prompts).input_ids
                for req, input_ids  in tqdm.tqdm(zip(reqs, input_idss), desc = 'updating prompt', total = len(reqs)):
                    if len(input_ids) < req.input_length:
                        input_ids = sum([input_ids for _ in range(int(math.ceil(
                            req.input_length / len(input_ids))))], [])
                    input_ids = input_ids[-req.input_length+1:]
                    req.prompt = tokenizer.decode(input_ids)
                    input_length = len(tokenizer(req.prompt).input_ids)

                    # Ensure `req.input_length` matches the original length of `input_ids`
                    while req.input_length < input_length:
                        input_ids = input_ids[1:]
                        req.prompt = tokenizer.decode(input_ids)
                        input_length = len(tokenizer(req.prompt).input_ids)
                    req.input_length = input_length
                    assert req.input_length + req.output_length <= max_seq_len
            
            input_lengths = [req.input_length for req in reqs]
            output_lengths = [req.output_length for req in reqs]
            print(f'Load {len(reqs)} from {ds_tag} using {prompt} as promt.')
            print(f'#Dropped: {n_dropped}')
            print(f'input_length: {np.mean(input_lengths)} +- {np.std(input_lengths)}')
            print(f'output_length: {np.mean(output_lengths)} +- {np.std(output_lengths)}')
            
            rets.extend(reqs)
            
        rets = sorted(rets, key = lambda x: x.arrive_time)
        
        print(f'load {len(rets)} data from ', self)
        
        return rets

    def easy_str(self) -> str:
        return '\n'.join(f'{ds}-{prompt}-{start}:{end}-{stretch}-{[_.easy_str() for _ in sla]}' for ds, prompt, (start, end), sla, stretch in zip(self.datasets, self.prompts, self.regions, self.slas, self.stretches))

@dataclass
class ParaConfig:
    dp: int
    tp: int
    pp: int
    world_size: int = field(init = False)
    
    def __post_init__(self):
        self.world_size = self.dp * self.tp * self.pp

    @staticmethod
    def from_str(s: str) -> 'ParaConfig':
        dp, tp, pp = [int(_) for _ in s.split('-')]
        return ParaConfig(
            dp, tp, pp
        )
    

@dataclass
class ArrivalPattern:
    ds: str
    start: int = 0
    n: int = 100
    
    def to_str(self)->str:
        return f'{self.ds}:{self.start}:{self.n}'
    
    @staticmethod
    def from_str(s: str):
        if s == 'synthetic':
            return ArrivalPattern(s)
        ds, start, n = s.split(':')
        start = int(start)
        n = int(n)
        return ArrivalPattern(ds, start, n)
    
    def sample(self, 
               req_rate: float | None = None, 
               n_req_at_once: int | None = None)-> Tuple[float, List[float]]:
        if self.ds == 'synthetic':
            if req_rate is None: 
                req_rate = 5
            if n_req_at_once is None:
                n_req_at_once = 5
            return req_rate, poisson_geo_process(req_rate / n_req_at_once,
                        n_req_at_once, self.n)
        arrival_time_ds = Dataset.load(f'dataset/{self.ds}.ds')
        assert arrival_time_ds.has_arrive_time

        tot_time = arrival_time_ds.reqs[-1].arrive_time
        arrive_times = [tot_time * (i // len(arrival_time_ds.reqs)) \
                + arrival_time_ds.reqs[(i+self.start) % len(arrival_time_ds.reqs)].arrive_time
                for i in range(self.n)]
        cur_req_rate = len(arrive_times) / arrive_times[-1]
        
        if req_rate is not None:
            arrive_times = [t * cur_req_rate / req_rate for t in arrive_times]
        
        req_rate = round(len(arrive_times) / arrive_times[-1], 2)
        
        return req_rate, arrive_times
    
@dataclass
class Request:
    input_length: int
    output_length: int
    arrive_time: float
    sla: 'SLA'
    batch_timer: BatchTimer
    prompt: str | None = None
    num_blocks: int = field(init = False)
    generation_start: int = 0
    follow_up_req: Tuple[float, 'Request'] | None = None 
    
    def __post_init__(self):
        self.set_num_blocks(16)
    
    def set_num_blocks(self, block_size: int) -> 'Request':
        # Plus one for the drafter.
        self.num_blocks = int(math.ceil((self.input_length + self.output_length) / block_size))
        return self
    
    def profit(self):
        if self.sla.is_best_effort: return 0.
        return self.input_length * self.sla.ttft_per_token_profit +\
            self.output_length * self.sla.tpot_per_token_profit + self.sla.base_profit
    
    def ttft(self):
        if self.sla.is_best_effort:
            return -1
        if self.sla.ttft_per_token is None: 
            assert self.sla.ttft_slown_rate is not None 
            return self.batch_timer(self.input_length) * self.sla.ttft_slown_rate
        return self.sla.ttft_per_token * self.input_length
    
    def ttft_zeroload(self):
        if self.sla.is_best_effort: return -1
        assert self.sla.ttft_slown_rate is not None 
        return self.batch_timer(self.input_length)
        # return self.input_length * self.sla.ttft_per_token

    def tpot(self):
        if self.sla.is_best_effort: return -1
        if self.sla.tpot_per_token is None:
            assert self.sla.tpot_slown_rate is not None
            return self.batch_timer(1) * self.sla.tpot_slown_rate
        return self.sla.tpot_per_token
        
    def ttnt(self, n):
        if self.sla.is_best_effort: return -1
        assert self.sla.tpot_per_token_thinking is not None
        assert self.sla.tpot_per_token is not None 
        return (min(self.generation_start, n) - 1) * self.sla.tpot_per_token_thinking + \
            max(n - self.generation_start, 0) * self.sla.tpot_per_token 
        
    
class RequestState(enum.Enum):
    Arrived = enum.auto()
    Prefill = enum.auto()
    Decode = enum.auto()
    Declined = enum.auto()
    Finished = enum.auto()

PREFILL_VIOLATION = 0x1
DECODE_VIOLATION = 0x2

@dataclass 
class RequestInstanceBase:
    id: int
    req: Request
    
    arrive_time: float
    time_skew: float = 0.0
    
    state: RequestState = RequestState.Arrived
    admitted: bool = False
    best_effort: bool = False
    n_preempted: int = 0
    
    input_length: int = field(init = False)
    output_length: int = field(init = False)
    _n_prefill_tokens: int = 0
    _n_decode_tokens: int = 0
    _spec_history: List[Tuple[int, int]] = field(default_factory = list)
    _timestamps: List[Tuple[int, float]] = field(default_factory = list)
    _sla_fullfilled: int | None = None
    _sch_requirement: Any = None
    
    def preempt(self):
        assert not self.is_finished()
        self.state = RequestState.Arrived
        # Update this request as a new prefill request.
        self.input_length += self._n_decode_tokens
        self.output_length -= self._n_decode_tokens
        self._n_prefill_tokens = 0
        self._n_decode_tokens = 0
        self._spec_history = []
        self._timestamps = []
        self.best_effort = False
        self.n_preempted += 1
    
    def __post_init__(self):
        self.input_length = self.req.input_length
        self.output_length = self.req.output_length
    
    def _satisfy_sla(self):
        if not self.admitted: 
            return
        
        self._sla_fullfilled = 0x0
        if self._timestamps[0][1] > self.arrive_time + self.req.ttft():
            self._sla_fullfilled |= PREFILL_VIOLATION
        
        checkpoints = list(range(self.req.sla.decode_check_freq, 
                                 self.req.output_length, 
                                 self.req.sla.decode_check_freq))
        if len(checkpoints) == 0 or checkpoints[-1] != self.req.output_length:
            checkpoints.append(self.req.output_length)
        
        t0 = max(self.arrive_time + self.req.ttft(), self._timestamps[0][1])
        idx = 0
        for n_tokens, t in self._timestamps[1:]: 
            if idx < len(checkpoints) and n_tokens >= checkpoints[idx]:
                # TODO: The tolerance should be integrated to 
                # if (self.req.tpot() * (n_tokens - 1)) <= (t - t0) :
                if (self.req.ttnt(n_tokens)) <= (t - t0) :
                    self._sla_fullfilled |= DECODE_VIOLATION
                idx += 1
            
    def satisfy_sla(self) -> bool:
        if self.state not in (RequestState.Finished, RequestState.Declined):
            return False
        # assert self.state in (RequestState.Finished, RequestState.Declined)
        if self._sla_fullfilled is None:
            self._satisfy_sla()
        return self.admitted and self._sla_fullfilled == 0
        
    def get_latenesses(self):
        
        ttft_lateness = (self._timestamps[0][1] - self.arrive_time) / self.req.ttft_zeroload()
        
        tpot_latenesses = []
        checkpoints = list(range(self.req.sla.decode_check_freq, 
                                 self.req.output_length, 
                                 self.req.sla.decode_check_freq))
        
        if len(checkpoints) == 0 or checkpoints[-1] != self.req.output_length:
            checkpoints.append(self.req.output_length)
        
        t0 = max(self.arrive_time + self.req.ttft(), self._timestamps[0][1])
        idx = 0
        for n_tokens, t in self._timestamps[1:]: 
            if idx < len(checkpoints) and n_tokens >= checkpoints[idx]:
                tpot_latenesses.append((t - t0) / (self.req.ttnt(n_tokens)))
                idx += 1
        
        return ttft_lateness, tpot_latenesses
        
    def is_finished(self) -> bool: 
        return self.state == RequestState.Finished

    def is_prefill(self) -> bool:
        return self.state in (RequestState.Prefill, RequestState.Arrived)
    
    def is_decode(self) -> bool:
        return self.state == RequestState.Decode
    
    def decline(self):
        assert self.state == RequestState.Arrived
        self.state = RequestState.Declined

    def __eq__(self, other: 'RequestInstanceBase'):
        return self.id == other.id
    
    def ttft(self):
        return self.req.ttft()
    
    def tpot(self, current_time: float):
        # tpot = self.req.tpot()
        # slo_time = (self._n_decode_tokens) * self.req.tpot() + self.ttft() + self.arrive_time
        slo_time = self.req.ttnt(self._n_decode_tokens + 1) + self.arrive_time + self.ttft()
        # return max(tpot, slo_time - current_time)
        return slo_time - current_time
        
    def propogate_skew(self):
        self.arrive_time += self.time_skew
        for i, t in enumerate(self._timestamps):
            self._timestamps[i] = (t[0], t[1] + self.time_skew)
        self.time_skew = 0.
    
    
    
    def __repr__(self):
        return f'Request({self.id}, is_prefill: {self.is_prefill()}, is_decode: {self.is_decode()}, state: {self.state}, {self.input_length}/{self.output_length}, #prefill {self._n_prefill_tokens}, #decode {self._n_decode_tokens}, admitted: {self.admitted}, best_effort {self.best_effort})'
@dataclass 
class RequestOutput:
    id: int 
    input_length: int 
    output_length: int 
    arrive_time: float
    admitted: bool
    best_effort: bool
    slo_attainment: int | None
    ttft_slo: float
    tpot_slo: float
    timestamps: List[Tuple[int, float]]
    check_freq: int
    generation_start: int = 0
    profit: float = 0 
    # n_decode_scheduled: int
    
    def get_latenesses(self, ttft_tolerance = 0.0, tpot_tolerance = 0.0):
        ttft_lateness = (self.timestamps[0][1] - self.arrive_time) / self.ttft_slo / (1 + ttft_tolerance)
        
        tpot_latenesses = []
        checkpoints = list(range(self.check_freq, 
                                 self.output_length, 
                                 self.check_freq))
        
        if len(checkpoints) == 0 or checkpoints[-1] != self.output_length:
            checkpoints.append(self.output_length)
        
        t0 = max(self.arrive_time + self.ttft_slo, self.timestamps[0][1])
        idx = 0
        for n_tokens, t in self.timestamps[1:]: 
            if idx < len(checkpoints) and n_tokens >= checkpoints[idx]:
                tpot_latenesses.append((t - t0) / (self.tpot_slo * (n_tokens-1) * (1+tpot_tolerance)))
                idx += 1
        
        return ttft_lateness, tpot_latenesses
    
    def satisfy_sla(self, ttft_tolerance = 0.05, tpot_tolerance = 0.05) -> bool:
        if not self.admitted: 
            return False
        
        _sla_fullfilled = 0x0
        if self.timestamps[0][1] > (self.arrive_time + self.ttft_slo * (1 + ttft_tolerance)) :
            _sla_fullfilled |= PREFILL_VIOLATION
        
        checkpoints = list(range(20, 
                                 self.output_length, 
                                 20))
        if len(checkpoints) == 0 or checkpoints[-1] != self.output_length:
            checkpoints.append(self.output_length)
        
        t0 = max(self.arrive_time + self.ttft_slo, self.timestamps[0][1])
        idx = 0
        for n_tokens, t in self.timestamps[1:]: 
            if idx < len(checkpoints) and n_tokens >= checkpoints[idx]:
                # TODO: The tolerance should be integrated to 
                if (self.tpot_slo * (n_tokens - 1) * (1 + tpot_tolerance)) <= (t - t0) :
                    _sla_fullfilled |= DECODE_VIOLATION
                idx += 1
        return _sla_fullfilled == 0 
            
    # def satisfy_sla(self) -> bool:
    #     if self.state not in (RequestState.Finished, RequestState.Declined):
    #         return False
    #     # assert self.state in (RequestState.Finished, RequestState.Declined)
    #     if self._sla_fullfilled is None:
    #         self._satisfy_sla()
    #     return self.admitted and self._sla_fullfilled == 0
    
    @staticmethod 
    def from_instance(req: RequestInstanceBase) -> 'RequestOutput':
        req.propogate_skew()
        return RequestOutput(
            req.id,
            req.req.input_length,
            req.req.output_length,
            req.arrive_time,
            req.admitted,
            req.best_effort,
            req._sla_fullfilled,
            req.req.ttft(),
            req.req.tpot(),
            req._timestamps,
            req.req.sla.decode_check_freq,
            req.req.generation_start,
            req.req.profit()
        )
    
    @property
    def ddl(self):
        return self.arrive_time + self.ttft_slo + self.tpot_slo * self.output_length

@dataclass
class ReqBatchSchedule:
    id: int 
    is_prefill: bool 
    n: int
    def __repr__(self):
        return f'({self.id},{"P" if self.is_prefill else "D"},{self.n})'

@dataclass 
class RequestInitMsg:
    id: int
    prompt: str | None
    input_length: int | None
    output_length: int | None

@dataclass
class RevokeMsg:
    id: int
    input_length: int 
    output_length: int

@dataclass
class RequestInitResult:
    id: int 
    input_length: int

@dataclass 
class BatchSchedule:
    reqs: List[ReqBatchSchedule] = field(default_factory=list)
    next: int = 1
    remain_budget: int = 0
    batch_size: int | None = None
    decode_steps: int | None = None
    idx: int = 0
    start_time: float = 0
    estimated_time: float = 0.
    profiled_time: float = 0.
    profiled_time_ft: float = 0.
    sch_overhead: float = 0.
    
    def __post_init__(self):
        # assert len(set(x.id for x in self.reqs)) == len(self.reqs)
        if self.batch_size == None:
            self.batch_size = sum([req.n for req in self.reqs])
        self.decode_steps = max([req.n for req in self.reqs if not req.is_prefill], default=0)

    def add_req(self, req:ReqBatchSchedule):
        assert self.remain_budget >= req.n
        self.reqs.append(req)
        assert self.batch_size is not None and self.decode_steps is not None
        self.batch_size += req.n
        self.remain_budget -= req.n
        if not req.is_prefill:
            self.decode_steps = max(self.decode_steps, req.n)
    

    def get_effective_bs(self):
        return sum([req.n for req in self.reqs])

    def __repr__(self) -> str:
        return f'Batch(BS={self.get_effective_bs()},' + ','.join(str(req) for req in self.reqs) + f'next:{self.next})'
    
    # def get_remain_bs(self):

@dataclass
class SchedulerConfig:
    name: str
    sch_strategy: str
    sch_thresh_timeout: float
    sch_thresh_arrive: int 
    sch_thresh_finish: int
    enable_best_effort: bool
    

@dataclass
class Schedule:
    name: str
    batches: List[BatchSchedule]
    failure_rate: float
    overhead: float 
    tail_time: float
        
    # Stat
    reqs: List[RequestOutput] = field(init = False)
    profit: float = field(init = False)
    sla_satisfy_rate: float = field(init = False)
    ttft_tolerance: float = field(init = False, default = 0.05)
    tpot_tolerance: float = field(init = False, default = 0.05)
    
    ttft_mean: float  = field(init = False)
    ttft_p50: float  = field(init = False)
    ttft_p80: float  = field(init = False)
    ttft_p90: float = field(init = False)
    ttft_p99: float  = field(init = False)
    tpot_mean: float = field(init = False)
    tpot_p50: float = field(init = False)
    tpot_p80: float = field(init = False)
    tpot_p90: float = field(init = False)
    tpot_p99: float = field(init = False)
    
    alpha: float = field(init = False)
    # alpha_std: float = field(init = False)
    admission_rate: float = field(init = False)
    
    req_trace: str = field(init = False)
    req_rate: float = field(init = False)
    real_rate: float = field(init = False)
    model_tag: str = field(init = False)
    
    sub_schedules: List['Schedule'] = field(default_factory=list)

    @staticmethod
    def from_schedules(schedules: List['Schedule']):
        if len(schedules) == 1:
            return schedules[0]
        reqs = {}
        for sch in schedules:
            for req in sch.reqs:
                if req.id not in reqs:
                    reqs[req.id] = req
                else:
                    if req.admitted or req.best_effort:
                        reqs[req.id] = req
        
        return Schedule(
            schedules[0].name,
            reqs = sorted(list(reqs.values()), key = lambda req: req.id),
            real_rate = 0.,
            alpha = np.mean([sch.alpha for sch in schedules]),
            sub_schedules = schedules
        )

    def __init__(self,  name: str, 
                        reqs: List[RequestInstanceBase],
                        batches: List[BatchSchedule] = [],
                        failure_rate: float = 0.0,
                        overhead: float = 0.0,
                        tail_time: float = 0.0,
                        **kwargs):
        
        self.name = name
        self.batches = batches
        self.failure_rate = failure_rate
        self.overhead = overhead
        self.tail_time = tail_time
        
        if len(kwargs):
            self.reqs = reqs
            for k, v in kwargs.items():
                setattr(self, k, v)
            self._update()
            return
        self.sub_schedules = []
        if len(reqs):
            self.real_rate = len(reqs) / (reqs[-1].arrive_time - reqs[0].arrive_time + 1e-3)
        else: self.real_rate = 0.0
        
        self.reqs = [RequestOutput.from_instance(req) for req in reqs]
        self._update()
        
        admitted_reqs = [req for req in reqs if req.admitted]
        spec_histories = {}
        n_data = 0
        n_scheduled_tot = 0
        n_yield_tot = 0
        for n, n_yield in sum([req._spec_history for req in admitted_reqs], []):
            n_scheduled_tot += n 
            n_yield_tot += n_yield
            if n not in spec_histories: spec_histories[n] = []
            spec_histories[n].append(n_yield)
            n_data += 1

        est_alpha = 0
        for k, v in spec_histories.items():
            n_eq = sum((x == k) for x in v)
            est = (n_eq / len(v)) ** (1/k)
            est_alpha += est * len(v) / n_data
        
        self.alpha = est_alpha
    
    def draw_loads(self, file_path: str | None, ax = None):
        events = []
        best_effort_events = []

        # Parse events
        for req in self.reqs:
            if req.admitted:
                events.append((req.arrive_time, True))
                events.append((req.timestamps[-1][-1], False))
            elif req.best_effort:
                best_effort_events.append((req.arrive_time, True))
                best_effort_events.append((req.timestamps[-1][-1], False))
            else:
                print("UNEXPECTED", req)

        # Add standard label tag
        # _tag = "-standard" if label == "Ours" else ""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()

        def _draw_event(events, label, ax):   
            events = sorted(events, key = lambda x: x[0])
            load = 0
            loads = []
            times = []
            for event in events:
                if event[1]: load += 1
                else: load -= 1
                loads.append(load)
                times.append(event[0])
            t0 = times[0]
            times = [t - t0 for t in times]
            ax.plot(times, loads, label = label, linewidth=2)

        _draw_event(events, label="admitted", ax=ax)
        if best_effort_events:
            _draw_event(best_effort_events, f"Best Effort", ax=ax)
        ax.legend()
        if file_path is not None:
            fig.savefig(file_path)

    def draw_lateness(self,
                     file_path: str | None = None):
        if file_path is None: 
            return 
        latenesses: List[Tuple[float, List[float]]] = [req_instance.get_latenesses() 
                for req_instance in self.reqs if req_instance.admitted]
        ttft_latenesses = [lateness[0] for lateness in latenesses]
        tpot_latenesses = sum([lateness[1] for lateness in latenesses], [])

        import matplotlib.pyplot as plt
        fig, (ax0, ax1) = plt.subplots(2, tight_layout = True)
        ttft_latenesses = sorted(ttft_latenesses)
        ax0.plot(ttft_latenesses, [i/len(ttft_latenesses) for i in range(len(ttft_latenesses))])            
        ax0.set_title(f'ttft_lateness PDF max {round(max(ttft_latenesses, default=0), 2)}')
        ax0.set_xlabel('Lateness')
        ax0.set_ylabel('Density')

        # Compute and plot PDF for tpot_latenesses
        tpot_latenesses = sorted(tpot_latenesses)
        ax1.plot(tpot_latenesses, [i/len(tpot_latenesses) for i in range(len(tpot_latenesses))])            
        ax1.set_title(f'tpot_lateness PDF max {round(max(tpot_latenesses, default=0), 2)}')
        ax1.set_xlabel('Lateness')
        ax1.set_ylabel('Density')
        
        fig.savefig(file_path)
        print('lateness saved to', file_path)

    def draw_batches(self,
        file_path: str | None = None):
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1,2, tight_layout = True)
        bss = [batch.get_effective_bs() for batch in self.batches]
        estimated_times = [batch.estimated_time for batch in self.batches]
        profiled_times = [batch.profiled_time for batch in self.batches]
        profiled_times_ft = [batch.profiled_time_ft for batch in self.batches]
        sch_overheads = [batch.sch_overhead for batch in self.batches]
        print('E[estimated]', np.mean(estimated_times))
        print('E[profiled]', np.mean(profiled_times))
        print('E[profiled_ft]', np.mean(profiled_times_ft))
        print('E[sch overhead]', np.mean(sch_overheads))
        if file_path is None: return
        ax2.twinx().plot(list(range(len(self.batches))), [b.get_effective_bs() for b in self.batches], label = 'BS', c = 'black')
        for tag, data in [
            ('estimated_time', estimated_times),
            ('profiled_time', profiled_times),
            ('profiled_time_ft',profiled_times_ft),
            ('sch_overhead', sch_overheads)
        ]:
            ax1.scatter(bss, data, label = tag)
            ax2.plot(list(range(len(self.batches))), data, label = tag)
        ax1.set_xlabel('# Token')
        ax1.set_ylabel('time (s)')
        ax1.set_title('Cost Model')
        ax2.set_ylabel('time (s)')
        ax2.set_xlabel('Time')
        ax2.set_title('Batch over Time')
        ax1.legend()
        ax2.legend()
        fig.savefig(file_path)
        print('Batches saved to', file_path)


    def _update(self):
        if not hasattr(self, 'real_rate'):
            self.real_rate = None
        # last_arrive_time = max((req.arrive_time for req in self.reqs), default = 0)
        last_finish_time = max((req.timestamps[-1][-1] for req in self.reqs if len(req.timestamps)), default=0)
        last_ddl = max((req.ddl for req in self.reqs), default = 0)
        self.tail_time = last_finish_time - last_ddl
        
        self.profit = sum([req.satisfy_sla(ttft_tolerance=self.ttft_tolerance, tpot_tolerance=self.tpot_tolerance) * req.profit\
            for req in self.reqs]).__float__()
        # count all request that serve under SLO
        self.sla_satisfy_rate = np.mean([req.satisfy_sla(ttft_tolerance=self.ttft_tolerance, tpot_tolerance=self.tpot_tolerance) \
            for req in self.reqs if req.tpot_slo != -1]).__float__()
        
        latenesses: List[Tuple[float, List[float]]] = [req.get_latenesses() 
                for req in self.reqs if req.admitted]
        ttft_latenesses = [lateness[0] for lateness in latenesses]
        tpot_latenesses = sum([lateness[1] for lateness in latenesses], [])

        if len(ttft_latenesses):
            self.ttft_mean: float = np.mean(ttft_latenesses).__float__()
            self.ttft_p50: float = np.percentile(ttft_latenesses, 50).__float__()
            self.ttft_p80: float = np.percentile(ttft_latenesses, 80).__float__()
            self.ttft_p90: float = np.percentile(ttft_latenesses, 90).__float__()
            self.ttft_p99: float = np.percentile(ttft_latenesses, 99).__float__()
        else: 
            self.ttft_mean = 0
            self.ttft_p50 = 0
            self.ttft_p80 = 0
            self.ttft_p90 = 0
            self.ttft_p99 = 0
            
            
        if len(tpot_latenesses):
            self.tpot_mean: float = np.mean(tpot_latenesses).__float__()
            self.tpot_p50: float = np.percentile(tpot_latenesses, 50).__float__()
            self.tpot_p80: float = np.percentile(tpot_latenesses, 80).__float__()
            self.tpot_p90: float = np.percentile(tpot_latenesses, 90).__float__()
            self.tpot_p99: float = np.percentile(tpot_latenesses, 99).__float__()
        else: 
            self.tpot_mean = 0
            self.tpot_p50 = 0
            self.tpot_p80 = 0
            self.tpot_p90 = 0
            self.tpot_p99 = 0
            
        if len(self.reqs):
            self.admission_rate = sum(req.admitted for req in self.reqs) / len(self.reqs)
        else: self.admission_rate = 0

    # def update_sla(self, sla: SLA, batch_timer: BatchTimer):
    #     assert sla.ttft_slown_rate is not None
    #     assert sla.tpot_per_token is not None
    #     trace = Trace.from_str(self.req_trace)
    #     assert len(trace.datasets) == 1
    #     trace.slas[0] = sla
    #     for req in self.reqs:
    #         req.ttft_slo = batch_timer(req.input_length) * sla.ttft_slown_rate
    #         req.tpot_slo = sla.tpot_per_token
    #         req.profit = sla.ttft_per_token_profit * req.input_length + sla.tpot_per_token_profit * req.output_length + sla.base_profit
    #     self._update()
      
    def save(self, 
             prefix: str,
             save_sch: bool = False):
        self._update()
        # schedule_dict = asdict(self)
        
        indices = ['req_trace', 'name', 'req_rate', 'model_tag']
        name = '-'.join([self.req_trace, self.name, self.model_tag, str(self.req_rate)])
        # if store_json:
            # key = '$'.join([str(schedule_dict[idx]) for idx in indices])
            
        if len(self.batches):
            os.makedirs(f'{prefix}/batches', exist_ok=True)
            os.makedirs(f'{prefix}/profile', exist_ok=True)
            self.draw_batches(f'{prefix}/batches/{name}.png')
            
            from .profiler import ProfileDatapoint, save_profile_data
            profile_datapoints = [ProfileDatapoint(sch.get_effective_bs(), 
                                                   sch.profiled_time_ft, 
                                                   0., 
                                                   0., 
                                                   sch, 
                                                   sch.profiled_time,) \
                    for sch in self.batches]
            save_profile_data(profile_datapoints, f'{prefix}/profile/{name}.json')

        os.makedirs(f'{prefix}/lateness', exist_ok=True)
        self.draw_lateness(f'{prefix}/lateness/{name}.png')
        

        os.makedirs(f'{prefix}/loads', exist_ok=True)
        self.draw_loads(f'{prefix}/loads/{name}.png')        
        
        os.makedirs(f'{prefix}/schedule', exist_ok=True)
        schedule_dict = asdict(self)
        # if save_sch:
        file_path = f'{prefix}/schedule/{name}.json'
        with open(file_path, 'w') as f:
            json.dump(schedule_dict, f, indent=4)
        print(f'schedule saved to {file_path}.')

        schedule_dict['batches'] = []
        schedule_dict.pop('reqs')
        schedule_dict.pop('sub_schedules')
        import pandas as pd 
        df = pd.DataFrame([schedule_dict])
        file_path = f'{prefix}/schedule.csv'

        if os.path.exists(file_path):
            prev_df = pd.read_csv(file_path)
            merged_df = pd.concat([prev_df, df])
            merged_df = merged_df.sort_values(by = indices + ['profit'], ascending=False).drop_duplicates(indices, keep = 'first')

        else:
            merged_df = df
        
        merged_df.to_csv(file_path, index=False)
    
    def __repr__(self):
        return f'''
name:{self.name}
trace:{self.req_trace}
admission_rate: {self.admission_rate}
req_rate: {self.req_rate}
failure_rate: {self.failure_rate}
sla_rate: {self.sla_satisfy_rate}
tail_time: {self.tail_time}
real_rate: {self.real_rate}
profit: {self.profit}
overhead: {self.overhead}
ttft_mean : {self.ttft_mean} 
ttft_p50 : {self.ttft_p50} 
ttft_p99 : {self.ttft_p99} 
tpot_mean : {self.tpot_mean}
tpot_p50 : {self.tpot_p50}
tpot_p99 : {self.tpot_p99}
alpha: {self.alpha}
'''

    @staticmethod
    def from_file(file_path: str) -> 'Schedule':
        with open(file_path, 'r') as f:
            data = json.load(f)
        schedule = from_json(Schedule, json_dict = data)
        assert isinstance(schedule, Schedule)
        if hasattr(schedule, 'sub_schedules'):
            schedule.sub_schedules = [from_json(Schedule, kwargs) for kwargs in schedule.sub_schedules]
        else: 
            schedule.sub_schedules = []
        return schedule
@dataclass 
class RequestResult(ReqBatchSchedule):
    n_generated: int = None
    is_finished: bool = None 
    generated_text: str = None
    
@dataclass
class ExecutionResult:
    results: List[RequestResult] = field(default_factory=list)
    draft_time: float = 0
    verifier_time: float = 0
    
@dataclass(kw_only = True)
class Problem:
    tag: str | None = None
    # random seed
    seed: int = 1234
    #model config
    model: str
    para_config: str
    n_param: int = int(7e9)

    # Hardware config
    mem_bw: float = 2e12
    n_block: int | None = None
    compute_speed: float = 312e12
    block_size: int = 16
    
    # Trace config
    request_trace: Trace
    max_seq_len: int = -1
    req_rate: float = 5
    ttft_tolerance: float = 0.05 
    tpot_tolerance: float = 0.05
    
    # Speculative Decode Config
    spec_model: str | None = None
    alpha: float = 0.8
    max_spec_decode_size: int = 6
    spec_decode: SpecDecode = field(init = False)
    fixed_bs: bool = False
    fixed_spec: bool = False
    
    # Requests & solutions
    reqs: List[Request] = field(default_factory = list)
    schedules: List['Schedule'] = field(default_factory=list)
    
    batch_timer: BatchTimer | None = field(init = False)

    save_schedule: bool = False

    def update_slo_scale(self, *, ttft_scale: float = 1.0, tpot_scale: float = 1.0):
        for sla in self.request_trace.slas:
            for sla_ in sla:
                sla_.update(ttft_scale = ttft_scale, tpot_scale = tpot_scale)

    def update_req_rate(self, req_rate: float | None = None):
        if not len(self.reqs): return 
        cur_req_rate = len(self.reqs)  / (self.reqs[-1].arrive_time - self.reqs[0].arrive_time)
        if req_rate is None:
            req_rate = cur_req_rate
        else:
            for req in self.reqs:
                req.arrive_time = req.arrive_time * cur_req_rate / req_rate
                
        self.req_rate = req_rate
    
        
    def update_trace(self, trace_str: str, update_prompt: bool):
        self.request_trace = Trace.from_str(trace_str)
        self.reqs = self.request_trace.sample(self.max_seq_len, self.batch_timer, self.block_size, self.model,\
            update_prompt=update_prompt)
        self.update_req_rate(self.req_rate)
        print('rate, ', self.req_rate)
    
    def __post_init__(self):
        if self.request_trace is None:
            assert len(self.reqs)
            return                     

        self.batch_timer = BatchTimer.from_model(self.model, self.spec_model, self.para_config)
        self.spec_decode = SpecDecode(self.alpha, self.max_spec_decode_size)
        
        self.reqs = self.request_trace.sample(self.max_seq_len, self.batch_timer, self.block_size, self.model)
        self.update_req_rate(self.req_rate)
        print('rate, ', self.req_rate)

    def asdict(self):
        data = asdict(self)
        data.pop('reqs')
        data.pop('batch_timer')
        data.pop('spec_decode')
        # def _recursive(d, l:list):
        #     if isinstance(d, dict):
        #         for k, v in d.items():
        #             l.append('.'+k)
        #             _recursive(v, l)
        #             l.pop()
        #     elif isinstance(d, list):
        #         for i, v in enumerate(d):
        #             l.append(f'[{i}]')
        #             _recursive(v, l)
        #             l.pop()
        #     else:
        #         print (''.join(l), f' = {type(d)}')
        # _recursive(data, [])
                
        return data

    # def save(self, prefix: str):
    #     with open(f'{prefix}/problem.json', 'w') as f:
    #         json.dump(self.asdict(), f, indent = 4)
                        
    #     print(f'data saved to {prefix}/problem.json')
    
    def save(self, schedule: Schedule, prefix: str):
        model_tag = self.model.split('/')[-1]
        for sch in [schedule] + schedule.sub_schedules:
            sch.req_trace = self.request_trace.to_str()                        
            sch.ttft_tolerance = self.ttft_tolerance
            sch.tpot_tolerance = self.tpot_tolerance
            sch.req_rate = round(self.req_rate, 2)
            sch.model_tag = '-'.join([model_tag, self.para_config])
        schedule.save(prefix, self.save_schedule)
        self.schedules.append(schedule)

    @staticmethod 
    def add_cli_args(parser):
        def add_argument(s, *args, **kwargs):
            if s not in parser._option_string_actions:
                parser.add_argument(s, *args, **kwargs)
            else: 
                print('conflict parameter:', s)
        add_argument('--model', type = str, default = 'facebook/opt-6.7b')
        add_argument('--spec-model', type = str, default = None) # 'facebook/opt-125m')
        add_argument('--para_config', type = str, default = '1-1-1')
        add_argument('--tag', type = str, default = 'run')
        add_argument('--trace', type = str, default = None, help='[{ds}-{prompt}-{start}:{end}-{sla}-{stretch},]')
        add_argument('--req_rate', type = int, default=None)
        add_argument('--ttft-tolerance', type = float, default = 0.05)
        add_argument('--tpot-tolerance', type = float, default = 0.05)
        # add_argument('--M', type = int, default = 1875)
        add_argument('--cache-mem-gb', type = int, default = 10)
        add_argument('--alpha', type=float, default = 0.6)
        add_argument('--start', type = int, default = 0)
        add_argument('--block_size', default = 16, type = int)
        add_argument('--max_spec_decode', default = 10, type = int)
        add_argument('--decode_check_freq', default = 10, type = int)
        add_argument('--fixed_bs', action = 'store_true')
        add_argument('--fixed_spec', action = 'store_true')
        add_argument('--save_schedule', action = 'store_true')

        return parser
        
    @staticmethod
    def from_cli_args(args) -> 'Problem':
        from ..models import get_model_config
        
        model_config = get_model_config(args.model)
        mem_per_token = model_config.get_token_cache_mem()
        n_tokens = int(args.cache_mem_gb * 1e9 / mem_per_token)
        n_block = n_tokens // args.block_size
        print(f'Model: {model_config}, {model_config.n_param / 1e9:.2f} B, {n_tokens} Token')
        # exit(0)        
        return Problem(
            tag = args.tag,
            
            model = args.model,
            para_config = args.para_config,
            n_param = model_config.n_param,
            mem_bw = 2e12,
            n_block = n_block,
            compute_speed = 312e12,
            block_size = args.block_size,
            
            max_seq_len = model_config.max_seq_len - args.max_spec_decode,
            request_trace=Trace.from_str(args.trace),
            req_rate = args.req_rate,
            fixed_bs=args.fixed_bs,
            fixed_spec=args.fixed_spec,
            
            spec_model = args.spec_model,
            max_spec_decode_size=args.max_spec_decode,
            alpha = args.alpha,
            ttft_tolerance = args.ttft_tolerance, 
            tpot_tolerance = args.tpot_tolerance,

            save_schedule=args.save_schedule
        )

    def __str__(self):
        return f'''
Problem(
    trace: {self.request_trace},
    req_rate: {self.req_rate},
    spec_decode: {self.spec_decode},
    batch timer: {self.batch_timer},
)
'''    

@dataclass
class MemoryInstance:
    num_blocks: int
    num_best_effort_blocks: int = field(init = False, default = 0)
    tot_num_blocks: int = field(init = False)
    def __post_init__(self):
        self.num_blocks = int(self.num_blocks)
        self.tot_num_blocks = self.num_blocks

    def optional_alloc(self, req: RequestInstanceBase) -> bool:
        assert req.admitted or req.best_effort
        if req.best_effort:
            self.num_best_effort_blocks += req.req.num_blocks
        if self.num_blocks >= req.req.num_blocks:
            self.num_blocks -= req.req.num_blocks
            return True
        return False
    
    def get_n_avail(self, remove_best_effort = False) -> int:
        if remove_best_effort:
            return self.num_blocks + self.num_best_effort_blocks
        return self.num_blocks
    
    def free(self, req: RequestInstanceBase):
        assert req.admitted or req.best_effort
        self.num_blocks += req.req.num_blocks
        if req.best_effort:
            self.num_best_effort_blocks -= req.req.num_blocks
        assert self.num_blocks <= self.tot_num_blocks

    def occupancy(self):
        return round(1 - self.num_blocks / self.tot_num_blocks, 2)    

'''
name:sarathi
profit: 44668
admission_rate: 1.0
sla_rate: 0.88
compute_utilization: 0.03953085300819783
on_ratio: 0.07530303430427243
overhead: 0.11801714729517698
ave_bs: 72.46456692913385
, 
name:vllm
profit: 42843
admission_rate: 1.0
sla_rate: 0.88
compute_utilization: 0.040395358358168876
on_ratio: 0.0815582283305581
overhead: 0.08216163609176874
ave_bs: 70.18164435946463
, 
name:slo
profit: 278
admission_rate: 0.94
sla_rate: 0.02
compute_utilization: 0.03980285046993176
on_ratio: 0.06898510941228207
overhead: 2.88735709246248
ave_bs: 65.60394265232975
'''