from dataclasses import dataclass, field
from typing import List, Any, Dict
import torch
from itertools import accumulate
import enum
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import warnings

from vllm.distributed import get_tp_group, broadcast_tensor_dict, get_pp_group, set_group_tag
# import time
from .scheduler import Timer 

from .engine_base import BaseEngine
from .scheduler.struct import (
    BatchSchedule,
    ReqBatchSchedule,
    RequestResult,
    ExecutionResult,
    RequestInitMsg,
    RequestInitResult,
    RevokeMsg)
from .models import RequestCacheManager, ModelImpl

class ReqState(enum.Enum):
    PREFILL = enum.auto()
    DECODE = enum.auto()
    PREEMPTED = enum.auto()

@dataclass
class GeneratedIDs:
    output_length: int | None = None
    generated_ids: torch.Tensor = field(init = False)
    
    def __post_init__(self):
        self.generated_ids = torch.tensor([], dtype = torch.int32, device = 'cuda')

    def commit(self, generated_ids_: torch.Tensor):
        # if self.generated_ids is not None:
        self.generated_ids = torch.cat((self.generated_ids, generated_ids_), dim = -1)
    
    def __getitem__(self, *args, **kwargs):
        # assert self.generated_ids is not None
        return self.generated_ids.__getitem__(*args, **kwargs)
    
    def __setitem__(self, *args, **kwargs):
        # assert self.generated_ids is not None
        return self.generated_ids.__setitem__(*args, **kwargs)

    def size(self) -> int:
        return self.generated_ids.size(0)
    
    def is_end(self)->bool:
        return (self.output_length is not None) and (self.size() >= self.output_length)
    
    def drop(self, n):
        if n == 0: return
        self.generated_ids = self.generated_ids[:-n]
    
    def __repr__(self):
        return f'len: {self.size()}/{self.output_length}'

    def preempt(self):
        assert self.output_length is not None
        self.output_length -= self.generated_ids.size(0)
        self.generated_ids = torch.tensor([], dtype = torch.int32, device = 'cuda')

@dataclass 
class RequestInstance:
    id: int
    prompt: str | None
    generated_ids: GeneratedIDs
    cache_manager: RequestCacheManager 
    
    input_ids: torch.Tensor | None = None
    tmp_generated_ids: torch.Tensor | None = None
    # generated_ids: torch.Tensor | None = None
    n_prefill_tokens: int = 0
    state: ReqState = ReqState.PREFILL
    num_block: int = field(init = False)
    
    def preempt(self):
        self.cache_manager.free()
        assert self.input_ids is not None
        self.tmp_generated_ids = None
        self.n_prefill_tokens = 0
        self.state = ReqState.PREEMPTED
        self.input_ids = torch.concat([self.input_ids, self.generated_ids.generated_ids], dim = -1)

    def __post_init__(self):
        assert self.generated_ids.output_length is not None
        self.num_block = self.cache_manager.global_manager.get_num_block(
            self.get_input_length() + 
            self.generated_ids.output_length) 
        
    def get_input_length(self) -> int:
        assert self.input_ids is not None
        return self.input_ids.size(0)
    
    def get_input_ids(self, req: ReqBatchSchedule, verify: bool ) -> torch.Tensor:
        assert req.id == self.id
        assert self.input_ids is not None
        
        if req.is_prefill:
            assert (self.n_prefill_tokens + req.n) <= self.input_ids.size(0)
            return self.input_ids[self.n_prefill_tokens:self.n_prefill_tokens + req.n]
        
        assert self.generated_ids is not None
        if verify:
            return self.generated_ids[-req.n-1:-1]
        return self.generated_ids[-req.n:]
    
    def get_n_samples_from_last(self, req: ReqBatchSchedule) -> int:
        if req.is_prefill:
            assert self.input_ids is not None
            return 1 if req.n + self.n_prefill_tokens >= self.input_ids.size(0) else 0
        return req.n
    
    def get_last_generated_ids(self, req: ReqBatchSchedule) -> torch.Tensor:
        assert not req.is_prefill and req.id == self.id
        assert self.generated_ids is not None
        return self.generated_ids[-req.n:]
    
    def forward(self, req:ReqBatchSchedule, sampled_ids: torch.Tensor):
        assert self.id == req.id 
        if req.is_prefill:
            self.n_prefill_tokens += req.n
            assert self.n_prefill_tokens <= self.get_input_length()
            if self.n_prefill_tokens == self.get_input_length():
                self.state = ReqState.DECODE
        else: 
            assert self.state == ReqState.DECODE
            
        if sampled_ids.size(0):
            self.tmp_generated_ids = sampled_ids
    
    def commit(self):
        if self.tmp_generated_ids is not None:
            self.generated_ids.commit(self.tmp_generated_ids)
            self.tmp_generated_ids = None
    
    def drop(self, n: int):
        if n > 0:
            assert self.generated_ids is not None
            self.cache_manager.update(0, n)

    def __repr__(self):
        return f'''(id {self.id}, #prefill {self.n_prefill_tokens}, S {self.state}, TMP {self.tmp_generated_ids}, generated_ids {id(self.generated_ids)})'''

    def __del__(self):
        del self.cache_manager

@dataclass
class ModelState:
    model: ModelImpl 
    req_states: Dict[int, RequestInstance] = field(default_factory=dict)
    n_blocks: int = 0
    
    def add_batch(self, msgs: List[RequestInitMsg],
                  input_idss: List[List[int]],
                  generated_ids: Dict[int, GeneratedIDs]):
        for msg, input_ids in zip(msgs, input_idss):
            assert msg.input_length
            if msg.id not in generated_ids: 
                generated_ids[msg.id] = GeneratedIDs(msg.output_length)
            if len(input_ids) != msg.input_length:
                warnings.warn(f"len(input_ids) {len(input_ids)} != msg.input_length {msg.input_length}, {msg.prompt}")

            msg.input_length = min(len(input_ids), msg.input_length)
            input_ids = input_ids[:msg.input_length]
            self.req_states[msg.id] = req_state = RequestInstance(
                msg.id, 
                msg.prompt, 
                generated_ids[msg.id], 
                self.model.cache_manager.new(),
                torch.tensor(input_ids, dtype = torch.int32, device = 'cuda'))
            self.n_blocks += req_state.num_block
            if self.n_blocks > self.model.cache_manager.num_blocks:
                print('WARNING', 'memory leaking!!')
    
    def batch_forward(
        self,
        reqs: List[ReqBatchSchedule],
        verify: bool = False
    ):
        set_group_tag(self.model.name)
        if not len(reqs): return
        # print('batch_forward', self.model.name, 'reqs', reqs, 'req_states', self.req_states)
        # for req in reqs:
        #     if req.is_prefill:
        #         req_state = self.req_states[req.id]
        #         req.n = min(req.n, req_state.get_input_length() - req_state.n_prefill_tokens)
        reqs = sorted([req for req in reqs if req.n > 0], key = lambda req: req.n, reverse = True)
        
        input_idss = [self.req_states[req.id].get_input_ids(req, verify) for req in reqs]
        
        assert input_idss is not None
        assert all(x.ndim == 1 for x in input_idss)
        
        cur_seq_lens: List[int] = [input_ids.size(0) for input_ids in input_idss]
        input_ids = torch.concat(input_idss, dim = -1)
        req_cache_managers = [self.req_states[req.id].cache_manager for req in reqs]
        rewind_sizes = [0] * len(reqs)
        
        if not (torch.any(input_ids < self.model.vocab_size) and torch.any(input_ids >= 0)):
            print('input_ids: ', input_ids.size(0), input_ids.max(), input_ids.min())
            assert False
        
        hidden_states = self.model.forward(input_ids, 
                                            req_cache_managers,
                                            cur_seq_lens,
                                            rewind_sizes)
        n_samples = [self.req_states[req.id].get_n_samples_from_last(req) for req in reqs]
        
        sampling_indices = sum([
            list(range(acc_seq_len - n_sample, acc_seq_len)) 
            for n_sample, acc_seq_len in zip(
                n_samples, accumulate(cur_seq_lens)
            )
        ], start = [])
        
        if len(sampling_indices):
            if max(sampling_indices) >= hidden_states.size(0) or \
                min(sampling_indices) < 0:
                sampling_indices = [min(max(_, 0), hidden_states.size(0) - 1) for _ in sampling_indices]
                print('ERROR', 'reqs', reqs, 'n_samples', n_samples, 'cur_seq_lens', cur_seq_lens)
        
        from vllm.model_executor.sampling_metadata import SamplingMetadata
        sampling_meta = SamplingMetadata([], sampling_indices, {}, -1)
        logits = self.model.module.compute_logits(hidden_states, sampling_meta) # [tot_seq_len, vocab_size]

        if get_tp_group().is_first_rank:
            sampled_ids = torch.argmax(logits, dim = -1)
            broadcast_tensor_dict({"sampled_ids": sampled_ids})
        else:
            sampled_ids = broadcast_tensor_dict()['sampled_ids']

        for req, n_sample, idx in zip(reqs, n_samples, accumulate(n_samples)):
            self.req_states[req.id].forward(req, sampled_ids[idx - n_sample: idx])
    
    def commit(
        self,
        reqs
    ):
        for req in reqs:
            self.req_states[req.id].commit()

    def finish(self, req_id: int):
        req_state = self.req_states.pop(req_id)
        self.n_blocks -= req_state.num_block
        del req_state 
    
    def preempt(self, req_id: int):
        req = self.req_states[req_id]
        self.n_blocks -= req.num_block
        req.preempt()
    
    def revoke(self, revoke: RevokeMsg):
        assert revoke.id in self.req_states
        req = self.req_states[revoke.id]
        assert req.state == ReqState.PREEMPTED
        req.state = ReqState.PREFILL
        assert req.get_input_length() == revoke.input_length
        assert req.generated_ids.output_length == revoke.output_length
        self.n_blocks += req.num_block
        
    def __repr__(self):
        return f'''
Model: {self.model.name},
Reqs: {self.req_states}
'''

class SpecDecodeEngine(BaseEngine):
    tokenizer: PreTrainedTokenizerBase
    base_model: ModelState
    spec_model: ModelState | None
    generated_idss: Dict[int, GeneratedIDs]
    finished_reqs: set
    verbose: bool
    timer: Timer
    executed_batches: List[BatchSchedule]
    
    def init(self, base_model_tag, 
             spec_model_tag = None, 
             verbose: bool = False):
        self.base_model_tag = base_model_tag
        self.spec_model_tag = spec_model_tag
        self.verbose = verbose
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_tag)
        self.timer = Timer()
        print('init with tokenizer', self.tokenizer)
        print(f'RANK{self.rank}> TP Group: {get_tp_group().ranks}, PP Group {get_pp_group().ranks}')
        self.reset()
    
    def reset(self):
        self.generated_idss = {}
        self.executed_batches = []
        self.finished_reqs = set()
        self.base_model = ModelState(self.model_impls[self.base_model_tag])
        self.spec_model = ModelState(self.model_impls[self.spec_model_tag])\
            if self.spec_model_tag is not None else None
        self.timer.start()
        self.n_block_used = 0
        # print(self.base_model.model.name, self.base_model.model.cache_manager.get_status())
        # if self.spec_model is not None:
        #     print(self.spec_model.model.name, self.spec_model.model.cache_manager.get_status())
    
    @torch.inference_mode()
    def init_requests(self, msgs: List[RequestInitMsg], revokes: List[RevokeMsg]) -> List[RequestInitResult]:
        self.timer('init_requests BEGIN')
        
        for revoke in revokes:
            self.base_model.revoke(revoke)
            if self.spec_model is not None:
                self.spec_model.revoke(revoke)
        
        if not len(msgs): return []
        # print('init reqs', msgs)
        if msgs[0].prompt is not None:
            assert all(msg.prompt is not None for msg in msgs)
            strs = [msg.prompt for msg in msgs]
            input_idss = self.tokenizer(strs).input_ids
        else:
            assert all(msg.input_length is not None for msg in msgs)
            input_idss = [[x % 100 for x in range(msg.input_length)] for msg in msgs]

        rets: List[RequestInitResult] = [RequestInitResult(msg.id, len(input_ids)) for msg, input_ids in zip(msgs, input_idss)]

        if self.verbose:
            print(f'RANK{self.rank}> init requests')
            # for msg, input_ids in zip(msgs, input_idss):
            #     print('received', msg, 'input len', len(input_ids))
        
        self.base_model.add_batch(msgs, input_idss, self.generated_idss)
        self.timer('init tokenizing END')
        if self.spec_model is not None:
            self.spec_model.add_batch(msgs, input_idss, self.generated_idss)
            # self.spec_model.batch_forward(
            #     [ReqBatchSchedule(msg.id, True, \
            #         self.spec_model.req_states[msg.id].get_input_length()) for msg in msgs]
            # )
        
        if self.verbose:
            print(f'RANK{self.rank}> init requests END')

        self.timer('init_requests END')
        return rets
    
    # @torch.inference_mode()
    # def execute_batches(self, batch_schs: List[BatchSchedule]) -> List[ExecutionResult]:
    #     rets = []
    #     for batch_sch in batch_schs:
    #         rets.append(self.execute_batch(batch_sch))
    #     return sum(rets, [])
    
    @torch.inference_mode()
    def preempt(self, preempt_reqs: List[int]):
        for req_id in preempt_reqs:
            if req_id in self.finished_reqs:
                print('WARNING: the to preempt request has finished')
                continue
            self.base_model.preempt(req_id)
            if self.spec_model is not None:
                self.spec_model.preempt(req_id)
            self.generated_idss[req_id].preempt()
    
    @torch.inference_mode()
    def execute_batch(self, batch_sch: BatchSchedule):
        # print('execute_batch', batch_sch)
        # breakpoint()

        self.timer('execute_batch BEGIN')
        
        if self.verbose:
            print(f'RANK{self.rank}> execute_batch', batch_sch)
        
        def _process_and_filter(req: ReqBatchSchedule):
            if req.id in self.finished_reqs: return False
            s = self.base_model.req_states[req.id]
            if s.state == ReqState.PREEMPTED:
                print('Warning Running preempted op')
                return False
            if not req.is_prefill:
                req.n = min(req.n, 
                        self.base_model.model.max_seq_len - 
                        (s.get_input_length() + s.generated_ids.size()))
            else: 
                req.n = min(req.n, s.get_input_length() - s.n_prefill_tokens)
            return req.n > 0
        
        reqs = [req for req in batch_sch.reqs if _process_and_filter(req)]

        self.executed_batches.append(BatchSchedule(reqs))
        

        if self.verbose:
            print(f'RANK{self.rank}> exec batch:: preprocess reqs', batch_sch)

        if not len(reqs): return ExecutionResult() 
        # torch.cuda.synchronize()
        drafter_start = self.timer.current_time()
        cur_lens = [self.generated_idss[req.id].size() for req in reqs]
        
        decode_reqs = [req for req in reqs if not req.is_prefill]
        if len(decode_reqs):
            max_n = max(decode_reqs, key = lambda x: x.n).n
            
            assert max_n == 1 or self.spec_model is not None
            
            if self.spec_model is not None:
                for i in range(max_n):
                    tmp_reqs = [ReqBatchSchedule(req.id, False, 1) for req in decode_reqs]
                    self.spec_model.batch_forward(tmp_reqs)
                    self.spec_model.commit(tmp_reqs)
                    while len(decode_reqs) and decode_reqs[-1].n == (i+1):
                        decode_reqs.pop()
        self.timer('Spec Model Ends')
        
        # torch.cuda.synchronize()
        verifier_start = self.timer.current_time()
        
        self.base_model.batch_forward(reqs, self.spec_model is not None)
        # torch.cuda.synchronize()
        if self.spec_model is not None:
            self.spec_model.batch_forward(
                [req for req in reqs if req.is_prefill]
            )
        # torch.cuda.synchronize()

        self.timer('Base model Ends')

        if self.verbose:
            print(f'RANK{self.rank}> finsih batch forward', batch_sch)
        
        self.commit(reqs)

        gen_id_lists = [self.generated_idss[req.id][cur_len:].tolist() for req, cur_len in zip(reqs, cur_lens)]
        generated_texts = self.tokenizer.batch_decode(gen_id_lists)
        
        rets: List[RequestResult] = []
        for req, gen_id_list, text in zip(reqs, gen_id_lists, generated_texts):
            gen_ids = self.generated_idss[req.id]
            is_finished = gen_ids.is_end() or (gen_ids.output_length is None \
                    and self.tokenizer.eos_token_id in gen_id_list)
            rets.append(RequestResult(
                req.id, req.is_prefill, req.n,
                n_generated = len(gen_id_list),
                is_finished = is_finished,
                generated_text=text
            ))
            if is_finished: 
                self.base_model.finish(req.id)
                if self.spec_model is not None:
                    self.spec_model.finish(req.id)
                self.generated_idss.pop(req.id)
                self.finished_reqs.add(req.id)
        # torch.cuda.synchronize()
        res = ExecutionResult(
            rets,
            verifier_start - drafter_start,
            self.timer.current_time() - verifier_start
        )
        
        self.timer('execute_batch END')
        if self.verbose:
            print(f'RANK{self.rank}>', batch_sch)
        return res 
        
    def commit(self, reqs: List[ReqBatchSchedule]):
        for req in reqs:
            if req.is_prefill or self.spec_model is None: 
                self.base_model.commit([req])
                continue
            assert self.spec_model is not None
            spec_length = req.n
            base_state = self.base_model.req_states[req.id]
            spec_state = self.spec_model.req_states[req.id]
            generated_ids = spec_state.generated_ids
            true_tokens = base_state.tmp_generated_ids
            guessed_tokens = generated_ids[-req.n:]
            
            # print('generated_ids', generated_ids.generated_ids)
            # print('guessed_tokens', guessed_tokens)
            # print('true_tokens', true_tokens)
            
            is_correct = (guessed_tokens == true_tokens).tolist() + [False]
            n_acc = is_correct.index(False)
            
            # if n_acc == req.n:
            #     # print('generated_ids', generated_ids.generated_ids)
            #     print('guessed_tokens', guessed_tokens)
            #     print('true_tokens', true_tokens)
            
            generated_ids[-req.n:] = true_tokens
            n_keeped = min(n_acc + 1, spec_length)
            rewind_size = spec_length - n_keeped
            
            # print('n_keeped', n_keeped, 'rewind_size', rewind_size)
            # print('generated_ids', generated_ids.generated_ids)
            
            
            generated_ids.drop(rewind_size)
            base_state.drop(rewind_size)
            spec_state.drop(rewind_size)
    
    def display(self):
        # print('generated_ids', self.generated_idss)
        # print('base model', self.base_model)
        # print('spec model', self.spec_model)
        # print('executed batches', self.executed_batches[-10:])
        # print('finished_reqs', sorted(self.finished_reqs))
        for tag, model in [
            ('Base Model', self.base_model),
            ('Spec Model', self.spec_model)
        ]:
            print(tag, end = '')
            if model is None:
                print()
                continue
            print(model.model.name)
            set_group_tag(model.model.name)
            print('\tRank:', self.rank)
            print('\tTP:', get_tp_group().ranks)
            print('\tPP:', get_pp_group().ranks)
        self.timer.display()