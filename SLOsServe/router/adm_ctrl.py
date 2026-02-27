import heapq 
import time
from dataclasses import dataclass, field
import copy
import math

from motivation.common import PerfModel
from SLOsServe.router.execplan_bus import ExecPlan

import logging 

logger = logging.getLogger(__name__)

now = time.time
DEBUG = False
@dataclass(kw_only = True)
class Request: 
    request_id: str 
    num_prompt_tokens: int
    prefill_ddl: float
    slo_tpot: float
    prefill_only: bool
    kv_ready_time: float | None
    
    num_computed_tokens: int = 0
    last_sch_bid: int = -1
    
    def get_next_ddl(self) -> float | None:
        if self.num_computed_tokens < self.num_prompt_tokens:
            return self.prefill_ddl
        return self.prefill_ddl + self.slo_tpot * (self.num_computed_tokens - self.num_prompt_tokens + 1)
        
    def get_next_load(self) -> int | None:
        if self.num_computed_tokens < self.num_prompt_tokens:
            return self.num_prompt_tokens - self.num_computed_tokens
        return 1
    
    def commit(self, n) -> float:    
        if self.num_computed_tokens < self.num_prompt_tokens:
            assert self.num_computed_tokens + n <= self.num_prompt_tokens
        else:
            assert n == 1
        self.num_computed_tokens += n
    
    def finished(self) -> bool:
        if not self.prefill_only: return False
        return self.num_computed_tokens >= self.num_prompt_tokens 

@dataclass 
class Batch:
    unscheduled_tokens: int 
    n_scheduled_tokens: dict[str, int] = field(default_factory = dict)
    
    start_time: float | None = None

@dataclass(kw_only=True)
class BatchPlanner:
    _perf_model: PerfModel
    _max_lookahead: int = 100
    _block_size: int
    _max_decode_length: int
    _profile_events: list = field(default_factory=list)
    
    _num_free_blocks: int # the number of free blocks
    _last_commit_time: float | None = None
    _batch_plan: list[Batch] = field(default_factory=list)
    _requests: dict[str, Request] = field(default_factory=dict)
    _admitted_requests: list[str] = field(default_factory=list)
    _next_batch_time: float | None = None # the begining time of the next batch 
    _perf_model_constant_headroom: float = 0.005
    
    
    def __post_init__(self):
        self._perf_model.hardware_params[4] += self._perf_model_constant_headroom
    
    def _get_batch_time(self, batch: Batch):
        return self._perf_model.get_batch_time([(self._requests[req_id].num_computed_tokens, v) for req_id, v in batch.n_scheduled_tokens.items()])
    def add_request(self, 
                        *,
                        request_id: str, 
                        num_prompt_tokens: int,
                        num_computed_tokens: int,
                        prefill_ddl: float, 
                        slo_tpot: float, 
                        prefill_only: bool = False,
                        kv_ready_time: float | None = None,
                        must_admit: bool = False):
        self._requests[request_id] = Request(request_id = request_id, 
                                             num_prompt_tokens=num_prompt_tokens,
                                             num_computed_tokens=num_computed_tokens,
                                             prefill_ddl = prefill_ddl, 
                                             slo_tpot = slo_tpot, 
                                             prefill_only = prefill_only,
                                             kv_ready_time = kv_ready_time)
        feasible, batch_plan, reason = self._refresh() 
        if (not feasible) and (not must_admit):
            self._requests.pop(request_id)
        else:
            if not feasible:
                logger.info(f'[BatchPlanner] Forced admission of {request_id}')
            self._batch_plan = batch_plan
            self._admitted_requests.append(request_id)
        return must_admit or feasible
    
    def get_next_batch_and_admitted_reqs(self) -> tuple[dict[str, int], set[str], ExecPlan]:
        if not len(self._batch_plan):
            assert len(self._requests) == 0
            self._next_batch_time = None 
            return {}, set(), ExecPlan()
        assert len(self._batch_plan)
        admitted_requests = set(self._admitted_requests)
        self._admitted_requests.clear()
        next_batch = self._batch_plan[0]
        self._next_batch_time = now() + self._get_batch_time(next_batch)
        
        for req_id, n_token in next_batch.n_scheduled_tokens.items():
            self._requests[req_id].commit(n_token)
                    
        exec_plan = ExecPlan()
        t = now()
        next_batch.start_time = t
        num_computed_tokens = {req_id: req.num_computed_tokens for req_id, req in self._requests.items()}
        for i, b in enumerate(self._batch_plan):
            t += self._get_batch_time(b)
            exec_plan.batch_times.append(t)
            for req_id, n_sch in b.n_scheduled_tokens.items():
                num_computed_tokens[req_id] += n_sch
                exec_plan.req_plans[req_id].append((num_computed_tokens[req_id], i))
        exec_plan.num_free_blocks = self._num_free_blocks
        return next_batch.n_scheduled_tokens, admitted_requests, exec_plan
    
    def finish_request(self, request_id: str):
        self._requests.pop(request_id)
        is_feasible, self._batch_plan, reason = self._refresh()
        if not is_feasible:
            logger.info(f'[BatchPlanner]: Infeasibility detected. {reason=}')
            
    def commit_batch(self,
                     num_scheduled_tokens: dict[str, int],
                     finished_reqs: list[str],
                     num_free_blocks: int):
        if self._last_commit_time is not None:
            t = now()
            elapsed = t - self._last_commit_time
            self._perf_model.update(
                [(self._requests[req_id].num_computed_tokens, num_sched) for req_id, num_sched in num_scheduled_tokens.items()],
                elapsed
            )
            assert len(self._batch_plan)
            if DEBUG:
                self._profile_events.append({
                    "event_type": 'batch_planner_batch_commit',
                    "timestamp": t,
                    "extra_args": {
                        "estimated_finish_time": self._next_batch_time,
                        "slack": self._next_batch_time - t,
                        "start_time": self._batch_plan[0].start_time
                    }
                })
            
        self._last_commit_time = now()
        if not num_scheduled_tokens == self._batch_plan[0].n_scheduled_tokens:
            logger.warning(f"num_scheduled_tokens mismatch: {num_scheduled_tokens=}, {self._batch_plan[0].n_scheduled_tokens=}")
            
        for req_id in finished_reqs:
            self._requests.pop(req_id) 
        
        self._num_free_blocks = num_free_blocks 
        self._next_batch_time = now()
        is_feasible, self._batch_plan, reason = self._refresh()
        if not is_feasible:
            logger.info(f'[BatchPlanner]: Infeasibility detected. {reason=}')
        
    def _refresh(self) -> tuple[bool, list[Batch], str | None]:
        '''
        A memory feasibility check and an EDF-based compute feasibility check
        Note that this function does not change states; 
        '''
        is_feasible = True
        infeasible_reason = "|"
        # TODO: Optimize the memory check, the current logic is that _num_free_blocks should be greator than the per-request overprovision for remained blocks and the new_request's size
        mem_ub = 0
        for req in self._requests.values():
            mem_ub += math.ceil((self._max_decode_length + req.num_prompt_tokens - req.num_computed_tokens) / self._block_size)
        if mem_ub > self._num_free_blocks:
            is_feasible = False
            infeasible_reason += f"MEM {mem_ub=}>{self._num_free_blocks=} |"
        
        # Step 2: the compute check 
        Q = []
        B: list[Batch] = []
        for i, req in enumerate(self._requests.values()): 
            req_copy = copy.deepcopy(req)
            req_copy.last_sch_bid = -1
            if not req_copy.finished():
                heapq.heappush(Q, (req_copy.get_next_ddl(), i, req_copy)) # add i to break ties 
        T = now()
        if self._next_batch_time is not None: 
            T = max(T, self._next_batch_time)
        for _ in range(self._max_lookahead): 
            if not len(Q): break
            next_ddl, req_uuid, req = heapq.heappop(Q)
            assert isinstance(req, Request)
            _next_load = next_load = req.get_next_load()
            if req.kv_ready_time is not None and req.kv_ready_time > T:
                # Try to use the gap before KV is ready.
                n_token_next_batch = self._perf_model.get_bs(req.kv_ready_time - T, num_reqs=1)
                if n_token_next_batch > 0:
                    batch = Batch(
                        unscheduled_tokens=n_token_next_batch,
                        start_time=T,
                    )
                    B.append(batch)
                # If the gap is too short (get_bs <= 0), just advance time.
                T = req.kv_ready_time + 1e-6
                    
            if req.last_sch_bid == -1 and req.kv_ready_time is not None:
                # we set last_sch_bid to the index of first batch begins after req.kv_ready_time minus 1
                idx = 0
                while idx < len(B) and B[idx].start_time < req.kv_ready_time:
                    idx += 1
                req.last_sch_bid = idx - 1
                assert (idx == len(B) and T >= req.kv_ready_time) or B[idx].start_time >= req.kv_ready_time
                
            tot_remained_tokens = sum(B[bid].unscheduled_tokens for bid in range(req.last_sch_bid + 1, len(B)))
            if tot_remained_tokens < next_load:
                n_token_next_batch = self._perf_model.get_bs(next_ddl - T, num_reqs = 1)
                start_time = T
                if n_token_next_batch + tot_remained_tokens < next_load:
                    is_feasible = False
                    # This is the fallback; we 
                    n_token_next_batch = next_load - tot_remained_tokens
                    T += self._perf_model.get_batch_time([(req.num_computed_tokens + tot_remained_tokens, n_token_next_batch)])
                    infeasible_reason = f' CMP Delay: {T - next_ddl:.3f}s |'
                else:
                    T = next_ddl
                batch = Batch(
                    unscheduled_tokens= n_token_next_batch,
                    start_time = start_time
                )
                B.append(batch)
                tot_remained_tokens += n_token_next_batch
            
            for bid in range(req.last_sch_bid + 1, len(B)):
                if next_load == 0: break
                n_sch = min(B[bid].unscheduled_tokens, next_load)
                tot_remained_tokens -= n_sch
                B[bid].unscheduled_tokens -= n_sch
                next_load -= n_sch
                if n_sch > 0:
                    assert req.request_id not in B[bid].n_scheduled_tokens
                    B[bid].n_scheduled_tokens[req.request_id] = n_sch
                    req.last_sch_bid = bid
                    
            req.commit(_next_load)
            if not req.finished():
                heapq.heappush(Q, (req.get_next_ddl(), req_uuid, req))
            assert next_load == 0
        
        logger.info(f'[BatchPlanner] Planned {len(B)} batches')
        return is_feasible, B, infeasible_reason
    
