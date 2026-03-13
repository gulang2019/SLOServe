import heapq 
import json
import os
import time
from dataclasses import dataclass, field
import copy
import math
import SLOsServe_C

from SLOsServe.perf_model import PerfModel
from SLOsServe.router.execplan_bus import ExecPlan

import logging 

logger = logging.getLogger(__name__)


DEBUG = False
_LOCAL_DUMP_DIR = "adm_ctrl_dumps"

@dataclass(kw_only = True)
class Request: 
    request_id: str 
    num_prompt_tokens: int
    prefill_ddl: float
    slo_tpot: float
    prefill_only: bool
    kv_ready_time: float | None
    output_length: int 
    
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
    
    def finished(self, use_oracle: bool = False) -> bool:
        if not self.prefill_only: 
            if use_oracle:
                assert self.output_length is not None 
                return self.num_computed_tokens >= self.num_prompt_tokens + self.output_length
            return False 
        return self.num_computed_tokens >= self.num_prompt_tokens 

@dataclass 
class Batch:
    unscheduled_tokens: int 
    n_scheduled_tokens: dict[str, int] = field(default_factory = dict)
    
    start_time: float | None = None

@dataclass(kw_only=True)
class BatchPlanner:
    _perf_model: PerfModel
    _max_lookahead: int = 5000
    _block_size: int
    _max_decode_length: int
    _profile_events: list = field(default_factory=list)
    _is_oracle: bool = False
    _now = time.time
    
    _num_free_blocks: int # the number of free blocks
    _last_commit_time: float | None = None
    _batch_plan: list[Batch] = field(default_factory=list)
    _requests: dict[str, Request] = field(default_factory=dict)
    _admitted_requests: list[str] = field(default_factory=list)
    _next_batch_time: float | None = None # the begining time of the next batch 
    _perf_model_constant_headroom: float = 0.005
    _last_infeasible_reason: str | None = None
    _adm_ctrler: any = None
    _adm_ctrler_tpot: float | None = None
    _max_time: int = 5
    _cpp_adm_ctrl_iterations: int = 0
    _cpp_schedule_iterations: int = 0
    
    
    def __post_init__(self):
        self._perf_model.hardware_params[4] += self._perf_model_constant_headroom
        # Initialize C++ admission controller; TPOT is set lazily per request mix.
        self._adm_ctrler = SLOsServe_C.AdmCtrlScheduler("edf_sim", self._block_size, False, False)
        self._adm_ctrler_tpot = None

    def _get_batch_time(self, batch: Batch):
        return self._perf_model.get_batch_time([(self._requests[req_id].num_computed_tokens, v) for req_id, v in batch.n_scheduled_tokens.items()])

    def _ensure_cpp_planner(self, tpot: float):
        if self._adm_ctrler_tpot is not None and abs(self._adm_ctrler_tpot - tpot) < 1e-9:
            return
        tpots = [tpot]
        hw = list(self._perf_model.hardware_params)
        self._adm_ctrler.set_ar_planner(tpots=tpots, hardware_params=hw, fixed_bs=False)
        self._adm_ctrler_tpot = tpot

    def _dump_schedule_inputs(
        self,
        c_reqs: list,
        accepted_ids: list[str],
        *,
        num_free_blocks: int,
        current_time: float,
        is_feasible: bool,
        surfix: str
    ) -> None:
        try:
            now = time.time()
            os.makedirs(_LOCAL_DUMP_DIR, exist_ok=True)
            filename = os.path.join(
                _LOCAL_DUMP_DIR,
                f"schedule_{self.tag}_{surfix}.txt",
            )
            with open(filename, "w") as f:
                f.write("SLOPACKER_SCHEDULE_DUMP_V1\n")
                f.write(f"timestamp {now}\n")
                f.write("did 0\n")
                f.write(f"mode {json.dumps('adm_ctrl_refresh')}\n")
                f.write("scheduler_mode \"edf_sim\"\n")
                f.write("scheduler_fifo_fair 0\n")
                f.write("scheduler_continuous 0\n")
                f.write("planner_type \"ar\"\n")
                f.write("planner_fixed_bs 0\n")
                f.write("planner_max_bs 16384\n")
                tpot = self._adm_ctrler_tpot if self._adm_ctrler_tpot is not None else 0.0
                f.write(f"tpots 1 {tpot}\n")
                hardware_params = self._perf_model.hardware_params
                f.write(f"hardware_params {len(hardware_params)}")
                for x in hardware_params:
                    f.write(f" {x}")
                f.write("\n")
                f.write(f"M {num_free_blocks}\n")
                f.write(f"current_time {current_time}\n")
                f.write(f"max_time {self._max_time}\n")
                f.write(f"observed_is_feasible {int(is_feasible)}\n")
                f.write("observed_schedule_elapsed_s 0.0\n")
                f.write("observed_total_elapsed_s 0.0\n")
                f.write(f"observed_accepted_ids {len(accepted_ids)}\n")
                for req_id in accepted_ids:
                    f.write(f"accepted_id {json.dumps(str(req_id))}\n")
                f.write(f"reqs {len(c_reqs)}\n")
                for req in c_reqs:
                    f.write(
                        "req "
                        f"{json.dumps(str(req.id))} "
                        f"{int(req.is_new_req)} "
                        f"{float(req.ddl)} "
                        f"{int(req.input_length)} "
                        f"{int(req.n_computed_tokens)} "
                        f"{float(req.profit)} "
                        f"{int(req.mem)} "
                        f"{int(req.tpot_idx)} "
                        f"{int(req.prefill_mem)} "
                        f"{int(req.prefill_device_id)} "
                        f"{int(req.decode_device_id)} "
                        f"{int(req.prefill_only)} "
                        f"{float(req.arrival_time)} "
                        f"{int(req.max_tokens)}\n"
                    )
            logger.warning(f'dumping takes {time.time()-now:.3f}s')
            logger.warning(f"[BatchPlanner] Dumped anomalous cpp schedule inputs to {filename}")
        except Exception:
            logger.exception("Failed to dump anomalous cpp schedule inputs")

    def _to_cpp_request(self, req: Request, *, is_new_req: bool, now: float):
        # Keep the same memory model as Python _refresh.
        if req.prefill_only:
            max_decode_length = 0
        else:
            max_decode_length = self._max_decode_length if not self._is_oracle else req.output_length
        assert max_decode_length is not None
        mem = math.ceil((max_decode_length + req.num_prompt_tokens - req.num_computed_tokens) / self._block_size)
        prefill_mem = math.ceil(req.num_prompt_tokens / self._block_size)
        return SLOsServe_C.Request(
            id=req.request_id,
            is_new_req=is_new_req,
            ddl=req.prefill_ddl - now,
            input_length=req.num_prompt_tokens,
            n_computed_tokens=req.num_computed_tokens,
            profit=1.0,
            mem=mem,
            tpot_idx=0,
            prefill_mem=prefill_mem,
            prefill_device_id=0,
            decode_device_id=0,
            prefill_only=req.prefill_only,
            arrival_time=0.0,
            max_tokens = max_decode_length
        )

    def _cpp_feasible_with_new(self, new_req: Request, now: float) -> bool:
        start = time.time()
        reqs = list(self._requests.values()) + [new_req]
        tpot = min(r.slo_tpot for r in reqs) if reqs else new_req.slo_tpot
        self._ensure_cpp_planner(tpot)
        c_reqs = [self._to_cpp_request(r, is_new_req=False, now=now) for r in self._requests.values()]
        c_reqs.append(self._to_cpp_request(new_req, is_new_req=True, now=now))
        is_feasible, is_accepteds = self._adm_ctrler.adm_ctrl(c_reqs, self._num_free_blocks, 0.0)
        # print('c_reqs', c_reqs, 'num_free_blocks', self._num_free_blocks)
        # print('is_feasible', is_feasible, 'is_accepteds', is_accepteds)
        self._cpp_adm_ctrl_iterations += 1
        elapsed = time.time() - start
        self._dump_schedule_inputs(
            c_reqs,
            [req.id for req, is_acc in zip(c_reqs, is_accepteds) if is_acc],
            num_free_blocks=self._num_free_blocks,
            current_time=0.0,
            is_feasible=bool(is_feasible),
            surfix = f'adm_ctrl_{new_req.request_id}'
        )
        return bool(is_feasible and len(is_accepteds) == len(c_reqs) and is_accepteds[-1])
    
    def add_request(self, 
                        *,
                        request_id: str, 
                        num_prompt_tokens: int,
                        num_computed_tokens: int,
                        prefill_ddl: float, 
                        slo_tpot: float, 
                        prefill_only: bool = False,
                        kv_ready_time: float | None = None,
                        must_admit: bool = False,
                        output_length: int | None = None):
        assert not self._is_oracle or output_length is not None
        new_req = Request(request_id = request_id, 
                          num_prompt_tokens=num_prompt_tokens,
                          num_computed_tokens=num_computed_tokens,
                          prefill_ddl = prefill_ddl, 
                          slo_tpot = slo_tpot, 
                          prefill_only = prefill_only,
                          kv_ready_time = kv_ready_time,
                          output_length = output_length)
        now = self._next_batch_time if self._next_batch_time is not None else self._now()
        feasible = self._cpp_feasible_with_new(new_req, now)
        if (not feasible) and (not must_admit):
            self._last_infeasible_reason = "CPP admission rejected new request"
        else:
            self._requests[request_id] = new_req
            if not feasible:
                logger.info(f'[BatchPlanner] Forced admission of {request_id}')
            self._admitted_requests.append(request_id)
            self._last_infeasible_reason = None
        return must_admit or feasible
        
    
    def get_next_batch_and_admitted_reqs(self) -> tuple[dict[str, int], set[str], ExecPlan]:
        start = time.time()
        
        if self._next_batch_time is not None:
            print(f"laxity {self._now() - self._next_batch_time:.3f}")
            
        self._last_commit_time = self._now()
        
        is_feasible, self._batch_plan, reason = self._refresh_fast()
        if not is_feasible:
            logger.info(f'[BatchPlanner]: Infeasibility detected before dispatch. {reason=}')
        if not len(self._batch_plan):
            assert len(self._requests) == 0 or not is_feasible
            self._next_batch_time = None 
            return {}, set(), ExecPlan()
        assert len(self._batch_plan)
        admitted_requests = set(self._admitted_requests)
        self._admitted_requests.clear()
        next_batch = self._batch_plan[0]
        self._next_batch_time = self._now() + self._get_batch_time(next_batch)
        
        for req_id, n_token in next_batch.n_scheduled_tokens.items():
            self._requests[req_id].commit(n_token)
            
        refresh_time = time.time() - start 
                    
        exec_plan = ExecPlan()
        t = self._now()
        next_batch.start_time = t
        num_computed_tokens = {req_id: req.num_computed_tokens for req_id, req in self._requests.items()}
        for i, b in enumerate(self._batch_plan):
            t += self._get_batch_time(b)
            exec_plan.batch_times.append(t)
            for req_id, n_sch in b.n_scheduled_tokens.items():
                num_computed_tokens[req_id] += n_sch
                exec_plan.req_plans[req_id].append((num_computed_tokens[req_id], i))
        exec_plan.num_free_blocks = self._num_free_blocks
        self._last_scheduled_tokens = copy.copy(next_batch.n_scheduled_tokens)
        
        
        e2e_time = time.time() - start
        if e2e_time > 0.1:
            logger.warning(f'[BatchPlanner] LONGSCH {refresh_time=}, {e2e_time=},{len(self._batch_plan)=}')
        return next_batch.n_scheduled_tokens, admitted_requests, exec_plan
    
    def finish_request(self, request_id: str):
        self._requests.pop(request_id)
        # is_feasible, self._batch_plan, reason = self._refresh()
        # if not is_feasible:
        #     logger.info(f'[BatchPlanner]: Infeasibility detected. {reason=}')
            
    def commit_batch(self,
                     num_scheduled_tokens: dict[str, int],
                     finished_reqs: list[str],
                     num_free_blocks: int):
        
        # if not num_scheduled_tokens == self._batch_plan[0].n_scheduled_tokens:
        #     logger.warning(f"num_scheduled_tokens mismatch: {num_scheduled_tokens=}, {self._batch_plan[0].n_scheduled_tokens=}")
            
        for req_id in finished_reqs:
            self._requests.pop(req_id) 
        
        self._num_free_blocks = num_free_blocks 
        self._next_batch_time = max(self._next_batch_time or 0, self._now())
        # is_feasible, self._batch_plan, reason = self._refresh_c()
        # if not is_feasible:
            # logger.info(f'[BatchPlanner]: Infeasibility detected. {reason=}')

    def _refresh_fast(self) -> tuple[bool, list[Batch], str | None]:
        now = max(self._next_batch_time or 0, self._now())
        baseline_batch_time = self._perf_model.get_batch_time([(0, 256)])
        load_ddls = sorted(
            [
                (req_id, req.get_next_ddl(), req.get_next_load())
                for req_id, req in self._requests.items()
                if not req.finished(self._is_oracle)
            ],
            key=lambda x: (x[1], x[0]),
        )
        if not load_ddls:
            return True, [], "|"

        cutoff = now + baseline_batch_time
        feasible_load_ddls = [x for x in load_ddls if x[1] >= cutoff]
        overdue_load_ddls = [x for x in load_ddls if x[1] < cutoff]

        if feasible_load_ddls:
            batch_time_budget = feasible_load_ddls[0][1] - now
            batch_size = max(0, self._perf_model.get_bs(batch_time_budget, num_reqs=1))
        else:
            batch_size = 16384  # Recover overdue work aggressively when no request is still feasible.
            batch_time_budget = self._perf_model.get_batch_time([(0, batch_size)])

        b = Batch(start_time=now, unscheduled_tokens=batch_size)
        total_scheduled_tokens = 0
        total_past_tokens = 0
        num_scheduled_reqs = 0
        # Give missed deadlines only whatever capacity remains after EDF-feasible work.
        for req_id, _next_ddl, next_load in feasible_load_ddls + overdue_load_ddls:
            if b.unscheduled_tokens <= 0:
                break
            req = self._requests[req_id]
            composition_limited_bs = max(
                0,
                self._perf_model.get_bs(
                    batch_time_budget,
                    num_reqs=num_scheduled_reqs + 1,
                    num_past_tokens=total_past_tokens + req.num_computed_tokens,
                ),
            )
            remaining_tokens = min(batch_size, composition_limited_bs) - total_scheduled_tokens
            if remaining_tokens <= 0:
                break
            n_sch = min(remaining_tokens, next_load)
            if n_sch <= 0:
                continue
            b.n_scheduled_tokens[req_id] = n_sch
            total_scheduled_tokens += n_sch
            total_past_tokens += req.num_computed_tokens
            num_scheduled_reqs += 1
            b.unscheduled_tokens = batch_size - total_scheduled_tokens
        return True, [b], None
            

    def _refresh_c(self) -> tuple[bool, list[Batch], str | None]:
        if not self._requests:
            return True, [], "|"
        start = time.time()
        now = self._next_batch_time if self._next_batch_time is not None else self._now()
        tpot = min(req.slo_tpot for req in self._requests.values())
        self._ensure_cpp_planner(tpot)
        setup_elapsed = time.time() - start
        c_reqs = [self._to_cpp_request(req, is_new_req=False, now=now) for req in self._requests.values()]
        is_feasible, c_batches = self._adm_ctrler.schedule(
            c_reqs, 0.0, self._max_time, False,
        )
        accepted_ids = [req.id for req in c_reqs]
        sch_elapsed = time.time() - start
        reqs = self._requests
        py_batches: list[Batch] = []
        for c_batch in c_batches:
            n_scheduled_tokens: dict[str, int] = {}
            for req_batch in c_batch.req_batches:
                n = req_batch.n
                if n <= 0:
                    continue
                req = reqs.get(req_batch.id)
                if req is None:
                    continue
                if req.num_computed_tokens < req.num_prompt_tokens:
                    max_sched = req.num_prompt_tokens - req.num_computed_tokens
                else:
                    max_sched = 1
                if max_sched <= 0:
                    continue
                if n > max_sched:
                    n = max_sched
                prev = n_scheduled_tokens.get(req_batch.id)
                n_scheduled_tokens[req_batch.id] = n if prev is None else (prev + n)

            if n_scheduled_tokens:
                py_batches.append(Batch(unscheduled_tokens=0, n_scheduled_tokens=n_scheduled_tokens))
        elapsed = time.time() - start
        logger.info(f'[BatchPlanner] Planned {len(py_batches)} batches (cpp)')
        self._cpp_schedule_iterations += 1
        # if not is_feasible or self._cpp_schedule_iterations % 1 == 0:
            # logger.info(
            #     "[BatchPlanner] Periodic cpp schedule dump at iteration %d",
            #     self._cpp_schedule_iterations,
            # )
        if elapsed > 0.05:
            logger.warning(f'[BatchPlanner] LONG SCHEDULE: {setup_elapsed=}, {sch_elapsed=}, {elapsed=}')
            self._dump_schedule_inputs(
                c_reqs,
                list(accepted_ids),
                num_free_blocks=self._num_free_blocks,
                current_time=0.0,
                is_feasible=bool(is_feasible),
                surfix = 'sch' + "" if elapsed < 0.05 else '_LONG'
            )
        return bool(is_feasible), py_batches, ("|" if is_feasible else " CPP schedule infeasible |")

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
            if req.prefill_only:
                max_decode_length = 0
            else:
                max_decode_length = self._max_decode_length if not self._is_oracle else req.output_length
            assert max_decode_length is not None
            mem_ub += math.ceil((max_decode_length + req.num_prompt_tokens - req.num_computed_tokens) / self._block_size)
                
        if mem_ub > self._num_free_blocks:
            is_feasible = False
            infeasible_reason += f"MEM {mem_ub=}>{self._num_free_blocks=} |"
        
        # Step 2: the compute check 
        Q = []
        B: list[Batch] = []
        n_reqs = len(self._requests)
        has_committed = [False] * n_reqs
        is_stop_ready = [False] * n_reqs
        n_stop_ready = 0
        for i, req in enumerate(self._requests.values()): 
            req_copy = copy.deepcopy(req)
            req_copy.last_sch_bid = -1
            if req_copy.finished(self._is_oracle):
                is_stop_ready[i] = True
                n_stop_ready += 1
            if not req_copy.finished(self._is_oracle):
                heapq.heappush(Q, (req_copy.get_next_ddl(), i, req_copy)) # add i to break ties 
        T = self._now()
        if self._next_batch_time is not None: 
            T = max(T, self._next_batch_time)
        n_lookahead = 0
        while (
            len(Q)
            and n_stop_ready < n_reqs
            and n_lookahead < self._max_lookahead
            and (self._max_plan_batches < 0 or len(B) < self._max_plan_batches)
        ):
            n_lookahead += 1
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
            has_committed[req_uuid] = has_committed[req_uuid] or (_next_load > 0)
            if (
                not is_stop_ready[req_uuid]
                and (
                    req.finished(self._is_oracle)
                    or (req.num_computed_tokens >= req.num_prompt_tokens and has_committed[req_uuid])
                )
            ):
                is_stop_ready[req_uuid] = True
                n_stop_ready += 1
            if not req.finished(self._is_oracle):
                heapq.heappush(Q, (req.get_next_ddl(), req_uuid, req))
            assert next_load == 0
        
        logger.info(f'[BatchPlanner] Planned {len(B)} batches')
        return is_feasible, B, infeasible_reason
    
    def set_state(self,
                  num_free_blocks: int,
                  next_batch_time: float,
                  requests: dict[str, Request]):
        self._num_free_blocks = num_free_blocks
        self._next_batch_time = next_batch_time
        self._requests = requests
        feasible, batch_plan, reason = self._refresh()
        if not feasible: 
            logger.info(f'[BatchPlanner] initial state not feasible {num_free_blocks=}, {next_batch_time=}, {requests=}, {reason=}')
        self._batch_plan = batch_plan
        return feasible
