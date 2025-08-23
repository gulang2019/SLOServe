import abc
from typing import List, Tuple
import abc
import math
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass, field
import numpy as np
from .spec_decode import SpecDecode
import time
from .C import promax


from .struct import (
    BatchSchedule,
    ReqBatchSchedule,
    Problem,
    RequestInstanceBase,
    RequestState,
    MemoryInstance
)
from .batch_timer import BatchTimer

MAX_BS = 2048

options = {
    'WLSACCESSID': 'cbb10db7-5ebb-4679-a160-809f24f51bb8',
    'WLSSECRET': '09942742-4a87-4664-ae88-09e703c9e46b',
    'LICENSEID': 2578678,
    # "OutputFlag": 0,
    "TimeLimit": 0.05,
}

@dataclass
class ScheduleALG(abc.ABC): 
    memory: MemoryInstance
    
    @abc.abstractmethod
    def schedule(self, reqs: List[RequestInstanceBase], 
                 new_reqs: List[RequestInstanceBase], current_time: float) \
        -> Tuple[List[BatchSchedule], List[RequestInstanceBase], List[RequestInstanceBase]]:
        ...

    def _decline_requests_mem(self, reqs: List[RequestInstanceBase])\
        -> Tuple[List[RequestInstanceBase], List[RequestInstanceBase]]:
            n_avail = self.memory.get_n_avail()
            for i in range(len(reqs)):
                if (n_avail - reqs[i].req.num_blocks) < 0:
                    return reqs[:i], reqs[i:]
                n_avail -= reqs[i].req.num_blocks
            return reqs, []
            
class OracleScheduler(ScheduleALG):
    def __init__(self, mem: MemoryInstance, prob: Problem):
        super(OracleScheduler, self).__init__(mem)
        from .milp import compute_oracle
        self.sch = compute_oracle(prob)
        self.unit_time = prob.batch_timer(prob.batch_timer.shift_bs)
        
    def schedule(self, 
                 reqs: List[RequestInstanceBase],
                 new_reqs: List[RequestInstanceBase], current_time: float)\
        -> Tuple[List[BatchSchedule], List[RequestInstanceBase], List[RequestInstanceBase]]:
        admitted_reqs = [req for req in new_reqs if self.sch.keep_requests[req.id]]
        declined_reqs = [req for req in new_reqs if not self.sch.keep_requests[req.id]]
        bid = round(current_time / self.unit_time)
        return [self.sch.batches[bid]], admitted_reqs, declined_reqs
    
class vLLMScheduler(ScheduleALG):
    def __init__(self, mem: MemoryInstance): 
        super(vLLMScheduler, self).__init__(mem)
        
    def schedule(self, reqs: List[RequestInstanceBase], new_reqs: List[RequestInstanceBase], current_time: float) \
        -> Tuple[List[BatchSchedule], List[RequestInstanceBase], List[RequestInstanceBase]]:
        
        new_reqs, declined_reqs = self._decline_requests_mem(new_reqs)
        
        batches = []
        for req in reqs + new_reqs:
            if req.is_prefill():
                batches.append(BatchSchedule(reqs = [ReqBatchSchedule(req.id, 
                                                is_prefill = True,
                                                n = req.input_length)]))
                # return [BatchSchedule(reqs = [ReqBatchSchedule(req.id, 
                #                                 is_prefill = True,
                #                                 n = req.input_length)])], new_reqs, declined_reqs
        batches.append(BatchSchedule(reqs=[ReqBatchSchedule(req.id, is_prefill = False, n = 1) for req in reqs + new_reqs], next = 0))
        return batches, new_reqs, declined_reqs

class SarathiServeScheduler(ScheduleALG):
    def __init__(self,
                 mem: MemoryInstance, 
                 batch_timer: BatchTimer):
        super(SarathiServeScheduler, self).__init__(mem)
        self.batch_timer = batch_timer
        
    def schedule(self, reqs: List[RequestInstanceBase], new_reqs: List[RequestInstanceBase], current_time: float) \
        -> Tuple[List[BatchSchedule], List[RequestInstanceBase], List[RequestInstanceBase]]:
        
        new_reqs, declined_reqs = self._decline_requests_mem(new_reqs)
        
        decode_reqs = [req for req in reqs + new_reqs if req.state == RequestState.Decode]
        prefill_reqs = [req for req in reqs + new_reqs if req.state in (RequestState.Arrived, RequestState.Prefill)]

        if len(decode_reqs) == 0:
            return [BatchSchedule([
                ReqBatchSchedule(req.id, is_prefill=True, n = req.input_length - req._n_prefill_tokens) for req in prefill_reqs])], new_reqs, declined_reqs
            
        strongest_tpot = min([req.req.tpot() for req in decode_reqs])
                
        max_bs = max(range(1, 2048, 32), key=lambda i: i if self.batch_timer(i) <= strongest_tpot else 0)                

        remain_bs = max_bs - len(decode_reqs)
        
        if len(prefill_reqs):
            prefill_bs = remain_bs // len(prefill_reqs)
        
        return [
            BatchSchedule(
                [ReqBatchSchedule(req.id, is_prefill = False, n = 1) for req in decode_reqs] +\
                [ReqBatchSchedule(req.id, is_prefill = True, n = min(prefill_bs, req.input_length - req._n_prefill_tokens)) 
                 for req in prefill_reqs],
            )
        ], new_reqs, declined_reqs

class SLOAwareScheduler(ScheduleALG):
    def __init__(self,
                 mem: MemoryInstance,  
                 bs: int,
                 t0: float,
                 n_sch_after_last: int):
        super(SLOAwareScheduler, self).__init__(mem)
        self.bs = bs
        self.t0 = t0
        self.n_sch_after_last = n_sch_after_last
        self.sch_cnt = 0
    
    @dataclass
    class ReqRequirement:
        is_new_req: bool
        prefill_ddl: int
        n_prefill_token: int
        n_token_per_unit: float
        profit: int
        num_blocks: int 
        
    def solve(self, m: gp.Model,
              old_reqs_requirements: List[ReqRequirement],
              new_reqs_requirements: List[ReqRequirement],):
        # print('old_reqs', old_reqs_requirements)
        # print('new_reqs_requirements', new_reqs_requirements)
        
        n_old_reqs = len(old_reqs_requirements)
        n_new_reqs = len(new_reqs_requirements)
        
        keep = m.addMVar((n_new_reqs,), name = 'keep', vtype = GRB.BINARY)
        profits = np.array([r.profit for r in new_reqs_requirements])
        m.setObjective(keep @ profits, GRB.MAXIMIZE)
        old_token_rate = sum(r.n_token_per_unit for r in old_reqs_requirements)
        new_token_rates = np.array([r.n_token_per_unit for r in new_reqs_requirements])
        
        m.addConstr(new_token_rates @ keep <= (self.bs - old_token_rate))
        memory_consumptions = np.array([r.num_blocks for r in new_reqs_requirements], dtype = np.int32)
        # print('mem consumption', memory_consumptions)
        m.addConstr(memory_consumptions @ keep <= self.memory.get_n_avail())

        reqs = old_reqs_requirements + new_reqs_requirements
        indices = sorted(range(n_old_reqs + n_new_reqs), key = lambda idx: reqs[idx].prefill_ddl)
                
        for i, idx in enumerate(indices):
            exprs = []
            ws = 0
            req = reqs[idx]
            for j in range(i+1):
                prior_req = reqs[indices[j]]
                assert req.prefill_ddl >= prior_req.prefill_ddl
                w = (req.prefill_ddl - prior_req.prefill_ddl) * prior_req.n_token_per_unit + prior_req.n_prefill_token
                if prior_req.is_new_req:
                    exprs.append(w*keep[indices[j] - n_old_reqs])
                else: ws += w
            if len(exprs) != 0:
                m.addConstr(gp.quicksum(exprs) <= (self.bs * req.prefill_ddl - ws))
            else:
                if ws > self.bs * req.prefill_ddl:
                    print('ws', ws, 'bs', self.bs, 'prefill_ddl', req.prefill_ddl)
                assert ws <= self.bs * req.prefill_ddl
        start = time.perf_counter()
        m.optimize()
        print('optimize takes ', time.perf_counter() - start, 's')
        keep_results = [bool(var.X) for var in keep.tolist()]
        return keep_results
    
    def _get_requirements(
        self,
        reqs: List[RequestInstanceBase],
        is_new_req: bool,
        current_time: float
    ):
        # print('t0', self.t0)
        for req in reqs:
            # if req.is_prefill():
            # print('req', req, 'ttft', req.req.ttft())
            prefill_ddl = math.floor((math.floor(req.arrive_time / self.t0) * self.t0 + req.req.ttft() - current_time) / self.t0)
            prefill_ddl = max(prefill_ddl, 0)
            n_prefill_token = req.input_length - req._n_prefill_tokens
            assert n_prefill_token >= 0

            req._sch_requirement = SLOAwareScheduler.ReqRequirement(
                is_new_req,
                prefill_ddl,
                n_prefill_token,
                req.req.sla.tpot_slown_rate,
                req.req.profit(),
                req.req.num_blocks
            )
    
    def _advance_prefill_ddls(
        self,
        reqs: List[RequestInstanceBase]
    ):
        
        remains = []
        for i, req in enumerate(reqs):
            assert isinstance(req._sch_requirement, self.ReqRequirement)
            cur_ddl = req._sch_requirement.prefill_ddl
            w = 0
            for j in range(i+1):
                prior_req: SLOAwareScheduler.ReqRequirement = reqs[j]._sch_requirement
                w += (cur_ddl - prior_req.prefill_ddl) * prior_req.n_token_per_unit + prior_req.n_prefill_token
            remain = self.bs * cur_ddl - w 
            remains.append(remain)
        remains = np.array(remains)
        prev_prefill_ddl = 0
        
        for i, req in enumerate(reqs):
            delta = min(math.floor(remains[i:].min() / req._sch_requirement.n_token_per_unit), 
                        math.floor(remains[i] / self.bs),
                        req._sch_requirement.prefill_ddl - prev_prefill_ddl)
            req._sch_requirement.prefill_ddl -= delta
            remains[i:] -= delta * req._sch_requirement.n_token_per_unit
            prev_prefill_ddl = req._sch_requirement.prefill_ddl
        assert np.all(remains >= 0)
    
    def _push_prefill_ddls(self, reqs: List[RequestInstanceBase]):
        reqs = sorted(reqs, key = lambda req: req._sch_requirement.prefill_ddl)
        '''
        t_i >= Ceil[n_i + Sum_{j<i}(n_j - t_j k_j) /B - Sum_j(k_j)]
        '''
        numerator = 0
        denominator = 0
        for req in reqs:
            prefill_ddl = math.ceil((req._sch_requirement.n_prefill_token+ numerator) / (self.bs - denominator))
            numerator += req._sch_requirement.n_prefill_token - req._sch_requirement.prefill_ddl * req._sch_requirement.n_token_per_unit
            denominator += req._sch_requirement.n_token_per_unit
            if prefill_ddl > req._sch_requirement.prefill_ddl:
                print(f'push prefill ddl of Req #{req.id} from {req._sch_requirement.prefill_ddl} to {prefill_ddl}')
                req._sch_requirement.prefill_ddl = prefill_ddl
    
    def _draw(self, reqs):
        import matplotlib.pyplot as plt 
        fig, ax = plt.subplots()
        n_schs = max([req._sch_requirement.prefill_ddl for req in reqs]) + self.n_sch_after_last
        xs = np.arange(n_schs)
        ax.plot(xs, xs * self.bs, label = 'tot_cap')
        for req in reqs:
            tmp = xs[req._sch_requirement.prefill_ddl:]
            ax.plot(tmp, 
                    req._sch_requirement.n_prefill_token + (tmp - req._sch_requirement.prefill_ddl) * req._sch_requirement.n_token_per_unit,
                    label = f'req {req.id}')
        ax.set_xticks([req._sch_requirement.prefill_ddl for req in reqs])
        ax.set_yticks([req._sch_requirement.n_prefill_token for req in reqs])
        plt.legend()
        fig.savefig(f'schedule/solve.png')
    
    def schedule(self, 
                 reqs: List[RequestInstanceBase], 
                 new_reqs: List[RequestInstanceBase],
                 current_time: float) \
        -> Tuple[List[BatchSchedule], List[RequestInstanceBase], List[RequestInstanceBase]]:

        if len(reqs) + len(new_reqs) == 0:
            return [BatchSchedule([])], [], []

        self._get_requirements(reqs, False, current_time)
        self._push_prefill_ddls(reqs)
        
        self._get_requirements(new_reqs, True, current_time)
        
        admitted_reqs = []
        declined_reqs = []
        
        if len(new_reqs):
            with gp.Env(params = options) as env, gp.Model(env=env) as model:
                keep_requests = self.solve(model, 
                                [req._sch_requirement for req in reqs],
                                [req._sch_requirement for req in new_reqs])
        
            for req, keep in zip(new_reqs, keep_requests):
                if keep: 
                    admitted_reqs.append(req)
                    assert self.memory.optional_alloc(req.req)
                else: 
                    req.decline()
                    declined_reqs.append(req)
            
        reqs = reqs + admitted_reqs

        if not len(reqs):
            return [BatchSchedule([])], [], new_reqs
            
        reqs = sorted(reqs, key = lambda req: req._sch_requirement.prefill_ddl)
        
        self._advance_prefill_ddls(reqs)        
        
        last_ddl = reqs[-1]._sch_requirement.prefill_ddl
        
        n_schs = last_ddl + self.n_sch_after_last
        allocated_token = np.zeros((n_schs,), dtype = np.int32)
        
        
        self.sch_cnt += 1
        req_schss = [[] for _ in range(n_schs)]

        '''
        Schedule
        '''

        for req in reqs:
            for j in range(req._sch_requirement.prefill_ddl, n_schs, req.req.sla.tpot_slown_rate):
                for k in range(min(j + req.req.sla.tpot_slown_rate, n_schs) - 1, j - 1, -1):
                    if allocated_token[k] < self.bs:
                        allocated_token[k] += 1
                        req_schss[k].append(ReqBatchSchedule(
                            req.id, 
                            False,
                            1
                        ))
                        break
        for req in reqs:
            remain_prefills = req._sch_requirement.n_prefill_token
            for i in range(req._sch_requirement.prefill_ddl):
                scheduled_tokens = min(self.bs - allocated_token[i], remain_prefills)
                remain_prefills -= scheduled_tokens
                allocated_token[i] += scheduled_tokens
                if scheduled_tokens > 0:
                    req_schss[i].append(ReqBatchSchedule(
                        req.id, 
                        True,
                        int(scheduled_tokens)
                    ))
                if remain_prefills == 0: break
                
            if remain_prefills > 0:
                print('remain_prefills', remain_prefills)
                # model.write('problem.lp')

            assert (remain_prefills == 0)
            
        return [BatchSchedule(req_schs) for req_schs in req_schss], admitted_reqs, declined_reqs

class MaxThroughputScheduler(ScheduleALG):
    def __init__(self, mem: MemoryInstance): 
        super(MaxThroughputScheduler, self).__init__(mem)

    
    def schedule(self, reqs: List[RequestInstanceBase], new_reqs: List[RequestInstanceBase], current_time: float) \
        -> Tuple[List[BatchSchedule], List[RequestInstanceBase], List[RequestInstanceBase]]:
        
        new_reqs, declined_reqs = self._decline_requests_mem(new_reqs)
        
        schs: List[ReqBatchSchedule] = []
        for req in reqs + new_reqs:
            if req.is_prefill():
                schs.append(ReqBatchSchedule(req.id, 
                                                is_prefill = True,
                                                n = req.input_length))
            else: 
                schs.append(ReqBatchSchedule(req.id, is_prefill = False, n = 1))
        
        return [BatchSchedule(schs)], new_reqs, declined_reqs
    
class MaxThroughputSpecScheduler(ScheduleALG):
    def __init__(self, mem: MemoryInstance, 
                 spec_decode: SpecDecode,
                 batch_timer: BatchTimer): 
        super(MaxThroughputSpecScheduler, self).__init__(mem)
        self.spec_decode = spec_decode
        self.batch_timer = batch_timer
    
    def _get_optimal_sd_size(self, n_decode: int, n_prefill_tokens: int):
        sd_size = 1
        best_tpt = 1e-9
        for spec_decode_size in range(1, self.spec_decode.max_spec_decode + 1):
            t = self.batch_timer(spec_decode_size * n_decode + n_prefill_tokens, spec_decode_size)
            tpt = (self.spec_decode.exp(spec_decode_size) * n_decode + n_prefill_tokens) / t 
            if tpt > best_tpt:
                best_tpt = tpt
                sd_size = spec_decode_size
        return sd_size
    
    def schedule(self, reqs: List[RequestInstanceBase], new_reqs: List[RequestInstanceBase], current_time: float) \
        -> Tuple[List[BatchSchedule], List[RequestInstanceBase], List[RequestInstanceBase]]:
        
        new_reqs, declined_reqs = self._decline_requests_mem(new_reqs)
        
        schs: List[ReqBatchSchedule] = []
        
        n_decode = len(reqs)
        
        prefill_tokens = sum(req.input_length for req in reqs if new_reqs)
        
        sd_size = self._get_optimal_sd_size(n_decode, prefill_tokens)
        
        for req in reqs + new_reqs:
            if req.is_prefill():
                schs.append(ReqBatchSchedule(req.id, 
                                                is_prefill = True,
                                                n = req.input_length))
            else: 
                schs.append(ReqBatchSchedule(req.id, is_prefill = False, n = sd_size))
        
        return [BatchSchedule(schs)], new_reqs, declined_reqs

class PromaxAdaSpecSchedulerBF(ScheduleALG):
    def __init__(self,
                 mem: MemoryInstance,  
                 batch_timer: BatchTimer,
                 spec_decode: SpecDecode,
                 tpot: float,
                 t_sch_after: float = 5e-2,):
        super(PromaxAdaSpecSchedulerBF, self).__init__(mem)
        self.batch_timer = batch_timer
        self.spec_decode = spec_decode
        self.mtx_01 = np.array([[0], [1]])
        self._gen_01_mtx(10)
        self.tpot = tpot
        self.t_sch_after = t_sch_after
        self.opt_tpts = [self.batch_timer.get_max_throughput()]
        self.opt_sds = [0] # best speculative decoding size
        print('self.tpot', self.tpot)
        
    def _gen_01_mtx(self, n):
        while n > self.mtx_01.shape[-1]:
            self.mtx_01_top = np.concatenate([self.mtx_01, np.ones(shape = (self.mtx_01.shape[0], 1))], axis = -1)
            self.mtx_01_bot = np.concatenate([self.mtx_01, np.zeros(shape = (self.mtx_01.shape[0], 1))], axis = -1)
            self.mtx_01 = np.concatenate([self.mtx_01_top, self.mtx_01_bot], axis = 0)  
    
    def _get_best_tpts_and_sd(self, n: int):
        for i in range(len(self.opt_tpts), n + 1):
            prefill_tpts = [
            (self.batch_timer.time_to_token(self.spec_decode.exp(sd) * self.tpot) - i*sd)\
            / (self.spec_decode.exp(sd) * self.tpot)\
                for sd in range(1, self.spec_decode.max_spec_decode + 1)]
            idx = np.argmax(prefill_tpts)
            self.opt_tpts.append(prefill_tpts[idx])
            self.opt_sds.append(int(idx) + 1)
    
    @dataclass
    class ReqInfo:
        id: int
        is_new_req: bool
        idx: int
        n_prefill_left: int 
        prefill_ddl: float
        num_block: int 
        profit: float

    def schedule(self, 
                 reqs: List[RequestInstanceBase], 
                 new_reqs: List[RequestInstanceBase], 
                 current_time: float) \
        -> Tuple[List[BatchSchedule], List[RequestInstanceBase], List[RequestInstanceBase]]:
        
        if len(reqs) + len(new_reqs) == 0:
            return [BatchSchedule([])], [], []
        
        print(f'schedule {len(reqs)} reqs, new_reqs {len(new_reqs)}')
        
        self._gen_01_mtx(len(new_reqs))
        
        req_infos = [PromaxAdaSpecSchedulerBF.ReqInfo(
            req.id,
            i >= len(reqs), 
            i - len(reqs) if i >= len(reqs) else i, 
            max(req.input_length - req._n_prefill_tokens, 0),
            max(req.req.ttft() + req.arrive_time - current_time, 0.),
            req.req.num_blocks,
            req.req.profit()
        ) for i, req in enumerate(reqs + new_reqs)]
            
        req_infos = sorted(req_infos, key = lambda info: info.prefill_ddl)
        indices = [i for i, info in enumerate(req_infos) if info.is_new_req]
        
        keep_requests = np.ones((2**len(new_reqs), len(req_infos)), dtype = np.int32)
        keep_requests[:, indices] = self.mtx_01[:2**len(new_reqs), :len(new_reqs)]
        
        n_reqs = np.cumsum(keep_requests, axis = -1).astype(np.int32)[:, :-1]
        n_reqs = np.concatenate([np.zeros((n_reqs.shape[0], 1), dtype = np.int32), n_reqs], axis = -1)
        
        # print('tpot', tpot)
        self._get_best_tpts_and_sd(len(req_infos))

        # tpts[i] the throughput between t[i-1] and t[i] (the)
        tpt_mtx = np.array(self.opt_tpts)[n_reqs]
        n_tokens_required = np.array([info.n_prefill_left for info in req_infos])
        delta_times = n_tokens_required / tpt_mtx
        times = np.cumsum(delta_times, axis = -1)
        ddls = np.array([info.prefill_ddl for info in req_infos])
        ddls = keep_requests * ddls

        # print('n_tokens_required', n_tokens_required)
        prefill_valid = np.all(times <= ddls, axis = -1)
        mem_required = np.array([info.num_block for info in req_infos])
        profits_per_req = np.array([info.profit for info in req_infos])
        profits = keep_requests @ profits_per_req
        mem_required = keep_requests @ mem_required
        mem_valid = mem_required <= self.memory.get_n_avail()
        decode_constr = np.min(tpt_mtx, axis = -1) >= 0

        profits_valid = prefill_valid * mem_valid * profits * decode_constr
        best_index = np.argmax(profits_valid)
        if profits_valid[best_index] == 0.:
            best_index = 0
        keep_request = keep_requests[best_index]
        indices = [i for i, keep_req in enumerate(keep_request) if keep_req]
        timepoints = times[best_index, indices]
        kept_reqs: List[PromaxAdaSpecSchedulerBF.ReqInfo] = [req_infos[i] for i in indices]
        
        batches: List[BatchSchedule] = []

        admitted_reqs = [new_reqs[info.idx] \
            for info, keep_req in zip(req_infos, keep_request) if keep_req and info.is_new_req]
        declined_reqs = [new_reqs[info.idx] \
            for info, keep_req in zip(req_infos, keep_request) if not keep_req and info.is_new_req]
                    
        idx = 0
        start_time = 0
        while idx < len(kept_reqs):
            # print(f'kept_reqs[{idx}]',kept_reqs[idx])
            # Find the first request that is different from start time
            while idx < len(kept_reqs) and\
                timepoints[idx] < start_time + 1e-6:
                # all requests should finish the prefill phase
                assert kept_reqs[idx].n_prefill_left == 0
                idx += 1
            
            if idx == len(kept_reqs):
                end_time = start_time + self.t_sch_after
            else: 
                end_time = timepoints[idx]

            if idx == 0:
                unit_time = end_time
                # set a large enough prefill budget
                n_prefill_tokens = 10000
            else: 
                best_sd = self.opt_sds[idx]
                unit_time = self.tpot * self.spec_decode.exp(best_sd)
                n_prefill_tokens = self.batch_timer.time_to_token(unit_time)\
                    - idx * best_sd
            # print('idx', idx, 'start_time', start_time, 'end_time', end_time, 'unit_time', unit_time)
            # print('prefill reqs', kept_reqs[idx:])
            # print('decode reqs', kept_reqs[:idx])
            # print('len(batches)', len(batches))
                        
            for _ in np.arange(start_time, end_time, unit_time):
                req_schs = [
                    ReqBatchSchedule(req.id, False, best_sd)\
                        for req in kept_reqs[:idx]
                ]
                n_prefill_tokens_cur_batch = n_prefill_tokens
                for info in kept_reqs[idx:]:
                    n_scheduled = min(n_prefill_tokens_cur_batch, info.n_prefill_left)
                    if n_scheduled > 0: 
                        req_schs.append(ReqBatchSchedule(info.id, True, n_scheduled))
                    n_prefill_tokens_cur_batch -= n_scheduled
                    info.n_prefill_left -= n_scheduled

                batches.append(BatchSchedule(req_schs))
            
            start_time = end_time

        return batches, admitted_reqs, declined_reqs

@dataclass
class ReqInfo:
    req: RequestInstanceBase
    is_new_req: bool
    n_prefill_left: int
    prefill_ddl: float
    num_block: int
    profit: float
    n_prefill_left_rt: int = field(init = False)
    def __post_init__(self):
        self.n_prefill_left_rt = self.n_prefill_left
    
class PromaxAdaSpecSchedulerMILP(ScheduleALG):
    def __init__(self,
                 mem: MemoryInstance,  
                 batch_timer: BatchTimer,
                 spec_decode: SpecDecode,
                 tpot: float,
                 max_spec_decode: int):
        super(PromaxAdaSpecSchedulerMILP, self).__init__(mem)
        self.batch_timer = batch_timer
        self.spec_decode = spec_decode
        self.tpot = tpot
        self.max_spec_decode = max_spec_decode
        self.spec_decode_sizes = range(1, self.max_spec_decode + 1)
        print('self.tpot', self.tpot)
    
    def _solve(
        self,
        m: gp.Model,
        infos: List[ReqInfo],
        current_time: float 
    ):
        rs = []
        n_old_reqs = 0
        n_resource_used = 0
        n_mem = 0
        
        remained_tokens = []
        resource_used = []
        memory_used = []
        profits = []
        
        last_prefill_ddl = current_time
        for req_id, info in enumerate(infos):
            prefill_tpt_vars = m.addVars(len(self.spec_decode_sizes), name = f'prefill_tpt_vars-{req_id}')
            for i, sd in enumerate(self.spec_decode_sizes):
                if n_old_reqs > 0:
                    tpt_tot = min(1 / k - b / self.tpot / k / self.spec_decode.exp(sd)
                                for k, b in [(self.batch_timer.k1, self.batch_timer.b1),
                                (self.batch_timer.k2, self.batch_timer.b2)])
                elif len(rs) == 0:
                    tpt_tot = MAX_BS / self.batch_timer(MAX_BS)
                else:
                    r_max = m.addVar(vtype = GRB.BINARY, name = f'r_max-{req_id}-{sd}')
                    m.addConstr(r_max == gp.max_(rs))
                    tpt_vars = m.addVars(2, name = f'linear-vars-{req_id}-{sd}')
                    for tpt_var, k, b in [(tpt_vars[0], self.batch_timer.k1, self.batch_timer.b1),
                                (tpt_vars[1], self.batch_timer.k2, self.batch_timer.b2)]:
                        m.addConstr(tpt_var == 1/k - r_max * b /(k*self.tpot*self.spec_decode.exp(sd)))                    
                    tpt_tot = m.addVar(name = f'tpt-req-{req_id}-sd-{sd}')
                    m.addConstr(tpt_tot == gp.min_(tpt_vars))
                # print('tpt_tot', tpt_tot, 'sd', sd, 'rs', rs, 'n_old_reqs', n_old_reqs)
                m.addConstr(prefill_tpt_vars[i] == tpt_tot - sd * (gp.quicksum(rs) + n_old_reqs) / self.spec_decode.exp(sd) / self.tpot) 
            prefill_tpt = m.addVar(name = f'prefill-tpt-{req_id}')
            m.addConstr(prefill_tpt == gp.max_(prefill_tpt_vars))
            
            if info.is_new_req:
                r = m.addVar(vtype = GRB.BINARY, name = f'r_{req_id}')
                rs.append(r)
                resource_used.append(r * info.n_prefill_left)
                memory_used.append(r * info.num_block) 
                profits.append(r * info.profit)
            else:
                n_old_reqs += 1
                n_resource_used += info.n_prefill_left
                n_mem += info.num_block
            
            remained_tokens.append(prefill_tpt * (info.prefill_ddl - last_prefill_ddl))
            m.addConstr(gp.quicksum(remained_tokens) >= (gp.quicksum(resource_used) + n_resource_used))
            last_prefill_ddl = info.prefill_ddl
            
        m.addConstr(gp.quicksum(memory_used) + n_mem <= self.memory.get_n_avail())
        m.setObjective(gp.quicksum(profits), GRB.MAXIMIZE)
        
        
        # print('problem creation takes', time.perf_counter() - start, 's')
        # start = time.perf_counter()
        m.optimize()
        # print('optimize takes', time.perf_counter() - start, 's')
        m.write('problem.lp')
        m.write("problem.json")
        
        # print('rs', rs)s
        if m.Status == GRB.INFEASIBLE:
            return False, [False for _ in range(len(rs))]
        try: 
            return True, [r.X for r in rs]
        except:
            print('no solution found')
        return False, [False for _ in range(len(rs))]
    
    def _get_best_config(
        self, 
        n_reqs: int,
    ):
        if n_reqs == 0:
            return 0, 2048, 2048 / self.batch_timer(2048)

        configs = []
        for sd in self.spec_decode_sizes:
            t = self.tpot * self.spec_decode.exp(sd)
            bs = self.batch_timer.reverse(t)
            prefill_tpt = (bs - n_reqs * sd) / t
            configs.append((sd, bs, prefill_tpt))

        sd, bs, tpt = max(configs, key = lambda x: x[-1])

        return sd, bs, tpt
    
    def _solution_to_sch(
        self,
        infos: List[ReqInfo],
        is_admitted: List[bool],
        current_time: float,
        feasible: bool
    ):
        admitted_reqs = []
        reject_reqs = []
        idx = 0
        prev_time = current_time
        remains = []
        tpts = []
        bss = []
        sds = []
        deltas = []
        admitted_infos: List[ReqInfo] = []
        n_reqs = 0
        sum_resources = 0
        sum_prefills = 0
        
        for info in infos:
            if info.is_new_req:
                (admitted_reqs if is_admitted[idx] else reject_reqs).append(info.req)
                idx += 1
                if not is_admitted[idx-1]:
                    continue
            admitted_infos.append(info)
            
            sd, bs, tpt = self._get_best_config(n_reqs)
            # delta = max((sum_prefills + info.n_prefill_left - sum_resources) / tpt, info.prefill_ddl - prev_time)
            delta = info.prefill_ddl - prev_time
            sum_resources += tpt * delta
            sum_prefills += info.n_prefill_left
            print('sd', sd, 'bs', bs, 'tpt', tpt, 'delta', delta, 'n_prefill_left', info.n_prefill_left, 'sum_resources', sum_resources, 'sum_prefills', sum_prefills)
            if feasible:
                if not sum_resources >= sum_prefills:
                    print('sum resources', sum_resources, 'sum_prefills', sum_prefills) 
                assert sum_resources + 1 >= sum_prefills
            remains.append(sum_resources - sum_prefills)
            tpts.append(tpt)
            sds.append(sd)
            bss.append(bs)
            
            n_reqs += 1
            deltas.append(delta)
            prev_time = info.prefill_ddl
            
        # print('tpts', tpts, 'deltas', deltas, 'sds', sds, 'remains', remains, 'bss', bss)
        # print('prefill_lefts', [info.n_prefill_left for info in admitted_infos])
        
        remains = np.array(remains)
        # assert np.all(remains >= 0)
        
        schs: List[BatchSchedule] = []
        idx = 0
        n_prefill_resource = 0
        n_prefill_used = 0
        for i, (info, tpt, bs, delta, sd) in enumerate(zip(admitted_infos, tpts, bss, deltas, sds)):
            # print('remains', remains[i:], 'tpt:', tpt, 'delta', delta)
            if remains[i] < 0: # when the bound cannot be satisfied, we losen the bound.
                gap = remains[i] / tpt
                remains[i:] -= gap * tpt
                delta -= gap
            # elif remains[i:].min() > 0: # when the bound is loose, we push the bound tighter 
            #     gap = min(remains[i:].min() / tpt, delta)
            #     remains[i:] -= gap * tpt
            #     delta -= gap
            n_prefill_resource += tpt * delta
            n_prefill_used += info.n_prefill_left
            # assert n_prefill_resource >= n_prefill_used
            
            
            print('idx', idx, 'bs', bs, 'tpt', tpt, 'delta', delta, 'sd', sd, 'n_prefill_resource', n_prefill_resource, 'n_prefill_used', n_prefill_used,)
            assert n_prefill_resource + 1e-3 >= n_prefill_used
            
            # if i == 0 and delta > max(self.batch_timer.b1, self.batch_timer.b2):
            #     bs = min(bs, self.batch_timer.reverse(delta))
            
            bs = math.ceil(bs / 128) * 128
            n_prefill_tokens = math.ceil(tpt * delta)
            # tpt_bs =  (bs - i * sd) / self.batch_timer(bs)
            # # print('bs', bs, 'tpt_bs', tpt_bs, 'tpt', tpt)
            # assert bs == 2048 or tpt_bs >= tpt
            # n_batches = math.floor((tpt * delta) / (bs - i * sd))
            
            while idx < len(admitted_infos) and admitted_infos[idx].n_prefill_left_rt == 0:
                idx += 1
            
            while n_prefill_tokens > 0 and idx < len(admitted_infos):
                if idx == len(admitted_infos): break
                req_schs: List[ReqBatchSchedule] = []
                for j in range(i):
                    req_schs.append(ReqBatchSchedule(admitted_infos[j].req.id, False, sd))
                cur_bs = bs - i * sd
                while idx < len(admitted_infos) and cur_bs and n_prefill_tokens:
                    scheduled = min(cur_bs, admitted_infos[idx].n_prefill_left_rt, n_prefill_tokens)
                    admitted_infos[idx].n_prefill_left_rt -= scheduled
                    cur_bs -= scheduled
                    n_prefill_tokens -= scheduled
                    if scheduled > 0:
                        req_schs.append(ReqBatchSchedule(admitted_infos[idx].req.id, True, scheduled))
                    if admitted_infos[idx].n_prefill_left_rt == 0:
                        idx += 1

                schs.append(BatchSchedule(req_schs))
            # after the above scheduling, i'th request's prefill must be done.
            # if idx <= i:
            #     print('bss', bss, 'tpts', tpts, 'deltas', deltas, 'sds', sds, 'remains', remains)
            #     print('prefill_lefts', [info.n_prefill_left_rt for info in admitted_infos])
            #     raise RuntimeError
            # TODO: analyze the reason for extra delay
            req_schs = []
            while idx <= i:
                info = admitted_infos[idx]
                if info.n_prefill_left_rt:
                    req_schs.append(
                        ReqBatchSchedule(info.req.id, True, 
                        info.n_prefill_left_rt))
                info.n_prefill_left_rt = 0                
                idx += 1
            if len(req_schs):
                print('WARNING: Extra Prefill batch ', sum(req.n for req in req_schs))
                # if not len(schs):
                schs.append(BatchSchedule(req_schs))
                # else: schs[-1].reqs.extend(req_schs)
        
        # one last batch for all decoding
        sd, bs, tpt = self._get_best_config(len(admitted_infos))
        schs.append(BatchSchedule(
            reqs = [ReqBatchSchedule(info.req.id, False, sd) for info in admitted_infos],
            next = 0))
        
        return schs, admitted_reqs, reject_reqs

    def schedule(self,
                 reqs: List[RequestInstanceBase], 
                 new_reqs: List[RequestInstanceBase], 
                 current_time: float)\
    -> Tuple[List[BatchSchedule],  List[RequestInstanceBase], List[RequestInstanceBase]]:
        
        infos = [ReqInfo(
            req,
            i >= len(reqs), 
            req.input_length - req._n_prefill_tokens,
            max(req.req.ttft() + req.arrive_time, current_time),
            req.req.num_blocks,
            req.req.profit()
        ) for i, req in enumerate(reqs + new_reqs)]
            
        infos = sorted(infos, key = lambda info: info.prefill_ddl)
        
        # start = time.perf_counter()
        with gp.Env(params = options) as env, gp.Model(env=env) as model:
            feasible, is_admitted = self._solve(
                model, 
                infos,
                current_time
            )
            print('feasible', feasible, 'is_admitted', is_admitted)
        # print('solve takes', time.perf_counter() - start, 's')
        
        return self._solution_to_sch(infos, is_admitted, current_time, feasible)

class PromaxSpecSchedulerDP(ScheduleALG):
    def __init__(
        self,
        memory: MemoryInstance,
        prob: Problem,
    ):
        super(PromaxSpecSchedulerDP, self).__init__(memory)
        assert prob.batch_timer is not None
        tpots = set()
        for sla in sum(prob.request_trace.slas, start = []):
            if sla.tpot_per_token is not None:
                tpots.add(sla.tpot_per_token)
            elif sla.tpot_slown_rate is not None: 
                tpots.add(sla.tpot_slown_rate * prob.batch_timer(0))
            if sla.tpot_per_token_thinking is not None:
                tpots.add(sla.tpot_per_token_thinking)
        
        # for i in range(len(self.tpots) - 1):
        #     assert self.tpots[i] < self.tpots[i+1]
        # assert prob.best_effort_sla.tpot_per_token is not None
        # self.tpots.append(prob.best_effort_sla.tpot_per_token)
        tpots = sorted(list(tpots))
        self.tpots = [x - 0.001 for x in tpots]
        self.tpots.append(self.tpots[-1] + 0.1)
       
        self.hardware_params = prob.batch_timer.params
        self.spec_decode_alpha = prob.spec_decode.alpha
        self.max_spec_decode_size = prob.max_spec_decode_size
        self.fixed_bs = prob.fixed_bs
        self.fixed_spec = prob.fixed_spec
        
        self.scheduler = promax.PromaxScheduler()
        if prob.spec_model is None:
            self.scheduler.set_ar_planner(
                tpots = self.tpots, 
                hardware_params = self.hardware_params,
                fixed_bs = self.fixed_bs
            )
        else:
            self.scheduler.set_sd_planner(
                tpots = self.tpots,
                hardware_params = self.hardware_params,
                fixed_bs = self.fixed_bs,
                alpha = self.spec_decode_alpha,
                max_sd_size = self.max_spec_decode_size,
                fixed_spec = self.fixed_spec
            )
        
        self.infeasible_cnt = 0
        print('PromaxSpecSchedulerDP, tpots:', self.tpots, 
              'hardware_params', self.hardware_params, 
              'spec_decode_alpha', self.spec_decode_alpha,
              'max_spec_decode', self.max_spec_decode_size,
              'fixed_bs', self.fixed_bs, 
              'fixed_spec', self.fixed_spec)
        self.i = 0
        
    def _get_tpot_idx(self, t: float) -> int:
        idx = len(self.tpots) - 1
        while idx >= 0:
            if self.tpots[idx] <= t: 
                return idx
            idx -= 1
        return 0
        # raise RuntimeError
    
    def schedule(self, reqs: List[RequestInstanceBase], 
                 new_reqs: List[RequestInstanceBase], 
                 current_time: float) \
        -> Tuple[List[BatchSchedule], List[RequestInstanceBase], List[RequestInstanceBase]] | None:
        if len(reqs) + len(new_reqs) == 0:
            return [], [], []
        
        c_reqs = [
            promax.Request(id = req.id,
                           is_new_req = i >= len(reqs),
                           ddl = max(req.req.ttft() + req.arrive_time, current_time) if req.is_prefill() else current_time,
                           input_length = max(req.input_length - req._n_prefill_tokens, 0),
                           profit = req.req.profit(),
                           mem = req.req.num_blocks,
                           tpot_idx = self._get_tpot_idx(req.tpot(current_time)))
            for i, req in enumerate(reqs + new_reqs)
        ]

        # self.infeasible_cnt += 1
        # if len(c_reqs) > 40:
        #     with open(f'errors/error{self.i}.in', 'w') as f:
        #         self.i += 1
        #         to_str = lambda l: ' '.join([str(x) for x in [len(l)] + l]) + '\n'
        #         f.write(to_str(self.tpots))
        #         f.write(to_str(self.hardware_params))
        #         f.write(f'{self.spec_decode_alpha} {self.max_spec_decode_size}\n')
        #         f.write(f'{self.memory.get_n_avail()} {current_time} {len(c_reqs)}\n')
        #         for r in c_reqs:
        #             f.write(f'{r.id} {int(r.is_new_req)} {r.ddl} {r.input_length} {r.profit} {r.mem} {r.tpot_idx}\n')

            # with open(f'errors/error{self.i:03}.in', 'w') as f:
            #     self.i += 1
            #     to_str = lambda l: ' '.join([str(x) for x in [len(l)] + l]) + '\n'
            #     f.write(to_str(self.tpots))
            #     f.write(to_str(self.hardware_params))
            #     f.write(f'{self.spec_decode_alpha} {self.max_spec_decode_size}\n')
            #     f.write(f'{self.memory.get_n_avail()} {current_time} {len(c_reqs)}\n')
            #     for r in c_reqs:
            #         f.write(f'{r.id} {int(r.is_new_req)} {r.ddl} {r.input_length} {r.profit} {r.mem} {r.tpot_idx}\n')

        start = time.perf_counter()
        
        is_feasible, acc_ids, batches = self.scheduler.schedule(
            reqs = c_reqs,
            M = self.memory.get_n_avail(remove_best_effort=True),
            current_time = current_time,
            verbose = False
        )
        
        elasped = time.perf_counter() - start
        
        if elasped > 0.5:
            with open(f'errors/error.in', 'w') as f:
                self.i += 1
                to_str = lambda l: ' '.join([str(x) for x in [len(l)] + l]) + '\n'
                f.write(to_str(self.tpots))
                f.write(to_str(self.hardware_params))
                f.write(f'{self.spec_decode_alpha} {self.max_spec_decode_size}\n')
                f.write(f'{self.memory.get_n_avail()} {current_time} {len(c_reqs)}\n')
                for r in c_reqs:
                    f.write(f'{r.id} {int(r.is_new_req)} {r.ddl} {r.input_length} {r.profit} {r.mem} {r.tpot_idx}\n')
            exit(0)
        
        # if not is_feasible:
        #     self.infeasible_cnt += 1
        
        # exit(0)
        
        
        
        if not is_feasible: 
            return None
        
        # except Exception as e:
        #     print(f"An error occurred: {e}")
            
        batches = [BatchSchedule(
            [ReqBatchSchedule(
                req_sch.id, 
                req_sch.is_prefill,
                req_sch.n) for req_sch in batch.req_batches], 
                next = batch.next,
                remain_budget=batch.prefill_bs) for batch in batches]
        
        if len(batches):
            assert batches[-1].next <= 0
        
        acc_ids = set(acc_ids)
        
        acc_reqs, declined_reqs = [], []
        for req in new_reqs:
            if req.id in acc_ids:
                acc_reqs.append(req)
            else: 
                declined_reqs.append(req)
                
        return batches, acc_reqs, declined_reqs

def get_schedule_alg(tag: str, problem: Problem, mem: MemoryInstance) -> ScheduleALG:
    if tag == 'vllm':
        alg = vLLMScheduler(mem)
    elif tag == 'sarathi':
        alg = SarathiServeScheduler(mem, problem.batch_timer)
    # elif tag == 'slo':
    #     bs = problem.batch_timer.shift_bs
    #     print('B for SLO scheduler:', bs)
    #     alg = SLOAwareScheduler(
    #         mem,
    #         bs,
    #         problem.batch_timer(bs), 10)
    elif tag == 'oracle':
        alg = OracleScheduler(mem, problem)
    elif tag == 'tpt':
        alg = MaxThroughputScheduler(mem)
    elif tag == 'tpt-spec':
        assert problem.batch_timer is not None
        alg = MaxThroughputSpecScheduler(mem, problem.spec_decode, problem.batch_timer)
    elif tag == 'promax-milp':
        assert len(problem.sla_distribution) == 1
        alg = PromaxAdaSpecSchedulerMILP(mem, 
                                     problem.batch_timer, 
                                     problem.spec_decode,
                                     tpot = problem.batch_timer(0) * problem.sla_distribution[0][1].tpot_slown_rate,
                                     max_spec_decode = problem.max_spec_decode_size)
    elif tag == 'promax-dp':
        alg = PromaxSpecSchedulerDP(mem, problem)
    else: raise NotImplementedError(f"unknown tag {tag}")
    return alg

'''
The problem w/ promax scheduler is it has exponential complexity. 
The problem w/ knapsack scheduler is that it cannot model the change in sizes. 

'''