import asyncio.queues
from typing import List, Dict
from dataclasses import dataclass, field
import asyncio
import numpy as np
import tqdm
import warnings
import ray
import copy
from .utils import Timer
from .schedule_algs import get_schedule_alg
from ..engine_wrapper import EngineWrapper
from .struct import (
    Request,
    BatchSchedule,
    BatchTimer,
    RequestResult,
    SchedulerConfig,
    ExecutionResult,
    RequestInstanceBase,
    RequestState,
    Schedule,
    RequestInitMsg,
    EOS,
    REJ,
    MemoryInstance,
    Problem,
    ReqBatchSchedule,
    RevokeMsg
)
from .schedule_algs import (
    ScheduleALG,
)

@dataclass 
class RequestInstance(RequestInstanceBase):
    _output_queue: asyncio.queues.Queue = field(default_factory = asyncio.queues.Queue)
    _generated_text: str = ''
    do_best_effort: bool = True
    
    # async def async_decline(self):
    #     await self._output_queue.put(EOS)
    
    def decline(self):
        self._output_queue.put_nowait(REJ)
                
    def commit(self, 
                req_res: RequestResult, 
                current_time: float):
        # assert self.input_length + self.output_length <= 2048
        assert req_res.id == self.id
        if req_res.is_prefill:
            assert self.state in (RequestState.Prefill, RequestState.Arrived)
            self.state = RequestState.Prefill
            self._n_prefill_tokens += req_res.n
            assert self._n_prefill_tokens <= self.input_length
            if self._n_prefill_tokens == self.input_length:
                assert req_res.n_generated == 1
                self._n_decode_tokens += req_res.n_generated
                self._generated_text += req_res.generated_text
                self._output_queue.put_nowait(req_res.generated_text)
                self._timestamps.append((self._n_decode_tokens, current_time))
                self.state = RequestState.Decode
        else:
            if not self.state == RequestState.Decode:
                print(self.state)
                return False
            self._n_decode_tokens += req_res.n_generated
            self._generated_text += req_res.generated_text
            self._timestamps.append((self._n_decode_tokens, current_time))
            self._spec_history.append((req_res.n, req_res.n_generated))
            self._output_queue.put_nowait(req_res.generated_text)
        if self._n_decode_tokens >= self.output_length or req_res.is_finished:
            self._output_queue.put_nowait(EOS)
            self.state = RequestState.Finished
        return True
    def __repr__(self):
        return super(RequestInstance, self).__repr__()
'''
add one tier of requests that is best effort serving w/ preemption.

First, we need to have a best effort serving tier that is managed separately.
'''
@dataclass
class Scheduler:
    id: int
    engines: List[ray.ObjectRef]
    config: SchedulerConfig
    problem: Problem
    memory: MemoryInstance
    
    alg: ScheduleALG = field(init = False)
    
        
    timer: Timer = field(default_factory=Timer)
    req_cnt: int = 0
    all_reqs: Dict[int, RequestInstance] = field(default_factory=dict)
    new_reqs: List[RequestInstance] = field(default_factory=list)
    running_reqs: List[RequestInstance] = field(default_factory=list)
    declined_reqs: List[RequestInstance] = field(default_factory=list)
    running_best_effort_reqs: List[RequestInstance] = field(default_factory=list)
    scheduled_batches: List[BatchSchedule] = field(default_factory=list)
    stop: bool = False
    schedule_time: float = 0.
    last_arrive_time: float = 0.
    time_skew = 0.
    
    def __post_init__(self):
        self.alg = get_schedule_alg(self.config.sch_strategy, self.problem, self.memory)
    
    def add_new_req(self, 
                    id: int,
                    req: Request, 
                    time_skew: float,
                    arrive_time = None,
                    do_best_effort = True) -> asyncio.queues.Queue:
        self.time_skew = time_skew
        # assert req.input_length + req.output_length <= 2048
        new_req = RequestInstance(id = id, 
            req = req, 
            arrive_time = arrive_time or self.timer.current_time(),
            time_skew = time_skew,
            do_best_effort=do_best_effort)
        self.req_cnt += 1
        self.all_reqs[id] = new_req
        if req.sla.is_best_effort:
            assert self.config.enable_best_effort
            self.declined_reqs.append(new_req)
            
        else:
            self.new_reqs.append(new_req)
        self.last_arrive_time = self.timer.current_time()

        return new_req._output_queue
                
        
    async def async_schedule(self):
        self.schedule_overhead = 0
        n_declined = 0
        n_accepted = 0
        n_arrived = 0
        tot_n_finished = 0
        n_preempt = 0
        
        assert self.problem.batch_timer is not None
        pbar = tqdm.tqdm(desc = 'scheduler')
        batches = []
        bid = 0
        self.failure_cnt = 0
        self.n_schedules = 0
        self.timer.start()
        self.declined_by_failure = 0
        
        while not self.stop:
            self.timer('before everything')
            n_arrived += len(self.new_reqs)
            n_new_reqs = len(self.new_reqs)
            pbar.update(1)

            if pbar.n == 10000:
                print('stop here')
                
            # pbar.set_description(f'schedule #new {n_new_reqs} #finshed {tot_n_finished} #old {len(self.running_reqs)} #declined {len(self.declined_reqs)}, #running best effort {len(self.running_best_effort_reqs)} #preempt {n_preempt} mem in use {self.memory.occupancy()}')
            self.timer('before schedule')
            sch_start = self.timer.current_time()
            
            # if len(self.new_reqs) == 0 and self.config.sch_strategy == 'promax-dp':
            #     sch = (batches[bid:], [], [])
            # else:
            sch = self.alg.schedule(
                self.running_reqs,
                self.new_reqs,
                self.timer.current_time()
            )
            # self.schedule_overhead = self.timer.current_time() - self.schedule_time
            self.timer('schedule')
            self.n_schedules += 1
            
            if sch is None:
                admitted_reqs = []
                declined_reqs = self.new_reqs 
                self.failure_cnt += 1
                self.declined_by_failure += len(declined_reqs)
            else: 
                batches, admitted_reqs, declined_reqs = sch
                bid = 0
                       
            assert (len(admitted_reqs) + len(declined_reqs)) == n_new_reqs
            self.schedule_overhead += self.timer.current_time() - sch_start
            
            # n_declined += len(declined_reqs)
            
            for req in declined_reqs:
                if not req.do_best_effort:
                    req.decline()
                    n_declined += 1
                else: 
                    self.declined_reqs.append(req)
        
            self.running_reqs.extend(admitted_reqs)
            self.new_reqs = []
            
            # if not (tot_n_finished \
            #         + n_declined \
            #         + len(self.running_reqs)\
            #         + len(self.running_best_effort_reqs)\
            #         + len(self.declined_reqs)\
            #         + len(self.new_reqs)) == self.req_cnt:
            #     print('HERE req_cnt', self.req_cnt, 
            #           '#finished', tot_n_finished, 
            #           '#r', len(self.running_reqs), 
            #           '#best effort', len(self.running_best_effort_reqs), 
            #           'declined', len(self.declined_reqs),
            #           '#new', len(self.new_reqs))
            #     raise RuntimeError
            
            
            if self.config.enable_best_effort:
                n_block = sum(req.req.num_blocks for req in admitted_reqs)
                preempt_reqs = []
                while n_block > self.memory.get_n_avail() \
                    and len(self.running_best_effort_reqs):
                    req = self.running_best_effort_reqs.pop()
                    assert not req.is_finished()
                    preempt_reqs.append(req.id)
                    for batch in batches: 
                        batch.reqs = [req_sch for req_sch in batch.reqs if req_sch.id != req.id]
                        
                    assert req.best_effort
                    self.memory.free(req)
                    req.preempt()
                    self.declined_reqs.append(req)
                assert n_block <= self.memory.get_n_avail()
                n_preempt += len(preempt_reqs)
                if len(preempt_reqs):
                    # await self.engine.execute_method.remote('preempt', preempt_reqs)
                    await asyncio.gather(*[engine.execute_method.remote('preempt', \
                                            preempt_reqs) for engine in self.engines])

            for req in admitted_reqs:
                req.admitted = True
                assert self.memory.optional_alloc(req)
                n_accepted += 1
            
            # if not (tot_n_finished + len(self.running_reqs) + len(self.running_best_effort_reqs) + len(self.declined_reqs) + len(self.new_reqs)) == self.req_cnt:
            #     print('HERE req_cnt', self.req_cnt, '#finished', tot_n_finished, '#r', len(self.running_reqs), '#best effort', len(self.running_best_effort_reqs), 'declined', len(self.declined_reqs), '#new', len(self.new_reqs))
            #     raise RuntimeError
            revokes = []
            if not len(self.running_reqs) or not len(batches):
                bid = 0
                batches = [BatchSchedule([], next = 0, remain_budget = self.problem.max_seq_len)]
            if self.config.enable_best_effort:
                while len(self.declined_reqs) and \
                    self.memory.get_n_avail() >= self.declined_reqs[0].req.num_blocks:
                    req = self.declined_reqs.pop(0)
                    req.best_effort = True
                    assert self.memory.optional_alloc(req)
                    self.running_best_effort_reqs.append(req)
                    if req.n_preempted == 0:
                        admitted_reqs.append(req)
                    else:
                        revokes.append(RevokeMsg(req.id, req.input_length, req.output_length))

                # if len(self.running_reqs) == 0:
                #     token_budget = 2048 
                # else: token_budget = (batches[-1].get_remain_bs())
                # # if len(self.scheduled_batches) % 1000 == 0:
                # #     print('token budget', token_budget, 'batches', batches)
                # for req in self.running_best_effort_reqs:
                #     assert not req.is_finished()
                #     if req.is_prefill() and token_budget >= req.input_length:
                #         batches[-1].add_req(ReqBatchSchedule(req.id, True, req.input_length))
                #         token_budget -= req.input_length
                #     elif req.is_decode() and token_budget >= 1:
                #         batches[-1].add_req(ReqBatchSchedule(req.id, False, 1))
                #         token_budget -= 1
                #     if token_budget <= 0: break
            else:
                for req in declined_reqs:
                    req.decline()
                    n_declined += 1
            
            # if not (tot_n_finished + len(self.running_reqs) + len(self.running_best_effort_reqs) + len(self.declined_reqs) + len(self.new_reqs)) == self.req_cnt:
            #     print('HERE req_cnt', self.req_cnt, '#finished', tot_n_finished, '#r', len(self.running_reqs), '#best effort', len(self.running_best_effort_reqs), 'declined', len(self.declined_reqs), '#new', len(self.new_reqs))
            #     raise RuntimeError
            # if len(self.scheduled_batches) % 1000 == 0:
            #     print('run batches', batches)
            
            if self.n_schedules % 1000 == 0:
                pbar.set_description(f'#running_reqs {len(self.running_reqs)}' 
                                     f'#declined {n_declined} #n_accepted {n_accepted},'
                                     f'#finished {tot_n_finished}, #arrived {n_arrived}'
                                     f'#best effort {len(self.running_best_effort_reqs)}'
                                     f'#declined {len(self.declined_reqs)}'
                                     f'#new  {len(self.new_reqs)}'
                                     f'#req {self.req_cnt}'
                                     f'#mem {self.memory.get_n_avail()}'
                                     f'batch {len(batches)}'
                                     f'failure {self.declined_by_failure}')
                print('Rank: ', self.id)
                print('Running:', [req.id for req in self.running_reqs])
                print('Best Effort:', [req.id for req in self.running_best_effort_reqs])
                print('Batch: ', batches[bid])
                self.engines[0].execute_method.remote('display')
                # print(f'#running_reqs {self.running_reqs}',
                #                      f'#declined {n_declined} #n_accepted {n_accepted},',
                #                      f'#finished {tot_n_finished}, #arrived {n_arrived}',
                #                      f'#best effort {self.running_best_effort_reqs}',
                #                     #  f'#declined {self.declined_reqs}',
                #                      f'#new  {self.new_reqs}',
                #                      f'#req {self.req_cnt}',
                #                      f'#mem {self.memory.get_n_avail()}',
                #                      f'batche {batches[bid]}')
                # print('Running:', self.running_reqs)
                # print('Batches')
                # for i, batch in enumerate(self.scheduled_batches):
                #     print(f'{i}:', batch)

                # self.engines[0].execute_method.remote('display')
            
            bid, n_finished = await self._run_batches(bid, batches, 
                                                      admitted_reqs,
                                                      revokes)
            tot_n_finished += n_finished
            # if not (tot_n_finished + len(self.running_reqs) + len(self.running_best_effort_reqs) + len(self.declined_reqs) + len(self.new_reqs)) == self.req_cnt:
            #     print('THERE req_cnt', self.req_cnt, '#finished', tot_n_finished, '#r', len(self.running_reqs), '#best effort', len(self.running_best_effort_reqs), 'declined', len(self.declined_reqs), '#new', len(self.new_reqs))
            #     raise RuntimeError
        print('Finish Job!')
        print('#Declined by failure', self.declined_by_failure)
        self.engines[0].execute_method.remote('display')
        self.timer.display()
        
    async def _run_batches(self,
                           bid: int,
                           batches: List[BatchSchedule],
                           admitted_reqs: List[RequestInstanceBase],
                           revokes: List[RevokeMsg]):
        self.timer('run batches begin')

        init_msgs = [RequestInitMsg(req.id, 
                                    req.req.prompt, 
                                    req.input_length, 
                                    req.output_length)
                    for req in admitted_reqs]

        # print(f'Scheduler{self.id}::init_requests BEGIN')
        await asyncio.gather(*[eng.execute_method.remote('init_requests',\
                    init_msgs, revokes)for eng in self.engines])
        # print(f'Scheduler{self.id}::init_requests END')
        
        # for eng in self.engines:
        #     eng.execute_method.remote('init_requests', init_msgs, revokes)
        
        self.timer('init request ends')

        assert (not len(self.running_reqs) and not len(self.running_best_effort_reqs)) or len(batches)
        
        t = self.timer.current_time()
        n_finished = 0

        while True:
            batch: BatchSchedule = copy.deepcopy(batches[bid])
            
            # if len(self.scheduled_batches) % 1000 == 0:
            #     print('token budget', token_budget, 'batches', batches)
            for req in self.running_best_effort_reqs:
                if batch.remain_budget <= 0:
                    break
                assert not req.is_finished()
                if req.is_prefill() and batch.remain_budget >= req.input_length:
                    batch.add_req(ReqBatchSchedule(req.id, True, req.input_length))
                elif req.is_decode() and batch.remain_budget >= 1:
                    batch.add_req(ReqBatchSchedule(req.id, False, 1))
                

            bid += batch.next
                
            # elapsed_time = self.timer.current_time() - self.schedule_time
            # print(len(self.scheduled_batches), bid,
            #       'elapsed', elapsed_time, 'exec', 
            #       self.scheduled_batches[-1].profiled_time if len(self.scheduled_batches) else 0)
            # print(f'Scheduler{self.id}::execute{batch} BEGIN')
            results: List[ExecutionResult] = await asyncio.gather(*[eng.execute_method.remote('execute_batch', 
                                                batch) for eng in self.engines])
            # print(f'Scheduler{self.id}::execute{batch} END: results {results}')
            result = results[0]
            batch.idx = len(self.scheduled_batches)
            batch.estimated_time = self.problem.batch_timer(batch.get_effective_bs(), batch.decode_steps)
            batch.profiled_time = result.draft_time + result.verifier_time
            batch.start_time = self.timer.current_time() + self.time_skew

            # this batch is abnormal
            # if batch.profiled_time > 5e-2 and batch.get_effective_bs() < 100:
            #     print('Long execution batch')
            #     print(batch)
                
            
            batch.batch_size = batch.get_effective_bs()
            self.scheduled_batches.append(batch)

            

            for res in result.results:
                req = self.all_reqs[res.id]
                if not req.commit(res, self.timer.current_time()):
                    print(self.scheduled_batches)
                    print(req)
                    raise RuntimeError
                if req.is_finished():
                    if req.admitted:
                        self.running_reqs.remove(req)
                    else:
                        assert req.best_effort
                        self.running_best_effort_reqs.remove(req)
                    self.memory.free(req)
                    n_finished += 1
            # batch.sch_overhead = self.schedule_overhead
            # self.schedule_overhead = 0.
            batch.profiled_time_ft  = self.timer.current_time() - self.schedule_time
            self.schedule_time = self.timer.current_time()
            
            if not len(result.results) and (
                len(self.running_reqs) or (
                    self.config.enable_best_effort and (len(self.running_best_effort_reqs) or len(self.declined_reqs))
                )
            ):
                warnings.warn(f'no progress {batch}, running {self.running_reqs}, best effort {self.running_best_effort_reqs}, declined: {self.declined_reqs}')
            
            if len(self.new_reqs) > self.config.sch_thresh_arrive or\
                n_finished > self.config.sch_thresh_finish or\
                (self.timer.current_time() - t) > self.config.sch_thresh_timeout:
                break
            

            # self.timer('execution')
            # await asyncio.sleep(0)
            # self.timer('sleep')

        self.timer('run batches end')
        
        return bid, n_finished


    def get_schedule(self) -> Schedule:
            
        return Schedule(self.config.name,
            reqs = sorted(self.all_reqs.values(), key=lambda req: req.id),
            batches = self.scheduled_batches,
            overhead = self.schedule_overhead,
            failure_rate = self.failure_cnt / self.n_schedules,
            tail_time=self.timer.current_time() - self.last_arrive_time)
    