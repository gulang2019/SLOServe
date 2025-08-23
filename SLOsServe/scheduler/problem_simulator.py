import asyncio

from .struct import (
    Problem, SchedulerConfig) 
from .profiler import Profiler
from .scheduler import Scheduler
from .schedule_algs import get_schedule_alg
from ..engine_wrapper import EngineWrapper

async def req_simulator(
    problem: Problem,
    scheduler: Scheduler
):
    for i, req in enumerate(problem.reqs):
        scheduler.add_new_req(req)
        await asyncio.sleep(problem.reqs[i+1].arrive_time - req.arrive_time)

def run(prob: Problem, 
             engine: EngineWrapper,
             scheduler_config: SchedulerConfig):
    profiler = Profiler(engine)
    data = profiler.profile_prob(prob)
    prob.batch_timer.fit([(x.bs, x.e2e_time) for x in data])
    alg = get_schedule_alg(scheduler_config.sch_strategy, prob)
    scheduler = Scheduler(engine, 
                          config=scheduler_config,
                          batch_timer = prob.batch_timer,
                          alg=alg)
    
    async def main():
        await asyncio.gather(
            scheduler.async_schedule(),
            req_simulator(prob, scheduler)
        )
    
    asyncio.run(main())
    
    prob.add_schedule(scheduler.get_schedule())
    
    