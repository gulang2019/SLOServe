import random
import numpy as np
import argparse
# import openbox as ob
import asyncio
import os
import tqdm
import ray
from typing import Tuple, List
from itertools import product
import time
import copy


from ..engine_wrapper import EngineWrapper
from ..scheduler import (
    Scheduler, 
    get_schedule_alg,
    Profiler,
    Problem, 
    SchedulerConfig,
    EOS,
    REJ,
    MemoryInstance,
    Simulator,
    BatchTimer,
    Schedule,
    Request,
    ParaConfig
)

engines = None
args = None

async def req_simulator(prob: Problem, 
                        schedulers: List[Scheduler]):
    n_try = args.n_try or len(schedulers)
    total_reqs = 0  # Total number of requests
    for req in prob.reqs:
        total_reqs += 1
        while req.follow_up_req is not None:
            total_reqs += 1
            req = req.follow_up_req[1]

    finish_pbar = tqdm.tqdm(total=total_reqs, desc="Requests Finished")
    arrive_pbar = tqdm.tqdm(total=total_reqs, desc="Requests Arrived")

    async def process_request(id, req: Request, 
                              time_skew, 
                              start_idx = 0):
        while req is not None:
            idx = 0
            arrive_time = time.perf_counter()
            # indices = list(range(len(schedulers)))
            # random.shuffle(indices)
            queue = schedulers[(idx + start_idx) % len(schedulers)].add_new_req(id, 
                req, 
                time_skew, 
                arrive_time,
                do_best_effort = (idx == (n_try - 1)))
            while True:
                token = await queue.get()
                if token == EOS:
                    finish_pbar.update(1)
                    break
                elif token == REJ:
                    idx += 1
                    if idx == n_try:
                        finish_pbar.update(1)
                        break
                    queue = schedulers[(idx + start_idx) % len(schedulers)].add_new_req(
                        id,
                        req,
                        time_skew, 
                        arrive_time, 
                        do_best_effort = (idx == (n_try - 1)))
                else:
                    pass
            if req.follow_up_req is None:
                break
            await asyncio.sleep(req.follow_up_req[0])
            req = req.follow_up_req[1]
            id += 1

    tasks = []
    time_skew = 0
    start_time = time.perf_counter()
    rid = 0
    for i, req in enumerate(prob.reqs):
        # Add the new request to the scheduler and get its output queue.
        n = 1
        tmp = req
        while tmp.follow_up_req:
            n += 1
            tmp = tmp.follow_up_req[1]
        arrive_pbar.update(n)

        arrive_pbar.set_description(f'real time: {time_skew + time.perf_counter() - start_time:.2f}, time: {req.arrive_time:.2f}')
        # Start a task to process outputs from the queue.
        tasks.append(asyncio.create_task(process_request(rid, req, time_skew, i % len(schedulers))))
        rid += n

        # Calculate the time until the next request arrives.
        if i < len(prob.reqs) - 1:
            sleep_duration = prob.reqs[i+1].arrive_time - req.arrive_time
            try:
                if sleep_duration < 1e-6: continue
                for t in np.arange(0, sleep_duration, 1):
                    cur_duration = min(1, sleep_duration - t)
                    await asyncio.sleep(cur_duration)
                    if arrive_pbar.n == finish_pbar.n: 
                        time_skew += sleep_duration - t - cur_duration
                        break
            except:
                print('sleep_duration', sleep_duration)
        else:
            # If this is the last request, no need to wait.
            pass
    
    await asyncio.gather(*tasks)
    for scheduler in schedulers:
        scheduler.stop = True
    finish_pbar.close()
    arrive_pbar.close()

def run(prob: Problem,
        replicas: List[List[ray.ObjectRef]],
        scheduler_config: SchedulerConfig):
    # profiler = Profiler(engine)
    # # data = profiler.profile_prob(prob, f'{output_dir}/{prob.tag}/{scheduler_config.sch_strategy}-profile.json')
    # if prob.batch_timer is None:
    #     data = profiler.profile_prob(prob)
    #     prob.batch_timer = BatchTimer.from_data(data)
    assert prob.batch_timer is not None
    
    configs = [copy.copy(scheduler_config) for _ in range(len(replicas))]
    # for config in configs:
    #     config.enable_best_effort = False
    # configs[-1].enable_best_effort = scheduler_config.enable_best_effort
    
    print('#Replica', len(replicas))

    schedulers = []
    for i, (engines, config) in enumerate(zip(replicas, configs)):
        assert hasattr(engines[0], 'num_block')
        print('#block', engines[0].num_block)
        memory = MemoryInstance(engines[0].num_block)
        ray.get([eng.execute_method.remote('reset') for eng in engines])
        scheduler = Scheduler(i, 
                            engines, 
                            config=config,
                            problem = prob,
                            memory=memory)
        schedulers.append(scheduler)
        
    async def main():
        await asyncio.gather(
            *[scheduler.async_schedule() for scheduler in schedulers],
            req_simulator(prob, schedulers)
        )
    
    asyncio.run(main())
    ray.get([replica[0].execute_method.remote('display') for replica in replicas])
    return Schedule.from_schedules(
        [scheduler.get_schedule() for scheduler in schedulers])

def run_simulation(
    prob: Problem,
    config: SchedulerConfig,
):
    
    profiler = Profiler()
    if prob.batch_timer is None:
        data = profiler.profile_prob(prob)
        prob.batch_timer = BatchTimer.from_data(data)
    simulator = Simulator(config)
    assert prob.n_block is not None
    memory = MemoryInstance(prob.n_block)
    alg = get_schedule_alg(config.sch_strategy, prob, memory)
    sch = simulator.simulate(prob, alg, memory)
    return sch

def parse_tuple(arg: str) -> List[str]:
    try:
        # Split the input on a comma and convert to int and float
        return arg.split(";")
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Input must be in the form 'int,float', e.g., '1,0.5'"
        )

def run_dyserve(
    problem: Problem, 
    sch_name: str,
):    
    if args.tuning:
        n_sch = len(problem.schedules)
        def _simulate(config):
            sch_config = SchedulerConfig(args.name,
                                        sch_name,
                                        config['timeout'] * 2e-2,
                                        config['arrive_thresh'] * 2,
                                        config['finish_thresh'] * 2,
                                        enable_best_effort=args.best_effort)
            if args.simulation:
                schedule = run_simulation(problem, sch_config)
            else: 
                schedule = run(problem, engines, sch_config)

            return {'objectives': [-schedule.profit]}
        space = ob.sp.Space()
        space.add_variable(ob.sp.Int('timeout', 2, 4))
        space.add_variable(ob.sp.Int('arrive_thresh', 3, 8))
        space.add_variable(ob.sp.Int('finish_thresh', 3, 8))
        opt = ob.Optimizer(_simulate, space, max_runs=10, task_id = sch_name)
        history = opt.run()
        configs = history.get_config_dicts()
        objs = history.get_objectives()
        idx = min(range(len(objs)), key = lambda i: objs[i])
        config = configs[idx]
        sch_config = SchedulerConfig(args.name, 
                                     sch_name, 
                                     config['timeout'] * 2e-2, 
                                     config['arrive_thresh'] * 2, 
                                     config['finish_thresh'] * 2,
                                     enable_best_effort=args.best_effort)
        problem.schedules = problem.schedules[:n_sch]
    else:
        # sch_config =  SchedulerConfig(
        #     sch, 0.02, 4, 2
        # )
        sch_config =  SchedulerConfig(
            args.name, sch_name, 0.1, 10, 6,
            enable_best_effort=args.best_effort
        )
    
    if args.simulation:
        schedule = run_simulation(problem, sch_config)
    else:
        schedule = run(problem, engines, sch_config)
        
    problem.save(schedule, f'{args.output}/{args.tag}')
    print(schedule)
    return schedule


class SLOsServeRunner:
    @staticmethod
    def add_cli_args(parser):
        parser.add_argument('--tuning', action = 'store_true')
        parser.add_argument('--simulation', action = 'store_true')
        parser.add_argument('--best-effort', action = 'store_true')
        parser.add_argument('--port', type = int, default = 29500)
        parser.add_argument('--n_try', type = int, default = None)
        return parser
    
    @staticmethod
    def init(Args):
        global engines
        global args 
        args = Args
        
        if not args.simulation:
            from SLOsServe.initialize_spec import init_spec
            engines = init_spec(
                base_model=args.model,
                spec_model = args.spec_model,
                dtype = 'float16',
                para_config = ParaConfig.from_str(args.para_config),
                seed = 1234,
                backend_tag = 'vllm',
                use_cuda_graph = True,
                block_size= args.block_size,
                verbose = args.verbose,
                port = args.port
            )
            print('engine initialized')
            
    @staticmethod
    def get_sch_name(name):
        if args.best_effort:
            name = name + '-best_effort'
        if args.simulation:
            name = name + '-simulated'
        if args.para_config != '1-1-1': 
            name = name + '-' + args.para_config
        return name

    @staticmethod
    def run(problem: Problem, sch_name: str) -> Schedule:
        return run_dyserve(problem, sch_name)

if __name__ == '__main__':
    random.seed(123)
    np.random.seed(123)
    parser = argparse.ArgumentParser()
    parser.add_argument('--sch', type=str, nargs = '+', choices = ['vllm', 'sarathi', 'slo', 'oracle', 'tpt', 'promax-dp', 'promax-milp', 'tpt-spec'])
    parser.add_argument('-o', '--output', type = str, default = 'result')
    # parser.add_argument('--req_rates', type = float, nargs = '+', default = [])
    parser.add_argument('--P', type = float, default = 0.90)
    parser.add_argument('--port', type = int, default = 29500)
    
    parser.add_argument(
        "--traces",
        type=str,
        nargs="+",  # Accept one or more tuples
    )
    Problem.add_cli_args(parser)
    args = parser.parse_args()

    os.makedirs(f'{args.output}/{args.tag}', exist_ok=True)
    
    if not args.simulation:
        from SLOsServe.initialize_spec import init_spec
        engines = init_spec(
            args.model,
            args.spec_model,
            'float16',
            1234,
            'vllm',
            para_config = ParaConfig.from_str(args.para_config),
            use_cuda_graph = True,
            block_size= args.block_size,
            verbose = args.verbose,
            port = args.port
        )
        print('engine initialized')
    
    # for slo_ttft_slowby, slo_tpot in args.slo:
    #     args.slo_ttft_slowby = slo_ttft_slowby
    #     args.slo_tpot = slo_tpot
    prefix = f'{args.output}/{args.tag}/'
    os.makedirs(prefix, exist_ok=True)
    cur_schs = os.path.join(prefix, 'schedule.csv')
    
    problem = Problem.from_cli_args(args)
    results = {}
    for trace in args.traces:
        problem.update_trace(trace, update_prompt = not args.simulation)
            
        # if len(args.req_rates) == 0:
        #     args.req_rates = [problem.req_rate] 
        # tdf = df[(df['name'] == args.name) & (df['trace'] == problem.request_trace)]
        results[trace] = {}
        for sch_name in args.sch:
            
            # for req_rate in args.req_rates:
            #     run_problem_w_rate(
            #         problem, req_rate = req_rate
            #     )
                
            sch_file = os.path.join(prefix, 'schedule.csv')

            left, right = (1.0, 1.0), (32, 0.0)
            if os.path.exists(sch_file):
                import pandas as pd
                prev_schs = pd.read_csv(sch_file, index_col=False)
                if len(prev_schs):
                    prev_schs = prev_schs[(prev_schs['req_trace'] == problem.request_trace.to_str()) & (prev_schs['name'] == sch_name)][['req_rate', 'sla_satisfy_rate', 'tail_time']]
                    prev_schs = [tuple(_) for _ in prev_schs.itertuples(index = False)]
                    prev_schs = sorted(prev_schs, key = lambda x: x[0])
                    idx = 0
                    while idx < len(prev_schs) and\
                        prev_schs[idx][1] > args.P\
                        and prev_schs[idx][2] < 20:
                        idx += 1
                    print(prev_schs)
                    if idx != len(prev_schs):
                        left = prev_schs[idx - 1]
                        right = prev_schs[idx]
            # print('left', left, 'right', right)
            # exit(0)
            assert left[1] > args.P
            tail_violation = False
            while left[0] + 0.3 < right[0]:
                print('left', left, 'right', right)
                # if tail_violation:
                req_rate = (left[0] + right[0]) / 2
                # else:
                #     req_rate = left[0] + (right[0] - left[0]) / (right[1] - left[1]) * (args.P - left[1])
                problem.update_req_rate(req_rate = req_rate)
                sch = run_problem_w_rate(problem, sch_name)
                if sch.sla_satisfy_rate > args.P and sch.tail_time < 20:
                    tail_violation = False
                    left = (req_rate, sch.sla_satisfy_rate)
                else: 
                    tail_violation = sch.tail_time >= 20
                    right = (req_rate, sch.sla_satisfy_rate)
            results[trace][sch_name] = left[0]
            
    import pandas as pd
    
    print('results', results)
    df = pd.DataFrame(results)
    print(df)
    file_path = f'{prefix}/max_loads.csv'
    if os.path.exists(file_path):
        pdf = pd.read_csv(file_path, index_col = 0)
        df = pd.concat([pdf, df], axis = 0)
    df.to_csv(file_path)
    print(df)
    