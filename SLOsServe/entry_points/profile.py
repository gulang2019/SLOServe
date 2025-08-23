import random
import numpy as np
import argparse
import openbox as ob

from ..scheduler.schedule_algs import (
    SLOAwareScheduler,
    vLLMScheduler,
     OracleScheduler,
     SarathiServeScheduler)
from ..scheduler.simulator import Simulator
from ..scheduler.struct import Problem, SLA, SchedulerConfig
from ..models import get_model_config
from ..scheduler.profiler import Profiler

if __name__ == '__main__':
    random.seed(123)
    np.random.seed(123)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'facebook/opt-6.7b')
    parser.add_argument('--spec-model', type = str, default = 'facebook/opt-125m')
    parser.add_argument('--tag', type = str, default = 'run')
    parser.add_argument('--trace', type = str, default= 'humaneval')
    parser.add_argument('--arrival_pattern', type = str, default = 'synthetic')
    parser.add_argument('--req_rate', type = int, default=None)
    parser.add_argument('--burstiness', type = int, default = 5)
    # parser.add_argument('--sch', type=str, nargs = '+', choices = ['vllm', 'sarathi', 'slo', 'oracle'])
    parser.add_argument('--slo-ttft', default = 3, type = int)
    parser.add_argument('--slo-tpot', default = 1, type = int)
    parser.add_argument('--n_req', type = int, default = 1000)
    parser.add_argument('-o', '--output', type = str, default = 'simulation')
    parser.add_argument('--tuning', action = 'store_true')
    args = parser.parse_args()

    model_config = get_model_config(args.model)

    problem = Problem(
        model = args.model,
        n_param = model_config.n_param,
        mem_bw = 2e12,
        compute_speed = 312e12,
        
        spec_model = args.spec_model, 
        max_seq_len = model_config.max_seq_len,
        request_trace=args.trace,
        n_req = args.n_req,
        arrival_pattern=args.arrival_pattern,
        req_rate = args.req_rate,
        n_req_at_once = args.burstiness,
        
        alpha = 0.6,
        
        sla_distribution = [(1., SLA(args.slo_ttft, args.slo_tpot, 10, 1, 8))]
    )
    
    from SLOsServe.initialize_spec import init_spec

    engines = init_spec(
        args.model,
        args.spec_model,
        'float16',
        1234,
        'vllm',
        1,
        use_cuda_graph = True
    )
    
    profiler = Profiler(engines[0])
    
    profiler.profile_prob(problem)
    
    # for sch in problem.schedules:
    #     profiler.profile_sch(problem.reqs, sch)
    
     # for sch_tag in args.sch:
    #     if sch_tag == 'vllm':
    #         scheduler = vLLMScheduler()
    #     elif sch_tag == 'sarathi':
    #         scheduler = SarathiServeScheduler(problem.batch_timer)
    #     elif sch_tag == 'slo':
    #         scheduler = SLOAwareScheduler(
    #             problem.batch_timer.shift_bs,
    #             problem.batch_timer(problem.batch_timer.shift_bs), 10)
    #     elif sch_tag == 'oracle':
    #         scheduler = OracleScheduler(problem)

    #     config = {
    #         'sch_thresh_timeout': 5e-3,
    #         'sch_thresh_arrive': 10,
    #         'sch_thresh_finish': 10,
    #     }
    #     if args.tuning:
    #         space = ob.sp.Space()
    #         space.add_variable(ob.sp.Int('sch_thresh_timeout', 1, 5))
    #         space.add_variable(ob.sp.Int('sch_thresh_arrive', 1, 5))
    #         space.add_variable(ob.sp.Int('sch_thresh_finish', 1, 5))
            
    #         def _simulate(config):
    #             simulator = Simulator(config['sch_thresh_timeout'] * 5e-3, 
    #                                   config['sch_thresh_arrive'], 
    #                                   config['sch_thresh_finish'])
                
    #             _sch = simulator.simulate(problem, scheduler)
    #             return {'objectives': [-_sch.profit]}
            
    #         opt = ob.Optimizer(_simulate, space, max_runs=2, task_id = sch_tag)
    #         history = opt.run()
    #         configs = history.get_config_dicts()
    #         objs = history.get_objectives()
    #         idx = min(range(len(objs)), key = lambda i: objs[i])
    #         config = configs[idx]
        
    #     simulator = Simulator(**config)
    #     sch = simulator.simulate(problem, scheduler)
    #     sch.config = SchedulerConfig(
    #         sch_tag,
    #         **config
    #     )
    #     problem.add_schedule(sch)
    
    # print('problem.schs', problem.schedules)