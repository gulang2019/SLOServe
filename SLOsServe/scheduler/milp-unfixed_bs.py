
import gurobipy as gp
from gurobipy import GRB
from benchmark.scripts.structs import Dataset, TestRequest
import pprint
from .struct import Request, Problem, SLA, Schedule, ReqBatchSchedule, BatchSchedule
import time 
from dataclasses import dataclass, field
from typing import List, Any
import math 
import numpy as np
'''
MILP formulation of scheduling for LLM

ai, pi,oi,ttfti,tpoti, wi: arrival time, prompt length, output length, perrequest SLO, request weight 
maximize iwi(piti,0<ttftipi + jti, (j+1)F -ti, jF < tpotiF)
xi, t:= resources allocated to request i at time t;
ixi,t<B
bit= t<ti, 0  prefill indicator
yi,t := bit xit + (1-bit) f(xi,t)
ti, kF := the time to generate the kF token of this request;
tbityit >= pi: prefill resource constraint
t(t>=di, jF & t <di, (j+1)F ) yit >= F; decode resource constraint

'''


options = {
    'WLSACCESSID': 'cbb10db7-5ebb-4679-a160-809f24f51bb8',
    'WLSSECRET': '09942742-4a87-4664-ae88-09e703c9e46b',
    'LICENSEID': 2578678
}

# @dataclass
# class SolverConfig:
#     name: str
#     max_spec_decode: int
    
#     B: int = field(init = False)
#     unit_time: float = field(init = False)
#     num_vars: int = field(init = False)
#     num_constrs: int = field(init = False)
    
#     def init(self, problem: Problem):
#         self.B = problem.batch_timer.shift_bs
#         self.unit_time = problem.batch_timer(self.B)

class SolverRequestVariables:
    def __init__(self, 
                 model: gp.Model,
                 request_id: int,
                 n_prefill_slots: int, 
                 n_slots: int):
        self.request_id = request_id
        self.n_prefill_slots = n_prefill_slots
        self.n_slots = n_slots
        
        self.prefill_allocateds = model.addVars(n_prefill_slots, lb = 0, 
                                                ub = config.B, 
                                                vtype=GRB.INTEGER, 
                                                name = f'prefill_allocated_{request_id}')
        self.decode_indicators = model.addMVar((n_slots, config.max_spec_decode + 1),
                                               vtype = GRB.BINARY,
                                               name = f'indicators_{request_id}')
        self.times = model.addMVars((n_slots,), 
                                    vtype = GRB.CONTINUOUS,
                                    name = f'times_{request_id}')
                    
    def __repr__(self):
        return f'''
prefill_allocateds: {self.prefill_allocateds},
decode_indicators: {self.decode_indicators}
'''

def solve(problem: Problem,
          m: gp.Model) -> Schedule:
    start_time = time.time()
    
    n_accs = [0.] + [(1-problem.alpha**(i+1)) / (1-problem.alpha) for i in range(config.max_spec_decode)]
    n_accs = np.array(n_accs)
    n_decode_tokens = np.arange(config.max_spec_decode + 1)
    resource_consumptions = []
    
    requests_weights = np.array([req.profit() for req in problem.reqs])
    keep_requests = m.addMVar((len(problem.reqs), ), name = 'keep_req', vtype = GRB.BINARY)
    m.setObjective(requests_weights @ keep_requests, GRB.MAXIMIZE)
        
    vars_by_reqs = []
    T = 0
    for i, req in enumerate(problem.reqs):
        arrive_time_idx = math.ceil(req.arrive_time / config.unit_time)
        n_prefill_slots = math.floor(req.ttft() / config.unit_time) - 1 # math.ceil(req.input_length / config.B) * req.sla.ttft_slown_rate
        n_decode_slots = req.output_length * req.sla.tpot_slown_rate
        
        vars = SolverRequestVariables(m, config, i, n_prefill_slots, n_decode_slots, arrive_time_idx)
        vars_by_reqs.append(vars)        
        T = max(T, arrive_time_idx + n_prefill_slots + n_decode_slots)
        
        resource_consumptions.extend([[] for _ in  range(arrive_time_idx + n_prefill_slots + n_decode_slots - len(resource_consumptions))])
        
        generated_tokens = []
        for j in range(n_prefill_slots + n_decode_slots):
            m.addGenConstrIndicator(vars.decode_indicators[j, 0], 0, vars.prefill_finish_time <= j)
            m.addConstr(vars.decode_indicators[j, :].sum() <= 1)
            
            if j < n_prefill_slots:
                prefill_allocated = vars.prefill_allocateds[j]
                resource_consumptions[j + arrive_time_idx].append(prefill_allocated)
                m.addConstr(prefill_allocated <= (vars.decode_indicators[j, 0] * config.B))
        
            generated_tokens.append(n_accs @ vars.decode_indicators[j, :])
            if ((j % req.sla.decode_check_freq) == 0) or (j == (n_prefill_slots + n_decode_slots - 1)):
                m.addGenConstrIndicator(keep_requests[i], 1, gp.quicksum(generated_tokens) * req.sla.tpot_slown_rate >= (j - vars.prefill_finish_time))
            resource_consumptions[j + arrive_time_idx].append(n_decode_tokens @ vars.decode_indicators[j, :])
        
        m.addGenConstrIndicator(keep_requests[i], 1, gp.quicksum(vars.prefill_allocateds) >= req.input_length)
        
    for resources in resource_consumptions:
        if len(resources):
            m.addConstr(gp.quicksum(resources) <= config.B)


    config.num_vars = m.NumVars
    config.num_constrs = m.NumConstrs
    m.write('model.lp')
    
    m.optimize()
        
    allocated_cnt = np.zeros((T,), dtype = np.int32)

    batches = [BatchSchedule() for _ in range(T)]
    for i, (vars, req) in enumerate(zip(vars_by_reqs, problem.reqs)):
        if bool(keep_requests[i].X):
            print(f'req {i} start at', vars.arrive_time_idx, ', finishes prefill at', vars.prefill_finish_time.X)
            print(vars.decode_indicators.X)
            print(f'req{i}', req, vars)
            is_prefill = True
            for j in range(vars.n_prefill_slots + vars.n_decode_slots):
                t = vars.arrive_time_idx + j
                is_prefill = is_prefill and vars.decode_indicators[j, 0].X == 1
                if is_prefill and j < vars.n_prefill_slots:
                    prefill_allocated = int(vars.prefill_allocateds[j].X)
                    batches[t].reqs.append(ReqBatchSchedule(i, True, prefill_allocated))
                    # for k in range(prefill_allocated):
                    #     schedule_viz[allocated_cnt[t]+k][t] = f'P{i}'
                    allocated_cnt[t] += prefill_allocated
                for k in range(config.max_spec_decode):
                    if bool(vars.decode_indicators[j, k+1].X):
                        assert not is_prefill
                        # for l in range(k+1):
                        #     schedule_viz[allocated_cnt[t]+l][t] = f'D{i}'
                        batches[t].reqs.append(ReqBatchSchedule(i, False, k + 1))
                        allocated_cnt[t] += k+1
        else: 
            print(f'req {i} is dropped')
    
    for batch in batches:
        batch.batch_size = config.B
    
    profit = m.getObjective().getValue()
    keep_requests = [bool(var.X) for var in keep_requests.tolist()]
    
    utilization = allocated_cnt.sum() / (config.B * T)
    elasped_time = round(time.time() - start_time, 3)
    print('reward', profit)
    print('utilization', utilization)
    print('elasped_time', elasped_time)
    
    return Schedule(
        name = 'oracle-milp',
        keep_requests = keep_requests,
        batches = batches,
        profit = profit,
        sla_satisfy_rate = sum(keep_requests) / len(keep_requests)
    )

# def test1():
#     problem = Problem(
        
#         'problem1',
#         reqs = [
#             Request(6, 20, 0, 3, 1, 1.0),
#             Request(6, 10, 0, 3, 1, 1.0),
#             Request(6, 20, 0, 3, 1, 1.0),
#             Request(6, 20, 4, 3, 1, 1.0),
#             Request(6, 10, 4, 3, 1, 1.0),
#             Request(6, 20, 4, 3, 1, 1.0),
#             Request(6, 20, 4, 3, 1, 1.0),
#         ],
#         B = 6,
#         unit_time = 1,
#         period = 1,
#         alpha = 0.8,
#         max_spec_decode = 2
#     )

#     solve(problem)

def compute_oracle(problem: Problem)->Schedule:
    config = SolverConfig(
        name = 'MILP',
        max_spec_decode = 1
    )
    
    config.init(problem)
    
    with gp.Env(params = options) as env, gp.Model(env=env) as model:
        model.setParam('TimeLimit', 5*60)
        solution = solve(problem, config, model)

    return solution

if __name__ == '__main__':
    problem = Problem(
        n_param = 7e9,
        mem_bw = 2e12,
        compute_speed = 312e12,
        
        request_trace='humaneval',
        n_req = 10,
        req_rate = 25,
        n_req_at_once = 5,
        
        alpha = 0.6,
        
        sla_distribution = [(1., SLA(5, 1, 10, 1, 8))]
    )
    
    import random
    random.seed(problem.seed)
    np.random.seed(problem.seed)
    
    config = SolverConfig(
        name = 'MILP',
        max_spec_decode = 1
    )
    
    config.init(problem)
    
    pprint.pprint(problem)

    with gp.Env(params = options) as env, gp.Model(env=env) as model:
        model.setParam('TimeLimit', 5*60)
        solution = solve(problem, config, model)
        problem.add_solution(solution)

    problem.save('solution.json')