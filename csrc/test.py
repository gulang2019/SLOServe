from typing import Any, Callable, Literal
import SLOsServe_C as C
import json
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "axes.labelsize": 20,   # x/y label size
    "axes.titlesize": 20,
})


def test_scheduler():
    reqs = [C.Request(
        id = "0", 
        is_new_req = True,
        ddl = 0.54, 
        input_length = 6707,
        n_computed_tokens = 6707,
        profit = 1.0,
        mem = 2048,
        tpot_idx = 0,
        prefill_mem = 1,
        prefill_device_id = 0,
        decode_device_id = 0,
        prefill_only = False,
    )]


    # Print Request details
    print(reqs)

    # Create a AdmCtrlScheduler instance
    tpots = [0.1]
    hardware_params = [4.1e-5, 0, 0, 0, 1.3e-2]
    scheduler = C.AdmCtrlScheduler("dp", False)
    # scheduler.set_sd_planner(tpots, hardware_params, False, 0.8, 10, False)
    scheduler.set_ar_planner(tpots, hardware_params, False)
    # Call schedule
    M = 100
    current_time = 0.0
    is_feasible, accepted_ids, batch_schedules = scheduler.schedule(reqs, M, current_time, True)

    print('feasible:', is_feasible)

    print('acc_ids', accepted_ids)

    print(reqs)

    # Print the schedule results
    print("Schedule result:")
    for batch_sch in batch_schedules:
        print(batch_sch)
        for req_batch_sch in batch_sch.req_batches:
            print("  ", req_batch_sch)


def test_router():
    hardware_params = [4.1e-5, 0, 1.3e-2]
    tpot = 0.1
    router = C.AdmCtrlRouter(4, hardware_params, tpot)
    reqs = [C.Request(
        id = "0", 
        is_new_req = True,
        ddl = 0.54, 
        input_length = 6707,
        profit = 1.0,
        mem = 20,
        tpot_idx = 0,
        prefill_mem = 1,
    ) for _ in range(4)]
    old_reqs = [C.Request(
        id = "0", 
        is_new_req = False,
        ddl = 0.54, 
        input_length = 6707,
        profit = 1.0,
        mem = 20,
        tpot_idx = 0,
        prefill_mem = 1,
        prefill_device_id = _,
        decode_device_id = _,
    ) for _ in range(4)]
    Ms = [100, 100, 100, 100]
    current_time = 0.0
    result = router.schedule(reqs + old_reqs, Ms, current_time, True)
    print(result)


def json2txt(input: dict, tpot, hardware_params, idx: int):
    '''
    std::string mode;
    infile >> n_avail >> current_time >> num_requests >> mode;

    // Read each request
    for (int i = 0; i < num_requests; ++i) {
        std::string id;
        int input_length, mem, tpot_idx;
        bool is_new_req;
        double ddl, profit;
        infile >> id >> is_new_req >> ddl >> input_length >> profit >> mem >> tpot_idx;
        requests.emplace_back(id, is_new_req, ddl, input_length, profit, mem, tpot_idx);
    }
    '''
    with open(f'problems/{idx}.in', 'w') as f:
        f.write(f'1 {tpot}\n')
        f.write(f'5 {hardware_params[0]} {hardware_params[1]} {hardware_params[2]} {hardware_params[3]} {hardware_params[4]}\n')
        f.write(str(input['num_free_blocks']) + ' 0.0 ' + str(len(input['reqs'])) + ' dp\n')
        for req in input['reqs']:
            f.write(req['id'] + ' ' + str(int(req['is_new_req'])) + ' ' + str(req['ddl']) + ' ' + str(req['input_length']) + ' ' + str(req['n_computed_tokens']) + ' ' + str(req['profit']) + ' ' + str(req['mem']) + ' ' + str(req['tpot_idx']) + ' ' + str(req['prefill_mem']) + ' ' + str(req['prefill_device_id']) + ' ' + str(req['decode_device_id']) + ' ' + str(int(req['prefill_only'])) + '\n')

def get_real_problems(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    data = [e for e in data if e['event_type'] == 'schedule_problem']
    data = [{
        'reqs': [C.Request(**req) for req in e['reqs']],
        'num_free_blocks': e['num_free_blocks'],
        'current_time': 0.0,
    } for e in data]
    return data

def test_problems(filepath = 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4579_anytime/slosserve-edf_auto_scaling-load_slo_req-1.0_1.2_1_anytime_3.0_0.025.events.jsonl',
                  model_name = 'Qwen/Qwen2.5-7B-Instruct', 
                  slo_tpot = 0.025):
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    data = [e for e in data if e['event_type'] == 'schedule_problem']
    from motivation.common import PerfModel
    perf_model = PerfModel.get_perf_model(model_name, 'azure_chat_23')
    bs = perf_model.get_bs(slo_tpot, 1)
    print('bs', bs)
    scheduler_dp = C.AdmCtrlScheduler("dp", False)
    scheduler_dp.set_ar_planner([slo_tpot], perf_model.hardware_params, False)
    
    scheduler_edf = C.AdmCtrlScheduler("edf", False)
    scheduler_edf.set_ar_planner([slo_tpot], perf_model.hardware_params, False)
    
    scheduler_ar_fixed_bs = C.AdmCtrlScheduler("edf", False)
    scheduler_ar_fixed_bs.set_ar_planner([slo_tpot], perf_model.hardware_params, True, 1769)
    
    num_accepted_reqs = []
    is_feasibles = []
    idx = 0
    n_new_reqs = []
    n_old_reqs = []
    import time 
    dp_times = []
    edf_times = []
    ar_fixed_bs_times = []
    print('# problems:', len(data))
    for problem in data:
        
        reqs = [C.Request(**req) for req in problem['reqs']]
        new_reqs = [req.id for req in reqs if req.is_new_req]
        new_reqs = set(new_reqs)
        n_new_reqs.append(len(new_reqs))
        n_old_reqs.append(len(reqs) - len(new_reqs))
        num_free_blocks = problem['num_free_blocks']
        start = time.time()
        is_feasible, accepted_ids, batch_schedules = scheduler_dp.schedule(
            reqs, num_free_blocks, 0.0, False)
        accepted_ids = set(accepted_ids) & set(new_reqs)
        
        dp_time = time.time() - start
        start = time.time()
        is_feasible_edf, accepted_ids_edf, batch_schedules_edf = scheduler_edf.schedule(
            reqs, num_free_blocks, 0.0, False)
        edf_time = time.time() - start
        accepted_ids_edf = set(accepted_ids_edf) & set(new_reqs)
        
        start = time.time()
        is_feasible_ar_fixed_bs, accepted_ids_ar_fixed_bs, batch_schedules_ar_fixed_bs = scheduler_ar_fixed_bs.schedule(
            reqs, num_free_blocks, 0.0, False)
        ar_fixed_bs_time = time.time() - start
        accepted_ids_ar_fixed_bs = set(accepted_ids_ar_fixed_bs) & set(new_reqs)

        num_accepted_reqs.append((len(accepted_ids), len(accepted_ids_edf), len(accepted_ids_ar_fixed_bs)))
        is_feasibles.append((is_feasible, is_feasible_edf, is_feasible_ar_fixed_bs))
        dp_times.append(dp_time)
        edf_times.append(edf_time)
        ar_fixed_bs_times.append(ar_fixed_bs_time)
            
        # if is_feasible_ar_fixed_bs and not is_feasible:
        json2txt(problem, slo_tpot, perf_model.hardware_params, idx)
        idx += 1
    accepted_reqs_dp, accepted_reqs_edf, accepted_reqs_ar_fixed_bs = zip(*num_accepted_reqs)
    import numpy as np
    print('average new reqs', np.mean(n_new_reqs), np.mean(n_old_reqs), np.max(n_new_reqs), np.min(n_new_reqs), np.max(n_old_reqs), np.min(n_old_reqs))
    print('average accepted reqs', np.mean(accepted_reqs_dp), np.mean(accepted_reqs_edf), np.mean(accepted_reqs_ar_fixed_bs))
    is_feasibles_dp, is_feasibles_edf, is_feasibles_ar_fixed_bs = zip(*is_feasibles)
    print('average is_feasibles', np.mean(is_feasibles_dp), np.mean(is_feasibles_edf), np.mean(is_feasibles_ar_fixed_bs))
    print('dp_times', np.mean(dp_times), np.std(dp_times), np.max(dp_times), np.min(dp_times))
    print('edf_times', np.mean(edf_times), np.std(edf_times), np.max(edf_times), np.min(edf_times))
    print('ar_fixed_bs_times', np.mean(ar_fixed_bs_times), np.std(ar_fixed_bs_times), np.max(ar_fixed_bs_times), np.min(ar_fixed_bs_times))


def vllm_scheduler(reqs, num_free_blocks, perf_model, tpot, skip_violations = False):
    t = 0
    violated = set()
    decode_starts = []
    n_accepted = 0
    n_acc_violated = 0
    n_rejected = 0
    
    for req in reqs:
        if num_free_blocks < req.mem:
            n_rejected += 1
            continue
        if skip_violations and (t + perf_model.get_zero_load_ttft(req.input_length) > req.ddl):
            n_rejected += 1
            continue
        n_accepted += 1
        t += perf_model.get_zero_load_ttft(req.input_length)
        decode_starts.append((req, t))
        if t > req.ddl:
            n_acc_violated += 1
            violated.add(req.id)
        num_free_blocks -= req.mem
    
    for req, start in decode_starts:
        if t > req.ddl:
            if req.id not in violated:
                n_acc_violated += 1
    return n_accepted, n_acc_violated, n_rejected

def sarathi_scheduler(reqs, num_free_blocks, perf_model, tpot, skip_violations = False):
    import numpy as np
    import copy
    max_decode_batch_size = perf_model.get_max_decode_batch_size(tpot, np.mean([req.input_length for req in reqs]))
    t = 0
    n_remained = {}
    for req in reqs:
        n_remained[req.id] = req.input_length
    
    n_accepted = 0
    n_acc_violated = 0
    n_rejected = 0
    
    idx = 0
    while idx < len(reqs) and num_free_blocks < reqs[idx].mem:
        n_rejected += 1
        idx += 1
    if idx < len(reqs):
        num_free_blocks -= reqs[idx].mem

    while idx < len(reqs):
        new_finished_reqs = []
        batch = [(1, 0) for _ in range(n_accepted)]
        token_budget = max_decode_batch_size - len(batch)
        while idx < len(reqs) and token_budget > 0:
            req = reqs[idx]
            to_sch = min(token_budget, n_remained[req.id])
            n_remained[req.id] -= to_sch
            token_budget -= to_sch
            batch.append((to_sch, req.input_length - n_remained[req.id] - to_sch))
            if n_remained[req.id] == 0:
                new_finished_reqs.append(req)
                n_accepted += 1
                idx += 1
                while idx < len(reqs) and num_free_blocks < reqs[idx].mem:
                    n_rejected += 1
                    idx += 1
                if idx < len(reqs):
                    num_free_blocks -= reqs[idx].mem
        # print('HERE', token_budget)
        # print('batch', batch)
        # print('new_finished_reqs', new_finished_reqs)
        # print('n_accepted', n_accepted)
        # print('n_rejected', n_rejected)
        # print('n_acc_violated', n_acc_violated)
        # print('idx', idx)
        # print('num_free_blocks', num_free_blocks)
        # print('reqs', reqs)
        t += perf_model.get_batch_time(batch)
        for req in new_finished_reqs:
            if t > req.ddl:
                n_acc_violated += 1
    return n_accepted, n_acc_violated, n_rejected            
            
def qlm_scheduler(reqs, num_free_blocks, perf_model, tpot, skip_violations = False):
    reqs = sorted(reqs, key = lambda x: x.ddl)    
    return sarathi_scheduler(reqs, num_free_blocks, perf_model, tpot, skip_violations)


def get_plus_scheduler(scheduler_fn):
    def plus_scheduler(reqs, num_free_blocks, perf_model, tpot, skip_violations = True):
        best_result = (0, 0, 0)
        _reqs = []
        for i in range(0, len(reqs)):
            n_accepted, n_acc_violated, n_rejected = scheduler_fn(_reqs + [reqs[i]], num_free_blocks, perf_model, tpot, skip_violations)
            # print('HERE', i, n_accepted, n_acc_violated, n_rejected)
            if (n_accepted - n_acc_violated) > (best_result[0] - best_result[1]):
                _reqs = _reqs + [reqs[i]]
                best_result = (n_accepted, n_acc_violated, n_rejected + len(reqs) - len(_reqs))
        return best_result
    return plus_scheduler
                

def synthesize_problems(datasets, model_name):
    from Dataset.dataset import Requests, Request
    import random
    from motivation.common import PerfModel
    import numpy as np
    import tqdm
    REQS = []
    for dataset in datasets:
        ds = Requests.load(dataset)
        REQS.extend(ds.requests)
    
    ttft_scales = [3,5,10]
    ttft_constants = [0.00, 0.05, 0.10]
    tpot = 0.025
    perf_model = PerfModel.get_perf_model(model_name, datasets[0])
    max_bs = perf_model.get_max_decode_batch_size(tpot, np.mean([req.input_length for req in ds.requests]))
    print('max_bs', max_bs)
    def sample_problem(n_requests: int):
        requests = random.sample(REQS, n_requests)
        ttft_scale = random.choice(ttft_scales)
        ttft_constant = random.choice(ttft_constants)
        
        C_requests = [
            C.Request(
                id = f'{i}',
                is_new_req = True, 
                ddl = perf_model.get_zero_load_ttft(req.input_length) * ttft_scale + ttft_constant,
                input_length = req.input_length,
                n_computed_tokens = 0,
                profit = 1,
                mem = 1,
                tpot_idx = 0,
                prefill_mem = 1,
                prefill_device_id = 0, 
                decode_device_id = 0,
                prefill_only = False
            ) for i, req in enumerate(requests)
        ]
        return {
            'reqs': C_requests,
            'num_free_blocks': max_bs,
            'current_time': 0.0,
            'mode': 'dp',
        }
    
    problems = []
    for n_requests in [2,4,8,16]:
        for _ in range(100):
            problems.append(sample_problem(n_requests))
    return problems

def run(problems, model_name, tpot, dataset):
    from motivation.common import PerfModel
    from Dataset.dataset import Requests
    import numpy as np
    import tqdm
    requests = Requests.load(dataset).requests
    perf_model = PerfModel.get_perf_model(model_name, dataset)
    max_bs = perf_model.get_max_decode_batch_size(tpot, np.mean([req.input_length for req in requests]))
    
    scheduler_dp = C.AdmCtrlScheduler("dp", False)
    scheduler_dp.set_ar_planner([tpot], perf_model.hardware_params, False)
    
    scheduler_edf = C.AdmCtrlScheduler("edf", False)
    scheduler_edf.set_ar_planner([tpot], perf_model.hardware_params, False)
    
    scheduler_ar_fixed_bs = C.AdmCtrlScheduler("edf", False)
    scheduler_ar_fixed_bs.set_ar_planner([tpot], perf_model.hardware_params, True, max_bs)
    all_results = {}
    for scheduler, name in zip([scheduler_dp, scheduler_edf, scheduler_ar_fixed_bs], 
                               ['OPT', 'edf', 'qlm+']):
        results = []
        for problem in tqdm.tqdm(problems): 
            is_feasible, accepted_ids, batch_schedules = scheduler.schedule(
                problem['reqs'], problem['num_free_blocks'], problem.get('current_time', 0.0), False)
            results.append((len(accepted_ids), 0, len(problem['reqs']) - len(accepted_ids)))
        
        accepted, accepted_violations, rejected = zip(*results)
        
        all_results[name] = {
            'accepted': np.mean(accepted),
            'accepted_violations': np.mean(accepted_violations),
            'rejected': np.mean(rejected),
        }
        
        
    for scheduler, name in zip(
        [vllm_scheduler, sarathi_scheduler, qlm_scheduler, get_plus_scheduler(vllm_scheduler), get_plus_scheduler(sarathi_scheduler)],
        ['vllm', 'sarathi', 'qlm', 'vllm+', 'sarathi+']):
        results = []
        for problem in tqdm.tqdm(problems):
            n_accepted, n_accpted_violations, n_rejected = scheduler(problem['reqs'], problem['num_free_blocks'], perf_model, tpot)
            results.append((n_accepted, n_accpted_violations, n_rejected))
        n_accepted, n_accpted_violations, n_rejected = zip(*results)
        all_results[name] = {
            'accepted': np.mean(n_accepted),
            'accepted_violations': np.mean(n_accpted_violations),
            'rejected': np.mean(n_rejected),
        }
    import pandas as pd
    df = pd.DataFrame(all_results)
    return df

# test_problems('experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4579_anytime/slosserve-edf_auto_scaling-load_slo_req-1.0_1.2_1_anytime_3.0_0.025.events.jsonl',
#               'Qwen/Qwen2.5-7B-Instruct',
#               0.025)

# test_problems('experiments/Qwen-7B_constant_azure_chat_23:azure_chat_23_601:1202_anytime/slosserve-edf_auto_scaling-all-0.18_4.0_4_anytime_5.0_0.1.events.jsonl',
#               'google/gemma-3-27b-it',
#               0.1)

# test_problem_synthesis('azure_code_23', 'Qwen/Qwen2.5-7B-Instruct')

# test_problem_synthesis('azure_chat_23', 'Qwen/Qwen2.5-7B-Instruct')
# test_problem_synthesis('arxiv_summary', 'Qwen/Qwen2.5-7B-Instruct')

# real_problems_1 = get_real_problems('experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4579_anytime/slosserve-edf_auto_scaling-load_slo_req-1.0_1.2_1_anytime_3.0_0.025.events.jsonl')

def plot_stacked_normalized(df: pd.DataFrame, cols=None, title=None, save_prefix=None, 
                            legend=True, figsize=None, ax=None):
    """
    Draw a normalized stacked bar chart with pretty styling and annotate bottom bar values.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import numpy as np

    if cols is None:
        cols = ['slo_satisfied', 'accepted_violations', 'rejected']
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    totals = df[cols].sum(axis=1)
    max_total = float(totals.max()) if len(totals) else 1.0
    df_norm = df[cols] / (max_total if max_total > 0 else 1.0)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (10, 5), tight_layout=True)

    # Draw stacked bars
    bars = df_norm.plot.bar(ax=ax, stacked=True)

    ax.set_ylabel("Request Fraction")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    if title:
        ax.set_title(title)

    # Legend
    if legend:
        ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    else: 
        ax.legend().remove()

    # Style
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for tick in ax.get_xticklabels():
        tick.set_rotation(15)
        tick.set_ha("right")

    # --- Annotate the bottom bar segment (first col in `cols`) ---
    bottom_col = cols[0]
    for i, (x, val) in enumerate(zip(df.index, df_norm[bottom_col])):
        abs_val = df[bottom_col].iloc[i]
        ax.text(
            i,                               # x-position (bar index)
            val / 2,                         # halfway up the bottom bar segment
            f"{abs_val * 100 / 7:.0f}%",                # text (integer value)
            ha="center", va="center",
            color="white", fontsize=11, fontweight="bold"
        )

    plt.tight_layout()

    # Save optional
    if save_prefix:
        df.to_csv(f"{save_prefix}.csv", index=True)
        plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches="tight")

    return ax

def test_synthetic_problems(tpot = 0.025, legend = True, figsize = None, ax = None, title = None):
    import os
    if not os.path.exists(f'adm_eval-{tpot}.csv'):
        synthetic_problems = synthesize_problems(['azure_chat_23', 'sharegpt_code', 'arxiv_summary'], 'Qwen/Qwen2.5-7B-Instruct')

        df = run(synthetic_problems, 'Qwen/Qwen2.5-7B-Instruct', tpot, 'azure_chat_23')
        df[['vLLM', 'vLLM+', 'Sarathi', 'Sarathi+', 'QLM', 'QLM+', 'Ours', 'OPT']] = df[['vllm', 'vllm+', 'sarathi', 'sarathi+', 'qlm', 'qlm+', 'edf', 'OPT']]
        df = df[['vLLM', 'vLLM+', 'Sarathi', 'Sarathi+', 'QLM', 'QLM+', 'Ours', 'OPT']]
        
        df = df.T
        df['slo_satisfied'] = df['accepted'] - df['accepted_violations']
    else:
        df = pd.read_csv(f'adm_eval-{tpot}.csv', index_col = 0)
    print(df)
    df.rename(columns = {'SLO Satisfied': 'SLO Attained'}, inplace = True)
    df.rename(columns = {'Violations (Accepted)': 'SLO Violated (Admitted)'}, inplace = True)
    df.rename(columns = {'Violation (Rejected)': 'SLO Violated (Rej.)'}, inplace = True)
    # df.to_csv(f'adm_eval-{tpot}.csv')
    df = df.T[['vLLM', 'Sarathi', 'QLM', 'vLLM+', 'Sarathi+', 'QLM+', 'Ours', 'OPT']].T
    print(df)
    plot_stacked_normalized(
        df,
        cols=['SLO Attained', 'SLO Violated (Admitted)', 'SLO Violated (Rej.)'],
        save_prefix=f"adm_eval-{tpot}",
        legend=legend,
        figsize=figsize,
        ax=ax
    )
    ax.set_title(title)

def main():
    fig, ax = plt.subplots(1, 1, figsize = (12, 3.5))
    test_synthetic_problems(0.025, False, (10,5), ax, 'TPOT=25ms')
    fig.savefig(f'adm_eval.pdf')
    fig.savefig(f'adm_eval.png')
    # test_synthetic_problems(0.1, True, (10, 4.5), ax2, 'TPOT=100ms')
    
# test_problems('experiments/Qwen-7B_constant_azure_chat_23:azure_code_23_3978:4579_anytime/slosserve-edf_auto_scaling_resch-all_chat-1.0_0.9_1_anytime_5.0_0.1.events.jsonl')
# test_problems('tmp.json')/
main()