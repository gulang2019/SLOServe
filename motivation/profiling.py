import httpx
import random 
random.seed(42)

'''
curl -X POST http://localhost:8000/profile_step \
  -H 'Content-Type: application/json' \
  -d '{
        "batch": [[512, 64], [256, 128]],
        "n": 1000,
        "verbose": true,
        "hz": 100,
        "warmup": 10
      }'
'''

import asyncio


max_model_len = 16384

def get_prefill_only_single_batches(
    n_reqs: list[int] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 11],
    n_tokens: list[int] = [3, 22, 31, 37, 44, 53, 60, 72, 98, 122, 126, 126, 126, 126, 129, 170, 222, 346, 572, 2541, 14986],
    n_past_tokens: list[int] = [0, 0, 0, 0, 0, 0, 0, 0, 1260, 3848, 10000],
    n: int = 1000,
) -> list[list[tuple[int, int]]]:
    for _ in range(n):
        n_req = random.choice(n_reqs)
        batch = []
        for _ in range(n_req):
            n_token = random.choice(n_tokens)
            n_past_token = random.choice(n_past_tokens)
            batch.append((n_past_token, n_token))
        yield batch

def get_decode_only_batches(
    n_reqs: list[int] = [1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 8, 9, 12, 14, 19, 26, 46, 133],
    past_tokens: list[int] = [14, 122, 195, 263, 330, 412, 508, 728, 1224, 1468],
    n: int = 1000,
) -> list[list[tuple[int, int]]]:
    for _ in range(n):
        n_req = random.choice(n_reqs)
        batch = []
        for _ in range(n_req):
            n_past_token = random.choice(past_tokens)
            batch.append([n_past_token, 1])
        yield batch
    

def get_mixed_batches(
    n_reqs: list[int] = [2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 9, 15, 26, 44, 67, 96, 145],
    n_prefill_tokens: list[int] = [2, 36, 82, 119, 120, 121, 122, 122, 123, 123, 124, 124, 125, 125, 130, 208, 350, 526, 674, 1132, 13000],
    n_prefill_reqs: list[int] =  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 6],
    n_past_tokens: list[int] = [0, 169, 300, 442, 908, 1145, 1247, 1355, 1494, 4251, 10000],
    n: int = 1000,
) -> list[list[tuple[int, int]]]:
    import random
    for _ in range(n):
        n_req = random.choice(n_reqs)
        batch = []
        n_prefill_req = random.choice(n_prefill_reqs)
        if n_prefill_req >= n_req: continue
        n_decode_req = n_req - n_prefill_req
        batch = []
        for _ in range(n_decode_req):
            batch.append([random.choice(n_past_tokens), 1])
        for _ in range(n_prefill_req):
            batch.append([random.choice(n_past_tokens), random.choice(n_prefill_tokens)])
        yield batch

async def run(batch_generator_name: str, n: int):
    import pandas as pd
    batches = []
    for name in batch_generator_name.split('-'):
        if name == 'prefill_only':
            batch_generator = get_prefill_only_single_batches
        elif name == 'decode_only':
            batch_generator = get_decode_only_batches
        elif name == 'mixed':
            batch_generator = get_mixed_batches
        else:
            raise ValueError(f'Unknown batch generator: {name}')
        print(f'Running {name}')
        
        import json
        from tqdm import tqdm
        _batches = list(batch_generator(n=n))
        import pprint
        # pprint.pprint(_batches[:10])
        _batches = [b for b in _batches if all(((x[0] + x[1]) <= (max_model_len - 2)) for x in b)]
        print(f'{len(_batches)} batches remaining for {name}')
        batches.extend(_batches)
    filename = f'profiling_{batch_generator_name}-{n}.csv'
    import os 
    if os.path.exists(filename):
        print(f'{filename} already exists, loading results')
        results = pd.read_csv(filename)
    else: 
        results = pd.DataFrame(columns=['total_current_length', 'num_reqs', 'total_past_length', 'energy', 'time', 'batch'])
    print(f'{len(batches)} batches collected, {len(results)} batches already profiled')
    # exit(0)
    
    from motivation.common import PerfModel
    perf_model = PerfModel.get_perf_model('Qwen/Qwen2.5-7B-Instruct')

    for batch in tqdm(batches[len(results):]):
        import math 
        total_current_length = sum(x[1] for x in batch)
        total_past_length = sum(x[0] for x in batch)
        num_reqs = len(batch)
        url = "http://localhost:8000/profile_step"
        headers = {"Content-Type": "application/json"}
        eta = perf_model.get_batch_time(batch)
        max_counts = int(math.ceil(5 / eta))
        data = {
            "batch": batch,
            "n": min(max_counts, 50),
            "verbose": True,
            "warmup": min(max_counts, 20),
            "hz": 100
        }
        try:
            timeout = httpx.Timeout(connect=5.0, read=120.0, write=5.0, pool=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=data, headers=headers)
                resp_json = resp.json()
                energy = resp_json['energy']
                time = resp_json['time']
                results.loc[len(results)] = [total_current_length, num_reqs, total_past_length, energy, time, json.dumps(batch)]
        except Exception as e:
            print(f'Error profiling batch: {batch}, {e}')
            continue
        if len(results) % 50 == 0:
            results.to_csv(filename, index=False)
            print(f'Saved {len(results)} to {filename}')        
    results = pd.DataFrame(results, columns=['total_current_length', 'num_reqs', 'total_past_length', 'energy', 'time', 'batch'])
    results.to_csv(filename, index=False)
    print(f'Saved {len(results)} to {filename}')      

async def test():
    import numpy as np
    url = "http://localhost:8000/profile_step"
    headers = {"Content-Type": "application/json"}

    data = {
        "batch": [[512, 64], [256, 128]],
        "n": 50,
        "verbose": True,
        "warmup": 10,
        "hz": 20
    }
    logns = list(range(15))
    energies = []
    times = []
    stds = []
    for logn in logns:
        n = int(2**logn)
        data["n"] = n
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=data, headers=headers)
            energy = resp.json()['energy']
            time = resp.json()['time']
            energies.append(energy)
            times.append(time)
            stds.append(np.std(resp.json()['times']))
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (4, 8), tight_layout = True)

    ax1.plot(logns, energies, label=f"n={n}")
    ax1.set_xlabel('log2(n)')
    ax1.set_ylabel('Energy (J)')
    ax2.errorbar(logns, times, yerr=stds, fmt='-o', label=f"n={n}")
    ax2.set_xlabel('log2(n)')
    ax2.set_ylabel('Time (s)')
    
    ax1.legend()
    ax2.legend()
    fig.savefig('profiling.png')
    print('Saved profiling.png')
    
def collect_batches(
    filenames = ['experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4579_anytime/results.jsonl',
                 'experiments/Qwen-7B_constant_azure_chat_23:azure_chat_23_600:1201_anytime/results.jsonl',
                 'experiments/Qwen-7B_constant_arxiv_summary:burstgpt_GPT-4_Conversation_450:551_anytime/results.jsonl']                 
):
    import json 
    from tqdm import tqdm
    from collections import Counter
    import random
    from motivation.events_analysis import analyze_events
    all_batches = []
    for filename in filenames:
        with open(filename, 'r') as f:
            lines = list(f.readlines())
            subsampled = random.sample(lines, 10)
            for line in tqdm(subsampled): 
                try: 
                    data = json.loads(line)
                    events, reqs = analyze_events(data['event_file'])
                    batches = [b for b in events if b.event_type == 'batch']
                    types = [b.type for b in batches]
                except Exception as e:
                    print(f'Error analyzing events for {filename}: {e}')
                    continue

                print(f'{len(batches)} batches collected, example: {batches[100]}')
                print(f'Types: {Counter(types)}')
                all_batches.extend(batches)

    import pickle
    with open('all_batches.pkl', 'wb') as f:
        pickle.dump(all_batches, f)
    print('Saved all_batches.pkl')
    # all_batches = [list([[num_computed_token, b.num_scheduled_tokens[req_id]]] \
    #                 for req_id, num_computed_token in zip(b.req_ids, b.num_computed_tokens)) for b in all_batches]
                
    print(f'{len(all_batches)} batches collected')
    

def analyze_batches():
    import pickle as pkl 
    import numpy as np
    import random
    with open('all_batches.pkl', 'rb') as f:
        all_batches = pkl.load(f)
    from collections import Counter
    from collections import defaultdict
    types = [b.type for b in all_batches]
    print(f'Types: {Counter(types)}')
    batch_by_type = defaultdict(list)
    for batch in all_batches:
        batch_by_type[batch.type].append(batch)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,4, figsize=(24, 18), tight_layout=True)
    batch_by_type['all'] = all_batches
    for i, (t, batches) in enumerate(batch_by_type.items()):
        # print(list(batches[0].num_scheduled_tokens.values()))
        # print(batches[0].num_computed_tokens)
        batches = random.sample(batches, 1000)
        current_lengths = sum([list(b.num_scheduled_tokens.values()) for b in batches], start = [])
        past_lengths = sum([b.num_computed_tokens for b in batches], start = [])
        ax = axes[0, i]
        ax1 = axes[1, i]
        ax2 = axes[2, i]
        ax.hist(current_lengths, bins=100, label=f'Current Length')
        ax.set_xlabel(f'Current Length:\n{np.mean(current_lengths):.2f}+-{np.std(current_lengths):.2f}\nP20: {np.percentile(current_lengths, 20):.0f}, p99: {np.percentile(current_lengths, 99):.0f}, p50: {np.percentile(current_lengths, 50):.0f}')
        ax.set_ylabel('Count')
        ax.set_title(f'{t}')
        ax1.hist(past_lengths, bins=100, label=f'Past Length', color='red')
        ax1.set_xlabel(f'Past Length:\n{np.mean(past_lengths):.2f}+-{np.std(past_lengths):.2f}\nP20: {np.percentile(past_lengths, 20):.0f}, p99: {np.percentile(past_lengths, 99):.0f}, p50: {np.percentile(past_lengths, 50):.0f}')
        ax1.set_ylabel('Count')
        
        curr_perteniles = [
            int(np.percentile(current_lengths, pertentile).__float__()) for pertentile in range(0, 101, 5)
        ]
        past_perteniles = [
            int(np.percentile(past_lengths, pertentile).__float__()) for pertentile in range(0, 101, 10)
        ]
        prefill_cur_lengths = [x for x in current_lengths if x > 1]
        if len(prefill_cur_lengths) > 0:
            prefill_curlengths_percentiles = [
                int(np.percentile(prefill_cur_lengths, pertentile).__float__()) for pertentile in range(0, 101, 5)
            ]
            print(f'{t} prefill_curlengths_percentiles: {prefill_curlengths_percentiles}')
        print(f'{t} curr_perteniles: {curr_perteniles}')
        print(f'{t} past_perteniles: {past_perteniles}')
        n_reqs = [len(b.req_ids) for b in batches]
        n_reqs_perteniles = [
            int(np.percentile(n_reqs, pertentile).__float__()) for pertentile in range(0, 101, 5)
        ]
        print(f'{t} n_reqs_perteniles: {n_reqs_perteniles}')
        n_prefill_reqs = [sum(n > 1 for n in b.num_scheduled_tokens.values()) for b in batches]
        if len(n_prefill_reqs) > 0:
            n_prefill_reqs_perteniles = [
                int(np.percentile(n_prefill_reqs, pertentile).__float__()) for pertentile in range(0, 101, 5)
            ]
            print(f'{t} n_prefill_reqs_perteniles: {n_prefill_reqs_perteniles}')
        ax2.hist(n_reqs, label=f'Number of Requests', alpha=0.5)
        ax2.set_xlabel(f'#Req: {np.mean(n_reqs):.2f}+-{np.std(n_reqs):.2f},\n#Prefill: {np.mean(n_prefill_reqs):.2f}+-{np.std(n_prefill_reqs):.2f}\nP20: {np.percentile(n_reqs, 20):.0f}, p99: {np.percentile(n_reqs, 99):.0f}, p50: {np.percentile(n_reqs, 50):.0f}')
        ax2.set_ylabel('Count')
        ax2_twinx = ax2.twinx()
        ax2_twinx.hist(n_prefill_reqs, label=f'Number of Prefill Requests', color='green', alpha=0.5)
        ax2_twinx.set_ylabel('Count')
        # ax2.legend()
        # ax2_twinx.legend()
    fig.savefig('batches_by_type.png')
    print('Saved batches_by_type.png')

    

def analysis():
    filename = 'profiling_prefill_only-mixed-decode_only-500.csv'
    import pandas  as pd
    df = pd.read_csv(filename)
    "total_current_length,num_reqs,total_past_length,energy,time"
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1,2, figsize=(6, 3), tight_layout=True)
    for i, feature in enumerate(['total_current_length', 'time']):
        for j, feature2 in enumerate(['energy']):
            ax = axes[i]
            ax.scatter(df[feature], df[feature2])
            # Linear fit
            x = df[feature].to_numpy()
            y = df[feature2].to_numpy()
            mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
            x = x[mask]
            y = y[mask]
            if x.size >= 2:
                slope, intercept = np.polyfit(x, y, 1)
                order = np.argsort(x)
                x_sorted = x[order]
                y_fit = slope * x_sorted + intercept
                ax.plot(x_sorted, y_fit, color='red', linewidth=2,
                        label=f'fit: y={slope:.4f}x+{intercept:.2f}')
                ax.legend()
            ax.set_xlabel(feature)
            ax.set_ylabel(feature2)
    fig.savefig('profiling_scatter.png')
    print('Saved profiling_scatter.png')
    
    
if __name__ == "__main__":
    analysis()
    exit(0)
    # collect_batches()
    # analyze_batches()
    # test()
    # exit(0)
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('BATCH', type=str, default='mixed')
    parser.add_argument('--n', type=int, default=1000)
    args = parser.parse_args()
    asyncio.run(run(args.BATCH, args.n))
