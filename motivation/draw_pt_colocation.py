import json 
import os
import pathlib

MARKER_CYCLE = ['o', 's', '^', 'v', 'P', 'X', 'D', '*']
COLOR_CYCLE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
          '#bcbd22', '#17becf']
COLOR_MAP = {}
COLOR_IDX = 0
router_raname = {
        'p_td': 'P-D disagg',
        'pt_d': 'P&T-A disagg',
    }

data = {}
for t in ['p_td', 'pt_d']:
    for load in [100, 200, 400, 600]:
        filename = f'jsons/Qwen/Qwen2.5-7B-Instruct_{t}_deepseek-r1_azure_chat_0:{load}.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data[(t, load)] = json.load(f)

import matplotlib.pyplot as plt
import pandas as pd

def draw(metric, sub_metric):
    new_data = [(k[0], k[1], v[metric][sub_metric]) for k, v in data.items()]
    print(new_data)
    '''
    {('p_td', 100): 0.05559890032708283, ('p_td', 200): 0.06851719372222735, ('pt_d', 100): 0.018508737799006927, ('pt_d', 200): 0.025070332844560277, ('pt_d', 400): 0.03683363884830198, ('pt_d', 600): 0.043170876271124876}
    '''
    df = pd.DataFrame(new_data, columns = ['router_type', 'load', metric])
    
    fig, ax = plt.subplots(figsize = (4, 4), tight_layout = True)
    
    for router_type, tdf in df.groupby('router_type'):
        ax.plot(tdf['load'], tdf[metric], label = router_raname[router_type], marker = MARKER_CYCLE[COLOR_IDX % len(MARKER_CYCLE)], linewidth = 0.5)
    ax.set_xlabel('Load')
    ax.set_ylabel(f'{sub_metric} {metric}')
    ax.legend()
    fig.savefig(f'figs/{metric}_{sub_metric}.png', dpi = 300, bbox_inches = 'tight')
    print(f'Saved {metric}_{sub_metric}.png')

# draw('TPOT', 'p99')
# draw('TTFT', 'mean')
# draw('TTFAT', 'mean')
# draw('TPOT', 'mean')

import pickle as pkl

from motivation.disagg_profile import ExecutionResult


def draw_normalized_ttfat():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    fig, ax = plt.subplots(figsize = (4, 4), tight_layout = True)
    for router in ['p_td', 'pt_d']:
        filename = f'/u/gulang/workspace/SLOsServe/csvs/Qwen/Qwen2.5-7B-Instruct_{router}_deepseek-r1_azure_chat_0:200.pkl'
        with open(filename, 'rb') as f:
            data = pkl.load(f)

        normalized_ttfats = []
        for item in data: 
            assert isinstance(item, ExecutionResult)
            idx = item.request.thinking_length
            if idx >= len(item.timestamps):
                # import warnings
                # warnings.warn(f"Index {idx} out of bounds for timestamps of length {len(item.timestamps)}. Skipping this item.")
                continue
            ttfat = item.timestamps[idx] - item.timestamps[1]
            denom = item.request.thinking_length
            if denom == 0:
                # import warnings
                # warnings.warn("Denominator for normalized TTFAT is zero. Skipping this item.")
                continue
            normalized_ttfat = ttfat / denom
            normalized_ttfats.append(normalized_ttfat)
        print(f'colected {len(normalized_ttfats)}/{len(data)} normalized ttfats, skipped {len(data) - len(normalized_ttfats)} items')
        ax.hist(normalized_ttfats, bins = 100, label = router_raname[router], alpha = 0.5)
    ax.set_xlabel('Normalized TTFAT')
    ax.set_ylabel('Frequency')
    ax.legend()
    fig.savefig('figs/normalized_ttfats.png', dpi = 300, bbox_inches = 'tight')
    print('Saved normalized_ttfats.png')



def draw_tpot_w_index():
    router = 'pt_d'
    filename = f'/u/gulang/workspace/SLOsServe/csvs/Qwen/Qwen2.5-7B-Instruct_{router}_deepseek-r1_azure_chat_0:200.pkl'
    with open(filename, 'rb') as f:
        data = pkl.load(f)

    fig, ax = plt.subplots(figsize = (4, 4), tight_layout = True)
    import random
    COLOR_IDX = 0
    for i in range(30):
        item = random.choice(data)
        assert isinstance(item, ExecutionResult)
        token_lengths = [0, item.request.input_length] + list(range(item.request.input_length + 1, item.request.input_length + 1 + len(item.timestamps) - 2))
        timestamps = [x - item.timestamps[0] for x in item.timestamps]
        ax.plot(token_lengths, timestamps, marker = MARKER_CYCLE[COLOR_IDX % len(MARKER_CYCLE)], linewidth = 0.5)
        COLOR_IDX += 1
    ax.set_xlabel('Token Lengths')
    ax.set_ylabel('Timestamp')
    ax.legend()
    fig.savefig('figs/tpot_w_index.png', dpi = 300, bbox_inches = 'tight')
    print('Saved tpot_w_index.png')
draw_normalized_ttfat()
draw_tpot_w_index()