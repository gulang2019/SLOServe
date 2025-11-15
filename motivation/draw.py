import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import math
import os
import numpy as np

matplotlib.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 24,
    "axes.titlesize": 24,
    "legend.fontsize": 14,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.grid": True
})

global_handles = {}
color_list = plt.get_cmap("tab10").colors
# print(color_list)
# exit(0)
marker_list = ['o', 's', '^', 'D', 'P', 'X', 'v', 'h', '*']
linestyle_list = ['-', '--', ':', '-.']
style_dict = {
    ('slosserve-edf', 'round_robin'): (color_list[0], 's', '-'),
    ('vllm', 'round_robin'): (color_list[1], 'o', '-'),
}

def nice_label(feature):
    pretty_names = {
        'load_scale': 'Load Scale',
        'n_device': '# Devices',
        'ttft_slo_scale': 'TTFT SLO Scale',
        'slo_tpot': 'SLO Tpot (s)',
        'tpot_slo_scale': 'TPOT SLO Scale',
        'profit': 'Profit',
        'slo_violation_rate': 'SLO Violation Rate',
        'scheduling_policy': 'Scheduling Policy',
        'routing_policy': 'Routing Policy',
        'energy_consumption': 'Energy Consumption (kJ)'
    }
    return pretty_names.get(feature, feature)

def format_group_title(other_features_dict):
    # Prettier display, e.g. "Load Scale=1.0, # Devices=8, TTFT SLO Scale=2.0"
    return ", ".join([f"{nice_label(k)}={float(v)}" for k, v in other_features_dict.items()])

def pareto_min_frontier(x, y):
    points = np.array(list(zip(x, y)))
    # Sort by x ascending, then y ascending
    points = points[np.argsort(points[:, 0])]
    
    pareto = [points[0]]
    for px, py in points[1:]:
        # keep if strictly better in y than the last kept point
        if py < pareto[-1][1]:
            pareto.append((px, py))
    return np.array(pareto)

def draw(experiment_dir,
         save_name, 
         results_file = 'results.jsonl', 
         label_format = lambda router, sched: f"{router} / {sched}",
         is_included = lambda router, sched: True,
         included_setups = {},
         more_funcs = lambda df: df,
         include_title = True,
         title_text = None,
         ylim = None,
         rotate_x_ticks = None,
         _ax = None,
         x = None,
         y = None,
         xleft = None,
         xright = None,
         annotate = False,
         pareto = False):
    
    df = pd.read_json(f'{experiment_dir}/{results_file}', lines=True)
    arrival_pattern = None
    for arrival_pattern in ['azure_code_23', 'azure_chat_23', 'burstgpt_GPT-4_Conversation']:
        if arrival_pattern in experiment_dir:
            arrival_pattern = arrival_pattern
            break
    df.fillna(0, inplace=True)
    if arrival_pattern is None:
        raise ValueError(f"Arrival pattern not found in experiment directory: {experiment_dir}")
    
    import re
    window_pattern = r'\d+:\d+'
    window = re.search(window_pattern, experiment_dir).group(0)
    from Dataset.dataset import ArrivalTimes
    window_start, window_end = window.split(':')
    window_start = int(window_start)
    window_end = int(window_end)
    print(f"Window: {window_start} to {window_end}")
    arrivals = ArrivalTimes.load(arrival_pattern, window_start = window_start, window_end = window_end)
    arrivals = arrivals.arrival_times
    # print(f"Arrivals: {arrivals}")
    rps = len(arrivals) / (arrivals[-1] - arrivals[0])
    # df['load_scale'] = rps * df['n_device']
    for k, v in included_setups.items():
        df = df[df[k] == v]
    df = more_funcs(df)
    df['Request / s'] = df['load_scale'] * rps
    # df = df[df['n_device'] != 5]
    # df = df[df['slo_tpot'] == 0.025]
    # df = df[df['ttft_slo_scale'] == 3.0]
    # df = df[df['n_device'] == 1.0]
    # df = df[df['load_scale'] == 0.8]
    # df = df[df['load_scale'] == 1.0]
    # df['routing_policy'] = df['routing_policy']
    # os.makedirs(f'{experiment_dir}/figs', exist_ok=True)
    features = ['Request / s', 'n_device', 'ttft_slo_scale', 'slo_tpot', 'average_n_active_servers', 'energy_consumption', 'burstiness_level']
    features = [f for f in features if f in df.columns]
    print('features', features)
    # Deduplicate by features and select the last row for each unique combination
    subset = [
        'Request / s', 'n_device', 'ttft_slo_scale', 'slo_tpot', 'scheduling_policy', 'routing_policy', 'burstiness_level'
    ]
    subset = [f for f in subset if f in df.columns]
    df = df.sort_index().drop_duplicates(subset=subset, keep='last')

    # df['n_device'] = df['avg_n_active_servers'].astype(int)
    df['routing_policy'] = df['routing_policy'].str.split("-").str[:2].str.join('-')
    # print(df)



    for feature in features:
        # print('plotting', feature, df[feature])
        if len(df[feature].unique()) == 1:
            continue
        other_features = [f for f in features if f != feature]
        if feature == 'average_n_active_servers' or feature == 'energy_consumption':
            other_features.remove('n_device')
        if 'average_n_active_servers' in other_features:
            other_features.remove('average_n_active_servers')
        if 'energy_consumption' in other_features:
            other_features.remove('energy_consumption')
        n_groups = 0
        for feature_value, grouped in df.groupby(other_features):
            if len(grouped[feature].unique()) > 1: 
                n_groups += 1
        if n_groups == 0:
            continue
        ncols = min(3, n_groups)
        nrows = math.ceil(n_groups / ncols)
        
        for ylabel in ['slo_violation_rate']:
            if y and ylabel != y: continue 
            if _ax is None:
                fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows), squeeze=False)
            # if hasattr(fig, 'suptitle'):
            #     global_title = f"{nice_label(ylabel)} vs {nice_label(feature)}"
            #     # Move title slightly higher to leave more room for the legend
            #     fig.suptitle(global_title, fontsize=24, y=1.06)
            idx = 0

            # For global legend, collect all unique line labels and handles
            local_handles = {}
            for other_feature_values, group in df.groupby(other_features):
                if x and feature != x: continue
                if len(group[feature].unique()) == 1: 
                    continue
                row, col = divmod(idx, ncols)
                if _ax is None:
                    ax = axes[row][col]
                else:
                    ax = _ax
                idx += 1

                # Sort for deterministic color/marker assignment
                sched_route_pairs = sorted(group.groupby(['scheduling_policy', 'routing_policy']).groups.keys())
                for i, (sched, route) in enumerate(sched_route_pairs):
                    # if route == 'disaggregated-edf': continue
                    if feature == 'n_device' and 'auto_scaling' in route: continue
                    subg = group[(group['scheduling_policy'] == sched) & (group['routing_policy'] == route)]
                    subg = subg.sort_values(feature)
                    if (sched, route) not in style_dict:
                        style_dict[(sched, route)] = (
                            color_list[len(style_dict) % len(color_list)],
                            marker_list[len(style_dict) % len(marker_list)],
                            linestyle_list[len(style_dict) % len(linestyle_list)],
                        )
                    color, marker, ls = style_dict[(sched, route)]
                    # if sched == 'slosserve-dp': continue
                    
                    if not is_included(route, sched):
                        continue
                    label = label_format(route, sched)
                     
                    if pareto:
                        data = pareto_min_frontier(subg[feature], subg[ylabel])
                        line, = ax.plot(data[:, 0], data[:, 1], marker=marker, linestyle=ls, color=color, linewidth=3.0, markersize=15, label=label)
                    else:
                        line, = ax.plot(
                            subg[feature],
                            subg[ylabel],
                            marker=marker,
                            linestyle=ls,
                            color=color,
                            linewidth=3.0,
                            markersize=15,
                            label=label,
                        )
                    
                    if annotate: 
                    # INSERT_YOUR_CODE
                        for xval, yval in zip(subg[feature], subg[ylabel]):
                            ax.annotate(f'({xval:.2f}, {yval:.2f})', (xval, yval),
                                        textcoords="offset points",
                                        xytext=(0,10),
                                        ha='center',
                                        fontsize=10)
                    # Only add unique labels for legend
                    if label not in global_handles:
                        global_handles[label] = line
                    local_handles[label] = line
                other_features_dict = {f: v for f, v in zip(other_features, other_feature_values)} \
                    if isinstance(other_feature_values, (tuple, list)) else {other_features[0]: other_feature_values}
                ax.set_xlabel(nice_label(feature))
                if feature == 'energy_consumption':
                    ax.ticklabel_format(style='sci', axis='x', scilimits=(3,3))
                
                ax.set_ylabel(nice_label(ylabel))
                if include_title:
                    ax.set_title(title_text or format_group_title(other_features_dict), pad=0)
                # Remove subplot legend; will do it on the main fig
                # ax.legend(loc='best', frameon=True)
                if df[feature].dtype in [float, int]:
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=df[feature].dtype==int))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=False))
                ax.yaxis.grid(True, which='major', linestyle='--', linewidth=1.2, alpha=1.0)
                ax.xaxis.grid(True, which='major', linestyle='--', linewidth=1.2, alpha=1.0)
                ax.set_ylim(bottom=0, top=ylim)
                if xleft is not None:
                    ax.set_xlim(left = xleft)
                if xright is not None:
                    ax.set_xlim(right = xright)
                if rotate_x_ticks is not None:
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotate_x_ticks)
            # ax.hlines(y=0.05, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyle='--', color='red', linewidth=1.5)
            # ax.set_xlim(left = 0)
            # Remove empty subplots if any
            if _ax is None:
                for j in range(idx, nrows * ncols):
                    row, col = divmod(j, ncols)
                    fig.delaxes(axes[row][col])
                # Add global legend outside the main figure, above the title with larger font and no overlap
                fig.tight_layout(rect=[0,0,1,0.91])  # Add more space at top for legend
                if local_handles:
                    # Place legend above the figure title, with larger font size
                    fig.legend(
                        list(local_handles.values()),
                        list(local_handles.keys()),
                        loc='upper center',
                        bbox_to_anchor=(0.5, 1.10),  # move legend further up
                        ncol=min(2, len(local_handles)),
                        frameon=True,
                        fontsize=20,  # larger legend font
                    )
                if save_name is not None:
                    save_path = f'figs_mlsys26/{ylabel}_vs_{feature}-{save_name}.png'
                else:
                    save_path = f'{experiment_dir}/figs/{ylabel}_vs_{feature}.png'
                
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                fig.savefig(save_path.replace('.png', '.pdf').replace('figs_mlsys26', 'figs_mlsys26/pdf'), dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved plot to {save_path}")

def modify_events(filepath):
    from pathlib import Path
    from pprint import pprint
    import json 
    import copy
    from motivation.events_analysis import analyze_events, analyze_slo_violation
    file = Path(filepath)
    datapoints = []
    with open(file / 'results.jsonl', 'r') as f:
        for line in f:
            datapoints.append(json.loads(line))
    new_results = []
    for datapoint in datapoints:
        for ttft_slo_scale in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
            event_file = datapoint['event_file']
            events, reqs = analyze_events(event_file)
            # for req in reqs.values():
            #     pprint(req)
            #     break
            slo_tpot = datapoint['slo_tpot']
            analysis = analyze_slo_violation(reqs, events, 
                                             'Qwen/Qwen2.5-7B-Instruct',
                                             ttft_slo_scale = ttft_slo_scale,
                                             slo_tpot = slo_tpot, 
                                             draw = False,
                                             slo_ttft_overhead = 0.05)
            result = copy.deepcopy(datapoint)
            result.update({
                'slo_violation_rate': 1 - analysis['slo_attainment_rate'],
                'ttft_slo_scale': ttft_slo_scale,
            })
            new_results.append(result)
    with open(file / 'results_modified.jsonl', 'w') as f:
        for result in new_results:
            json.dump(result, f)
            f.write('\n')

def draw_auto_scaling_motivation():
    import numpy as np
    points = np.array([
        [1, 0.29],
        [2, 0.12],
        [3, 0.04],
        [4, 0.03],
    ])
    auto_scaling_points = np.array([
        [1.39, 0.32],
        [1.6, 0.20],
        [2.5, 0.10],
        [3.5, 0.065],
    ])
    fig, ax = plt.subplots(figsize = (4.5, 4.5))
    ax.plot(points[:, 0], points[:, 1], 'o-', label = 'RR LB', linewidth = 2)
    ax.plot(auto_scaling_points[:, 0], auto_scaling_points[:, 1], 'o-', label = 'Vanilla LC', linewidth = 2)
    for i, x in enumerate(auto_scaling_points):
        ax.text(x[0], x[1], "ABCDEFGHI"[i], fontsize=25, ha="left", va="bottom", color = "black")
    ax.legend(fontsize = 18)
    ax.grid(True)
    ax.set_ylim(0, 0.38)
    ax.set_xlabel('Avg. # Active Servers', fontsize = 20)
    ax.set_ylabel('SLO Violation Rate', fontsize = 20)
    fig.savefig('figs/auto_scaling_baseline.png', dpi = 300, bbox_inches = 'tight')
    fig.savefig('figs/auto_scaling_baseline.pdf', dpi = 300, bbox_inches = 'tight')

    print('Saved figs/auto_scaling_baseline.png')

def draw_energy(filename):
    import pandas as pd
    df = pd.read_json(filename + '/results.jsonl', lines=True)
    df = df[df['scheduling_policy'] == 'slosserve-edf']
    fig, ax = plt.subplots(figsize = (4.5, 4.5))
    df = df.sort_values('load_scale')
    df = df[df['load_scale'] > 0.05]
    df = df[df['n_device'] == 1]
    df['energy_consumption'] /= df['load_scale']
    
    ax.plot(df['load_scale'], df['energy_consumption'], 'o-')
    ax.set_xlabel('Load Scale')
    ax.set_ylabel('Energy Consumption (kJ)')
    ax.grid(True)
    fig.savefig('figs/energy.png', dpi = 300, bbox_inches = 'tight')
    fig.savefig('figs/energy.pdf', dpi = 300, bbox_inches = 'tight')
    print('Saved figs/energy.png')



# draw('experiments/Qwen-7B_constant_azure_chat_0:1000_anytime')

# draw('experiments/Qwen-7B_constant_azure_chat_0:1000_anytime')

# draw('experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime', 'Qwen7B-Code', results_file = 'results_new.jsonl')


# draw_auto_scaling_motivation()
# exit(0)
SCHED_LABELS = {
    'slosserve-dp': 'SLOsServe (dp)',
    'slosserve-edf': 'Ours',
    'vllm': 'vLLM',
    'sarathi': 'Sarathi',
    'qlm': 'QLM',
    'vllm-priority': 'vllm-edf',
    'vllm+': 'vLLM+',
    'sarathi+': 'Sarathi+',
    'qlm+': 'QLM+',
}
ROUTER_LABELS = {
    'round_robin': 'RR',
    'auto_scaling-all': 'LC',
    'auto_scaling_resch-all': 'LC (Rerouting).',
}
single_gpu_label_format = lambda router, sched: SCHED_LABELS.get(sched, sched)
single_gpu_is_included = lambda router, sched: sched in ['vllm', 'sarathi', 'qlm', 'slosserve-edf']
single_gpu_is_included_all = lambda router, sched: sched in ['vllm', 'sarathi', 'qlm', 'slosserve-edf', 'vllm+', 'sarathi+', 'qlm+']

def draw_single_gpu():
    
    fig, axes = plt.subplots(1, 4, figsize = (24, 6), tight_layout = True, sharey = True)
    X = 'Request / s'
    Y = 'slo_violation_rate'
    kwargs = {
        'x': X,
        'y': Y,
        'include_title': True,
        'ylim': 0.20,
        'label_format': single_gpu_label_format,
        'is_included': single_gpu_is_included_all,
    }
    draw('experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4581_anytime', 
        'Qwen7B-Code', 
        results_file = 'results.jsonl',
        more_funcs = lambda df: df[df['load_scale'] < 0.35],
        rotate_x_ticks = 45,
        _ax = axes[0],
        title_text = 'Qwen7B-Code',
        **kwargs)

    draw('experiments/Qwen-7B_constant_azure_chat_23:azure_chat_23_600:1201_anytime', 'Qwen7B-Chat',
        more_funcs = lambda df: df[df['load_scale'] < 1.3],
        _ax = axes[1],
        title_text = 'Qwen7B-Chat',
        **kwargs)

    # draw('experiments/Qwen-7B_constant_arxiv_summary:burstgpt_GPT-4_Conversation_450:551_anytime/', 'Qwen7B-Arxiv',
    #     _ax = axes[2],
    #     title_text = 'Qwen7B-Arxiv',
    #     rotate_x_ticks = 45,
    #     **kwargs)


    draw('experiments/Gemma-3-27B-IT_constant_sharegpt_code:azure_code_23_3979:4579_anytime', 'Gemma27B-Code',
        _ax = axes[2],
        title_text = 'Gemma27B-Code', xright = 5.6,
        **kwargs
    )

    draw('experiments/Gemma-3-27B-IT_constant_sharegpt_chat:azure_chat_23_601:1201_anytime', 'Gemma27B-Chat', 
        _ax = axes[3],
        title_text = 'Gemma27B-Chat',
        **kwargs)
    for ax in axes[1:]: ax.set_ylabel('')

    # draw('experiments/Gemma-3-27B-IT_constant_arxiv_summary:burstgpt_GPT-4_Conversation_450:551_anytime/',
    #     'Gemma27B-Arxiv',
    #     _ax = axes[1][2],
    #     title_text = 'Gemma27B-Arxiv',
    #     rotate_x_ticks = 45,
    #     **kwargs)

    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Remove duplicates
    by_label = dict(zip(labels, handles))
    items = list(by_label.items())
    ORDER = ['vLLM', 'Sarathi', 'QLM', 'vLLM+', 'Sarathi+', 'QLM+', 'Ours']
    items.sort(key=lambda x: ORDER.index(x[0]))
    labels, handles = zip(*items)

    fig.legend(handles, labels,
            loc="upper center",  bbox_to_anchor=(0.5, 1.10), ncol=len(by_label), fontsize=22)
    fig.savefig('figs/single_gpu.png', dpi = 300, bbox_inches = 'tight')
    fig.savefig('figs/single_gpu.pdf', dpi = 300, bbox_inches = 'tight')
    print('Saved figs/single_gpu.png')
    print('Saved figs/single_gpu.pdf')


distributed_label_format = lambda router, sched: SCHED_LABELS.get(sched, sched) + ' / ' + ROUTER_LABELS.get(router, router)
distributed_is_included = lambda router, sched: sched in ['vllm', 'sarathi', 'qlm', 'slosserve-edf']
    
# fig, axes = plt.subplots(1, 2, figsize = (10, 5), tight_layout = True)
def draw_distributed():
    X = 'n_device'
    Y = 'slo_violation_rate'
    kwargs = {
        'x': X,
        'y': Y,
        'include_title': True,
        'label_format': distributed_label_format,
        'is_included': distributed_is_included,
    }
    fig, axes = plt.subplots(1, 2, figsize = (10, 5), tight_layout = True)
    draw('experiments/Qwen-7B_constant_azure_chat_23:azure_chat_23_601:1202_anytime/',
        'Qwen7B-Chat-Distributed', results_file = 'tmp.json',
        title_text = 'Qwen7B-Chat',
        _ax = axes[0],
        **kwargs)
    # ax.legend(fontsize = 24)
    
    
    # exit(0)
    draw('experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4579_anytime/',
        'Qwen7B-Code-Distributed', results_file = 'results.jsonl',
        title_text = 'Qwen7B-Code',
        included_setups = {'load_scale': 1.0, 'ttft_slo_scale': 3.0, 'slo_tpot': 0.025},
        more_funcs = lambda df: df[df['n_device'] <= 4],
        _ax = axes[1],
        **kwargs)
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

    # # Remove duplicates
    by_label = dict(zip(labels, handles))
    items = list(by_label.items())
    ORDER = ['vLLM / Round-Robin', 
             'Sarathi / Round-Robin',
             'QLM / Round-Robin',
             'vLLM+ / Round-Robin',
             'Sarathi+ / Round-Robin',
             'QLM+ / Round-Robin', 
             'Ours / Round-Robin']
    items.sort(key=lambda x: ORDER.index(x[0]))
    labels, handles = zip(*items)



    fig.legend(handles, labels,
            loc="upper center",  bbox_to_anchor=(0.5, 1.20), ncol=2, fontsize=22)

    fig.savefig('figs/distributed.png', dpi = 300, bbox_inches = 'tight')
    fig.savefig('figs/distributed.pdf', dpi = 300, bbox_inches = 'tight')
    print('Saved figs/distributed.png')
    print('Saved figs/distributed.pdf')

def draw_auto_scaling():
    fig, axes = plt.subplots(1,2,figsize = (10, 5), tight_layout = True, sharey = True)
    label_format = lambda router, sched: ROUTER_LABELS.get(router, router)

    draw('experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3978:4579_anytime', 'Qwen7B-Code-auto_scaling',
          results_file = 'auto_scaling.jsonl', ylim = 0.10, xleft = 2.0,
          is_included = lambda router, sched:  router in ['round_robin', 'auto_scaling-all', 'auto_scaling_resch-all'] \
              and sched == 'slosserve-edf', _ax = axes[0],
              x = 'average_n_active_servers',
              y = 'slo_violation_rate',
              include_title = False,
              label_format = label_format,
              rotate_x_ticks = 45,)
    axes[0].text(2.75, 0.075, "A", fontsize = 24,  color="black", weight="bold", ha="center", bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3"))
    axes[0].text(3.60, 0.005, "B", fontsize = 24,  color="black", weight="bold", ha="center", bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3"))
    axes[0].text(3.20, 0.035, "(3)", fontsize = 24,  color="black", weight="bold", ha="center", bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3"))
    axes[0].text(4.00, 0.02, "(4)", fontsize = 24,  color="black", weight="bold", ha="center", bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3"))
    axes[1].text(2.9, 0.06, '(3)', fontsize = 24,  color="black", weight="bold", ha="center", bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3"))
    axes[1].text(3.5, 0.03, '(4)', fontsize = 24,  color="black", weight="bold", ha="center", bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3"))
    draw('experiments/Qwen-7B_constant_azure_chat_23:azure_code_23_3978:4579_anytime', 
         'Qwen7B-Chat-auto_scaling',
        results_file = 'results.jsonl', ylim = 0.10, xleft = 2.0,
        is_included = lambda router, sched: router in ['round_robin', 'auto_scaling-all', 'auto_scaling_resch-all'],
        _ax = axes[1],
        x = 'average_n_active_servers',
        y = 'slo_violation_rate',
        include_title = False,
        label_format = label_format,
        rotate_x_ticks = 45,
        pareto = True
    )
    
    axes[0].set_title('Qwen7B-Code')
    axes[1].set_title('Qwen7B-Chat')
    axes[1].set_xlabel('Average # Active Servers')
    axes[1].set_ylabel('')
    axes[0].set_xlabel('Average # Active Servers')
    # axes[1].legend(fontsize = 20)
    
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Remove duplicates
    by_label = dict(zip(labels, handles))
    items = list(by_label.items())
    ORDER = ['RR', 'LC', 'LC (Rerouting).']
    items.sort(key=lambda x: ORDER.index(x[0]))
    labels, handles = zip(*items)

    fig.legend(handles, labels,
            loc="upper center",  bbox_to_anchor=(0.5, 1.10), ncol=len(by_label), fontsize=22)

    
    # draw('experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3978:4579_anytime', 'Qwen7B-Code-auto_scaling',
    #       results_file = 'auto_scaling.jsonl', ylim = 0.10, xleft = 2.0,
    #       is_included = lambda router, sched: 'qlm' in sched or router in ['round_robin', 'auto_scaling-all', 'auto_scaling_resch-all'] \
    #           and sched == 'slosserve-edf', _ax = axes[1],
    #           x = 'energy_consumption',
    #           y = 'slo_violation_rate',
    #           include_title = False,
    #           label_format = label_format,
    #           rotate_x_ticks = 45)
    # ax.legend()
    # ax.set_xlabel('Average # Active Servers')
    fig.savefig('figs_mlsys26/auto_scaling.png', dpi = 300, bbox_inches = 'tight')
    fig.savefig('figs_mlsys26/auto_scaling.pdf', dpi = 300, bbox_inches = 'tight')
    print('Saved figs_mlsys26/auto_scaling.png')



def draw_ablation():
    X = 'slo_tpot'
    Y = 'slo_violation_rate'
    single_gpu_is_included = lambda router, sched: sched in ['vllm', 'sarathi', 'qlm', 'slosserve-edf']
    single_gpu_label_format = lambda router, sched: SCHED_LABELS.get(sched, sched)
    fig, axes = plt.subplots(1, 2, figsize = (10, 5), tight_layout = True)
    kwargs = {
        'x': X,
        'y': Y,
        'include_title': False,
        'label_format': single_gpu_label_format,
        'is_included': single_gpu_is_included,
    }
    draw('experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4578_anytime', 
            'Qwen7B-Code', 
            results_file = 'slo_ablation.jsonl',
            rotate_x_ticks = 45,
            more_funcs = lambda df: df[df['ttft_slo_scale'] == 3.0][df['slo_tpot'].isin([0.025, 0.05, 0.10, 0.125, 0.15])],
            # title_text = 'Qwen7B-Code-ablation',
            _ax = axes[0],
            **kwargs)
    kwargs = {
        'x': 'ttft_slo_scale',
        'y': Y,
        'include_title': False,
        'label_format': single_gpu_label_format,
        'is_included': single_gpu_is_included,
    }
    draw('experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4578_anytime', 
            'Qwen7B-Code', 
            results_file = 'slo_ablation.jsonl',
            rotate_x_ticks = 45,
            more_funcs = lambda df: df[df['slo_tpot'] == 0.025],
            # title_text = 'Qwen7B-Code-ablation',
            _ax = axes[1],
            **kwargs)
    
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Remove duplicates
    by_label = dict(zip(labels, handles))
    items = list(by_label.items())
    ORDER = ['vLLM', 'Sarathi', 'QLM', 'vLLM+', 'Sarathi+', 'QLM+', 'Ours']
    items.sort(key=lambda x: ORDER.index(x[0]))
    labels, handles = zip(*items)

    fig.legend(handles, labels,
            loc="upper center",  bbox_to_anchor=(0.5, 1.10), ncol=4, fontsize=22)
    
    fig.savefig('figs/slo_ablation.png', dpi = 300, bbox_inches = 'tight')
    fig.savefig('figs/slo_ablation.pdf', dpi = 300, bbox_inches = 'tight')
    print('Saved figs/slo_ablation.png')
    print('Saved figs/slo_ablation.pdf')

def plot_windowed_average_step(time_value_pairs, window_size=5.0, ax=None, step_where="post", label = "None", reduction = 'sum', **kwargs):
    """
    Plot average value over time in a step graph using a sliding window.

    Args:
        time_value_pairs: list of (time, value) tuples (sorted ascending)
        window_size: width of sliding window in same time unit (default: 5.0)
        ax: optional matplotlib Axes to draw on
        step_where: 'pre', 'post', or 'mid' (controls step alignment)
    """
    # Convert to numpy arrays
    data = np.array(time_value_pairs)
    times, values = data[:, 0], data[:, 1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # Compute averages at discrete centers
    centers = np.linspace(times.min(), times.max(), 200)
    avg_values = []
    for t in centers:
        mask = (times >= t - window_size / 2) & (times <= t + window_size / 2)
        avg_values.append(values[mask].sum() if reduction == 'sum' else values[mask].mean() if np.any(mask) else np.nan)

    # Remove NaN gaps (optional)
    centers = np.array(centers)
    avg_values = np.array(avg_values)
    valid = ~np.isnan(avg_values)
    centers, avg_values = centers[valid], avg_values[valid]
    print('centers:', centers)
    print('avg_values:', avg_values)
    # Plot step line
    ax.step(centers, avg_values, where=step_where, linewidth=2, label=label, **kwargs)

    ax.set_xlabel("Time")
    ax.set_ylabel("Average value")
    ax.grid(True, linewidth=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return avg_values

def draw_dataset():
    import numpy as np
    from Dataset.dataset import Requests, ArrivalTimes 
    for dataset_name, window_start, window_end, app_name in [
        ('sharegpt_code', 3979, 4579, 'Coder'),
        ('azure_chat_23', 601, 1202, 'ChatBot'),
        ('arxiv_summary', 450, 551, 'Summarizer'),
    ]:
        requests = Requests.load(dataset_name, window_start = window_start, window_end = window_end).requests
        input_lengths = [req.input_length for req in requests]
        output_lengths = [req.output_length for req in requests]
        prompt_mean, prompt_std, prompt_p99 = np.mean(input_lengths), np.std(input_lengths), np.percentile(input_lengths, 99)
        output_mean, output_std, output_p99 = np.mean(output_lengths), np.std(output_lengths), np.percentile(output_lengths, 99)
        print(app_name, prompt_mean, prompt_std, prompt_p99, output_mean, output_std, output_p99)

def draw_arrivals():
    import numpy as np
    from Dataset.dataset import ArrivalTimes
    fig, (ax1, ax2) = plt.subplots(2, figsize = (6, 8), tight_layout = True)
    for arrival_pattern, window_start, window_end, app_name, ax in [
        ('azure_chat_23', 601, 1202, 'ChatBot', ax1),
        ('burstgpt_GPT-4_Conversation', 450, 551, 'Summarizer', ax2),
    ]:
        arrivals = ArrivalTimes.load(arrival_pattern)
        arrivals = arrivals.arrival_times[window_start:window_end]
        events = [(t, 1) for t in arrivals]
        print('example events:', events[:10])
        plot_windowed_average_step(events, window_size = 0.1, ax = ax, label = None)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Arrival Rate (req/s)')
        ax.grid(True)
        ax.legend()
    fig.savefig(f'figs/arrivals.png', dpi = 300, bbox_inches = 'tight')
    fig.savefig(f'figs/arrivals.pdf', dpi = 300, bbox_inches = 'tight')
    print('Saved figs/arrivals.png')
    print('Saved figs/arrivals.pdf')
    

# # draw_ablation()
# draw('experiments/Qwen-7B_constant_arxiv_summary:burstgpt_GPT-4_Conversation_450:551_anytime/', 
#      'Qwen7B-Arxiv', results_file = 'results.jsonl', ylim = 0.30)


# draw('experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4579_anytime/',
#         'Qwen7B-Code-Distributed-New', results_file = 'new_results.jsonl',
#         title_text = 'Qwen7B-Code',
#         included_setups = {'load_scale': 1.0, 'ttft_slo_scale': 3.0, 'slo_tpot': 0.025}, annotate = True, ylim = 0.30)

# draw('experiments/Qwen-7B_constant_azure_chat_23:azure_chat_23_601:1202_anytime/',
#         'Qwen7B-Chat-Distributed-New', results_file = 'tmp.json',
#         title_text = 'Qwen7B-Chat')
def draw_burstiness():
    fig, ax = plt.subplots(figsize = (5, 5), tight_layout = True)
    draw('experiments/Qwen-7B_constant_azure_chat_23:bursty_600:1200_anytime', 'Qwen7B-Chat-burstiness',
    results_file = 'results.jsonl', 
    ylim = 0.2,
    include_title = False,
    y = 'slo_violation_rate',
    x = 'burstiness_level',
    _ax = ax,
    xleft = 0, xright = 0.6,rotate_x_ticks = 45)
    fig.savefig('figs/burstiness.png', dpi = 300, bbox_inches = 'tight')
    fig.savefig('figs/burstiness.pdf', dpi = 300, bbox_inches = 'tight')
    print('Saved figs/burstiness.png')
    print('Saved figs/burstiness.pdf')
# draw_burstiness()
# draw('Qwen-7B_constant_sharegpt_code:bursty_3979:4581_anytime', 'Qwen7B-Code-burstiness',
#       results_file = 'results.jsonl', ylim = 0.1)
# draw_auto_scaling()
# draw_distributed()
# exit(0)

# draw('experiments/Qwen-7B_constant_sharegpt_code:bursty_3979:4581_anytime/', 
#      'Qwen7B-Coder-Burstiness', results_file = 'results.jsonl')

# draw('experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4581_anytime', 'Qwen7B-Code', 
#      results_file = 'results.jsonl',
#      label_format = single_gpu_label_format,
#      is_included = single_gpu_is_included,
#      more_funcs = lambda df: df[df['load_scale'] < 0.35])
# draw_single_gpu()

# draw_ablation()
# draw_distributed()
# draw_arrivals()
# draw_dataset()
# draw('.', 'Qwen7B-Code-auto_scaling', results_file = 'tmp.jsonl')
# draw('experiments/Gemma-3-27B-IT_constant_arxiv_summary:burstgpt_GPT-4_Conversation_450:551_anytime/', 'Gemma27B-Arxiv', ylim = 0.20)


def draw_teaser():
    fig, ax = plt.subplots(figsize = (6, 6), tight_layout = True)
    draw('experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4581_anytime', 
        'Qwen7B-Code', 
        results_file = 'results.jsonl',
        more_funcs = lambda df: df[df['load_scale'] < 0.35],
        rotate_x_ticks = 45,
        _ax = ax,
        title_text = '',
        x = 'Request / s',
        y = 'slo_violation_rate',
        label_format = single_gpu_label_format,
        is_included = lambda router, sched: sched in ['vllm', 'sarathi', 'qlm', 'slosserve-edf'],
        ylim = 0.20,
        annotate = False,
        include_title = False,
    )
    handles, labels = ax.get_legend_handles_labels()
    # print(labels)
    # print(handles)
    # # Make a custom order: Ours first, then vLLM, then Sarathi, then QLM, then rest
    desired_order = ['vLLM', 'Sarathi', 'QLM', 'Ours']
    new_handles_labels = sorted(zip(handles, labels), key=lambda hl: (desired_order.index(hl[1]) if hl[1] in desired_order else len(desired_order)))
    handles, labels = zip(*new_handles_labels)
    ax.legend(handles, labels)
    ax.text(0.75, 0.10, "High Load", color="black", fontsize=20, weight="bold", ha="center", bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3"))
    ax.text(2.00, 0.10, "Over load", color="black", fontsize=20, weight="bold", ha="center", bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3"))

    # Draw a vertical line to separate regions
    ax.axvline(x=1.50, color="red", linestyle="--", linewidth=4)
    ax.set_xlabel('Request / Server / s')

    fig.savefig('figs/teaser.png', dpi = 300, bbox_inches = 'tight')
    fig.savefig('figs/teaser.pdf', dpi = 300, bbox_inches = 'tight')
# draw_teaser()
# exit(0)
# draw_energy('experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4581_anytime')
# exit(0)

draw('experiments/Gemma-3-27B-IT_constant_sharegpt_chat:azure_chat_23_601:1201_anytime', 'Gemma27B-Chat', 
     label_format = single_gpu_label_format,
     is_included = single_gpu_is_included)

draw('experiments/Qwen-7B_constant_azure_chat_23:azure_chat_23_601:1202_anytime/',
     'Qwen7B-Chat-Distributed', results_file = 'results.jsonl')
# exit(0)
draw('experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4579_anytime/',
     'Qwen7B-Code-Distributed', results_file = 'results.jsonl',
     included_setups = {'load_scale': 1.0, 'ttft_slo_scale': 3.0, 'slo_tpot': 0.025},)
# exit(0)
# exit(0)
# draw('experiments/Qwen-7B_constant_azure_chat_23:azure_chat_23_598:1200_anytime',
#      'Qwen7B-Chat-Distributed', results_file = 'results.jsonl')
# exit(0)


draw('experiments/Qwen-7B_constant_azure_chat_23:azure_chat_23_600:1201_anytime', 'Qwen7B-Chat',
     label_format = single_gpu_label_format,
     is_included = single_gpu_is_included,
     more_funcs = lambda df: df[df['load_scale'] < 1.3])

draw('experiments/Qwen-7B_constant_arxiv_summary:burstgpt_GPT-4_Conversation log_400:601_anytime/', 'Qwen7B-Arxiv',
     label_format = single_gpu_label_format,
     is_included = lambda router, sched: sched in ['vllm', 'slosserve-dp', 'slosserve-edf'])

draw('experiments/Gemma-3-27B-IT_constant_sharegpt_code:azure_code_23_3979:4579_anytime', 'Gemma27B-Code',
     label_format = single_gpu_label_format,
     is_included  = lambda router, sched: sched in ['vllm', 'sarathi', 'qlm', 'slosserve-edf']
)



draw('experiments/Gemma-3-27B-IT_constant_arxiv_summary:burstgpt_GPT-4_Conversation log_400:600_anytime/', 'Gemma27B-Arxiv')



# draw('experiments/Qwen-7B_constant_sharegpt_chat:burstgpt_GPT-4_Conversation log_400:800_anytime/')


# modify_events('experiments/Qwen-7B_constant_sharegpt_chat:azure_chat_23_600:1200_anytime')

# draw('experiments/Qwen-7B_constant_sharegpt_chat:azure_chat_23_600:1200_anytime', results_file = 'results_modified.jsonl')


# draw('experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4581_anytime', 'Qwen7B-Code-Distributed', results_file = 'results.jsonl')

