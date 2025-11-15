from Dataset.dataset import ArrivalTimes, Requests
import os
import numpy as np
import matplotlib.pyplot as plt

FIGDIR = 'figs/loads'
os.makedirs(FIGDIR, exist_ok=True)

from motivation.common import PerfModel, get_easy_name

def count_intervals(intervals: list[tuple[float, float, float, str]], 
                    window: float,
                    mode: str = 'min'):
    import numpy as np
    import math
    start_time = intervals[0][0]
    end_time = intervals[-1][1]
    windows = [[window_start, window_start + window, 0, 0, 0] for window_start in np.arange(start_time, end_time, window)]

    for s, e, time, type in intervals:
        first_window_idx = math.floor((s - start_time) / window)
        last_window_idx = math.ceil((e - start_time) / window)
        if mode == 'min' and first_window_idx != last_window_idx - 1:
            continue
        if first_window_idx >= len(windows) or last_window_idx >= len(windows):
            continue
        for window_idx in range(first_window_idx, last_window_idx + 1):
            if type == 'P':
                windows[window_idx][2] += time
            else:
                windows[window_idx][3] += time
            windows[window_idx][4] += time
    for w in windows: 
        for i in range(2,5):
            w[i] = w[i] / window
    return windows

def visualize_load(arrival_times_name: str, 
                   requests_name: str, 
                   model_name: str,
                   ttft_slo_scale: float,
                   slo_tpot: float,
                   window_start: int,
                   window_end: int,
                   window: float,
                   load_scale: float = 1.0,
                   figname: str = 'figs/loads',
                   slo_constant: float = 0.0):
    
    arrival_times = ArrivalTimes.load(arrival_times_name, load_scale = load_scale).arrival_times
    requests = Requests.load(requests_name, max_tokens = 32768).requests
    requests = requests[window_start:window_end]
    arrival_times = arrival_times[:window_end - window_start]
    arrival_times = [t - arrival_times[0] for t in arrival_times]
    
    print('Req/s:', len(requests) / (arrival_times[-1] - arrival_times[0]))
    
    # INSERT_YOUR_CODE
    import numpy as np

    tot_input_lengths = [req.input_length for req in requests]
    input_lengths = [req.input_length - req.cached_length for req in requests]
    output_lengths = [req.output_length for req in requests]
    perf_model = PerfModel.get_perf_model(model_name)

    def get_stats(arr):
        arr = np.array(arr)
        return {
            'mean': float(np.mean(arr)),
            'p50': float(np.percentile(arr, 50)),
            'p90': float(np.percentile(arr, 90)),
            'p99': float(np.percentile(arr, 99))
        }
        
    tot_input_stats = get_stats(tot_input_lengths)
    input_stats = get_stats(input_lengths)
    output_stats = get_stats(output_lengths)

    print("Tot Input Lengths stats:")
    for k, v in tot_input_stats.items():
        print(f"  {k}: {v}")
    print("Input Lengths stats:")
    for k, v in input_stats.items():
        print(f"  {k}: {v}")
    print("Output Lengths stats:")
    for k, v in output_stats.items():
        print(f"  {k}: {v}")
        

        
    prefill_times = [perf_model.get_batch_time([(req.cached_length, req.input_length - req.cached_length)]) for req in requests]
    prefill_stats = get_stats(prefill_times)
    print("Prefill times stats:")
    for k, v in prefill_stats.items():
        print(f"  {k}: {v}")
        
    slo_ttft_per_token = perf_model.hardware_params[0] * ttft_slo_scale
    slo_ttft_constant = perf_model.hardware_params[4] * ttft_slo_scale + slo_constant
    
    get_ttft_slo = lambda _: slo_ttft_per_token * _ + slo_ttft_constant

    max_decode_batch_size = perf_model.get_max_decode_batch_size(slo_tpot, np.mean([req.input_length for req in requests]))
    
    print('max_decode_batch_size', max_decode_batch_size)
    
    intervals = []
    for request, arrival_time in zip(requests, arrival_times):
        ttft_slo = get_ttft_slo(request.input_length - request.cached_length)
        intervals.append((arrival_time, arrival_time + ttft_slo, perf_model.get_batch_time([(request.cached_length, request.input_length - request.cached_length)]), 'P'))
        for i in range(request.output_length + request.thinking_length - 1):
            intervals.append((arrival_time + ttft_slo + slo_tpot * i, arrival_time + ttft_slo + slo_tpot * (i + 1), 1 / max_decode_batch_size * slo_tpot, 'D'))
    intervals = sorted(intervals, key=lambda x: x[0])
    
    print('counting intervals')
    
    min_intervals = count_intervals(intervals, window, mode = 'min')
    max_intervals = count_intervals(intervals, window, mode = 'max')
    
    print('drawing intervals')

    window_starts, window_ends, min_prefill_servers, min_decode_servers, min_tot_servers = zip(*min_intervals)
    _, _, max_prefill_servers, max_decode_servers, max_tot_servers = zip(*max_intervals)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    labels = ["Prefill", "Decode", "Total"]
    min_series = [min_prefill_servers, min_decode_servers, min_tot_servers]
    max_series = [max_prefill_servers, max_decode_servers, max_tot_servers]
    print('min servers', 'MIN', np.min(min_tot_servers), 'MAX', np.max(min_tot_servers), 'MEAN', np.mean(min_tot_servers))
    print('max servers', 'MIN', np.min(max_tot_servers), 'MAX', np.max(max_tot_servers), 'MEAN', np.mean(max_tot_servers))

    # INSERT_YOUR_CODE
    # Calculate the fraction of time intervals that require N servers (after ceiling)
    import collections
    min_tot_servers_ceil = np.ceil(min_tot_servers).astype(int)
    durations = np.array(window_ends) - np.array(window_starts)
    server_duration = collections.defaultdict(float)
    for n_servers, duration in zip(min_tot_servers_ceil, durations):
        server_duration[n_servers] += duration
    total_time = durations.sum()
    if total_time > 0:
        print("Fraction of time requiring at least N servers (ceil(min_servers)):")
        for n in sorted(server_duration):
            frac = server_duration[n] / total_time
            print(f"  {n} servers: {frac:.2%} of the time")
    else:
        print("Warning: total_time is zero, cannot compute fraction of time.")

    for i, (ax, label) in enumerate(zip(axes, labels)):
        # Plot min servers as horizontal lines per window
        # Plot min servers as horizontal lines per window and connect with verticals
        prev_we, prev_ms = None, None
        for ws, we, ms in zip(window_starts, window_ends, min_series[i]):
            ax.hlines(ms, ws, we, color='b', linewidth=2, label='min' if ws == window_starts[0] else "")
            if prev_we is not None and prev_we == ws:  # connect contiguous intervals
                ax.vlines(ws, prev_ms, ms, color='b', linewidth=2)
            prev_we, prev_ms = we, ms
        # Plot max servers as horizontal lines per window and connect with verticals
        prev_we, prev_mx = None, None
        for ws, we, mx in zip(window_starts, window_ends, max_series[i]):
            ax.hlines(mx, ws, we, color='r', linewidth=2, linestyle="--", label='max' if ws == window_starts[0] else "")
            if prev_we is not None and prev_we == ws:  # connect contiguous intervals
                ax.vlines(ws, prev_mx, mx, color='r', linewidth=2, linestyle="--")
            prev_we, prev_mx = we, mx

        ax.set_ylabel(f"{label} servers")
        ax.legend(loc="upper right")
        ax.grid(True)
    axes[-1].set_xlabel("Time")

    # fig.savefig(f'{fig_prefix}/loads-{model_name}-{slo_ttft}-{slo_tpot}-{window}.png')
    fig.savefig(figname)
    print(f'Saved {figname}')
    
    return {
        'ttft_slo_scale': ttft_slo_scale,
        'slo_tpot': slo_tpot,
        'model_name': model_name,
        'prefill_p99': np.percentile(min_prefill_servers, 99),
        'decode_p99': np.percentile(min_decode_servers, 99),
        'tot_p99': np.percentile(min_tot_servers, 99),
        'prefill_mean': np.mean(min_prefill_servers),
        'decode_mean': np.mean(min_decode_servers),
        'tot_mean': np.mean(min_tot_servers),
        'prefill_p50': np.percentile(min_prefill_servers, 50),
        'decode_p50': np.percentile(min_decode_servers, 50),
        'tot_p50': np.percentile(min_tot_servers, 50),
        'prefill_p25': np.percentile(min_prefill_servers, 25),
        'decode_p25': np.percentile(min_decode_servers, 25),
        'tot_p25': np.percentile(min_tot_servers, 25),
    }
    
def main(arrival_times_name: str, requests_name: str, window: float = 1.0):
    ress = []
    fig_prefix = f'{FIGDIR}/{arrival_times_name}-{requests_name}'
    os.makedirs(fig_prefix, exist_ok=True)
    for tpot_slo_scale in np.arange(1.5, 5.0, 0.5):
        res = visualize_load(
            arrival_times_name=arrival_times_name,
            requests_name=requests_name,
            model_name='Qwen2.5-7B',
            ttft_slo_scale=2.0,
            tpot_slo_scale=tpot_slo_scale,
            window_start=0,
            window_end=1000,
            window=window,
            slo_routing_overhead=0.0,
            fig_prefix=fig_prefix)
        ress.append(res)
    import pandas as pd
    df = pd.DataFrame(ress)
    
    # plot the p99, mean change with tpot_slo_scale
    import matplotlib.ticker as mticker

    # To make it look better, and plot candlestick-like charts for p99/mean
    from matplotlib.patches import Rectangle
    # ----------- Figure 1: Prefill and Decode together -----------
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bar_width = 0.18
    x = df['tpot_slo_scale'].values
    offsets = [-bar_width / 2, bar_width / 2]
    cats = [
        ('prefill', 'Prefill Servers', 'skyblue'),
        ('decode', 'Decode Servers', 'salmon'),
    ]

    for i, (cat, label, color) in enumerate(cats):
        mean = df[f'{cat}_mean'].values
        p99 = df[f'{cat}_p99'].values
        # shift bars for each category to avoid overlap
        bar_x = x + offsets[i]
        for xi, m, p in zip(bar_x, mean, p99):
            rect = Rectangle((xi - bar_width/2, 0), bar_width, m, color=color, alpha=0.8, label=label if xi == bar_x[0] else "")
            ax1.add_patch(rect)
            ax1.vlines(xi, m, p, color='navy', linewidth=2, label='p99' if (xi == bar_x[0] and i == 0) else "")
            ax1.hlines(p, xi - bar_width/4, xi + bar_width/4, color='navy', linewidth=2)
    ax1.set_xticks(x)
    ax1.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.2f}"))
    ax1.set_xlabel('TPOT SLO Scale')
    ax1.set_ylabel('Servers')
    ax1.set_title('Prefill and Decode Server Usage (Mean and P99)')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    handles = [
        Rectangle((0,0),1,1,color='skyblue',alpha=0.8,label='Prefill Mean'),
        Rectangle((0,0),1,1,color='salmon',alpha=0.8,label='Decode Mean'),
        plt.Line2D([0],[0],color='navy',lw=2,label='p99')
    ]
    ax1.legend(handles=handles, loc='upper right')
    fig1.tight_layout()
    fig1.savefig(f'{fig_prefix}/loads-p99-mean-candlestick-prefill-decode-{window}.png')
    print(f"Saved {fig_prefix}/loads-p99-mean-candlestick-prefill-decode-{window}.png")


    # ----------- Figure 2: Total -----------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    cat = 'tot'
    color = 'mediumseagreen'
    label = 'Total Servers'
    mean = df[f'{cat}_mean'].values
    p99 = df[f'{cat}_p99'].values
    bar_x = x  # no offset needed for single bar
    for xi, m, p in zip(bar_x, mean, p99):
        rect = Rectangle((xi - bar_width/2, 0), bar_width, m, color=color, alpha=0.8, label=label if xi == bar_x[0] else "")
        ax2.add_patch(rect)
        ax2.vlines(xi, m, p, color='navy', linewidth=2, label='p99' if xi == bar_x[0] else "")
        ax2.hlines(p, xi - bar_width/4, xi + bar_width/4, color='navy', linewidth=2)
    ax2.set_xticks(x)
    ax2.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.2f}"))
    ax2.set_xlabel('TPOT SLO Scale')
    ax2.set_ylabel('Servers')
    ax2.set_title('Total Server Usage (Mean and P99)')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    handles2 = [
        Rectangle((0,0),1,1,color=color,alpha=0.8,label='Total Mean'),
        plt.Line2D([0],[0],color='navy',lw=2,label='p99')
    ]
    ax2.legend(handles=handles2, loc='upper right')
    fig2.tight_layout()
    fig2.savefig(f'{fig_prefix}/loads-p99-mean-candlestick-total-{window}.png')
    print(f"Saved {fig_prefix}/loads-p99-mean-candlestick-total-{window}.png")   


for model_name in [
    'Qwen/Qwen2.5-7B-Instruct',
    # 'google/gemma-3-27b-it',
    # 'meta-llama/Llama-3.1-70B',
]:
    easy_name = get_easy_name(model_name)


# for burstiness in [0.0, 0.6]: 
    # visualize_load(
    #     arrival_times_name='bursty_0.0',
    #     requests_name='azure_chat_23',
    #     model_name = model_name,
    #     ttft_slo_scale=5.0,
    #     slo_tpot=0.1 if 'Qwen' in model_name else 0.2,
    #     window_start=600,
    #     window_end=1200,
    #     window=1,
    #     figname=f'figs/loads/{easy_name}-chatbot.png',
    #     load_scale=0.1
    # )
    

    visualize_load(
        arrival_times_name='azure_code_23',
        requests_name='sharegpt_code',
        model_name = model_name,
        ttft_slo_scale=3.0,
        slo_tpot=0.025 if 'Qwen' in model_name else 0.05,
        window_start=0,
        window_end=10000,
        window=0.3,
        figname=f'figs/loads/{easy_name}-codebot.png',
        load_scale = 1.0,
        slo_constant = 0.2
    )
    
    

    # visualize_load(
    #     arrival_times_name='burstgpt_GPT-4_Conversation log',
    #     requests_name='arxiv_summary',
    #     model_name = model_name,
    #     ttft_slo_scale=2.0,
    #     slo_tpot=0.1,
    #     window_start=450,
    #     window_end=550,
    #     window=3,
    #     load_scale=5,
    #     figname=f'figs/loads/{easy_name}-arxiv.png'
    # )

# visualize_load(
#     arrival_times_name='azure_chat',
#     requests_name='deepseek-r1',
#     model_name='Qwen2.5-7B',
#     slo_ttft=1.5,
#     slo_tpot=0.025,
#     window_start=0,
#     window_end=1000,
#     window=0.3
# )



# main('azure_chat', 'azure_chat', window = 0.3)
# main('azure_code_23', 'azure_code_23', window = 0.3)
# main('azure_code_23', 'deepseek-r1', window = 0.3)


# for arrival_times_name in ['azure_chat_23', 'azure_code_23', 'azure_chat', 'azure_code']:
#     for requests_name in ['azure_chat_23', 'azure_code_23', 'azure_chat', 'azure_code', 'sharegpt_chat', 'deepseek_r1']:
#         main(arrival_times_name, requests_name)
