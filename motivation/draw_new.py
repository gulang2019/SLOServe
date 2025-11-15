import json
import matplotlib.pyplot as plt

data = {}
loads = [1.0, 2.0, 3.0, 4.0, 4.5]
for load_scale in loads:
    filename = f'jsons/Qwen_Qwen2.5-7B-Instruct_pd_azure_code_azure_code_0:5000_{load_scale}.json'
    with open(filename, 'r') as f:
        data[load_scale] = json.load(f)

fig, axes = plt.subplots(5, figsize = (4, 20), tight_layout = True)

for i, metrics in enumerate(['TPOT', 'TTFAT', 'TTFT', 'Normalized TTFAT', 'Normalized TTFT']):
    ax = axes[i]
    for p in ['p50', 'p90', 'p99', 'mean']:
        trace = []
        for load_scale in loads:
            trace.append(data[load_scale][metrics][p])
            data[load_scale][metrics][p]
        ax.plot(trace, label = p)
    ax.set_xlabel('Load Scale')
    ax.set_ylabel(metrics)
    ax.legend()
fig.savefig('figs/new.png', dpi = 300, bbox_inches = 'tight')
print('Saved figs/new.png')
            