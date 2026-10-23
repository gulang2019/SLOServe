#!/usr/bin/env python3
"""Compare four explicitly modeled BEAM overlays against Figure 7 data."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plots import chat_7b_n_device_triptych as plot
from plots import common

WORKLOADS = {'chat_7b': 'chat_7B', 'code_7b': 'code_7B', 'mixed_7b': 'mixed_7B'}
VARIANTS = {
    'sloserve_raw': ('SLOServe SLO / PCIe', '#008B8B', '-'),
    'beam_raw': ('BEAM goodput SLO / PCIe', '#D55E00', '-'),
    'sloserve_scaled': ('SLOServe SLO / scaled power', '#008B8B', '--'),
    'beam_scaled': ('BEAM goodput SLO / scaled power', '#D55E00', '--'),
}


def scale_energy(energy_j, duration_s):
    # Integral of P' = 70 + (P - 52) * (400 - 70)/(300 - 52).
    a = 330 / 248
    return a * energy_j + (70 - a * 52) * duration_s


def passes_sloserve(row):
    if not row['success'] or row['ttft_s'] > row['ttft_budget_s']:
        return False
    gaps = np.asarray(row['itls_s'])
    lateness = (row['ttft_s'] + np.cumsum(gaps) - row['ttft_budget_s']
                - np.arange(1, len(gaps) + 1) * row['slo_tpot_s'])
    return not len(lateness) or np.percentile(lateness, 90) <= 0


def passes_beam(row, config):
    # benchmarks/energy/benchmark_script.py calculate_metrics goodput semantics.
    length = row['actual_output_tokens']
    tpot = (row['e2e_s'] - row['ttft_s']) / (length - 1) if length > 1 else 0
    return (row['success'] and row['ttft_s'] <= float(config['beam_global_ttft_s'])
            and tpot <= float(config['beam_tbt_s']))


def active_duration(rows, duration):
    intervals = sorted((max(0, r['client_entry_time_s']),
                        min(duration, r['client_entry_time_s'] + r['e2e_s']))
                       for r in rows if r['success'])
    end = total = 0.0
    for start, stop in intervals:
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def read_run(path):
    n = int(path.name.split('_')[1])
    rows = [json.loads(line) for line in (path / 'per_request_slo.jsonl').read_text().splitlines()]
    config = dict(line.split('=', 1) for line in (path / 'slo_config.txt').read_text().splitlines())
    result = json.loads((path / 'beam_detailed.json').read_text())
    assert len(rows) == len(result['ttfts'])
    with (path / 'power_timeline_gpu_0.csv').open() as handle:
        power = list(csv.DictReader(handle))
    energy = np.asarray([float(r['cumulative_energy_mj']) for r in power])
    assert np.all(np.diff(energy) >= 0), path
    duration = float(power[-1]['elapsed_s']) - float(power[0]['elapsed_s'])
    joules = (energy[-1] - energy[0]) / 1000
    return dict(n_device=n, requests=len(rows),
                slo_serve_violation=1 - sum(map(passes_sloserve, rows)) / len(rows),
                beam_violation=1 - sum(passes_beam(r, config) for r in rows) / len(rows),
                beam_global_ttft_s=float(config['beam_global_ttft_s']),
                raw_gpu0_energy_j=joules, scaled_gpu0_energy_j=scale_energy(joules, duration),
                monitor_duration_s=duration, benchmark_duration_s=result['duration'],
                average_n_active_servers=n * active_duration(rows, result['duration']) / result['duration'],
                source=str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir', type=Path, default=ROOT / 'beam-figure7-7b-20261020-215014')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'analysis_outputs/beam_figure7_comparisons')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    report = {}
    for column, (workload, stem) in enumerate(WORKLOADS.items()):
        runs = sorted((read_run(p) for p in (args.archive_dir / workload).glob('split_*_0')), key=lambda r: r['n_device'])
        assert runs, workload
        report[workload] = runs
        original = ROOT / 'Paper/data/e2e' / f'{stem}.csv'
        with original.open() as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            original_rows = list(reader)
        base_rows, _ = plot._load_rows(original)
        series, order = plot._build_method_series(base_rows)
        series, order = plot._filter_default_e2e_methods(series, order)
        series, order, _ = plot._collapse_baseline_families(series, order, family_specs=plot.DEFAULT_BASELINE_FAMILY_COLLAPSE)
        for method in order:
            if 'disagg' in method:
                continue
            values = sorted(series[method], key=lambda r: r['n_device'])
            for axis, key in zip(axes[:, column], ['energy_consumption', 'average_n_active_servers']):
                selected = [r for r in values if r[key] is not None]
                axis.plot([r['slo_violation_rate'] for r in selected], [r[key] for r in selected],
                          label=common.get_method_label(method), alpha=.7, **common.get_method_style(method))
        for variant, (label, color, linestyle) in VARIANTS.items():
            method = f'beam_{variant} / round_robin'
            common.LABEL_MAP[method] = 'BEAM (RR est.)'
            common.COLOR_MAP[method] = color
            common.MARKER_MAP[method] = 'X'
            additions = []
            for run in runs:
                row = {k: '' for k in fields}
                row.update({k: original_rows[0][k] for k in plot.CONSTANT_FIELDS if k in fields})
                row.update(scheduling_policy=f'beam_{variant}', routing_policy='round_robin',
                           n_device=run['n_device'],
                           slo_violation_rate=run['slo_serve_violation' if variant.startswith('sloserve') else 'beam_violation'],
                           energy_consumption=run['n_device'] * run['scaled_gpu0_energy_j' if variant.endswith('scaled') else 'raw_gpu0_energy_j'],
                           average_n_active_servers=run['average_n_active_servers'])
                additions.append(row)
            folder = args.output_dir / variant
            folder.mkdir(exist_ok=True)
            csv_path = folder / f'{stem}.csv'
            with csv_path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(original_rows + additions)
            plot.draw_n_device_triptych(csv_path=csv_path, output_stem=folder / stem,
                panel_keys=plot.DEFAULT_ASPLOS_PANEL_KEYS, large_format=True,
                slo_violation_x_log_scale=True, show_threshold_gap_labels=False,
                show_threshold_gap_connectors=False)
            for axis, key, divisor in [(axes[0, column], 'energy_consumption', 1000), (axes[1, column], 'average_n_active_servers', 1)]:
                axis.plot([r['slo_violation_rate'] * 100 for r in additions],
                          [r[key] / divisor for r in additions], color=color,
                          linestyle=linestyle, marker='X', label='BEAM: ' + label)
        for axis in axes[:, column]:
            axis.set_xscale('symlog', linthresh=.1)
            axis.set_xlim(0, 100)
            axis.set_xlabel('SLO violation (%) — symlog; zero included')
            axis.grid(alpha=.2)
            axis.axvline(1, color='gray', lw=.6)
            axis.axvline(5, color='gray', lw=.6)
        axes[0, column].set_title(workload.replace('_', ' ').title())
        axes[0, column].set_ylabel('Energy (kJ)')
        axes[1, column].set_ylabel('Average active GPUs (BEAM: request-occupancy estimate)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9)
    fig.suptitle('Figure 7 + four BEAM variants (RR shard-0 extrapolation)\nExisting baseline data unchanged; scaled power is a sensitivity estimate, not an SXM rerun')
    fig.tight_layout(rect=(0, .12, 1, .93))
    for suffix in ['png', 'pdf']:
        fig.savefig(args.output_dir / f'comparison_overview.{suffix}', dpi=180)
    plt.close(fig)
    (args.output_dir / 'run_metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output_dir / 'METHODOLOGY.txt').write_text(
        'Four variants: SLOServe cumulative token deadlines or BEAM benchmark goodput '
        '(global configured TTFT and mean TPOT), crossed with raw or linearly scaled GPU power.\n'
        'BEAM native goodput was reconstructed; archived runs did not enable --goodput.\n'
        'SLOServe deadlines use saved client observations and no thinking-token adjustment.\n'
        'Power mapping: P_new = 70 + (P_old - 52)*330/248; no clipping. '
        'Energy mapping: E_new = E_old*330/248 + duration*(70 - 52*330/248).\n'
        'Energy uses FULL GPU-0 monitor window, including pre/post serving overhead. '
        'It is NOT an exactly aligned benchmark-window energy measurement.\n'
        'Cluster energy is N times GPU-0 energy; SLO outcomes are shard 0 only. '
        'Active GPUs estimate is N times union of successful client request lifetimes / benchmark duration, '
        'not measured GPU residency or the original sampled active-server counter.\n'
        'Existing baselines retain original SLOServe scoring and original energy. '
        'BEAM-native panels compare different SLO definitions. '
        'Power scaling leaves latency and schedule unchanged; not a predicted SXM execution.\n'
        'Overview includes 0-100% violation; individual Figure-7-style panels retain 0-10% focus. '
        'Raw/scaled active-GPU curves coincide because only energy is transformed.\n')
    print(f'Outputs: {args.output_dir}')


if __name__ == '__main__':
    main()
