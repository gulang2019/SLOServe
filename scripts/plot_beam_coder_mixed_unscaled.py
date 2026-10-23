#!/usr/bin/env python3
"""Show unscaled PCIe BEAM energy; display the Coder Pareto frontier.

Dominated Coder measurements remain in CSV/provenance, not the displayed curve.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plots import chat_7b_n_device_triptych as paper
from plots.common import get_method_label
from scripts.plot_beam_figure7_energy_preview import load_paper_series
from scripts.plot_beam_figure8_mixed import file_hash, source_name

RAW = 'BEAM (unscaled)'
STYLES = {
    RAW: dict(color='#008B8B', marker='D', linewidth=1.6, markersize=5),
}


def select_coder_frontier(points):
    """Minimize raw energy and violation rate; retain ties and all source data."""
    for point in points:
        values = [point['raw_energy_kj'], point['violation_pct_upper']]
        if not np.isfinite(values).all() or any(value < 0 for value in values):
            raise ValueError('Pareto coordinates must be finite and nonnegative')
    kept, omitted = [], []
    for point in points:
        dominators = [other['n_device'] for other in points
                      if other['raw_energy_kj'] <= point['raw_energy_kj']
                      and other['violation_pct_upper'] <= point['violation_pct_upper']
                      and (other['raw_energy_kj'] < point['raw_energy_kj']
                           or other['violation_pct_upper'] < point['violation_pct_upper'])]
        if dominators:
            omitted.append(dict(point, dominated_by_n_devices=sorted(dominators)))
        else:
            kept.append(point)
    return kept, omitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-preview', type=Path, default=ROOT / (
        'analysis_outputs/beam_figure7_energy_preview_old_coding_multiround_tpot25ms/provenance.json'))
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'analysis_outputs/beam_figure7_coder_mixed_unscaled_frontier')
    args = parser.parse_args()
    source = json.loads(args.source_preview.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'pdf.fonttype': 42,
                         'axes.labelsize': 10, 'axes.titlesize': 12,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    legend_styles = {}
    rows = []
    omissions = []
    for axis, workload, title in zip(axes, ('code_7b', 'mixed_7b'), ('Coder — 7B', 'Mixed — 7B')):
        original = source['original_sources'][workload]
        csv_path = ROOT / original['source']
        if file_hash(csv_path) != original['sha256']:
            raise ValueError(f'Original paper data changed since the source preview: {csv_path}')
        series, order, _ = load_paper_series(csv_path)
        styles = paper._build_method_styles(order)
        for method in order:
            legend_styles.setdefault(get_method_label(method), styles[method])
        points = sorted((p for p in source['beam_points'] if p['workload'] == workload),
                        key=lambda p: p['n_device'])
        if not points:
            raise ValueError(f'No {workload} points in the source preview')
        displayed, omitted = select_coder_frontier(points) if workload == 'code_7b' else (points, [])
        omissions.extend(omitted)
        omitted_by_n = {point['n_device']: point for point in omitted}
        for point in points:
            expected_raw = point['n_device'] * np.mean(
                [run['raw_gpu_energy_j'] for run in point['per_shard']]) / 1000
            if not np.isclose(point['raw_energy_kj'], expected_raw, rtol=0, atol=1e-8):
                raise ValueError('Raw energy must equal N times the mean measured shard energy')
            rows.append(dict(workload=workload, n_device=point['n_device'],
                             violation_pct=point['violation_pct_upper'],
                             violation_pct_lower=point['violation_pct_lower'],
                             raw_energy_kj=point['raw_energy_kj'],
                             included_in_curve=point['n_device'] not in omitted_by_n,
                             omission_reason=('Pareto-dominated by N=' + ','.join(
                                 str(n) for n in omitted_by_n[point['n_device']]['dominated_by_n_devices'])
                                 if point['n_device'] in omitted_by_n else ''),
                             within_paper_x_range=point['violation_pct_upper'] <= 10,
                             sources=';'.join(point['sources'])))
        for label, energy_key in ((RAW, 'raw_energy_kj'),):
            series[label] = [dict(n_device=p['n_device'],
                                  slo_violation_rate=p['violation_pct_upper'],
                                  energy_consumption=p[energy_key]) for p in displayed]
            order.append(label)
            styles[label] = STYLES[label]
            legend_styles[label] = STYLES[label]
            for point in displayed:
                if point['violation_pct_lower'] != point['violation_pct_upper']:
                    axis.errorbar(point['violation_pct_upper'], point[energy_key],
                                  xerr=[[point['violation_pct_upper'] - point['violation_pct_lower']], [0]],
                                  fmt='none', ecolor=STYLES[label]['color'], capsize=2, zorder=4)
        paper._plot_metric_pair(
            axis, series, order, styles, x_key='slo_violation_rate',
            y_key='energy_consumption', sort_key='n_device', gradient_key='n_device',
            solid_markers=True, line_collection_cls=LineCollection, style_scale=.8)
        paper._apply_slo_violation_log_xscale(
            axis, series_by_method=series, max_slo_violation=10, include_left_margin=True)
        paper._add_zero_slo_reference_line(axis)
        for target in (1, 5):
            axis.axvline(target, color='#888888', linestyle='--', linewidth=.65, alpha=.7)
        axis.set_ylim(bottom=0)
        axis.set_title(title)
        axis.set_xlabel('SLO violation rate (%)')
        axis.set_ylabel('Energy (kJ)')
        axis.grid(axis='y', alpha=.2)
    preferred = ['vLLM', 'Llumnix', 'Sarathi', 'SLO Packer', RAW]
    labels = [label for label in preferred if label in legend_styles]
    labels += [label for label in legend_styles if label not in labels]
    handles = [Line2D([], [], label=label, color=legend_styles[label]['color'],
                      marker=legend_styles[label]['marker'], lw=1.8, markersize=6)
               for label in labels]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .995),
               ncol=5, frameon=False, fontsize=10)
    omitted_label = ', '.join(str(point['n_device']) for point in omissions) or 'none'
    fig.text(.5, .02,
             f'BEAM Coder Pareto frontier: dominated N={omitted_label} omitted from the plot only; all measurements retained in CSV. Mixed unchanged.\n'
             'BEAM: unscaled PCIe energy × replica count (shard estimate), not hardware-matched to SXM baselines. No idle add-back.\n'
             'BEAM TPOT 25 ms + total TTFT allowance 200 ms; original Mixed baselines use 20 ms TPOT.',
             ha='center', va='bottom', fontsize=8, color='#555555')
    fig.tight_layout(rect=(0, .16, 1, .89), w_pad=2)
    stem = args.output_dir / 'coder_mixed_unscaled'
    for extension in ('png', 'pdf'):
        fig.savefig(f'{stem}.{extension}', dpi=240, bbox_inches='tight')
    plt.close(fig)
    with (args.output_dir / 'beam_points.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(source_preview=source_name(args.source_preview),
                  source_preview_sha256=file_hash(args.source_preview),
                  original_sources={w: source['original_sources'][w] for w in ('code_7b', 'mixed_7b')},
                  x_axis='Original paper 0-10% piecewise-focus scale; CSV retains all points.',
                  raw_energy_formula='N * mean(raw_gpu_energy_j) / 1000',
                  retained_point_values_unchanged=True, original_curves_unchanged=True,
                  beam_energy_variant='unscaled PCIe only; no affine hardware normalization',
                  point_selection='Coder only: minimize (raw energy, upper-bound SLO violation); Mixed unchanged.',
                  omitted_coder_points=omissions,
                  raw_measurements_retained_in_csv=True,
                  paper_assets_modified=False,
                  limitations=['Unscaled BEAM is PCIe, original baselines are SXM: not hardware-matched.',
                               'BEAM curves extrapolate independent shards, not measured cluster totals.',
                               'Coder BEAM alone is Pareto-filtered; excluded points are not claimed to be erroneous measurements.',
                               'Full monitoring windows retained; no idle addback or duration normalization.',
                               'BEAM Mixed evaluation remains 25 ms versus original baseline 20 ms.'])
    (args.output_dir / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
