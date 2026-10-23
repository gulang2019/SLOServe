#!/usr/bin/env python3
"""Preview Figure 7 with energy/SLO panels only and explicit BEAM estimates.

Writes only new analysis artifacts, never paper assets or original results.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plots import chat_7b_n_device_triptych as paper
from plots.common import get_method_label
from scripts.beam_figure7_metrics import collect_beam_points
from scripts.plot_beam_figure8_mixed import file_hash, source_name

WORKLOADS = (
    ('chat_7b', 'chat_7B', '(a) ChatBot — 7B'),
    ('code_7b', 'code_7B', '(b) Coder — 7B'),
    ('mixed_7b', 'mixed_7B', '(c) Mixed — 7B'),
    ('chat_32b', '32B_chat', '(d) ChatBot — 32B'),
    ('multiround_chat', 'e2e_mulitiround_chatting', '(e) Multi-round chatting — 7B'),
    ('large_scale_chat', 'chatting_large_scale', '(f) Large-scale chatting — 7B'),
)
BEAM_KEY = 'beam / round_robin'
BEAM_STYLE = dict(color='#8E44AD', marker='X', linewidth=1.6, markersize=5.0)


def load_paper_series(csv_path):
    rows, duplicates = paper._load_rows(csv_path)
    series, order = paper._build_method_series(rows)
    series, order = paper._filter_default_e2e_methods(series, order)
    series, order, selections = paper._collapse_baseline_families(
        series, order, family_specs=paper.DEFAULT_BASELINE_FAMILY_COLLAPSE)
    series, order = paper._split_methods_by_serving_mode(series, order)['normal']
    with csv_path.open(newline='') as handle:
        raw_rows = list(csv.DictReader(handle))
    context = {key: sorted({row[key] for row in raw_rows if row.get(key)})
               for key in paper.CONSTANT_FIELDS}
    return series, order, dict(duplicate_rows=duplicates, family_selections=selections,
                               context=context, source=source_name(csv_path),
                               sha256=file_hash(csv_path))


def render_large_scale_overview(beam_points, output_dir):
    """Companion view showing the points cropped by the paper's 10% limit."""
    series, order, styles = {}, [BEAM_KEY], {BEAM_KEY: BEAM_STYLE}
    selected = sorted((p for p in beam_points if p['workload'] == 'large_scale_chat'),
                      key=lambda p: p['n_device'])
    series[BEAM_KEY] = [dict(n_device=p['n_device'],
                            slo_violation_rate=p['violation_pct_upper'],
                            energy_consumption=p['energy_kj']) for p in selected]
    fig, axis = plt.subplots(figsize=(7.1, 4.5))
    paper._plot_metric_pair(
        axis, series, order, styles, x_key='slo_violation_rate',
        y_key='energy_consumption', sort_key='n_device', gradient_key='n_device',
        solid_markers=True, line_collection_cls=LineCollection, style_scale=.8)
    for point in selected:
        axis.annotate(f"N={point['n_device']}",
                      (point['violation_pct_upper'], point['energy_kj']),
                      xytext=(5, 7), textcoords='offset points',
                      color=BEAM_STYLE['color'], fontsize=8.5)
    axis.axvspan(0, 10, color='#eeeeee', zorder=0)
    axis.axvline(10, color='#888888', linestyle='--', lw=.8)
    axis.set_xlim(0, 70)
    axis.set_ylim(bottom=0)
    axis.set_xlabel('SLO violation rate (%) — full linear range')
    axis.set_ylabel('Energy (kJ)')
    axis.set_title('BEAM: large-scale chatting — 7B\nGray region: Figure 7’s 0–10% view')
    axis.grid(axis='y', alpha=.2)
    handles = [Line2D([], [], color=styles[key]['color'], marker=styles[key]['marker'],
                      linewidth=1.8, markersize=6,
                      label='BEAM' if key == BEAM_KEY else get_method_label(key)) for key in order]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .035),
               ncol=1, frameon=False, fontsize=9)
    fig.text(.5, .012, 'Same BEAM data as preview: scaled shard estimate; 25 ms TPOT, 50 ms total TTFT allowance.',
             ha='center', fontsize=7.5, color='#555555')
    fig.tight_layout(rect=(0, .12, 1, 1))
    for extension in ('png', 'pdf'):
        fig.savefig(output_dir / f'large_scale_chat_full_range.{extension}',
                    dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'Paper/data/e2e')
    parser.add_argument('--output-dir', type=Path,
                        help='Default: a separate directory named for the coding source and multi-round TPOT.')
    parser.add_argument('--coding-source', choices=('old', 'latest'), default='old',
                        help='Old uses the complete original shard-0 Coding sweep.')
    parser.add_argument('--multiround-tpot-ms', type=float, default=25,
                        help='Post-hoc BEAM multi-round evaluation budget; recorded scheduler target remains 50 ms.')
    args = parser.parse_args()
    if not math.isfinite(args.multiround_tpot_ms) or args.multiround_tpot_ms <= 0:
        parser.error('--multiround-tpot-ms must be finite and positive')
    if args.output_dir is None:
        args.output_dir = ROOT / (
            f'analysis_outputs/beam_figure7_energy_preview_{args.coding_source}_coding'
            f'_multiround_tpot{args.multiround_tpot_ms:g}ms')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    beam_points = collect_beam_points(ROOT, coding_source=args.coding_source,
                                     multiround_tpot_s=args.multiround_tpot_ms / 1000)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                         'axes.labelsize': 10, 'axes.titlesize': 11,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'pdf.fonttype': 42, 'axes.linewidth': .8})
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4))
    legend_styles = {}
    exported = []
    sources = {}
    visibility = {}
    for axis, (workload, stem, title) in zip(axes.flat, WORKLOADS):
        series, order, info = load_paper_series(args.data_dir / f'{stem}.csv')
        sources[workload] = info
        styles = paper._build_method_styles(order)
        for method in order:
            legend_styles.setdefault(method, styles[method])
            for point in series[method]:
                exported.append(dict(workload=workload, method=get_method_label(method),
                                     n_device=point['n_device'],
                                     violation_pct=point['slo_violation_rate'],
                                     energy_kj=point['energy_consumption'],
                                     violation_pct_lower=point['slo_violation_rate'],
                                     energy_extrapolated_for_preview=False,
                                     within_paper_x_range=point['slo_violation_rate'] <= 10))
        selected = sorted((point for point in beam_points if point['workload'] == workload),
                          key=lambda point: point['n_device'])
        if selected:
            series[BEAM_KEY] = [dict(method=BEAM_KEY, n_device=point['n_device'],
                                    slo_violation_rate=point['violation_pct_upper'],
                                    energy_consumption=point['energy_kj']) for point in selected]
            order.append(BEAM_KEY)
            styles[BEAM_KEY] = BEAM_STYLE
            legend_styles[BEAM_KEY] = BEAM_STYLE
            for point in selected:
                exported.append(dict(workload=workload, method='BEAM',
                                     n_device=point['n_device'],
                                     violation_pct=point['violation_pct_upper'],
                                     energy_kj=point['energy_kj'],
                                     violation_pct_lower=point['violation_pct_lower'],
                                     energy_extrapolated_for_preview=True,
                                     within_paper_x_range=point['violation_pct_upper'] <= 10))
                if point['violation_pct_lower'] != point['violation_pct_upper']:
                    axis.errorbar(point['violation_pct_upper'], point['energy_kj'],
                                  xerr=[[point['violation_pct_upper'] - point['violation_pct_lower']], [0]],
                                  fmt='none', ecolor=BEAM_STYLE['color'], capsize=2, zorder=4)
        else:
            axis.text(.97, .05, 'BEAM: no 32B measurements', transform=axis.transAxes,
                      ha='right', va='bottom', fontsize=8.5, color='#666666')
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
        axis.set_title(title, pad=8)
        axis.set_xlabel('SLO violation rate (%)')
        axis.set_ylabel('Energy (kJ)')
        axis.grid(axis='y', alpha=.2)
        if selected and all(p['violation_pct_upper'] > 10 for p in selected):
            axis.text(.97, .05, 'BEAM: all points above 10%', transform=axis.transAxes,
                      ha='right', va='bottom', fontsize=8.5, color=BEAM_STYLE['color'])
        visibility[workload] = {
            'beam_devices_visible': [p['n_device'] for p in selected if p['violation_pct_upper'] <= 10],
            'beam_devices_above_10_pct': [p['n_device'] for p in selected if p['violation_pct_upper'] > 10],
        }
    preferred_order = ['vllm / round_robin', 'qlm / llumnix_load', 'sarathi / round_robin',
                       'atfc / slosserve_planner', 'vllm / round_robin_session', BEAM_KEY]
    methods = [key for key in preferred_order if key in legend_styles]
    methods += [key for key in legend_styles if key not in methods]
    handles = []
    seen_legend_styles = set()
    for key in methods:
        label = 'BEAM' if key == BEAM_KEY else get_method_label(key)
        style = legend_styles[key]
        identity = (label, style['color'], style['marker'])
        if identity in seen_legend_styles:
            continue
        seen_legend_styles.add(identity)
        handles.append(Line2D([], [], color=style['color'], marker=style['marker'],
                              linewidth=1.8, markersize=6, label=label))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .99),
               ncol=len(handles), frameon=False, fontsize=10, columnspacing=1.3)
    fig.text(.5, .024,
             'Preview: original paper energies retained; BEAM uses scaled full-monitor energy × replica count (shard estimate).\n'
             f'BEAM evaluation: 25 ms single-round, {args.multiround_tpot_ms:g} ms multi-round; recorded multi-round scheduler target remains 50 ms.\n'
             'Original Mixed/Large-scale/multi-round baselines retain 20/20/50 ms TPOT; this is a post-hoc sensitivity comparison.',
             ha='center', va='bottom', fontsize=8, color='#555555')
    fig.tight_layout(rect=(0, .10, 1, .93), w_pad=1.4, h_pad=1.7)
    output_stem = args.output_dir / 'figure7_energy_only'
    for extension in ('png', 'pdf'):
        fig.savefig(f'{output_stem}.{extension}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    render_large_scale_overview(beam_points, args.output_dir)
    with (args.output_dir / 'plotted_points.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    report = {
        'figure': 7, 'preview_only': True, 'paper_assets_modified': False,
        'panels': ['slo_violation_rate_vs_energy'], 'original_sources': sources,
        'beam_points': beam_points, 'visibility': visibility,
        'x_axis': 'Original paper piecewise-focus mapping, 0-10%; out-of-range points retained in CSV.',
        'energy_convention': 'Original saved paper energy, no idle addback. BEAM full monitor energy is affine-scaled per GPU then N times mean of selected shards.',
        'coding_source': args.coding_source,
        'beam_multiround_evaluation_tpot_ms': args.multiround_tpot_ms,
        'beam_selection': (
            ('Coding uses the complete old shard-0 sweep. ' if args.coding_source == 'old'
             else 'Coding uses latest reruns where available. ')
            + 'Other workloads use latest archive for each workload/N when available; retain earlier runs at other N. Do not mix old/new shard measurements at the same N.'),
        'companion_plot': 'large_scale_chat_full_range.png shows all five BEAM points over a linear 0-70% axis.',
        'scoring': 'SLOServe cumulative token deadlines, p90 lateness <=0 and TTFT pass; conservative upper violation bound for unlocated tokens.',
        'beam_total_allowances_ms': {'chat_7b': 100, 'code_7b': 200, 'mixed_7b': 200,
                                     'large_scale_chat': 50, 'multiround_chat': 50},
        'limitations': [
            'BEAM is a shard-based extrapolation, not a full-cluster energy or SLO measurement.',
            'BEAM-only 25ms evaluation continues the earlier preview sensitivity; original Mixed/Large-scale CSVs use 20ms. Not an equal-SLO comparison.',
            f'BEAM multi-round is evaluated at {args.multiround_tpot_ms:g} ms TPOT; archived scheduler targets and original multi-round baseline scores remain at 50 ms.',
            'BEAM control targets can differ from evaluation targets; post-hoc scoring does not alter measured clocks, latencies, or energy.',
            'Full-monitor durations differ across runs and are not identical to the original paper measurement windows.',
            'No BEAM 32B measurements are available; that panel is unchanged.',
            'Original savings-percentage annotations are omitted rather than reused after adding a baseline.',
        ],
    }
    (args.output_dir / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(output_stem.with_suffix('.png'))
    print(json.dumps(visibility, indent=2))


if __name__ == '__main__':
    main()
