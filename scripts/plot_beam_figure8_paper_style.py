#!/usr/bin/env python3
"""Use the original Figure 8 renderer, then overlay scaled BEAM measurements.

Paper panels retain indices 0/4/7. The user-approved middle-panel proxy uses
measured BEAM index 3 for paper index 4; it is not a measurement of index 4.
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
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plots import timeline_comparison_replay as renderer
from scripts.plot_beam_figure8_mixed import read_beam_run, original_series, file_hash, scale_power, source_name

OUTPUT = ROOT / 'analysis_outputs/beam_figure8_paper_style'
PANEL_TO_BEAM = {0: 0, 4: 3, 7: 7}
SNAPSHOT_DIR = ROOT / 'figure_data/beam_paper/figure8'
METADATA = ROOT / 'figure_data/beam_paper/timeline/energy_consumption-mixed.meta.json'


def load_snapshot(snapshot_dir, metadata, edges):
    """Load the frozen per-device curves without reopening measurement archives."""
    with (snapshot_dir / 'aligned_power.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    report = json.loads((snapshot_dir / 'provenance.json').read_text())
    if len(rows) != len(PANEL_TO_BEAM) * (len(edges) - 1):
        raise ValueError('Snapshot must contain every panel and one-second bin')
    runs = {run['paper_device_index']: run for run in report['beam_runs']}
    measurements = {}
    for paper_index, beam_index in PANEL_TO_BEAM.items():
        selected = [row for row in rows if int(row['paper_device_index']) == paper_index]
        times = np.asarray([float(row['time_s']) for row in selected])
        if len(times) != len(edges) - 1 or not np.array_equal(times, edges[:-1]):
            raise ValueError('Snapshot must contain complete ordered one-second bins per panel')
        if any(int(row['beam_source_device_index']) != beam_index
               or row['beam_is_proxy'] != str(paper_index != beam_index) for row in selected):
            raise ValueError('Snapshot must preserve the recorded BEAM proxy mapping')
        for label, column in (('Baseline', 'vllm_power_w'), ('Ours', 'slo_packer_power_w')):
            if not np.array_equal([float(row[column]) for row in selected],
                                  original_series(metadata, label, paper_index, edges)):
                raise ValueError('Snapshot original curves do not match the metadata')
        raw = np.asarray([float(row['beam_raw_pcie_power_w']) for row in selected])
        scaled = np.asarray([float(row['beam_scaled_power_w']) for row in selected])
        if (not np.isfinite(raw).all() or not np.isfinite(scaled).all()
                or not np.allclose(scaled, scale_power(raw), rtol=0, atol=1e-9)):
            raise ValueError('Snapshot power must be finite and preserve the recorded affine scaling')
        if paper_index not in runs or runs[paper_index]['device_index'] != beam_index:
            raise ValueError('Snapshot provenance does not match the BEAM proxy mapping')
        measurements[paper_index] = (raw, scaled, runs[paper_index])
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=METADATA)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--snapshot-dir', type=Path, default=SNAPSHOT_DIR)
    parser.add_argument('--from-raw', action='store_true',
                        help='Recompute aligned power from the optional original measurement archives.')
    args = parser.parse_args()
    metadata_path = args.metadata
    metadata = json.loads(metadata_path.read_text())
    edges = np.arange(601, dtype=float)
    sources = {
        0: ROOT / 'beam-figure7-7b-20261020-215014/mixed_7b/split_8_0',
        3: ROOT / 'beam-jobs-mlsys/mixed/split_8_3',
        7: ROOT / 'beam-jobs-mlsys/mixed/split_8_7',
    }
    measurements = ({paper_index: read_beam_run(sources[beam_index], beam_index, edges)
                     for paper_index, beam_index in PANEL_TO_BEAM.items()}
                    if args.from_raw else load_snapshot(args.snapshot_dir, metadata, edges))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_dir / 'figure8_mixed_beam_scaled'
    # These are the exact evaluation-layout arguments used by the original
    # replay_timeline_comparisons(..., layout='evaluation') implementation.
    fig, axes = renderer.render_timeline_comparison_metadata(
        metadata, output_stem=output_stem, device_ids=tuple(PANEL_TO_BEAM),
        xlim=(0.0, 600.0), show_arrivals=False, show_total_power=False,
        show_device_power=True, show_active_devices=False, bottom_panel=None,
        row_height_in=.86, min_fig_height_in=3.55,
        device_panel_tags=('TOP', 'MID', 'BOTTOM'), shared_ylabel='Power (W)',
        shared_ylabel_x=-.075, legend_y_anchor=1.10, font_scale=1.20,
    )
    assert len(axes) == 3
    original_lines = [tuple(axis.lines) for axis in axes]
    original_limits = [(axis.get_xlim(), axis.get_ylim()) for axis in axes]
    csv_rows = []
    provenance = []
    for axis, (paper_index, beam_index) in zip(axes, PANEL_TO_BEAM.items()):
        assert len(axis.lines) == 2
        raw, scaled, info = measurements[paper_index]
        if not np.all(np.isfinite(scaled)):
            raise ValueError('This comparison requires complete 0-600s BEAM coverage')
        if np.max(scaled) > axis.get_ylim()[1] or np.min(scaled) < axis.get_ylim()[0]:
            raise ValueError('BEAM would be clipped by the original power limits')
        axis.step(edges, np.r_[scaled, scaled[-1]], where='post', color='#8E44AD',
                  lw=1.0, linestyle='-', label='BEAM')
        baseline = original_series(metadata, 'Baseline', paper_index, edges)
        ours = original_series(metadata, 'Ours', paper_index, edges)
        for i, time in enumerate(edges[:-1]):
            csv_rows.append(dict(time_s=time, paper_device_index=paper_index,
                                 beam_source_device_index=beam_index,
                                 beam_is_proxy=paper_index != beam_index,
                                 vllm_power_w=baseline[i], slo_packer_power_w=ours[i],
                                 beam_raw_pcie_power_w=raw[i], beam_scaled_power_w=scaled[i]))
        provenance.append(dict(info, paper_device_index=paper_index,
                               beam_is_proxy=paper_index != beam_index))
    # Retain the renderer's original two curves and limits, replacing only its
    # legend so that the added BEAM artist is included.
    for axis, lines, limits in zip(axes, original_lines, original_limits):
        assert tuple(axis.lines[:2]) == lines
        assert (axis.get_xlim(), axis.get_ylim()) == limits
    handles, labels = axes[0].get_legend_handles_labels()
    for legend in list(fig.legends):
        legend.remove()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.10),
               fontsize=9.2, frameon=False, handlelength=1.35, handletextpad=.35,
               columnspacing=.65, ncol=3)
    for extension in ('png', 'pdf'):
        fig.savefig(f'{output_stem}.{extension}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    with (args.output_dir / 'aligned_power.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    report = {
        'renderer': 'plots/timeline_comparison_replay.py:render_timeline_comparison_metadata',
        'renderer_sha256': file_hash(ROOT / 'plots/timeline_comparison_replay.py'),
        'original_metadata': source_name(metadata_path),
        'original_metadata_sha256': file_hash(metadata_path),
        'input_mode': 'raw_archives' if args.from_raw else 'frozen_aligned_snapshot',
        'snapshot_sources': {} if args.from_raw else {
            source_name(args.snapshot_dir / name): file_hash(args.snapshot_dir / name)
            for name in ('aligned_power.csv', 'provenance.json')},
        'paper_device_indices': list(PANEL_TO_BEAM),
        'paper_to_beam_source_index': PANEL_TO_BEAM,
        'proxy_assumption': 'User-approved similarity: BEAM GPU 4 (index 3) proxies paper GPU 5 (index 4).',
        'scaling': '70 + (P_pcie - 52) * 330/248, no clipping; not measured SXM power',
        'original_curves_and_limits_preserved': True,
        'presentation': {
            'beam_linestyle': 'solid', 'added_plot_caption': False,
            'proxy_annotation_on_plot': False,
            'proxy_documentation': 'Preserved here and in aligned_power.csv, omitted from plot at user request.',
        },
        'beam_runs': provenance,
        'limitations': [
            'No BEAM shard-4 measurement is available; the middle curve is explicitly a proxy.',
            'BEAM shards were replayed independently, with 20ms TBT for shard0 and 25ms for shards3/7.',
            'Power time alignment is approximate, using the previous measured-overlay alignment methods.',
            'This rendering script does not modify the original renderer, paper metadata, or experimental results.',
        ],
    }
    (args.output_dir / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(output_stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
