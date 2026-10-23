#!/usr/bin/env python3
"""Replay the original Figure 3 and overlay an estimated BEAM cluster curve.

Only three RR shards are available, not a measured eight-GPU cluster. The
user-selected estimate is 8 * mean(scaled power of shards 0, 3, 7). These
independent runs mix 20/25 ms TBT targets and have approximate time alignment.
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
from scripts.plot_beam_figure8_mixed import file_hash, read_beam_run, scale_power, source_name

SNAPSHOT_DIR = ROOT / 'figure_data/beam_paper/figure3'
METADATA = ROOT / 'figure_data/beam_paper/timeline/energy_consumption-mixed.meta.json'

SOURCES = {
    0: ROOT / 'beam-figure7-7b-20261020-215014/mixed_7b/split_8_0',
    3: ROOT / 'beam-jobs-mlsys/mixed/split_8_3',
    7: ROOT / 'beam-jobs-mlsys/mixed/split_8_7',
}


def estimate_cluster_power(scaled_shards, n_device=8):
    """Extrapolate the sample mean; never silently drop uncovered samples."""
    if not isinstance(n_device, int) or isinstance(n_device, bool) or n_device < 1:
        raise ValueError('n_device must be a positive integer')
    if not scaled_shards:
        raise ValueError('At least one measured shard is required')
    arrays = [np.asarray(values, dtype=float) for values in scaled_shards.values()]
    if (any(a.ndim != 1 or a.size == 0 for a in arrays)
            or len({a.size for a in arrays}) != 1
            or any(not np.isfinite(a).all() for a in arrays)):
        raise ValueError('All shards must have equal-length, finite, complete power bins')
    return n_device * np.mean(np.stack(arrays), axis=0)


def original_total_power(metadata, label, edges):
    trace = next(t for t in metadata['traces'] if t['label'] == label)
    if trace['n_device'] != 8:
        raise ValueError('Figure 3 requires the original eight-device traces')
    power = trace['power']
    times = np.asarray(power['time_s'], dtype=float)
    values = np.asarray(power['values_w'], dtype=float)
    indices = np.searchsorted(times, edges[:-1])
    if (len(times) != len(values) or np.any(indices >= len(times))
            or not np.allclose(times[indices], edges[:-1])):
        raise ValueError('Original metadata must supply matching one-second bins')
    return values[indices]


def load_snapshot(snapshot_dir, metadata, edges, shards):
    """Replay frozen plotted bins, checking them against the original metadata."""
    with (snapshot_dir / 'aligned_power.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    report = json.loads((snapshot_dir / 'provenance.json').read_text())
    times = np.asarray([float(row['time_s']) for row in rows])
    if len(times) != len(edges) - 1 or not np.array_equal(times, edges[:-1]):
        raise ValueError('Snapshot must contain the complete ordered one-second bins')
    for label, column in (('Baseline', 'vllm_total_power_w'), ('Ours', 'slo_packer_total_power_w')):
        if not np.array_equal([float(row[column]) for row in rows],
                              original_total_power(metadata, label, edges)):
            raise ValueError('Snapshot original curves do not match the metadata')
    runs = {run['device_index']: run for run in report['beam_runs']}
    measurements = {}
    for shard in shards:
        if shard not in report['beam_source_shards'] or shard not in runs:
            raise ValueError(f'Snapshot does not contain BEAM shard {shard}')
        raw = np.asarray([float(row[f'beam_shard_{shard}_raw_power_w']) for row in rows])
        scaled = np.asarray([float(row[f'beam_shard_{shard}_scaled_power_w']) for row in rows])
        if (not np.isfinite(raw).all() or not np.isfinite(scaled).all()
                or not np.allclose(scaled, scale_power(raw), rtol=0, atol=1e-9)):
            raise ValueError('Snapshot power must be finite and preserve the recorded affine scaling')
        measurements[shard] = (raw, scaled, runs[shard])
    if set(shards) == set(report['beam_source_shards']):
        expected = estimate_cluster_power({i: run[1] for i, run in measurements.items()})
        recorded = np.asarray([float(row['beam_estimated_total_power_w']) for row in rows])
        if not np.allclose(expected, recorded, rtol=0, atol=1e-9):
            raise ValueError('Snapshot cluster estimate differs from its measured shard mean')
    return measurements


def render_figure(metadata, cluster_power, edges, output_stem):
    # Exact --layout intro arguments: the active paper PDF reproduces byte-for-
    # byte after creation-date normalization. Its old PNG is a stale layout.
    fig, axes = renderer.render_timeline_comparison_metadata(
        metadata, output_stem=output_stem, xlim=(0.0, 600.0),
        show_arrivals=True, show_total_power=True, show_device_power=False,
        show_active_devices=False, bottom_panel=None, row_height_in=.70,
        min_fig_height_in=2.85, arrival_ylabel='Request / Window',
        total_power_ylabel='Power (W)',
        legend_y_anchor=1.10,
    )
    if len(axes) != 2:
        raise ValueError('Expected arrival and total-power panels')
    original_lines = [tuple(axis.lines) for axis in axes]
    axes[1].step(edges, np.r_[cluster_power, cluster_power[-1]], where='post',
                 color='#8E44AD', lw=1.0, linestyle='-', label='BEAM')
    for axis, lines in zip(axes, original_lines):
        assert tuple(axis.lines[:len(lines)]) == lines
    handles, labels = axes[1].get_legend_handles_labels()
    for legend in list(fig.legends):
        legend.remove()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.10),
               fontsize=8.0, frameon=False, handlelength=1.35, handletextpad=.35,
               columnspacing=.65, ncol=3)
    for extension in ('png', 'pdf'):
        fig.savefig(f'{output_stem}.{extension}', dpi=300, bbox_inches='tight')
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path,
                        default=METADATA)
    parser.add_argument('--from-raw', action='store_true',
                        help='Recompute aligned power from the optional original measurement archives.')
    parser.add_argument('--snapshot-dir', type=Path, default=SNAPSHOT_DIR)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'analysis_outputs/beam_figure3_paper_style')
    parser.add_argument('--shards', type=int, nargs='+', choices=tuple(SOURCES),
                        default=[0, 3, 7], help='Independent RR shards used for the estimate')
    args = parser.parse_args()
    if len(args.shards) != len(set(args.shards)):
        parser.error('--shards must not contain duplicates')
    metadata = json.loads(args.metadata.read_text())
    edges = np.arange(601, dtype=float)
    measurements = ({i: read_beam_run(SOURCES[i], i, edges) for i in args.shards}
                    if args.from_raw else load_snapshot(args.snapshot_dir, metadata, edges, args.shards))
    scaled_shards = {i: values[1] for i, values in measurements.items()}
    cluster_power = estimate_cluster_power(scaled_shards)
    baseline = original_total_power(metadata, 'Baseline', edges)
    ours = original_total_power(metadata, 'Ours', edges)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_dir / 'figure3_mixed_beam_estimated'
    fig, _ = render_figure(metadata, cluster_power, edges, output_stem)
    plt.close(fig)
    rows = []
    for index, time in enumerate(edges[:-1]):
        row = dict(time_s=time, vllm_total_power_w=baseline[index],
                   slo_packer_total_power_w=ours[index],
                   beam_estimated_total_power_w=cluster_power[index])
        for shard, (raw, scaled, _) in measurements.items():
            row[f'beam_shard_{shard}_raw_power_w'] = raw[index]
            row[f'beam_shard_{shard}_scaled_power_w'] = scaled[index]
        rows.append(row)
    with (args.output_dir / 'aligned_power.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        'figure': 3, 'workload': 'mixed', 'server_count': 8,
        'window_s': [0, 600], 'bin_width_s': 1,
        'beam_is_estimated_cluster_power': True,
        'beam_source_shards': args.shards,
        'estimator': '8 * arithmetic_mean(selected scaled one-GPU shard powers), per time bin',
        'power_scaling_per_gpu': '70 + (P_pcie - 52) * 330/248, no clipping',
        'renderer': 'plots/timeline_comparison_replay.py:render_timeline_comparison_metadata',
        'renderer_sha256': file_hash(ROOT / 'plots/timeline_comparison_replay.py'),
        'script_sha256': file_hash(Path(__file__)),
        'input_mode': 'raw_archives' if args.from_raw else 'frozen_aligned_snapshot',
        'snapshot_sources': {} if args.from_raw else {
            source_name(args.snapshot_dir / name): file_hash(args.snapshot_dir / name)
            for name in ('aligned_power.csv', 'provenance.json')},
        'original_metadata': source_name(args.metadata),
        'original_metadata_sha256': file_hash(args.metadata),
        'original_curves_preserved': True,
        'presentation': {'beam_legend': 'BEAM', 'beam_linestyle': 'solid',
                         'added_plot_caption': False},
        'beam_runs': [values[2] for values in measurements.values()],
        'plotted_energy_kj': {
            'vllm': float(np.dot(baseline, np.diff(edges)) / 1000),
            'slo_packer': float(np.dot(ours, np.diff(edges)) / 1000),
            'beam_estimated': float(np.dot(cluster_power, np.diff(edges)) / 1000),
        },
        'limitations': [
            'Not measured simultaneous eight-GPU power; sample mean may not represent unmeasured shards or aggregate peaks.',
            'Independent replay targets: shard 0 uses 20 ms TBT; shards 3 and 7 use 25 ms TBT.',
            'PCIe-to-SXM scaling is a heuristic, not measured SXM power.',
            'Time alignment is approximate; original metadata starts at its first power event.',
            'No missing-data imputation, clipping, idle addback, or modification of original curves.',
        ],
    }
    (args.output_dir / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(output_stem.with_suffix('.png'))
    print(json.dumps(report['plotted_energy_kj'], indent=2))


if __name__ == '__main__':
    main()
