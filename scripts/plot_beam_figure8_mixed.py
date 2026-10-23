#!/usr/bin/env python3
"""Plot mixed Figure 8 with three distinct, independently measured BEAM shards.

Original paper data and the earlier repeated-shard sketch are left unchanged.
BEAM PCIe power is linearly scaled, not a measurement on SXM hardware.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEVICES = (0, 3, 7)


def scale_power(power):
    return 70.0 + (np.asarray(power) - 52.0) * 330.0 / 248.0


def interval_mean_power(times_s, energy_j, edges_s, max_gap_s=1.0):
    """Average counter-difference power; no extrapolation or gap filling."""
    times = np.asarray(times_s, dtype=float)
    energy = np.asarray(energy_j, dtype=float)
    edges = np.asarray(edges_s, dtype=float)
    if (times.ndim != 1 or energy.ndim != 1 or edges.ndim != 1
            or len(times) < 2 or len(times) != len(energy) or len(edges) < 2
            or not all(np.all(np.isfinite(a)) for a in (times, energy, edges))
            or not np.isfinite(max_gap_s) or max_gap_s <= 0):
        raise ValueError('Expected finite aligned 1D arrays and positive max_gap_s')
    if np.any(np.diff(times) <= 0) or np.any(np.diff(edges) <= 0):
        raise ValueError('Sample times and bin edges must be strictly increasing')
    if np.any(np.diff(energy) < 0):
        raise ValueError('Cumulative energy must not decrease')
    power = np.diff(np.interp(edges, times, energy)) / np.diff(edges)
    covered = (edges[:-1] >= times[0]) & (edges[1:] <= times[-1])
    for index in np.flatnonzero(np.diff(times) > max_gap_s):
        covered &= ~((edges[:-1] < times[index + 1]) & (edges[1:] > times[index]))
    power[~covered] = np.nan
    return power


def estimate_replay_start(result, final_batch_time=None):
    if final_batch_time is not None:
        completion = max(t + d for t, d in zip(result['entry_times'], result['e2el']))
        return final_batch_time - timedelta(seconds=completion)
    # date is written after metric calculation and truncated to whole seconds.
    # Midsecond reduces rounding bias but does NOT remove postprocessing delay.
    return (datetime.strptime(result['date'], '%Y%m%d-%H%M%S')
            + timedelta(seconds=0.5 - float(result['duration'])))


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_name(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def read_beam_run(path, device, edges):
    detail_path = path / 'beam_detailed.json'
    power_path = path / 'power_timeline_gpu_0.csv'
    request_path = path / 'per_request_slo.jsonl'
    config_path = path / 'slo_config.txt'
    result = json.loads(detail_path.read_text())
    config = dict(line.split('=', 1) for line in config_path.read_text().splitlines() if '=' in line)
    with request_path.open() as handle:
        first_request = json.loads(next(handle))
    source_timestamp = datetime.strptime(first_request['trace_timestamp'], '%Y-%m-%d %H:%M:%S.%f%z')
    trace_offset = (source_timestamp - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()
    batch_path = path / 'batch_log_GPU_0.csv'
    final_batch = None
    if batch_path.exists():
        with batch_path.open(newline='') as handle:
            final_batch = max(datetime.fromisoformat(row['current_time']) for row in csv.DictReader(handle))
    start = estimate_replay_start(result, final_batch)
    with power_path.open(newline='') as handle:
        samples = list(csv.DictReader(handle))
    # Both stamps belong to the same remote local clock. Subtract naive datetimes
    # directly instead of applying this plotting machine's timezone to them.
    times = np.asarray([(datetime.fromisoformat(row['timestamp']) - start).total_seconds()
                        + trace_offset for row in samples])
    energy = np.asarray([float(row['cumulative_energy_mj']) / 1000 for row in samples])
    raw = interval_mean_power(times, energy, edges)
    scaled = scale_power(raw)
    hashes = {source_name(p): file_hash(p) for p in (detail_path, power_path, request_path, config_path)}
    if final_batch is not None:
        hashes[source_name(batch_path)] = file_hash(batch_path)
    info = {
        'device_index': device, 'source_directory': source_name(path),
        'requests': result['completed'], 'benchmark_duration_s': result['duration'],
        'beam_ttft_target_s': float(config['beam_global_ttft_s']),
        'beam_tbt_target_s': float(config['beam_tbt_s']),
        'estimated_replay_start_remote_local': start.isoformat(),
        'alignment_method': ('final_batch_minus_last_client_completion' if final_batch is not None
                             else 'result_date_midsecond_minus_duration'),
        'alignment_is_approximate': True,
        'restored_global_trace_offset_s': trace_offset,
        'monitor_relative_global_start_s': float(times[0]),
        'monitor_relative_global_end_s': float(times[-1]),
        'full_monitor_raw_energy_kj': float(energy[-1] - energy[0]) / 1000,
        'plotted_bins': len(raw), 'covered_bins': int(np.isfinite(raw).sum()),
        'plotted_raw_energy_kj': float(np.nansum(raw * np.diff(edges))) / 1000,
        'plotted_scaled_energy_kj': float(np.nansum(scaled * np.diff(edges))) / 1000,
        'date_anchor_minus_selected_anchor_s': (estimate_replay_start(result) - start).total_seconds(),
        'source_sha256': hashes,
    }
    return raw, scaled, info


def original_series(metadata, label, device, edges):
    trace = next(t for t in metadata['traces'] if t['label'] == label)
    if trace['n_device'] != 8:
        raise ValueError('Figure 8 requires the eight-device original traces')
    power = trace['power']
    times = np.asarray(power['time_s'], dtype=float)
    values = np.asarray(power['per_device_values_w'][str(device)], dtype=float)
    indices = np.searchsorted(times, edges[:-1])
    if (len(times) != len(values) or np.any(indices >= len(times))
            or not np.allclose(times[indices], edges[:-1])):
        raise ValueError('Original metadata must supply matching one-second bins')
    return values[indices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-archive-dir', type=Path, default=ROOT / 'beam-figure7-7b-20261020-215014')
    parser.add_argument('--new-archive-dir', type=Path, default=ROOT / 'beam-jobs-mlsys')
    parser.add_argument('--metadata', type=Path, default=ROOT / 'Paper/data/timeline/energy_consumption-mixed.meta.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'analysis_outputs/beam_figure8_mixed_measured')
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    edges = np.arange(601, dtype=float)
    sources = {
        0: args.old_archive_dir / 'mixed_7b/split_8_0',
        3: args.new_archive_dir / 'mixed/split_8_3',
        7: args.new_archive_dir / 'mixed/split_8_7',
    }
    data = {}
    provenance = []
    csv_rows = []
    for device in DEVICES:
        raw, scaled, info = read_beam_run(sources[device], device, edges)
        original = {label: original_series(metadata, label, device, edges)
                    for label in ('Baseline', 'Ours')}
        data[device] = (original, raw, scaled, info)
        provenance.append(info)
        for index, time in enumerate(edges[:-1]):
            csv_rows.append({
                'device_index': device, 'time_s': time,
                'vllm_sxm_power_w': original['Baseline'][index],
                'slo_packer_sxm_power_w': original['Ours'][index],
                'beam_raw_pcie_power_w': raw[index],
                'beam_scaled_power_w': scaled[index],
            })

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.2), sharex=True, sharey=True)
    styles = [('Baseline', 'vLLM (SXM4)', '#4C78A8'),
              ('Ours', 'SLO-Packer (SXM4)', '#2E7D32')]
    all_values = []
    for axis, device, tag in zip(axes, DEVICES, ('TOP', 'MID', 'BOTTOM')):
        original, raw, scaled, info = data[device]
        for label, name, color in styles:
            axis.stairs(original[label], edges, color=color, lw=0.95, alpha=.85,
                        baseline=None, label=name)
            all_values.extend(original[label])
        axis.stairs(scaled, edges, color='#8E44AD', lw=1.15, linestyle='--',
                    baseline=None, label='BEAM (scaled PCIe)')
        all_values.extend(scaled)
        axis.set_title(f'{tag} — GPU {device + 1} (index {device})', loc='left', fontsize=11)
        target = info['beam_tbt_target_s'] * 1000
        batch_label = 'earlier run' if device == 0 else 'new run'
        axis.set_title(f'BEAM shard {device} · TBT target {target:g} ms · {batch_label}',
                       loc='right', fontsize=9, color='#6E3487')
        axis.set_ylabel('GPU power (W)')
        axis.grid(axis='y', alpha=.2)
        axis.set_xlim(0, 600)
    axes[-1].set_xlabel('Workload time (s); cross-run alignment approximate')
    maximum = np.nanmax(np.asarray(all_values))
    axes[0].set_ylim(0, max(450, 50 * np.ceil(maximum / 50)))
    axes[0].set_yticks(np.arange(0, axes[0].get_ylim()[1] + 1, 100))
    fig.suptitle('Mixed workload · 8-server configuration\n'
                 'Distinct measured BEAM shards 0 / 3 / 7, independently replayed',
                 fontsize=13, y=.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .057), ncol=3,
               frameon=False, fontsize=10)
    fig.text(.5, .025, 'BEAM scaling: 70 + (PPCIe − 52) × 330/248; not measured SXM power.\n'
             'Separate runs with different TBT targets—not a simultaneous, uniform-policy 8-GPU measurement.',
             ha='center', va='center', fontsize=8.5, color='#444444')
    fig.tight_layout(rect=(0, .105, 1, .925), h_pad=1.6)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(args.output_dir / f'figure8_mixed_beam_scaled.{extension}', dpi=220)
    plt.close(fig)
    with (args.output_dir / 'aligned_power.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    report = {
        'workload': 'mixed', 'server_count': 8, 'device_indices': list(DEVICES),
        'window_s': [0, 600], 'bin_width_s': 1,
        'power_scaling': '70 + (P_pcie - 52) * 330 / 248; no clipping',
        'original_metadata': source_name(args.metadata),
        'original_metadata_sha256': file_hash(args.metadata), 'beam_runs': provenance,
        'limitations': [
            'Independent one-GPU shard replays, not simultaneous eight-GPU execution.',
            'BEAM old shard 0 target is 20 ms; new shards 3/7 targets are 25 ms. Original paper traces use 20 ms.',
            'Power scaling is a heuristic, not an SXM hardware measurement.',
            'New run starts estimated from saved result date midpoint minus duration; metric/postprocessing delay is unknown.',
            'Old run start estimated from final batch minus final client completion; transport delay is unknown.',
            'Original metadata uses first measured power event as time origin, not a verified common first-request timestamp.',
            'Preserve original metadata values, including bins exceeding 400 W; no idle addback or floor is applied.',
            'Power bins with missing coverage or sample gaps exceeding one second remain NaN.',
            'Use 0-600s plotted interval only; omit later completions/monitoring tails from this figure.',
        ],
    }
    (args.output_dir / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output_dir / 'METHODOLOGY.txt').write_text(
        'Figure 8 mixed: original per-device paper power plus three distinct BEAM measurements.\n'
        'Panels: zero-based device indices 0, 3, 7 (GPU 1, 4, 8).\n'
        'BEAM: one-second mean power from differences of interpolated cumulative energy counters.\n'
        'No extrapolation outside monitor coverage and no interpolation across sample gaps > 1 s.\n'
        'Scaled PCIe power = 70 + (P - 52) * 330/248, without clipping.\n'
        'Restore per-shard trace offsets 0 / 0.553 / 1.351 s after approximate client-clock alignment.\n'
        'All timestamps are subtracted within the remote local clock, without using host timezone.\n'
        'New run date is written AFTER metric calculation: +/-0.5 s is quantization only, not a full error bound.\n'
        'Monitor end is NOT used to align new runs; it can contain a 20+ second idle tail.\n'
        'Original power arrays are copied unchanged from the paper metadata, without idle addback/floor.\n'
        'See provenance.json for source hashes, exact anchors, targets, coverage, and limitations.\n'
        'This is an independently measured per-shard comparison, not a uniform-policy simultaneous cluster run.\n')
    print(json.dumps({'output_dir': str(args.output_dir), 'beam_runs': provenance}, indent=2))


if __name__ == '__main__':
    main()
