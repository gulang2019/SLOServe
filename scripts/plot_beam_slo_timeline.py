#!/usr/bin/env python3
"""Mixed SLO violation timelines: original cluster counts and BEAM shard estimates.

Uses arrival cohorts, cumulative token deadlines, and explicit bounds for streamed
chunks containing multiple tokens. Original results are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def evaluate_request(row, saved_allowance_s=.02, total_allowance_s=.05, tpot_s=.02):
    """Bound violations by placing unlocated tokens at earliest/latest chunks.

    Each timestamped output event is assumed to contain at least one token.
    TTFT and stream times are client observations; no schedule-gap clipping.
    """
    if (not all(np.isfinite(x) and x >= 0 for x in
                (saved_allowance_s, total_allowance_s, tpot_s)) or tpot_s == 0):
        raise ValueError('Invalid allowance or TPOT')
    budget = float(row['ttft_budget_s']) - saved_allowance_s + total_allowance_s
    if not np.isfinite(budget) or budget < 0:
        raise ValueError('Invalid TTFT budget')
    result = dict(budget_s=budget, ttft_violation=False, error=not row['success'],
                  violation_lower=True, violation_upper=True, missing_token_timestamps=0)
    if not row['success']:
        return result
    count = row['actual_output_tokens']
    if type(count) is not int or count < 1:
        raise ValueError('Output token count must be a positive integer')
    ttft = float(row['ttft_s'])
    gaps = np.asarray(row['itls_s'], dtype=float)
    if (not np.isfinite(ttft) or ttft < 0 or gaps.ndim != 1
            or not np.all(np.isfinite(gaps)) or np.any(gaps < 0)):
        raise ValueError('Expected finite nonnegative TTFT and one-dimensional gaps')
    missing = count - len(gaps) - 1
    if missing < 0:
        raise ValueError('More timestamped events than output tokens')
    result['missing_token_timestamps'] = missing
    result['ttft_violation'] = ttft > budget
    if result['ttft_violation']:
        return result
    observed = np.r_[ttft, ttft + np.cumsum(gaps)]
    earliest = np.r_[np.repeat(observed[0], missing), observed]
    latest = np.r_[observed, np.repeat(observed[-1], missing)]
    deadlines = budget + np.arange(1, count) * tpot_s
    def fails(times):
        return bool(count > 1 and np.percentile(times[1:] - deadlines, 90) > 0)
    result['violation_lower'] = fails(earliest)
    result['violation_upper'] = fails(latest)
    return result


def rebin_counts(time_s, counts, source_window_s=5, target_window_s=30, duration_s=600):
    """Sum native-bin counts before division; never average window percentages."""
    windows = (source_window_s, target_window_s, duration_s)
    if not all(np.isfinite(x) and x > 0 for x in windows):
        raise ValueError('Bin widths and duration must be finite and positive')
    factor = target_window_s / source_window_s
    if not np.isclose(factor, round(factor)) or factor < 1:
        raise ValueError('Target window must be an integer multiple of source window')
    if not np.isclose(duration_s / target_window_s, round(duration_s / target_window_s)):
        raise ValueError('Duration must be a multiple of target window')
    times = np.asarray(time_s, dtype=float)
    values = np.asarray(counts, dtype=float)
    expected = np.arange(round(duration_s / source_window_s)) * source_window_s
    if (times.ndim != 1 or values.ndim != 1 or len(times) != len(expected)
            or len(values) != len(expected) or not np.allclose(times, expected)
            or not np.all(np.isfinite(values)) or np.any(values < 0)):
        raise ValueError('Expected complete aligned nonnegative source bins')
    edges = np.arange(round(duration_s / target_window_s) + 1) * target_window_s
    return edges, values.reshape(-1, round(factor)).sum(axis=1)


def rates(numerator, denominator):
    result = np.full(len(denominator), np.nan)
    np.divide(100 * numerator, denominator, out=result, where=denominator > 0)
    return result


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def beam_series(path, device, allowance, tpot):
    summary_path = path / 'per_request_slo.summary.json'
    request_path = path / 'per_request_slo.jsonl'
    config_path = path / 'slo_config.txt'
    summary = json.loads(summary_path.read_text())
    rows = [json.loads(line) for line in request_path.read_text().splitlines() if line.strip()]
    config = dict(line.split('=', 1) for line in config_path.read_text().splitlines() if '=' in line)
    first_stamp = datetime.strptime(rows[0]['trace_timestamp'], '%Y-%m-%d %H:%M:%S.%f%z')
    offset = (first_stamp - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()
    scored = []
    for row in rows:
        result = evaluate_request(row, summary['routing_overhead_s'], allowance, tpot)
        result.update(request_index=row['request_index'],
                      arrival_time_s=float(row['client_entry_time_s']) + offset)
        scored.append(result)
    arrival = np.asarray([r['arrival_time_s'] for r in scored])
    if np.any(~np.isfinite(arrival)) or np.any(arrival < 0) or np.any(arrival >= 600):
        raise ValueError(f'{path}: arrivals extend outside [0, 600)')
    edges = np.arange(0, 605, 5)
    counts = np.histogram(arrival, bins=edges)[0]
    lower = np.histogram(arrival, bins=edges, weights=[int(r['violation_lower']) for r in scored])[0]
    upper = np.histogram(arrival, bins=edges, weights=[int(r['violation_upper']) for r in scored])[0]
    ttft = np.histogram(arrival, bins=edges, weights=[int(r['ttft_violation']) for r in scored])[0]
    info = {
        'source': str(path.relative_to(ROOT)), 'device_index': device,
        'scope': 'single independently replayed RR shard, not cluster aggregate',
        'requests': len(rows), 'violation_count_lower': int(lower.sum()),
        'violation_count_upper': int(upper.sum()), 'ttft_violations': int(ttft.sum()),
        'violation_pct_lower': 100 * float(lower.sum()) / len(rows),
        'violation_pct_upper': 100 * float(upper.sum()) / len(rows),
        'saved_allowance_s': summary['routing_overhead_s'],
        'evaluated_total_allowance_s': allowance, 'evaluated_tpot_s': tpot,
        'control_tbt_target_s': float(config['beam_tbt_s']),
        'restored_global_trace_offset_s': offset,
        'requests_with_missing_token_timestamps': sum(r['missing_token_timestamps'] > 0 for r in scored),
        'ambiguous_request_indices': [r['request_index'] for r in scored if r['violation_lower'] != r['violation_upper']],
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in (request_path, summary_path, config_path)},
    }
    return dict(key=f'beam_shard_{device}', label=f'BEAM shard {device}',
                time_s=edges[:-1], arrivals=counts, lower=lower, upper=upper,
                ttft=ttft, info=info), scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'analysis_outputs/beam_figure8_mixed_slo_timeline')
    parser.add_argument('--tpot-ms', type=float, default=20)
    parser.add_argument('--allowance-ms', type=float, default=50)
    args = parser.parse_args()
    metadata_path = ROOT / 'Paper/data/timeline/energy_consumption-mixed.meta.json'
    metadata = json.loads(metadata_path.read_text())
    series = []
    for trace in metadata['traces']:
        arrivals = trace['arrivals']
        violations = trace['slo_violations']
        if arrivals['window_s'] != 5 or violations['window_s'] != 5 or arrivals['time_s'] != violations['time_s']:
            raise ValueError('Original metadata must have aligned five-second bins')
        counts = np.asarray(arrivals['values'], dtype=float)
        failures = np.asarray(violations['values'], dtype=float)
        label = 'vLLM' if trace['label'] == 'Baseline' else 'SLO-Packer'
        series.append(dict(key=label.lower(), label=label + ' (8-GPU aggregate)',
                           time_s=np.asarray(arrivals['time_s']), arrivals=counts,
                           lower=failures, upper=failures, ttft=None,
                           info={'requests': int(counts.sum()), 'violation_count': int(failures.sum()),
                                 'violation_pct': 100 * float(failures.sum() / counts.sum()),
                                 'scope': 'original eight-GPU aggregate; saved scoring unchanged'}))
    per_request = []
    for device, path in [
        (0, ROOT / 'beam-figure7-7b-20261020-215014/mixed_7b/split_8_0'),
        (3, ROOT / 'beam-jobs-mlsys/mixed/split_8_3'),
        (7, ROOT / 'beam-jobs-mlsys/mixed/split_8_7'),
    ]:
        s, scored = beam_series(path, device, args.allowance_ms / 1000, args.tpot_ms / 1000)
        series.append(s)
        per_request.extend(dict(device_index=device, **r) for r in scored)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    colors = ['#4C78A8', '#2E7D32', '#8E44AD', '#D97706', '#C44E52']
    linestyles = ['-', '-', '--', '-.', ':']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    for window in (30, 5):
        fig, ax = plt.subplots(figsize=(12, 6.3))
        for s, color, linestyle in zip(series, colors, linestyles):
            edges, counts = rebin_counts(s['time_s'], s['arrivals'], target_window_s=window)
            _, lower = rebin_counts(s['time_s'], s['lower'], target_window_s=window)
            _, upper = rebin_counts(s['time_s'], s['upper'], target_window_s=window)
            lower_rate, upper_rate = rates(lower, counts), rates(upper, counts)
            label = s['label']
            if s['key'].startswith('beam'):
                label += ' (client estimate)'
            ax.stairs(upper_rate, edges, color=color, linestyle=linestyle, lw=1.8,
                      baseline=None, label=label)
            if np.any(lower != upper):
                ax.fill_between(edges, np.r_[lower_rate, lower_rate[-1]],
                                np.r_[upper_rate, upper_rate[-1]], step='post',
                                color=color, alpha=.15)
            for i, time in enumerate(edges[:-1]):
                csv_rows.append(dict(series=s['key'], window_s=window, time_s=float(time),
                                     requests=int(counts[i]), violations_lower=int(lower[i]),
                                     violations_upper=int(upper[i]), violation_pct_lower=float(lower_rate[i]),
                                     violation_pct_upper=float(upper_rate[i])))
        ax.set_xlim(0, 600)
        ax.set_ylim(0, 105)
        ax.set_xticks(np.arange(0, 601, 60))
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_xlabel('Request arrival time in workload (s)')
        ax.set_ylabel(f'SLO violations (% of arrivals per {window}-s window)')
        ax.grid(alpha=.2)
        ax.set_title(f'Mixed workload — SLO violation timeline ({window}-second windows)\n'
                     f'BEAM rescore: cumulative deadlines, {args.tpot_ms:g} ms TPOT + '
                     f'{args.allowance_ms:g} ms total TTFT allowance', fontsize=12, pad=13)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .10),
                   ncol=3, frameon=False, fontsize=9)
        fig.text(.5, .045, 'Original systems: 8-GPU aggregates with server schedule-gap correction. '
                 'BEAM: individual shards, raw client timestamps.\n'
                 'BEAM control targets differ: shard 0 = 20 ms; shards 3/7 = 25 ms. '
                 'Shading bounds token/chunk ambiguity; no full-cluster BEAM claim.',
                 ha='center', va='center', fontsize=8.3, color='#444444')
        fig.tight_layout(rect=(0, .20, 1, 1))
        for extension in ('png', 'pdf'):
            fig.savefig(args.output_dir / f'mixed_slo_violation_timeline_{window}s.{extension}', dpi=220)
        plt.close(fig)
    with (args.output_dir / 'timeline_bins.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    with (args.output_dir / 'beam_rescored_requests.jsonl').open('w') as handle:
        for row in per_request:
            handle.write(json.dumps(row) + '\n')
    report = {
        'original_metadata': str(metadata_path.relative_to(ROOT)),
        'original_metadata_sha256': digest(metadata_path),
        'original_scoring': {'tpot_s': .02, 'ttft_allowance_s': .05, 'schedule_gap_clip_s': .10},
        'beam_evaluation': {'tpot_s': args.tpot_ms / 1000, 'total_ttft_allowance_s': args.allowance_ms / 1000,
                            'schedule_gap_clip_s': None, 'decode_rule': 'p90 cumulative token deadline lateness <= 0'},
        'series': {s['key']: s['info'] for s in series},
        'limitations': [
            'Cluster-level original curves and independently replayed single-shard BEAM curves have different scope.',
            'Original violations are binned by router arrival, denominators by global arrival; window ratios approximate cohort matching.',
            'Original binned outcomes are unchanged; exact matching client/server scoring is impossible from this bundle.',
            'Original zero-load TTFT calibration cannot be reconstructed from aggregate metadata; BEAM uses archived per-request budgets.',
            'BEAM interval assignment uses actual client entry plus original shard first-arrival offset; all selected arrivals are within [0,600).',
            'All completions are scored, including completions after 600s, and attributed to their arrival windows.',
            'Thirty-second overview sums numerator and denominator counts before division; native five-second plot is also provided.',
            'Missing-token bounds assume every recorded stream event includes at least one token; no timestamps are fabricated.',
            'Empty bins are NaN, not zero; main BEAM line uses upper violation bound and shaded range uses lower bound.',
            'No new server execution, SLO API updates, or changes to source results were performed.',
        ],
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output_dir / 'METHODOLOGY.txt').write_text(
        'Mixed SLO violation timelines; original cluster aggregates plus independent BEAM shards 0, 3, 7.\n'
        'Primary display: 30-second count-rebinned rates. Companion: native 5-second rates.\n'
        'BEAM TTFT budget = saved budget - saved allowance + selected total allowance.\n'
        'Pass if successful, TTFT <= budget, and p90(token_time[k] - budget - k*TPOT) <= 0 for k>=1.\n'
        'The archived p90-ITL violation field is ignored. No client-gap clipping is applied.\n'
        'Unknown token locations within chunks are bounded by earliest/latest feasible assignments, not invented.\n'
        'Outcomes are binned by client arrival plus restored shard offset, never by completion time.\n'
        'Original numerator uses router arrival and denominator uses global arrival; approximate window cohort matching.\n'
        'Original metadata scoring has TPOT20ms, TTFT allowance50ms, and 100ms schedule-gap correction.\n'
        'These estimates are not an exact apples-to-apples client/server or full-cluster comparison.\n'
        'See summary.json for parameters, counts, source hashes, and all limitations.\n')
    print(json.dumps({'output_dir': str(args.output_dir), 'series': report['series']}, indent=2))


if __name__ == '__main__':
    main()
