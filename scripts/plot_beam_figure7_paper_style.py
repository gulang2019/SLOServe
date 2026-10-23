#!/usr/bin/env python3
"""Merge the previous BEAM previews using Figure 7's original paper renderer.

Energy/SLO panels only; no captions or savings annotations. Coder and Mixed
retain unscaled PCIe energy, and Coder retains the requested Pareto selection.
Other BEAM workloads retain the previous scaled results in the six-panel mode.
Writes only the selected output directory, never measurement archives. See the
generated provenance for caveats.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plots import chat_7b_n_device_triptych as paper
from plots import common
from scripts.plot_beam_coder_mixed_unscaled import select_coder_frontier
from scripts.plot_beam_figure7_energy_preview import load_paper_series
from scripts.plot_beam_figure8_mixed import file_hash, source_name
from scripts.recover_main_multi_replica_figure import LEGEND_METHOD_ORDER

SNAPSHOTS = ROOT / 'figure_data/beam_paper'
SOURCE_PREVIEW = SNAPSHOTS / 'figure7/source_preview.json'
UNSCALED_PREVIEW = SNAPSHOTS / 'figure7/unscaled_preview.json'
OUTPUT_DIR = ROOT / 'analysis_outputs/beam_figure7_paper_style'
MAIN_OUTPUT_DIR = ROOT / 'analysis_outputs/beam_figure7_main'
BEAM_KEY = 'beam / round_robin'
BEAM_COLOR = '#008B8B'
BEAM_MARKER = 'D'
# Match the font scaling of the original large-format two-panel Figure 7.
CONTENT_SCALE = 1.3 / 1.75
WORKLOADS = (
    ('chat_7b', 'chat_7B', 'ChatBot (7B)'),
    ('code_7b', 'code_7B', 'Coder (7B)'),
    ('mixed_7b', 'mixed_7B', 'Mixed (7B)'),
    ('chat_32b', '32B_chat', 'ChatBot (32B)'),
    ('multiround_chat', 'e2e_mulitiround_chatting', 'Multi-Round Chatting'),
    ('large_scale_chat', 'chatting_large_scale', 'Large-Scale Chatting'),
)
MAIN_WORKLOADS = ('chat_7b', 'code_7b', 'mixed_7b')


def build_merge(source_preview=SOURCE_PREVIEW, unscaled_preview=UNSCALED_PREVIEW):
    """Return panels, exported_points, omissions, and auditable source records.

No source files or loaded points are mutated. The unscaled preview must refer
to the exact same six-panel snapshot, preventing accidental mixed revisions.
"""
    source_preview, unscaled_preview = Path(source_preview), Path(unscaled_preview)
    source = json.loads(source_preview.read_text())
    unscaled = json.loads(unscaled_preview.read_text())
    if unscaled['source_preview_sha256'] != file_hash(source_preview):
        raise ValueError('Unscaled and six-panel previews refer to different source revisions')
    panels, exported, omissions, original_sources = [], [], [], {}
    seen = set()
    for point in source['beam_points']:
        identity = (point['workload'], point['n_device'])
        if identity in seen:
            raise ValueError(f'Duplicate BEAM point: {identity}')
        seen.add(identity)
        expected = point['n_device'] * sum(
            run['raw_gpu_energy_j'] for run in point['per_shard']) / len(point['per_shard']) / 1000
        if not math.isclose(point['raw_energy_kj'], expected, rel_tol=0, abs_tol=1e-8):
            raise ValueError(f'Inconsistent raw energy for {identity}')
    for workload, stem, title in WORKLOADS:
        original = source['original_sources'][workload]
        # Keep source identities/hashes in the frozen provenance, but replay from
        # tracked copies rather than an untracked directory in the old Paper repo.
        csv_path = SNAPSHOTS / 'original_e2e' / Path(original['source']).name
        if file_hash(csv_path) != original['sha256']:
            raise ValueError(f'Original paper data changed since the preview: {csv_path}')
        series, order, info = load_paper_series(csv_path)
        original_sources[workload] = info
        for method in order:
            for point in series[method]:
                exported.append(dict(
                    workload=workload, method=common.get_method_label(method),
                    method_key=method, n_device=point['n_device'],
                    violation_pct=point['slo_violation_rate'],
                    violation_pct_lower=point['slo_violation_rate'],
                    energy_kj=point['energy_consumption'], energy_variant='original_paper',
                    included_in_curve=True, omission_reason='',
                    within_paper_x_range=point['slo_violation_rate'] <= 10,
                    sources=info['source']))
        points = sorted((p for p in source['beam_points'] if p['workload'] == workload),
                        key=lambda p: p['n_device'])
        displayed, omitted = select_coder_frontier(points) if workload == 'code_7b' else (points, [])
        omissions.extend(omitted)
        omitted_by_n = {p['n_device']: p for p in omitted}
        energy_key = 'raw_energy_kj' if workload in ('code_7b', 'mixed_7b') else 'energy_kj'
        energy_variant = 'unscaled_pcie' if energy_key == 'raw_energy_kj' else 'affine_scaled'
        if displayed:
            series[BEAM_KEY] = [dict(method=BEAM_KEY, n_device=p['n_device'],
                                    slo_violation_rate=p['violation_pct_upper'],
                                    energy_consumption=p[energy_key]) for p in displayed]
            order.append(BEAM_KEY)
        for point in points:
            omitted_point = omitted_by_n.get(point['n_device'])
            exported.append(dict(
                workload=workload, method='BEAM', method_key=BEAM_KEY,
                n_device=point['n_device'], violation_pct=point['violation_pct_upper'],
                violation_pct_lower=point['violation_pct_lower'],
                energy_kj=point[energy_key], energy_variant=energy_variant,
                included_in_curve=omitted_point is None,
                omission_reason=('Pareto-dominated by N=' + ','.join(map(
                    str, omitted_point['dominated_by_n_devices'])) if omitted_point else ''),
                within_paper_x_range=point['violation_pct_upper'] <= 10,
                sources=';'.join(point['sources'])))
        panels.append(dict(workload=workload, stem=stem, title=title,
                           series=series, order=order, energy_variant=energy_variant))
    if omissions != unscaled['omitted_coder_points']:
        raise ValueError('Coder selection differs from the previous unscaled preview')
    return dict(panels=panels, exported_points=exported, omitted_coder_points=omissions,
                original_sources=original_sources, source=source,
                source_preview=source_name(source_preview),
                source_preview_sha256=file_hash(source_preview),
                unscaled_preview=source_name(unscaled_preview),
                unscaled_preview_sha256=file_hash(unscaled_preview))


def select_main_panels(merged):
    """Select Chat/Code/Mixed without altering any measurements or the merge."""
    return dict(
        merged,
        panels=[panel for panel in merged['panels'] if panel['workload'] in MAIN_WORKLOADS],
        exported_points=[row for row in merged['exported_points']
                         if row['workload'] in MAIN_WORKLOADS],
        original_sources={key: value for key, value in merged['original_sources'].items()
                          if key in MAIN_WORKLOADS},
    )


@contextmanager
def beam_style():
    """Register a plain BEAM label only during rendering, not in shared files."""
    maps = ((common.LABEL_MAP, 'BEAM'), (common.COLOR_MAP, BEAM_COLOR),
            (common.MARKER_MAP, BEAM_MARKER))
    previous = [(mapping, BEAM_KEY in mapping, mapping.get(BEAM_KEY)) for mapping, _ in maps]
    try:
        for mapping, value in maps:
            mapping[BEAM_KEY] = value
        yield
    finally:
        for mapping, existed, value in previous:
            if existed:
                mapping[BEAM_KEY] = value
            else:
                mapping.pop(BEAM_KEY, None)


@contextmanager
def reference_lines_only():
    """Suppress all savings artifacts, including the renderer's extrapolations.

The original label/connector flags still leave estimated endpoint dots and
extrapolation segments. Replace only that annotation hook during this preview;
the original curve rendering, axes, and style are unchanged.
    """
    original = paper._add_threshold_gap_annotations

    def references(ax, *, max_slo_violation, show_target_reference_lines=True, **kwargs):
        if show_target_reference_lines:
            for target in paper.TARGET_SLO_VIOLATIONS:
                if target <= max_slo_violation:
                    ax.axvline(target, color=paper.ANNOTATION_COLOR,
                               linestyle=':', linewidth=1.0, zorder=0)
        return []

    try:
        paper._add_threshold_gap_annotations = references
        yield
    finally:
        paper._add_threshold_gap_annotations = original


def render_panels(merged, output_dir):
    plt, line_collection, line2d, max_locator = paper._get_plotting_dependencies()
    options = dict(panel_keys=('energy',), large_format=True, show_legend=False,
                   show_panel_titles=False, show_energy_gradient_note=False,
                   annotation_placement='below_ours', show_threshold_gap_connectors=False,
                   show_threshold_gap_labels=False, show_slo_reference_lines=True,
                   slo_violation_x_log_scale=True, content_scale=CONTENT_SCALE)
    with beam_style(), reference_lines_only(), plt.rc_context():
        paper._apply_paper_style(plt, panel_keys=('energy',), large_format=True,
                                content_scale=CONTENT_SCALE)
        for panel in merged['panels']:
            paper._render_triptych_figure(
                plt=plt, line_collection_cls=line_collection, line2d_cls=line2d,
                max_n_locator_cls=max_locator, output_stem=output_dir / panel['stem'],
                series_by_method=panel['series'], method_order=panel['order'],
                max_slo_violation=10, **options)
        present = {method for panel in merged['panels'] for method in panel['order']}
        legend_order = [method for method in (*LEGEND_METHOD_ORDER, BEAM_KEY) if method in present]
        paper.save_shared_legend(output_dir / 'shared_legend',
                                 legend_order, large_format=True,
                                 ncol=len(legend_order), include_modeled_optimal=False)
    return options


def write_composition_tex(panels, output_dir):
    """Compose three main panels in one row, or six preview panels in 3x2."""
    ncol = 3 if len(panels) == 3 else 2
    fraction = '0.32' if ncol == 3 else '0.49'
    cells = [rf'\parbox[c]{{{fraction}\linewidth}}{{\centering\small\textbf{{{p["title"]}}}\\'
             + '\n' + rf'\includegraphics[width=\linewidth]{{{p["stem"]}.pdf}}}}'
             for p in panels]
    rows = [' &\n'.join(cells[i:i + ncol]) for i in range(0, len(cells), ncol)]
    body = r' \\[8pt]' + '\n'
    tex = r'''\documentclass{article}
\usepackage{graphicx}
\usepackage{mathptmx}
\pagestyle{empty}
\newsavebox{\figurebox}
\begin{document}
\sbox{\figurebox}{%
\begin{minipage}{7in}
\centering
\includegraphics[width=0.96\linewidth]{shared_legend.pdf}\\[5pt]
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.0}
''' + r'\begin{tabular}{@{}' + 'c' * ncol + r'@{}}' + '\n' + body.join(rows) + r'''
\end{tabular}
\end{minipage}%
}
\pdfpagewidth=\dimexpr\wd\figurebox+16pt\relax
\pdfpageheight=\dimexpr\ht\figurebox+\dp\figurebox+16pt\relax
\hoffset=\dimexpr-1in+8pt\relax
\voffset=\dimexpr-1in+8pt\relax
\shipout\vbox{\hbox{\usebox{\figurebox}}}
\end{document}
'''
    path = output_dir / 'figure7_energy_only.tex'
    path.write_text(tex)
    return path


def compose(panels, output_dir):
    for command in ('pdflatex', 'gs'):
        if shutil.which(command) is None:
            raise RuntimeError(f'{command} is required to compose/rasterize the panel PDFs')
    tex_path = write_composition_tex(panels, output_dir)
    commands = [
        ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', tex_path.name],
        ['gs', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-sDEVICE=png16m', '-r240',
         '-dTextAlphaBits=4', '-dGraphicsAlphaBits=4',
         '-sOutputFile=figure7_energy_only.png', 'figure7_energy_only.pdf'],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=output_dir, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f'{command[0]} failed:\n{result.stdout}\n{result.stderr}')
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-preview', type=Path, default=SOURCE_PREVIEW)
    parser.add_argument('--unscaled-preview', type=Path, default=UNSCALED_PREVIEW)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--main-only', action='store_true',
                        help='Render the main-paper Chat/Code/Mixed figure; no other BEAM workloads.')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = MAIN_OUTPUT_DIR if args.main_only else OUTPUT_DIR
    merged = build_merge(args.source_preview, args.unscaled_preview)
    if args.main_only:
        merged = select_main_panels(merged)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    options = render_panels(merged, args.output_dir)
    commands = compose(merged['panels'], args.output_dir)
    with (args.output_dir / 'plotted_points.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged['exported_points'][0]))
        writer.writeheader()
        writer.writerows(merged['exported_points'])
    report = {key: value for key, value in merged.items() if key not in ('panels', 'source')}
    report.update(
        figure=7, preview_only=not args.main_only,
        paper_assets_modified=args.output_dir.resolve().is_relative_to(ROOT / 'SLO-Packer-MLSys-27'),
        panels=['slo_violation_rate_vs_energy'],
        workloads=[panel['workload'] for panel in merged['panels']],
        layout='1 row x 3 columns' if args.main_only else '3 rows x 2 columns, paper order',
        caption=None, legend_label='BEAM', renderer_options=options,
        original_driver='scripts/recover_main_multi_replica_figure.py',
        original_renderer='plots/chat_7b_n_device_triptych.py:_render_triptych_figure',
        implementation_hashes={str(p): file_hash(ROOT / p) for p in (
            Path('scripts/recover_main_multi_replica_figure.py'),
            Path('plots/chat_7b_n_device_triptych.py'), Path('plots/common.py'),
            Path('scripts/plot_beam_figure7_paper_style.py'))},
        beam_points=[point for point in merged['source']['beam_points']
                     if point['workload'] in {panel['workload'] for panel in merged['panels']}],
        x_axis=merged['source']['x_axis'], scoring=merged['source']['scoring'],
        beam_multiround_evaluation_tpot_ms=merged['source']['beam_multiround_evaluation_tpot_ms'],
        beam_total_allowances_ms=merged['source']['beam_total_allowances_ms'],
        energy_by_workload={p['workload']: p['energy_variant'] for p in merged['panels']
                            if BEAM_KEY in p['series']},
        energy_formula={
            'unscaled_pcie': 'N * mean(raw_gpu_energy_j) / 1000',
            'affine_scaled': 'N * mean((330/248)*raw_gpu_energy_j + (70-52*330/248)*monitor_duration_s) / 1000'},
        uncertainty='Conservative upper violation bound plotted; lower bounds retained in CSV/provenance.',
        annotation_override='Temporarily replace savings annotation hook with reference lines only; no estimated endpoints or extrapolations.',
        omissions='Coder N=7,8 omitted from curve only, using previous Pareto selection. All records retained.',
        limitations=merged['source']['limitations'] + [
            'Coder/Mixed BEAM energy is unscaled PCIe, unlike other scaled BEAM panels and SXM baselines.',
            'Only Coder BEAM is Pareto-filtered; omitted points are not invalidated measurements.',
            'No captions or automatic savings annotations are displayed; assumptions are retained here.',
        ],
        composition_commands=commands,
    )
    # Replace the old preview's energy statement: the merged panels intentionally
    # have different energy conventions, as requested.
    report['energy_convention'] = 'Original paper energy retained; raw BEAM for Coder/Mixed, previously scaled BEAM elsewhere. No idle addback.'
    if args.main_only:
        report.pop('beam_multiround_evaluation_tpot_ms')
        report['beam_total_allowances_ms'] = {
            key: value for key, value in report['beam_total_allowances_ms'].items()
            if key in MAIN_WORKLOADS}
        report['limitations'] = [
            'BEAM extrapolates independent GPU shards, not a measured simultaneous full cluster.',
            'Chat uses heuristic affine power scaling; Coder/Mixed use raw PCIe energy against SXM baselines.',
            'BEAM evaluation TPOT is 25 ms; original Mixed baseline scores retain 20 ms. Not an equal-SLO comparison.',
            'Post-hoc scoring and allowances do not change recorded BEAM control targets or measured execution.',
            'Full monitoring intervals differ across runs; no idle addback or duration normalization.',
            'Only Coder BEAM is Pareto-filtered; omitted N=7,8 measurements are retained, not invalidated.',
            'Original saved scores and historical calibration are not independently revalidated by this replay.',
            'Conservative upper violation bounds are plotted when individual token timestamps are ambiguous.',
        ]
        report['energy_convention'] = 'Original paper energy retained; raw BEAM for Coder/Mixed and affine-scaled BEAM for Chat. No idle addback.'
    (args.output_dir / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(args.output_dir / 'figure7_energy_only.png')


if __name__ == '__main__':
    main()
