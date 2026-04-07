from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_left
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

from plots.common import get_method_label, get_method_style, get_paper_figure_dir

TARGET_SLO_VIOLATIONS = (1.0, 5.0)
ANNOTATION_COLOR = '#808080'
OURS_METHOD = 'atfc / slosserve_planner'
BASELINE_METHOD = 'qlm / round_robin'
ANNOTATED_YLABELS = {
    'energy_consumption': 'saved',
    'average_n_active_servers': 'fewer',
    'n_device': 'fewer',
}
METADATA_SUFFIX = '.meta.json'
FIGURE_METADATA_VERSION = 1


def _default_result_files():
    return [
        {
            'name': 'chat_23_7B',
            'file': 'traces/report/chat23.jsonl',
            'xlim': 20,
        },
        {
            'name': 'code_7B',
            'file': 'traces/report/code.jsonl',
            'xlim': 20,
        },
        {
            'name': 'mixed_7B',
            'file': 'traces/report/mixed.jsonl',
            'xlim': 20,
        },
    ]


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return None
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if hasattr(value, 'item'):
        value = value.item()
    if pd is not None and pd.isna(value):
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _get_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'matplotlib is required to render or replay e2e figures'
        ) from exc
    return plt


def _require_pandas():
    if pd is None:
        raise RuntimeError('pandas is required to build e2e figure metadata from raw trace files')


def _df_points(curve: pd.DataFrame):
    x_col, y_col = curve.columns[:2]
    clean = curve[[x_col, y_col]].dropna()
    return {
        'x': [_json_ready(v) for v in clean[x_col].tolist()],
        'y': [_json_ready(v) for v in clean[y_col].tolist()],
    }


def _estimate_value_at_target(curve: pd.DataFrame, target: float):
    points_by_x = {}
    x_col, y_col = curve.columns[:2]
    for x, y in curve[[x_col, y_col]].itertuples(index=False, name=None):
        if pd.isna(x) or pd.isna(y):
            continue
        points_by_x[float(x)] = float(y)

    points = sorted(points_by_x.items())
    if not points:
        return None, None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    idx = bisect_left(xs, target)

    if idx < len(xs) and math.isclose(xs[idx], target):
        return ys[idx], None
    if len(points) == 1:
        return None, None

    if idx == 0:
        x0, y0 = points[0]
        x1, y1 = points[1]
        extension_start = (x0, y0)
    elif idx == len(points):
        x0, y0 = points[-2]
        x1, y1 = points[-1]
        extension_start = (x1, y1)
    else:
        x0, y0 = points[idx - 1]
        x1, y1 = points[idx]
        extension_start = None

    if math.isclose(x0, x1):
        return None, None

    slope = (y1 - y0) / (x1 - x0)
    target_y = y0 + slope * (target - x0)
    extension = None
    if extension_start is not None:
        extension = (extension_start, (target, target_y))
    return target_y, extension


def _build_threshold_annotation_ops(method_curves: dict[str, pd.DataFrame], ylabel: str):
    ours_curve = method_curves.get(OURS_METHOD)
    if ours_curve is None:
        ours_curve = next(
            (curve for method, curve in method_curves.items() if 'slosserve_planner' in method),
            None,
        )

    baseline_curve = method_curves.get(BASELINE_METHOD)
    if baseline_curve is None:
        baseline_curve = next(
            (
                curve
                for method, curve in method_curves.items()
                if 'slosserve_planner' not in method and method != OURS_METHOD
            ),
            None,
        )

    if ours_curve is None or baseline_curve is None:
        return {
            'line_overlays': [],
            'scatter_overlays': [],
            'annotations': [],
        }

    label_suffix = ANNOTATED_YLABELS.get(ylabel, 'saved')
    ops = {
        'line_overlays': [],
        'scatter_overlays': [],
        'annotations': [],
    }
    for target in TARGET_SLO_VIOLATIONS:
        baseline_y, baseline_extension = _estimate_value_at_target(baseline_curve, target)
        ours_y, ours_extension = _estimate_value_at_target(ours_curve, target)

        for extension in (baseline_extension, ours_extension):
            if extension is None:
                continue
            (x0, y0), (x1, y1) = extension
            ops['line_overlays'].append(
                {
                    'x': [x0, x1],
                    'y': [y0, y1],
                    'style': {
                        'color': ANNOTATION_COLOR,
                        'linestyle': '--',
                        'linewidth': 1.2,
                        'zorder': 3,
                    },
                }
            )

        if baseline_y is None or ours_y is None or baseline_y <= 0:
            continue

        y_low, y_high = sorted((ours_y, baseline_y))
        ops['line_overlays'].append(
            {
                'x': [target, target],
                'y': [y_low, y_high],
                'style': {
                    'color': ANNOTATION_COLOR,
                    'linewidth': 1.2,
                    'zorder': 4,
                },
            }
        )
        ops['scatter_overlays'].append(
            {
                'x': [target, target],
                'y': [ours_y, baseline_y],
                'style': {
                    'color': ANNOTATION_COLOR,
                    's': 18,
                    'zorder': 5,
                },
            }
        )

        savings = (baseline_y - ours_y) / baseline_y * 100
        label = f'{savings:.1f}% {label_suffix}' if savings >= 0 else f'{abs(savings):.1f}% higher'
        ops['annotations'].append(
            {
                'text': label,
                'xy': [target, (ours_y + baseline_y) / 2],
                'xytext': [6, 0],
                'textcoords': 'offset points',
                'color': ANNOTATION_COLOR,
                'fontsize': 9,
                'va': 'center',
            }
        )
    return ops


def _build_metric_figure_specs(
    df: pd.DataFrame,
    name: str,
    output_dir: Path,
    file: str,
    xlim=None,
    is_disagg: bool = False,
    whitelist=None,
):
    specs = []
    features = [f for f in ['load_scale', 'n_device', 'ttft_slo_scale', 'slo_tpot'] if f in df.columns]
    if not features:
        return specs

    for feature in features:
        if df[feature].nunique(dropna=False) <= 1:
            continue

        other_features = [f for f in features if f != feature]
        if other_features:
            grouped = list(df.groupby(other_features, dropna=False, sort=False))
            n_groups = len(grouped)
        else:
            grouped = [((), df)]
            n_groups = 1

        ncols = min(3, n_groups)
        nrows = math.ceil(n_groups / ncols)

        for xlabel, ylabel in [
            ('slo_violation_rate', 'energy_consumption'),
            ('slo_violation_rate', 'average_n_active_servers'),
            ('slo_violation_rate', feature),
        ]:
            if xlabel not in df.columns or ylabel not in df.columns:
                continue

            axis_specs = []
            for idx, (other_feature_values, group) in enumerate(grouped):
                if not isinstance(other_feature_values, tuple):
                    other_feature_values = (other_feature_values,)

                method_curves = {}
                series_specs = []
                for (sched, route), pair_group in group.groupby(
                    ['scheduling_policy', 'routing_policy'],
                    dropna=False,
                    sort=False,
                ):
                    route_str = '' if pd.isna(route) else str(route)
                    if is_disagg ^ ('disagg' in route_str):
                        continue

                    method = f"{sched} / {route}"
                    if whitelist is not None and method not in whitelist:
                        continue

                    group_sorted = pair_group.sort_values(xlabel)
                    clean_curve = group_sorted[[xlabel, ylabel]].dropna()
                    if clean_curve.empty:
                        continue

                    method_curves[method] = clean_curve
                    series_specs.append(
                        {
                            'x': _df_points(clean_curve)['x'],
                            'y': _df_points(clean_curve)['y'],
                            'label': get_method_label(method),
                            'style': _json_ready(get_method_style(method)),
                            'method': method,
                        }
                    )

                other_features_dict = {
                    f: _json_ready(v) for f, v in zip(other_features, other_feature_values)
                }
                axis_spec = {
                    'row': idx // ncols,
                    'col': idx % ncols,
                    'visible': True,
                    'xlabel': xlabel,
                    'ylabel': ylabel,
                    'title': f'{ylabel} vs {xlabel}\n({other_features_dict})',
                    'legend': True,
                    'series': series_specs,
                    'axvlines': [],
                    'line_overlays': [],
                    'scatter_overlays': [],
                    'annotations': [],
                    'xlim': [-0.2, xlim] if xlim is not None else None,
                }
                if xlabel == 'slo_violation_rate':
                    axis_spec['axvlines'] = [
                        {
                            'x': target,
                            'style': {
                                'color': ANNOTATION_COLOR,
                                'linestyle': ':',
                                'linewidth': 1.0,
                                'zorder': 0,
                            },
                        }
                        for target in TARGET_SLO_VIOLATIONS
                    ]
                if xlabel == 'slo_violation_rate' and ylabel in ANNOTATED_YLABELS:
                    annotation_ops = _build_threshold_annotation_ops(method_curves, ylabel)
                    axis_spec['line_overlays'].extend(annotation_ops['line_overlays'])
                    axis_spec['scatter_overlays'].extend(annotation_ops['scatter_overlays'])
                    axis_spec['annotations'].extend(annotation_ops['annotations'])

                axis_specs.append(axis_spec)

            for extra_idx in range(len(axis_specs), nrows * ncols):
                axis_specs.append(
                    {
                        'row': extra_idx // ncols,
                        'col': extra_idx % ncols,
                        'visible': False,
                    }
                )

            output_stem = output_dir / f'{ylabel}_vs_{xlabel}_change_{feature}_{name}'
            specs.append(
                {
                    'schema_version': FIGURE_METADATA_VERSION,
                    'figure_kind': 'metric_grid',
                    'name': name,
                    'source_file': file,
                    'feature': feature,
                    'xlabel': xlabel,
                    'ylabel': ylabel,
                    'layout': {
                        'nrows': nrows,
                        'ncols': ncols,
                    },
                    'figsize': [6 * ncols, 5 * nrows],
                    'tight_layout': True,
                    'output_stem': str(output_stem),
                    'filters': {
                        'is_disagg': is_disagg,
                        'whitelist': None if whitelist is None else list(whitelist),
                        'xlim': xlim,
                    },
                    'axes': axis_specs,
                }
            )
    return specs


def _build_power_figure_spec(
    df: pd.DataFrame,
    name: str,
    output_dir: Path,
    file: str,
    is_disagg: bool = False,
):
    power_columns = {
        'n_device',
        'load_scale',
        'ttft_slo_scale',
        'slo_tpot',
        'per_server_power',
    }
    if not power_columns.issubset(df.columns):
        return None

    power_df = df[(df['n_device'] == 8) & df['per_server_power'].notna()]
    power_groups = list(
        power_df.groupby(
            ['load_scale', 'ttft_slo_scale', 'slo_tpot'],
            dropna=False,
            sort=False,
        )
    )
    if not power_groups:
        return None

    ncols = min(4, len(power_groups))
    nrows = math.ceil(len(power_groups) / ncols)
    axis_specs = []

    for idx, ((load_scale, ttft_slo_scale, slo_tpot), tdf) in enumerate(power_groups):
        series_specs = []
        for (sched, route), pair_group in tdf.groupby(
            ['scheduling_policy', 'routing_policy'],
            dropna=False,
            sort=False,
        ):
            route_str = '' if pd.isna(route) else str(route)
            if is_disagg ^ ('disagg' in route_str):
                continue

            method = f"{sched} / {route}"
            label = get_method_label(method)
            style = get_method_style(method)
            for line_idx, (_, power_row) in enumerate(pair_group.iterrows()):
                per_server_power = power_row['per_server_power']
                if not isinstance(per_server_power, (list, tuple)):
                    continue
                series_specs.append(
                    {
                        'x': list(range(len(per_server_power))),
                        'y': [_json_ready(v) for v in per_server_power],
                        'label': label if line_idx == 0 else None,
                        'style': {
                            'color': style['color'],
                            'marker': style['marker'],
                            'linewidth': style['linewidth'],
                            'markersize': style['markersize'],
                        },
                        'method': method,
                    }
                )

        axis_specs.append(
            {
                'row': idx // ncols,
                'col': idx % ncols,
                'visible': True,
                'xlabel': 'Server',
                'ylabel': 'Power (W)',
                'title': (
                    f'load_scale={_json_ready(load_scale)}, '
                    f'ttft_slo_scale={_json_ready(ttft_slo_scale)}, '
                    f'slo_tpot={_json_ready(slo_tpot)}'
                ),
                'legend': True,
                'series': series_specs,
                'axvlines': [],
                'line_overlays': [],
                'scatter_overlays': [],
                'annotations': [],
                'xlim': None,
            }
        )

    for extra_idx in range(len(axis_specs), nrows * ncols):
        axis_specs.append(
            {
                'row': extra_idx // ncols,
                'col': extra_idx % ncols,
                'visible': False,
            }
        )

    output_stem = output_dir / f'per_server_power_{name}'
    return {
        'schema_version': FIGURE_METADATA_VERSION,
        'figure_kind': 'power_grid',
        'name': name,
        'source_file': file,
        'layout': {
            'nrows': nrows,
            'ncols': ncols,
        },
        'figsize': [6 * ncols, 4.5 * nrows],
        'tight_layout': True,
        'output_stem': str(output_stem),
        'filters': {
            'is_disagg': is_disagg,
            'n_device': 8,
        },
        'axes': axis_specs,
    }


def build_figure_specs(name, file, xlim=None, is_disagg=False, whitelist=None):
    _require_pandas()
    df = pd.read_json(file, lines=True)
    if df.empty:
        return []

    required_columns = {'scheduling_policy', 'routing_policy'}
    if not required_columns.issubset(df.columns):
        return []

    output_dir = get_paper_figure_dir('e2e_figure', 'draw_figures')
    df = df.copy()
    df['slo_violation_rate'] *= 100

    specs = _build_metric_figure_specs(
        df=df,
        name=name,
        output_dir=output_dir,
        file=file,
        xlim=xlim,
        is_disagg=is_disagg,
        whitelist=whitelist,
    )
    power_spec = _build_power_figure_spec(
        df=df,
        name=name,
        output_dir=output_dir,
        file=file,
        is_disagg=is_disagg,
    )
    if power_spec is not None:
        specs.append(power_spec)
    return specs


def _render_axis_spec(ax, axis_spec):
    if not axis_spec.get('visible', True):
        ax.set_visible(False)
        return

    for series in axis_spec.get('series', []):
        ax.plot(
            series['x'],
            series['y'],
            label=series.get('label'),
            **series.get('style', {}),
        )

    for axvline in axis_spec.get('axvlines', []):
        ax.axvline(axvline['x'], **axvline.get('style', {}))

    for overlay in axis_spec.get('line_overlays', []):
        ax.plot(
            overlay['x'],
            overlay['y'],
            **overlay.get('style', {}),
        )

    for scatter in axis_spec.get('scatter_overlays', []):
        ax.scatter(
            scatter['x'],
            scatter['y'],
            **scatter.get('style', {}),
        )

    for annotation in axis_spec.get('annotations', []):
        ax.annotate(
            annotation['text'],
            xy=tuple(annotation['xy']),
            xytext=tuple(annotation['xytext']),
            textcoords=annotation.get('textcoords', 'offset points'),
            color=annotation.get('color'),
            fontsize=annotation.get('fontsize'),
            va=annotation.get('va'),
            ha=annotation.get('ha'),
        )

    ax.set_xlabel(axis_spec.get('xlabel', ''))
    ax.set_ylabel(axis_spec.get('ylabel', ''))
    ax.set_title(axis_spec.get('title', ''))

    xlim = axis_spec.get('xlim')
    if xlim is not None:
        ax.set_xlim(*xlim)

    ylim = axis_spec.get('ylim')
    if ylim is not None:
        ax.set_ylim(*ylim)

    if axis_spec.get('legend', False):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()


def render_figure_spec(spec: dict, save_outputs: bool = True):
    plt = _get_pyplot()
    nrows = spec['layout']['nrows']
    ncols = spec['layout']['ncols']
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=tuple(spec['figsize']),
        squeeze=False,
    )

    for axis_spec in spec['axes']:
        ax = axes[axis_spec['row']][axis_spec['col']]
        _render_axis_spec(ax, axis_spec)

    if spec.get('tight_layout', False):
        fig.tight_layout()

    if save_outputs:
        output_stem = Path(spec['output_stem'])
        fig.savefig(f'{output_stem}.png', dpi=300)
        fig.savefig(f'{output_stem}.pdf', dpi=300)
        print(f"Saved plot to {output_stem}.png")
        print(f"Saved plot to {output_stem}.pdf")

    plt.close(fig)


def _metadata_path(output_stem: str | Path) -> Path:
    return Path(f'{output_stem}{METADATA_SUFFIX}')


def write_figure_metadata(spec: dict):
    metadata_path = _metadata_path(spec['output_stem'])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open('w', encoding='utf-8') as f:
        json.dump(_json_ready(spec), f, indent=2, sort_keys=True)
    print(f"Saved metadata to {metadata_path}")


def load_figure_metadata(path: str | Path):
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def _expand_metadata_paths(paths):
    expanded_paths = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_dir():
            expanded_paths.extend(sorted(path.glob(f'*{METADATA_SUFFIX}')))
        else:
            expanded_paths.append(path)
    return expanded_paths


def replay_figures_from_metadata(paths):
    metadata_paths = _expand_metadata_paths(paths)
    for metadata_path in metadata_paths:
        spec = load_figure_metadata(metadata_path)
        render_figure_spec(spec, save_outputs=True)


def draw_figures(
    name,
    file,
    xlim=None,
    is_disagg=False,
    whitelist=None,
    export_metadata: bool = True,
    render: bool = True,
):
    specs = build_figure_specs(
        name=name,
        file=file,
        xlim=xlim,
        is_disagg=is_disagg,
        whitelist=whitelist,
    )
    for spec in specs:
        if export_metadata:
            write_figure_metadata(spec)
        if render:
            render_figure_spec(spec, save_outputs=True)


def main(
    result_files=None,
    replay_metadata=None,
    export_metadata: bool = True,
    metadata_only: bool = False,
):
    if replay_metadata:
        replay_figures_from_metadata(replay_metadata)
        return

    if result_files is None:
        result_files = _default_result_files()

    for kwargs in result_files:
        draw_figures(
            **kwargs,
            export_metadata=export_metadata,
            render=not metadata_only,
        )


def _parse_args():
    parser = argparse.ArgumentParser(description='Draw end-to-end paper figures.')
    parser.add_argument(
        '--replay-metadata',
        nargs='+',
        help='Replay one or more metadata JSON files or directories containing *.meta.json files.',
    )
    parser.add_argument(
        '--metadata-only',
        action='store_true',
        help='Export metadata without rendering figures from raw trace files.',
    )
    parser.add_argument(
        '--no-export-metadata',
        action='store_true',
        help='Do not emit figure metadata JSON files when generating from raw traces.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(
        replay_metadata=args.replay_metadata,
        export_metadata=not args.no_export_metadata,
        metadata_only=args.metadata_only,
    )
