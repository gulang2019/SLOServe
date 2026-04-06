import math
import os
import sys
from bisect import bisect_left

import matplotlib.pyplot as plt
import pandas as pd
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


def _annotate_threshold_comparison(ax, method_curves: dict[str, pd.DataFrame], ylabel: str):
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
        return

    label_suffix = ANNOTATED_YLABELS.get(ylabel, 'saved')
    for target in TARGET_SLO_VIOLATIONS:
        baseline_y, baseline_extension = _estimate_value_at_target(baseline_curve, target)
        ours_y, ours_extension = _estimate_value_at_target(ours_curve, target)

        for extension in (baseline_extension, ours_extension):
            if extension is None:
                continue
            (x0, y0), (x1, y1) = extension
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=ANNOTATION_COLOR,
                linestyle='--',
                linewidth=1.2,
                zorder=3,
            )

        if baseline_y is None or ours_y is None or baseline_y <= 0:
            continue

        y_low, y_high = sorted((ours_y, baseline_y))
        ax.plot(
            [target, target],
            [y_low, y_high],
            color=ANNOTATION_COLOR,
            linewidth=1.2,
            zorder=4,
        )
        ax.scatter(
            [target, target],
            [ours_y, baseline_y],
            color=ANNOTATION_COLOR,
            s=18,
            zorder=5,
        )

        savings = (baseline_y - ours_y) / baseline_y * 100
        label = f'{savings:.1f}% {label_suffix}' if savings >= 0 else f'{abs(savings):.1f}% higher'
        ax.annotate(
            label,
            xy=(target, (ours_y + baseline_y) / 2),
            xytext=(6, 0),
            textcoords='offset points',
            color=ANNOTATION_COLOR,
            fontsize=9,
            va='center',
        )


def draw_figures(name, file, xlim = None, is_disagg = False):
    df = pd.read_json(file, lines = True)
    if df.empty:
        return

    required_columns = {'scheduling_policy', 'routing_policy'}
    if not required_columns.issubset(df.columns):
        return

    output_dir = get_paper_figure_dir('e2e_figure', 'draw_figures')
    features = [f for f in ['load_scale', 'n_device', 'ttft_slo_scale', 'slo_tpot'] if f in df.columns]
    if not features:
        return
    
    df['slo_violation_rate'] *= 100

    for feature in features:
        if df[feature].nunique(dropna=False) <= 1:
            continue

        other_features = [f for f in features if f != feature]
        if other_features:
            grouped = df.groupby(other_features, dropna=False, sort=False)
            n_groups = grouped.ngroups
        else:
            grouped = [((), df)]
            n_groups = 1

        ncols = min(3, n_groups)
        nrows = math.ceil(n_groups / ncols)

        for xlabel, ylabel in [
            ('slo_violation_rate', 'energy_consumption'),
            ('slo_violation_rate', 'average_n_active_servers'),
            ('slo_violation_rate', feature)
        ]:
            if xlabel not in df.columns or ylabel not in df.columns:
                continue

            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
            idx = 0
            for other_feature_values, group in grouped:
                row, col = divmod(idx, ncols)
                ax = axes[row][col]
                idx += 1

                if not isinstance(other_feature_values, tuple):
                    other_feature_values = (other_feature_values,)

                method_curves = {}
                for (sched, route), pair_group in group.groupby(['scheduling_policy', 'routing_policy'], dropna=False, sort=False):
                    group_sorted = pair_group.sort_values(xlabel)
                    if is_disagg ^ ('disagg' in route):
                        continue 
                    method = f"{sched} / {route}"
                    method_curves[method] = group_sorted[[xlabel, ylabel]].dropna()
                    label = get_method_label(method)
                    ax.plot(
                        group_sorted[xlabel],
                        group_sorted[ylabel],
                        label=label,
                        **get_method_style(method),
                    )

                other_features_dict = {f: v for f, v in zip(other_features, other_feature_values)}
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(f'{ylabel} vs {xlabel}\n({other_features_dict})')
                if xlabel == 'slo_violation_rate':
                    for target in TARGET_SLO_VIOLATIONS:
                        ax.axvline(
                            target,
                            color=ANNOTATION_COLOR,
                            linestyle=':',
                            linewidth=1.0,
                            zorder=0,
                        )
                if xlabel == 'slo_violation_rate' and ylabel in ANNOTATED_YLABELS:
                    _annotate_threshold_comparison(ax, method_curves, ylabel)
                ax.legend()

            for extra_idx in range(idx, nrows * ncols):
                row, col = divmod(extra_idx, ncols)
                axes[row][col].set_visible(False)
            
            if xlim is not None: 
                for ax in axes.flatten():
                    ax.set_xlim(-0.2,xlim)

            fig.tight_layout()
            output_stem = output_dir / f'{ylabel}_vs_{xlabel}_change_{feature}_{name}'
            fig.savefig(f'{output_stem}.png', dpi=300)
            fig.savefig(f'{output_stem}.pdf', dpi=300)
            plt.close(fig)
            print(f"Saved plot to {output_stem}.png")
            print(f"Saved plot to {output_stem}.pdf")

    power_columns = {
        'n_device',
        'load_scale',
        'ttft_slo_scale',
        'slo_tpot',
        'per_server_power',
    }
    if power_columns.issubset(df.columns):
        power_df = df[(df['n_device'] == 8) & df['per_server_power'].notna()]
        power_groups = list(
            power_df.groupby(
                ['load_scale', 'ttft_slo_scale', 'slo_tpot'],
                dropna=False,
                sort=False,
            )
        )
        if power_groups:
            ncols = min(4, len(power_groups))
            nrows = math.ceil(len(power_groups) / ncols)
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(6 * ncols, 4.5 * nrows),
                squeeze=False,
            )

            for idx, ((load_scale, ttft_slo_scale, slo_tpot), tdf) in enumerate(power_groups):
                row, col = divmod(idx, ncols)
                ax = axes[row][col]

                for (sched, route), pair_group in tdf.groupby(
                    ['scheduling_policy', 'routing_policy'],
                    dropna=False,
                    sort=False,
                ):
                    if is_disagg ^ ('disagg' in route):
                        continue

                    method = f"{sched} / {route}"
                    label = get_method_label(method)
                    style = get_method_style(method)
                    for line_idx, (_, power_row) in enumerate(pair_group.iterrows()):
                        per_server_power = power_row['per_server_power']
                        if not isinstance(per_server_power, (list, tuple)):
                            continue
                        ax.plot(
                            range(len(per_server_power)),
                            per_server_power,
                            color=style['color'],
                            marker=style['marker'],
                            linewidth=style['linewidth'],
                            markersize=style['markersize'],
                            label=label if line_idx == 0 else None,
                        )

                ax.set_title(
                    f'load_scale={load_scale}, ttft_slo_scale={ttft_slo_scale}, '
                    f'slo_tpot={slo_tpot}'
                )
                ax.set_xlabel('Server')
                ax.set_ylabel('Power (W)')
                if ax.lines:
                    ax.legend()

            for extra_idx in range(len(power_groups), nrows * ncols):
                row, col = divmod(extra_idx, ncols)
                axes[row][col].set_visible(False)

            fig.tight_layout()
            output_stem = output_dir / f'per_server_power_{name}'
            fig.savefig(f'{output_stem}.png', dpi=300)
            fig.savefig(f'{output_stem}.pdf', dpi=300)
            plt.close(fig)
            print(f"Saved plot to {output_stem}.png")
            print(f"Saved plot to {output_stem}.pdf")

def main(
    result_files = [
        {
            'name': 'chat_23_7B', 
            'file': 'traces/report/chat23.jsonl',
            'xlim': 20
        },
        {
            'name': 'code_7B', 
            'file': 'traces/report/code.jsonl',
            'xlim': 20
        },
        {
            'name': 'mixed_7B', 
            'file': 'traces/report/mixed.jsonl',
            'xlim': 20
        }
    ]
):
    for kwargs in result_files:
        draw_figures(**kwargs)


if __name__ == '__main__':
    main()
