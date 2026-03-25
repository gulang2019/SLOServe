import math
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


def draw_figures(experiment_dir, df):
    if df.empty:
        return

    required_columns = {'scheduling_policy', 'routing_policy'}
    if not required_columns.issubset(df.columns):
        return

    os.makedirs(f'{experiment_dir}/figs', exist_ok=True)
    features = [f for f in ['load_scale', 'n_device', 'ttft_slo_scale', 'slo_tpot'] if f in df.columns]
    if not features:
        return

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
            (feature, 'energy_est'),
            (feature, 'slo_violation_rate'),
            (feature, 'energy_consumption'),
            (feature, 'energy_consumption_active'),
            (feature, 'energy_consumption_non_idle'),
            ('slo_violation_rate', 'energy_consumption'),
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

                for (sched, route), pair_group in group.groupby(['scheduling_policy', 'routing_policy'], dropna=False, sort=False):
                    group_sorted = pair_group.sort_values(xlabel)
                    label = f"{sched} / {route}"
                    ax.plot(group_sorted[xlabel], group_sorted[ylabel], marker='o', label=label)

                other_features_dict = {f: v for f, v in zip(other_features, other_feature_values)}
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(f'{ylabel} vs {xlabel}\n({other_features_dict})')
                ax.legend()

            for extra_idx in range(idx, nrows * ncols):
                row, col = divmod(extra_idx, ncols)
                axes[row][col].set_visible(False)

            fig.tight_layout()
            output_path = f'{experiment_dir}/figs/{ylabel}_vs_{xlabel}_change_{feature}.png'
            fig.savefig(output_path, dpi=300)
            plt.close(fig)
            print(f"Saved plot to {output_path}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(f'Usage: {sys.argv[0]} <directory>')

    root_dir = sys.argv[1]
    for entry in sorted(os.scandir(root_dir), key=lambda item: item.name):
        if not entry.is_dir():
            continue

        experiment_dir = entry.path
        results_path = os.path.join(experiment_dir, 'results.jsonl')
        if not os.path.isfile(results_path):
            continue

        df = pd.read_json(results_path, lines=True)
        draw_figures(experiment_dir, df)


if __name__ == '__main__':
    main()
