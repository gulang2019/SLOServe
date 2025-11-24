
import json
import os
import math
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)

# For nice DataFrame display in this environment
try:
    from caas_jupyter_tools import display_dataframe_to_user
except Exception:
    display_dataframe_to_user = None


plt.rcParams.update({
    # --- Figure layout ---
    "figure.figsize": (7, 4),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",

    # --- Font / text ---
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 18,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,

    # --- Axes and ticks ---
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,

    # --- Lines and markers ---
    "lines.linewidth": 2.0,
    "lines.markersize": 6,

    # --- Legend ---
    "legend.frameon": False,
    "legend.loc": "best",

    # --- PDF/LaTeX export compatibility ---
    "pdf.fonttype": 42,   # TrueType
    "ps.fonttype": 42,    # TrueType
})

# === Utilities ===
def load_json_or_jsonl(paths: list[str]) -> pd.DataFrame:
    rows = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
            rows.extend(data)
    return pd.DataFrame(rows)

def fit_logistic_regression(df, features, target):
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train logistic regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_s, y_train)

    y_prob = log_reg.predict_proba(X_test_s)[:, 1]

    # --- Find the best threshold ---
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    youdens_j = tpr - fpr
    best_threshold = thresholds[np.argmax(youdens_j)]

    # Optionally check F1-optimized threshold
    f1_scores = [f1_score(y_test, y_prob >= t) for t in thresholds]
    best_f1_threshold = thresholds[np.argmax(f1_scores)]

    print(f"Best threshold (Youden’s J): {best_threshold:.3f}")
    print(f"Best threshold (F1): {best_f1_threshold:.3f}")

    # Evaluate using best threshold
    y_pred_opt = (y_prob >= best_threshold).astype(int)
    rep = classification_report(y_test, y_pred_opt, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)

    print("Validation accuracy:", rep["accuracy"])
    print("ROC-AUC:", auc)

    return log_reg, scaler, best_threshold

def export_feature_oriented(model, scaler, feature_names, output_name, threshold=0.5):
    import json
    import numpy as np

    coef = model.coef_[0]
    intercept = float(model.intercept_[0])

    means = scaler.mean_
    stds = getattr(scaler, "scale_", None)  # StandardScaler std
    if stds is None:
        stds = getattr(scaler, "std_", None)
    if stds is None:
        raise ValueError("Scaler must provide standard deviations via .scale_ or .std_")

    out = {}
    for name, c, m, s in zip(feature_names, coef, means, stds):
        out[name] = {"coeff": float(c), "mean": float(m), "std": float(s)}

    out["intercept"] = intercept
    out["threshold"] = float(threshold)
    return out

    
    return out

def fit(data_paths, features, target, name, threshold=0.5, per_device = False):
    df = load_json_or_jsonl(data_paths)
    if per_device:
        model = {}
        for device_id, device_df in df.groupby('device_id'):
            print('device_id:', device_id, 'len(device_df):', len(device_df))
            # print(device_df)
            log_reg, scaler, best_threshold = fit_logistic_regression(device_df, features, target)
            model[device_id] = export_feature_oriented(log_reg, scaler, features, name, best_threshold)
    else: 
        log_reg, scaler, best_threshold = fit_logistic_regression(df, features, target)
        model = export_feature_oriented(log_reg, scaler, features, name, best_threshold)
    with open('auto_scaling_model.json', "r") as f:
        models = json.load(f)
    model['per_device'] = per_device
    models[name] = model
    with open('auto_scaling_model.json', "w") as f:
        json.dump(models, f, indent=2)

def calc_rejection_prob(model, features, threshold=None) -> float:
    def inner(model, features, threshold=None):
        intercept = float(model.get("intercept", 0.0))
        feature_keys = [k for k in model.keys() if k not in ("intercept", "threshold")]

        # Compute standardized linear score: b + sum_i w_i * (x_i - mean_i) / std_i
        score = intercept
        for feat in feature_keys:
            if feat not in features:
                # If missing, treat as mean (contributes 0) or raise—your call
                continue
            x = float(features[feat])
            m = float(model[feat]["mean"])
            s = float(model[feat]["std"])
            s = s if s != 0.0 else 1.0  # guard against divide-by-zero
            z = (x - m) / s
            score += float(model[feat]["coeff"]) * z

        # Logistic probability
        # if score < -10 or score > 10:
        #     print(f"Score {score} out of expected range [-10, 10]. Clipping. model = {model}, features = {features}")
        score = max(min(score, 10), -10)
        prob = 1.0 / (1.0 + math.exp(-score))
        threshold = threshold or model.get('threshold', 0.5)
        return prob, prob >= threshold
    # print('model:', model)
    # print('features:', features)
    # print(model[features['device_id']])
    if model.get('per_device', False):
        return inner(model[str(features['device_id'])], features)
    else:
        return inner(model, features)
    
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.ticker import MaxNLocator

def _merge_ticks(base_ticks, target_ticks, *, min_sep=0.00, max_bins=8):
    """Merge tick sets while keeping spacing and limiting count."""
    merged = sorted(set(np.round(target_ticks, 2)))  # ensure 2-digit precision
    for t in sorted(set(np.round(base_ticks, 2))):
        if any(abs(t - m) < min_sep for m in merged):
            continue
        merged.append(t)
        merged = sorted(merged)
        if len(merged) >= max_bins:
            break
    return merged

def plot_fpr_fnr_tradeoff(
    y_true, probs, ax=None, name=None,
    n_labels_low=10, n_labels_high=5, low_fnr_cutoff=0.25,
    fnr_targets=(0.70, 0.40, 0.10, 0.01),
    fontsize=20,
    do_threshold_labeling=True,
    max_xticks=8, max_yticks=8,
    min_tick_sep=0.10, rotate_xticks=60, tick_pad=8,
):
    """Plot FPR–FNR trade-off with red dashed reference lines and clean 2-digit ticks."""
    y_true, probs = np.array(y_true), np.array(probs)

    thresholds = np.linspace(0, 1, 300)
    fprs, fnrs = [], []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        fprs.append(fpr)
        fnrs.append(fnr)
    fprs, fnrs = np.array(fprs), np.array(fnrs)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 7.5))

    ax.plot(fprs, fnrs, lw=2.5, label=f"{name or 'Model'}")
    # ax.plot(fprs, fnrs, lw=2.5)

    if do_threshold_labeling:
        low_idxs = np.where(fnrs <= low_fnr_cutoff)[0]
        high_idxs = np.where(fnrs > low_fnr_cutoff)[0]
        label_idxs = []
        if len(low_idxs) > 0:
            label_idxs.extend(np.linspace(low_idxs[0], low_idxs[-1], n_labels_low, dtype=int))
        if len(high_idxs) > 0:
            label_idxs.extend(np.linspace(high_idxs[0], high_idxs[-1], n_labels_high, dtype=int))
        for idx in label_idxs:
            ax.text(fprs[idx], fnrs[idx], f"{thresholds[idx]:.2f}",
                    fontsize=fontsize * 0.7, ha="left", va="bottom")

    target_x, target_y = [], []
    for i, target in enumerate(fnr_targets):
        idx = np.argmin(np.abs(fnrs - target))
        fpr_val, fnr_val = float(fprs[idx]), float(fnrs[idx])
        # ax.axhline(y=fnr_val, color="red", linestyle="--", alpha=0.6)
        ax.plot([0, fpr_val], [fnr_val, fnr_val], color="red", linestyle="--", alpha=0.6, zorder=2)
        ax.plot([fpr_val, fpr_val], [0, fnr_val], color="red", linestyle="--", alpha=0.6)
        ax.text(fpr_val, fnr_val, "ABCDEFGHI"[i], fontsize=25, ha="left", va="bottom", color = "black")
        target_x.append(round(fpr_val, 2))
        target_y.append(round(fnr_val, 2))

    ax.plot([0, 1], [1, 0], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=fontsize)
    ax.set_ylabel("False Negative Rate", fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=fontsize, pad=tick_pad)
    
    # ---- tick de-crowding ----
    xloc = MaxNLocator(nbins=max_xticks)
    yloc = MaxNLocator(nbins=max_yticks)
    base_xticks = xloc.tick_values(*ax.get_xlim())
    base_yticks = yloc.tick_values(*ax.get_ylim())
    xticks = _merge_ticks(base_xticks, target_x, min_sep=min_tick_sep, max_bins=max_xticks)
    yticks = _merge_ticks(base_yticks, fnr_targets, min_sep=min_tick_sep, max_bins=max_yticks)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Format with 2-digit precision
    ax.set_xticklabels([f"{x:.2f}" for x in xticks], fontsize=fontsize, rotation=rotate_xticks)
    ax.set_yticklabels([f"{y:.2f}" for y in yticks], fontsize=fontsize)

    ax.margins(x=0.1, y=0.1)
    # ax.set_xlim(-0.1, 1.1)
    # ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    return fprs, fnrs, thresholds


def eval_auto_scaling(name, data_path, label_key="is_rejected", threshold=None, ax = None, do_threshold_labeling = False, label = None, flip = False):
    # Load model JSON (feature-oriented)
    with open('auto_scaling_model.json', "r") as f:
        models = json.load(f)
    model = models[name]

    # Extract constants and feature list
    intercept = float(model.get("intercept", 0.0))
    feature_keys = [k for k in model.keys() if k not in ("intercept", "threshold")]

    # Load data: array of dicts
    with open(data_path, "r") as f:
        data = json.load(f)

    n_correct = 0
    tp, fp, fn, tn = 0, 0, 0, 0
    n_total = 0
    
    scores = []
    preds, probs, y_true = [], [], []
    from collections import defaultdict
    device_id2idx = defaultdict(list)
    for i, row in enumerate(data):
        prob, pred = calc_rejection_prob(model, row, threshold)
        probs.append(prob)
        preds.append(pred)
        y_true.append(row[label_key])
        # Accuracy only if label present
        if label_key in row:
            if pred == 1 and row[label_key] == 0:
                fp += 1   # False Positive: predicted 1 but label is 0
            elif pred == 0 and row[label_key] == 1:
                fn += 1   # False Negative: predicted 0 but label is 1
            elif pred == 1 and row[label_key] == 1:
                tp += 1   # True Positive
            elif pred == 0 and row[label_key] == 0:
                tn += 1   # True Negative

            n_correct += int(pred == int(row[label_key]))
            n_total += 1
        device_id2idx[row['device_id']].append(i)

    import matplotlib.pyplot as plt
    # plot_fpr_fnr_tradeoff(y_true, probs, do_threshold_labeling = do_threshold_labeling)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True)
    if model.get('per_device', False):
        for device_id, idx in device_id2idx.items():
            y_true_device = [y_true[i] for i in idx]
            probs_device = [probs[i] for i in idx]
            plot_fpr_fnr_tradeoff(y_true_device, probs_device, ax, f"{label or name} (Device {device_id})", do_threshold_labeling = do_threshold_labeling)
    plot_fpr_fnr_tradeoff(y_true, probs, ax, label or name, do_threshold_labeling = do_threshold_labeling)
    if not flip:
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("False Negative Rate")
    else:
        ax.set_xlabel("False Negative Rate")
        ax.set_ylabel("False Positive Rate")
    ax.legend()
    ax.grid(True)
    # fig.savefig(f'{name}_fpr_fnr.png')
    # print(f'Saved {name}_fpr_fnr.png')
    # plt.close()
    
    test_acc = (n_correct / n_total) if n_total else float("nan")
    print(f"Test accuracy for {name}: {test_acc}")
    print(f"fn: {fn}, fp: {fp}, tp: {tp}, tn: {tn}")
    
    return {
        'scaling_pred_fpr': fp / (fp + tn + 0.00001),
        'scaling_pred_fnr': fn / (fn + tp + 0.00001),
        'scaling_pred_acc': test_acc
    }

def fit_ours():
    # === Configuration ===
    TRAIN_PATHS = [
        "profile_admission_history_auto_scaling.jsonl",
        "profile_admission_history_auto_scaling-1.jsonl",
        "profile_admission_history_auto_scaling-2.jsonl"
    ]
    TRAIN_PATHS = [
        # "experiments/Qwen-7B_constant_azure_chat_23:azure_chat_23_601:1202_anytime/slosserve-edf_auto_scaling-all-0.12_4.0_4_anytime_5.0_0.1.0.admission_history.jsonl"
        # "admission_history-slosserve.json",
        # "experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4579_anytime/slosserve-edf_auto_scaling-load_slo_req-1.0_1.2_1_anytime_3.0_0.025.admission_history.jsonl"
        # "admission_history.jsonl",
        # "experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3978:4579_anytime/slosserve-edf_auto_scaling_resch-load_slo-1.0_1.0_1_anytime_3.0_0.025.admission_history.jsonl",
        # "experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3978:4579_anytime/slosserve-edf_auto_scaling-all-0.3_1.5_4_anytime_3.0_0.025.admission_history.jsonl"
        # "experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3978:4579_anytime/slosserve-edf_auto_scaling_resch-load_slo-0.4_1.0_4_anytime_3.0_0.025.admission_history.jsonl"
        # "experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3978:4579_anytime/slosserve-edf_auto_scaling_resch-all-0.04_1.5_4_anytime_3.0_0.025.admission_history.jsonl"
        # "experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3978:4579_anytime/slosserve-edf_auto_scaling_resch-all-0.15_1.5_4_anytime_3.0_0.025.admission_history.jsonl",
        # "experiments/Qwen-7B_constant_azure_chat_23:azure_code_23_3978:4579_anytime/slosserve-edf_auto_scaling_resch-all_chat-1.0_0.9_1_anytime_5.0_0.1.admission_history.jsonl"
        # "experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3978:4579_anytime/slosserve-edf_auto_scaling-all-0.07_1.5_4_anytime_3.0_0.025.admission_history.jsonl"
        # "experiments/Qwen-7B_constant_azure_chat_23:azure_code_23_3978:4579_anytime/slosserve-edf_auto_scaling_resch-all_chat-1.0_0.9_1_anytime_5.0_0.1.admission_history.jsonl",
        "experiments_mock/Qwen-7B_constant_sharegpt_code:azure_chat_23_3978:5978_anytime_0.0/slosserve-edf_round_robin_1.0_1_anytime_3.0_0.025.admission_history.jsonl"
    ]
    TEST_PATH = "experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3978:4579_anytime/slosserve-edf_auto_scaling-all-0.10_1.5_4_anytime_3.0_0.025.admission_history.jsonl"
    # TEST_PATH = "experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3978:4579_anytime/slosserve-edf_auto_scaling_resch-all-0.05_1.5_4_anytime_3.0_0.025.events.jsonl"
    # # TEST_PATH = "experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4579_anytime/slosserve-edf_auto_scaling_resch-load_slo_req-0.08_1.0_4_anytime_3.0_0.025.admission_history.jsonl"
    # TEST_PATH = "experiments/Qwen-7B_constant_azure_chat_23:azure_code_23_3978:4579_anytime/slosserve-edf_auto_scaling_resch-all-0.015_0.9_4_anytime_5.0_0.1.events.jsonl"
    # TEST_PATH = "experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4579_anytime/slosserve-edf_auto_scaling-load_slo_req-1.0_1.2_1_anytime_3.0_0.025.admission_history.jsonl"
    # TEST_PATH = TRAIN_PATHS[-1]
    # TEST_PATH = "experiments/Qwen-7B_constant_azure_chat_23:azure_chat_23_601:1202_anytime/slosserve-edf_auto_scaling-all-0.12_4.0_4_anytime_5.0_0.1.0.admission_history.jsonl"
    TEST_PATH = "experiments_mock/Qwen-7B_constant_sharegpt_code:azure_chat_23_3978:4100_anytime_0.0/slosserve-edf_auto_scaling_resch-all_chat-0.015_1.0_4_anytime_3.0_0.025.admission_history.jsonl"
    TEST_PATH = "experiments_mock/Qwen-7B_constant_sharegpt_code:azure_chat_23_3978:5978_anytime_0.0/slosserve-edf_round_robin_1.0_1_anytime_3.0_0.025.admission_history.jsonl"
    TARGET = "slo_violation"
    
    # REQUEST_FEATURES = []
    SLO_FEATURES = ["past_utilization", "future_utilization", "rejection_rate"]
    LOAD_FEATURES = ["n_requests", "input_length", "prefill_ddl", "running_size", "waiting_size"]
    PREDICTOR_NAME = "all_mock"

    # df = load_json_or_jsonl([TEST_PATH])
    # y_prob = df['rejection_prob']
    # y_pred = (y_prob >= 0.5).astype(int)
    # y_true = df[TARGET]
    # rep = classification_report(y_true, y_pred, output_dict=True)
    # print(f'Execution Accuracy: {rep["accuracy"]}')
    threshold = 0.05
    # fit(TRAIN_PATHS, REQUEST_FEATURES + SLO_FEATURES + LOAD_FEATURES, TARGET, "load_slo_req", threshold)
    # fit(TRAIN_PATHS, LOAD_FEATURES, TARGET, "load", threshold)
    # fit(TRAIN_PATHS, LOAD_FEATURES + SLO_FEATURES, TARGET, "all_chat", threshold)
    fit(TRAIN_PATHS, LOAD_FEATURES + SLO_FEATURES, TARGET, PREDICTOR_NAME, threshold, per_device = False)
    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
    # eval_auto_scaling("load_slo_req", TEST_PATH, TARGET, ax = ax, do_threshold_labeling = True)
    # eval_auto_scaling("load_slo", TEST_PATH, TARGET, ax = ax, do_threshold_labeling = True, label = "Load & SLO Features")
    # eval_auto_scaling("load", TEST_PATH, TARGET, ax = ax, do_threshold_labeling = False, label = "Load Features")
    eval_auto_scaling(PREDICTOR_NAME, TEST_PATH, TARGET, ax = ax, do_threshold_labeling = False, label = "In-domain")
    eval_auto_scaling('all', TEST_PATH, TARGET, ax = ax, do_threshold_labeling = False, label = "Out-of-domain")
    # eval_auto_scaling("all_chat", TEST_PATH, TARGET, ax = ax, do_threshold_labeling = False, label = "All+")
    
    # eval_auto_scaling("load", TEST_PATH, TARGET, ax = ax, do_threshold_labeling = True, label = "Load & Req. Features Only")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(f'auto_scaling_fpr_fnr.png')
    print(f'Saved auto_scaling_fpr_fnr.png')
    plt.close()

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="vllm+", help="name of the model")
    parser.add_argument("--test_only", 'store_true', default=False, help="only test the model")
    return parser.parse_args()

def main():
    # === Configuration ===
    args = parse_args()
    name = args.name
    if not args.test_only:
        TRAIN_PATHS = [
            f"experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3978:4579_anytime/qlm+_round_robin_1.5_1_anytime_3.0_0.025.admission_history.jsonl",
        ]
        # TEST_PATH = "experiments/Qwen-7B_constant_azure_chat_23:azure_chat_23_601:1202_anytime/slosserve-edf_auto_scaling-all-0.12_4.0_4_anytime_5.0_0.1.0.admission_history.jsonl"
        TARGET = "slo_violation"
        REQUEST_FEATURES = ["running_size", "waiting_size", "input_length", "output_length", "prefill_ddl"]
        SLO_FEATURES = ["past_utilization", "future_utilization", "rejection_rate"]
        LOAD_FEATURES = ["n_requests"]

        threshold = 0.1
        fit(TRAIN_PATHS, REQUEST_FEATURES + SLO_FEATURES + LOAD_FEATURES, TARGET, f"load_slo_req-{name}", threshold)
        fit(TRAIN_PATHS, LOAD_FEATURES + SLO_FEATURES, TARGET, f"load_slo-{name}", threshold)
        fit(TRAIN_PATHS, LOAD_FEATURES + REQUEST_FEATURES, TARGET, f"load_reqs_{name}", threshold)
        fit(TRAIN_PATHS, LOAD_FEATURES, TARGET, f"load_{name}", threshold)
    
    TEST_PATH = f"experiments_mock/Qwen-7B_constant_sharegpt_code:azure_chat_23_3978:4100_anytime_0.0/slosserve-edf_auto_scaling_resch-all_chat-0.015_1.0_4_anytime_3.0_0.025.events.jsonl"
    
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    # eval_auto_scaling("load_slo_req", TEST_PATH, TARGET, ax = ax)
    # eval_auto_scaling("load_slo", TEST_PATH, TARGET, ax = ax)
    eval_auto_scaling(f"load_{name}", TEST_PATH, TARGET, ax = ax, label = "Load Features Only")
    eval_auto_scaling(f"load_reqs_{name}", TEST_PATH, TARGET, ax = ax, flip = True, label = "Load & Request Features")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(f'auto_scaling_fpr_fnr-{name}.png')
    fig.savefig(f'auto_scaling_fpr_fnr-{name}.pdf')
    print(f'Saved auto_scaling_fpr_fnr-{name}.png')
    plt.close()
    
if __name__ == "__main__":
    fit_ours()
    # main('qlm')

# auto_scaling-load_slo_req-0.05:slosserve-edf auto_scaling-load_slo_req-0.10:slosserve-edf auto_scaling-load_slo_req-0.2:slosserve-edf auto_scaling-load_slo_req-0.4:slosserve-edf auto_scaling-load_slo_req-0.8:slosserve-edf
# auto_scaling-load_slo-0.05:slosserve-edf auto_scaling-load_slo-0.10:slosserve-edf auto_scaling-load_slo-0.2:slosserve-edf auto_scaling-load_slo-0.4:slosserve-edf auto_scaling-load_slo-0.8:slosserve-edf
# auto_scaling-load-0.05:slosserve-edf auto_scaling-load-0.10:slosserve-edf auto_scaling-load-0.2:slosserve-edf auto_scaling-load-0.4:slosserve-edf auto_scaling-load-0.8:slosserve-edf
