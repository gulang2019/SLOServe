#!/usr/bin/env python3

"""
Train a simple, illustrative predictor (Logistic Regression) and plot ROC + Confusion Matrix.
Also supports comparing two feature sets:
  - "queue": only ["running_size", "waiting_size"]
  - "all":   all columns except the target
  - "both":  train/evaluate/save for both sets

Usage examples:
  # Train both sets and save artifacts next to the data file
  python train_predictor.py --data "/path/to/admission_history.jsonl" --target is_rejected --feature_set both

  # Predict one example using previously saved model/scaler
  python train_predictor.py --predict \
    --predict_one '{"rejection_rate":0.1,"past_utilization":0.2,"future_utilization":0.15,"waiting_size":3,"running_size":2,"input_length":300,"output_length":80,"prefill_ddl":0.18}' \
    --model_out "./model_all.joblib" --scaler_out "./scaler_all.joblib"

Artifacts saved:
  - model_*.joblib, scaler_*.joblib
  - roc_curve_*.png
  - confusion_matrix_*.png
  - metrics_*.json
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)

import joblib


QUEUE_FEATURES = ["running_size", "waiting_size"]


def load_json_or_jsonl(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        head = f.read(2)
    if head.strip().startswith("["):
        with open(path, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)

    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def plot_confusion_matrix(cm: np.ndarray, out_path: str, title: str):
    fig = plt.figure(figsize=(4.5, 4))
    ax = plt.gca()
    im = ax.imshow(cm)  # default colormap; no explicit colors set
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    # tick labels for binary classification
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0 (accept)", "1 (reject)"], rotation=30, ha="right")
    ax.set_yticklabels(["0 (accept)", "1 (reject)"])
    # annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def train_eval_save(df: pd.DataFrame, target: str, feature_set: str, outdir: str) -> Dict[str, Any]:
    # Choose features
    if feature_set == "queue":
        features = [f for f in QUEUE_FEATURES if f in df.columns]
    else:
        # all non-target columns
        features = [c for c in df.columns if c != target]

    X = df[features]
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    rep = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    # Save artifacts
    suffix = feature_set
    model_path = os.path.join(outdir, f"model_{suffix}.joblib")
    scaler_path = os.path.join(outdir, f"scaler_{suffix}.joblib")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # ROC curve
    fig = plt.figure(figsize=(6, 4))
    RocCurveDisplay.from_estimator(model, X_test_s, y_test)
    plt.title(f"ROC Curve ({suffix}) AUC = {auc:.3f}")
    roc_path = os.path.join(outdir, f"roc_curve_{suffix}.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Confusion matrix
    cm_path = os.path.join(outdir, f"confusion_matrix_{suffix}.png")
    plot_confusion_matrix(cm, cm_path, f"Confusion Matrix ({suffix})")

    # Metrics JSON
    metrics = {
        "feature_set": suffix,
        "features_used": features,
        "accuracy": rep["accuracy"],
        "precision_1": rep["1"]["precision"],
        "recall_1": rep["1"]["recall"],
        "f1_1": rep["1"]["f1-score"],
        "auc": auc,
        "confusion_matrix": cm.tolist(),
        "model_path": model_path,
        "scaler_path": scaler_path,
        "roc_curve_path": roc_path,
        "confusion_matrix_path": cm_path,
    }
    with open(os.path.join(outdir, f"metrics_{suffix}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def predict_one(json_str: str, model_path: str, scaler_path: str) -> Dict[str, Any]:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    feat = json.loads(json_str)
    X = pd.DataFrame([feat])
    Xs = scaler.transform(X)
    prob = float(model.predict_proba(Xs)[:, 1][0])
    pred = int(prob >= 0.5)
    return {"predicted_is_rejected": pred, "probability": prob}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to JSON/JSONL data file.")
    parser.add_argument("--target", default="is_rejected")
    parser.add_argument("--feature_set", choices=["all", "queue", "both"], default="both")
    parser.add_argument("--outdir", default=None, help="Where to save models/plots (default: data dir).")
    parser.add_argument("--predict_one", help="JSON string of one example to score (used with --predict).")
    parser.add_argument("--model_out", default="model_all.joblib", help="Model path for --predict.")
    parser.add_argument("--scaler_out", default="scaler_all.joblib", help="Scaler path for --predict.")
    parser.add_argument("--predict", action="store_true", help="Run prediction with saved model/scaler.")
    args = parser.parse_args()

    if args.predict and args.predict_one:
        res = predict_one(args.predict_one, args.model_out, args.scaler_out)
        print(json.dumps(res, indent=2))
        return

    if not args.data:
        print("--data is required when training.", file=sys.stderr)
        sys.exit(1)

    df = load_json_or_jsonl(args.data)
    if df.empty:
        print("Loaded an empty DataFrame. Check your input file.", file=sys.stderr)
        sys.exit(1)

    if df[args.target].dtype != int:
        df[args.target] = df[args.target].astype(int)

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.data)) or "."
    os.makedirs(outdir, exist_ok=True)

    sets = ["all", "queue"] if args.feature_set == "both" else [args.feature_set]

    results = {}
    for s in sets:
        metrics = train_eval_save(df, args.target, s, outdir)
        results[s] = metrics

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
