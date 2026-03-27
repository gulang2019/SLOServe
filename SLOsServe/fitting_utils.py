from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls


BatchTimingSample = tuple[list[tuple[int, int]], float]


def sanitize_filename(text: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return sanitized or "default"


def write_json(path: str | Path, payload: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


def save_prediction_scatter(
    path: str | Path,
    measured_times: list[float],
    predicted_times: list[float],
    title: str,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    measured = np.asarray(measured_times, dtype=float)
    predicted = np.asarray(predicted_times, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
    ax.scatter(measured, predicted, s=14, alpha=0.7)
    if measured.size > 0 and predicted.size > 0:
        lo = float(min(measured.min(), predicted.min()))
        hi = float(max(measured.max(), predicted.max()))
        ax.plot([lo, hi], [lo, hi], "--r", linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured Time (s)")
    ax.set_ylabel("Predicted Time (s)")
    ax.set_title(title)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def fit_linear_perf_model(
    batch_times: list[BatchTimingSample],
    *,
    min_abs_num_reqs_coef: float = 1e-9,
) -> dict[str, Any]:
    xs: list[list[float]] = []
    ys: list[float] = []
    records: list[dict[str, Any]] = []

    for batch, measured_time in batch_times:
        normalized_batch = [
            (int(past_tokens), int(current_tokens))
            for past_tokens, current_tokens in batch
        ]
        if not normalized_batch:
            continue

        total_current_tokens = int(sum(current_tokens for _, current_tokens in normalized_batch))
        total_past_tokens = int(sum(past_tokens for past_tokens, _ in normalized_batch))
        num_reqs = len(normalized_batch)
        num_decode_steps = 1.0
        measured = float(measured_time)

        xs.append([
            float(total_current_tokens),
            float(num_reqs),
            float(total_past_tokens),
            1.0,
        ])
        ys.append(measured)
        records.append({
            "batch": [
                {
                    "past_tokens": past_tokens,
                    "scheduled_tokens": current_tokens,
                }
                for past_tokens, current_tokens in normalized_batch
            ],
            "batch_size": num_reqs,
            "total_current_tokens": total_current_tokens,
            "total_past_tokens": total_past_tokens,
            "num_decode_steps": int(num_decode_steps),
            "measured_time": measured,
        })

    if not xs:
        raise ValueError("No valid batch samples to fit.")

    x_data = np.asarray(xs, dtype=float)
    y_data = np.asarray(ys, dtype=float)
    params, _ = nnls(x_data, y_data)

    adjusted_params = np.asarray(params, dtype=float).copy()
    raw_num_reqs_coef = float(adjusted_params[1])
    clamped_num_reqs_coef = False
    min_abs_num_reqs_coef = float(abs(min_abs_num_reqs_coef))
    if min_abs_num_reqs_coef > 0.0 and abs(raw_num_reqs_coef) < min_abs_num_reqs_coef:
        sign = -1.0 if raw_num_reqs_coef < 0.0 else 1.0
        adjusted_params[1] = sign * min_abs_num_reqs_coef
        clamped_num_reqs_coef = True

    predicted = x_data @ adjusted_params

    residuals = y_data - predicted
    centered = y_data - np.mean(y_data)
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum(centered ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    predicted_list = predicted.astype(float).tolist()
    for record, predicted_time in zip(records, predicted_list):
        record["predicted_time"] = float(predicted_time)

    return {
        "hardware_params": [
            float(adjusted_params[0]),
            float(adjusted_params[1]),
            float(adjusted_params[2]),
            0.0,
            float(adjusted_params[3]),
        ],
        "predicted_times": predicted_list,
        "measured_times": y_data.astype(float).tolist(),
        "records": records,
        "stats": {
            "num_samples": len(records),
            "mae": mae,
            "rmse": rmse,
            "r2": float(r2),
            "fit_method": "nnls",
            "non_negative_constraints_applied": True,
            "raw_num_reqs_coef": raw_num_reqs_coef,
            "min_abs_num_reqs_coef": min_abs_num_reqs_coef,
            "num_reqs_coef_was_clamped": clamped_num_reqs_coef,
        },
    }
