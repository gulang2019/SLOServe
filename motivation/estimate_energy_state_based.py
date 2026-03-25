import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from motivation.events_analysis import Energy, _compute_measured_power_series

IDLE_POWER_W = 70.0
NDEV = 8


def load_events(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def common_window(left: list[dict], right: list[dict]) -> tuple[float, float]:
    left_times = [float(e["timestamp"]) for e in left if e.get("event_type") == "energy"]
    right_times = [float(e["timestamp"]) for e in right if e.get("event_type") == "energy"]
    start = max(min(left_times), min(right_times))
    end = min(max(left_times), max(right_times))
    if end <= start:
        raise ValueError("No overlapping energy window between traces.")
    return start, end


def batch_type(event: dict) -> str | None:
    vals = list(event.get("num_scheduled_tokens", {}).values())
    if not vals:
        return None
    if max(vals) == 1:
        return "decode"
    if min(vals) > 1:
        return "prefill"
    return "mixed"


def _window_count(start: float, end: float, window_size: float) -> int:
    return max(1, int(math.ceil((end - start) / window_size)))


def measured_energy_bins(
    events: list[dict],
    start: float,
    end: float,
    window_size: float,
    recover_idle_gaps: bool = False,
    idle_power_per_device: float = IDLE_POWER_W,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    typed_events = [
        Energy(
            event_type="energy",
            timestamp=float(event["timestamp"]),
            device_id=int(event.get("device_id", -1)),
            energy=float(event.get("energy", 0.0) or 0.0),
            power=float(event.get("power", 0.0) or 0.0),
            mhz=float(event.get("mhz", 0.0) or 0.0),
        )
        for event in events
        if event.get("event_type") == "energy"
    ]
    summary = _compute_measured_power_series(
        typed_events,
        n_device=NDEV,
        window_size=window_size,
        start_time=start,
        end_time=end,
        recover_idle_gaps=recover_idle_gaps,
        idle_power_per_device=idle_power_per_device,
    )
    total = np.asarray(summary["total_power"], dtype=float) * window_size
    per_dev = {
        d: np.asarray(summary["per_device_power"].get(d, np.zeros_like(total)), dtype=float) * window_size
        for d in range(NDEV)
    }
    return total, per_dev


def state_features_by_window(
    events: list[dict],
    start: float,
    end: float,
    window_size: float,
) -> dict[int, dict[str, np.ndarray]]:
    nwin = _window_count(start, end, window_size)
    features = {
        d: {
            "decode_active": np.zeros(nwin, dtype=float),
            "mixed_active": np.zeros(nwin, dtype=float),
            "prefill_active": np.zeros(nwin, dtype=float),
            "decode_present": np.zeros(nwin, dtype=float),
            "mixed_present": np.zeros(nwin, dtype=float),
            "prefill_present": np.zeros(nwin, dtype=float),
        }
        for d in range(NDEV)
    }

    for event in events:
        if event.get("event_type") != "batch":
            continue
        ts = float(event["timestamp"])
        elapsed = float(event.get("elapsed", 0.0) or 0.0)
        batch_start = max(start, ts - elapsed)
        batch_end = min(end, ts)
        if batch_end <= batch_start:
            continue
        d = int(event.get("device_id", -1))
        if not 0 <= d < NDEV:
            continue
        typ = batch_type(event)
        if typ is None:
            continue
        active_count = float(len(event.get("req_ids", [])))
        left = int(math.floor((batch_start - start) / window_size))
        right = int(math.ceil((batch_end - start) / window_size))
        left = max(0, left)
        right = min(nwin, right)
        for idx in range(left, right):
            win_start = start + idx * window_size
            win_end = win_start + window_size
            overlap = max(0.0, min(batch_end, win_end) - max(batch_start, win_start))
            if overlap <= 0:
                continue
            frac = overlap / window_size
            features[d][f"{typ}_active"][idx] += active_count * frac
            features[d][f"{typ}_present"][idx] = 1.0

    return features


def fit_linear_nonnegative(X: np.ndarray, y: np.ndarray, force_idle: float | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if force_idle is None:
        A = np.c_[np.ones(len(y)), X]
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        return beta

    target = y - force_idle
    beta = np.linalg.lstsq(X, target, rcond=None)[0]
    beta = np.maximum(beta, 0.0)
    return np.concatenate([[force_idle], beta])

def add_recent_features(
    states: dict[int, dict[str, np.ndarray]],
    window_size: float,
    tau_seconds: float,
) -> dict[int, dict[str, np.ndarray]]:
    decay = math.exp(-window_size / max(tau_seconds, 1e-6))
    out: dict[int, dict[str, np.ndarray]] = {}
    for device_id, feat in states.items():
        enriched = {key: np.array(value, copy=True) for key, value in feat.items()}
        for typ in ("decode", "mixed", "prefill"):
            present = feat[f"{typ}_present"]
            recent = np.zeros_like(present)
            for idx in range(1, len(present)):
                recent[idx] = present[idx - 1] + decay * recent[idx - 1]
            enriched[f"{typ}_recent"] = recent
        out[device_id] = enriched
    return out


def predict_linear(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    A = np.c_[np.ones(len(X)), X]
    return A @ beta


def build_training_data(
    events: list[dict],
    start: float,
    end: float,
    window_size: float,
    tau_seconds: float = 0.0,
    recover_idle_gaps: bool = False,
    idle_power_per_device: float = IDLE_POWER_W,
) -> tuple[np.ndarray, np.ndarray]:
    total_energy, per_dev_energy = measured_energy_bins(
        events,
        start,
        end,
        window_size,
        recover_idle_gaps=recover_idle_gaps,
        idle_power_per_device=idle_power_per_device,
    )
    del total_energy
    states = state_features_by_window(events, start, end, window_size)
    if tau_seconds > 0.0:
        states = add_recent_features(states, window_size, tau_seconds)
    X_rows = []
    y_rows = []
    for d in range(NDEV):
        cols = [
            states[d]["decode_active"],
            states[d]["mixed_active"],
            states[d]["prefill_active"],
            states[d]["decode_present"],
            states[d]["mixed_present"],
            states[d]["prefill_present"],
        ]
        if tau_seconds > 0.0:
            cols.extend(
                [
                    states[d]["decode_recent"],
                    states[d]["mixed_recent"],
                    states[d]["prefill_recent"],
                ]
            )
        X = np.column_stack(cols)
        y = per_dev_energy[d] / window_size
        X_rows.append(X)
        y_rows.append(y)
    return np.vstack(X_rows), np.concatenate(y_rows)


def estimate_trace(
    events: list[dict],
    start: float,
    end: float,
    window_size: float,
    beta: np.ndarray,
    tau_seconds: float = 0.0,
    recover_idle_gaps: bool = False,
    idle_power_per_device: float = IDLE_POWER_W,
) -> tuple[np.ndarray, np.ndarray]:
    total_energy, per_dev_energy = measured_energy_bins(
        events,
        start,
        end,
        window_size,
        recover_idle_gaps=recover_idle_gaps,
        idle_power_per_device=idle_power_per_device,
    )
    del per_dev_energy
    states = state_features_by_window(events, start, end, window_size)
    if tau_seconds > 0.0:
        states = add_recent_features(states, window_size, tau_seconds)
    nwin = len(total_energy)
    pred_power = np.zeros(nwin, dtype=float)
    for d in range(NDEV):
        cols = [
            states[d]["decode_active"],
            states[d]["mixed_active"],
            states[d]["prefill_active"],
            states[d]["decode_present"],
            states[d]["mixed_present"],
            states[d]["prefill_present"],
        ]
        if tau_seconds > 0.0:
            cols.extend(
                [
                    states[d]["decode_recent"],
                    states[d]["mixed_recent"],
                    states[d]["prefill_recent"],
                ]
            )
        X = np.column_stack(cols)
        pred_power += np.maximum(predict_linear(beta, X), 0.0)
    return pred_power, total_energy / window_size


def summarize(label: str, pred_power: np.ndarray, meas_power: np.ndarray, window_size: float) -> None:
    pred_energy = float(np.sum(pred_power) * window_size)
    meas_energy = float(np.sum(meas_power) * window_size)
    mae = float(np.mean(np.abs(pred_power - meas_power)))
    print(label)
    print(f"  estimated_energy_j: {pred_energy:.3f}")
    print(f"  measured_energy_j: {meas_energy:.3f}")
    print(f"  ratio: {pred_energy / meas_energy:.3f}")
    print(f"  mae_w: {mae:.3f}")


def plot_comparison(
    rr_pred: np.ndarray,
    rr_meas: np.ndarray,
    pk_pred: np.ndarray,
    pk_meas: np.ndarray,
    window_size: float,
    output_prefix: str,
) -> None:
    time = np.arange(len(rr_meas), dtype=float) * window_size
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), tight_layout=True)

    axes[0].step(time, rr_meas, where="post", color="#c73e1d", label="RR measured")
    axes[0].step(time, rr_pred, where="post", color="#7f2704", linestyle="--", label="RR estimated")
    axes[0].set_title("Round Robin: Measured vs State-Based Estimated Power")
    axes[0].set_ylabel("Power (W)")
    axes[0].legend()

    axes[1].step(time, pk_meas, where="post", color="#1f78b4", label="Packer measured")
    axes[1].step(time, pk_pred, where="post", color="#08519c", linestyle="--", label="Packer estimated")
    axes[1].set_title("Packer: Measured vs State-Based Estimated Power")
    axes[1].set_xlabel("Time Since Common Window Start (s)")
    axes[1].set_ylabel("Power (W)")
    axes[1].legend()

    fig.savefig(f"{output_prefix}.png", dpi=250, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rr", required=True)
    parser.add_argument("--packer", required=True)
    parser.add_argument("--window-size", type=float, default=1.0)
    parser.add_argument("--recover-idle-gaps", action="store_true")
    parser.add_argument("--idle-power-per-device", type=float, default=IDLE_POWER_W)
    parser.add_argument(
        "--output-prefix",
        default="figs/state_based_energy_estimate",
    )
    args = parser.parse_args()

    rr_events = load_events(args.rr)
    pk_events = load_events(args.packer)
    start, end = common_window(rr_events, pk_events)

    best = None
    for tau_seconds in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
        X_rr, y_rr = build_training_data(
            rr_events,
            start,
            end,
            args.window_size,
            tau_seconds=tau_seconds,
            recover_idle_gaps=args.recover_idle_gaps,
            idle_power_per_device=args.idle_power_per_device,
        )
        X_pk, y_pk = build_training_data(
            pk_events,
            start,
            end,
            args.window_size,
            tau_seconds=tau_seconds,
            recover_idle_gaps=args.recover_idle_gaps,
            idle_power_per_device=args.idle_power_per_device,
        )
        X = np.vstack([X_rr, X_pk])
        y = np.concatenate([y_rr, y_pk])
        beta = fit_linear_nonnegative(X, y, force_idle=IDLE_POWER_W)
        rr_pred, rr_meas = estimate_trace(
            rr_events,
            start,
            end,
            args.window_size,
            beta,
            tau_seconds=tau_seconds,
            recover_idle_gaps=args.recover_idle_gaps,
            idle_power_per_device=args.idle_power_per_device,
        )
        pk_pred, pk_meas = estimate_trace(
            pk_events,
            start,
            end,
            args.window_size,
            beta,
            tau_seconds=tau_seconds,
            recover_idle_gaps=args.recover_idle_gaps,
            idle_power_per_device=args.idle_power_per_device,
        )
        mae = float(np.mean(np.abs(rr_pred - rr_meas))) + float(np.mean(np.abs(pk_pred - pk_meas)))
        if best is None or mae < best["mae"]:
            best = {
                "tau_seconds": tau_seconds,
                "beta": beta,
                "rr_pred": rr_pred,
                "rr_meas": rr_meas,
                "pk_pred": pk_pred,
                "pk_meas": pk_meas,
                "mae": mae,
            }

    tau_seconds = best["tau_seconds"]
    beta = best["beta"]
    rr_pred = best["rr_pred"]
    rr_meas = best["rr_meas"]
    pk_pred = best["pk_pred"]
    pk_meas = best["pk_meas"]

    print("Model")
    print(f"  tail_tau_seconds = {tau_seconds:.3f}")
    print(
        "  power = "
        f"{beta[0]:.3f}"
        f" + {beta[1]:.3f} * decode_active"
        f" + {beta[2]:.3f} * mixed_active"
        f" + {beta[3]:.3f} * prefill_active"
        f" + {beta[4]:.3f} * decode_present"
        f" + {beta[5]:.3f} * mixed_present"
        f" + {beta[6]:.3f} * prefill_present"
    )
    if len(beta) >= 10:
        print(
            "  tail = "
            f"{beta[7]:.3f} * decode_recent"
            f" + {beta[8]:.3f} * mixed_recent"
            f" + {beta[9]:.3f} * prefill_recent"
        )

    summarize("round_robin", rr_pred, rr_meas, args.window_size)
    summarize("packer", pk_pred, pk_meas, args.window_size)

    output_prefix = str(Path(args.output_prefix))
    plot_comparison(rr_pred, rr_meas, pk_pred, pk_meas, args.window_size, output_prefix)
    print(f"Saved {output_prefix}.png")
    print(f"Saved {output_prefix}.pdf")


if __name__ == "__main__":
    main()
