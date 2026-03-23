import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from SLOsServe.analysis.headroom_analysis import (
    Instance,
    _get_dataset_data,
    calc_avg_num_servers,
)


def _policy_label(is_oracle: bool) -> str:
    return "oracle_mem" if is_oracle else "baseline"


def run_scaling(
    *,
    arrival_pattern: str,
    length_pattern: str,
    model_name: str,
    slo_ttft_scale: float,
    slo_tpot: float,
    server_counts: list[int],
    enforce_batch_memory_budget: bool,
) -> list[dict]:
    arrival_times, requests = _get_dataset_data(arrival_pattern, length_pattern)
    n_requests = len(arrival_times)
    rows: list[dict] = []
    for n_server in server_counts:
        for is_oracle in (False, True):
            Instance._printed_kv_cache_info = False
            (
                _avg2peak,
                _peak2min,
                _n_active,
                _n_total,
                _idle_ratio,
                aggregated_failures,
                _per_instance_failures,
                rejection_summary,
            ) = calc_avg_num_servers(
                arrival_pattern=arrival_pattern,
                length_pattern=length_pattern,
                model_name=model_name,
                slo_ttft_scale=slo_ttft_scale,
                slo_tpot=slo_tpot,
                n_server=n_server,
                is_oracle=is_oracle,
                arrival_times_list=arrival_times,
                requests_list=requests,
                verbose=False,
                enforce_batch_memory_budget=enforce_batch_memory_budget,
            )
            row = {
                "policy": _policy_label(is_oracle),
                "n_server": n_server,
                "n_requests": n_requests,
                "admitted_requests": n_requests - rejection_summary["total"],
                "reject_total": rejection_summary["total"],
                "reject_comp": rejection_summary["comp"],
                "reject_mem": rejection_summary["mem"],
                "reject_unknown": rejection_summary["unknown"],
                "fail_comp_attempts": aggregated_failures["comp"],
                "fail_mem_attempts": aggregated_failures["mem"],
                "fail_oom": aggregated_failures["oom"],
                "enforce_batch_memory_budget": int(enforce_batch_memory_budget),
            }
            rows.append(row)
            print(row)
    return rows


def write_csv(rows: list[dict], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header = [
        "policy",
        "n_server",
        "n_requests",
        "admitted_requests",
        "reject_total",
        "reject_comp",
        "reject_mem",
        "reject_unknown",
        "fail_comp_attempts",
        "fail_mem_attempts",
        "fail_oom",
        "enforce_batch_memory_budget",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_rows(rows: list[dict], fig_path: str, title: str) -> None:
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    server_counts = sorted({int(row["n_server"]) for row in rows})
    policy_rows = {
        policy: sorted(
            [row for row in rows if row["policy"] == policy],
            key=lambda row: row["n_server"],
        )
        for policy in ("baseline", "oracle_mem")
    }

    colors = {
        "baseline": "#2b6cb0",
        "oracle_mem": "#dd6b20",
        "comp": "#1f4e79",
        "mem": "#c53030",
        "unknown": "#718096",
    }

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax = axes[0]
    for policy in ("baseline", "oracle_mem"):
        xs = [row["n_server"] for row in policy_rows[policy]]
        ys = [row["reject_total"] for row in policy_rows[policy]]
        ax.plot(xs, ys, marker="o", linewidth=2, label=policy, color=colors[policy])
    ax.set_ylabel("Rejected Requests")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    ax = axes[1]
    x = np.arange(len(server_counts), dtype=float)
    width = 0.34
    positions = {
        "baseline": x - width / 2,
        "oracle_mem": x + width / 2,
    }
    for policy in ("baseline", "oracle_mem"):
        rows_for_policy = policy_rows[policy]
        comp = np.array([row["reject_comp"] for row in rows_for_policy], dtype=float)
        mem = np.array([row["reject_mem"] for row in rows_for_policy], dtype=float)
        unknown = np.array([row["reject_unknown"] for row in rows_for_policy], dtype=float)
        ax.bar(
            positions[policy],
            comp,
            width=width,
            color=colors["comp"] if policy == "baseline" else colors[policy],
            alpha=0.9,
            label=f"{policy} compute",
        )
        ax.bar(
            positions[policy],
            mem,
            width=width,
            bottom=comp,
            color=colors["mem"],
            alpha=0.8,
            label=f"{policy} memory",
        )
        if np.any(unknown):
            ax.bar(
                positions[policy],
                unknown,
                width=width,
                bottom=comp + mem,
                color=colors["unknown"],
                alpha=0.8,
                label=f"{policy} other",
            )
    ax.set_xticks(x)
    ax.set_xticklabels([str(count) for count in server_counts])
    ax.set_xlabel("# Devices")
    ax.set_ylabel("Rejected Requests")
    ax.set_title("Request Rejection Breakdown")
    ax.grid(True, axis="y", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    ax.legend(unique_handles, unique_labels, ncol=2)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot request rejections vs device count.")
    parser.add_argument("--arrival-pattern", default="azure_chat_23")
    parser.add_argument("--length-pattern", default="reasoning")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--slo-ttft-scale", type=float, default=5.0)
    parser.add_argument("--slo-tpot", type=float, default=0.1)
    parser.add_argument("--server-counts", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--output-name", default="rejection_scaling")
    parser.add_argument("--enforce-batch-memory-budget", action="store_true")
    args = parser.parse_args()

    out_dir = "headroom_outputs"
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{args.output_name}.csv")
    fig_path = os.path.join(fig_dir, f"{args.output_name}.png")

    rows = run_scaling(
        arrival_pattern=args.arrival_pattern,
        length_pattern=args.length_pattern,
        model_name=args.model_name,
        slo_ttft_scale=args.slo_ttft_scale,
        slo_tpot=args.slo_tpot,
        server_counts=sorted(set(args.server_counts)),
        enforce_batch_memory_budget=args.enforce_batch_memory_budget,
    )
    write_csv(rows, csv_path)
    plot_rows(
        rows,
        fig_path,
        title=(
            f"{args.arrival_pattern}:{args.length_pattern} "
            f"(recheck={'on' if args.enforce_batch_memory_budget else 'off'})"
        ),
    )
    print(f"Saved CSV to {csv_path}")
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
