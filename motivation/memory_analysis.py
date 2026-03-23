import json
import os
from typing import Any

import matplotlib.pyplot as plt


def _safe_name(text: str) -> str:
    safe = str(text).replace(" ", "_").replace("(", "").replace(")", "")
    safe = safe.replace("γ", "theta").replace(":", "_")
    return safe


def _to_int_request_id(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 10**18


def _classify_finish_reason(finish_reason: str | None) -> str:
    reason = str(finish_reason or "")
    if "reject" in reason:
        return "reject"
    if reason == "length":
        return "success"
    return "fail"


def _build_finish_records(events: list[Any]) -> list[tuple[float, str]]:
    records = []
    for event in events:
        if getattr(event, "event_type", None) != "finish":
            continue
        records.append((
            float(getattr(event, "timestamp", 0.0)),
            _classify_finish_reason(getattr(event, "finish_reason", None)),
        ))
    records.sort(key=lambda x: x[0])
    return records


def _write_connectivity_log(path: str, admission_history: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in admission_history:
            if row.get("memory_check_called", False):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def analyze_memory_run(
    *,
    store_prefix: str,
    policy_name: str,
    admission_history: list[dict[str, Any]],
    reqs: dict[str, Any],
    events: list[Any],
) -> dict[str, Any]:
    run_dir = os.path.join(
        "memory_analysis",
        os.path.normpath(store_prefix).replace(":", "_"),
    )
    os.makedirs(run_dir, exist_ok=True)

    connectivity_log = os.path.join(run_dir, "memory_check_connectivity.jsonl")
    _write_connectivity_log(connectivity_log, admission_history)

    finish_records = _build_finish_records(events)
    finish_idx = 0
    success_so_far = 0
    fail_so_far = 0
    accept_so_far = 0
    reject_so_far = 0

    sorted_history = sorted(
        admission_history,
        key=lambda row: (
            _to_int_request_id(row.get("request_id")),
            float(row.get("timestamp", 0.0)),
        ),
    )

    debug_rows = []
    for row in sorted_history:
        decision_ts = float(row.get("timestamp", 0.0))
        while finish_idx < len(finish_records) and finish_records[finish_idx][0] <= decision_ts:
            _, outcome = finish_records[finish_idx]
            if outcome == "success":
                success_so_far += 1
            elif outcome == "fail":
                fail_so_far += 1
            finish_idx += 1

        is_rejected = bool(row.get("is_rejected", False))
        if is_rejected:
            reject_so_far += 1
        else:
            accept_so_far += 1

        debug_rows.append({
            "request_id": row.get("request_id"),
            "active_request": int(row.get("active_request", -1)),
            "OOM_rate": float(row.get("oom_rate", -1.0)),
            "memory_occupy": float(row.get("memory_occupy", 0.0)),
            "success_request": success_so_far,
            "fail_request": fail_so_far,
            "accept_request": accept_so_far,
            "reject_request": reject_so_far,
            "timestamp": decision_ts,
            "memory_check_called": bool(row.get("memory_check_called", False)),
            "memory_check_policy": row.get("memory_check_policy", policy_name),
            "decision_reason": row.get("decision_reason", ""),
        })

    debug_tsv = os.path.join(run_dir, "debug_memory_trace.tsv")
    with open(debug_tsv, "w", encoding="utf-8") as f:
        headers = [
            "request_id",
            "active_request",
            "OOM_rate",
            "memory_occupy",
            "success_request",
            "fail_request",
            "accept_request",
            "reject_request",
        ]
        f.write("\t".join(headers) + "\n")
        for row in debug_rows:
            f.write("\t".join(str(row[h]) for h in headers) + "\n")

    debug_rows_by_req = sorted(debug_rows, key=lambda row: _to_int_request_id(row["request_id"]))
    request_ids = [_to_int_request_id(row["request_id"]) for row in debug_rows_by_req]
    memory_occupies = [float(row["memory_occupy"]) for row in debug_rows_by_req]
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(request_ids, memory_occupies, marker="o", markersize=2, linewidth=1, alpha=0.6)
    ax.set_xlabel("Request ID")
    ax.set_ylabel("Memory Occupy (ratio)")
    ax.set_title(f"Memory Occupy vs Request ID - {policy_name}")
    ymax = max([1.0] + memory_occupies)
    ax.set_ylim(0.0, max(1.0, ymax * 1.1))
    fig.tight_layout()
    debug_plot = os.path.join(run_dir, f"debug_memory_occupy_{_safe_name(policy_name)}.png")
    fig.savefig(debug_plot, dpi=300)
    plt.close(fig)

    total_requests = len(reqs)
    outcomes = {"success": 0, "fail": 0, "reject": 0}
    for req in reqs.values():
        outcome = _classify_finish_reason(getattr(req, "finish_reason", None))
        outcomes[outcome] += 1
    accept_count = total_requests - outcomes["reject"]
    summary = {
        "total_requests": total_requests,
        "reject_count": outcomes["reject"],
        "accept_count": accept_count,
        "fail_count": outcomes["fail"],
        "success_count": outcomes["success"],
        "reject_rate": outcomes["reject"] / max(total_requests, 1),
        "accept_rate": accept_count / max(total_requests, 1),
        "fail_rate": outcomes["fail"] / max(total_requests, 1),
        "success_rate": outcomes["success"] / max(total_requests, 1),
        "connectivity_log": connectivity_log,
        "debug_tsv": debug_tsv,
        "debug_plot": debug_plot,
    }

    summary_path = os.path.join(run_dir, "memory_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["summary_path"] = summary_path
    return summary


def analyze_memory_sweep(
    *,
    experiment_dir: str,
    results: list[dict[str, Any]],
) -> None:
    if not results:
        return
    out_dir = os.path.join(
        "memory_analysis",
        os.path.normpath(experiment_dir).replace(":", "_"),
    )
    os.makedirs(out_dir, exist_ok=True)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in results:
        key = (
            row.get("n_device"),
            row.get("ttft_slo_scale"),
            row.get("slo_tpot"),
            row.get("perf_model_err"),
        )
        grouped.setdefault(key, []).append(row)

    for key, rows in grouped.items():
        lines: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            label = (
                f"{row.get('routing_policy')}:{row.get('scheduling_policy')}:"
                f"{row.get('memory_check_policy', 'mc')}"
            )
            lines.setdefault(label, []).append(row)

        suffix = _safe_name(
            f"n{key[0]}_ttft{key[1]}_tpot{key[2]}_err{key[3]}"
        )

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        for label, line_rows in sorted(lines.items()):
            line_rows = sorted(line_rows, key=lambda x: float(x.get("rps", 0.0)))
            ax.plot(
                [float(row.get("rps", 0.0)) for row in line_rows],
                [float(row.get("success_rate", 0.0)) for row in line_rows],
                marker="o",
                label=label,
            )
        ax.set_xlabel("Arrival rate (requests/s)")
        ax.set_ylabel("Success rate")
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=6)
        fig.tight_layout()
        success_with_suffix = os.path.join(out_dir, f"arrival_rate_vs_success_rate_{suffix}.png")
        fig.savefig(success_with_suffix, dpi=300)
        if len(grouped) == 1:
            fig.savefig(os.path.join(out_dir, "arrival_rate_vs_success_rate.png"), dpi=300)
        plt.close(fig)

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        for label, line_rows in sorted(lines.items()):
            line_rows = sorted(line_rows, key=lambda x: float(x.get("rps", 0.0)))
            ax.plot(
                [float(row.get("rps", 0.0)) for row in line_rows],
                [float(row.get("fail_rate", 0.0)) for row in line_rows],
                marker="o",
                label=label,
            )
        ax.set_xlabel("Arrival rate (requests/s)")
        ax.set_ylabel("Fail rate")
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=6)
        fig.tight_layout()
        fail_with_suffix = os.path.join(out_dir, f"arrival_rate_vs_fail_rate_{suffix}.png")
        fig.savefig(fail_with_suffix, dpi=300)
        if len(grouped) == 1:
            fig.savefig(os.path.join(out_dir, "arrival_rate_vs_fail_rate.png"), dpi=300)
        plt.close(fig)

        last_rows = []
        for label, line_rows in sorted(lines.items()):
            line_rows = sorted(line_rows, key=lambda x: float(x.get("rps", 0.0)))
            last_rows.append((label, line_rows[-1]))
        metrics_path = os.path.join(out_dir, f"metrics_{suffix}.txt")
        total_requests = int(last_rows[0][1].get("total_requests", 0)) if last_rows else 0
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(f"total_requests={total_requests}\n")
            f.write("policy\treject_rate\taccept_rate\tfail_rate\tsuccess_rate\n")
            for label, row in last_rows:
                f.write(
                    f"{label}\t"
                    f"{float(row.get('reject_rate', 0.0))}\t"
                    f"{float(row.get('accept_rate', 0.0))}\t"
                    f"{float(row.get('fail_rate', 0.0))}\t"
                    f"{float(row.get('success_rate', 0.0))}\n"
                )
        if len(grouped) == 1:
            with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
                f.write(f"total_requests={total_requests}\n")
                f.write("policy\treject_rate\taccept_rate\tfail_rate\tsuccess_rate\n")
                for label, row in last_rows:
                    f.write(
                        f"{label}\t"
                        f"{float(row.get('reject_rate', 0.0))}\t"
                        f"{float(row.get('accept_rate', 0.0))}\t"
                        f"{float(row.get('fail_rate', 0.0))}\t"
                        f"{float(row.get('success_rate', 0.0))}\n"
                    )
