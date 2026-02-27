#!/usr/bin/env python3
"""Generate a 200-batch seed grid for performance-model regression.

Model target:
    max_i (a_i * query + b_i * past + c_i * query * past + d_i * n_req + e_i)

The seed set is intentionally structured to cover:
1) single-request (query, past) grid points,
2) batch-size sweeps (n_req effect),
3) mixed prefill/decode competition (max-over-requests behavior),
4) heterogeneous decode/prefill packs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def infer_batch_type(requests: list[dict[str, int]]) -> str:
    has_prefill = any(r["query_len"] > 1 for r in requests)
    has_decode = any(r["query_len"] == 1 for r in requests)
    if has_prefill and not has_decode:
        return "prefill"
    if has_decode and not has_prefill:
        return "decode"
    return "mixed"


def canonical_signature(requests: list[dict[str, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((r["context_len"], r["query_len"]) for r in requests))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a structured 200-batch seed set for regression."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "regression_seed_batches_200.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=200,
        help="Number of seed batches to generate.",
    )
    args = parser.parse_args()

    batches: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[tuple[int, int], ...]]] = set()

    def add(requests: list[dict[str, int]]) -> None:
        if len(batches) >= args.target:
            return
        reqs = [
            {
                "context_len": int(max(0, r["context_len"])),
                "query_len": int(max(0, r["query_len"])),
            }
            for r in requests
            if r["context_len"] > 0 or r["query_len"] > 0
        ]
        if not reqs:
            return
        batch_type = infer_batch_type(reqs)
        sig = (batch_type, canonical_signature(reqs))
        if sig in seen:
            return
        seen.add(sig)
        name = f"{batch_type}_{len([b for b in batches if b['type'] == batch_type]) + 1:03d}"
        batches.append({"name": name, "type": batch_type, "requests": reqs})

    # 1) Single-request grid (query, past).
    context_levels = [0, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    query_levels = [1, 16, 32, 64, 128, 256, 512, 1024, 2048]
    for p in context_levels:
        for q in query_levels:
            add([{"context_len": p, "query_len": q}])

    # 2) Batch-size sweeps for key archetypes.
    archetypes = [(0, 256), (0, 1024), (512, 1), (2048, 1), (4096, 1), (2048, 64)]
    n_levels = [1, 2, 4, 8, 16, 32]
    for p, q in archetypes:
        for n in n_levels:
            add([{"context_len": p, "query_len": q} for _ in range(n)])

    # 3) Mixed 2-request competition points.
    prefills = [(0, 256), (0, 1024), (512, 256), (1024, 128), (2048, 64)]
    decodes = [(128, 1), (512, 1), (2048, 1), (4096, 1), (8192, 1)]
    for p_ctx, p_q in prefills:
        for d_ctx, d_q in decodes:
            add(
                [
                    {"context_len": p_ctx, "query_len": p_q},
                    {"context_len": d_ctx, "query_len": d_q},
                ]
            )

    # 4) Mixed with variable decode fan-in around one prefill.
    for p_ctx, p_q in prefills[:3]:
        for d_ctx, d_q in decodes[:3]:
            for n in [2, 4, 8, 16]:
                add(
                    [{"context_len": p_ctx, "query_len": p_q}]
                    + [{"context_len": d_ctx, "query_len": d_q} for _ in range(n)]
                )

    # 5) Heterogeneous decode packs.
    decode_packs = [
        [128, 256],
        [128, 512],
        [256, 512, 1024],
        [512, 1024, 2048],
        [1024, 2048, 4096],
        [2048, 4096, 8192],
        [256, 512, 1024, 2048],
        [512, 1024, 2048, 4096],
        [128, 256, 512, 1024, 2048],
    ]
    for ctxs in decode_packs:
        add([{"context_len": p, "query_len": 1} for p in ctxs])
        add([{"context_len": p, "query_len": 1} for p in (ctxs + ctxs[:1])])

    # 6) Heterogeneous prefill packs.
    prefill_query_sets = [
        [16, 32],
        [32, 64, 128],
        [64, 128, 256],
        [128, 256, 512],
        [256, 512, 1024],
    ]
    prefill_ctx_bases = [0, 512, 2048]
    for base in prefill_ctx_bases:
        for qs in prefill_query_sets:
            add([{"context_len": base, "query_len": q} for q in qs])

    # 7) Fill remainder with systematic mixed stress points.
    for q in [32, 64, 128, 256, 512, 1024, 2048]:
        for p in [128, 512, 2048, 8192]:
            for n in [2, 4, 8]:
                add(
                    [{"context_len": 0, "query_len": q}]
                    + [{"context_len": p, "query_len": 1} for _ in range(n)]
                )
                if len(batches) >= args.target:
                    break
            if len(batches) >= args.target:
                break
        if len(batches) >= args.target:
            break

    if len(batches) < args.target:
        raise SystemExit(
            f"Could only generate {len(batches)} unique seeds (target={args.target})."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(batches[: args.target], f, indent=2)
        f.write("\n")

    type_counts = {"prefill": 0, "decode": 0, "mixed": 0}
    for b in batches[: args.target]:
        type_counts[b["type"]] += 1
    print(f"Wrote {args.target} seed batches to {args.output}")
    print(type_counts)


if __name__ == "__main__":
    main()

