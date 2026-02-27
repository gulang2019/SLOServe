#!/usr/bin/env python3
"""Export batch datasets from trace event files.

This script scans trace files (e.g. experiments_emulation_new/**/*events.jsonl),
extracts `event_type == "batch"` events, and writes:
1) all extracted batches in a normalized dataset format
2) a typical subset (most frequent unique batch signatures per type)
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


BatchRequest = dict[str, int]
BatchRecord = dict[str, Any]
Signature = tuple[tuple[int, int], ...]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Export full and typical batch datasets from trace events."
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="experiments_emulation_new/**/*events.jsonl",
        help="Glob pattern for input event files (recursive glob).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "exported_batches.json",
        help="Path for full exported batches JSON.",
    )
    parser.add_argument(
        "--typical-output",
        type=Path,
        default=here / "typical_batches.json",
        help="Path for typical subset JSON.",
    )
    parser.add_argument(
        "--top-k-per-type",
        type=int,
        default=20,
        help=(
            "Number of typical signatures to keep per type: prefill/decode/mixed. "
            "Use 0 or negative to keep all signatures above --min-frequency."
        ),
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=1,
        help="Minimum count for a signature to be included in the typical subset.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="Optional cap on number of files to scan (0 means no cap).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Optional cap on exported batches (0 means no cap).",
    )
    parser.add_argument(
        "--include-counts",
        action="store_true",
        help="Include `count` in typical subset entries.",
    )
    parser.add_argument(
        "--skip-exported-output",
        action="store_true",
        help="Skip writing full exported batches JSON (faster when only refreshing typical subset).",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=0,
        help=(
            "If > 0, downsample typical subset to exactly this many batches "
            "(subject to available entries)."
        ),
    )
    parser.add_argument(
        "--target-split",
        type=str,
        default="prefill=1,decode=1,mixed=1",
        help=(
            "Type weights used with --target-total, format: "
            "prefill=0.3,decode=0.4,mixed=0.3"
        ),
    )
    parser.add_argument(
        "--typical-bucket-mode",
        type=str,
        choices=("pow2", "multiple", "exact"),
        default="pow2",
        help=(
            "Bucketing mode used only for typical-subset dedup/signature building. "
            "`pow2` groups lengths to next power-of-two, `multiple` groups to "
            "next multiple of --typical-bucket-size, `exact` keeps original lengths."
        ),
    )
    parser.add_argument(
        "--typical-bucket-size",
        type=int,
        default=128,
        help="Bucket size used when --typical-bucket-mode=multiple.",
    )
    return parser.parse_args()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_by_req_id(container: Any, req_id: Any, index: int, default: int = 0) -> int:
    if isinstance(container, dict):
        # req_id in traces is often a string.
        if req_id in container:
            return _safe_int(container[req_id], default)
        req_id_str = str(req_id)
        if req_id_str in container:
            return _safe_int(container[req_id_str], default)
        return default
    if isinstance(container, (list, tuple)):
        if 0 <= index < len(container):
            return _safe_int(container[index], default)
    return default


def infer_batch_type(requests: list[BatchRequest]) -> str:
    has_prefill = any(req["query_len"] > 1 for req in requests)
    has_decode = any(req["query_len"] == 1 for req in requests)
    if has_prefill and not has_decode:
        return "prefill"
    if has_decode and not has_prefill:
        return "decode"
    return "mixed"


def extract_batch_from_event(event: dict[str, Any]) -> tuple[list[BatchRequest], str] | None:
    if event.get("event_type") != "batch":
        return None
    req_ids = event.get("req_ids")
    if not isinstance(req_ids, list) or not req_ids:
        return None

    computed_tokens = event.get("num_computed_tokens", [])
    scheduled_tokens = event.get("num_scheduled_tokens", {})
    requests: list[BatchRequest] = []

    for idx, req_id in enumerate(req_ids):
        context_len = max(0, _resolve_by_req_id(computed_tokens, req_id, idx, default=0))
        query_len = max(0, _resolve_by_req_id(scheduled_tokens, req_id, idx, default=0))
        # Keep entries where at least one side is present.
        if context_len == 0 and query_len == 0:
            continue
        requests.append({"context_len": context_len, "query_len": query_len})

    if not requests:
        return None
    return requests, infer_batch_type(requests)


def iter_events_from_json_array(path: Path) -> Iterable[dict[str, Any]]:
    # Stream JSON array values to avoid loading very large traces into memory.
    decoder = json.JSONDecoder()
    buffer = ""
    in_array = False
    eof = False

    with path.open("r", encoding="utf-8") as f:
        while True:
            if not eof:
                chunk = f.read(1 << 20)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            idx = 0
            while True:
                while idx < len(buffer) and buffer[idx].isspace():
                    idx += 1

                if not in_array:
                    if idx >= len(buffer):
                        break
                    if buffer[idx] != "[":
                        raise json.JSONDecodeError(
                            "Expected JSON array start '['",
                            buffer,
                            idx,
                        )
                    in_array = True
                    idx += 1
                    continue

                while idx < len(buffer) and buffer[idx].isspace():
                    idx += 1
                if idx >= len(buffer):
                    break

                if buffer[idx] == ",":
                    idx += 1
                    continue
                if buffer[idx] == "]":
                    # End of array.
                    return

                try:
                    event, next_idx = decoder.raw_decode(buffer, idx)
                except json.JSONDecodeError:
                    # Need more bytes.
                    break

                idx = next_idx
                if isinstance(event, str):
                    try:
                        event = json.loads(event)
                    except json.JSONDecodeError:
                        continue
                if isinstance(event, dict):
                    yield event

            if idx > 0:
                buffer = buffer[idx:]

            if eof:
                break


def iter_events_from_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw in {"[", "]"}:
                continue
            if raw.endswith(","):
                raw = raw[:-1]
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(event, str):
                try:
                    event = json.loads(event)
                except json.JSONDecodeError:
                    continue
            if isinstance(event, dict):
                yield event


def iter_events(path: Path) -> Iterable[dict[str, Any]]:
    # Auto-detect: some *.jsonl files here are actually JSON arrays.
    first = ""
    with path.open("r", encoding="utf-8") as f:
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first = ch
                break

    if first == "[":
        yield from iter_events_from_json_array(path)
    else:
        yield from iter_events_from_jsonl(path)


def _bucket_length(value: int, mode: str, bucket_size: int) -> int:
    if value <= 1:
        # Keep 0/1 untouched so decode token=1 remains distinguishable.
        return value
    if mode == "exact":
        return value
    if mode == "multiple":
        if bucket_size <= 1:
            return value
        return ((value + bucket_size - 1) // bucket_size) * bucket_size
    # mode == "pow2"
    bucket = 1
    while bucket < value:
        bucket <<= 1
    return bucket


def batch_signature(
    requests: list[BatchRequest],
    bucket_mode: str,
    bucket_size: int,
) -> Signature:
    # Sort so signature is stable regardless of req_id ordering.
    pairs = sorted(
        (
            _bucket_length(req["context_len"], bucket_mode, bucket_size),
            _bucket_length(req["query_len"], bucket_mode, bucket_size),
        )
        for req in requests
    )
    return tuple(pairs)


def signature_to_requests(signature: Signature) -> list[BatchRequest]:
    return [{"context_len": c, "query_len": q} for c, q in signature]


def collect_batches(
    paths: list[Path],
    max_batches: int,
    typical_bucket_mode: str,
    typical_bucket_size: int,
    collect_exported: bool,
) -> tuple[list[BatchRecord], dict[str, Counter[Signature]], int]:
    exported: list[BatchRecord] = []
    counts_by_type: dict[str, Counter[Signature]] = defaultdict(Counter)
    file_count = 0

    stop = False
    for path in paths:
        file_count += 1
        try:
            event_iter = iter_events(path)
            for event in event_iter:
                extracted = extract_batch_from_event(event)
                if extracted is None:
                    continue
                requests, batch_type = extracted
                if collect_exported:
                    exported.append(
                        {
                            "name": f"batch_{len(exported) + 1:07d}",
                            "type": batch_type,
                            "requests": requests,
                        }
                    )
                counts_by_type[batch_type][
                    batch_signature(
                        requests,
                        bucket_mode=typical_bucket_mode,
                        bucket_size=typical_bucket_size,
                    )
                ] += 1
                if max_batches > 0 and len(exported) >= max_batches:
                    stop = True
                    break
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] Failed to parse {path}: {exc}")
        if stop:
            break

    return exported, counts_by_type, file_count


def build_typical_subset(
    counts_by_type: dict[str, Counter[Signature]],
    top_k_per_type: int,
    min_frequency: int,
    include_counts: bool,
) -> list[BatchRecord]:
    out: list[BatchRecord] = []
    for batch_type in ("prefill", "decode", "mixed"):
        ranked = [
            (signature, count)
            for signature, count in counts_by_type.get(batch_type, Counter()).items()
            if count >= min_frequency
        ]
        ranked.sort(key=lambda item: (-item[1], len(item[0]), item[0]))
        selected = ranked if top_k_per_type <= 0 else ranked[:top_k_per_type]
        for idx, (signature, count) in enumerate(selected, start=1):
            record: BatchRecord = {
                "name": f"{batch_type}_{idx:03d}",
                "type": batch_type,
                "requests": signature_to_requests(signature),
            }
            if include_counts:
                record["count"] = count
            out.append(record)
    return out


def parse_target_split(spec: str) -> dict[str, float]:
    weights: dict[str, float] = {"prefill": 1.0, "decode": 1.0, "mixed": 1.0}
    if not spec:
        return weights

    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in weights:
            continue
        try:
            parsed = float(value.strip())
        except ValueError:
            continue
        if parsed >= 0:
            weights[key] = parsed

    if sum(weights.values()) <= 0:
        return {"prefill": 1.0, "decode": 1.0, "mixed": 1.0}
    return weights


def spread_select(records: list[BatchRecord], k: int) -> list[BatchRecord]:
    n = len(records)
    if k <= 0 or n == 0:
        return []
    if k >= n:
        return list(records)
    if k == 1:
        return [records[0]]

    # Choose indices spread over ranking to preserve both popular and diverse shapes.
    raw_indices = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    chosen: list[int] = []
    used = set()
    for idx in raw_indices:
        if idx not in used:
            chosen.append(idx)
            used.add(idx)
    if len(chosen) < k:
        for idx in range(n):
            if idx in used:
                continue
            chosen.append(idx)
            used.add(idx)
            if len(chosen) >= k:
                break
    chosen.sort()
    return [records[idx] for idx in chosen[:k]]


def compute_type_quotas(
    target_total: int,
    available: dict[str, int],
    weights: dict[str, float],
) -> dict[str, int]:
    order = ("prefill", "decode", "mixed")
    quotas = {t: 0 for t in order}
    if target_total <= 0:
        return quotas

    total_weight = sum(max(0.0, weights.get(t, 0.0)) for t in order)
    if total_weight <= 0:
        total_weight = float(len(order))
        weights = {t: 1.0 for t in order}

    raw = {
        t: target_total * (max(0.0, weights.get(t, 0.0)) / total_weight)
        for t in order
    }
    for t in order:
        quotas[t] = min(available.get(t, 0), int(raw[t]))

    assigned = sum(quotas.values())
    # Distribute remaining by largest fractional parts, respecting availability.
    while assigned < target_total:
        candidates = []
        for t in order:
            if quotas[t] >= available.get(t, 0):
                continue
            frac = raw[t] - int(raw[t])
            candidates.append((frac, t))
        if not candidates:
            break
        candidates.sort(key=lambda x: (-x[0], x[1]))
        _, pick = candidates[0]
        quotas[pick] += 1
        assigned += 1

    # If still short due capacity constraints, fill from any type with room.
    while assigned < target_total:
        progressed = False
        for t in order:
            if quotas[t] < available.get(t, 0):
                quotas[t] += 1
                assigned += 1
                progressed = True
                if assigned >= target_total:
                    break
        if not progressed:
            break
    return quotas


def downsample_typical(
    records: list[BatchRecord],
    target_total: int,
    target_split: str,
) -> list[BatchRecord]:
    if target_total <= 0:
        return records

    grouped: dict[str, list[BatchRecord]] = {"prefill": [], "decode": [], "mixed": []}
    for rec in records:
        t = rec.get("type")
        if t in grouped:
            grouped[t].append(rec)

    weights = parse_target_split(target_split)
    quotas = compute_type_quotas(
        target_total=target_total,
        available={k: len(v) for k, v in grouped.items()},
        weights=weights,
    )

    out: list[BatchRecord] = []
    for t in ("prefill", "decode", "mixed"):
        out.extend(spread_select(grouped[t], quotas[t]))

    # Stable rename for readability.
    by_type_idx = {"prefill": 0, "decode": 0, "mixed": 0}
    for rec in out:
        t = rec["type"]
        by_type_idx[t] += 1
        rec["name"] = f"{t}_{by_type_idx[t]:03d}"
    return out


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in sorted(glob.glob(args.input_glob, recursive=True))]
    if args.limit_files > 0:
        paths = paths[: args.limit_files]

    if not paths:
        raise SystemExit(f"No files matched input glob: {args.input_glob}")

    exported, counts_by_type, scanned_files = collect_batches(
        paths=paths,
        max_batches=args.max_batches,
        typical_bucket_mode=args.typical_bucket_mode,
        typical_bucket_size=args.typical_bucket_size,
        collect_exported=not args.skip_exported_output,
    )
    typical = build_typical_subset(
        counts_by_type=counts_by_type,
        top_k_per_type=max(0, args.top_k_per_type),
        min_frequency=max(1, args.min_frequency),
        include_counts=args.include_counts,
    )
    typical = downsample_typical(
        records=typical,
        target_total=max(0, args.target_total),
        target_split=args.target_split,
    )

    if not args.skip_exported_output:
        write_json(args.output, exported)
    write_json(args.typical_output, typical)

    if args.skip_exported_output:
        print(
            f"Scanned {scanned_files} file(s). "
            f"Wrote typical subset:\n- {args.typical_output}"
        )
    else:
        print(
            f"Scanned {scanned_files} file(s), exported {len(exported)} batch(es). "
            f"Wrote:\n- {args.output}\n- {args.typical_output}"
        )
    for batch_type in ("prefill", "decode", "mixed"):
        n_unique = len(counts_by_type.get(batch_type, Counter()))
        print(f"{batch_type}: {n_unique} unique signatures")


if __name__ == "__main__":
    main()
