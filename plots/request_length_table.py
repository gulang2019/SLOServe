from __future__ import annotations

import argparse
import contextlib
import io
import math
import statistics
import sys
import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))


def _install_import_stubs() -> None:
    """Allow Dataset.dataset to load even in lightweight environments.

    The request pickles only need the dataclass definitions from Dataset.dataset.
    Some shells in this repo do not have optional analysis dependencies such as
    python-dotenv, numpy, or pandas installed, so stub them if they are absent.
    """

    for module_name in ("dotenv", "numpy", "pandas"):
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            sys.modules[module_name] = types.ModuleType(module_name)


_install_import_stubs()

with contextlib.redirect_stdout(io.StringIO()):
    from Dataset.dataset import Requests


DEFAULT_DATASETS = [
    "sharegpt",
    "azure_chat_23",
    "azure_code_23",
    "azure_chat",
]

DATASET_ALIASES = {
    "sharegpt": "sharegpt_chat",
}

DEFAULT_CAPTION = (
    "Request length statistics for the evaluation traces. Token statistics are "
    "reported as mean / std / p99. For ShareGPT, we additionally report the "
    "number of chats and the distribution of rounds per chat."
)
DEFAULT_LABEL = "tab:request_length_stats"


@dataclass(frozen=True)
class NumericStats:
    mean: float
    std: float
    p99: int


@dataclass(frozen=True)
class DatasetSummary:
    display_name: str
    dataset_name: str
    num_requests: int
    prompt_tokens: NumericStats
    generation_tokens: NumericStats
    num_chats: int | None = None
    rounds_per_chat: NumericStats | None = None


def _resolve_dataset_name(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def _percentile_nearest_rank(values: list[int], percentile: float) -> int:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty list.")
    ordered = sorted(values)
    index = max(0, math.ceil((percentile / 100.0) * len(ordered)) - 1)
    return int(ordered[index])


def _summarize(values: list[int]) -> NumericStats:
    if not values:
        raise ValueError("Cannot summarize an empty list.")
    return NumericStats(
        mean=statistics.fmean(values),
        std=statistics.pstdev(values),
        p99=_percentile_nearest_rank(values, 99),
    )


def _load_summary(display_name: str) -> DatasetSummary:
    dataset_name = _resolve_dataset_name(display_name)
    with contextlib.redirect_stdout(io.StringIO()):
        requests = Requests.load(dataset_name).requests
    prompt_lengths = [int(req.input_length) for req in requests]
    generation_lengths = [int(req.output_length) for req in requests]

    session_counts = Counter(
        req.session_id for req in requests if getattr(req, "session_id", None)
    )
    if session_counts:
        rounds_per_chat = _summarize(list(session_counts.values()))
        num_chats = len(session_counts)
    else:
        rounds_per_chat = None
        num_chats = None

    return DatasetSummary(
        display_name=display_name,
        dataset_name=dataset_name,
        num_requests=len(requests),
        prompt_tokens=_summarize(prompt_lengths),
        generation_tokens=_summarize(generation_lengths),
        num_chats=num_chats,
        rounds_per_chat=rounds_per_chat,
    )


def _format_int(value: int) -> str:
    return f"{int(value):,}"


def _format_token_stats(stats: NumericStats) -> str:
    return (
        f"{_format_int(round(stats.mean))} / "
        f"{_format_int(round(stats.std))} / "
        f"{_format_int(stats.p99)}"
    )


def _format_round_stats(stats: NumericStats) -> str:
    return f"{stats.mean:.1f} / {stats.std:.1f} / {stats.p99}"


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def build_latex_table(
    summaries: list[DatasetSummary],
    *,
    caption: str = DEFAULT_CAPTION,
    label: str = DEFAULT_LABEL,
    tabular_only: bool = False,
) -> str:
    lines = []
    if not tabular_only:
        lines.extend(
            [
                r"\begin{table*}[t]",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                r"\centering",
                r"\small",
                r"\setlength{\tabcolsep}{4pt}",
                r"\resizebox{\textwidth}{!}{%",
            ]
        )

    lines.extend(
        [
            r"\begin{tabular}{@{}lrrrrr@{}}",
            r"\toprule",
            (
                r"Dataset & Requests & Prompt Tokens (mean / std / p99) & "
                r"Generation Tokens (mean / std / p99) & Chats & "
                r"Rounds / Chat (mean / std / p99) \\"
            ),
            r"\midrule",
        ]
    )

    for summary in summaries:
        num_chats = "--" if summary.num_chats is None else _format_int(summary.num_chats)
        rounds = (
            "--"
            if summary.rounds_per_chat is None
            else _format_round_stats(summary.rounds_per_chat)
        )
        lines.append(
            " & ".join(
                [
                    _escape_latex(summary.display_name),
                    _format_int(summary.num_requests),
                    _format_token_stats(summary.prompt_tokens),
                    _format_token_stats(summary.generation_tokens),
                    num_chats,
                    rounds,
                ]
            )
            + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )

    if not tabular_only:
        lines.extend(
            [
                r"}",
                r"\end{table*}",
            ]
        )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a LaTeX table with request-length statistics for the "
            "saved dataset pickles in assets/datasets/."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=(
            "Dataset names to include. 'sharegpt' is treated as an alias for "
            "'sharegpt_chat'."
        ),
    )
    parser.add_argument(
        "--caption",
        default=DEFAULT_CAPTION,
        help="Caption to use when emitting the full table environment.",
    )
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help="LaTeX label to use when emitting the full table environment.",
    )
    parser.add_argument(
        "--tabular-only",
        action="store_true",
        help="Emit only the tabular block, without the surrounding table env.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the generated LaTeX to.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [_load_summary(name) for name in args.datasets]
    latex = build_latex_table(
        summaries,
        caption=args.caption,
        label=args.label,
        tabular_only=args.tabular_only,
    )

    if args.output is not None:
        args.output.write_text(latex, encoding="utf-8")
    else:
        sys.stdout.write(latex)


if __name__ == "__main__":
    main()
