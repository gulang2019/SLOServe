#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plots.chat_7b_n_device_triptych import (  # noqa: E402
    DEFAULT_ASPLOS_PANEL_KEYS,
    draw_n_device_triptych,
    save_shared_legend,
)


WORKLOAD_STEMS = (
    "chat_7B",
    "code_7B",
    "mixed_7B",
    "32B_chat",
    "e2e_mulitiround_chatting",
    "chatting_large_scale",
)

LEGEND_METHOD_ORDER = (
    "vllm / round_robin",
    "qlm / llumnix_load",
    "sarathi / round_robin",
    "atfc / slosserve_planner",
    "vllm / round_robin_session",
)


def render_main_multi_replica_figure(
    *,
    data_dir: Path,
    output_dir: Path,
    content_scale: float = 1.0,
) -> list[Path]:
    saved_paths: list[Path] = []

    for stem in WORKLOAD_STEMS:
        saved_paths.extend(
            draw_n_device_triptych(
                csv_path=data_dir / f"{stem}.csv",
                output_stem=output_dir / f"{stem}_two_panel_slo_cap_10_logx",
                panel_keys=DEFAULT_ASPLOS_PANEL_KEYS,
                large_format=True,
                show_legend=False,
                show_panel_titles=False,
                show_energy_gradient_note=False,
                annotation_placement="below_ours",
                slo_violation_x_log_scale=True,
                content_scale=content_scale,
            )
        )

    saved_paths.extend(
        save_shared_legend(
            output_dir / "multi_replica_shared_legend",
            LEGEND_METHOD_ORDER,
            large_format=True,
            ncol=6,
            include_modeled_optimal=True,
            content_scale=content_scale,
        )
    )
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the paper's main multi-replica figure set "
            "(Figure 'distributed' plus shared legend)."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Paper/data/e2e"),
        help="Directory containing the e2e CSV inputs. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Paper/figs/e2e_new"),
        help="Directory for the exported figure assets. Default: %(default)s",
    )
    parser.add_argument(
        "--content-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply the internal plot and legend sizing for the regenerated "
            "figure set without changing the output layout. Default: %(default)s"
        ),
    )
    args = parser.parse_args()

    saved_paths = render_main_multi_replica_figure(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        content_scale=args.content_scale,
    )
    print(f"Rendered {len(saved_paths)} asset(s).")


if __name__ == "__main__":
    main()
