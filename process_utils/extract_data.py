from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping

def extract_ttft_tpots(experiment_dir: str | Path) -> Dict[str, Mapping[str, List[float] | float]]:
    """
    Load per-request TTFT and TPOT lists from a Sarathi round-robin experiment.

    The function searches the given experiment directory for a ``*.reqs.jsonl`` file
    (these files are stored as a single JSON array) and builds a dictionary of
    ``req_id -> {"ttft": float, "tpots": [float, ...]}``.
    """
    exp_path = Path(experiment_dir)
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {exp_path}")

    req_files = list(exp_path.glob("*.reqs.jsonl"))
    if not req_files:
        raise FileNotFoundError(f"No *.reqs.jsonl file found in {exp_path}")

    req_file = req_files[0]
    requests = json.loads(req_file.read_text())

    results: Dict[str, Mapping[str, List[float] | float]] = {}
    for req in requests:
        timestamps: List[float] = req.get("timestamps", [])
        if not timestamps:
            # Skip malformed entries with no timing info.
            continue

        arrival_time = float(req.get("arrival_time", 0.0))
        ttft = timestamps[0] - arrival_time
        tpots = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        results[str(req.get("req_id"))] = {"ttft": ttft, "tpots": tpots}

    return results
