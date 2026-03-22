# Experiment Plan

## Goal

Run the current `configs/batch` suite on a cluster with:

- 4 nodes
- 8 A100 GPUs per node
- 32 GPUs total

The current batch configs do not fully use a 32-GPU cluster as a single pool. The best practical plan is:

- use two 16-GPU multi-node runs first for the configs that actually need them
- then split back into single-node 8-GPU runs for the remaining configs


## Current Runtime Baseline

Measured from:

```bash
./.venv/bin/python -m SLOsServe.batch_runtime_report --config-dir configs/batch
```

Current serial runtime by config:

| Config | Serial Hours | Notes |
| --- | ---: | --- |
| `e2e.json` | 10.667 | single-node, 8-GPU class |
| `e2e_multinode.json` | 6.667 | requires 16 logical clients |
| `e2e_30B.jsonl` | 5.333 | TP=2, up to 8 replicas, needs 16 GPUs |
| `reasoning.json` | 2.500 | single-node |
| `sensitivity_slo.json` | 2.500 | single-node |
| `energy.json` | 2.000 | single-node |
| `sensitivity_perf_model.json` | 2.000 | single-node |
| `sensitivity_slo_tpot.json` | 2.000 | single-node |
| `ablation_cluster.json` | 1.000 | single-node |
| `multiround_chat.json` | 0.833 | single-node |
| `sensitivity_n_device.json` | 0.667 | single-node |
| **Total** | **36.167** | fully serial |

Target wall-clock with the schedule below: about `16` hours, plus setup and rerun buffer.

Recommended reservation: `18` hours.


## Preflight

Run these before starting experiments:

```bash
./.venv/bin/python -m SLOsServe.batch_sanity_check --config-dir configs/batch
./.venv/bin/python -m SLOsServe.batch_runtime_report --config-dir configs/batch
```

Current sanity status is acceptable. The only warning is in `e2e_30B.jsonl`: benchmark `extra_args` overrides `tensor_parallel_size` from `1` to `2`. That is intentional for now and not blocking.

On every node:

```bash
cd /path/to/SLOServe
conda activate sloserve
export PYTHONPATH=$PWD
```

If there is no shared filesystem, sync the repo from a control node:

```bash
rsync -ah --delete \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude 'experiments*' \
  /path/to/SLOServe/ \
  user@worker:/path/to/SLOServe/
```


## Resource Layout

Use the nodes like this:

- `node0 + node1`: Cluster A, 16 GPUs
- `node2 + node3`: Cluster B, 16 GPUs

Recommended role assignment:

- `node0`: Ray head for Cluster A
- `node2`: Ray head for Cluster B

Use different Ray ports so the clusters are independent:

- Cluster A: `6379`
- Cluster B: `6380`


## Phase 0: Start Ray

Cluster A:

```bash
# node0
HEAD_A=<node0-ip>
ray stop -f || true
ray start --head --node-ip-address "$HEAD_A" --port 6379
```

```bash
# node1
HEAD_A=<node0-ip>
MY_IP=<node1-ip>
ray stop -f || true
ray start --address "$HEAD_A:6379" --node-ip-address "$MY_IP"
```

Cluster B:

```bash
# node2
HEAD_B=<node2-ip>
ray stop -f || true
ray start --head --node-ip-address "$HEAD_B" --port 6380
```

```bash
# node3
HEAD_B=<node2-ip>
MY_IP=<node3-ip>
ray stop -f || true
ray start --address "$HEAD_B:6380" --node-ip-address "$MY_IP"
```


## Phase 1: Run the Two 16-GPU Jobs First

### 1A. `e2e_multinode.json` on Cluster A

This config is already wired for `server_clients=0-15`, so it is the natural 16-replica TP1 run.

Use:

```bash
# node0
HEAD_A=<node0-ip>
SERVER_EXTRA_ARGS_SHELL="--ray_address ${HEAD_A}:6379" \
RUN_BATCH_PORT=8000 \
./run_batch.sh configs/batch/e2e_multinode.json
```

Estimated runtime: `6.667` hours.


### 1B. `e2e_30B.jsonl` on Cluster B

This config already sets TP=2 in `extra_server_args`, but it also needs `--ray_address` to place actors across both nodes.

Recommended approach: create a temporary launch copy with `--ray_address` appended.

```bash
# node2
HEAD_B=<node2-ip>
python - <<'PY'
from pathlib import Path
src = Path("configs/batch/e2e_30B.jsonl")
dst = Path("/tmp/e2e_30B_clusterB.jsonl")
text = src.read_text()
needle = '"extra_server_args": "'
insert = '--ray_address ' + __import__("os").environ["HEAD_B"] + ':6380 '
text = text.replace(needle, needle + insert, 1)
dst.write_text(text)
print(dst)
PY

RUN_BATCH_PORT=8001 \
./run_batch.sh /tmp/e2e_30B_clusterB.jsonl
```

Estimated runtime: `5.333` hours.


## Phase 2: Split Back to Single-Node Runs

When `e2e_30B.jsonl` finishes, stop Cluster B and reuse `node2` and `node3` as two independent 8-GPU workers.

```bash
# on node2 and node3
ray stop -f || true
```

Do the same for Cluster A when `e2e_multinode.json` finishes, then reuse `node0` and `node1` independently.

For single-node jobs, no `--ray_address` is needed. `run_batch.sh` will start a local Ray runtime on that node.


## Recommended Schedule

### Stage 1

Start at time `T=0`:

- `node0+node1`: `e2e_multinode.json` (`6.667h`)
- `node2+node3`: `e2e_30B.jsonl` (`5.333h`)

### Stage 2

At about `T=5.333h`, `node2` and `node3` become free.

Run:

- `node2`: `e2e.json` (`10.667h`)
- `node3`: `reasoning.json` (`2.500h`) -> `sensitivity_slo.json` (`2.500h`) -> `ablation_cluster.json` (`1.000h`) -> `sensitivity_n_device.json` (`0.667h`)

### Stage 3

At about `T=6.667h`, `node0` and `node1` become free.

Run:

- `node0`: `energy.json` (`2.000h`) -> `sensitivity_perf_model.json` (`2.000h`) -> `multiround_chat.json` (`0.833h`)
- `node1`: `sensitivity_slo_tpot.json` (`2.000h`)


## Expected Completion Time

Critical path:

- `e2e_30B.jsonl`: `5.333h`
- then `e2e.json` on one freed node: `10.667h`

Total expected wall-clock:

- about `16.0h` from first launch to last completion

Practical reservation:

- `16h` if everything is stable
- `18h` if you want setup time, failures, and reruns absorbed


## Why This Schedule

This schedule is recommended because:

- `e2e_multinode.json` actually uses a 16-client setup and benefits from a 2-node Ray cluster
- `e2e_30B.jsonl` needs 16 GPUs because it runs up to 8 replicas at TP=2
- most remaining configs are only 8-GPU class jobs, so after the two large jobs finish, the best move is to split into single-node runs
- `e2e.json` is the longest remaining 8-GPU job, so it should start immediately when the first 2-node cluster frees up


## Operational Notes

- Use distinct `RUN_BATCH_PORT` values for concurrent jobs.
- Keep one `run_batch.sh` process per node or per Ray cluster.
- Do not run two batch jobs against the same Ray cluster at once.
- Capture logs per node with `tee` or `tmux`.
- Re-run sanity check after config edits:

```bash
./.venv/bin/python -m SLOsServe.batch_sanity_check --config-dir configs/batch
```


## Optional Improvements

These are useful but not required for the first pass:

- clean up `e2e_30B.jsonl` by setting `"tensor_parallel_size": 2` explicitly and removing the TP override from `extra_args`
- add `extra_server_args` with `--ray_address` directly to multi-node configs so temporary copies are unnecessary
- add a thin launcher script that starts Ray, injects `--ray_address`, and runs the schedule automatically
