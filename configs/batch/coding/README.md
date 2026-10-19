# Single-Server Batch Templates

This directory contains a minimal 8-way sharded template for running one
policy on one logical replica per shard with `run_batch.sh`.

Files:
- `base_single_policy.json`: shared settings for the sweep.
- `gpu0.json` ... `gpu7.json`: example non-overlapping shards of the grid.
- `launch_8way.sh`: launcher template with one visible GPU, one port, and one
  log directory per shard.

How to edit:
- Change `policies` in `base_single_policy.json` to the full
  `routing_policy:scheduling_policy` string you want to test.
- Change `traces`, `window`, and `n_devices` in `base_single_policy.json`.
- Keep `n_devices: [1]` if each shard should consume one physical GPU.
- Repartition the grid by editing `load_scales`, `ttft_slo_scales`, and
  `slo_tpots` in the `gpu*.json` files.
- Keep `output_dir` unique per shard so results do not collide.

Current example split:
- `gpu0` ... `gpu3`: `load_scale=0.5` and TTFT SLOs `2, 3, 5, 10`
- `gpu4` ... `gpu7`: `load_scale=1.0` and TTFT SLOs `2, 3, 5, 10`
- every shard sweeps `slo_tpot` over `0.025, 0.05`

Launch:

```bash
bash configs/batch/single_server/launch_8way.sh
```

Validate a shard before launching:

```bash
./.venv/bin/python -m SLOsServe.batch_config \
  configs/batch/single_server/gpu0.json
```
