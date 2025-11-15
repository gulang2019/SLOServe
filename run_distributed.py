import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

baselines = [
    'round_robin:qlm',
    # 'round_robin:sarathi',
    # 'disaggregated:vllm',
    'disaggregated-edf:qlm',
    'round_robin:slosserve-edf',
    'renaming:slosserve-edf',
    'auto_scaling:slosserve-edf'
]

baselines = [
    'round_robin:vllm',
    'round_robin:sarathi',
    'round_robin:slosserve-edf',
    'auto_scaling:slosserve-edf'
    # 'disaggregated:vllm',
    # 'disaggregated:slosserve-edf',
]


jobs = {
    'Coder-Qwen7B': {
        'baseline': baselines,
        'n_devices': [5,6,7,8],
        'load_scale': 1.0,
        'slo_tpots': 0.025,
        'ttft_slo_scales': 3.0,
        'window': '3979:4579',
        'trace': 'sharegpt_code:azure_code_23',
        'model_name': 'Qwen/Qwen2.5-7B-Instruct',
    },
    'ChatBot-Qwen7B': {
        'baseline': baselines,
        'n_devices': [8,7,6,5],
        'load_scale': 4.0,
        'slo_tpots': 0.10,
        'ttft_slo_scales': 5.0,
        'window': '601:1202',
        'trace': 'azure_chat_23:azure_chat_23',
        'model_name': 'Qwen/Qwen2.5-7B-Instruct',
    },
    'Arxiv-Qwen7B': {
        'baseline': baselines,
        'n_devices': [1,2,4,8],
        'load_scale': 6,
        'slo_tpots': 0.10,
        'ttft_slo_scales': 1.5,
        'window': '400:600',
        'trace': 'arxiv_summary:burstgpt_GPT-4_Conversation log',
        'model_name': 'Qwen/Qwen2.5-7B-Instruct',
    },
    'Coder-Gemma27B': {
        'baseline': baselines,
        'n_devices': [4,3,2,1],
        'load_scale': 1.0,
        'slo_tpots': 0.05,
        'ttft_slo_scales': 3.0,
        'window': '3978:4579',
        'trace': 'sharegpt_code:azure_code_23',
        'model_name': 'google/gemma-3-27b-it',
    },
    'ChatBot-Gemma27B': {
        'baseline': baselines,
        'n_devices': [4,3,2,1],
        'load_scale': 4.0,
        'slo_tpots': 0.10,
        'ttft_slo_scales': 5.0,
        'window': '600:1202',
        'trace': 'azure_chat_23:azure_chat_23',
        'model_name': 'google/gemma-3-27b-it',
    },
    'Arxiv-Gemma27B': {
        'baseline': baselines,
        'n_devices': [1,2,4,8],
        'load_scale': 6,
        'slo_tpots': 0.05,
        'ttft_slo_scales': 1.5,
        'window': '400:800',
        'trace': 'arxiv_summary:burstgpt_GPT-4_Conversation log',
        'model_name': 'google/gemma-3-27b-it',
    },
}


def run_job(job_name: str):
    """
    Allocate GPUs and run the requested experiment grid (baseline x load_scale x n_device)
    using the distributed harness.

    run_distributed.sh <gpu_csv> <router_port> <n_device> <baseline> <load_scale>
                       [slo_tpots] [ttft_slo_scales] [window] [trace] [model_name]
    """
    import torch
    import json
    import time
    import subprocess
    from collections import deque, defaultdict
    from pathlib import Path
    from datetime import datetime, timedelta

    job_cfg = jobs.get(job_name)
    if job_cfg is None:
        raise ValueError(f'Unknown job "{job_name}". Available jobs: {list(jobs)}')

    def _ensure_list(value, *, split_commas: bool = False):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        if isinstance(value, str):
            if split_commas:
                items = [item.strip() for item in value.split(',') if item.strip()]
                return items or [value.strip()]
            return [value]
        return [value]

    def _to_int(value):
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return int(value)
        return int(str(value).strip())

    baselines = job_cfg.get('baseline', [])
    if isinstance(baselines, str):
        baselines = [baselines]
    elif isinstance(baselines, (set, tuple)):
        baselines = list(baselines)

    load_scales = _ensure_list(job_cfg.get('load_scale', []))
    if not baselines or not load_scales:
        raise ValueError(f'Job "{job_name}" must specify both "baseline" and "load_scale".')

    n_device_values_raw = job_cfg.get('n_devices', [1])
    n_device_values = _ensure_list(n_device_values_raw, split_commas=True) or [1]
    try:
        n_device_values = [_to_int(v) for v in n_device_values]
    except ValueError as exc:
        raise ValueError(f'Invalid n_devices specification {n_device_values_raw!r}: {exc}') from exc
    if any(n <= 0 for n in n_device_values):
        raise ValueError(f'n_devices must be positive integers, got {n_device_values}')

    def _normalize_gpu_pool(value):
        if value is None:
            return None
        if isinstance(value, int):
            return list(range(value))
        if isinstance(value, (list, tuple, set)):
            return sorted({_to_int(v) for v in value})
        if isinstance(value, str):
            tokens = [token.strip() for token in value.split(',') if token.strip()]
            return sorted({_to_int(token) for token in tokens})
        raise TypeError(f'Unsupported gpu_pool specification: {value!r}')

    gpu_pool = _normalize_gpu_pool(job_cfg.get('gpu_pool'))
    detected_gpus = torch.cuda.device_count()
    if gpu_pool is None:
        if detected_gpus <= 0:
            raise RuntimeError(
                'run_job requires at least one visible CUDA device; '
                'set "gpu_pool" in the job configuration to override detection.'
            )
        gpu_pool = list(range(detected_gpus))
    if not gpu_pool:
        raise RuntimeError('GPU pool is empty after applying configuration.')

    gpu_pool = sorted(gpu_pool)
    n_gpus = len(gpu_pool)
    max_requested = max(n_device_values)
    if max_requested > n_gpus:
        raise ValueError(
            f'Job "{job_name}" requests up to {max_requested} GPUs but only {n_gpus} are available: {gpu_pool}'
        )

    tasks_by_n = defaultdict(deque)
    total_runs = 0
    for n_device in n_device_values:
        for baseline in baselines:
            for load_scale in load_scales:
                tasks_by_n[int(n_device)].append((baseline, load_scale))
                total_runs += 1

    if total_runs == 0:
        logger.info('Job %s has no tasks to run', job_name)
        return []

    logger.info(
        'Starting job %s: %d baselines x %d load scales x %d n_device options (%d runs) on GPU pool %s',
        job_name,
        len(baselines),
        len(load_scales),
        len(n_device_values),
        total_runs,
        gpu_pool,
    )

    script_path = Path(__file__).resolve().parent / 'run_distributed.sh'
    if not script_path.exists():
        raise FileNotFoundError(f'Unable to locate helper script at {script_path}')

    log_root = Path('job_logs') / job_name
    log_root.mkdir(parents=True, exist_ok=True)
    failure_log = log_root / 'failures.txt'
    failure_records = []

    logger.info('Writing per-run logs under %s', log_root)

    router_port_base = int(job_cfg.get('router_port_base', 8000))
    router_port_stride = int(job_cfg.get('router_port_stride', 1))
    router_port_max = int(job_cfg.get('router_port_max', 65535))
    if router_port_max <= router_port_base:
        router_port_max = router_port_base + 1000 * max(router_port_stride, 1)

    active_router_ports: set[int] = set()
    router_port_cursor = router_port_base
    router_port_span = max(1, ((router_port_max - router_port_base) // max(router_port_stride, 1)) + 1)

    completed_runs = 0
    completed_durations = []
    overall_start = time.monotonic()

    def _format_load(load_value):
        try:
            return str(load_value).replace('.', 'p')
        except Exception:
            return str(load_value)

    def _format_duration(seconds):
        if seconds is None:
            return 'unknown'
        seconds = max(float(seconds), 0.0)
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f'{hours}h {minutes:02d}m'
        if minutes:
            return f'{minutes}m {secs:02d}s'
        return f'{seconds:.1f}s'

    def _estimate_time_remaining(active_running):
        if not completed_durations:
            return None, None
        avg_duration = sum(completed_durations) / len(completed_durations)
        now = time.monotonic()
        queued = total_runs - completed_runs - len(active_running)
        eta = max(queued, 0) * avg_duration
        for entry in active_running:
            elapsed = now - entry.get('start_time', now)
            remaining_single = avg_duration - elapsed
            if remaining_single <= 0:
                remaining_single = max(avg_duration * 0.25, 0.0)
            eta += remaining_single
        return avg_duration, eta

    def _log_progress(active_running):
        avg_duration, eta_seconds = _estimate_time_remaining(active_running)
        if avg_duration is None or eta_seconds is None:
            return
        finished = completed_runs
        in_flight = len(active_running)
        queued = total_runs - finished - in_flight
        elapsed = time.monotonic() - overall_start
        eta_text = _format_duration(eta_seconds)
        avg_text = _format_duration(avg_duration)
        elapsed_text = _format_duration(elapsed)
        finish_time = datetime.now() + timedelta(seconds=eta_seconds)
        logger.info(
            'Progress %d/%d complete (%d running, %d queued). Avg run %s, elapsed %s, ETA %s (finish ~%s)',
            finished,
            total_runs,
            in_flight,
            max(queued, 0),
            avg_text,
            elapsed_text,
            eta_text,
            finish_time.strftime('%H:%M:%S'),
        )

    def _allocate_router_port():
        nonlocal router_port_cursor
        attempts = 0
        while attempts <= router_port_span:
            candidate = router_port_cursor
            router_port_cursor += router_port_stride
            if router_port_cursor > router_port_max:
                router_port_cursor = router_port_base
            attempts += 1
            if candidate in active_router_ports:
                continue
            active_router_ports.add(candidate)
            return candidate
        raise RuntimeError('Unable to allocate a free router port.')

    def _build_command(gpu_csv: str, router_port: int, n_device_value: int,
                       baseline_value: str, load_scale_value):
        cmd = [
            'bash',
            str(script_path),
            gpu_csv,
            str(router_port),
            str(n_device_value),
            str(baseline_value),
            str(load_scale_value),
        ]
        optional_keys = ['slo_tpots', 'ttft_slo_scales', 'window', 'trace', 'model_name']
        for key in optional_keys:
            if key not in job_cfg or job_cfg[key] is None:
                break
            cmd.append(str(job_cfg[key]))
        return cmd

    def _stop_router(info):
        router_port = info['router_port']
        if not info.get('started'):
            active_router_ports.discard(router_port)
            return
        stop_cmd = ['bash', str(script_path), 'stop', str(router_port)]
        subprocess.run(stop_cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        active_router_ports.discard(router_port)

    running_entries = []
    router_infos = []

    try:
        for n_device in sorted(tasks_by_n.keys(), reverse=True):
            tasks_queue = tasks_by_n[n_device]
            if not tasks_queue:
                continue

            concurrency = min(len(tasks_queue), len(gpu_pool) // n_device)
            if concurrency <= 0:
                raise ValueError(
                    f'n_device={n_device} requires {n_device} GPUs but only {len(gpu_pool)} are available'
                )

            subsets = []
            offset = 0
            for _ in range(concurrency):
                subset = gpu_pool[offset:offset + n_device]
                if len(subset) < n_device:
                    break
                subsets.append(subset)
                offset += n_device
            if not subsets:
                raise ValueError(f'Unable to allocate GPU subsets for n_device={n_device}')

            router_infos = [
                {
                    'gpus': subset,
                    'gpu_csv': ','.join(str(g) for g in subset),
                    'router_port': _allocate_router_port(),
                    'busy': False,
                    'started': False,
                }
                for subset in subsets
            ]

            running_entries = []
            logger.info(
                'n_device=%d: %d task(s), %d router(s) active (%s)',
                n_device,
                len(tasks_queue),
                len(router_infos),
                ', '.join(str(info['router_port']) for info in router_infos),
            )

            def _launch_task(info, baseline_value, load_scale_value):
                gpu_csv = info['gpu_csv']
                router_port = info['router_port']
                load_suffix = _format_load(load_scale_value)
                safe_baseline = baseline_value.replace(':', '_').replace('/', '_')
                gpu_suffix = gpu_csv.replace(',', '-')
                log_path = log_root / f'{safe_baseline}_load_{load_suffix}_gpus_{gpu_suffix}_n{n_device}.log'
                log_file = log_path.open('w', encoding='utf-8')
                cmd = _build_command(gpu_csv, router_port, n_device, baseline_value, load_scale_value)
                log_file.write(f'GPUs: {gpu_csv}\n')
                log_file.write(f'Router port: {router_port}\n')
                log_file.write(f'Command: {" ".join(cmd)}\n')
                log_file.flush()
                try:
                    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
                except Exception as exc:  # pragma: no cover - defensive
                    log_file.write(f'Failed to start job: {exc}\n')
                    log_file.flush()
                    log_file.close()
                    logger.error(
                        'Failed to launch baseline=%s load_scale=%s n_device=%d on GPUs %s: %s',
                        baseline_value,
                        load_scale_value,
                        n_device,
                        gpu_csv,
                        exc,
                    )
                    failure_records.append({
                        'baseline': baseline_value,
                        'load_scale': load_scale_value,
                        'n_device': n_device,
                        'gpus': gpu_csv,
                        'router_port': router_port,
                        'reason': str(exc),
                        'log': str(log_path),
                        'cmd': cmd,
                    })
                    info['busy'] = False
                    return

                info['busy'] = True
                info['started'] = True
                running_entries.append({
                    'proc': proc,
                    'info': info,
                    'log_file': log_file,
                    'log_path': log_path,
                    'baseline': baseline_value,
                    'load_scale': load_scale_value,
                    'n_device': n_device,
                    'gpus': gpu_csv,
                    'router_port': router_port,
                    'cmd': cmd,
                    'start_time': time.monotonic(),
                })
                logger.info(
                    'Launching baseline=%s load_scale=%s n_device=%d on GPUs %s (router %d). %d queued runs remain for this n_device',
                    baseline_value,
                    load_scale_value,
                    n_device,
                    gpu_csv,
                    router_port,
                    len(tasks_queue),
                )

            try:
                while tasks_queue or running_entries:
                    for info in router_infos:
                        if not tasks_queue:
                            break
                        if info['busy']:
                            continue
                        baseline_value, load_scale_value = tasks_queue.popleft()
                        _launch_task(info, baseline_value, load_scale_value)

                    new_running = []
                    for entry in running_entries:
                        proc = entry['proc']
                        return_code = proc.poll()
                        if return_code is None:
                            new_running.append(entry)
                            continue

                        info = entry['info']
                        info['busy'] = False
                        now = time.monotonic()
                        duration = now - entry.get('start_time', now)
                        completed_durations.append(duration)
                        completed_duration_text = _format_duration(duration)

                        log_file = entry['log_file']
                        if not log_file.closed:
                            log_file.write(f'Process exited with code {return_code} after {duration:.2f}s\n')
                            log_file.flush()
                            log_file.close()

                        if return_code == 0:
                            logger.info(
                                'Completed baseline=%s load_scale=%s n_device=%d on GPUs %s (router %d) in %s',
                                entry['baseline'],
                                entry['load_scale'],
                                entry['n_device'],
                                entry['gpus'],
                                entry['router_port'],
                                completed_duration_text,
                            )
                        else:
                            logger.error(
                                'Run baseline=%s load_scale=%s n_device=%d on GPUs %s exited with code %d after %s; see %s',
                                entry['baseline'],
                                entry['load_scale'],
                                entry['n_device'],
                                entry['gpus'],
                                return_code,
                                completed_duration_text,
                                entry['log_path'],
                            )
                            failure_records.append({
                                'baseline': entry['baseline'],
                                'load_scale': entry['load_scale'],
                                'n_device': entry['n_device'],
                                'gpus': entry['gpus'],
                                'router_port': entry['router_port'],
                                'returncode': return_code,
                                'duration_sec': duration,
                                'log': str(entry['log_path']),
                                'cmd': entry['cmd'],
                            })

                        completed_runs += 1
                        _log_progress(new_running)

                    running_entries = new_running
                    if tasks_queue and not any(not info['busy'] for info in router_infos):
                        time.sleep(0.5)
            finally:
                for entry in running_entries:
                    proc = entry['proc']
                    if proc.poll() is None:
                        proc.terminate()
                    log_file = entry['log_file']
                    if not log_file.closed:
                        log_file.write('Process terminated unexpectedly.\n')
                        log_file.flush()
                        log_file.close()
                running_entries = []

            for info in router_infos:
                _stop_router(info)
            router_infos = []
    except KeyboardInterrupt:
        logger.warning('KeyboardInterrupt received; terminating active jobs and routers')
        for entry in running_entries:
            entry['proc'].terminate()
            if not entry['log_file'].closed:
                entry['log_file'].write('Process terminated due to KeyboardInterrupt.\n')
                entry['log_file'].flush()
                entry['log_file'].close()
        for info in router_infos:
            _stop_router(info)
        raise

    if failure_records:
        with failure_log.open('w', encoding='utf-8') as fh:
            for record in failure_records:
                fh.write(json.dumps(record) + '\n')
        logger.warning('Recorded %d failed runs for job %s (details: %s)',
                       len(failure_records), job_name, failure_log)
    else:
        logger.info('Job %s completed successfully', job_name)

    return failure_records



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--job', type=str, default='Coder-Qwen7B')
args = parser.parse_args()
for job in args.job.split(','):
    run_job(job)

# start = time.time()
# run_job('Coder-Qwen7B')
# end = time.time()
# print(f'Coder-Qwen7B took {end - start} seconds')

# start = time.time()
# run_job('ChatBot-Qwen7B')
# end = time.time()
# print(f'ChatBot-Qwen7B took {end - start} seconds')

# start = time.time()
# run_job('Arxiv-Qwen7B')
# end = time.time()
# print(f'Arxiv-Qwen7B took {end - start} seconds')
