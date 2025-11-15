
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
baselines = ['vllm', 'sarathi', 'slosserve-dp', 'slosserve-edf', 'vllm+', 'sarathi+', 'qlm', 'qlm+']
jobs = {
    'Coder-Qwen7B-bustiness': {
        'baseline': ['slosserve-edf', 'sarathi', 'qlm', 'vllm'],
        'load_scale': 0.1,
        'slo_tpots': 0.025,
        'ttft_slo_scales': 3,
        'window': '3979:4582',
        'trace': ['sharegpt_code:bursty_0.0', 'sharegpt_code:bursty_0.2', 'sharegpt_code:bursty_0.4', 'sharegpt_code:bursty_0.6', 'sharegpt_code:bursty_0.8', 'sharegpt_code:bursty_1.0'],
        'model_name': 'Qwen/Qwen2.5-7B-Instruct',
    },
    'ChatBot-Qwen7B-bustiness': {
        'baseline': ['slosserve-edf', 'sarathi', 'qlm', 'vllm'],
        'load_scale': 0.1,
        'slo_tpots': 0.10,
        'ttft_slo_scales': 5.0,
        'window': '600:1199',
        'trace': ['azure_chat_23:bursty_0.0', 'azure_chat_23:bursty_0.2', 'azure_chat_23:bursty_0.4', 'azure_chat_23:bursty_0.6', 'azure_chat_23:bursty_0.8', 'azure_chat_23:bursty_1.0'],
        'model_name': 'Qwen/Qwen2.5-7B-Instruct',
    },
    'Coder-Qwen7B': {
        'baseline': ['vllm+'],
        'load_scale': [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        'slo_tpots': 0.025,
        'ttft_slo_scales': 3,
        'window': '3979:4581',
        'trace': 'sharegpt_code:azure_code_23',
        'model_name': 'Qwen/Qwen2.5-7B-Instruct',
    },
    'Coder-Qwen7B-tpot-ablation': {
        'baseline': ['sarathi', 'qlm', 'slosserve-edf'],
        'load_scale': [0.30],
        'slo_tpots': [0.025, 0.05, 0.10, 0.15, 0.20],
        'ttft_slo_scales': 3,
        'window': '3979:4581',
        'trace': 'sharegpt_code:azure_code_23',
        'model_name': 'Qwen/Qwen2.5-7B-Instruct',
    },
    'Coder-Qwen7B-ttft-ablation': {
        'baseline': [ 'slosserve-edf'],
        'load_scale': [0.30],
        'slo_tpots': 0.025,
        'ttft_slo_scales': [2,3,4,5,6,7,8],
        'window': '3979:4581',
        'trace': 'sharegpt_code:azure_code_23',
        'model_name': 'Qwen/Qwen2.5-7B-Instruct',
    },
    'ChatBot-Qwen7B': {
        'baseline': baselines,
        'load_scale': [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        'slo_tpots': 0.10,
        'ttft_slo_scales': 5.0,
        'window': '600:1201',
        'trace': 'azure_chat_23:azure_chat_23',
        'model_name': 'Qwen/Qwen2.5-7B-Instruct',
    },
    'ChatBot-Qwen7B': {
        'baseline': baselines,
        'load_scale': [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        'slo_tpots': 0.10,
        'ttft_slo_scales': 5.0,
        'window': '600:1201',
        'trace': 'azure_chat_23:azure_chat_23',
        'model_name': 'Qwen/Qwen2.5-7B-Instruct',
    },
    'Arxiv-Qwen7B': {
        'baseline':  ['vllm', 'sarathi', 'slosserve-edf', 'vllm+', 'sarathi+', 'qlm+'],
        'load_scale': [5,6,7,8,9,10,11],
        'slo_tpots': 0.10,
        'ttft_slo_scales': 3.0,
        'window': '450:551',
        'trace': 'arxiv_summary:burstgpt_GPT-4_Conversation',
        'model_name': 'Qwen/Qwen2.5-7B-Instruct',
    },
    'Coder-Gemma27B': {
        'baseline': baselines,
        'load_scale': [0.10, 0.20, 0.30, 0.40, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'slo_tpots': 0.05,
        'ttft_slo_scales': 3,
        'window': '3979:4579',
        'trace': 'sharegpt_code:azure_code_23',
        'model_name': 'google/gemma-3-27b-it',
    },
    'ChatBot-Gemma27B': {
        'baseline': baselines,
        'load_scale': [0.1,0.2,0.3,0.4,0.5],
        'slo_tpots': 0.10,
        'ttft_slo_scales': 5.0,
        'window': '601:1201',
        'trace': 'azure_chat_23:azure_chat_23',
        'model_name': 'google/gemma-3-27b-it',
    },
    'Arxiv-Gemma27B': {
        'baseline': baselines,
        'load_scale': [1, 2, 3, 4, 5, 6],
        'slo_tpots': 0.10,
        'ttft_slo_scales': 3.0,
        'window': '450:551',
        'trace': 'arxiv_summary:burstgpt_GPT-4_Conversation',
        'model_name': 'google/gemma-3-27b-it',
    }
}


def run_job(job_name: str):
    '''
    write a script the allocate the GPUs and run the jobs (baseline X load_scale)
    source run_unit.sh <gpu> <port> <baseline> <load_scale> [slo_tpots] [ttft_slo_scales] [window] [trace] [model_name]
    check for failures and store them in a file, run another job.
    '''
    import torch
    import json
    import time
    import subprocess
    from collections import deque
    from pathlib import Path
    from datetime import datetime, timedelta

    job_cfg = jobs.get(job_name)
    if job_cfg is None:
        raise ValueError(f'Unknown job "{job_name}". Available jobs: {list(jobs)}')

    n_gpus = torch.cuda.device_count()
    if n_gpus <= 0:
        raise RuntimeError('run_job requires at least one visible CUDA device.')

    baselines = job_cfg.get('baseline', [])
    if isinstance(baselines, str):
        baselines = [baselines]
    load_scales = job_cfg.get('load_scale', [])
    if isinstance(load_scales, (int, float, str)):
        load_scales = [load_scales]
    if not baselines or not load_scales:
        raise ValueError(f'Job "{job_name}" must specify both "baseline" and "load_scale".')

    def _ensure_list(value, *, default=None):
        if value is None:
            if default is not None:
                value = default
            else:
                return [None]
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    slo_tpots_values = _ensure_list(job_cfg.get('slo_tpots'))
    ttft_slo_scales_values = _ensure_list(job_cfg.get('ttft_slo_scales'))
    window_values = _ensure_list(job_cfg.get('window'))
    trace_values = _ensure_list(job_cfg.get('trace'))
    model_name_values = _ensure_list(job_cfg.get('model_name', 'Qwen/Qwen2.5-7B-Instruct'))

    work_queue = deque(
        {
            'baseline': baseline,
            'load_scale': load_scale,
            'slo_tpots': slo_tpot,
            'ttft_slo_scales': ttft_scale,
            'window': window,
            'trace': trace,
            'model_name': model_name,
        }
        for baseline in baselines
        for load_scale in load_scales
        for slo_tpot in slo_tpots_values
        for ttft_scale in ttft_slo_scales_values
        for window in window_values
        for trace in trace_values
        for model_name in model_name_values
    )

    total_runs = len(work_queue)
    if not total_runs:
        logger.info('Job %s has no tasks to run', job_name)
        return []

    def _describe_dimension(label: str, values):
        if len(values) == 1 and values[0] is None:
            return None
        return f"{len(values)} {label}"

    dimension_parts = [
        f"{len(baselines)} baselines",
        f"{len(load_scales)} load scales",
        _describe_dimension('slo_tpots', slo_tpots_values),
        _describe_dimension('ttft_slo_scales', ttft_slo_scales_values),
        _describe_dimension('windows', window_values),
        _describe_dimension('traces', trace_values),
        _describe_dimension('models', model_name_values),
    ]
    dimension_text = ' x '.join(part for part in dimension_parts if part)
    logger.info(
        'Starting job %s with %d runs (%s) on %d GPU(s)',
        job_name,
        total_runs,
        dimension_text,
        n_gpus,
    )

    script_path = Path(__file__).resolve().parent / 'run_unit.sh'
    if not script_path.exists():
        raise FileNotFoundError(f'Unable to locate helper script at {script_path}')

    log_root = Path('job_logs') / job_name
    log_root.mkdir(parents=True, exist_ok=True)
    failure_log = log_root / 'failures.txt'
    failure_records = []

    logger.info('Writing per-run logs under %s', log_root)

    port_base = int(job_cfg.get('port_base', 8100))
    port_stride = int(job_cfg.get('port_stride', 10))

    vllm_script_path = Path(__file__).resolve().parent / 'launch_vllm.sh'
    if not vllm_script_path.exists():
        raise FileNotFoundError(f'Unable to locate vLLM launcher at {vllm_script_path}')

    gpu_list = ','.join(str(i) for i in range(n_gpus))
    port_list = ','.join(str(port_base + i * port_stride) for i in range(n_gpus))

    non_null_models = [m for m in model_name_values if m is not None]
    if non_null_models:
        model_name = str(non_null_models[0])
        unique_models = {str(m) for m in non_null_models}
        if len(unique_models) > 1:
            logger.warning('Job %s has multiple model_name values; using %s for vLLM bootstrap', job_name, model_name)
    else:
        model_name = 'Qwen/Qwen2.5-7B-Instruct'

    is_distributed = str(int(bool(job_cfg.get('is_distributed', 0))))
    vllm_cmd = ['bash', str(vllm_script_path), gpu_list, port_list, model_name, is_distributed]
    logger.info('Launching vLLM servers: %s', ' '.join(vllm_cmd))
    try:
        subprocess.run(vllm_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f'Failed to launch vLLM servers (exit {exc.returncode}).') from exc

    free_gpus = list(range(n_gpus))
    running: list[dict] = []

    completed_durations: list[float] = []
    overall_start = time.monotonic()

    def _sanitize_value(value) -> str:
        text = str(value)
        for src, dst in {'.': 'p', ':': '-', '/': '_', ' ': ''}.items():
            text = text.replace(src, dst)
        return ''.join(ch if ch.isalnum() or ch in '-_=' else '_' for ch in text)

    def _format_load(load_value):
        return _sanitize_value(load_value)

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

    def _estimate_time_remaining():
        if not completed_durations:
            return None, None
        avg_duration = sum(completed_durations) / len(completed_durations)
        now = time.monotonic()
        waiting = len(work_queue) * avg_duration
        running_remaining = 0.0
        for active in running:
            elapsed = now - active.get('start_time', now)
            remaining_single = avg_duration - elapsed
            if remaining_single <= 0:
                remaining_single = max(avg_duration * 0.25, 0.0)
            running_remaining += remaining_single
        eta = max(waiting + running_remaining, 0.0)
        return avg_duration, eta

    def _log_progress():
        avg_duration, eta_seconds = _estimate_time_remaining()
        if avg_duration is None or eta_seconds is None:
            return
        finished = len(completed_durations)
        in_flight = len(running)
        queued = len(work_queue)
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
            queued,
            avg_text,
            elapsed_text,
            eta_text,
            finish_time.strftime('%H:%M:%S'),
        )

    def _format_suffix(name: str, value):
        if value is None:
            return ''
        sanitized = _sanitize_value(value)
        if not sanitized:
            return ''
        return f'_{name}_{sanitized}'

    def _build_command(gpu_id: int, port: int, run_cfg: dict) -> list[str]:
        cmd = [
            'bash',
            str(script_path),
            str(gpu_id),
            str(port),
            str(run_cfg['baseline']),
            str(run_cfg['load_scale']),
        ]
        optional_keys = ['slo_tpots', 'ttft_slo_scales', 'window', 'trace', 'model_name']
        for key in optional_keys:
            value = run_cfg.get(key)
            if value is not None:
                cmd.append(str(value))
        return cmd

    def _launch_job(gpu_id: int, run_cfg: dict):
        port = port_base + gpu_id * port_stride
        load_suffix = _format_load(run_cfg['load_scale'])
        slo_suffix = _format_suffix('slo', run_cfg.get('slo_tpots'))
        ttft_suffix = _format_suffix('ttft', run_cfg.get('ttft_slo_scales'))
        window_suffix = _format_suffix('window', run_cfg.get('window'))
        log_name = f"{run_cfg['baseline']}_load_{load_suffix}{slo_suffix}{ttft_suffix}{window_suffix}_gpu{gpu_id}.log"
        log_path = log_root / log_name
        cmd = _build_command(gpu_id, port, run_cfg)
        log_file = log_path.open('w', encoding='utf-8')
        log_file.write(f"Command: {' '.join(cmd)}\n")
        log_file.flush()
        remaining = len(work_queue)
        logger.info(
            'Launching baseline=%s load_scale=%s slo_tpots=%s ttft_slo_scales=%s on GPU %d (port %d). %d queued runs remain',
            run_cfg['baseline'],
            run_cfg['load_scale'],
            run_cfg.get('slo_tpots'),
            run_cfg.get('ttft_slo_scales'),
            gpu_id,
            port,
            remaining,
        )
        try:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            start_time = time.monotonic()
        except Exception as exc:
            log_file.write(f'Failed to start job: {exc}\n')
            log_file.flush()
            log_file.close()
            logger.error('Failed to launch baseline=%s load_scale=%s on GPU %d: %s',
                         run_cfg['baseline'], run_cfg['load_scale'], gpu_id, exc)
            failure_records.append({
                'baseline': run_cfg['baseline'],
                'load_scale': run_cfg['load_scale'],
                'slo_tpots': run_cfg.get('slo_tpots'),
                'ttft_slo_scales': run_cfg.get('ttft_slo_scales'),
                'window': run_cfg.get('window'),
                'trace': run_cfg.get('trace'),
                'model_name': run_cfg.get('model_name'),
                'gpu': gpu_id,
                'port': port,
                'reason': str(exc),
                'log': str(log_path),
            })
            return

        running.append({
            'proc': proc,
            'gpu': gpu_id,
            'baseline': run_cfg['baseline'],
            'load_scale': run_cfg['load_scale'],
            'slo_tpots': run_cfg.get('slo_tpots'),
            'ttft_slo_scales': run_cfg.get('ttft_slo_scales'),
            'window': run_cfg.get('window'),
            'trace': run_cfg.get('trace'),
            'model_name': run_cfg.get('model_name'),
            'log_file': log_file,
            'log_path': log_path,
            'port': port,
            'cmd': cmd,
            'start_time': start_time,
        })

    try:
        while work_queue or running:
            while work_queue and free_gpus:
                gpu_id = free_gpus.pop(0)
                run_cfg = work_queue.popleft()
                _launch_job(gpu_id, run_cfg)

            for entry in list(running):
                return_code = entry['proc'].poll()
                if return_code is None:
                    continue
                now = time.monotonic()
                duration = now - entry.get('start_time', now)
                completed_durations.append(duration)
                duration_text = _format_duration(duration)
                if not entry['log_file'].closed:
                    entry['log_file'].write(f'Process exited with code {return_code} after {duration:.2f}s\n')
                    entry['log_file'].flush()
                    entry['log_file'].close()
                free_gpus.append(entry['gpu'])
                running.remove(entry)
                if return_code == 0:
                    logger.info(
                        'Completed baseline=%s load_scale=%s slo_tpots=%s ttft_slo_scales=%s on GPU %d (port %d) in %s',
                        entry['baseline'],
                        entry['load_scale'],
                        entry.get('slo_tpots'),
                        entry.get('ttft_slo_scales'),
                        entry['gpu'],
                        entry['port'],
                        duration_text,
                    )
                else:
                    logger.error(
                        'Run baseline=%s load_scale=%s slo_tpots=%s ttft_slo_scales=%s on GPU %d exited with code %d after %s; see %s',
                        entry['baseline'],
                        entry['load_scale'],
                        entry.get('slo_tpots'),
                        entry.get('ttft_slo_scales'),
                        entry['gpu'],
                        return_code,
                        duration_text,
                        entry['log_path'],
                    )
                    failure_records.append({
                        'baseline': entry['baseline'],
                        'load_scale': entry['load_scale'],
                        'slo_tpots': entry.get('slo_tpots'),
                        'ttft_slo_scales': entry.get('ttft_slo_scales'),
                        'window': entry.get('window'),
                        'trace': entry.get('trace'),
                        'model_name': entry.get('model_name'),
                        'gpu': entry['gpu'],
                        'port': entry['port'],
                        'returncode': return_code,
                        'duration_sec': duration,
                        'log': str(entry['log_path']),
                        'cmd': entry['cmd'],
                    })
                _log_progress()

            free_gpus.sort()
            if running and not free_gpus:
                time.sleep(1)

    except KeyboardInterrupt:
        logger.warning('KeyboardInterrupt received; terminating %d running jobs', len(running))
        for entry in running:
            entry['proc'].terminate()
        raise
    finally:
        if running:
            logger.info('Cleaning up %d unfinished runs', len(running))
        for entry in running:
            if not entry['log_file'].closed:
                entry['log_file'].write('Process terminated unexpectedly.\n')
                entry['log_file'].close()

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
parser.add_argument('--job', type=str, default='Coder-Qwen7B', choices=list(jobs.keys()))

args = parser.parse_args()
run_job(args.job)
