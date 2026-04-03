import asyncio
import csv
import os
from xxlimited import Str
import httpx
import copy
import random
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Any, List, Tuple, Dict, AsyncGenerator, Optional
import math
import time
from contextlib import asynccontextmanager
from abc import ABC
from enum import Enum
import json
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
import numpy as np
from ray.util.queue import Queue as RayQueue
from contextlib import contextmanager
try:
    import SLOsServe_C
except ImportError:
    SLOsServe_C = None

# NEW: Ray
import ray
from asyncio import Queue

from vllm.sampling_params import SamplingParams
from vllm.inputs import TokensPrompt


import logging
from SLOsServe.client_spec import parse_client_spec
from SLOsServe.perf_model import PerfModel
from SLOsServe.router.execplan_bus import (ExecPlanBus, ExecPlan,
                                           extract_load_statistics,
                                           make_default_load_stats)
from SLOsServe.router.adm_ctrl import Request as BatchPlannerRequest
from SLOsServe.router.engine_shutdown import shutdown_engine_instance
from SLOsServe.router.macro import DUMP_SCHS, DUMP_ADM
from SLOsServe.router.sem_util import MaxCapSemaphore
from motivation.energy_measure import EnergyMeter, EnergyHistoryRecorder

DEBUG = False

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s [pid=%(process)d] %(message)s"
        ))
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            handler.setLevel(log_level)
    return logger


logger = setup_logger("SLOsServe.router.api_server", os.getenv("SLOSSERVE_LOG_LEVEL", "INFO"))


def _build_rejection_response(
    rejection_reason: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"finish_reason": "rejected"}
    if rejection_reason is not None:
        payload["stop_reason"] = rejection_reason
        payload["rejection_reason"] = rejection_reason
    return payload


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _sorted_metric_columns(fieldnames: list[str], prefix: str) -> list[str]:
    return sorted(
        [name for name in fieldnames if name.startswith(prefix)],
        key=lambda name: int(name[len(prefix):]),
    )


def _to_float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _parse_energy_csv(csv_path: str) -> tuple[list[dict[str, Any]], list[float], float]:
    if not os.path.exists(csv_path):
        return [], [], 0.0

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        j_cols = _sorted_metric_columns(fieldnames, "J_gpu")
        w_cols = _sorted_metric_columns(fieldnames, "W_gpu")
        mhz_cols = _sorted_metric_columns(fieldnames, "MHz_gpu")
        per_gpu = [0.0 for _ in j_cols]
        events: list[dict[str, Any]] = []
        for row in reader:
            timestamp = float(row["ts"])
            for local_idx, j_col in enumerate(j_cols):
                energy = float(row.get(j_col, 0.0) or 0.0)
                per_gpu[local_idx] += energy
                w_col = w_cols[local_idx] if local_idx < len(w_cols) else None
                mhz_col = mhz_cols[local_idx] if local_idx < len(mhz_cols) else None
                events.append({
                    "event_type": "energy",
                    "timestamp": timestamp,
                    "device_id": local_idx,
                    "energy": energy,
                    "power": float(row.get(w_col, 0.0) or 0.0) if w_col else 0.0,
                    "mhz": _to_float_or_none(row.get(mhz_col)) or 0.0,
                })
    return events, per_gpu, sum(per_gpu)


def _read_text_if_exists(path: str | None) -> str | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _build_worker_dump_path(filename: str, device_id: int) -> str:
    base, ext = os.path.splitext(filename)
    if not ext:
        ext = ".json"
    return f"{base}.device{device_id}{ext}"


def _build_worker_energy_csv_path(filename: str, device_id: int) -> str:
    base, _ = os.path.splitext(filename)
    return f"{base}.device{device_id}.energy.csv"


class WorkerEnergyProfiler:
    def __init__(self, *, device_id: int, interval_s: float = 0.1):
        self.device_id = device_id
        self.interval_s = interval_s
        self.physical_gpu_ids = self._detect_physical_gpu_ids()
        self.csv_path: str | None = None
        self._meter: EnergyMeter | None = None
        self._recorder: EnergyHistoryRecorder | None = None
        self._started = False

    def _detect_physical_gpu_ids(self) -> list[int]:
        gpu_ids: list[int] = []
        try:
            for gpu_id in ray.get_gpu_ids():
                if isinstance(gpu_id, str):
                    if gpu_id.isdigit():
                        gpu_ids.append(int(gpu_id))
                elif isinstance(gpu_id, (int, float)) and float(gpu_id).is_integer():
                    gpu_ids.append(int(gpu_id))
        except Exception:
            gpu_ids = []
        if gpu_ids:
            return gpu_ids

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        parsed: list[int] = []
        for token in visible.split(","):
            token = token.strip()
            if token.isdigit():
                parsed.append(int(token))
        return parsed

    def restart(self, store_prefix: str | None) -> None:
        self.stop()
        if not store_prefix:
            self.csv_path = None
            return
        self.csv_path = os.path.abspath(
            f"{store_prefix}.device{self.device_id}.energy.csv")
        _ensure_parent_dir(self.csv_path)
        self._meter = EnergyMeter(devices=self.physical_gpu_ids or None)
        self._meter.start()
        self._recorder = EnergyHistoryRecorder(
            self._meter,
            interval_s=self.interval_s,
            csv_path=self.csv_path,
        )
        self._recorder.start()
        self._started = True
        logger.info(
            "Started worker energy profiler for device %s on physical GPUs %s -> %s",
            self.device_id,
            self.physical_gpu_ids,
            self.csv_path,
        )

    def stop(
        self,
        *,
        keep_csv: bool = True,
        include_csv_content: bool = True,
    ) -> dict[str, Any]:
        recorder = self._recorder
        meter = self._meter
        csv_path = self.csv_path
        was_started = self._started
        self._started = False
        self._recorder = None
        self._meter = None

        per_gpu_joules: list[float] = []
        total_joules = 0.0
        if recorder is not None:
            recorder.stop()
        if meter is not None:
            per_gpu_joules, total_joules = meter.stop()

        energy_events: list[dict[str, Any]] = []
        energy_csv_content: str | None = None
        if was_started and csv_path:
            energy_events, csv_per_gpu_joules, csv_total_joules = _parse_energy_csv(
                csv_path)
            if include_csv_content:
                energy_csv_content = _read_text_if_exists(csv_path)
            if not per_gpu_joules:
                per_gpu_joules = csv_per_gpu_joules
            if total_joules == 0.0:
                total_joules = csv_total_joules
            if not keep_csv and os.path.exists(csv_path):
                try:
                    os.remove(csv_path)
                except OSError as exc:
                    logger.warning(
                        "Failed to remove worker energy CSV %s: %s",
                        csv_path,
                        exc,
                    )

        return {
            "energy_csv_path": csv_path if was_started and keep_csv else None,
            "energy_csv_content": energy_csv_content,
            "energy_events": energy_events,
            "per_gpu_joules": per_gpu_joules,
            "total_joules": total_joules,
            "physical_gpu_ids": list(self.physical_gpu_ids),
        }

@dataclass 
class EngineActor:
    engine_actor: ray.ObjectRef
    shared_q: RayQueue
    local_queues: dict[str, Queue] = field(default_factory=dict)
    _profile_events: list | None = None
    
    async def run(self):
        # drain_time_budget_s = 0.001
        while True:
            await asyncio.sleep(0.020)
            payload = await self.shared_q.get_async()
            qlen = self.shared_q.qsize()
            if qlen > 0:
                payloads = self.shared_q.get_nowait_batch(qlen)
                for _ in payloads:
                    payload.extend(_)
            if payload:
                self._handle_payload(payload)

    def _handle_payload(self, payload):
        for item in payload:
            rid = item.get("request_id")
            if not rid:
                continue
            q = self.local_queues.get(rid)
            if q is None:
                logger.warning(f'{rid} not found in local_queues')
                continue
            q.put_nowait(item)  # ✅ don’t await if you want speed
            if item.get('finish_reason') is not None:
                q.put_nowait(None)
    
    def connect(self, request_id):
        @contextmanager
        def _f():
            q = Queue()
            self.local_queues[request_id] = q
            try: 
                yield q
            finally:
                self.local_queues.pop(request_id)
        return _f()

engine_actors: list[EngineActor] | None = None 
engine_tasks: list | None = None 
# REPLACED: engine -> Ray actors (one per logical replica)
execplan_bus_actor = None

routing_loop_task: asyncio.Task | None = None


def _parse_clients_arg(raw_clients: str | None) -> list[str]:
    return parse_client_spec(raw_clients)


async def _wait_for_engine_actors_ready() -> None:
    if not engine_actors:
        return
    await asyncio.gather(*[
        ea.engine_actor.wait_until_ready.remote() for ea in engine_actors
    ])


def _engine_actor_shutdown_timeout_s() -> float:
    raw = os.getenv(
        "SLOSSERVE_ENGINE_ACTOR_SHUTDOWN_TIMEOUT_S",
        os.getenv("SLOSSERVE_ENGINE_SHUTDOWN_TIMEOUT_S", "5.0"),
    )
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 5.0


async def _shutdown_engine_actor_handle(
    actor_handle: Any,
    label: str,
    *,
    always_kill: bool = False,
) -> None:
    shutdown_succeeded = False
    timeout_s = _engine_actor_shutdown_timeout_s()

    try:
        shutdown_ref = actor_handle.shutdown.remote()
        if timeout_s == 0.0:
            await shutdown_ref
        else:
            await asyncio.wait_for(shutdown_ref, timeout=timeout_s)
        shutdown_succeeded = True
    except asyncio.TimeoutError:
        logger.warning(
            "%s shutdown hung for %.3fs; killing Ray actor",
            label,
            timeout_s,
        )
    except Exception as exc:
        logger.warning(
            "%s shutdown failed with %r; killing Ray actor",
            label,
            exc,
        )

    if always_kill or not shutdown_succeeded:
        try:
            ray.kill(actor_handle)
        except Exception as exc:
            logger.warning("Failed to kill %s: %r", label, exc)


async def _shutdown_engine_actors(
    actors: list[EngineActor] | None,
    *,
    always_kill: bool = False,
) -> None:
    if not actors:
        return

    await asyncio.gather(*[
        _shutdown_engine_actor_handle(
            actor.engine_actor,
            f"engine_actor[{idx}]",
            always_kill=always_kill,
        )
        for idx, actor in enumerate(actors)
    ])


def _parse_worker_env_args(raw_envs: list[str] | None) -> dict[str, str]:
    env_vars: dict[str, str] = {}
    if not raw_envs:
        return env_vars
    for raw_env in raw_envs:
        for item in raw_env.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(
                    f"Invalid --worker_env value {item!r}; expected KEY=VALUE")
            key, value = item.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(
                    f"Invalid --worker_env value {item!r}; empty KEY")
            env_vars[key] = value
    return env_vars

def _task_done(task: asyncio.Task):
    try:
        task.result()  # re-raises exception if task failed
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Background coroutine crashed")

# =========================
# Ray Actor: one replica/process
# =========================
@ray.remote(max_concurrency=1024)
class EngineWorker:
    def __init__(self,
                 model_name: str,
                 mock_connector: bool,
                 output_queue: RayQueue,
                 mock_engine: bool = False,
                 mock_engine_device_mem_bytes: int = 20_000_000_000,
                 device_id: int = -1,
                 tensor_parallel_size: int = 1,
                 worker_env: dict[str, str] | None = None,
                 vllm_port: int | None = None,
                 execplan_bus=None):
        setup_logger("SLOsServe.router.api_server", os.getenv("SLOSSERVE_LOG_LEVEL", "INFO"))
        if worker_env:
            for key, value in worker_env.items():
                os.environ[str(key)] = str(value)
            logger.info("Replica %s applied worker env: %s", device_id,
                        sorted(worker_env.items()))
        self.tensor_parallel_size = tensor_parallel_size
        if vllm_port is not None:
            os.environ["VLLM_PORT"] = str(vllm_port)
            logger.info("Replica %s using explicit VLLM_PORT=%s", device_id,
                        vllm_port)
        if not mock_engine:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.config import CompilationConfig, KVTransferConfig
            from vllm.v1.engine.async_llm import AsyncLLM
            # One logical model replica per actor/process. Ray assigns the
            # physical GPUs and vLLM uses TP within the actor-local GPU set.
            engine_args = AsyncEngineArgs(
                model=model_name,
                enable_chunked_prefill=True,
                enable_expert_parallel=False,
                max_num_batched_tokens=16384,
                tensor_parallel_size=tensor_parallel_size,
                data_parallel_size=1,
                distributed_executor_backend="mp",
                max_num_seqs=512,
                long_prefill_token_threshold=16384,
                compilation_config=CompilationConfig(
                    cudagraph_mode='FULL_AND_PIECEWISE',
                ),
                scheduler_cls='vllm.v1.core.sched.scheduler_adm_ctrl.SchedulerAdmCtrl',
                kv_transfer_config=(
                    KVTransferConfig(kv_connector='NixlConnector', kv_role='kv_both')
                    if not mock_connector else None
                ),
                enable_prefix_caching = True
            )
            self.engine = AsyncLLM.from_engine_args(engine_args)
            if execplan_bus is not None:
                self.engine.set_engine_state_publisher(execplan_bus, device_id)
        else: 
            from SLOsServe.router.mock_engine import MockEngine
            self.engine = MockEngine(
                model_name,
                mock_connector,
                device_id=device_id,
                tensor_parallel_size=tensor_parallel_size,
                device_mem=mock_engine_device_mem_bytes,
                execplan_bus=execplan_bus,
            )
        self.device_id = device_id
        self.output_queue = output_queue
        self._local_queue = asyncio.Queue()
        self._control_queue = asyncio.Queue()
        self.is_ready = True
        self._mux_task = asyncio.create_task(
            self._mux(),
            name=f"EngineWorker[{device_id}]._mux",
        )
        self._mux_task.add_done_callback(_task_done)
        self._profile_events = []
        self._energy_store_prefix: str | None = None
        self._energy_profiler = None if mock_engine else WorkerEnergyProfiler(
            device_id=device_id,
            interval_s=float(os.getenv("SLOSSERVE_ENERGY_INTERVAL_S", "0.1")),
        )
        
    async def wait_until_ready(self):
        while not self.is_ready:
            await asyncio.sleep(0.1)
        return True

    async def update_config(self, request_json: dict):
        await self.engine.update_config(request_json)
        self._profile_events.clear()
        if self._energy_profiler is not None:
            self._energy_profiler.stop()
            self._energy_store_prefix = request_json.get("store_prefix")

    async def start_energy_profiling(self, store_prefix: str | None = None):
        if self._energy_profiler is None:
            return False
        if store_prefix is not None:
            self._energy_store_prefix = store_prefix
        self._energy_profiler.restart(self._energy_store_prefix)
        return True
        
    async def profile_step(self, request_json: dict):
        batch = request_json['batch']
        n = request_json['n']
        verbose = request_json['verbose']
        return await self.engine.profile_step(batch, n, verbose)

    async def dump_profile_events(self, path: str, include_energy_csv: bool = True):
        energy_summary = {
            "energy_csv_path": None,
            "energy_csv_content": None,
            "energy_events": [],
            "per_gpu_joules": [],
            "total_joules": 0.0,
            "physical_gpu_ids": [],
        }
        if self._energy_profiler is not None:
            energy_summary = self._energy_profiler.stop(
                keep_csv=include_energy_csv,
                include_csv_content=include_energy_csv,
            )
        _ensure_parent_dir(path)
        await self.engine.dump_profile_events(path)
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        data.extend(self._profile_events)
        data.extend(energy_summary["energy_events"])
        data.sort(key=lambda event: event.get("timestamp", 0.0))
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        return {
            "profile_events_path": path,
            "profile_events": data,
            "energy_csv_path": energy_summary["energy_csv_path"],
            "energy_csv_content": energy_summary["energy_csv_content"],
            "per_gpu_joules": energy_summary["per_gpu_joules"],
            "total_joules": energy_summary["total_joules"],
            "physical_gpu_ids": energy_summary["physical_gpu_ids"],
        }

    async def shutdown(self):
        if self._energy_profiler is not None:
            self._energy_profiler.stop()
        shutdown_error = None
        try:
            await shutdown_engine_instance(self.engine)
        except Exception as exc:
            shutdown_error = exc
        finally:
            self.engine = None
            self._mux_task.cancel()
            try:
                await self._mux_task
            except asyncio.CancelledError:
                pass

        if shutdown_error is not None:
            raise shutdown_error

    async def _mux(self):
        FLUSH_T = 0.05
        FLUSH_N = 100
        now = time.time()
        buf = []

        async def _flush(payload: list[dict[str, Any]]):
            if not payload:
                return
            for item in payload:
                if DEBUG:
                    self._profile_events.append({
                        'event_type': '_mux_push',
                        'request_id': item['request_id'],
                        'device_id': self.device_id,
                        'timestamp': time.time()
                    })
            await self.output_queue.put_async(payload)

        while True: 
            get_control_task = asyncio.create_task(
                self._control_queue.get(),
                name=f"EngineWorker[{self.device_id}]._mux.control_get",
            )
            get_local_task = asyncio.create_task(
                self._local_queue.get(),
                name=f"EngineWorker[{self.device_id}]._mux.local_get",
            )
            done, pending = await asyncio.wait(
                {get_control_task, get_local_task},
                timeout=FLUSH_T,
                return_when=asyncio.FIRST_COMPLETED,
            )

            control_payload = []
            if get_control_task in done:
                control_payload.append(get_control_task.result())
            if get_local_task in done:
                item = get_local_task.result()
                if DEBUG:
                    self._profile_events.append({
                        'event_type': '_mux_read',
                        'request_id': item['request_id'],
                        'device_id': self.device_id, 
                        'timestamp': time.time()
                    })
                buf.append(item)
                while True:
                    try: 
                        item = self._local_queue.get_nowait()
                        buf.append(item)
                        if DEBUG:
                            self._profile_events.append({
                                'event_type': '_mux_read',
                                'request_id': item['request_id'],
                                'device_id': self.device_id, 
                                'timestamp': time.time()
                            })
                    except asyncio.QueueEmpty:
                        break 

            for task in pending:
                task.cancel()
            for task in pending:
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            while True:
                try:
                    item = self._control_queue.get_nowait()
                    control_payload.append(item)
                except asyncio.QueueEmpty:
                    break

            if control_payload:
                await _flush(control_payload)
                now = time.time()

            if len(buf) >= FLUSH_N or (len(buf) and (time.time() - now > FLUSH_T)):
                now = time.time()
                await _flush(buf)
                buf = []
                
    async def prefill_once(self, req_data: dict, request_id: str):
        """Prefill to first token (max_tokens=1). Return a single dict response."""
        from vllm.sampling_params import SamplingParams
        from vllm.outputs import RequestOutput
        from vllm.inputs import TokensPrompt

        extra_args = req_data['vllm_xargs'].copy()
        extra_args['kv_transfer_params'] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None
        }

        last_output: RequestOutput | None = None
        prompt = req_data['prompt'] if isinstance(req_data['prompt'], str) \
                 else TokensPrompt(prompt_token_ids=req_data['prompt'])

        admitted, generator, rejection_reason = await self.engine.add_request(prompt=prompt,
            request_id=request_id,
            sampling_params=SamplingParams.from_optional(
                max_tokens=1, ignore_eos=True, extra_args=extra_args
            ))
        
        if not admitted: 
            return False, None, rejection_reason

        async for out in generator:
            last_output = out
        assert last_output is not None 

        response = asdict(last_output.outputs[0]) if len(last_output.outputs) else None 
        if isinstance(response, dict):
            response['timestamp'] = time.time()
            if isinstance(last_output.kv_transfer_params, dict):
                response['kv_transfer_params'] = last_output.kv_transfer_params
        return True, response, None

    async def decode_stream(self, req_data: dict, request_id: str):
        
        # print('dispatch', request_id, time.time(), 'engineworker')

        extra_args = req_data['vllm_xargs'].copy()
        kv_transfer_params = req_data.get('kv_transfer_params')
        if kv_transfer_params is not None:
            extra_args['kv_transfer_params'] = kv_transfer_params

        prompt = req_data['prompt'] if isinstance(req_data['prompt'], str) \
                 else TokensPrompt(prompt_token_ids=req_data['prompt'])
        
        admitted, generator, rejection_reason = await self.engine.add_request(prompt=prompt,
            request_id=request_id,
            sampling_params=SamplingParams.from_optional(
                max_tokens=req_data['max_tokens'],
                extra_args=extra_args,
                ignore_eos=True,
            ))
        
        if not admitted:
            return False, rejection_reason

        async def _loop():
            n_tokens = 0
            text_len = 0
            async for output in generator:
                completion = output.outputs[0]
                chunk = {
                    'request_id': request_id,
                    'text': completion.text[text_len:],
                    'finish_reason': completion.finish_reason,
                    'stop_reason': completion.stop_reason,
                    'token_ids': completion.token_ids[n_tokens:],
                    'num_computed_tokens': completion.num_computed_tokens,
                    'timestamp': time.time(),
                }
                if (
                    completion.finish_reason == 'rejected'
                    and completion.stop_reason is not None
                ):
                    chunk['rejection_reason'] = completion.stop_reason
                n_tokens = len(completion.token_ids)
                text_len = len(completion.text)
                await self._local_queue.put(chunk)
        task = asyncio.create_task(
            _loop(),
            name=f"EngineWorker[{self.device_id}].decode_stream[{request_id}]",
        )
        task.add_done_callback(_task_done)
        return True, None

    async def get_load_statistics(self, n: int = 100) -> list[dict[str, Any]]:
        return await self.engine.get_load_statistics(n)
    
    async def abort_request(self, request_id: str):
        await self.engine.abort(request_id)

    async def health_check(self) -> dict[str, Any] | None:
        return await self.engine.check_health()
# =========================
# Helpers (engine RPC)
# =========================
async def send_request_to_service_engine(client_idx: int,
                                         req_data: dict,
                                         request_id: str):
    """
    Prefill step via the EngineWorker actor owning this replica.
    """
    assert engine_actors is not None
    actor = engine_actors[client_idx]
    # Use Ray async await on the ObjectRef
    admitted, response, rejection_reason = await actor.engine_actor.prefill_once.remote(req_data, request_id)
    return admitted, response, rejection_reason

async def stream_service_response_engine(
    client_idx,
    req_data: dict,
    request_id: str,
) -> tuple[bool, AsyncGenerator | None, str | None]:
    """
    Decode step via EngineWorker actor; returns list of chunks and yields them here as SSE.
    """
    assert engine_actors is not None
    actor = engine_actors[client_idx]
    
    Q = actor.local_queues[request_id] = Queue()
    admitted, rejection_reason = await actor.engine_actor.decode_stream.remote(req_data, request_id)
    
    if not admitted:
        actor.local_queues.pop(request_id)
        return False, None, rejection_reason 
    
    async def _generator():
        while True:
            item = await Q.get()
            if item is None:
                break
            yield item
            # wait for either next chunk OR remote task failure
            # try:
            #     item = await asyncio.wait_for(Q.get(), 10)
            #     if item is None:
            #         break
            #     yield item
            # except asyncio.TimeoutError:
            #     logger.error(f"timeout in stream service for {request_id}")
            #     break
        actor.local_queues.pop(request_id)
    
    return True, _generator(), None
    
        
async def abort_request(client_idx: int, request_id: str):
    assert engine_actors is not None
    actor = engine_actors[client_idx]
    await actor.engine_actor.abort.remote(request_id)

# =========================
# Router & Request types
# =========================
class RequestState(Enum):
    WAITING = 0
    PREFILL_REJECTED = 1
    PREFILL_REJECTED_WAITING = 2
    PREFILL_FINISHED = 3
    DECODE_REJECTED = 4
    DECODE_REJECTED_WAITING = 5
    DECODE_FINISHED = 6
    TIMEOUT = 7

    def __lt__(self, other):
        if isinstance(other, RequestState):
            return self.value < other.value
        return NotImplemented
    
    def __ge__(self, other):
        if isinstance(other, RequestState):
            return self.value >= other.value
        return NotImplemented

@dataclass
class ReqeustGroup:
    n_abortion: int = 0
    has_acceptance: bool = False
    requests: list['RequestInstance'] = field(default_factory=list)

class RequestInstance:
    def __init__(self,
                 request_id: str | None = None,
                 payload: Any = None,
                 response_queue: Queue | None = None,
                 arrival_time: float | None = None):
        self.request_id = request_id
        self.payload = payload
        self.payload.update({'request_id': request_id})
        self.response_queue = response_queue
        self.arrival_time = arrival_time

        # Router State
        self.admitted: bool | None = None
        self.prefill_device_id: int = -1
        self.decode_device_id: int = -1
        self.rejection_prob: float = 0.0
        self.admission_stat: dict[str, Any] | None = None
        self.rejection_reason: str | None = None

        # runtime state
        self.state: RequestState = RequestState.WAITING
        
        self._group: ReqeustGroup | None = None
        
        self._state: dict[str, Any] = {}
        self.kv_transfer_params: dict[str, Any] | None = None
        
        # Staled Request State
        self.num_computed_tokens = 0
        self.timestamps: list[tuple[int, float]] = []
    
    def update(self, data: dict):
        if (num_computed_tokens:= data.get('num_computed_tokens', None)) is not None: 
            self.num_computed_tokens = num_computed_tokens
            event_ts = data.get('timestamp', time.time())
            self.timestamps.append((num_computed_tokens, event_ts))
    
    @property 
    def is_prefill(self):
        return self.state < RequestState.PREFILL_FINISHED
    
    @property 
    def is_decode(self):
        return not self.is_prefill
    
    @property
    def num_prompt_tokens(self):
        return self.payload['vllm_xargs']['input_length']

    @property
    def session_id(self) -> str | None:
        return self.payload.get('vllm_xargs', {}).get('session_id')

    @property
    def service_tier(self) -> str:
        return str(self.payload.get('vllm_xargs', {}).get('service_tier', 'default'))

    @property
    def initial_service_tier(self) -> str:
        return str(
            self.payload.get('vllm_xargs', {}).get(
                'initial_service_tier',
                self.service_tier,
            )
        )

    @property
    def cached_tokens(self) -> int:
        try:
            return max(0, int(self.payload.get('vllm_xargs', {}).get('cached_tokens', 0)))
        except (TypeError, ValueError):
            return 0

    
    @property
    def is_slo_violation(self):
        expected_timestamps = [(self.payload['vllm_xargs']['input_length'], self.payload['vllm_xargs']['prefill_ddl'])]
        for _ in range(self.payload['vllm_xargs']['output_length'] - 1):
            expected_timestamps.append((expected_timestamps[-1][0] + 1, expected_timestamps[-1][1] + self.payload['vllm_xargs']['slo_tpot']))
        idx = 0
        for n_token, t in expected_timestamps:
            while idx < len(self.timestamps) and self.timestamps[idx][0] < n_token: 
                idx += 1
            if idx == len(self.timestamps):
                # violation
                return True 
            if self.timestamps[idx][1] > t:
                return True 
        return False 
                 

    def fork(self):
        if self._group is None: 
            self._group = ReqeustGroup()
            self._group.requests.append(self)
        
        new_request = RequestInstance(request_id=f'{self.request_id}', payload=self.payload.copy(), response_queue=self.response_queue, arrival_time = self.arrival_time)
        new_request._group = self._group
        self._group.requests.append(new_request)
        return new_request
            
    def is_finished(self):
        return self.state in [RequestState.DECODE_FINISHED, RequestState.TIMEOUT, RequestState.PREFILL_REJECTED, RequestState.DECODE_REJECTED]

    def update_state(self, key: str, value: Any):
        self._state[key] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    
@dataclass(kw_only=True)
class LoadEvent:
    type: str
    timestamp: float
    device_id: int = -1
    service_tier: str = "default"

@dataclass
class SLOsServeEvent(LoadEvent):
    future_batches: list[dict[str, Any]] = field(default_factory=list)
    execplan: list[dict[str, Any]] = field(default_factory=list)
    elapsed: float = -1.0

@dataclass
class PoolEvent(LoadEvent):
    waiting_size: int
    running_size: int

@dataclass
class ArrivalEvent(LoadEvent):
    request_id: str

@dataclass
class FinishEvent(LoadEvent):
    request_id: str
    is_rejection: bool 
    is_slo_violation: bool

@dataclass
class RejectEvent(LoadEvent):
    request_id: str
    
@dataclass 
class DeviceStat:
    device_id: int
    rejection_rate: float
    n_considered: int
    past_utilization: float
    future_utilization: float
    waiting_size: int
    running_size: int
    n_requests: int

class LoadStat:
    def __init__(self, max_window: float = 10, n_devices: int = 8):
        self.events: list[LoadEvent] = []
        self.n_reqs = 0
        self.max_window = max_window
        self.n_devices = n_devices
        self.n_requests = [0 for _ in range(n_devices)]
        self.n_regular_requests = [0 for _ in range(n_devices)]
        self.version_id = 0
        
    def reset(self, n_devices: int = 8):
        self.version_id = 0
        self.events = []
        self.n_devices = n_devices
        self.n_requests = [0 for _ in range(n_devices)]
        self.n_regular_requests = [0 for _ in range(n_devices)]

    @staticmethod
    def _is_best_effort_service_tier(service_tier: Any) -> bool:
        return str(service_tier or "default") == "best_effort"

    def _add_stat(self, event: dict[str, Any]):
        is_best_effort = self._is_best_effort_service_tier(
            event.get('service_tier', 'default'))
        if event['type'] == 'slosserve':
            self.events.append(SLOsServeEvent(**event))
        elif event['type'] == 'pool':
            self.events.append(PoolEvent(**event))
        elif event['type'] == 'arrival' and (self.n_devices > event['device_id'] >= 0):
            self.n_requests[event['device_id']] += 1
            if not is_best_effort:
                self.n_regular_requests[event['device_id']] += 1
            self.events.append(ArrivalEvent(**event))
        elif event['type'] == 'finish' and (self.n_devices > event['device_id'] >= 0):
            self.events.append(FinishEvent(**event))
            self.n_requests[event['device_id']] -= 1
            if not is_best_effort:
                self.n_regular_requests[event['device_id']] -= 1
        elif event['type'] == 'reject' and (self.n_devices > event['device_id'] >= 0):
            self.events.append(RejectEvent(**event))
            self.n_requests[event['device_id']] -= 1
            if not is_best_effort:
                self.n_regular_requests[event['device_id']] -= 1

    def get_rejection_rate(self, window = 5, include_best_effort: bool = True) -> float:
        earliest_time = time.time() - window
        per_device_stats = defaultdict(lambda: {'n_arrivals': 0, 'n_rejects': 0})
        rejected = set()
        for req in self.events[::-1]:
            if req.timestamp < earliest_time:
                break
            if ((not include_best_effort)
                    and self._is_best_effort_service_tier(
                        getattr(req, 'service_tier', 'default'))):
                continue
            if req.type == 'arrival':
                if req.device_id not in per_device_stats:
                    per_device_stats[req.device_id] = {
                        'n_arrivals': 0,
                        'n_rejects': 0,
                    }
                per_device_stats[req.device_id]['n_arrivals'] += 1
                if (req.request_id, req.device_id) in rejected:
                    per_device_stats[req.device_id]['n_rejects'] += 1
            
            if req.type == 'reject':
                rejected.add((req.request_id, req.device_id))

        return {device_id: {
            'rejection_rate': per_device_stats[device_id]['n_rejects'] / (per_device_stats[device_id]['n_arrivals'] + 1e-6),
            'n_considered': per_device_stats[device_id]['n_arrivals']
        } for device_id in range(self.n_devices)}
    
    def get_batch_utilization(self, window = 5) -> float:
        earliest_time = time.time() - window
        past_utilizations = defaultdict(list)
        future_utils = {}

        def _batch_util(batch: dict[str, Any]) -> float:
            # New schema: execplan with allocated tokens + optional capacity.
            if "allocated_tokens" in batch:
                allocated = batch.get("allocated_tokens", {})
                if not isinstance(allocated, dict):
                    allocated = {}
                scheduled = 0.0
                for n_tokens in allocated.values():
                    try:
                        scheduled += float(n_tokens)
                    except (TypeError, ValueError):
                        continue
                capacity = batch.get("capacity")
                if capacity is None:
                    return 1.0 if scheduled > 0 else 0.0
                try:
                    cap = float(capacity)
                except (TypeError, ValueError):
                    cap = 0.0
                if cap <= 0:
                    return 1.0 if scheduled > 0 else 0.0
                return min(1.0, scheduled / cap)

            # Backward-compatible schema: future_batches from scheduler stats.
            n_tokens = float(batch.get("n_tokens", 0.0))
            prefill_bs = float(batch.get("prefill_bs", 0.0))
            denom = n_tokens + prefill_bs
            if denom <= 0:
                return 0.0
            return n_tokens / denom

        def _batch_duration(batches: list[dict[str, Any]],
                            idx: int,
                            elapsed: float) -> float:
            batch = batches[idx]
            if "duration" in batch:
                try:
                    return max(0.0, float(batch["duration"]))
                except (TypeError, ValueError):
                    pass
            if idx + 1 < len(batches):
                try:
                    cur_t = float(batch.get("estimated_time", 0.0))
                    next_t = float(batches[idx + 1].get("estimated_time", cur_t))
                    return max(0.0, next_t - cur_t)
                except (TypeError, ValueError):
                    pass
            if idx == 0 and elapsed > 0:
                return float(elapsed)
            try:
                return max(0.0, float(batch.get("estimated_time", 0.0)))
            except (TypeError, ValueError):
                return 0.0

        for event in self.events[::-1]:
            if event.timestamp < earliest_time:
                break
            if event.type == 'slosserve':
                batches = event.execplan if len(event.execplan) > 0 else event.future_batches
                if len(batches) == 0:
                    continue
                executed_batch = batches[0]
                t = _batch_duration(batches, 0, event.elapsed)
                past_utilizations[event.device_id].append((_batch_util(executed_batch) * t, t))
                if event.device_id not in future_utils:
                    t = 0
                    util = 0
                    for i, batch in enumerate(batches[:10]):
                        batch_time = _batch_duration(batches, i, event.elapsed)
                        util += _batch_util(batch) * batch_time
                        t += batch_time
                    future_utils[event.device_id] = (util / t) if t > 0 else 0.0

        for k, v in past_utilizations.items():
            utils, times = zip(*v)
            past_utilizations[k] = float(np.sum(utils) / (np.sum(times) + 1e-6)) if len(v) > 0 else 0

        return {
            device_id: {
                'past_utilization': past_utilizations.get(device_id, 0),
                'future_utilization': future_utils.get(device_id, 0)
            } for device_id in range(self.n_devices)
        }
        
    def get_pool_size(self, window = 5) -> dict[int, int]:
        earliest_time = time.time() - window
        ret = {}
        for event in self.events[::-1]:
            if event.timestamp < earliest_time:
                break
            if event.type == 'pool' and event.device_id not in ret:
                ret[event.device_id] = {
                    'waiting_size': event.waiting_size,
                    'running_size': event.running_size,
                }
        return {device_id: ret.get(device_id, {'waiting_size': 0, 'running_size': 0}) for device_id in range(self.n_devices)}

    def get_stat(
        self,
        window = 5,
        include_best_effort: bool = True,
    ) -> list[DeviceStat]:
        rejection_rates = self.get_rejection_rate(
            window,
            include_best_effort=include_best_effort,
        )
        batch_utilizations = self.get_batch_utilization(window)
        pool_sizes = self.get_pool_size(window)
        ret = []
        for device_id in range(self.n_devices):
            ret.append(DeviceStat(
                device_id,
                rejection_rates[device_id]['rejection_rate'],
                rejection_rates[device_id]['n_considered'],
                batch_utilizations[device_id]['past_utilization'],
                batch_utilizations[device_id]['future_utilization'],
                pool_sizes[device_id]['waiting_size'],
                pool_sizes[device_id]['running_size'],
                (
                    self.n_requests[device_id]
                    if include_best_effort else
                    self.n_regular_requests[device_id]
                )))
        return ret

    def add_event(self, event: dict[str, Any] | list):
        if isinstance(event, list):
            for e in event:
                self._add_stat(e)
        else:
            self._add_stat(event)
        self.events.sort(key=lambda e: e.timestamp)
        
        idx = 0
        while idx < len(self.events) and self.events[idx].timestamp < time.time() - self.max_window:
            idx += 1
        self.events = self.events[idx:]

    def get_slo_violation_rate(self, window = 5):
        earliest_time = time.time() - window 
        for event in self.events[::-1]:
            if event.timestamp < earliest_time: break

    def update(self, stats: list):
        self.stats = stats
        self.version_id += 1

class Router(ABC):
    def _ensure_session_maps(self):
        if not hasattr(self, "_session_prefill_device_map"):
            self._session_prefill_device_map: dict[str, int] = {}
            self._session_decode_device_map: dict[str, int] = {}

    def set_load_stat(self, load_stat: LoadStat):
        self.load_stat = load_stat
    
    def set_profile_events(self, profile_events: dict):
        self._profile_events = profile_events

    def run(self, waiting_requests: List[RequestInstance],
            running_requests: List[RequestInstance]):
        '''
        Run the router to decide the admission and device assignment for the waiting requests.
        for every request in the waiting_requests, the router decide: 
        request.admitted: bool
        request.prefill_device_id: int
        request.decode_device_id: int
        the router is not required to make decisions, it can keep the request.admitted as None
        '''
        raise NotImplementedError

    def update(self, request: RequestInstance, new_state: RequestState):
        '''
        Optional update for the router to keep track of the request state.
        '''
        pass

    def get_session_home_prefill_device(self, session_id: str | None) -> int | None:
        if not session_id:
            return None
        self._ensure_session_maps()
        return self._session_prefill_device_map.get(session_id)

    def get_session_home_decode_device(self, session_id: str | None) -> int | None:
        if not session_id:
            return None
        self._ensure_session_maps()
        return self._session_decode_device_map.get(session_id)

    def note_request_state(self, request: RequestInstance, new_state: RequestState):
        session_id = request.session_id
        if not session_id:
            return
        if new_state not in (RequestState.PREFILL_FINISHED, RequestState.DECODE_FINISHED):
            return
        self._ensure_session_maps()
        if request.prefill_device_id >= 0:
            self._session_prefill_device_map[session_id] = request.prefill_device_id
        if request.decode_device_id >= 0:
            self._session_decode_device_map[session_id] = request.decode_device_id

    def update_json(self, request_json: dict, i: int):
        '''
        Optional update for the config passed to the i-th engine.
        For example, in KV disagg router, the config is updated for P devices to allow large batch size limit.
        '''
        return request_json
    
    def select_asap_server(
        self,
        load_stat: LoadStat,
        request: RequestInstance | None = None,
    ) -> tuple[int, int]:
        '''
        Optional server selection when a request missed the SLO.
        Reuse the last placement when available; otherwise start from the top.
        '''
        if request is not None and request.prefill_device_id >= 0:
            decode_device_id = request.decode_device_id
            if decode_device_id < 0:
                decode_device_id = request.prefill_device_id
            return request.prefill_device_id, decode_device_id
        if request is not None:
            return 0, 0
        n_reqs = [stat.n_requests for stat in load_stat.get_stat(
            include_best_effort=True)]
        min_n_req = min(n_reqs)
        freest_device_id = n_reqs.index(min_n_req)
        return freest_device_id, freest_device_id
    
    def set_iter(self, iter: int):
        self.iter = iter 

class AutoScalingRouter(Router):
    def __init__(self, n_devices: int, router_kwargs: dict[str, Any]):
        if isinstance(router_kwargs, str):
            router_kwargs = json.loads(router_kwargs)
        self.n_devices = n_devices
        self.round_robin_init = router_kwargs.get('round_robin_init', False)
        self.base_device_id = defaultdict(int)
        self.i = 0
        model_path = router_kwargs.get('model_path', 'auto_scaling_model.json')
        model_key = router_kwargs.get('model_key', 'all')
        self.fallback_policy = router_kwargs.get('fallback_policy', 'best')
        assert self.fallback_policy in ['best', 'random', 'round_robin', 'reject']
        logger.info(f"AutoScalingRouter: n_devices = {n_devices}, model_path = {model_path}")
        with open(model_path, 'r') as f:
            self.model = json.load(f)[model_key]
        self.threshold = router_kwargs.get('threshold', None)
        logger.info(f"AutoScalingRouter: model_key = {model_key}, model = {self.model}, threshold = {self.threshold}")
        self._iter = 0
        
    def calc_rejection_prob(self, device_stats: DeviceStat, request: RequestInstance) -> float:
        features = {
            'device_id': device_stats.device_id,
            'past_utilization': device_stats.past_utilization,
            'future_utilization': device_stats.future_utilization,
            'waiting_size': device_stats.waiting_size,
            'running_size': device_stats.running_size,
            'input_length': request.payload['vllm_xargs']['input_length'],
            'prefill_ddl': request.payload['vllm_xargs']['prefill_ddl'] - time.time(),
            'output_length': request.payload['vllm_xargs']['output_length'],
            'rejection_rate': device_stats.rejection_rate,
            'n_requests': device_stats.n_requests,
        }
        
        def inner(model, features):
            intercept = float(model.get("intercept", 0.0))
            feature_keys = [k for k in model.keys() if k not in ("intercept", "threshold")]

            # Compute standardized linear score: b + sum_i w_i * (x_i - mean_i) / std_i
            score = intercept
            for feat in feature_keys:
                if feat not in features:
                    # If missing, treat as mean (contributes 0) or raise—your call
                    continue
                x = float(features[feat])
                m = float(model[feat]["mean"])
                s = float(model[feat]["std"])
                s = s if s != 0.0 else 1.0  # guard against divide-by-zero
                z = (x - m) / s
                score += float(model[feat]["coeff"]) * z

            # Logistic probability
            if score < -10 or score > 10:
                logger.warning(f"Score {score} out of expected range [-10, 10]. Clipping. model = {model}, features = {features}")
            score = max(min(score, 10), -10)
            prob = 1.0 / (1.0 + math.exp(-score))
            threshold = self.threshold or model.get('threshold', 0.5)
            return prob, prob >= threshold
        
        if self.model.get('per_device', False):
            return inner(self.model[str(features['device_id'])], features)
        return inner(self.model, features)
        
    def calc_rejection_prob_q_only(self, device_stats: DeviceStat, request: RequestInstance) -> tuple[float, bool]:
        rej = bool(device_stats.n_requests > 50)
        return float(rej), rej
    
    def run(self, waiting_requests: List[RequestInstance],
            running_requests: List[RequestInstance]):
        # we implement a try next 
        waiting_requests.sort(key=lambda x: x.payload['vllm_xargs']['prefill_ddl'])
        
        stats = self.load_stat.get_stat(include_best_effort=False)
        
        if len(waiting_requests):
            self._iter += 1
            if self._iter % 100 == 0:
                probs = defaultdict(list)
                for requst in random.sample(waiting_requests, min(10, len(waiting_requests))):
                    for i, stat in enumerate(stats): 
                        prob, pred = self.calc_rejection_prob(stat, requst)
                        probs[i].append(prob)
                logger.info(f"AutoScalingRouter: ")
                for i in range(self.n_devices):
                    logger.info(f"AutoScalingRouter: device_id = {i}, probs = {np.mean(probs[i])} +- {np.std(probs[i])}")
        for request in waiting_requests:
            # if time.time() > request.payload['vllm_xargs']['prefill_ddl']:
            #     request.admitted = False
            #     continue
            
            if request.get_state('fall_back', False):
                request.admitted = False
                continue
            
            request.admitted = True

            device_to_try = request.prefill_device_id + 1
            # logger.info(f"AutoScalingRouter: device_to_try = {device_to_try}")
            best_device = (-1, 1.0)
            while device_to_try < self.n_devices:
                stat = stats[device_to_try]
                rejection_prob, pred = self.calc_rejection_prob(stat, request)
                # rejection_prob, pred = self.calc_rejection_prob_q_only(stat, request)
                if not pred:
                    best_device = (device_to_try, rejection_prob)
                    break
                if rejection_prob < best_device[1]:
                    best_device = (device_to_try, rejection_prob)
                device_to_try += 1
            if device_to_try == self.n_devices:
                request.update_state('fall_back', True)
                # if self.fallback_policy == 'reject':
                request.admitted = False
                continue
            # logger.info(f"AutoScalingRouter: best_device = {best_device}")
            
            assert self.fallback_policy == 'best'
            request.prefill_device_id = request.decode_device_id = best_device[0]
            request.rejection_prob = best_device[1]
            stat = stats[request.prefill_device_id]
            request.admission_stat = asdict(stat)
            
            stat.waiting_size += 1
            stat.n_considered += 1
            stat.rejection_rate = (stat.rejection_rate * (stat.n_considered - 1) + request.rejection_prob) / stat.n_considered
            stat.n_requests += 1

class RoundRobinRouter(Router):
    def __init__(self, n_devices: int, router_kwargs: dict):
        self.n_devices = n_devices
        self.group_i = 0
        self.sticky_sessions = bool(router_kwargs.get('sticky_sessions', False))
        self.is_pd_disagg = router_kwargs.get('is_pd_disagg', False)
        self.group_size = router_kwargs.get('group_size', self.n_devices)
        assert self.n_devices % self.group_size == 0
        self.n_group = self.n_devices // self.group_size
        self.per_group_indices = [0 for _ in range(self.n_group)]
        self.per_group_decode_indices = [0 for _ in range(self.n_group)]
        assert self.n_devices % self.n_group == 0
        self.n_prefill_or_mixed_per_group = router_kwargs.get('n_prefill_per_group', self.group_size)
        if self.is_pd_disagg:
            assert self.n_prefill_or_mixed_per_group < self.group_size
        

    def run(self, waiting_requests: List[RequestInstance],
            running_requests: List[RequestInstance]):
        # if not self.is_pd_disagg:
        for request in waiting_requests:
            request.admitted = True
            if self.sticky_sessions and request.session_id:
                home_prefill = self.get_session_home_prefill_device(request.session_id)
                if home_prefill is not None:
                    request.prefill_device_id = home_prefill
                    if not self.is_pd_disagg:
                        request.decode_device_id = home_prefill
                    else:
                        home_decode = self.get_session_home_decode_device(request.session_id)
                        if home_decode is not None:
                            request.decode_device_id = home_decode
                    continue
            group_i = self.group_i
            idx = self.per_group_indices[group_i]
            request.prefill_device_id = group_i * self.group_size + idx 
            self.group_i = (self.group_i + 1) % self.n_group
            self.per_group_indices[group_i] = (self.per_group_indices[group_i] + 1) % self.n_prefill_or_mixed_per_group

            if not self.is_pd_disagg:
                request.decode_device_id = request.prefill_device_id
                continue
            
            decode_idx = self.per_group_decode_indices[group_i]
            request.decode_device_id = decode_idx + self.n_prefill_or_mixed_per_group + group_i * self.group_size
            self.per_group_decode_indices[group_i] = (self.per_group_decode_indices[group_i] + 1) % (self.group_size - self.n_prefill_or_mixed_per_group)

class LlumnixLoadRouter(Router):
    def __init__(self, n_devices: int, router_kwargs: dict):
        if isinstance(router_kwargs, str):
            router_kwargs = json.loads(router_kwargs)
        self.n_devices = n_devices
        self.is_pd_disagg = router_kwargs.get('is_pd_disagg', False)
        self.group_size = router_kwargs.get('group_size', self.n_devices)
        self.block_size = router_kwargs.get('block_size', 16)
        assert self.block_size == 16, f"Unsupported block_size={self.block_size}"
        assert self.n_devices % self.group_size == 0
        self.n_group = self.n_devices // self.group_size
        assert self.n_devices % self.n_group == 0
        self.n_prefill_or_mixed_per_group = router_kwargs.get('n_prefill_per_group', self.group_size)
        if self.is_pd_disagg:
            assert self.n_prefill_or_mixed_per_group < self.group_size
        self.stats_version = -1

    def get_stats(self):
        if self.stats_version < self.load_stat.version_id:
            self.stats = copy.copy(self.load_stat.stats)
            self.stats_version = self.load_stat.version_id
        return self.stats

    def run(self, waiting_requests: List[RequestInstance],
            running_requests: List[RequestInstance]):
        stats = self.get_stats()
        if not (isinstance(stats, list) and len(stats) >= self.n_devices):
            logger.warning(f'wrong stats format {self.stats}')
            self.stats = [{
                'n_waitings': 0, 
                'n_running': 0,
                'num_free_blocks': 20000
            } for _ in range(self.n_devices)]
            
        
        def remaining_steps(stat: dict[str, float]) -> tuple[int, float, float]:
            num_requests = stat['n_waitings'] + stat['n_running']
            if num_requests == 0:
                return (1, stat['num_free_blocks'], 0.0)
            return (0, stat['num_free_blocks'] / num_requests, stat['num_free_blocks'])

        def pick_best(start: int, end: int) -> int:
            best_dids = []
            best_score = None
            for did in range(start, end):
                score = remaining_steps(self.stats[did])
                if best_score is None or score > best_score:
                    best_score = score
                    best_dids = [did]
                elif score == best_score:
                    best_dids.append(did)
            return random.choice(best_dids)

        def reserve(did: int, n_blocks: int):
            stat = self.stats[did]
            stat['n_waitings'] += 1
            stat['num_free_blocks'] -= n_blocks

        if not self.is_pd_disagg:
            for request in waiting_requests:
                did = pick_best(0, self.n_devices)
                request.admitted = True
                request.prefill_device_id = did
                request.decode_device_id = did
                prefill_blocks = math.ceil(request.num_prompt_tokens / self.block_size)
                reserve(did, prefill_blocks)
            return

        prefill_span = self.n_prefill_or_mixed_per_group
        decode_span = self.group_size - prefill_span
        for request in waiting_requests:
            prefill_blocks = math.ceil(request.num_prompt_tokens / self.block_size)
            best_prefill_did = 0
            best_decode_did = prefill_span
            best_score = None
            best_aux_score = None

            for group_i in range(self.n_group):
                group_start = group_i * self.group_size
                prefill_did = pick_best(group_start, group_start + prefill_span)
                decode_did = pick_best(group_start + prefill_span, group_start + prefill_span + decode_span)
                prefill_score = remaining_steps(self.stats[prefill_did])
                decode_score = remaining_steps(self.stats[decode_did])
                score = min(prefill_score, decode_score)
                aux_score = (
                    prefill_score[0] + decode_score[0],
                    prefill_score[1] + decode_score[1],
                    prefill_score[2] + decode_score[2],
                )
                if best_score is None or score > best_score or (score == best_score and aux_score > best_aux_score):
                    best_score = score
                    best_aux_score = aux_score
                    best_prefill_did = prefill_did
                    best_decode_did = decode_did

            request.admitted = True
            request.prefill_device_id = best_prefill_did
            request.decode_device_id = best_decode_did
            reserve(best_prefill_did, prefill_blocks)
            # Reserve decode-side capacity in the same routing pass to avoid collapse.
            reserve(best_decode_did, prefill_blocks)
            
    def update(self, req: RequestInstance, state: RequestState):
        if state == RequestState.PREFILL_FINISHED:
            if isinstance(self.stats, list) and req.prefill_device_id < len(self.stats):
                self.stats[req.prefill_device_id]['n_running'] -= 1
        elif state == RequestState.DECODE_FINISHED:
            if isinstance(self.stats, list) and req.decode_device_id < len(self.stats):
                self.stats[req.decode_device_id]['n_running'] -= 1

class LightestFirstRouter(Router):
    def __init__(self, n_devices: int, router_kwargs: str):
        self.n_devices = n_devices
        self.devices = np.zeros(n_devices)
        self.n_tries = defaultdict(int)

    def run(self, waiting_requests: List[RequestInstance],
            running_requests: List[RequestInstance]):
        for request in waiting_requests:
            if self.n_tries[request.request_id] >= self.n_devices - 1 or time.time() > request.payload['vllm_xargs']['prefill_ddl']:
                request.admitted = False
                continue
            self.n_tries[request.request_id] += 1
            request.admitted = True
            request.prefill_device_id = request.decode_device_id = int(np.argmin(self.devices))
            self.devices[request.prefill_device_id] += 1
    
    def update(self, request: RequestInstance, new_state: RequestState):
        if new_state == RequestState.DECODE_FINISHED:
            self.devices[request.decode_device_id] -= 1
        elif new_state == RequestState.TIMEOUT:
            self.devices[request.prefill_device_id] -= 1
        elif new_state == RequestState.PREFILL_REJECTED:
            self.devices[request.prefill_device_id] += 1
        elif new_state == RequestState.PREFILL_FINISHED:
            self.devices[request.prefill_device_id] += 1
        elif new_state == RequestState.WAITING:
            self.devices[request.prefill_device_id] -= 1

class DisaggregatedRouter(Router):
    def __init__(self, n_devices: int, router_kwargs: str):
        import re
        pattern = re.compile(r'(\d+)P(\d+)D')
        match = pattern.match(router_kwargs)
        if match:
            self.n_prefill_devices = int(match.group(1))
            self.n_decode_devices = int(match.group(2))
        else:
            raise ValueError(f"Invalid router kwargs: {router_kwargs}")
        assert self.n_prefill_devices + self.n_decode_devices == n_devices
        self.prefill_i = 0
        self.decode_i = 0

    def run(self, waiting_requests: List[RequestInstance],
            running_requests: List[RequestInstance]):
        for request in waiting_requests:
            request.admitted = True
            request.prefill_device_id = self.prefill_i
            request.decode_device_id = self.decode_i + self.n_prefill_devices
            self.prefill_i = (self.prefill_i + 1) % self.n_prefill_devices
            self.decode_i = (self.decode_i + 1) % self.n_decode_devices

    def update_json(self, request_json: dict, i: int):
        new_request_json = request_json.copy()
        if i < self.n_prefill_devices:
            if new_request_json['scheduling_policy'] in ['vllm-fcfs', 'vllm-priority']:
                new_request_json['scheduling_kwargs']['max_num_batched_tokens'] = 16384
                new_request_json['scheduling_kwargs']['long_prefill_token_threshold'] = 4096
        return new_request_json

class DisaggregatedLCRouter(AutoScalingRouter):
    def __init__(self, n_devices: int, router_kwargs: str):
        super().__init__(n_devices, router_kwargs)
        if isinstance(router_kwargs, str):
            router_kwargs = json.loads(router_kwargs)
        assert isinstance(router_kwargs, dict)
        self.n_devices = n_devices
        self.max_decode_batch_size = router_kwargs['max_decode_batch_size']

    def run(self, waiting_requests: List[RequestInstance],
            running_requests: List[RequestInstance]):
        stats = self.load_stat.get_stat(include_best_effort=False)
        for request in waiting_requests:
            if time.time() > request.payload['vllm_xargs']['prefill_ddl']:
                request.admitted = False
                continue
            
            if request.get_state('fall_back', False):
                request.admitted = False
                continue
            
            assert request.state in [RequestState.WAITING, RequestState.PREFILL_REJECTED_WAITING, RequestState.DECODE_REJECTED_WAITING]
            
            if request.decode_device_id == -1: 
                request.decode_device_id = self.n_devices
            
            request.admitted = True

            if request.state < RequestState.PREFILL_FINISHED:
                device_to_try = request.prefill_device_id + 1
                # logger.info(f"AutoScalingRouter: device_to_try = {device_to_try}")
                best_device = (-1, 1.0)
                while device_to_try < self.n_devices:
                    rejection_prob, pred = self.calc_rejection_prob(stats[device_to_try], request)
                    if not pred:
                        best_device = (device_to_try, rejection_prob)
                        break
                    if rejection_prob < best_device[1]:
                        best_device = (device_to_try, rejection_prob)
                    device_to_try += 1
                if device_to_try == self.n_devices:
                    request.update_state('fall_back', True)
                    if self.fallback_policy == 'reject':
                        request.admitted = False
                        continue
                # logger.info(f"AutoScalingRouter: best_device = {best_device}")
                
                assert self.fallback_policy == 'best'
                request.prefill_device_id = best_device[0]
                request.rejection_prob = best_device[1]
                stat = stats[request.prefill_device_id]
                request.admission_stat = asdict(stat)
                
                stat.waiting_size += 1
                stat.n_considered += 1
                stat.rejection_rate = (stat.rejection_rate * (stat.n_considered - 1) + request.rejection_prob) / stat.n_considered
                stat.n_requests += 1
            
            assert request.state < RequestState.DECODE_FINISHED
            
            decode_device_to_try = request.decode_device_id - 1
            while decode_device_to_try >= 0:
                if stats[decode_device_to_try].n_requests < self.max_decode_batch_size:
                    request.decode_device_id = decode_device_to_try
                    break
                decode_device_to_try -= 1
            if decode_device_to_try == -1:
                request.update_state('fall_back', True)
                if self.fallback_policy == 'reject':
                    request.admitted = False
                    continue
            stats[request.decode_device_id].n_requests += 1

class DisaggregatedEDFRouter(Router):
    def __init__(self, n_devices: int, router_kwargs: str):
        import re
        pattern = re.compile(r'(\d+)P(\d+)D')
        match = pattern.match(router_kwargs)
        if match:
            self.n_prefill_devices = int(match.group(1))
            self.n_decode_devices = int(match.group(2))
        else:
            raise ValueError(f"Invalid router kwargs: {router_kwargs}")
        assert self.n_prefill_devices + self.n_decode_devices == n_devices
        self.decode_i = 0
        self.prefill_device_states = [DeviceState(i, 'idle', 0) for i in range(self.n_prefill_devices)]

    def run(self, waiting_requests: List[RequestInstance],
            running_requests: List[RequestInstance]):
        waiting_requests = sorted(waiting_requests, key=lambda x: x.payload['vllm_xargs']['prefill_ddl'])
        for request in waiting_requests:
            if time.time() > request.payload['vllm_xargs']['prefill_ddl']:
                request.admitted = False
                continue

            for device in self.prefill_device_states:
                if device.state == 'idle':
                    device.state = 'prefill'
                    request.admitted = True
                    request.prefill_device_id = device.id
                    request.decode_device_id = self.decode_i + self.n_prefill_devices
                    self.decode_i = (self.decode_i + 1) % self.n_decode_devices
                    break

    def update(self, request: RequestInstance, new_state: RequestState):
        if new_state == RequestState.PREFILL_REJECTED:
            self.prefill_device_states[request.prefill_device_id].state = 'idle'
        elif new_state == RequestState.PREFILL_FINISHED:
            self.prefill_device_states[request.prefill_device_id].state = 'idle'

    def update_json(self, request_json: dict, i: int):
        new_request_json = request_json.copy()
        if i < self.n_prefill_devices:
            if new_request_json['scheduling_policy'] in ['vllm-fcfs', 'vllm-priority']:
                new_request_json['scheduling_kwargs']['max_num_batched_tokens'] = 16384
                new_request_json['scheduling_kwargs']['long_prefill_token_threshold'] = 4096
        return new_request_json

@dataclass 
class EngineState:
    next_batch_time: float
    num_free_blocks: int
    num_computed_tokens: dict = field(default_factory = dict)
    batch_id: int | None = None
    staleness: float | None = None

class SLOsServeRouter(Router):
    
    def __init__(self, n_devices: int, router_kwargs: str | dict):
        import json
        if isinstance(router_kwargs, str):
            router_kwargs = json.loads(router_kwargs)
        
        self.n_devices = n_devices
        from SLOsServe.perf_model import PerfModel
        self.model_name = router_kwargs['model_name']
        self.perf_model_task = str(router_kwargs.get('perf_model_task', 'default'))
        self.perf_model_err = float(router_kwargs.get('perf_model_err', 1.0))
        perf_model = PerfModel.get_perf_model(
            self.model_name,
            self.perf_model_task,
        )
        perf_model = perf_model.copy_with_adjustments(
            scale=self.perf_model_err,
            constant_offset=router_kwargs['scheduling_overhead'],
        )
        self.perf_model = perf_model
        self.hardware_params = perf_model.describe_hardware_params()
        logger.info(
            "Loaded router perf model: model=%s task=%s type=%s breakpoints=%s perf_model_err=%s",
            self.model_name,
            self.perf_model_task,
            (
                "piecewise_current_tokens"
                if perf_model.is_piecewise_current_tokens else "linear"
            ),
            (
                self.hardware_params.get("breakpoints")
                if isinstance(self.hardware_params, dict) else None
            ),
            self.perf_model_err,
        )
        self.tpot = router_kwargs['tpot']
        self.n_block = router_kwargs['device_mem']
        # self.device_mems = [router_kwargs['device_mem'] for _ in range(n_devices)]
        self.block_size = router_kwargs['block_size']
        assert 'max_decode_length' in router_kwargs
        self.max_decode_length = int(router_kwargs['max_decode_length'])
        self.admission_max_decode_length = int(
            router_kwargs.get('admission_max_decode_length', self.max_decode_length)
        )
        self.group_size = router_kwargs.get('group_size', self.n_devices)
        self.n_lb = router_kwargs.get('n_lb', 1)
        self.use_planner = router_kwargs.get('use_planner', False)
        self.ablation = router_kwargs.get('ablation', False)
        self.oracle_mem = router_kwargs.get('oracle_mem', router_kwargs.get('is_oracle', False))
        assert self.group_size >= 1
        assert self.n_devices % self.group_size == 0 
        self.n_group = self.n_devices // self.group_size
        assert self.n_lb >= 1 and self.n_lb <= self.group_size
        self.is_pd_disagg = router_kwargs.get('is_pd_disagg', False)
        assert not self.is_pd_disagg or router_kwargs.get('enable_rerouting', False) == True 
        self.n_prefill_or_mixed_per_group = self.group_size
        if self.is_pd_disagg:
            assert 'n_prefill_per_group' in router_kwargs
            self.n_prefill_or_mixed_per_group = router_kwargs['n_prefill_per_group']
            assert 'max_decode_bs' in router_kwargs
            self.max_decode_bs = router_kwargs['max_decode_bs']
            
            assert self.n_lb <= self.n_prefill_or_mixed_per_group < self.group_size
            self.decode_group_idx = 0
        self.group_idx = 0
        self.lb_indices_per_group = [0 for _ in range(self.n_group)] 
        
        # self.routing_overhead = router_kwargs.get('routing_overhead', 0.0)
        self.pre_adm_ctrler = SLOsServe_C.AdmCtrlScheduler(
            "edf", # policy
            self.block_size,
            False, # fairness 
            False, # continuous
        )
        self.perf_model.configure_cpp_ar_planner(
            self.pre_adm_ctrler,
            tpots=[self.tpot],
            fixed_bs=False,
        )
        self.is_oracle = self.oracle_mem
        
        self.adm_planner = SLOsServe_C.AdmCtrlScheduler(
            "edf_sim", # policy
            self.block_size,
            False, # fairness 
            False, # continuous
        )
        self.perf_model.configure_cpp_ar_planner(
            self.adm_planner,
            tpots=[self.tpot],
            fixed_bs=False,
        )
        self.kv_xfer_delay = router_kwargs.get('kv_xfer_delay', 0.05)
        self.pre_adm_schedule_dump_threshold_s = router_kwargs.get("pre_adm_schedule_dump_threshold_s", 0.05)
        self.pre_adm_schedule_dump_dir = router_kwargs.get("pre_adm_schedule_dump_dir", "pre_adm_schedule_debug")
        self.pre_adm_schedule_dump_cooldown_s = router_kwargs.get("pre_adm_schedule_dump_cooldown_s", 1.0)
        self._last_pre_adm_schedule_dump_ts = 0.0
        self._pre_adm_schedule_iter = 0
        
        
        logger.info(
            f'SLOServeRouter: n_group: {self.n_group}, n_lb: {self.n_lb}, '
            f'group_size: {self.group_size}, n_prefill_or_mixed: {self.n_prefill_or_mixed_per_group}, '
            f'use_planner: {self.use_planner}, ablation: {self.ablation}, oracle_mem: {self.oracle_mem}, '
            f'max_decode_length: {self.max_decode_length}, admission_max_decode_length: {self.admission_max_decode_length}, '
            f'perf_model_err: {self.perf_model_err}, hardware_params: {self.hardware_params}, '
            f'scheduling_overhead: {router_kwargs["scheduling_overhead"]}'
        )

    def _get_decode_length_ub(self, request: RequestInstance) -> int:
        if not self.oracle_mem:
            return self.admission_max_decode_length
        extra_args = request.payload.get('vllm_xargs', {})
        return int(extra_args.get('output_length', request.payload.get('max_tokens', self.admission_max_decode_length)))
        
    def get_req_data(self, request: RequestInstance):
        extra_args = request.payload['vllm_xargs']
        prefill_ddl = extra_args['prefill_ddl']
        input_length = extra_args['input_length']
        profit = extra_args['profit']
        prefill_mem = math.ceil(input_length / self.block_size)
        mem = math.ceil((input_length + self._get_decode_length_ub(request)) / self.block_size)
        return prefill_ddl, input_length, profit, prefill_mem, mem

    def _effective_cached_tokens(
        self,
        request: RequestInstance,
        did: int,
        mode: str = 'normal',
    ) -> int:
        if mode == 'decode_only':
            return 0
        cached_tokens = min(request.cached_tokens, request.num_prompt_tokens)
        if cached_tokens <= 0:
            return 0
        session_id = request.session_id
        if not session_id:
            return 0
        home_prefill_device = self.get_session_home_prefill_device(session_id)
        if home_prefill_device != did:
            return 0
        return cached_tokens
    
    def _run_pre_adm_planner(
        self, 
        did,
        running_requests: List[RequestInstance],
        waiting_requests: List[RequestInstance], 
        engine_state: EngineState | None = None,
        mode: str = 'normal',
    ) -> List[RequestInstance]:
        now = time.time()
        if engine_state is None:
            engine_state = EngineState(next_batch_time=now,
                                       num_free_blocks=self.n_block)
        else:
            now = max(now, engine_state.next_batch_time)
        t_start = time.perf_counter()
        if not len(waiting_requests):
            return []
        
        if self.ablation:
            for req in waiting_requests:
                if mode in ('normal', 'prefill_only'):
                    req.prefill_device_id = did
                if mode in ('normal', 'decode_only'):
                    req.decode_device_id = did
                req.admitted = True
                running_requests.append(req)
            return []

        # num_computed_tokens = {}
        # num_free_blocks = self.n_block
        # if exec_plan is not None and exec_plan['exec_plan'] is not None:
        #     exec_plan_ = exec_plan['exec_plan']
        #     if exec_plan_.num_free_blocks is not None:
        #         num_free_blocks = exec_plan_.num_free_blocks
        #     bid = 0
        #     while bid < len(exec_plan_.batch_times) and exec_plan_.batch_times[bid] < now:
        #         bid += 1
        #     if bid < len(exec_plan_.batch_times):
        #         now = max(now, exec_plan_.batch_times[bid])
        #     for req_id, req_plan in exec_plan_.req_plans.items():
        #         for n_token, cbid in req_plan:
        #             if cbid <= bid:
        #                 num_computed_tokens[req_id] = n_token
        #             else:
        #                 break

        c_reqs = []
        for req in waiting_requests:
            if req.admitted:
                continue
            prefill_ddl, input_length, profit, prefill_mem, _ = self.get_req_data(req)
            if mode == 'prefill_only': prefill_ddl -= self.kv_xfer_delay
            num_computed = req.num_prompt_tokens if mode == 'decode_only' else self._effective_cached_tokens(req, did, mode)
            decode_length_ub = self._get_decode_length_ub(req)
            remaining_prompt_tokens = max(req.num_prompt_tokens - num_computed, 0)
            n_block_ub = math.ceil((remaining_prompt_tokens + decode_length_ub) / self.block_size)
            c_reqs.append(
                SLOsServe_C.Request(
                    id=req.request_id,
                    is_new_req=True,
                    ddl=prefill_ddl - now,
                    input_length=input_length,
                    n_computed_tokens=num_computed,
                    max_tokens = 1 if mode == 'prefill_only' else decode_length_ub,
                    profit=profit,
                    mem=n_block_ub,
                    tpot_idx=0,
                    prefill_mem=prefill_mem,
                    prefill_device_id=0,
                    decode_device_id=0,
                    prefill_only=(mode == 'prefill_only'),
                    arrival_time=req.arrival_time - now,
                )
            )

        for req in running_requests:
            prefill_ddl, input_length, profit, prefill_mem, _ = self.get_req_data(req)
            num_computed = req.num_computed_tokens
            if engine_state is not None:
                num_computed = max(engine_state.num_computed_tokens.get(req.request_id, 0), num_computed)
            if (self.is_pd_disagg and req.is_decode):
                num_computed = max(num_computed, req.num_prompt_tokens)
            decode_length_ub = self._get_decode_length_ub(req)
            n_block_ub = math.ceil((decode_length_ub + req.num_prompt_tokens - num_computed) / self.block_size)
            prefill_only = self.is_pd_disagg and req.is_prefill
            if prefill_only: prefill_ddl -= self.kv_xfer_delay
            c_reqs.append(
                SLOsServe_C.Request(
                    id=req.request_id,
                    is_new_req=False,
                    ddl=prefill_ddl - now,
                    input_length=input_length,
                    n_computed_tokens=num_computed,
                    max_tokens = 1 if prefill_only else decode_length_ub,
                    profit=profit,
                    mem=n_block_ub,
                    tpot_idx=0,
                    prefill_mem=prefill_mem,
                    prefill_device_id=0,
                    decode_device_id=0,
                    prefill_only=prefill_only,
                    arrival_time=req.arrival_time - now,
                )
            )
            

        t_before_schedule = time.perf_counter()
        is_feasible, is_accepteds = self.adm_planner.adm_ctrl(c_reqs, engine_state.num_free_blocks, 0.0)
        accepted_ids = set(c_req.id for c_req, is_accepted in zip(c_reqs, is_accepteds) if is_accepted)
        t_after_schedule = time.perf_counter()
        # logger.info(f'[SLOPacker] {did=} {is_feasible=}')
        if DUMP_ADM and hasattr(self, '_profile_events'):
            self._profile_events.append({
                'event_type': 'global_admission', 
                'device_id': did, 
                'timestamp': time.time(), 
                'extra_args': {
                    'is_feasible': is_feasible,
                    'now': now,
                    'state': {
                        req.id: (req.is_new_req, is_acc, req.n_computed_tokens, req.input_length, req.ddl, req.max_tokens, req.arrival_time) for req, is_acc in zip(c_reqs, is_accepteds)
                    },
                }
            })
        
        # if schedule_s > self.pre_adm_schedule_dump_threshold_s:
        
        if DUMP_SCHS:
            t_end = time.perf_counter()
            total_s = t_end - t_start
            schedule_s = t_after_schedule - t_before_schedule
            # prepare_s = t_before_schedule - t_start
            # post_s = t_end - t_after_schedule
            self._dump_pre_adm_schedule_inputs(
                did=did,
                mode=mode,
                c_reqs=c_reqs,
                num_free_blocks=engine_state.num_free_blocks,
                current_time=0.0,
                is_feasible=is_feasible,
                accepted_ids=accepted_ids,
                schedule_elapsed_s=schedule_s,
                total_elapsed_s=total_s,
                force = True
            )
        if not is_feasible:
            return waiting_requests

        accepted_set = set(accepted_ids)
        remained_waiting_requests = []
        for req in waiting_requests:
            if req.request_id in accepted_set:
                if mode in ('normal', 'prefill_only'):
                    req.prefill_device_id = did
                if mode in ('normal', 'decode_only'):
                    req.decode_device_id = did
                req.admitted = True
                running_requests.append(req)
            else:
                remained_waiting_requests.append(req)
        return remained_waiting_requests

    def _dump_pre_adm_schedule_inputs(
        self,
        did: int,
        mode: str,
        c_reqs: list[Any],
        num_free_blocks: int,
        current_time: float,
        is_feasible: bool,
        accepted_ids: list[str],
        schedule_elapsed_s: float,
        total_elapsed_s: float,
        force: bool = False,
        dump_reason: str = "slow",
    ) -> None:
        now = time.time()
        if (not force) and (now - self._last_pre_adm_schedule_dump_ts) < self.pre_adm_schedule_dump_cooldown_s:
            return
        self._last_pre_adm_schedule_dump_ts = now
        try:
            os.makedirs(self.pre_adm_schedule_dump_dir, exist_ok=True)
            filename = os.path.join(
                self.pre_adm_schedule_dump_dir,
                f"pre_adm_schedule_{self.iter}_did{did}_{mode}.txt",
            )
            with open(filename, "w") as f:
                f.write("SLOPACKER_SCHEDULE_DUMP_V1\n")
                f.write(f"timestamp {now}\n")
                f.write(f"did {did}\n")
                f.write(f"mode {json.dumps(mode)}\n")
                f.write("scheduler_mode \"edf_sim\"\n")
                f.write("scheduler_fifo_fair 0\n")
                f.write("scheduler_continuous 0\n")
                f.write("planner_type \"ar\"\n")
                f.write("planner_fixed_bs 0\n")
                f.write("planner_max_bs 16384\n")
                f.write(f"tpots {1} {self.tpot}\n")
                planner_config = self.perf_model.get_cpp_planner_config()
                if planner_config["type"] == "piecewise_current_tokens":
                    hardware_params = [
                        value
                        for segment_params in planner_config["segment_hardware_params"]
                        for value in segment_params
                    ]
                else:
                    hardware_params = planner_config["hardware_params"]
                f.write(f"hardware_params {len(hardware_params)}")
                for x in hardware_params:
                    f.write(f" {x}")
                f.write("\n")
                f.write(f"M {num_free_blocks}\n")
                f.write(f"current_time {current_time}\n")
                f.write("max_time 1.0\n")
                f.write(f"observed_is_feasible {int(is_feasible)}\n")
                f.write(f"observed_schedule_elapsed_s {schedule_elapsed_s}\n")
                f.write(f"observed_total_elapsed_s {total_elapsed_s}\n")
                f.write(f"observed_accepted_ids {len(accepted_ids)}\n")
                for req_id in accepted_ids:
                    f.write(f"accepted_id {json.dumps(str(req_id))}\n")
                f.write(f"reqs {len(c_reqs)}\n")
                for req in c_reqs:
                    f.write(
                        "req "
                        f"{json.dumps(str(getattr(req, 'id', '')))} "
                        f"{int(bool(getattr(req, 'is_new_req', False)))} "
                        f"{float(getattr(req, 'ddl', 0.0))} "
                        f"{int(getattr(req, 'input_length', 0))} "
                        f"{int(getattr(req, 'n_computed_tokens', 0))} "
                        f"{float(getattr(req, 'profit', 0.0))} "
                        f"{int(getattr(req, 'mem', 0))} "
                        f"{int(getattr(req, 'tpot_idx', 0))} "
                        f"{int(getattr(req, 'prefill_mem', 0))} "
                        f"{int(getattr(req, 'prefill_device_id', 0))} "
                        f"{int(getattr(req, 'decode_device_id', 0))} "
                        f"{int(bool(getattr(req, 'prefill_only', False)))} "
                        f"{float(getattr(req, 'arrival_time', 0.0))} "
                        f"{int(getattr(req, 'max_tokens', -1))}\n"
                    )
            if dump_reason == "sanity":
                logger.info(
                    f"[SLOPackerTiming] sanity dump at iter={self._pre_adm_schedule_iter} "
                    f"did={did} mode={mode} feasible={is_feasible} "
                    f"schedule={schedule_elapsed_s:.6f}s total={total_elapsed_s:.6f}s "
                    f"dumped schedule inputs to {filename}"
                )
            else:
                logger.warning(
                    f"[SLOPackerTiming] slow schedule ({schedule_elapsed_s:.6f}s > "
                    f"{self.pre_adm_schedule_dump_threshold_s:.3f}s), dumped schedule inputs to {filename}"
                )
        except Exception:
            logger.exception("Failed to dump slow schedule inputs")

    def get_engine_states(self):
        now = time.time()
        def _compute_engine_state(exec_plan: ExecPlan, engine_state: EngineState):
            engine_state.batch_id = exec_plan.batch_id
            if exec_plan.num_free_blocks is not None:
                engine_state.num_free_blocks = exec_plan.num_free_blocks
            bid = 0
            while bid < len(exec_plan.batch_times) and exec_plan.batch_times[bid] < now:
                bid += 1
            if bid < len(exec_plan.batch_times):
                engine_state.next_batch_time = max(engine_state.next_batch_time, exec_plan.batch_times[bid])
                engine_state.staleness = exec_plan.batch_times[bid] - now
            for req_id, req_plan in exec_plan.req_plans.items():
                for n_token, cbid in req_plan:
                    if cbid <= bid:
                        engine_state.num_computed_tokens[req_id] = n_token
                    else:
                        break
            
        global execplan_bus_actor
        assert execplan_bus_actor is not None
        raw_engine_states = ray.get(execplan_bus_actor.get_all.remote())
        engine_states = {
            device_id: EngineState(next_batch_time=now,
                                   num_free_blocks=self.n_block)
            for device_id in range(self.n_devices)
        }
        if not isinstance(raw_engine_states, dict):
            return engine_states

        for device_id, state in raw_engine_states.items():
            if device_id not in engine_states or not isinstance(state, dict):
                continue
            engine_state = engine_states[device_id]
            load_stats = state.get('load_stats')
            if isinstance(load_stats, dict):
                try:
                    engine_state.num_free_blocks = int(
                        load_stats.get('effective_num_free_blocks',
                                       load_stats.get('num_free_blocks',
                                       engine_state.num_free_blocks)))
                except (TypeError, ValueError):
                    pass
            exec_plan = state.get('exec_plan')
            if exec_plan is not None:
                _compute_engine_state(exec_plan, engine_state)
        return engine_states

    def run_with_planner(self, waiting_requests: List[RequestInstance],
            running_requests: List[RequestInstance]):
        # logger.info(f'exec_plans: {exec_plans}')
        # logger.info(f'fetch exec plans takes {elapsed_fetch_exec_plan}s')
        
        engine_states = self.get_engine_states()
        if DUMP_ADM and hasattr(self, '_profile_events'):
            self._profile_events.append({
                'event_type': 'engine_states',
                'timestamp': time.time(),
                'extra_args': {
                    k: asdict(v) for k, v in engine_states.items()
                }
            })
        
        # Step.1 set the 
        running_requests_by_device = [[] for _ in range(self.n_devices)]
        for req in running_requests:
            if self.is_pd_disagg:
                if req.is_prefill:
                    assert req.prefill_device_id >= 0
                    running_requests_by_device[req.prefill_device_id].append(req)
                else:
                    assert req.decode_device_id >= 0
                    running_requests_by_device[req.decode_device_id].append(req)
            else: 
                running_requests_by_device[req.prefill_device_id].append(req)
        
        waiting_prefill_or_normal_requests = [[] for _ in range(self.n_devices)]
        waiting_decode_requests = [[] for _ in range(self.n_devices)]
        for req in waiting_requests:
            did = None 
            if not self.is_pd_disagg or req.is_prefill:
                if req.prefill_device_id == -1:
                    home_prefill_device = self.get_session_home_prefill_device(req.session_id)
                    if home_prefill_device is not None:
                        did = home_prefill_device
                    else:
                        did = (
                            self.group_idx * self.group_size
                            + self.lb_indices_per_group[self.group_idx]
                        )
                        self.lb_indices_per_group[self.group_idx] = (
                            self.lb_indices_per_group[self.group_idx] + 1
                        ) % self.n_lb
                        self.group_idx = (self.group_idx + 1) % self.n_group
                elif (req.prefill_device_id + 1) % self.group_size: 
                    did = req.prefill_device_id + 1
                else:
                    # begin the next cycle
                    did = req.prefill_device_id // self.group_size * self.group_size
                # if did is not None: 
                waiting_prefill_or_normal_requests[did].append(req)
            else:
                assert req.prefill_device_id >= 0
                if req.decode_device_id == -1:
                    # Keep decode in the same group (node) as prefill to avoid cross-node traffic.
                    did = (req.prefill_device_id // self.group_size) * self.group_size + self.group_size - 1
                elif req.decode_device_id % self.group_size:
                    did = req.decode_device_id - 1 
                else:
                    # begin the next cycle
                    did = req.decode_device_id + self.group_size - 1
                # if did is not None: 
                waiting_decode_requests[did].append(req)
        
        for group_i in range(self.n_group):
            if self.is_pd_disagg:
                for did in range(self.group_size * (group_i + 1) - 1, self.group_size * group_i - 1, -1):
                    remained_waiting_requests = self._run_pre_adm_planner(
                        did,
                        running_requests=running_requests_by_device[did],
                        waiting_requests= waiting_decode_requests[did],
                        engine_state=engine_states.get(did, None),
                        mode = 'decode_only'
                    )
                    if did != self.group_size * group_i:
                        waiting_decode_requests[did - 1].extend(remained_waiting_requests)
            
            for did in range(self.group_size * group_i, self.group_size * (group_i + 1)):
                waiting_requests = waiting_prefill_or_normal_requests[did]
                if not len(waiting_requests):
                    continue
                remained_waiting_requests = self._run_pre_adm_planner(
                    did, 
                    running_requests=running_requests_by_device[did],
                    waiting_requests= waiting_requests,
                    engine_state=engine_states.get(did, None),
                    mode = 'normal' if not self.is_pd_disagg else 'prefill_only'
                )
                if (did + 1) != (self.group_size * (group_i + 1)):
                    waiting_prefill_or_normal_requests[did + 1].extend(remained_waiting_requests)
            

    def _run_pre_adm(
        self, 
        did,
        waiting_requests: List[RequestInstance], 
        running_requests: List[RequestInstance],
        exec_plan: ExecPlan | None = None,
        prefill_only: bool = False
    ) -> List[RequestInstance]:
        if not len(waiting_requests): return []
        assert all(req.prefill_device_id == did for req in running_requests)
        num_computed_tokens = {}
        now = time.time()
        num_free_blocks = self.n_block
        if exec_plan is not None and exec_plan['exec_plan'] is not None:
            exec_plan_ = exec_plan['exec_plan']
            num_free_blocks = exec_plan['exec_plan'].num_free_blocks
            staleness = now - exec_plan['timestamp']
            per_req_stalenesses = []
            bid = 0
            while bid < len(exec_plan_.batch_times) and exec_plan_.batch_times[bid] < now:
                bid += 1
            
            # Set now to be the first batch finished after now. (the scheduling point when requests sent to that server)
            if bid < len(exec_plan_.batch_times):
                now = exec_plan_.batch_times[bid]
                
            for req_id, req_plan in exec_plan_.req_plans.items():
                per_req_staleness = staleness
                for n_token, cbid in req_plan:
                    if cbid <= bid:
                        per_req_staleness = now - exec_plan_.batch_times[cbid]
                        num_computed_tokens[req_id] = n_token 
                    else: break                
                per_req_stalenesses.append(per_req_staleness)
            
            logger.info(f'plan staleness for device {did}: {staleness:.06f}s, on device reqs: {exec_plan["exec_plan"].req_plans.keys()}, running_reqs: {[req.request_id for req in running_requests]}, now: {now}, num_computed_tokens: {num_computed_tokens}')
            if len(per_req_stalenesses):
                logger.info(f'Per Req: mean {np.mean(per_req_stalenesses)}, max: {np.max(per_req_stalenesses)},')
        
        c_reqs = []
        runned_waiting_reqs = {}
        
        for req in waiting_requests:
            if req.admitted: continue
            prefill_ddl, input_length, profit, prefill_mem, mem = self.get_req_data(req)
            decode_length_ub = self._get_decode_length_ub(req)
            num_computed_tokens = self._effective_cached_tokens(req, did, 'prefill_only' if prefill_only else 'normal')
            remaining_prompt_tokens = max(req.num_prompt_tokens - num_computed_tokens, 0)
            n_block_ub = math.ceil((remaining_prompt_tokens + decode_length_ub) / self.block_size)
            c_req = SLOsServe_C.Request(
                id = req.request_id,
                is_new_req = True,
                ddl = prefill_ddl,
                input_length = input_length,
                n_computed_tokens = num_computed_tokens,
                profit = profit,
                mem = n_block_ub,
                tpot_idx = 0,
                prefill_mem = prefill_mem,
                prefill_device_id = 0, 
                decode_device_id = 0,
                prefill_only = prefill_only,
                arrival_time = req.arrival_time
            )
            c_reqs.append(c_req)
            runned_waiting_reqs[req.request_id] = req
        
        for req in running_requests:
            if not (req.prefill_device_id == did): continue 
            prefill_ddl, input_length, profit, prefill_mem, mem = self.get_req_data(req)
            num_computed_token = max(num_computed_tokens.get(req.request_id, 0), req.num_computed_tokens)
            num_output_tokens = max(num_computed_token - input_length, 0) + 1
            decode_length_ub = self._get_decode_length_ub(req)
            num_block_ub = math.ceil((decode_length_ub + req.num_prompt_tokens - num_computed_token) / self.block_size)
            c_req = SLOsServe_C.Request(
                id = req.request_id,
                is_new_req = False,
                ddl = prefill_ddl + self.tpot * num_output_tokens,
                input_length = max(input_length - num_computed_token, 0),
                n_computed_tokens = num_computed_token,
                profit = profit,
                mem = num_block_ub,
                tpot_idx = 0,
                prefill_mem = 0,
                prefill_device_id = 0, 
                decode_device_id = 0,
                prefill_only = prefill_only,
                arrival_time = req.arrival_time
            )
            c_reqs.append(c_req)
        
        # logger.info(f'slosserverouter: {c_reqs}, Mem: {self.device_mems[did]}, now: {now}')
        
        is_feasible, is_accepteds = self.pre_adm_ctrler.adm_ctrl(
            c_reqs, num_free_blocks, now
        )
        accpeted_ids = [req.id for req, accepted in zip(c_reqs, is_accepteds) if accepted]
        logger.info(f'slosserverouter, is_feasible: {is_feasible}, len(waiting): {len(waiting_requests)}, len(running): {len(running_requests)}')
        
        if not is_feasible: return waiting_requests 
        
        to_remove = []
        for req_id in accpeted_ids:
            if not req_id in runned_waiting_reqs: 
                continue 
            req = runned_waiting_reqs[req_id]
            to_remove.append(req)
            prefill_ddl, input_length, profit, prefill_mem, mem = self.get_req_data(req)
            req.admitted = True
            req.prefill_device_id = did
            if not prefill_only:
                req.decode_device_id = did
        
        if len(to_remove): 
            return [req for req in waiting_requests if req.request_id not in accpeted_ids]
        
        return waiting_requests

    def _run_pre_adm_decode(
        self, 
        did,
        waiting_requests: List[RequestInstance], 
        running_requests: List[RequestInstance],
        exec_plan: ExecPlan | None = None,
    ) -> List[RequestInstance]:
        if not len(waiting_requests): return []
        assert all(req.decode_device_id == did for req in running_requests)
        req2num_computed_tokens = {}
        now = time.time()
        num_free_blocks = self.n_block
        if exec_plan is not None and exec_plan['exec_plan'] is not None:
            exec_plan_ = exec_plan['exec_plan']
            num_free_blocks = exec_plan['exec_plan'].num_free_blocks
            staleness = now - exec_plan['timestamp']
            per_req_stalenesses = []
            bid = 0
            while bid < len(exec_plan_.batch_times) and exec_plan_.batch_times[bid] < now:
                bid += 1
            
            # Set now to be the first batch finished after now. (the scheduling point when requests sent to that server)
            if bid < len(exec_plan_.batch_times):
                now = exec_plan_.batch_times[bid]
                
            for req_id, req_plan in exec_plan_.req_plans.items():
                per_req_staleness = staleness
                for n_token, cbid in req_plan:
                    if cbid <= bid:
                        per_req_staleness = now - exec_plan_.batch_times[cbid]
                        req2num_computed_tokens[req_id] = n_token 
                    else: break                
                per_req_stalenesses.append(per_req_staleness)
            
            logger.info(f'plan staleness for device {did}: {staleness:.06f}s, on device reqs: {exec_plan["exec_plan"].req_plans.keys()}, running_reqs: {[req.request_id for req in running_requests]}, exec_plan: {exec_plan}, now: {now}, num_computed_tokens: {req2num_computed_tokens}')
            if len(per_req_stalenesses):
                logger.info(f'Per Req: mean {np.mean(per_req_stalenesses)}, max: {np.max(per_req_stalenesses)},')
            
        for req in running_requests:
            num_computed_tokens = max(req2num_computed_tokens.get(req.request_id, 0), req.num_computed_tokens)
            decode_length_ub = self._get_decode_length_ub(req)
            num_block_ub = math.ceil((decode_length_ub + req.num_prompt_tokens - num_computed_tokens) / self.block_size)
            num_free_blocks -= num_block_ub
            
        idx = 0
        while idx < len(waiting_requests) and ((idx + len(running_requests) + 1) <= self.max_decode_bs):
            req = waiting_requests[idx]
            decode_length_ub = self._get_decode_length_ub(req)
            num_block_ub = math.ceil((decode_length_ub + req.num_prompt_tokens) / self.block_size)
            if num_free_blocks - num_block_ub >= 0:
                num_free_blocks -= num_block_ub
                
                req.admitted = True 
                req.decode_device_id = did
                idx += 1
            else: break                
        return waiting_requests[idx:]
    
    def _run(self, waiting_requests: List[RequestInstance],
            running_requests: List[RequestInstance],
            prefill_only: bool = False
        ):
        
        assert all(req.admitted is None for req in waiting_requests)
        
        global execplan_bus_actor
        assert execplan_bus_actor is not None 
        

        start = time.time()
        exec_plans = ray.get(execplan_bus_actor.get_all.remote())
        # logger.info(f'exec_plans: {exec_plans}')
        elapsed_fetch_exec_plan = time.time() - start
        logger.info(f'elapsed fetch exec plan: {elapsed_fetch_exec_plan:.06f}s')
        
        start_routing = time.time()
        # step 1. Figure out the next device for each request
        device_to_waiting_reqs = [[] for i in range(self.n_devices)]
        for req in waiting_requests: 
            if req.prefill_device_id == -1: 
                home_prefill_device = self.get_session_home_prefill_device(req.session_id)
                if home_prefill_device is not None:
                    did = home_prefill_device
                else:
                    did = self.group_idx * self.group_size + self.lb_indices_per_group[self.group_idx]
                    self.lb_indices_per_group[self.group_idx] = (self.lb_indices_per_group[self.group_idx] + 1) % self.n_lb
                    self.group_idx = (self.group_idx + 1) % self.n_group
            else:
                if ((req.prefill_device_id % self.group_size) < self.n_lb) and (self.n_lb < self.n_prefill_or_mixed_per_group):
                    did = (req.prefill_device_id // self.group_size * self.group_size) + self.n_lb
                elif (req.prefill_device_id + 1) < ((req.prefill_device_id) // self.group_size * self.group_size + self.n_prefill_or_mixed_per_group): 
                    did = req.prefill_device_id + 1
                else: 
                    did = None
            if did is None: 
                req.admitted = False
            else:
                if not (0 <= did < len(device_to_waiting_reqs)):
                    logger.error(f'Error indexing {did=}, {req.prefill_device_id=}, {self.group_size=}, {self.n_prefill_or_mixed_per_group=}, {self.n_lb=}, {self.group_idx=}, {self.lb_indices_per_group=}')
                    raise RuntimeError
                device_to_waiting_reqs[did].append(req)
        
        device_to_running_reqs = [[] for i in range(self.n_devices)]
        for req in running_requests:
            assert req.prefill_device_id >= 0 and req.prefill_device_id < self.n_devices, f"{req.prefill_device_id=} out of bound. {self.n_devices=}"
            device_to_running_reqs[req.prefill_device_id].append(req)
                
        # step 2: for severs using load balancing, do admission check (in parallel)
        for i in range(self.n_group):
            for j in range(self.n_lb):
                did = i * self.group_size + j
                remained_waiting_reqs = self._run_pre_adm(did, waiting_requests=device_to_waiting_reqs[did], 
                                                          running_requests=device_to_running_reqs[did],
                                                          prefill_only = prefill_only,
                                  exec_plan = exec_plans.get(did, None))
                # for req in remained_waiting_reqs:
                next_did = (i * self.group_size) + self.n_lb
                if self.n_lb < self.n_prefill_or_mixed_per_group:
                    device_to_waiting_reqs[next_did].extend(remained_waiting_reqs)
        
        
        # step 3: for severs using load packing, do admission check (TODO: in parallel)
        for i in range(self.n_group):
            for j in range(self.n_lb, self.n_prefill_or_mixed_per_group):
                did = i * self.group_size + j
                remained_waiting_reqs = self._run_pre_adm(did, 
                                                          waiting_requests=device_to_waiting_reqs[did],
                                                          running_requests=device_to_running_reqs[did],
                                                          exec_plan = exec_plans.get(did, None),
                                                          prefill_only = prefill_only)
                if j + 1 < self.n_prefill_or_mixed_per_group:
                    device_to_waiting_reqs[i * self.group_size + j + 1].extend(remained_waiting_reqs)
        
        routing_elapsed = time.time() - start_routing
    
    def _run_decode(self, waiting_requests: List[RequestInstance],
            running_requests: List[RequestInstance],
        ):
            start = time.time()
            exec_plans = ray.get(execplan_bus_actor.get_all.remote())
            # logger.info(f'exec_plans: {exec_plans}')
            elapsed_fetch_exec_plan = time.time() - start
            logger.info(f'fetch exec plans takes {elapsed_fetch_exec_plan}s')
            
            device_to_running_reqs = [[] for i in range(self.n_devices)]
            for req in running_requests:
                assert 0 <= req.decode_device_id < self.n_devices, \
                    f"{req.decode_device_id=} out of bound. {self.n_devices=}"
                assert (req.decode_device_id % self.group_size) >= self.n_prefill_or_mixed_per_group, \
                    f"{req.decode_device_id=} points to prefill region. {self.group_size=}, {self.n_prefill_or_mixed_per_group=}"
                device_to_running_reqs[req.decode_device_id].append(req)
            device_to_waiting_reqs = [[] for i in range(self.n_devices)]
            for req in waiting_requests:
                if req.decode_device_id == -1:
                    group_idx = self.decode_group_idx
                    self.decode_group_idx = (self.decode_group_idx + 1) % self.n_group
                    did = self.group_size * group_idx + self.group_size - 1
                else: 
                    assert 0 <= req.decode_device_id < self.n_devices, \
                        f"{req.decode_device_id=} out of bound. {self.n_devices=}"
                    local_decode_idx = req.decode_device_id % self.group_size
                    assert local_decode_idx >= self.n_prefill_or_mixed_per_group, \
                        f"{req.decode_device_id=} points to prefill region. {self.group_size=}, {self.n_prefill_or_mixed_per_group=}"
                    if local_decode_idx > self.n_prefill_or_mixed_per_group:
                        did = req.decode_device_id - 1
                    else:
                        # Already at the lowest decode slot in this group.
                        did = None
                if did is not None:
                    if not (0 <= did < len(device_to_waiting_reqs)):
                        logger.error(f'Error indexing {did=}, {req.prefill_device_id=}, {self.group_size=}, {self.n_prefill_or_mixed_per_group=}, {self.n_lb=}, {self.group_idx=}, {self.lb_indices_per_group=}')
                        raise RuntimeError
                    device_to_waiting_reqs[did].append(req)
                
            for i in range(self.n_group):
                for j in range(self.group_size - 1, self.n_prefill_or_mixed_per_group - 1, -1):
                    did = self.group_size * i + j
                    remained_waiting_requests = self._run_pre_adm_decode(waiting_requests = device_to_waiting_reqs[did], 
                                             running_requests = device_to_running_reqs[did],
                                             exec_plan = exec_plans.get(did, None),
                                             did = did
                                             )
                    if j > self.n_prefill_or_mixed_per_group: 
                        device_to_waiting_reqs[did - 1].extend(remained_waiting_requests)
    
    def run(self, waiting_requests: List[RequestInstance],
            running_requests: List[RequestInstance]):
        if self.use_planner:
            start = time.time()
            self.run_with_planner(waiting_requests, running_requests)
            # logger.info(f'[SLOPacker] routing takes {time.time() - start}s')
            return
        if not self.is_pd_disagg:
            self._run(waiting_requests, running_requests)
            return
        
        prefill_waiting_requests = [req for req in waiting_requests if req.state < RequestState.PREFILL_FINISHED]
        prefill_running_requests = [req for req in running_requests if req.state < RequestState.PREFILL_FINISHED]
        decode_waiting_requests = [req for req in waiting_requests if req.state >= RequestState.PREFILL_FINISHED]
        decode_running_requests = [req for req in running_requests if req.state >= RequestState.PREFILL_FINISHED]
        assert all(req.prefill_device_id >= 0 for req in prefill_running_requests)
        assert all(req.decode_device_id >= 0 for req in decode_running_requests)
        self._run(prefill_waiting_requests, prefill_running_requests, prefill_only = True)
        self._run_decode(decode_waiting_requests, decode_running_requests)

    def update(self, request: RequestInstance, new_state: RequestState):
        pass 
        # prefill_ddl, input_length, profit, prefill_mem, mem = self.get_req_data(request)
        # assert request.prefill_device_id == request.decode_device_id, "not implemented for PD scheduling in SLOsServe"
        # if new_state in [RequestState.WAITING, RequestState.DECODE_FINISHED]:
        #     self.device_mems[request.prefill_device_id] += mem
        # else: 
        #     raise NotImplementedError(f"state update for {new_state} is not impled")

    def select_asap_server(
        self,
        load_stat,
        request: RequestInstance | None = None,
    ):
        if not self.is_pd_disagg:
            return super().select_asap_server(load_stat, request=request)

        if request is not None:
            if request.prefill_device_id >= 0:
                prefill_device_id = request.prefill_device_id
            else:
                prefill_device_id = 0
            if request.decode_device_id >= 0:
                decode_device_id = request.decode_device_id
            else:
                group_start = (prefill_device_id // self.group_size) * self.group_size
                decode_device_id = group_start + self.group_size - 1
            return prefill_device_id, decode_device_id

        n_reqs = [stat.n_requests for stat in load_stat.get_stat()]
        is_decode, n_req, freest_prefill_device_id = min(((did % self.group_size) >= self.n_prefill_or_mixed_per_group,n_req,did) for did, n_req in enumerate(n_reqs))
        assert not is_decode
        is_prefill, n_req, freest_decode_device_id = min(((did % self.group_size) < self.n_prefill_or_mixed_per_group,n_req,did) for did, n_req in enumerate(n_reqs))
        assert not is_prefill
        return freest_prefill_device_id, freest_decode_device_id
        
        
@dataclass
class DeviceState:
    id: int
    state: str
    n_decode_reqs: int
    
class RenamingRouter(Router):
    def __init__(self, n_devices: int, router_kwargs: str):
        self.n_devices = n_devices
        self.devices = [DeviceState(i, 'idle', 0) for i in range(n_devices)]
        if isinstance(router_kwargs, str):
            router_kwargs = json.loads(router_kwargs)
        self.max_decode_batch_size = router_kwargs['max_decode_batch_size']

    def run(self, waiting_requests: List[RequestInstance],
            running_requests: List[RequestInstance]):
        waiting_requests = sorted(waiting_requests, key=lambda x: x.payload['vllm_xargs']['prefill_ddl'])
        for request in waiting_requests:
            for device in self.devices:
                if device.state == 'idle':
                    device.state = 'prefill'
                    request.admitted = True
                    request.prefill_device_id = request.decode_device_id = device.id
                    break

    def update(self, request: RequestInstance, new_state: RequestState):
        logger.info(f"Renaming Router:[update]: {self.devices}, {request}, {new_state}")
        if new_state == RequestState.PREFILL_REJECTED:
            self.devices[request.prefill_device_id].state = 'idle'
        elif new_state == RequestState.PREFILL_FINISHED:
            prefill_device = self.devices[request.prefill_device_id]
            decode_device = None
            for device in self.devices:
                if device.state == 'decode' and device.n_decode_reqs < self.max_decode_batch_size:
                    request.decode_device_id = device.id
                    device.n_decode_reqs += 1
                    decode_device = device
                    break
            if decode_device is not None:
                assert decode_device.id != request.prefill_device_id
                prefill_device.state = 'idle'
            else:
                request.decode_device_id = request.prefill_device_id
                prefill_device.state = 'decode'
                prefill_device.n_decode_reqs += 1
        elif new_state == RequestState.DECODE_FINISHED:
            device = self.devices[request.decode_device_id]
            device.n_decode_reqs -= 1
            if device.n_decode_reqs == 0:
                device.state = 'idle'

def create_router(t: str, n_devices: int, router_kwargs: str | dict[str, Any]) -> Router:
    logger.info(f"Creating router of type {t} with {n_devices} devices and kwargs: {router_kwargs}")
    if isinstance(router_kwargs, str):
        router_kwargs = json.loads(router_kwargs)
    if t == 'round_robin':
        return RoundRobinRouter(n_devices, router_kwargs)
    elif t == 'round_robin_session' or t.startswith('round_robin_session-'):
        return RoundRobinRouter(
            n_devices,
            dict(router_kwargs) | {'sticky_sessions': True},
        )
    elif 'disaggregated-edf' in t:
        return DisaggregatedEDFRouter(n_devices, router_kwargs)
    elif 'disaggregated' in t:
        return DisaggregatedRouter(n_devices, router_kwargs)
    elif 'slosserve' in t:
        return SLOsServeRouter(n_devices, router_kwargs)
    elif t == 'renaming':
        return RenamingRouter(n_devices, router_kwargs)
    elif 'disagg_auto_scaling' in t:
        return DisaggregatedLCRouter(n_devices, router_kwargs)
    elif 'auto_scaling' in t:
        return AutoScalingRouter(n_devices, router_kwargs)
    elif t == 'round_robin_retry':
        return AutoScalingRouter(n_devices, router_kwargs)        
    elif 'lightest_first' in t:
        return LightestFirstRouter(n_devices, router_kwargs)
    elif t == 'llumnix_load':
        return LlumnixLoadRouter(n_devices, router_kwargs)
    else:
        raise ValueError(f"Invalid router type: {t}")

# =========================
# Simple NVLink model for mock connector
# =========================
class Network:
    def __init__(self, bw_per_link_Bps=300e9,  # bytes/s per NVLink lane; adjust to your fabric
                 inj_cap_Bps=300e9,           # per-GPU injection cap (A100 NVSwitch ~300 GB/s)
                 ej_cap_Bps=300e9,            # per-GPU ejection cap
                 overhead_s=0.010,
                 n_devices=8,
                 model_name: str = 'Qwen/Qwen2.5-7B-Instruct'):
        self.bw_per_link = bw_per_link_Bps
        self.inj_cap = inj_cap_Bps
        self.ej_cap = ej_cap_Bps
        self.overhead = overhead_s
        self.n = n_devices
        # Next-free time for each resource
        self.link_free = {(i, j): 0.0 for i in range(n_devices) for j in range(n_devices) if i != j}
        self.inj_free = [0.0] * n_devices
        self.ej_free = [0.0] * n_devices
        # simple counters for concurrent use on ports (for naive rate sharing)
        self.inj_load = [0] * n_devices
        self.ej_load = [0] * n_devices
        self.per_token_bytes = {
            'Qwen/Qwen2.5-7B-Instruct': 53.4,
            'facebook/opt-125m': 34.329,
            # 'google/gemma-3-27b-it': 106.8,
        }.get(model_name, 53.4) * 1024

    def reset(self, n_devices=None):
        if n_devices is not None:
            self.n = n_devices
        self.link_free = {(i, j): 0.0 for i in range(self.n) for j in range(self.n) if i != j}
        self.inj_free = [0.0] * self.n
        self.ej_free = [0.0] * self.n
        self.inj_load = [0] * self.n
        self.ej_load = [0] * self.n

    def tx(self, src: int, dst: int, num_tokens: int, start_time: float) -> float:
        if src == dst: return start_time
        # Earliest time all three resources are free
        t0 = max(start_time, self.link_free[(src, dst)], self.inj_free[src], self.ej_free[dst])

        # Naive simultaneous-sharing model
        inj_rate = self.inj_cap / max(1, self.inj_load[src])
        ej_rate = self.ej_cap / max(1, self.ej_load[dst])
        path_rate = min(self.bw_per_link, inj_rate, ej_rate)

        tx_time = self.overhead + num_tokens * self.per_token_bytes / path_rate

        t_done = t0 + tx_time
        # Update resource availability
        self.link_free[(src, dst)] = t_done
        self.inj_free[src] = t_done
        self.ej_free[dst] = t_done
        return t_done

# =========================
# Auto Scaler
# =========================
@dataclass
class AutoScaler: 
    window_size: float # the time window for sampling  
    scale_up_policy: Any  # the 
    scale_down_policy: Any #  
    setup_cost: int # the setup cost in seconds 
    
# =========================
# Request Pool
# =========================
class RequestPool:
    def __init__(self, window_size: float,
                 router: Router,
                 clients: list = None,
                 auto_scaler: Optional[AutoScaler] = None, 
                 enable_rerouting: bool = False,
                 enable_rescheduling: bool = False,
                 admission_mode: str = 'anytime',
                 mock_connector: bool = False,
                 load_stat: LoadStat = None,
                 stat_window: float = 1,
                 fallback_policy: str = 'asap'):
        self.waiting_pool: List[RequestInstance] = []
        self.running_pool: List[RequestInstance] = []
        self.changed_requests: set[RequestInstance] = set()
        self.window_size = window_size
        self.router = router
        self.clients = clients
        self.n_devices = len(self.clients)
        self.auto_scaler = auto_scaler 
        assert isinstance(self.clients, list)
        self.request_id = 0
        self._profile_events: List[Dict[str, Any]] = []
        # self.routing_overhead = 0
        self.admission_mode = admission_mode
        self.enable_rerouting = enable_rerouting
        self.enable_rescheduling = enable_rescheduling
        self.mock_connector = mock_connector
        if mock_connector:
            self.network = Network(n_devices=len(clients))
        if isinstance(router, RenamingRouter):
            self.enable_rerouting = True
        self.running_tasks = []
        self.load_stat = load_stat
        router.set_load_stat(load_stat)
        self.stat_window = stat_window
        self.admission_history: list[dict[str, Any]] = []
        self.routing_overhead: float = -1.0
        self.fallback_policy = fallback_policy
        self.kv_xfer_delay = 0.05
        self.maximum_ingress = 20 
        self.sem = MaxCapSemaphore(self.maximum_ingress)
        
    
    @property
    def on(self):
        return (len(self.waiting_pool) + len(self.running_pool)) > 0
    
    async def empty(self):
        while len(self.waiting_pool) > 0 or len(self.running_pool) > 0:
            await asyncio.sleep(0.1)

    def update_config(self, request_json: dict):
        self.reset(request_json['n_devices'])
        global engine_actors
        for actor in engine_actors:
            actor._profile_events = self._profile_events

        self.router = create_router(request_json['routing_policy'],
                                    request_json['n_devices'],
                                    request_json['routing_kwargs'])
        self.router.set_load_stat(self.load_stat)
        self.router.set_profile_events(self._profile_events)
        # self.admission_mode = request_json.get('admission_mode', 'arrival')
        self.routing_overhead = request_json.get('routing_overhead', -1.0)
        self.fallback_policy = request_json.get('routing_fallback_policy', 'asap')
        assert self.fallback_policy in ['asap', 'reject'], \
            f"Unsupported routing_fallback_policy={self.fallback_policy}"
        # try:
        if isinstance(request_json['routing_kwargs'], dict):
            self.routing_overhead = request_json['routing_kwargs'].get('routing_overhead', self.routing_overhead)
            self.enable_rerouting = request_json['routing_kwargs'].get('enable_rerouting', False)
            self.enable_rescheduling = request_json['routing_kwargs'].get('enable_rescheduling', False)
            self.stat_window = request_json['routing_kwargs'].get('stat_window', self.stat_window)
            self.kv_xfer_delay = request_json['routing_kwargs'].get('kv_xfer_delay', 0.05)
        # except Exception:
        #     self.routing_overhead = 0
        if request_json['routing_policy'] == 'renaming':
            self.enable_rerouting = True
        
        logger.info(f"RequestPool:[update_config]: {request_json['routing_kwargs']} {request_json['routing_policy']}, Enable Rerouting: {self.enable_rerouting}, Enable Rescheduling: {self.enable_rescheduling}, Route Ovd. {self.routing_overhead}, kv_xfer_delay: {self.kv_xfer_delay}.")

    def reset(self, n_devices):
        assert n_devices <= len(self.clients), f'{n_devices=} <= {len(self.clients)=}'
        self.n_devices = n_devices
        self.waiting_pool = []
        self.running_pool = []
        self.changed_requests = set()
        self.request_id = 0
        self._profile_events = []
        self.enable_rerouting = False
        self.admission_mode = 'anytime'
        self.running_tasks = []
        if self.mock_connector:
            self.network.reset(self.n_devices)
        self.load_stat.reset(self.n_devices)
        self.admission_history: list[dict[str, Any]] = []
        self.routing_overhead = -1.0
        if execplan_bus_actor is not None:
            execplan_bus_actor.reset.remote()

    def add_request_json(self, request_json: dict) -> AsyncGenerator[bytes, Any]:
        # print('add_request_json', request_json)
        assert 'slo_tpot' in request_json['vllm_xargs']
        response_queue = Queue()
        current_time = time.time()
        request_id = request_json['vllm_xargs'].get('request_id', str(uuid.uuid4()))
        request_json['vllm_xargs']['cached_tokens'] = int(request_json['vllm_xargs'].get('cached_tokens', 0) or 0)
        request_json['vllm_xargs']['router_arrival_time'] = current_time
        request_json['vllm_xargs']['service_tier'] = str(
            request_json['vllm_xargs'].get('service_tier', 'default')
        )
        request_json['vllm_xargs']['initial_service_tier'] = str(
            request_json['vllm_xargs'].get(
                'initial_service_tier',
                request_json['vllm_xargs']['service_tier'],
            )
        )
        if 'prefill_ddl' not in request_json['vllm_xargs']:
            request_json['vllm_xargs']['prefill_ddl'] = current_time + request_json['vllm_xargs']['slo_ttft']
        request_instance = RequestInstance(request_id, request_json, response_queue, current_time)
        self.request_id += 1
        self.waiting_pool.append(request_instance)


        self._profile_events.append({
            "event_type": "arrival-router",
            "zero_load_ttft": request_json['vllm_xargs'].get('zero_load_ttft', 0),
            "device_id": -1,
            "timestamp": current_time,
            "request_id": request_id,
            "prefill_ddl": request_json['vllm_xargs'].get('prefill_ddl', 0),
            "profit": request_json['vllm_xargs'].get('profit', 1),
            "prompt_tokens": request_json['vllm_xargs'].get('input_length', 0),
            "num_cached_tokens": request_json['vllm_xargs'].get('cached_tokens', 0),
            "max_tokens": request_json['max_tokens'],
            "service_tier": request_json['vllm_xargs'].get('service_tier', 'default'),
            "initial_service_tier": request_json['vllm_xargs'].get(
                'initial_service_tier',
                request_json['vllm_xargs'].get('service_tier', 'default'),
            ),
            "extra_args": {
                "session_id": request_json['vllm_xargs'].get('session_id'),
            },
        })

        async def gen():
            while True:
                chunk = await response_queue.get()
                if chunk is None:
                    yield b"[done]\n"
                    break
                yield b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n"
        return gen

    async def add_request(self, request: Request) -> StreamingResponse:
        start = time.time()
        # async with self.parse_json_sem:
        request_json = await request.json()
        after_request_json = time.time()
        await self.sem.acquire()
        after_sem_time = time.time()
        gen = self.add_request_json(request_json)()
        now = time.time()
        self._profile_events.append({
            'event_type': 'add_req_req_pool',
            'request_id': request_json['vllm_xargs']['request_id'],
            'timestamp': now,
            'device_id': -1,
            'extra_args': {
                'parsing_time': after_request_json - start, 
                'sem_time': after_sem_time - after_request_json,
                'add_req_time': now - after_sem_time
            }
        })
        return StreamingResponse(gen, media_type="text/event-stream")

    def update_req_state(self, request: RequestInstance, state: RequestState):
        self.router.update(request, state)
        self.router.note_request_state(request, state)
        request.state = state
        self.changed_requests.add(request)

    def _emit_rescheduling_event(
        self,
        *,
        request_id: str,
        prefill_device_id: int,
        decode_device_id: int,
    ) -> None:
        self._profile_events.append({
            "event_type": "rescheduling",
            "timestamp": time.time(),
            "request_id": request_id,
            "prefill_device_id": prefill_device_id,
            "decode_device_id": decode_device_id,
            "device_id": -1,
        })

    async def _emit_request_error(self, request: RequestInstance, reason: str, exc: Exception | None = None):
        error_type = type(exc).__name__ if exc is not None else "RuntimeError"
        error_message = f"{error_type}: {exc}" if exc is not None else reason
        logger.error(f"Request {request.request_id} failed in {reason}: {error_message}")
        self._profile_events.append({
            "event_type": "finish",
            "timestamp": time.time(),
            "device_id": -1,
            "request_id": request.request_id,
            "finish_reason": "error",
            "error": error_message,
        })
        if request.state != RequestState.TIMEOUT:
            try:
                self.update_req_state(request, RequestState.TIMEOUT)
            except Exception:
                request.state = RequestState.TIMEOUT
                self.changed_requests.add(request)
        try:
            await request.response_queue.put({
                "finish_reason": "error",
                "error_type": error_type,
                "error": error_message,
            })
            await request.response_queue.put(None)
        except Exception as queue_error:
            logger.error(f"Failed to emit error chunk for request {request.request_id}: {queue_error}")

    async def dispatch_request_safe(self, request: RequestInstance):
        try:
            await self.dispatch_request(request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                f"dispatch_request crashed for request {request.request_id}: {exc}\n{traceback.format_exc()}"
            )
            await self._emit_request_error(request, "dispatch_request", exc)

    async def fail_all_active_requests(self, reason: str, exc: Exception | None = None):
        requests = []
        seen = set()
        for request in self.waiting_pool + self.running_pool:
            if id(request) in seen:
                continue
            seen.add(id(request))
            requests.append(request)
        for request in requests:
            await self._emit_request_error(request, reason, exc)
        self.waiting_pool = []
        self.running_pool = []

    async def run_dummy_request(self, input_length: int,
                                output_length: int,
                                src_device_id: int,
                                dst_device_id: int,
                                model_name: str):
        request_id = str(uuid.uuid4())
        request_json = {
            'model': model_name,
            'max_tokens': output_length,
            'prompt': [random.randint(1000, 2000) for _ in range(input_length)],
            'stream': True,
            'ignore_eos': True,
            'vllm_xargs': {
                'prefill_ddl': time.time() + 100,
                'input_length': input_length,
                'output_length': output_length,
                'profit': 1,
                'slo_ttft': 100
            }
        }
        admitted, response, rejection_reason = await send_request_to_service_engine(
            src_device_id,
            request_json,
            request_id,
        )
        if not admitted or response is None:
            raise RuntimeError(
                f"Dummy prefill request rejected: {request_id} reason={rejection_reason}"
            )
        first_token_time = time.time()
        if 'kv_transfer_params' in response:
            request_json['kv_transfer_params'] = response['kv_transfer_params']
        first_decode_response_time = None

        admitted, generator, rejection_reason = await stream_service_response_engine(
            dst_device_id,
            request_json,
            request_id,
        )
        if not admitted or generator is None:
            raise RuntimeError(
                f"Dummy decode request rejected: {request_id} reason={rejection_reason}"
            )

        async for data in generator:
            if first_decode_response_time is None:
                first_decode_response_time = time.time()

        return first_decode_response_time - first_token_time

    async def benchmark(self):
        from tqdm.asyncio import tqdm_asyncio
        from tqdm import tqdm

        async def benchmark_pair(src_device_id: int, dst_device_id: int):
            # Warmup
            for _ in tqdm(range(10), desc=f"Warmup [{src_device_id}->{dst_device_id}]"):
                await self.run_dummy_request(1000, 10, src_device_id, dst_device_id, "")
            latencies = []
            input_lens = [100, 200, 400, 800] + list(range(1600, 32001, 1600))
            for input_length in tqdm(input_lens, desc=f"Bench Input Lens [{src_device_id}->{dst_device_id}]"):
                for _ in tqdm(range(3), desc=f"Reps @ {input_length} [{src_device_id}->{dst_device_id}]", leave=False):
                    start_time = time.time()
                    await self.run_dummy_request(input_length, 10, src_device_id, dst_device_id, "")
                    elapsed = time.time() - start_time
                    latencies.append((input_length, elapsed))
            return latencies

        all_results = []
        n_clients = len(self.clients)
        outer = tqdm(range(n_clients), desc="Src Device")
        for src_device_id in outer:
            inner = tqdm(range(n_clients), desc=f"Dst Device for Src {src_device_id}", leave=False)
            for dst_device_id in inner:
                if src_device_id == dst_device_id:
                    continue
                latencies = await benchmark_pair(src_device_id, dst_device_id)
                all_results.extend([(src_device_id, dst_device_id, input_length, latency) for input_length, latency in latencies])
        import pandas as pd
        df = pd.DataFrame(all_results, columns=['src_device_id', 'dst_device_id', 'input_length', 'latency'])
        df.to_csv('network_latency.csv', index=False)
        return df

    async def warmup(self, request_json: dict):
        start_time = time.time()
        logger.info(f"Prewarming request: {request_json}")

        for i in range(len(self.clients)):
            for j in range(len(self.clients)):
                if i == j:
                    continue
                await self.run_dummy_request(100, 10, i, j, request_json['model'])
        logger.info(f"Prewarming request: {request_json} took {time.time() - start_time} seconds")

    async def sync(self, timeout: float | None = None):
        start_time = time.time()
        while len(self.waiting_pool):
            await asyncio.sleep(1)
            if timeout is not None and time.time() - start_time > timeout:
                logger.warning(f"sync: Timed out after {timeout} seconds waiting for requests to finish")
                break
        remained_time = timeout - (time.time() - start_time) if timeout is not None else None
        if remained_time is not None and remained_time <= 0:
            logger.warning(f"sync: Timed out after {timeout} seconds waiting for requests to finish")
            return
        corotines = [task for _, task, request in self.running_tasks if not request.is_finished()]
        if not corotines:
            return
        try:
            await asyncio.wait_for(asyncio.gather(*corotines), timeout=remained_time)
        except asyncio.TimeoutError:
            logger.warning(f"sync: Timed out after {remained_time} seconds waiting for tasks")

    async def get_load_statistics(self) -> list[dict[str, Any]]:
        engine_states = await self._get_engine_states()
        load_statistics = extract_load_statistics(engine_states, self.n_devices)
        if (
            load_statistics is not None
            and self._engine_state_load_stats_are_usable(
                engine_states,
                load_statistics,
            )
        ):
            return load_statistics

        return await self._get_load_statistics_from_engines()

    def _engine_state_load_stats_are_usable(
        self,
        engine_states: dict[int, dict[str, Any]] | None,
        load_statistics: list[dict[str, Any]],
    ) -> bool:
        if not isinstance(engine_states, dict):
            return False
        if len(load_statistics) != self.n_devices:
            return False

        default_stats = make_default_load_stats()
        has_non_default = False
        now = time.time()
        max_age_s = max(1.0, 2.0 * float(self.stat_window))

        for device_id in range(self.n_devices):
            state = engine_states.get(device_id)
            if not isinstance(state, dict):
                return False
            try:
                timestamp = float(state.get("timestamp", 0.0) or 0.0)
            except (TypeError, ValueError):
                return False
            if timestamp <= 0.0 or now - timestamp > max_age_s:
                return False

            stats = load_statistics[device_id]
            if not isinstance(stats, dict):
                return False
            if any(
                int(stats.get(key, default_value)) != int(default_value)
                for key, default_value in default_stats.items()
            ):
                has_non_default = True

        return has_non_default

    async def _get_engine_states(self) -> dict[int, dict[str, Any]]:
        global execplan_bus_actor
        if execplan_bus_actor is None:
            return {}
        return await execplan_bus_actor.get_all.remote()

    async def _get_load_statistics_from_engines(self) -> list[dict[str, Any]]:
        assert engine_actors is not None

        refs = [
            engine_actors[i].engine_actor.get_load_statistics.remote(self.stat_window)
            for i in range(self.n_devices)
        ]
        stats = list(await asyncio.gather(*refs))
        return stats

    def _route_best_effort_requests(
        self,
        waiting_requests: list[RequestInstance],
    ) -> None:
        for request in waiting_requests:
            if request.admitted is True:
                continue
            prefill_device_id, decode_device_id = self.router.select_asap_server(
                self.load_stat,
                request=None,
            )
            request.prefill_device_id = prefill_device_id
            request.decode_device_id = decode_device_id
            request.payload['vllm_xargs']['must_admit'] = True
            request.payload['vllm_xargs']['service_tier'] = 'best_effort'
            request.admitted = True

    def _get_regular_running_requests(self) -> list[RequestInstance]:
        return [
            request for request in self.running_pool
            if request.service_tier != 'best_effort'
        ]

    async def routing_loop(self):
        next_load_statistics_time = time.time()
        next_auto_scaling_time = time.time()
        get_load_statistics_task = None 
        load_statistics = await self.get_load_statistics()
        self.load_stat.update(load_statistics)
        routing_iter = 0
        while True:
            if self.auto_scaler is not None: 
                if time.time() >= next_auto_scaling_time:
                    next_auto_scaling_time += self.auto_scaler.window_size
                
            
            if not self.on: 
                await asyncio.sleep(self.window_size)
                continue
            
            
            it_start_time = time.time()
            await asyncio.sleep(self.window_size)
            waiting_time = time.time() - it_start_time            
            await self.sem.reset()
            
            
            if len(self.waiting_pool): 
                routing_iter += 1
                self.router.set_iter(routing_iter)
                regular_waiting_requests = [
                    request for request in self.waiting_pool
                    if request.service_tier != 'best_effort'
                ]
                best_effort_waiting_requests = [
                    request for request in self.waiting_pool
                    if request.service_tier == 'best_effort'
                ]
                if regular_waiting_requests:
                    self.router.run(
                        regular_waiting_requests,
                        self._get_regular_running_requests(),
                    )
                if best_effort_waiting_requests:
                    self._route_best_effort_requests(best_effort_waiting_requests)
            routing_elapsed = time.time() - it_start_time

            if len(self.waiting_pool) > 0:
                self._profile_events.append({
                    "event_type": "routing",
                    "timestamp": time.time(),
                    "routing_overhead": routing_elapsed,
                    "schedules": {
                        request.request_id: {
                            "prefill_device_id": request.prefill_device_id,
                            "decode_device_id": request.decode_device_id,
                            "admitted": request.admitted,
                        } for request in self.waiting_pool
                    },
                    "device_id": -1,
                    "extra_args": {
                        'waiting_time': waiting_time, 
                        'n_waiting': len(self.waiting_pool),
                        'n_running': len(self.running_pool), 
                        'routing_iter': routing_iter 
                    }
                })
                if routing_iter % 100 == 0:
                    logger.info(f"Routing loop: routing_iter={routing_iter}, load_statistics={self.load_stat.stats}")
            
            to_logging = time.time() - it_start_time

            if time.time() >= next_load_statistics_time:
                next_load_statistics_time = time.time() + self.stat_window
                if get_load_statistics_task is None:
                    get_load_statistics_task = asyncio.create_task(
                        self.get_load_statistics(),
                        name="RequestPool.get_load_statistics",
                    )
                    get_load_statistics_task.add_done_callback(_task_done)

            to_launch_stats = time.time() - it_start_time 

            if get_load_statistics_task is not None and get_load_statistics_task.done():
                load_statistics = get_load_statistics_task.result()
                get_load_statistics_task = None
                self.load_stat.update(load_statistics)
            
            to_get_stats = time.time() - it_start_time
                
            
            remained_waiting_requests = []
            for request in self.waiting_pool:
                prefill_ddl = request.payload['vllm_xargs']['prefill_ddl']
                now = time.time()
                is_best_effort = request.service_tier == 'best_effort'
                overdue = (
                    False if is_best_effort else
                    now > (
                        prefill_ddl
                        - (self.kv_xfer_delay if (request.is_prefill and self.enable_rerouting) else 0)
                    )
                )
                if (not is_best_effort) and self.routing_overhead < 0 and (
                    (request.state in (RequestState.WAITING, RequestState.PREFILL_REJECTED_WAITING) and overdue ) or 
                    (request.state == RequestState.PREFILL_FINISHED and overdue)
                ):
                    request.admitted = False
                if not request.admitted:
                    if self.admission_mode == 'anytime' and \
                            (request.admitted is None) and \
                            (not overdue or self.routing_overhead >= 0.0): # when routing overhead > 0.0, we guarantee the acceptance of this request
                        self._profile_events.append({
                            "event_type": "temporal_rej",
                            "timestamp": now,
                            "device_id": -1,
                            "request_id": request.request_id,
                            "extra_args": {
                                "routing_overhead": routing_elapsed,
                                "waiting_time": waiting_time,
                                "window_size": self.window_size,
                                "to_logging": to_logging,
                                "to_launch": to_launch_stats,
                                "to_get_stats": to_get_stats,
                                "to_prefill_ddl": now - prefill_ddl, 
                                "request.admitted": request.admitted,
                                "state": str(request.state),
                                "admission_mode": self.admission_mode,
                                "prefill_id": request.prefill_device_id,
                                "decode_id": request.decode_device_id,
                                "routing_iter": routing_iter
                            }
                        })
                        remained_waiting_requests.append(request)
                        continue

                    if self.fallback_policy == 'reject':
                        self._profile_events.append({
                            "event_type": "finish",
                            "timestamp": now,
                            "device_id": -1,
                            "request_id": request.request_id,
                            "finish_reason": "router_rejection",
                            "extra_args": {
                                "routing_overhead": routing_elapsed,
                                "waiting_time": waiting_time,
                                "window_size": self.window_size,
                                "to_logging": to_logging,
                                "to_launch": to_launch_stats,
                                "to_get_stats": to_get_stats,
                                "to_prefill_ddl": now - prefill_ddl, 
                                "request.admitted": request.admitted,
                                "state": str(request.state),
                                "admission_mode": self.admission_mode,
                                "prefill_id": request.prefill_device_id,
                                "decode_id": request.decode_device_id,
                                'routing_iter': routing_iter
                            }
                        })
                        
                        self.load_stat.add_event({
                            'type': 'finish',
                            'timestamp': now,
                            'device_id': request.prefill_device_id,
                            'request_id': request.request_id,
                            'is_rejection': True, 
                            'is_slo_violation': True,
                            'service_tier': request.service_tier,
                        })

                        request.rejection_reason = "ROUTER"
                        logger.debug(
                            "Request %s rejected by router fallback policy=%s",
                            request.request_id,
                            self.fallback_policy,
                        )
                        await request.response_queue.put(
                            _build_rejection_response(request.rejection_reason)
                        )
                        await request.response_queue.put(None)
                        continue
                    elif self.fallback_policy == 'asap': 
                        # When this request cannot be served w/in SLO, we keep it
                        # on its last placement when available; otherwise route it
                        # to the top of the packing chain and force admission.
                        logger.info(f'[SLO Packer] route {request.request_id} asap from ({request.prefill_device_id=}, {request.decode_device_id=}) ')
                        request.prefill_device_id, request.decode_device_id = self.router.select_asap_server(self.load_stat, request=request)
                        logger.info(f'[SLO Packer] route {request.request_id} asap to ({request.prefill_device_id=}, {request.decode_device_id=}) ')
                        assert 0 <= request.prefill_device_id < self.n_devices, f'not 0 <= {request.prefill_device_id=} < {self.n_devices=}'
                        assert 0 <= request.decode_device_id < self.n_devices, f'not 0 <= {request.decode_device_id=} < {self.n_devices=}'
                        request.payload['vllm_xargs']['must_admit'] = True
                        request.payload['vllm_xargs']['service_tier'] = 'best_effort'
                        request.admitted = True
                        
                logger.debug(f"Request {request.request_id} admitted, prefill_device_id={request.prefill_device_id}, decode_device_id={request.decode_device_id}")
                self._profile_events.append({
                    "event_type": "router_decision",
                    "timestamp": time.time(),
                    "device_id": -1,
                    "request_id": request.request_id,
                    "prefill_device_id": request.prefill_device_id,
                    "decode_device_id": request.decode_device_id,
                    "extra_args": {'routing_iter': routing_iter}
                })
                assert request.prefill_device_id is not None
                assert request.decode_device_id is not None
                self.running_pool.append(request)
                task = asyncio.create_task(
                    self.dispatch_request_safe(request),
                    name=f"RequestPool.dispatch_request_safe[{request.request_id}]",
                )
                task.add_done_callback(_task_done)
                self.running_tasks.append((time.time(), task, request))

            self.waiting_pool = remained_waiting_requests

            finished_requests = []

            for request in self.changed_requests:
                logger.debug(f"Request {request.request_id} state changed to {request.state}")
                if request.state in [RequestState.PREFILL_REJECTED,
                                     RequestState.DECODE_REJECTED,
                                     RequestState.DECODE_FINISHED,
                                     RequestState.TIMEOUT]:
                    finished_requests.append(request)

            self.changed_requests.clear()

            if finished_requests:
                self.running_pool = [request for request in self.running_pool if request not in finished_requests]

            remained_running_tasks = []
            for start_time, task, request in self.running_tasks:
                if task.done():
                    if task.cancelled():
                        logger.warning(f"dispatch task for request {request.request_id} was cancelled")
                    else:
                        exc = task.exception()
                        if exc is not None:
                            logger.error(f"dispatch task failed for request {request.request_id}: {exc!r}")
                    continue
                remained_running_tasks.append((start_time, task, request))
            self.running_tasks = remained_running_tasks

    async def dispatch_request(self, request: RequestInstance):
        assert request.admitted
        assert request.prefill_device_id is not None
        assert self.enable_rerouting or request.decode_device_id is not None
        # logger.info(f"Dispatching request {request.request_id}: prefill_device_id={request.prefill_device_id}, decode_device_id={request.decode_device_id}")
        # Single-device fast path
        if not self.enable_rerouting and request.prefill_device_id == request.decode_device_id:
            attempted_prefill_device_id = request.prefill_device_id
            attempted_decode_device_id = request.decode_device_id
            self.load_stat.add_event({
                'type': 'arrival',
                'timestamp': time.time(),
                'device_id': request.prefill_device_id,
                'request_id': request.request_id,
                'service_tier': request.service_tier,
            })
            # logger.info(f"Request {request.request_id} will be streamed from device {request.prefill_device_id}")
            self._profile_events.append({
                "event_type": "dispatch-both",
                "timestamp": time.time(),
                "request_id": request.request_id,
                "prefill_device_id": request.prefill_device_id,
                "decode_device_id": request.decode_device_id,
                "device_id": -1
            })
            
            # print('dispatch', request.request_id, time.time(), 'router')

            admission_history = request.admission_stat or asdict(
                self.load_stat.get_stat(
                    include_best_effort=(request.service_tier == 'best_effort'),
                )[request.prefill_device_id]
            )
            admission_history.update({
                'input_length': request.payload['vllm_xargs']['input_length'],
                'output_length': request.payload['vllm_xargs']['output_length'],
                'rejection_prob': request.rejection_prob,
                'request_id': request.request_id
            })
            admission_history['prefill_ddl'] = admission_history.get('prefill_ddl', request.payload['vllm_xargs']['prefill_ddl'] - time.time())

            admitted, generator, rejection_reason = await stream_service_response_engine(
                request.prefill_device_id,
                request.payload,
                request.request_id,
            )
            
            
            if not admitted:
                request.rejection_reason = rejection_reason
                self.load_stat.add_event({
                    'type': 'reject',
                    'timestamp': time.time(),
                    'device_id': request.prefill_device_id,
                    'request_id': request.request_id,
                    'service_tier': request.service_tier,
                })
                admission_history.update({
                    'is_rejected': True
                })
                if self.enable_rescheduling:
                    self.waiting_pool.append(request)
                    self.running_pool.remove(request)
                    request.admitted = None
                    request.prefill_device_id = request.decode_device_id = -1
                    self.update_req_state(request, RequestState.WAITING)
                    # request.payload['vllm_xargs']['prefill_ddl'] = time.time() + request.payload['vllm_xargs']['slo_ttft']
                    self._emit_rescheduling_event(
                        request_id=request.request_id,
                        prefill_device_id=attempted_prefill_device_id,
                        decode_device_id=attempted_decode_device_id,
                    )
                else:
                    await request.response_queue.put(
                        _build_rejection_response(request.rejection_reason)
                    )
                    await request.response_queue.put(None)
                    self.update_req_state(request, RequestState.DECODE_REJECTED)
                return 
            self._profile_events.append({
                "event_type": "admitted",
                "timestamp": time.time(),
                "request_id": request.request_id,
                "device_id": -1
            })
            async for data in generator:
                request.update(data)
                await request.response_queue.put(data)
            await request.response_queue.put(None)
            self.update_req_state(request, RequestState.DECODE_FINISHED)
            return 
        
        if request.state < RequestState.PREFILL_FINISHED:
            # Disaggregated path: prefill → kv → decode
            attempted_prefill_device_id = request.prefill_device_id
            attempted_decode_device_id = request.decode_device_id
            self.load_stat.add_event({
                'type': 'arrival',
                'timestamp': time.time(),
                'device_id': request.prefill_device_id,
                'request_id': request.request_id,
                'service_tier': request.service_tier,
            })
            self._profile_events.append({
                "event_type": "dispatch-prefill",
                "timestamp": time.time(),
                "request_id": request.request_id,
                "prefill_device_id": request.prefill_device_id,
                "decode_device_id": request.decode_device_id,
                "device_id": -1
            })

            logger.debug(f"Request {request.request_id} sending to prefill device {request.prefill_device_id}")
            request.payload['vllm_xargs']['prefill_ddl'] -= self.kv_xfer_delay
            request.payload['vllm_xargs']['slo_ttft'] -= self.kv_xfer_delay
            admitted, response, rejection_reason = await send_request_to_service_engine(
                request.prefill_device_id,
                request.payload,
                request.request_id,
            )
            request.payload['vllm_xargs']['prefill_ddl'] += self.kv_xfer_delay
            request.payload['vllm_xargs']['slo_ttft'] += self.kv_xfer_delay

            if not admitted:
                request.rejection_reason = rejection_reason
                logger.debug(f"Request {request.request_id} was rejected at prefill stage")
                self.load_stat.add_event({
                    'type': 'reject',
                    'timestamp': time.time(),
                    'device_id': request.prefill_device_id,
                    'request_id': request.request_id,
                    'service_tier': request.service_tier,
                })
                self.update_req_state(request, RequestState.PREFILL_REJECTED)
                if self.enable_rescheduling:
                    self.update_req_state(request, RequestState.PREFILL_REJECTED_WAITING)
                    self.waiting_pool.append(request)
                    self.running_pool.remove(request)
                    request.prefill_device_id = -1
                    request.admitted = None
                    self._emit_rescheduling_event(
                        request_id=request.request_id,
                        prefill_device_id=attempted_prefill_device_id,
                        decode_device_id=attempted_decode_device_id,
                    )
                else:
                    await request.response_queue.put(
                        _build_rejection_response(request.rejection_reason)
                    )
                    await request.response_queue.put(None)
                return
            kv_transfer_params = response.get('kv_transfer_params', {})
            request.kv_transfer_params = kv_transfer_params
            self.update_req_state(request, RequestState.PREFILL_FINISHED)
            # Prefill stage finished; request will either enter decode or wait for rerouting.
            self.load_stat.add_event({
                'type': 'finish',
                'timestamp': time.time(),
                'device_id': request.prefill_device_id,
                'request_id': request.request_id,
                'is_rejection': False,
                'is_slo_violation': False,
                'service_tier': request.service_tier,
            })
            request.update({'num_computed_tokens': request.num_prompt_tokens, 'timestamp': time.time()})

        assert request.state < RequestState.DECODE_FINISHED            
        
        if self.enable_rerouting and request.decode_device_id < 0:
            self.waiting_pool.append(request)
            self.running_pool.remove(request)
            request.admitted = None
            return
    
        assert request.decode_device_id >= 0
        self.load_stat.add_event({
            'type': 'arrival',
            'timestamp': time.time(),
            'device_id': request.decode_device_id,
            'request_id': request.request_id,
            'service_tier': request.service_tier,
        })
        
        self._profile_events.append({
            "event_type": "dispatch-decode",
            "timestamp": time.time(),
            "request_id": request.request_id,
            "prefill_device_id": request.prefill_device_id,
            "decode_device_id": request.decode_device_id,
            "device_id": -1
        })

        
        if request.kv_transfer_params:
            logger.debug(f"Request {request.request_id} updating kv_transfer_params for decode")
            kv_transfer_params = request.kv_transfer_params
            request.payload["kv_transfer_params"] = kv_transfer_params
            if self.mock_connector:
                assert 'num_tokens' in kv_transfer_params, f"num_tokens not found in kv_transfer_params: {kv_transfer_params}"
                assert 'dispatch_time' in kv_transfer_params, f"dispatch_time not found in kv_transfer_params: {kv_transfer_params}"
                kv_transfer_params['arrival_time'] = self.network.tx(request.prefill_device_id,
                                                                     request.decode_device_id,
                                                                     kv_transfer_params['num_tokens'],
                                                                     kv_transfer_params['dispatch_time'])

        # Stream response from decode service
        logger.debug(f"Request {request.request_id} streaming from decode device {request.decode_device_id}")
        is_rejected = False
        attempted_prefill_device_id = request.prefill_device_id
        attempted_decode_device_id = request.decode_device_id
        admitted, generator, rejection_reason = await stream_service_response_engine(
            request.decode_device_id,
            request.payload,
            request_id=request.request_id,
        )
        if not admitted: 
            request.rejection_reason = rejection_reason
            logger.debug(f"Request {request.request_id} was rejected at decode stage")
            is_rejected = True
            self.load_stat.add_event({
                'type': 'reject',
                'timestamp': time.time(),
                'device_id': request.decode_device_id,
                'request_id': request.request_id,
                'service_tier': request.service_tier,
            })
            self.update_req_state(request, RequestState.DECODE_REJECTED)
            if self.enable_rescheduling:
                self.update_req_state(request, RequestState.DECODE_REJECTED_WAITING)
                self.waiting_pool.append(request)
                self.running_pool.remove(request)
                request.admitted = None
                request.decode_device_id = -1
                self._emit_rescheduling_event(
                    request_id=request.request_id,
                    prefill_device_id=attempted_prefill_device_id,
                    decode_device_id=attempted_decode_device_id,
                )
                logger.debug(f"Request {request.request_id} waiting for rescheduling")
            else:
                await request.response_queue.put(
                    _build_rejection_response(request.rejection_reason)
                )
                await request.response_queue.put(None)
            return 
            
        async for data in generator:
            request.update(data)
            if not is_rejected:
                await request.response_queue.put(data)
                
        self.load_stat.add_event({
            'type': 'finish',
            'timestamp': time.time(),
            'device_id': request.decode_device_id,
            'request_id': request.request_id,
            'is_rejection': False,
            'is_slo_violation': request.is_slo_violation,
            'service_tier': request.service_tier,
        })
        await request.response_queue.put(None)
        self.update_req_state(request, RequestState.DECODE_FINISHED)
        logger.debug(f"Request {request.request_id} finished decode (disaggregated)")

request_pool: RequestPool | None = None

import traceback

async def routing_loop_with_error_monitoring():
    try:
        await request_pool.routing_loop()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"Exception in routing_loop: {e}\n{traceback.format_exc()}")
        if request_pool is not None:
            await request_pool.fail_all_active_requests("routing_loop", e)
# =========================
# FastAPI app lifecycle
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global routing_loop_task, execplan_bus_actor, engine_tasks
    # Start engine mux tasks now that an event loop exists
    engine_tasks = []
    if engine_actors:
        for idx, ea in enumerate(engine_actors):
            task = asyncio.create_task(
                ea.run(),
                name=f"EngineActor.run[{idx}]",
            )
            task.add_done_callback(_task_done)
            engine_tasks.append(task)
            
    routing_loop_task = asyncio.create_task(
        routing_loop_with_error_monitoring(),
        name="routing_loop_with_error_monitoring",
    )
    routing_loop_task.add_done_callback(_task_done)
    yield
    # Shutdown Ray actors cleanly
    if engine_actors:
        await _shutdown_engine_actors(engine_actors)
    if execplan_bus_actor is not None:
        execplan_bus_actor.reset.remote()
    routing_loop_task.cancel()
    for engine_task in engine_tasks:
        engine_task.cancel() 


app = FastAPI(lifespan=lifespan)


# =========================
# Endpoints
# =========================
@app.get('/health_check')
async def health_check():
    if engine_actors is None:
        return JSONResponse(status_code=200, content={'status': 'ok'})
    for actor in engine_actors:
        status = await actor.engine_actor.health_check.remote()
        if isinstance(status, dict) and 'loop_task_done' in status:
            if status.get('loop_task_done'):
                return JSONResponse(status_code=500, content=status)
            if status.get('loop_task_cancelled'):
                return JSONResponse(status_code=500, content=status)
            if status.get('loop_task_exception') is not None:
                return JSONResponse(status_code=500, content=status)
    return JSONResponse(status_code=200, content={'status': 'ok'})

@app.post("/v1/completions")
async def handle_completions(request: Request):
    if request_pool is None:
        logger.error("Request pool is not initialized")
        raise RuntimeError("Request pool is not initialized")
    return await request_pool.add_request(request)


@app.post('/dump_profile_events')
async def dump_profile_events(request: Request):
    request_json = await request.json()
    await request_pool.sync(timeout=request_json.get('timeout', 10.0))
    filename = os.path.abspath(request_json.get('filename', 'profile_events.jsonl'))
    include_energy_csv = bool(request_json.get('include_energy_csv', True))
    logger.info(f"Dumping profile events to {filename}")
    import json as _json
    if request_pool is None:
        logger.error("Request pool is not initialized")
        raise RuntimeError("Request pool is not initialized")
    all_events = []
    per_gpu_energy_consumption: list[float] = []
    energy_csv_files: list[str] = []
    local_energy_csv_files: list[str] = []
    worker_event_files: list[str] = []
    physical_gpu_ids: list[list[int]] = []
    for i, client in enumerate(request_pool.clients):
        try:
            logger.info(f"Dumping profile events from client {client}")
            worker_filename = _build_worker_dump_path(filename, i)
            dump_result = await engine_actors[i].engine_actor.dump_profile_events.remote(
                worker_filename,
                include_energy_csv,
            )
            worker_event_files.append(worker_filename)
            if dump_result.get("energy_csv_path"):
                energy_csv_files.append(dump_result["energy_csv_path"])
            energy_csv_content = dump_result.get("energy_csv_content")
            if energy_csv_content is not None:
                local_energy_csv = _build_worker_energy_csv_path(filename, i)
                _ensure_parent_dir(local_energy_csv)
                with open(local_energy_csv, "w", encoding="utf-8") as f:
                    f.write(energy_csv_content)
                local_energy_csv_files.append(local_energy_csv)
            physical_gpu_ids.append(dump_result.get("physical_gpu_ids", []))
            per_gpu_energy_consumption.append(
                float(dump_result.get("total_joules", 0.0)))
            events = dump_result.get("profile_events", [])
            _ensure_parent_dir(worker_filename)
            with open(worker_filename, 'w') as f:
                _json.dump(events, f, indent=4)
            for event in events:
                event['device_id'] = i
            all_events.extend(events)
        except Exception as e:
            logger.error(f"Error dumping profile events from client {client}, {i}: {e}")
    all_events.extend(request_pool._profile_events)
    all_events.sort(key=lambda x: x['timestamp'])
    _ensure_parent_dir(filename)
    with open(filename, 'w') as f:
        _json.dump(all_events, f, indent=4)
    logger.info(f"Dumped {len(all_events)} events to {filename}, example: {all_events[0]}")
    admission_filename = os.path.abspath(
        request_json.get('admission_filename', 'admission_history.jsonl'))

    _ensure_parent_dir(admission_filename)
    with open(admission_filename, 'w') as f:
        _json.dump(request_pool.admission_history, f, indent=4)
    logger.info(f"Dumped {len(request_pool.admission_history)} admission history to {filename}")
    return JSONResponse(status_code=200, content={
        "message": "Profile events dumped.",
        "energy_consumption": sum(per_gpu_energy_consumption),
        "per_gpu_energy_consumption": per_gpu_energy_consumption,
        "energy_csv_files": energy_csv_files,
        "local_energy_csv_files": local_energy_csv_files,
        "worker_event_files": worker_event_files,
        "physical_gpu_ids": physical_gpu_ids,
    })


@app.post('/start_energy_profile')
async def start_energy_profile(request: Request):
    request_json = await request.json()
    store_prefix = request_json.get("store_prefix")
    started = []
    for i, client in enumerate(request_pool.clients):
        started.append(
            await engine_actors[i].engine_actor.start_energy_profiling.remote(
                store_prefix))
        logger.info(f"Started energy profiler for {i}th client={client}")
    return JSONResponse(status_code=200, content={
        "message": "Energy profiling started.",
        "started": started,
    })


@app.post('/update_config')
async def update_config(request: Request):
    request_json = await request.json()
    requested_tp = int(request_json.get('tensor_parallel_size',
                                        args.tensor_parallel_size))
    if requested_tp != args.tensor_parallel_size:
        return JSONResponse(
            status_code=400,
            content={
                "error": "tensor_parallel_size mismatch",
                "requested_tensor_parallel_size": requested_tp,
                "configured_tensor_parallel_size": args.tensor_parallel_size,
            })
    request_pool.update_config(request_json)
    logger.info(f"Updated router: {request_pool.router}")
    await _wait_for_engine_actors_ready()

    # Fan out config to per-replica actors
    for i, client in enumerate(request_pool.clients):
        req = request_json.copy()
        req['engine_id'] = i
        req['is_mock_connector'] = args.mock_connector
        new_request_json = request_pool.router.update_json(req, i)
        if request_pool.enable_rescheduling:
            new_request_json['admission_mode'] = 'instant'
        else:
            new_request_json['admission_mode'] = 'anytime'
        await engine_actors[i].engine_actor.update_config.remote(new_request_json)
        logger.info(f"Updated config for {i}th client={client}")
    return JSONResponse(status_code=200, content={"message": "Config updated."})


@app.post('/warmup')
async def warmup(request: Request):
    request_json = await request.json()
    await request_pool.warmup(request_json)
    logger.info(f"Prewarming request: {request_json}")
    return JSONResponse(status_code=200, content={"message": "Prewarmed."})


@app.post('/update_clients')
async def update_clients(request: Request):
    request_json = await request.json()
    clients = _parse_clients_arg(request_json['clients'])
    logger.info(f'existing clients: {request_pool.clients}, new clients: {clients}')
    global engine_actors, engine_tasks
    # Recreate actors if client set changed
    global routing_loop_task
    if (engine_actors is None) or (clients != request_pool.clients):
        old_engine_actors = list(engine_actors) if engine_actors is not None else []
        if routing_loop_task is not None:
            routing_loop_task.cancel()
            try:
                await routing_loop_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error cancelling routing loop task: {e}")
        if engine_tasks:
            for engine_task in engine_tasks:
                engine_task.cancel()
            for engine_task in engine_tasks:
                try:
                    await engine_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error cancelling engine queue task: {e}")
            engine_tasks = []
        if old_engine_actors:
            await _shutdown_engine_actors(old_engine_actors, always_kill=True)
        start_engine(clients)
        engine_tasks = []
        for idx, ea in enumerate(engine_actors):
            task = asyncio.create_task(
                ea.run(),
                name=f"EngineActor.run[{idx}]",
            )
            task.add_done_callback(_task_done)
            engine_tasks.append(task)
        routing_loop_task = asyncio.create_task(
            routing_loop_with_error_monitoring(),
            name="routing_loop_with_error_monitoring",
        )
        routing_loop_task.add_done_callback(_task_done)
    else:
        logger.info("Client set unchanged; waiting for existing engine actors to report ready")
    await _wait_for_engine_actors_ready()
    request_pool.clients = clients
    logger.info(f"Updated clients: {clients}")
    return JSONResponse(status_code=200, content={"message": "Clients updated."})


@app.get('/benchmark')
async def benchmark(request: Request):
    df = await request_pool.benchmark()
    return JSONResponse(status_code=200, content={"message": "Benchmark results saved to network_latency.csv"})


example_usage = """
example:
python -m SLOsServe.router.api_server \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --window_size 1.0 \\
    --router slo \\
    --router_kwargs "{\\"hardware_params\\": [4.1e-5, 0, 1.3e-2], \\"tpot\\": 0.05, \\"device_mem\\": 16384, \\"block_size\\": 16}" \\
    --clients localhost:8100 localhost:8200
"""



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description=example_usage)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--window_size", type=float, default=0.005)
    parser.add_argument("--router", type=str, default="slo")
    parser.add_argument("--router_kwargs", type=str, default="{}")
    parser.add_argument("--clients", type=str, default=None)
    parser.add_argument("--enable_rerouting", action="store_true", default=False, help="enable rerouting of a rejected request")
    parser.add_argument("--enable_rescheduling", action="store_true", default=False, help="enable rescheduling after a request's P phase finishes")
    parser.add_argument("--admission_mode", type=str, default="anytime", help="admission mode: anytime or arrival, arrival: instant decision at arrival, anytime: admission can be made anytime.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="number of GPUs per logical replica")
    parser.add_argument(
        "--vllm_port_base",
        type=int,
        default=None,
        help="base port for per-replica vLLM internal rendezvous; replica i uses base + i * vllm_port_stride",
    )
    parser.add_argument(
        "--vllm_port_stride",
        type=int,
        default=64,
        help="port spacing between logical replicas when --vllm_port_base is set",
    )
    parser.add_argument("--mock_connector", action="store_true", default=False, help="use mock connector to simulate the network latency")
    parser.add_argument("--ray_address", type=str, default=None)  # NEW: allow Ray cluster connect
    parser.add_argument(
        "--worker_env",
        action="append",
        default=[],
        help="Environment variables propagated to EngineWorker actors, e.g. --worker_env NCCL_IB_DISABLE=1",
    )
    parser.add_argument("--stat_window", type=float, default=0.5, help="stat window for collecting load statistics from backend engine (used for load prediction)")
    parser.add_argument("--mock_engine", action="store_true", default=False, help="use mock engine to simulate the model inference")
    parser.add_argument(
        "--mock_engine_device_mem_bytes",
        type=int,
        default=20_000_000_000,
        help=(
            "mock-engine-only device memory budget in bytes; used to size the "
            "mock KV cache"
        ),
    )
    parser.add_argument("--log_level", type=str, default=os.getenv("SLOSSERVE_LOG_LEVEL", "INFO"), help="logging level (e.g., DEBUG, INFO, WARNING)")
    return parser.parse_args()

def start_engine(clients: list):
    # Initialize Ray locally or connect to a cluster
    if not ray.is_initialized():
        ray_log_level = getattr(logging, args.log_level.upper(), logging.INFO)
        if args.ray_address:
            ray.init(address=args.ray_address, log_to_driver=True, logging_level=ray_log_level)
        else:
            ray.init(log_to_driver=True, logging_level=ray_log_level)

    n_devices = len(clients)
    worker_env = _parse_worker_env_args(args.worker_env)
    print('clients: ', clients, 'n_devices: ', n_devices,
          'tensor_parallel_size: ', args.tensor_parallel_size,
          'worker_env: ', worker_env)

    # Create one EngineWorker per logical replica. Each actor may use
    # multiple GPUs via tensor parallelism.
    global engine_actors, engine_tasks, execplan_bus_actor, engine_queues
    if execplan_bus_actor is None:
        execplan_bus_actor = ExecPlanBus.remote()
    else:
        execplan_bus_actor.reset.remote()
    engine_actors = []
    engine_tasks = []
    remote_actors = []
    for device_id in range(n_devices):
        output_queue = RayQueue(maxsize=8192)
        vllm_port = None
        if args.vllm_port_base is not None:
            vllm_port = args.vllm_port_base + device_id * args.vllm_port_stride
        if args.mock_engine:
            actor = EngineWorker.remote(
                args.model_name,
                args.mock_connector,
                mock_engine=True,
                mock_engine_device_mem_bytes=args.mock_engine_device_mem_bytes,
                device_id=device_id,
                tensor_parallel_size=args.tensor_parallel_size,
                worker_env=worker_env,
                vllm_port=vllm_port,
                execplan_bus=execplan_bus_actor,
                output_queue = output_queue
            )
        else:
            actor_options: dict[str, Any] = {
                "num_gpus": args.tensor_parallel_size,
            }
            if worker_env:
                actor_options["runtime_env"] = {"env_vars": worker_env}
            actor = EngineWorker.options(**actor_options).remote(
                args.model_name,
                args.mock_connector,
                device_id=device_id,
                tensor_parallel_size=args.tensor_parallel_size,
                worker_env=worker_env,
                vllm_port=vllm_port,
                execplan_bus=execplan_bus_actor,
                output_queue = output_queue 
            )
        remote_actors.append(actor)
        engine_actor = EngineActor(
            actor,
            output_queue
        )
        engine_actors.append(engine_actor)
    ray.get([actor.wait_until_ready.remote() for actor in remote_actors])

if __name__ == "__main__":
    import uvicorn
    start_time = time.time()
    args = parse_args()
    os.environ["SLOSSERVE_LOG_LEVEL"] = args.log_level.upper()
    setup_logger("SLOsServe.router.api_server", args.log_level)
    clients = _parse_clients_arg(args.clients)
    logger.info(f"Starting API server on {args.host}:{args.port} with router={args.router} and clients={args.clients}")
    if args.clients:
        start_engine(clients)
    logger.info(f"Engine started: {engine_actors}")
    router = create_router(args.router, len(clients), args.router_kwargs)
    load_stat = LoadStat(max_window=args.stat_window, n_devices=len(clients))
    request_pool = RequestPool(args.window_size, router, clients, None,
                               args.enable_rerouting, args.enable_rescheduling, args.admission_mode, args.mock_connector, load_stat, args.stat_window)
    start_up_time = time.time() - start_time
    logger.info(f"Start up time: {start_up_time} seconds")
    uvicorn.run(app, host=args.host, port=args.port, access_log=False, log_level = args.log_level.lower())
