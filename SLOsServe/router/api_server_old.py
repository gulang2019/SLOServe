
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import itertools
import json
import math
import logging
import os
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

try:  # Optional dependency used only when pooling is enabled
    import SLOsServe_C  # type: ignore
except ImportError:  # pragma: no cover - only hit when extension is missing
    SLOsServe_C = None  # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class RouterAttributes:
    request_id: str
    ddl: float
    input_length: int
    profit: float
    mem: int
    tpot_idx: int
    prefill_mem: int


@dataclass
class PoolingJob:
    request_id: str
    arrival_time: float
    api: str
    payload: Dict[str, Any]
    stream: bool
    max_try: Optional[int]
    future: "asyncio.Future[JSONResponse | StreamingResponse]"


@dataclass
class ActiveAssignment:
    attrs: RouterAttributes
    prefill_device_id: int
    decode_device_id: int
    admitted_at: float
    prefill_reserved: int = 0
    decode_reserved: int = 0    

@dataclass
class ExecutionConfig:
    hardware_params: List[float]
    max_decode_size: int
    block_size: int
    memory_size: int

    def _normalize_decode_tokens(self, decode_tokens: Optional[int]) -> int:
        if decode_tokens is None:
            return self.max_decode_size
        return max(int(decode_tokens), 0)

    def calculate_memory(self, prompt_tokens: int, decode_tokens: Optional[int]) -> Tuple[int, int, int]:
        """Return (prefill_blocks, decode_blocks, execution_blocks)."""
        prompt = max(int(prompt_tokens), 0)
        normalized_decode = self._normalize_decode_tokens(decode_tokens)
        block_size = max(self.block_size, 1)
        prefill_blocks = math.ceil(prompt / block_size) if prompt > 0 else 0

        if normalized_decode <= 0:
            decode_blocks = 0
            execution_blocks = prefill_blocks
        else:
            total_blocks = math.ceil((prompt + normalized_decode) / block_size)
            decode_blocks = max(total_blocks - prefill_blocks, 0)
            execution_blocks = total_blocks

        return prefill_blocks, decode_blocks, execution_blocks

@dataclass
class Client:
    host: str
    port: int
    id: int
    
@dataclass
class ClientState:
    _client: Client
    _exec_config: ExecutionConfig = field(init = False)
    
    def __post_init__(self):
        self._exec_config = ExecutionConfig(
            hardware_params=[1.0],
            max_decode_size=1024,
            block_size=1024,
            memory_size=1024,
        )

class RouteAlg:
    def __init__(self, clients: List[Client], routing_hints: str):
        self._clients = clients
        self._routing_hints = routing_hints

    def schedule(self, reqs: List[Request]) -> List[RequestOutput]:
        pass

class PoolingRouter:
    def __init__(self, *,
                 clients: List[Client],
                 hardware_params: List[float],
                 tpot: float,
                 device_mems: List[int],
                 window_seconds: float,
                 verbose: bool = False) -> None:
        if SLOsServe_C is None:
            raise RuntimeError("SLOsServe_C extension not available; pooling router cannot be created")


        self._device_capacity = [int(m) for m in normalized_device_mems]
        self._device_available = self._device_capacity.copy()
        self._window_seconds = max(window_seconds, 0.0)
        self._verbose = verbose
        self._lock = asyncio.Lock()
        self._pending: Deque[PoolingJob] = deque()
        self._active: Dict[str, ActiveAssignment] = {}
        self._shutdown = asyncio.Event()
        self._jitter_counter = 0
        self._jitter_epsilon = 1e-6

        self._router = SLOsServe_C.AdmCtrlRouter(len(self._device_capacity), hardware_params, tpot)
        self._task = asyncio.create_task(self._run_loop())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def submit(self, api: str, payload: Dict[str, Any], request_id: str,
                     *, stream: bool, max_try: Optional[int]) -> JSONResponse | StreamingResponse:
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[JSONResponse | StreamingResponse]" = loop.create_future()
        job = PoolingJob(
            request_id=request_id,
            arrival_time=time.monotonic(),
            api=api,
            payload=payload,
            stream=stream,
            max_try=max_try,
            future=future,
        )
        async with self._lock:
            self._pending.append(job)
        # If pooling window is zero, flush immediately to emulate non-pooled routing.
        if self._window_seconds == 0.0:
            await self._flush()
        return await future

    async def aclose(self) -> None:
        self._shutdown.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:  # pragma: no cover - expected on shutdown
                pass
        # Reject any jobs still pending
        async with self._lock:
            while self._pending:
                job = self._pending.popleft()
                if not job.future.done():
                    job.future.set_result(JSONResponse({"finish_reason": "rejected"}))
        # Clear active assignments
        self._active.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _run_loop(self) -> None:
        try:
            while not self._shutdown.is_set():
                await asyncio.sleep(self._window_seconds if self._window_seconds > 0 else 0.01)
                await self._flush()
        except asyncio.CancelledError:  # pragma: no cover - cooperative cancellation
            raise

    async def _flush(self) -> None:
        async with self._lock:
            if not self._pending:
                return
            jobs = list(self._pending)
            self._pending.clear()

        await self._schedule_jobs(jobs)

    async def _schedule_jobs(self, jobs: List[PoolingJob]) -> None:
        if not jobs:
            return

        current_time = time.monotonic()

        reqs: List[Any] = []
        # Include ongoing assignments first so router accounts for in-flight load.
        for assignment in self._active.values():
            reqs.append(self._build_c_request(assignment.attrs, False,
                                              assignment.prefill_device_id,
                                              assignment.decode_device_id))

        new_entries: List[Tuple[PoolingJob, RouterAttributes]] = []
        for job in jobs:
            attrs = self._extract_attributes(job)
            reqs.append(self._build_c_request(attrs, True, -1, -1))
            new_entries.append((job, attrs))

        outputs = self._router.schedule(reqs, self._device_capacity, current_time, self._verbose)

        if len(outputs) != len(new_entries):  # pragma: no cover - defensive guard
            raise RuntimeError("Router output size mismatch with pending jobs")

        # Router sorts requests by deadline; mirror the same order to align outputs.
        ordered_entries = sorted(new_entries, key=lambda item: item[1].ddl)

        for (job, attrs), result in zip(ordered_entries, outputs):
            if not result.admitted:
                if not job.future.done():
                    job.future.set_result(JSONResponse({
                        "finish_reason": "rejected",
                        "request_id": job.request_id,
                    }))
                continue

            prefill_id = result.prefill_device_id
            decode_id = result.decode_device_id
            if self._shared_devices:
                decode_id = prefill_id

            if prefill_id < 0 or prefill_id >= len(self._prefill_clients):
                if not job.future.done():
                    job.future.set_result(JSONResponse({
                        "finish_reason": "rejected",
                        "reason": "invalid_prefill_assignment",
                    }))
                continue

            if decode_id < 0 or decode_id >= len(self._decode_clients):
                if not job.future.done():
                    job.future.set_result(JSONResponse({
                        "finish_reason": "rejected",
                        "reason": "invalid_decode_assignment",
                    }))
                continue

            assignment = ActiveAssignment(
                attrs=attrs,
                prefill_device_id=prefill_id,
                decode_device_id=decode_id,
                admitted_at=current_time,
            )
            self._reserve_resources(assignment)
            self._active[job.request_id] = assignment
            asyncio.create_task(self._dispatch_job(job, assignment))

    def _extract_attributes(self, job: PoolingJob) -> RouterAttributes:
        meta = job.payload.get("router_metadata") or {}
        defaults = self._defaults

        deadline_hint = meta.get("ddl", meta.get("deadline"))
        if deadline_hint is None:
            deadline = job.arrival_time + float(defaults["deadline"])
        else:
            deadline = float(deadline_hint)
            if deadline <= job.arrival_time:
                deadline = job.arrival_time + float(defaults["deadline"])

        # Ensure unique ordering when deadlines tie by injecting a tiny jitter.
        deadline += self._next_jitter()

        def _optional_int_field(*keys: str) -> Optional[int]:
            for key in keys:
                value = meta.get(key)
                if value is not None:
                    return int(value)
                value = job.payload.get(key)
                if value is not None:
                    return int(value)
            return None

        def _int_field(*keys: str, fallback: int) -> int:
            value = _optional_int_field(*keys)
            if value is not None:
                return value
            return int(fallback)

        def _float_field(*keys: str, fallback: float) -> float:
            for key in keys:
                value = meta.get(key)
                if value is not None:
                    return float(value)
                value = job.payload.get(key)
                if value is not None:
                    return float(value)
            return float(fallback)

        input_length = _int_field("input_length", "prompt_length", fallback=defaults["input_length"])
        decode_tokens = _optional_int_field("decode_tokens", "decode_length",
                                            "max_decode_tokens", "max_tokens",
                                            "max_completion_tokens")

        exec_prefill = exec_decode = exec_total = None
        normalized_decode_tokens: Optional[int] = decode_tokens
        if self._execution_configs:
            config = self._execution_configs[0]
            exec_prefill, exec_decode, exec_total = config.calculate_memory(input_length, decode_tokens)
            normalized_decode_tokens = config._normalize_decode_tokens(decode_tokens)

        prefill_override = _optional_int_field("prefill_mem", "prefill_memory")
        decode_override = _optional_int_field("decode_mem", "mem", "memory")

        prefill_mem = prefill_override if prefill_override is not None else (
            exec_prefill if exec_prefill is not None else int(defaults["prefill_mem"])
        )
        mem = decode_override if decode_override is not None else (
            exec_decode if exec_decode is not None else int(defaults["mem"])
        )

        if exec_prefill is not None:
            updated_meta = dict(meta)
            updated_meta["prefill_mem"] = prefill_mem
            updated_meta["decode_mem"] = mem
            updated_meta.setdefault("prompt_length", input_length)
            if normalized_decode_tokens is not None:
                updated_meta.setdefault("max_decode_tokens", normalized_decode_tokens)
            job.payload["router_metadata"] = updated_meta

            execution_meta = job.payload.setdefault("execution_metadata", {})
            if not isinstance(execution_meta, dict):
                execution_meta = {}
                job.payload["execution_metadata"] = execution_meta
            execution_meta["memory"] = {
                "prefill_mem": prefill_mem,
                "decode_mem": mem,
                "execution_mem": exec_total if exec_total is not None else prefill_mem + mem,
            }
            meta = updated_meta

        profit = _float_field("profit", fallback=defaults["profit"])
        tpot_idx = _int_field("tpot_idx", fallback=defaults["tpot_idx"])

        return RouterAttributes(
            request_id=job.request_id,
            ddl=deadline,
            input_length=input_length,
            profit=profit,
            mem=mem,
            tpot_idx=tpot_idx,
            prefill_mem=prefill_mem,
        )

    def _build_c_request(self, attrs: RouterAttributes, is_new: bool,
                         prefill_device_id: int, decode_device_id: int) -> Any:
        return SLOsServe_C.Request(
            id=attrs.request_id,
            is_new_req=is_new,
            ddl=attrs.ddl,
            input_length=attrs.input_length,
            profit=attrs.profit,
            mem=attrs.mem,
            tpot_idx=attrs.tpot_idx,
            prefill_mem=attrs.prefill_mem,
            prefill_device_id=prefill_device_id,
            decode_device_id=decode_device_id,
        )

    def _next_jitter(self) -> float:
        self._jitter_counter += 1
        return self._jitter_counter * self._jitter_epsilon

    def _adjust_device_mem(self, device_id: int, delta: int) -> None:
        if device_id < 0 or device_id >= len(self._device_available):
            return
        self._device_available[device_id] += delta
        capacity = self._device_capacity[device_id]
        if self._device_available[device_id] < 0:
            logger.warning("Device %s memory over-reserved by %s tokens", device_id, -self._device_available[device_id])
            self._device_available[device_id] = 0
        elif self._device_available[device_id] > capacity:
            self._device_available[device_id] = capacity

    def _reserve_resources(self, assignment: ActiveAssignment) -> None:
        prefill_tokens = max(int(assignment.attrs.prefill_mem), 0)
        if prefill_tokens > 0:
            self._adjust_device_mem(assignment.prefill_device_id, -prefill_tokens)
            assignment.prefill_reserved = prefill_tokens
        decode_tokens = max(int(assignment.attrs.mem), 0)
        if decode_tokens > 0:
            self._adjust_device_mem(assignment.decode_device_id, -decode_tokens)
            assignment.decode_reserved = decode_tokens

    def _release_prefill_resources(self, assignment: ActiveAssignment) -> None:
        tokens = assignment.prefill_reserved
        if tokens > 0:
            self._adjust_device_mem(assignment.prefill_device_id, tokens)
            assignment.prefill_reserved = 0
            assignment.attrs.prefill_mem = 0

    def _release_decode_resources(self, assignment: ActiveAssignment) -> None:
        tokens = assignment.decode_reserved
        if tokens > 0:
            self._adjust_device_mem(assignment.decode_device_id, tokens)
            assignment.decode_reserved = 0
            assignment.attrs.mem = 0

    async def _dispatch_job(self, job: PoolingJob, assignment: ActiveAssignment) -> None:
        cleanup_called = False

        def cleanup() -> None:
            nonlocal cleanup_called
            if cleanup_called:
                return
            cleanup_called = True
            self._release_decode_resources(assignment)
            self._release_prefill_resources(assignment)
            self._active.pop(job.request_id, None)

        try:
            prefill_client = self._prefill_clients[assignment.prefill_device_id]
            if not self._shared_devices and assignment.prefill_device_id == assignment.decode_device_id:
                response = await _send_request(
                    prefill_client,
                    job.api,
                    job.payload,
                    job.request_id,
                    stream=job.stream,
                    cleanup=cleanup,
                )
                if not job.future.done():
                    job.future.set_result(response)
                return
            prefill_response = await _send_prefill_request(
                prefill_client,
                job.api,
                job.payload,
                job.request_id,
            )

            if prefill_response.get('finish_reason') == 'rejected':
                cleanup()
                if not job.future.done():
                    job.future.set_result(JSONResponse(prefill_response))
                return

            decode_payload = dict(job.payload)
            kv_transfer_params = prefill_response.get('kv_transfer_params')
            if kv_transfer_params:
                decode_payload['kv_transfer_params'] = kv_transfer_params

            self._release_prefill_resources(assignment)

            decode_client = self._decode_clients[assignment.decode_device_id]

            if job.stream:
                response = await self._stream_decode_fixed(
                    decode_client,
                    job.api,
                    decode_payload,
                    job.request_id,
                    cleanup,
                )
                if not job.future.done():
                    job.future.set_result(response)
            else:
                response = await self._decode_fixed(
                    decode_client,
                    job.api,
                    decode_payload,
                    job.request_id,
                )
                cleanup()
                if not job.future.done():
                    job.future.set_result(response)
        except Exception as exc:  # pylint: disable=broad-except
            cleanup()
            if not job.future.done():
                job.future.set_exception(exc)

    async def _decode_fixed(self, decode_client: dict, api: str, payload: Dict[str, Any],
                            request_id: str) -> JSONResponse:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }
        response = await decode_client['client'].post(api, json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
        return JSONResponse(body)

    async def _stream_decode_fixed(self, decode_client: dict, api: str,
                                   payload: Dict[str, Any], request_id: str,
                                   cleanup: Callable[[], None]) -> JSONResponse | StreamingResponse:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }

        stream_ctx = decode_client['client'].stream("POST", api, json=payload, headers=headers)
        response = await stream_ctx.__aenter__()
        try:
            response.raise_for_status()
            aiter = response.aiter_bytes()
            try:
                first_chunk = await aiter.__anext__()
            except StopAsyncIteration:
                await stream_ctx.__aexit__(None, None, None)
                cleanup()
                return StreamingResponse(iter(()), media_type="application/json")

            rejection_payload = _extract_rejection_payload(first_chunk)
            if rejection_payload:
                await stream_ctx.__aexit__(None, None, None)
                cleanup()
                return JSONResponse(rejection_payload)

            async def stream() -> Any:
                try:
                    if first_chunk:
                        yield first_chunk
                    async for chunk in aiter:
                        yield chunk
                finally:
                    await stream_ctx.__aexit__(None, None, None)
                    cleanup()

            return StreamingResponse(stream(), media_type="application/json")
        except Exception:  # pylint: disable=broad-except
            await stream_ctx.__aexit__(*sys.exc_info())
            cleanup()
            raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize client pools for prefiller and decoder services
    app.state.prefill_clients = []
    app.state.decode_clients = []
    app.state.execution_configs = []

    # Create prefill clients
    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f'http://{host}:{port}/v1'
        app.state.prefill_clients.append({
            'client': httpx.AsyncClient(timeout=None, base_url=prefiller_base_url),
            'host': host,
            'port': port,
            'id': i,
        })

    # Create decode clients
    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f'http://{host}:{port}/v1'
        app.state.decode_clients.append({
            'client': httpx.AsyncClient(timeout=None, base_url=decoder_base_url),
            'host': host,
            'port': port,
            'id': i,
        })

    if not app.state.decode_clients:
        app.state.decode_clients = list(app.state.prefill_clients)
    app.state.pooling_shared_devices = True

    # Initialize round-robin iterators
    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))

    app.state.pooling_router = None

    if getattr(global_args, "enable_pooling", False):
        if not app.state.decode_clients:
            raise RuntimeError("Pooling router requires at least one client")

        pool_clients = _deduplicate_clients(app.state.decode_clients)

        provided_mems = global_args.router_device_mems
        target_device_count = len(pool_clients)
        if provided_mems:
            device_mems = list(provided_mems)
            if len(device_mems) == 1 and target_device_count > 1:
                device_mems = device_mems * target_device_count
            elif len(device_mems) != target_device_count:
                raise ValueError(
                    "router-device-mems must either have one value or match the number of pooled devices"
                )
        else:
            device_mems = [int(global_args.router_default_mem)] * target_device_count

        decode_count = target_device_count
        execution_memory_sizes = _expand_per_device(
            getattr(global_args, "execution_memory_sizes", None),
            decode_count,
            default=device_mems,
            name="execution-memory-sizes",
        )
        execution_block_sizes = _expand_per_device(
            getattr(global_args, "execution_block_sizes", None),
            decode_count,
            default=[global_args.execution_block_size],
            name="execution-block-sizes",
        )
        execution_max_decode_sizes = _expand_per_device(
            getattr(global_args, "execution_max_decode_sizes", None),
            decode_count,
            default=[global_args.execution_max_decode_size],
            name="execution-max-decode-sizes",
        )

        execution_configs = [
            ExecutionConfig(
                hardware_params=list(global_args.router_hardware_params),
                max_decode_size=execution_max_decode_sizes[i],
                block_size=execution_block_sizes[i],
                memory_size=execution_memory_sizes[i],
            )
            for i in range(decode_count)
        ]

        device_mems = execution_memory_sizes
        for client_info, exec_cfg in zip(app.state.decode_clients, execution_configs):
            client_info['execution_config'] = exec_cfg

        app.state.execution_configs = execution_configs

        defaults = {
            "deadline": float(global_args.pooling_default_deadline),
            "prefill_mem": int(global_args.pooling_default_prefill_mem),
            "mem": int(global_args.pooling_default_mem),
            "input_length": int(global_args.pooling_default_input_length),
            "profit": float(global_args.pooling_default_profit),
            "tpot_idx": int(global_args.pooling_default_tpot_idx),
        }

        client_roles = {
            'P': [],
            'D': [],
            'both': pool_clients,
        }

        app.state.pooling_router = PoolingRouter(
            client_roles=client_roles,
            hardware_params=list(global_args.router_hardware_params),
            tpot=float(global_args.router_tpot),
            device_mems=device_mems,
            window_seconds=float(global_args.pooling_window_seconds),
            defaults=defaults,
            execution_configs=execution_configs,
            verbose=bool(global_args.pooling_verbose),
        )

    print(
        f"Initialized {len(app.state.prefill_clients)} prefill clients "
        f"and {len(app.state.decode_clients)} decode clients."
    )

    yield

    # Shutdown: Close all clients
    if app.state.pooling_router is not None:
        await app.state.pooling_router.aclose()

    seen_clients = set()
    for client_info in list(app.state.prefill_clients) + list(app.state.decode_clients):
        client = client_info['client']
        key = id(client)
        if key in seen_clients:
            continue
        seen_clients.add(key)
        await client.aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")

    # For prefiller instances
    # parser.add_argument("--prefiller-hosts", "--prefiller-host", type=str, nargs="+", default=["localhost"])
    # parser.add_argument("--prefiller-ports", "--prefiller-port", type=int, nargs="+", default=[8100])

    # # For decoder instances
    # parser.add_argument("--decoder-hosts", "--decoder-host", type=str, nargs="+", default=["localhost"])
    # parser.add_argument("--decoder-ports", "--decoder-port", type=int, nargs="+", default=[8200])

    parser.add_argument("--enable-pooling", action="store_true",
                        help="Enable pooled admission control and routing")
    parser.add_argument("--pooling-window-ms", type=int, default=100,
                        help="Window size in milliseconds for pooling new requests")
    parser.add_argument("--pooling-default-deadline", type=float, default=1.0,
                        help="Fallback deadline (seconds) added to arrival if request does not specify one")
    parser.add_argument("--pooling-default-prefill-mem", type=int, default=2048,
                        help="Default prefill memory tokens if request omits it")
    parser.add_argument("--pooling-default-mem", type=int, default=4096,
                        help="Default decode memory tokens if request omits it")
    parser.add_argument("--pooling-default-input-length", type=int, default=1024,
                        help="Default input length for scheduler when unspecified")
    parser.add_argument("--pooling-default-profit", type=float, default=1.0,
                        help="Default profit/priority when unspecified")
    parser.add_argument("--pooling-default-tpot-idx", type=int, default=0,
                        help="Default throughput tier index when unspecified")
    parser.add_argument("--pooling-verbose", action="store_true",
                        help="Enable verbose logging for pooling scheduler decisions")
    parser.add_argument("--router-hardware-params", type=float, nargs="+", default=[4.1e-5, 0, 1.3e-2],
                        help="Hardware parameters vector passed to AdmCtrlRouter")
    parser.add_argument("--router-tpot", type=float, default=0.1,
                        help="TPOT value passed to AdmCtrlRouter")
    parser.add_argument("--router-device-mems", type=int, nargs="+", default=None,
                        help="Available memory tokens per device for the router (repeat or one per device)")
    parser.add_argument("--router-default-mem", type=int, default=65536,
                        help="Default per-device memory tokens when router-device-mems not specified")
    parser.add_argument("--execution-block-size", type=int, default=16,
                        help="Default KV block size (tokens) for execution memory calculations")
    parser.add_argument("--execution-block-sizes", type=int, nargs="+", default=None,
                        help="Override KV block size per decode device; provide one value or one per device")
    parser.add_argument("--execution-max-decode-size", type=int, default=4096,
                        help="Default maximum decode tokens used when a request omits a limit")
    parser.add_argument("--execution-max-decode-sizes", type=int, nargs="+", default=None,
                        help="Override maximum decode tokens per decode device; provide one value or one per device")
    parser.add_argument("--execution-memory-sizes", type=int, nargs="+", default=None,
                        help="Override execution memory capacity per decode device (tokens); defaults to router device mems")

    args = parser.parse_args()

    # Validate and pair hosts with ports
    # if len(args.prefiller_hosts) != len(args.prefiller_ports):
    #     raise ValueError("Number of prefiller hosts must match number of prefiller ports")

    # if len(args.decoder_hosts) != len(args.decoder_ports):
    #     raise ValueError("Number of decoder hosts must match number of decoder ports")

    # Create tuples of (host, port) for each service type
    # args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    # args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))

    args.pooling_window_seconds = max(args.pooling_window_ms / 1000.0, 0.0)

    return args


def _deduplicate_clients(clients: List[dict]) -> List[dict]:
    seen = set()
    deduped: List[dict] = []
    for client in clients:
        key = (client.get('host'), client.get('port'), client.get('id'))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(client)
    return deduped


def _expand_per_device(values: Optional[List[int]], count: int, *, default, name: str) -> List[int]:
    """Expand per-device CLI values, supporting single-value broadcast."""
    if values:
        expanded = list(values)
        if len(expanded) == 1 and count > 1:
            expanded *= count
        if len(expanded) != count:
            raise ValueError(f"{name} must provide either 1 value or {count} values")
        return [int(v) for v in expanded]

    if isinstance(default, list):
        if len(default) != count:
            raise ValueError(f"Default for {name} must have {count} entries")
        return [int(v) for v in default]

    return [int(default)] * count


def get_next_client(app: FastAPI, service_type: str) -> dict:
    """Get the next client in round-robin fashion."""
    if service_type == 'prefill':
        client_idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[client_idx]
    if service_type == 'decode':
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    raise ValueError(f"Unknown service type: {service_type}")


def _extract_rejection_payload(raw: bytes) -> Optional[dict]:
    """Attempt to parse a rejection payload from a streamed chunk."""
    if not raw:
        return None
    try:
        text = raw.decode("utf-8", errors="ignore")
    except UnicodeDecodeError:
        return None

    for line in text.splitlines():
        content = line.strip()
        if not content:
            continue
        if content.startswith("data:"):
            content = content[len("data:"):].strip()
            if content in {"", "[DONE]"}:
                continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue

        finish_reason = payload.get("finish_reason")
        choice_payload = None
        choices = payload.get("choices")
        if not finish_reason and isinstance(choices, list):
            for choice in choices:
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    if isinstance(choice, dict):
                        choice_payload = choice
                    break

        if finish_reason == "rejected":
            data = choice_payload or payload
            if isinstance(data, dict):
                data.setdefault("finish_reason", "rejected")
                return data
            return {"finish_reason": "rejected"}

    return None


async def _send_prefill_request(client_info: dict, endpoint: str, req_data: dict, request_id: str) -> dict:
    """Send the prefill request (non-streaming)."""
    payload = req_data.copy()
    payload['kv_transfer_params'] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    payload.setdefault("stream", False)
    payload["max_tokens"] = 1
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = 1
    payload.pop("stream_options", None)

    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    response = await client_info['client'].post(endpoint, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


async def _send_request(client_info: dict, endpoint: str, req_data: dict, request_id: str,
                        *, stream: bool, cleanup: Optional[Callable[[], None]] = None
                        ) -> JSONResponse | StreamingResponse:
    """Send the original request without forcing a decode-only limitation."""
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    if stream:
        if cleanup is None:
            raise ValueError("cleanup callback required for streaming requests")
        stream_ctx = client_info['client'].stream("POST", endpoint, json=req_data, headers=headers)
        response = await stream_ctx.__aenter__()
        try:
            response.raise_for_status()
            aiter = response.aiter_bytes()
            try:
                first_chunk = await aiter.__anext__()
            except StopAsyncIteration:
                await stream_ctx.__aexit__(None, None, None)
                cleanup()
                return StreamingResponse(iter(()), media_type="application/json")

            rejection_payload = _extract_rejection_payload(first_chunk)
            if rejection_payload:
                await stream_ctx.__aexit__(None, None, None)
                cleanup()
                return JSONResponse(rejection_payload)

            async def stream_iter():
                try:
                    if first_chunk:
                        yield first_chunk
                    async for chunk in aiter:
                        yield chunk
                finally:
                    await stream_ctx.__aexit__(None, None, None)
                    cleanup()

            return StreamingResponse(stream_iter(), media_type="application/json")
        except Exception:
            await stream_ctx.__aexit__(*sys.exc_info())
            cleanup()
            raise

    response = await client_info['client'].post(endpoint, json=req_data, headers=headers)
    response.raise_for_status()
    if cleanup:
        cleanup()
    return JSONResponse(response.json())


async def _decode_non_stream(app: FastAPI, api: str, base_payload: dict, request_id: str,
                              max_try: Optional[int]) -> JSONResponse:
    attempts = max_try or len(app.state.decode_clients)
    last_payload: Optional[dict] = None
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }
    for _ in range(max(1, attempts)):
        decode_client = get_next_client(app, 'decode')
        response = await decode_client['client'].post(api, json=base_payload, headers=headers)
        response.raise_for_status()
        body = response.json()
        if body.get('finish_reason') != 'rejected':
            return JSONResponse(body)
        last_payload = body
    return JSONResponse(last_payload or {"finish_reason": "rejected"})


async def _stream_decode_with_rejection(app: FastAPI, api: str, base_payload: dict,
                                        request_id: str, max_try: Optional[int]) -> JSONResponse | StreamingResponse:
    attempts = max_try or len(app.state.decode_clients)
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    for attempt in range(max(1, attempts)):
        decode_client = get_next_client(app, 'decode')
        stream_ctx = decode_client['client'].stream("POST", api, json=base_payload, headers=headers)
        response = await stream_ctx.__aenter__()
        try:
            response.raise_for_status()
            aiter = response.aiter_bytes()
            try:
                first_chunk = await aiter.__anext__()
            except StopAsyncIteration:
                await stream_ctx.__aexit__(None, None, None)
                return StreamingResponse(iter(()), media_type="application/json")

            rejection_payload = _extract_rejection_payload(first_chunk)
            if rejection_payload:
                await stream_ctx.__aexit__(None, None, None)
                if attempt == attempts - 1:
                    return JSONResponse(rejection_payload)
                continue

            async def stream():
                try:
                    if first_chunk:
                        yield first_chunk
                    async for chunk in aiter:
                        yield chunk
                finally:
                    await stream_ctx.__aexit__(None, None, None)

            return StreamingResponse(stream(), media_type="application/json")
        except Exception as exc:  # ensure the context manager exits
            await stream_ctx.__aexit__(*sys.exc_info())
            raise exc

    return JSONResponse({"finish_reason": "rejected"})


async def _handle_completions(api: str, request: Request, max_try: Optional[int] = None):
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())

        # Prefill stage (allow retries)
        prefill_attempts = max_try or len(request.app.state.prefill_clients)
        prefill_response: Optional[dict] = None
        last_prefill: Optional[dict] = None
        for _ in range(max(1, prefill_attempts)):
            prefill_client_info = get_next_client(request.app, 'prefill')
            current_response = await _send_prefill_request(prefill_client_info, api, req_data, request_id)
            if current_response.get('finish_reason') != 'rejected':
                prefill_response = current_response
                break
            last_prefill = current_response

        if prefill_response is None:
            return JSONResponse(last_prefill or {"finish_reason": "rejected"})

        kv_transfer_params = prefill_response.get('kv_transfer_params')
        decode_payload = req_data.copy()
        if kv_transfer_params:
            decode_payload['kv_transfer_params'] = kv_transfer_params

        if decode_payload.get('stream'):
            return await _stream_decode_with_rejection(request.app, api, decode_payload, request_id, max_try)
        return await _decode_non_stream(request.app, api, decode_payload, request_id, max_try)

    except Exception as e:
        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        if exc_info:
            import traceback
            print("".join(traceback.format_exception(*exc_info)))
        raise


async def _handle_completions_pooling(api: str, request: Request,
                                      max_try: Optional[int] = None):
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())
        pooling_router: Optional[PoolingRouter] = getattr(request.app.state, "pooling_router", None)
        if pooling_router is None:
            raise RuntimeError("Pooling router requested but not initialized")
        stream = bool(req_data.get("stream"))
        return await pooling_router.submit(api, req_data, request_id, stream=stream, max_try=max_try)
    except Exception as e:  # pylint: disable=broad-except
        exc_info = sys.exc_info()
        print(f"Error occurred in pooling handler - {api} endpoint")
        print(e)
        if exc_info:
            import traceback
            print("".join(traceback.format_exception(*exc_info)))
        raise

@app.post("/v1/completions")
async def handle_completions(request: Request):
    if getattr(global_args, "enable_pooling", False):
        return await _handle_completions_pooling("/completions", request)
    return await _handle_completions("/completions", request)

@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    if getattr(global_args, "enable_pooling", False):
        return await _handle_completions_pooling("/chat/completions", request)
    return await _handle_completions("/chat/completions", request)

@app.get("/healthcheck")
async def healthcheck():
    """Simple endpoint to check if the server is running."""
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
    }

if __name__ == '__main__':
    global global_args
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
