import asyncio
import os
import httpx
import random
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from asyncio import Queue
from typing import Any, List, Tuple, Dict
import math
import time
from contextlib import asynccontextmanager
from abc import ABC
from enum import Enum
import json
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
try: 
    import SLOsServe_C
except ImportError:
    SLOsServe_C = None

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.sampling_params import SamplingParams
from vllm.outputs import RequestOutput
from vllm.inputs import TokensPrompt

from motivation.bench_api_server import Problem
from motivation.common import PerfModel

import logging

logger = logging.getLogger("SLOsServe.router.api_server")
logging.basicConfig(level=logging.INFO)

engine: AsyncLLM | None = None

async def send_request_to_service_engine(client_idx: int,
                                  req_data: dict,
                                  request_id: str):
    """
    Send a request to a service using a client from the pool.
    """
    extra_args = req_data['vllm_xargs'].copy()
    extra_args['kv_transfer_params'] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None
    }
    # req_data['vllm_xargs'].pop('prefill_ddl', None)
    outputs: List[RequestOutput] = []
    assert engine is not None
    async for output in engine.generate(
        prompt=req_data['prompt'] if isinstance(req_data['prompt'], str) else TokensPrompt(prompt_token_ids=req_data['prompt']),
        request_id=request_id,
        sampling_params=SamplingParams.from_optional(
            max_tokens=1,
            ignore_eos=True,
            extra_args=extra_args,
        ),
        data_parallel_rank=client_idx,
    ):
        outputs.append(output)
    assert len(outputs) == 1
    output = outputs[0]
    response = asdict(output.outputs[0]) if outputs else None
    if isinstance(output.kv_transfer_params, dict):
        response['kv_transfer_params'] = output.kv_transfer_params
    return response

async def stream_service_response_engine(client_idx, req_data: dict, request_id: str):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
    extra_args = req_data['vllm_xargs'].copy()
    kv_transfer_params = req_data.get('kv_transfer_params', None)
    if kv_transfer_params is not None:
        extra_args['kv_transfer_params'] = kv_transfer_params
    assert engine is not None
    n_tokens = 0
    text_len = 0
    async for output in engine.generate(
        prompt = req_data['prompt'] if isinstance(req_data['prompt'], str) else TokensPrompt(prompt_token_ids=req_data['prompt']),
        request_id = request_id,
        sampling_params = SamplingParams.from_optional(
            max_tokens = req_data['max_tokens'],
            extra_args = extra_args,
            ignore_eos = True
        ),
        data_parallel_rank=client_idx,
    ):
        completion = output.outputs[0]
        yield {
            'text': completion.text[text_len:],
            'finish_reason': completion.finish_reason,
            'stop_reason': completion.stop_reason,
            'token_ids': completion.token_ids[n_tokens:],
        }
        n_tokens = len(completion.token_ids)
        text_len = len(completion.text)

class RequestState(Enum):
    WAITING = 'waiting'
    PREFILL_REJECTED = 'prefill_rejected'
    PREFILL_FINISHED = 'prefill_finished'
    DECODE_FINISHED = 'decode_finished'
    TIMEOUT = 'timeout'

class RequestInstance:
    def __init__(self,
                 request_id: str | None = None,
                 payload: Any = None,
                 response_queue: Queue | None = None):
        self.request_id = request_id
        self.payload = payload
        self.payload.update({'request_id': request_id})
        self.response_queue = response_queue

        # Router State
        self.admitted: bool | None = None
        self.prefill_device_id: int = -1
        self.decode_device_id: int = -1

        # runtime state
        self.state: RequestState = RequestState.WAITING
    
    def is_finished(self):
        return self.state in [RequestState.DECODE_FINISHED, RequestState.TIMEOUT, RequestState.PREFILL_REJECTED]

class Router(ABC):
    def run(self, waiting_requests: List[RequestInstance],
            running_requests: List[RequestInstance]):
        raise NotImplementedError

    def update(self, request: RequestInstance, new_state: RequestState):
        pass
    
    def update_json(self, request_json: dict, i: int):
        return request_json

class AutoScalingRouter(Router):
    def __init__(self, n_devices: int, router_kwargs: str):
        self.n_devices = n_devices
        self.n_tries = defaultdict(int)
        self.round_robin_init = router_kwargs.get('round_robin_init', False)
        self.base_device_id = defaultdict(int)
        self.i = 0
        logger.info(f"AutoScalingRouter: n_devices = {n_devices}")
        
    def run(self, waiting_requests: List[RequestInstance],
            running_requests: List[RequestInstance]):
        for request in waiting_requests:
            request.admitted = True
            n_try = self.n_tries[request.request_id]
            if n_try >= self.n_devices - 1:
                request.admitted = False
                continue
            if n_try == 0 and self.round_robin_init:
                self.base_device_id[request.request_id] = self.i
                self.i += 1
            device_id_to_try = (self.base_device_id[request.request_id] + n_try) % self.n_devices
            request.prefill_device_id = request.decode_device_id = device_id_to_try
            self.n_tries[request.request_id] = n_try + 1
            
class RoundRobinRouter(Router):
    def __init__(self, n_devices: int, router_kwargs: str):
        self.n_devices = n_devices
        self.i = 0
    
    def run(self, waiting_requests: List[RequestInstance],
            running_requests: List[RequestInstance]):
        # logger.info(f"RoundRobinRouter: Assigning {len(waiting_requests)} waiting requests")
        for request in waiting_requests:
            request.admitted = True
            request.prefill_device_id = request.decode_device_id = self.i
            logger.info(f"Request {request.request_id} assigned to device {self.i}")
            self.i = (self.i + 1) % self.n_devices

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
        # logger.info(f"DisaggregatedRouter: Assigning {len(waiting_requests)} waiting requests")
        for request in waiting_requests:
            request.admitted = True
            request.prefill_device_id = self.prefill_i
            request.decode_device_id = self.decode_i + self.n_prefill_devices
            logger.info(f"Request {request.request_id} assigned to prefill_device {self.prefill_i}, decode_device {self.decode_i}")
            self.prefill_i = (self.prefill_i + 1) % self.n_prefill_devices
            self.decode_i = (self.decode_i + 1) % self.n_decode_devices
            
    def update_json(self, request_json: dict, i: int):
        new_request_json = request_json.copy()
        if i < self.n_prefill_devices:
            if new_request_json['scheduling_policy'] in ['vllm-fcfs', 'vllm-priority']:
                new_request_json['scheduling_kwargs']['max_num_batched_tokens'] = 16384
                new_request_json['scheduling_kwargs']['long_prefill_token_threshold'] = 4096
        return new_request_json

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
        # logger.info(f"DisaggregatedRouter: Assigning {len(waiting_requests)} waiting requests")
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

class SLOsServeRouter(Router):
    def __init__(self, n_devices: int, router_kwargs: str | dict):
        import json
        if isinstance(router_kwargs, str):
            router_kwargs = json.loads(router_kwargs)
        self.hardware_params = router_kwargs['hardware_params']
        self.tpot = router_kwargs['tpot']
        self.device_mems = [router_kwargs['device_mem'] for _ in range(n_devices)]
        self.block_size = router_kwargs['block_size']
        self.routing_overhead = router_kwargs['routing_overhead']
        self.router = SLOsServe_C.AdmCtrlRouter(n_devices, self.hardware_params, self.tpot)

    def get_req_data(self, request: RequestInstance):
        extra_args = request.payload['vllm_xargs']
        prefill_ddl = extra_args['prefill_ddl']
        input_length = extra_args['input_length']
        profit = extra_args['profit']
        prefill_mem = math.ceil(input_length / self.block_size)
        mem = math.ceil((input_length + request.payload['max_tokens']) / self.block_size)
        return prefill_ddl, input_length, profit, prefill_mem, mem

    def run(self, waiting_requests: List[RequestInstance],
            running_requests: List[RequestInstance]):
        # logger.info(f"SLOsServeRouter: Scheduling {len(waiting_requests)} waiting and {len(running_requests)} running requests")
        c_reqs = []
        for i, request in enumerate(waiting_requests + running_requests):
            # logger.info(f"Request {request.request_id} converting to SLOsServe_C.Request: i={i}, is_new_req={is_new_req}, prefill_ddl={prefill_ddl}, input_length={input_length}, profit={profit}, prefill_mem={prefill_mem}, mem={mem}, prefill_device_id={request.prefill_device_id}, decode_device_id={request.decode_device_id}")
            if i < len(waiting_requests):
                is_new_req = True
            else:
                is_new_req = False
            # logger.info(f"Request {request.request_id} getting req data")
            prefill_ddl, input_length, profit, prefill_mem, mem = self.get_req_data(request)
            
            c_req = SLOsServe_C.Request(
                id = str(i),
                is_new_req = is_new_req,
                ddl = prefill_ddl - self.routing_overhead,
                input_length = input_length,
                profit = profit,
                mem = mem,
                tpot_idx = 0,
                prefill_mem = prefill_mem,
                prefill_device_id = request.prefill_device_id,
                decode_device_id = request.decode_device_id,
                prefill_only = False
            )
            c_reqs.append(c_req)
            # logger.info(f"Request {request.request_id} converted to SLOsServe_C.Request: id={c_req.id}, is_new_req={c_req.is_new_req}, ddl={c_req.ddl}, input_length={c_req.input_length}, profit={c_req.profit}, mem={c_req.mem}, tpot_idx={c_req.tpot_idx}, prefill_mem={c_req.prefill_mem}, prefill_device_id={c_req.prefill_device_id}, decode_device_id={c_req.decode_device_id}, prefill_only={c_req.prefill_only}")
        
        # logger.info(f"Scheduling {len(c_reqs)} requests")
        outputs, batches = self.router.schedule(c_reqs, self.device_mems, time.time(), False)
        
        for i, output in enumerate(outputs):
            request = waiting_requests[i]
            request.admitted = output.admitted
            request.prefill_device_id = output.prefill_device_id
            request.decode_device_id = output.decode_device_id
            logger.info(f"Request {request.request_id} scheduling result: admitted={output.admitted}, prefill_device_id={output.prefill_device_id}, decode_device_id={output.decode_device_id}")
            if request.admitted:
                prefill_ddl, input_length, profit, prefill_mem, mem = self.get_req_data(request)
                if request.prefill_device_id == request.decode_device_id:
                    self.device_mems[request.prefill_device_id] -= mem
                else:
                    self.device_mems[request.prefill_device_id] -= prefill_mem
                    self.device_mems[request.decode_device_id] -= mem
                
        
    def update(self, request: RequestInstance, new_state: RequestState):
        # P rejected, P finished, D rejected, D finished
        prefill_ddl, input_length, profit, prefill_mem, mem = self.get_req_data(request)
        logger.info(f"Updating device memory for request {request.request_id} with state {request.state}")
        if new_state == RequestState.PREFILL_REJECTED:
            if request.prefill_device_id == request.decode_device_id:
                self.device_mems[request.prefill_device_id] += mem
            else:
                self.device_mems[request.prefill_device_id] += prefill_mem
                self.device_mems[request.decode_device_id] += mem
        elif new_state == RequestState.PREFILL_FINISHED:
            if request.prefill_device_id != request.decode_device_id:
                self.device_mems[request.decode_device_id] += mem
        elif new_state == RequestState.DECODE_FINISHED:
            self.device_mems[request.decode_device_id] += mem
        elif new_state == RequestState.TIMEOUT:
            # TODO: we should be more careful here.
            if request.prefill_device_id == request.decode_device_id:
                self.device_mems[request.prefill_device_id] += mem
            else:
                self.device_mems[request.prefill_device_id] += prefill_mem
                self.device_mems[request.decode_device_id] += mem
        else: 
            raise ValueError(f"Invalid request state: {request.state}")


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
        # if len(waiting_requests) > 0:
        #     logger.info(f"Renaming Router[run]: {self.devices}")
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
    
def create_router(t: str, n_devices: int, router_kwargs: str) -> Router:
    logger.info(f"Creating router of type {t} with {n_devices} devices and kwargs: {router_kwargs}")
    if t == 'round_robin':
        return RoundRobinRouter(n_devices, router_kwargs)
    elif 'disaggregated-edf' in t:
        return DisaggregatedEDFRouter(n_devices, router_kwargs)
    elif 'disaggregated' in t:
        return DisaggregatedRouter(n_devices, router_kwargs)
    elif t == 'slosserve':
        return SLOsServeRouter(n_devices, router_kwargs)
    elif t == 'renaming':
        return RenamingRouter(n_devices, router_kwargs)
    elif t == 'auto_scaling':
        return AutoScalingRouter(n_devices, router_kwargs)
    elif t == 'round_robin_retry':
        return AutoScalingRouter(n_devices, router_kwargs)
    else:
        raise ValueError(f"Invalid router type: {t}")

class Network:
    def __init__(self, bw_per_link_Bps=25e9,  # bytes/s per NVLink lane; adjust to your fabric
                 inj_cap_Bps=300e9,           # per-GPU injection cap (A100 NVSwitch ~300 GB/s)
                 ej_cap_Bps=300e9,            # per-GPU ejection cap
                 overhead_s=0.030,
                 n_devices=8,
                 model_name: str = 'Qwen/Qwen2.5-7B-Instruct'):
        self.bw_per_link = bw_per_link_Bps
        self.inj_cap = inj_cap_Bps
        self.ej_cap = ej_cap_Bps
        self.overhead = overhead_s
        self.n = n_devices
        # Next-free time for each resource
        self.link_free = {(i, j): 0.0 for i in range(n_devices) for j in range(n_devices) if i != j}
        self.inj_free  = [0.0] * n_devices
        self.ej_free   = [0.0] * n_devices
        # simple counters for concurrent use on ports (for naive rate sharing)
        self.inj_load  = [0] * n_devices
        self.ej_load   = [0] * n_devices
        self.per_token_bytes = {
            'Qwen/Qwen2.5-7B-Instruct': 53.4,
            'facebook/opt-125m': 34.329,
            # 'google/gemma-3-27b-it': 106.8,
        }.get(model_name, 53.4) * 1024

    def reset(self, n_devices=None):
        if n_devices is not None:
            self.n = n_devices
        self.link_free = {(i, j): 0.0 for i in range(self.n) for j in range(self.n) if i != j}
        self.inj_free  = [0.0] * self.n
        self.ej_free   = [0.0] * self.n
        self.inj_load  = [0] * self.n
        self.ej_load   = [0] * self.n

    def tx(self, src: int, dst: int, num_tokens: int, start_time: float) -> float:
        # Earliest time all three resources are free
        t0 = max(start_time, self.link_free[(src, dst)], self.inj_free[src], self.ej_free[dst])

        # Naive simultaneous-sharing model: if multiple transfers overlap at t0 on a port,
        # divide the port cap among them. (In a real sim you'd maintain active intervals.)
        inj_rate = self.inj_cap / max(1, self.inj_load[src])
        ej_rate  = self.ej_cap  / max(1, self.ej_load[dst])
        path_rate = min(self.bw_per_link, inj_rate, ej_rate)

        tx_time = self.overhead + num_tokens * self.per_token_bytes / path_rate

        t_done = t0 + tx_time
        # Update resource availability
        self.link_free[(src, dst)] = t_done
        self.inj_free[src] = t_done
        self.ej_free[dst]  = t_done
        return t_done

class RequestPool:
    def __init__(self, window_size: float,
                 router: Router,
                 clients: list = None,
                 enable_rerouting: bool = False,
                 enable_rescheduling: bool = False,
                 admission_mode: str = 'anytime',
                 mock_connector: bool = False):
        self.waiting_pool: List[RequestInstance] = []
        self.running_pool: List[RequestInstance] = []
        self.changed_requests: List[RequestInstance] = []
        self.window_size = window_size
        self.router = router
        self.clients = clients
        assert isinstance(self.clients, list)
        self.request_id = 0
        self._profile_events: List[Dict[str, Any]] = []
        self.routing_overhead = 0
        self.admission_mode = admission_mode
        self.enable_rerouting = enable_rerouting
        self.enable_rescheduling = enable_rescheduling
        self.mock_connector = mock_connector
        if mock_connector:
            self.network = Network(n_devices=len(clients))
        if isinstance(router, RenamingRouter):
            self.enable_rerouting = True
        self.running_tasks = []
    
    async def empty(self):
        while len(self.waiting_pool) > 0 or len(self.running_pool) > 0:
            await asyncio.sleep(0.1)

    def update_config(self, request_json: dict):
        self.reset()
        
        self.router = create_router(request_json['routing_policy'], 
                                    request_json['n_devices'],
                                    request_json['routing_kwargs'])
        self.admission_mode = request_json.get('admission_mode', 'arrival')
        try:
            self.routing_overhead = request_json['routing_kwargs'].get('routing_overhead', 0)
            self.enable_rerouting = request_json['routing_kwargs'].get('enable_rerouting', False)
            self.enable_rescheduling = request_json['routing_kwargs'].get('enable_rescheduling', False)
        except:
            self.routing_overhead = 0
        if request_json['routing_policy'] == 'renaming':
            self.enable_rerouting = True
            
        
    def reset(self):
        self.waiting_pool = []
        self.running_pool = []
        self.changed_requests = []
        self.request_id = 0
        self._profile_events = []
        self.enable_rerouting = False
        self.admission_mode = 'arrival'
        self.running_tasks = []
        if self.mock_connector:
            self.network.reset(len(self.clients))
    
    async def add_request(self, request: Request) -> Queue:
        response_queue = Queue()
        request_json = await request.json()
        current_time = time.time()
        request_id = request_json['vllm_xargs'].get('request_id', str(uuid.uuid4()))
        if 'prefill_ddl' not in request_json['vllm_xargs']:
            request_json['vllm_xargs']['prefill_ddl'] = current_time + request_json['vllm_xargs']['slo_ttft']
        request_instance = RequestInstance(request_id, request_json, response_queue)
        logger.info(f"Received new request: request_id={request_id}")
        self.request_id += 1
        self.waiting_pool.append(request_instance)
        
        self._profile_events.append({
            "event_type": "arrival-router",
            "device_id": -1,
            "timestamp": current_time,
            "request_id": request_id,
            "prefill_ddl": request_json['vllm_xargs'].get('prefill_ddl', 0),
            "profit": request_json['vllm_xargs'].get('profit', 1),
            "prompt_tokens": request_json['vllm_xargs'].get('input_length', 0),
            "max_tokens": request_json['max_tokens'],
        })
        
        async def gen():
            while True:
                chunk = await response_queue.get()
                if chunk is None:
                    yield b"[done]\n"
                    break
                yield b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
        
    def update_req_state(self, request: RequestInstance, state: RequestState):
        logger.info(f"RequestPool:[update_req_state]: {request}, {state}")
        self.router.update(request, state)
        request.state = state
        self.changed_requests.append(request)
    
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
        response = await send_request_to_service_engine(src_device_id, request_json, request_id)
        first_token_time = time.time()
        if 'kv_transfer_params' in response:
            request_json['kv_transfer_params'] = response['kv_transfer_params']
        # logger.info(f'request_json: {request_json}')
        first_decode_response_time = None
        async for data in stream_service_response_engine(dst_device_id, request_json, request_id):
            if first_decode_response_time is None:
                first_decode_response_time = time.time()
        
        return first_decode_response_time - first_token_time
    
    async def benchmark(self):
        from tqdm.asyncio import tqdm_asyncio
        from tqdm import tqdm

        async def benchmark_pair(src_device_id: int, dst_device_id: int):
            # Warmup phase with tqdm
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
                if i == j: continue
                await self.run_dummy_request(100, 10, i, j, request_json['model'])
        logger.info(f"Prewarming request: {request_json} took {time.time() - start_time} seconds")
    
    async def sync(self, timeout: float | None = None):
        start_time = time.time()
        while len(self.waiting_pool):
            await asyncio.sleep(1)
            if timeout is not None and time.time() - start_time > timeout:
                logger.warning(f"sync: Timed out after {timeout} seconds waiting for requests to finish")
                break
        remained_time = timeout - (time.time() - start_time)
        if remained_time <= 0:
            logger.warning(f"sync: Timed out after {timeout} seconds waiting for requests to finish")
            return
        corotines = [task for _, task, request in self.running_tasks if not request.is_finished()]
        if not corotines: return
        # Wait until all coroutines complete or until the timeout occurs
        try:
            await asyncio.wait_for(asyncio.gather(*corotines), timeout=remained_time)
        except asyncio.TimeoutError:
            logger.warning(f"sync: Timed out after {remained_time} seconds waiting for tasks")
        
    async def routing_loop(self):
        while True:
            # logger.info("Routing loop: sleeping for window_size=%s", self.window_size)
            await asyncio.sleep(self.window_size)
            it_start_time = time.time()
            # logger.info("Routing loop: running router")
            self.router.run(self.waiting_pool, self.running_pool)
            
            it_end_time = time.time()
            
            if len(self.waiting_pool) > 0:
                self._profile_events.append({
                    "event_type": "routing",
                    "timestamp": it_start_time,
                    "routing_overhead": it_end_time - it_start_time,
                    "schedules": {
                        request.request_id: {
                            "prefill_device_id": request.prefill_device_id,
                            "decode_device_id": request.decode_device_id,
                            "admitted": request.admitted,
                        } for request in self.waiting_pool
                    },
                    "device_id": -1,
                })
            
            remained_waiting_requests = []
            for request in self.waiting_pool:
                # assert request.admitted is not None
                if not request.admitted:
                    prefill_ddl = request.payload['vllm_xargs']['prefill_ddl']
                    if self.admission_mode == 'anytime' and \
                        (request.admitted is None) and \
                            time.time() < prefill_ddl: 
                        remained_waiting_requests.append(request)
                        continue
                    self._profile_events.append({   
                        "event_type": "finish",
                        "timestamp": it_start_time,
                        "device_id": -1,
                        "request_id": request.request_id,
                        "finish_reason": "router_rejection",
                    })
                    
                    logger.info(f"Request {request.request_id} not admitted due to , sending rejection")
                    await request.response_queue.put({"finish_reason":"rejected"})
                    await request.response_queue.put(None)
                    continue
                logger.info(f"Request {request.request_id} admitted, prefill_device_id={request.prefill_device_id}, decode_device_id={request.decode_device_id}")
                self._profile_events.append({
                    "event_type": "router_decision",
                    "timestamp": it_start_time,
                    "device_id": -1,
                    "request_id": request.request_id,
                    "prefill_device_id": request.prefill_device_id,
                    "decode_device_id": request.decode_device_id,
                })
                assert request.prefill_device_id is not None
                assert request.decode_device_id is not None
                logger.info(f"Request {request.request_id} admitted, prefill_device_id={request.prefill_device_id}, decode_device_id={request.decode_device_id}")
                self.running_pool.append(request)
                self.running_tasks.append((time.time(), asyncio.create_task(self.dispatch_request(request)), request))
            
            self.waiting_pool = remained_waiting_requests
            
            finished_requests = []
            
            for request in self.changed_requests:
                logger.info(f"Request {request.request_id} state changed to {request.state}")
                if request.state in [RequestState.PREFILL_REJECTED, 
                                        RequestState.DECODE_FINISHED,
                                        RequestState.TIMEOUT]:
                    finished_requests.append(request)
                                            
            self.changed_requests.clear()
            
            if finished_requests:
                self.running_pool = [request for request in self.running_pool if request not in finished_requests]
            
    async def dispatch_request(self, request: RequestInstance):
        assert request.admitted
        assert request.prefill_device_id is not None
        assert request.decode_device_id is not None
        logger.info(f"Dispatching request {request.request_id}: prefill_device_id={request.prefill_device_id}, decode_device_id={request.decode_device_id}")
        # TODO: the protocol should be defined
        if not self.enable_rerouting and request.prefill_device_id == request.decode_device_id:
            logger.info(f"Request {request.request_id} will be streamed from device {request.prefill_device_id}")
            self._profile_events.append({
                "event_type": "dispatch-both",
                "timestamp": time.time(),
                "request_id": request.request_id,
                "prefill_device_id": request.prefill_device_id,
                "decode_device_id": request.decode_device_id,
                "device_id": -1
            })
            is_rejected = False
            async for data in stream_service_response_engine(request.prefill_device_id,
                                                   request.payload,
                                                   request.request_id):
                await request.response_queue.put(data)
                if data['finish_reason'] == 'rejected':
                    logger.info(f"Request {request.request_id} was rejected at prefill stage")
                    is_rejected = True
            if is_rejected and self.enable_rescheduling:
                self.waiting_pool.append(request)
                self.update_req_state(request, RequestState.WAITING)
                self._profile_events.append({
                    "event_type": "rescheduling",
                    "timestamp": time.time(),
                    "request_id": request.request_id,
                    "prefill_device_id": request.prefill_device_id,
                    "decode_device_id": request.decode_device_id,
                    "device_id": -1
                })
                logger.info(f"Request {request.request_id} waiting for rescheduling")
            else:
                await request.response_queue.put(None)
                self.update_req_state(request, RequestState.DECODE_FINISHED)
                logger.info(f"Request {request.request_id} finished decode (single device)")
            return
        
        self._profile_events.append({
            "event_type": "dispatch-prefill",
            "timestamp": time.time(),
            "request_id": request.request_id,
            "prefill_device_id": request.prefill_device_id,
            "decode_device_id": request.decode_device_id,
            "device_id": -1
        })
        
        logger.info(f"Request {request.request_id} sending to prefill device {request.prefill_device_id}")
        request.payload['vllm_xargs']['prefill_ddl'] -= self.routing_overhead
        response = await send_request_to_service_engine(request.prefill_device_id,
                                                        request.payload,
                                                        request.request_id)
        request.payload['vllm_xargs']['prefill_ddl'] += self.routing_overhead
        
        if response.get('finish_reason') == 'rejected':
            logger.info(f"Request {request.request_id} was rejected at prefill stage")
            self.update_req_state(request, RequestState.PREFILL_REJECTED)
            await request.response_queue.put(None)
            return
    
        self.update_req_state(request, RequestState.PREFILL_FINISHED)
        
        self._profile_events.append({
            "event_type": "dispatch-decode",
            "timestamp": time.time(),
            "request_id": request.request_id,
            "prefill_device_id": request.prefill_device_id,
            "decode_device_id": request.decode_device_id,
            "device_id": -1
        })
        
        kv_transfer_params = response.get('kv_transfer_params', {})
        if kv_transfer_params:
            logger.info(f"Request {request.request_id} updating kv_transfer_params for decode")
            request.payload["kv_transfer_params"] = kv_transfer_params
            if self.mock_connector:
                assert 'num_tokens' in kv_transfer_params, f"num_tokens not found in kv_transfer_params: {kv_transfer_params}"
                assert 'dispatch_time' in kv_transfer_params, f"dispatch_time not found in kv_transfer_params: {kv_transfer_params}"
                kv_transfer_params['arrival_time'] = self.network.tx(request.prefill_device_id,
                                                                     request.decode_device_id,
                                                                     kv_transfer_params['num_tokens'], 
                                                                     kv_transfer_params['dispatch_time'])
            
        # Stream response from decode service
        logger.info(f"Request {request.request_id} streaming from decode device {request.decode_device_id}")
        is_rejected = False
        async for data in stream_service_response_engine(request.decode_device_id,
                                                    request.payload,
                                                    request_id=request.request_id):
            await request.response_queue.put(data)
            if data['finish_reason'] == 'rejected':
                logger.info(f"Request {request.request_id} was rejected at prefill stage")
                is_rejected = True
    
        if is_rejected and self.enable_rescheduling:
            self.waiting_pool.append(request)
            self.update_req_state(request, RequestState.WAITING)
            logger.info(f"Request {request.request_id} waiting for rescheduling")
        else:
            self.update_req_state(request, RequestState.DECODE_FINISHED)
            await request.response_queue.put(None)
            self.update_req_state(request, RequestState.DECODE_FINISHED)
            logger.info(f"Request {request.request_id} finished decode (disaggregated)")
            
request_pool: RequestPool | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    import traceback
    assert engine is not None
    async def routing_loop_with_error_monitoring():
        try:
            await request_pool.routing_loop()
        except Exception as e:
            logger.error(f"Exception in routing_loop: {e}\n{traceback.format_exc()}")

    task = asyncio.create_task(routing_loop_with_error_monitoring())
    yield
    engine.shutdown()
    task.cancel()
app = FastAPI(lifespan=lifespan)

@app.post("/v1/completions")
async def handle_completions(request: Request):
    if request_pool is None:
        logger.error("Request pool is not initialized")
        raise RuntimeError("Request pool is not initialized")
    logger.info("Received /v1/completions request")
    return await request_pool.add_request(request)

@app.post('/dump_profile_events')
async def dump_profile_events(request: Request):
    request_json = await request.json()
    await request_pool.sync(timeout = request_json.get('timeout', 10.0))
    filename = request_json.get('filename', 'profile_events.jsonl')
    logger.info(f"Dumping profile events to {filename}")
    import json
    if request_pool is None:
        logger.error("Request pool is not initialized")
        raise RuntimeError("Request pool is not initialized")
    all_events = []
    for i, client in enumerate(request_pool.clients):
        try:
            logger.info(f"Dumping profile events from client {client}")
            await engine.dump_profile_events(f'profile_events_{i}.jsonl', i)
            with open(f'profile_events_{i}.jsonl', 'r') as f:
                events = json.load(f)
            for event in events:
                event['device_id'] = i
            all_events.extend(events)
        except Exception as e:
            logger.error(f"Error dumping profile events from client {client}, {i}: {e}")
    all_events.extend(request_pool._profile_events)
    all_events.sort(key=lambda x: x['timestamp'])
    print(f'all_events: {all_events[:5]}')
    with open(filename, 'w') as f:
        json.dump(all_events, f, indent=4)
    logger.info(f"Dumped {len(all_events)} events to {filename}, example: {all_events[0]}")
    return JSONResponse(status_code=200, content={"message": "Profile events dumped."})

@app.post('/update_config')
async def update_config(request: Request):
    request_json = await request.json()
    request_pool.update_config(request_json)
    logger.info(f"Updated router: {request_pool.router}")
    await engine.update_config(request_json)
    for i, client in enumerate(request_pool.clients):
        request_json['engine_id'] = i
        request_json['is_mock_connector'] = args.mock_connector
        new_request_json = request_pool.router.update_json(request_json, i)
        print(new_request_json)
        await engine.update_config(new_request_json, i)
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
    clients = request_json['clients'].split(',')
    logger.info(f'existing clients: {request_pool.clients}, new clients: {clients}')
    global engine 
    if engine is None or set(clients) != set(request_pool.clients):
        if engine is not None: 
            engine.shutdown()
        start_engine(clients)
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
    --router_kwargs "{\"hardware_params\": [4.1e-5, 0, 1.3e-2], \"tpot\": 0.05, \"device_mem\": 16384, \"block_size\": 16}" \\
    --clients localhost:8100 localhost:8200
"""

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description=example_usage)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--window_size", type=float, default=0.1)
    parser.add_argument("--router", type=str, default="slo")
    parser.add_argument("--router_kwargs", type=str, default="{}")
    parser.add_argument("--clients", type=str, default=None)
    parser.add_argument("--enable_rerouting", action="store_true", default=False)
    parser.add_argument("--enable_rescheduling", action="store_true", default=False)
    parser.add_argument("--admission_mode", type=str, default="anytime")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--mock_connector", action="store_true", default=False)
    return parser.parse_args()



def start_engine(clients: list):
    n_devices = len(clients)
    print('clients: ', clients, 'n_devices: ', n_devices)
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import SamplingParams
    from vllm.usage.usage_lib import UsageContext
    from vllm.config import CompilationConfig, KVTransferConfig

    engine_args = AsyncEngineArgs(
        model = args.model_name,
        enable_chunked_prefill = True,
        enable_expert_parallel = False,
        max_num_batched_tokens = 16384,
        data_parallel_size = n_devices,
        max_num_seqs = 512,
        long_prefill_token_threshold = 16384,
        compilation_config = CompilationConfig(
            cudagraph_mode = 'FULL_AND_PIECEWISE',
        ),
        scheduler_cls = 'vllm.v1.core.sched.scheduler_adm_ctrl.SchedulerAdmCtrl',
        kv_transfer_config = KVTransferConfig(
            kv_connector = 'NixlConnector',
            kv_role = 'kv_both',
        ) if not args.mock_connector else None,
    )

    global engine

    engine = AsyncLLM.from_engine_args(engine_args)

    logger.info(f'Engine {engine} Started with clients: {clients}')

if __name__ == "__main__":
    import uvicorn
    start_time = time.time()
    args = parse_args()
    clients = args.clients.split(',')
    logger.info(f"Starting API server on {args.host}:{args.port} with router={args.router} and clients={args.clients}")
    if args.clients:
        start_engine(clients)

    router = create_router(args.router, len(clients), args.router_kwargs)
    request_pool = RequestPool(args.window_size, router, clients,
                               args.enable_rerouting, args.enable_rescheduling, args.admission_mode, args.mock_connector)
    start_up_time = time.time() - start_time
    logger.info(f"Start up time: {start_up_time} seconds")
    uvicorn.run(app, host=args.host, port=args.port)