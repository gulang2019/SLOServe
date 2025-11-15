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
try: 
    import SLOsServe_C
except ImportError:
    SLOsServe_C = None

from motivation.bench_api_server import Problem
from motivation.common import PerfModel

import logging

logger = logging.getLogger("SLOsServe.router.api_server")
logging.basicConfig(level=logging.INFO)

async def send_request_to_service(client: httpx.AsyncClient,
                                  req_data: dict,
                                  request_id: str):
    """
    Send a request to a service using a client from the pool.
    """
    req_data = req_data.copy()
    req_data['kv_transfer_params'] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None
    }
    req_data['request_id'] = request_id
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }
    # req_data['vllm_xargs'].pop('prefill_ddl', None)
    logger.info(f"Sending request {request_id} with req_data: {req_data}")
    response = await client.post('/v1/completions',
                                json=req_data,
                                headers=headers)
    # response.raise_for_status()

    return response

async def stream_service_response(client: httpx.AsyncClient,
                                  req_data: dict, request_id: str):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }
    logger.info(f"Streaming response for request {request_id} with req_data: {req_data}")
    async with client.stream("POST",
                            '/v1/completions',
                            json=req_data,
                            headers=headers) as response:
        # response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk

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

from dataclasses import dataclass

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
    
class RequestPool:
    def __init__(self, window_size: float, timeout: float,
                 router: Router,
                 clients: List[httpx.AsyncClient],
                 enable_rerouting: bool,
                 enable_rescheduling: bool,
                 admission_mode: str):
        self.waiting_pool: List[RequestInstance] = []
        self.running_pool: List[RequestInstance] = []
        self.changed_requests: List[RequestInstance] = []
        self.window_size = window_size
        self.timeout = timeout
        self.router = router
        self.clients = clients
        self.request_id = 0
        self._profile_events: List[Dict[str, Any]] = []
        self.routing_overhead = 0
        self.admission_mode = admission_mode
        self.enable_rerouting = enable_rerouting
        self.enable_rescheduling = enable_rescheduling
        if isinstance(router, RenamingRouter):
            self.enable_rerouting = True
    
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
        
        async def generate_stream():
            while True: 
                chunk = await response_queue.get()
                if chunk is None:
                    break
                yield chunk
        
        return StreamingResponse(generate_stream(), media_type="application/json")
    
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
            }
        }
        response = await send_request_to_service(self.clients[src_device_id], request_json, request_id)
        response_json = response.json()
        # logger.info(f'response_json: {response_json}')
        assert not response_json.get('finish_reason') == 'rejected'
        if 'kv_transfer_params' in response_json:
            request_json['kv_transfer_params'] = response_json['kv_transfer_params']
        # logger.info(f'prewarm _req_data: {_req_data}')``
        try:
            # Wrap the async generator with a task and handle timeout properly
            async def consume_stream():
                async for chunk in stream_service_response(self.clients[dst_device_id], request_json, request_id):
                    first_response_time = time.time()
            await asyncio.wait_for(consume_stream(), timeout=60)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout {src_device_id} ->{dst_device_id}")
        except Exception as e:
            logger.exception(f"Error {src_device_id} ->{dst_device_id}: {e}")
        return request_json
    
    async def warmup(self, request_json: dict):
        start_time = time.time()
        logger.info(f"Prewarming request: {request_json}")
        
        
        import copy
        successful_prewarms = set()
        for i in range(len(self.clients)):
            for j in range(len(self.clients)):
                if i == j: continue
                request_id = str(uuid.uuid4())
                logger.info(f"Prewarming client {i} -> {j}")
                req_data = {
                    'model': request_json['model'],
                    'max_tokens': 100,
                    'prompt': [random.randint(1000, 2000) for _ in range(100)],
                    "stream": True,
                    "ignore_eos": True,
                    'vllm_xargs': {
                        'prefill_ddl': time.time() + 100,
                        'input_length': 100,
                        'output_length': 10,
                        'profit': 1,
                    }
                }
                response = await send_request_to_service(self.clients[i], req_data, request_id)
                response_json = response.json()
                # logger.info(f'response_json: {response_json}')
                assert not response_json.get('finish_reason') == 'rejected'
                if 'kv_transfer_params' in response_json:
                    req_data['kv_transfer_params'] = response_json['kv_transfer_params']
                # logger.info(f'prewarm _req_data: {_req_data}')``
                try:
                 # Wrap the async generator with a task and handle timeout properly
                    async def consume_stream():
                        async for chunk in stream_service_response(self.clients[j], req_data, request_id):
                            first_response_time = time.time()     
                            break                       

                    await asyncio.wait_for(consume_stream(), timeout=60)
                    successful_prewarms.add((i, j))
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout {i} ->{j}")
                except Exception as e:
                    logger.exception(f"Error {i} ->{j}: {e}")
        logger.info(f"Successful prewarms: {successful_prewarms}")
        logger.info(f"Prewarming request: {request_json} took {time.time() - start_time} seconds")
    
    
    async def routing_loop(self):
        running_tasks = []
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
                    await request.response_queue.put(b'{"finish_reason":"rejected"}')
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
                running_tasks.append((time.time(), asyncio.create_task(self.dispatch_request(request)), request))
            
            self.waiting_pool = remained_waiting_requests
            
            # update the running tasks 
            idx = 0
            while idx < len(running_tasks) and running_tasks[idx][0] < time.time() - self.timeout:
                idx += 1
            if idx > 0:
                logger.info(f"Routing loop: waiting for {idx} finished running tasks")
            
            finished_requests = []
            
            for _, task, request in running_tasks[:idx]:
                if not task.done():
                    self._profile_events.append({
                        "event_type": "finish",
                        "timestamp": time.time(),
                        "device_id": -1,
                        "request_id": request.request_id,
                        "finish_reason": "router_timeout",
                    })
                    logger.warning(f"Cancelling task {task} due to timeout")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        logger.info(f"Task {task} cancelled successfully")
                    finally:
                        await request.response_queue.put(None)
                        self.update_req_state(request, RequestState.TIMEOUT)

            running_tasks = running_tasks[idx:]                
            
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
            async for chunk in stream_service_response(self.clients[request.prefill_device_id],
                                                   request.payload,
                                                   request.request_id):
                await request.response_queue.put(chunk)
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8")
                # logger.info(f'received chunk: {chunk}')
                if chunk.startswith('data: {'):
                    chunk = json.loads(chunk[5:])
                    if chunk['choices'][0].get('finish_reason') == 'rejected':
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
        response = await send_request_to_service(self.clients[request.prefill_device_id],
                                                 request.payload, request.request_id)
        request.payload['vllm_xargs']['prefill_ddl'] += self.routing_overhead
        
        # Extract the needed fields
        response_json = response.json()
                
        if response_json.get('finish_reason') == 'rejected':
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
        
        kv_transfer_params = response_json.get('kv_transfer_params', {})
        if kv_transfer_params:
            logger.info(f"Request {request.request_id} updating kv_transfer_params for decode")
            request.payload["kv_transfer_params"] = kv_transfer_params
            
        # Stream response from decode service
        logger.info(f"Request {request.request_id} streaming from decode device {request.decode_device_id}")
        is_rejected = False
        async for chunk in stream_service_response(self.clients[request.decode_device_id],
                                                    request.payload,
                                                    request_id=request.request_id):
            await request.response_queue.put(chunk)
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")
            # print(f"chunk: {chunk}")
            if chunk.startswith('data: {'):
                chunk = json.loads(chunk[5:])
                if chunk['choices'][0].get('finish_reason') == 'rejected':
                    logger.info(f"Request {request.request_id} was rejected at decode stage")
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

    async def routing_loop_with_error_monitoring():
        try:
            await request_pool.routing_loop()
        except Exception as e:
            logger.error(f"Exception in routing_loop: {e}\n{traceback.format_exc()}")

    task = asyncio.create_task(routing_loop_with_error_monitoring())
    yield
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
    await request_pool.empty()
    request_json = await request.json()
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
            await client.post('/dump_profile_events', json={'filename': f'profile_events_{i}.jsonl'})
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
    for i, client in enumerate(request_pool.clients):
        request_json['engine_id'] = i
        new_request_json = request_pool.router.update_json(request_json, i)
        
        print(new_request_json)
        response = await client.post('/update_config', json=new_request_json)
        logger.info(f"Updated config for client {i}: {response}")
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
    logger.info(f"Updated clients: {clients}")
    request_pool.clients = [httpx.AsyncClient(timeout=None, base_url=client) for client in clients]
    return JSONResponse(status_code=200, content={"message": "Clients updated."})

example_usage = """
example:
python -m SLOsServe.router.api_server \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --window_size 1.0 \\
    --timeout 10.0 \\
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
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--router", type=str, default="slo")
    parser.add_argument("--router_kwargs", type=str, default="{}")
    parser.add_argument("--clients", type=str, default="8500:1")
    parser.add_argument("--enable_rerouting", action="store_true", default=False)
    parser.add_argument("--enable_rescheduling", action="store_true", default=False)
    parser.add_argument("--admission_mode", type=str, default="anytime")
    return parser.parse_args()

if __name__ == "__main__":
    import uvicorn
    args = parse_args()
    logger.info(f"Starting API server on {args.host}:{args.port} with router={args.router} and clients={args.clients}")
    if ':' in args.clients:
        client_port_start, n_client = args.clients.split(':')
        client_port_start = int(client_port_start)
        n_client = int(n_client)
        clients = [httpx.AsyncClient(timeout=None, base_url=f"http://localhost:{client_port_start + i}") for i in range(n_client)]
    elif ',' in args.clients:
        clients = [httpx.AsyncClient(timeout=None, base_url=f"http://localhost:{client}") for client in args.clients.split(',')]
        n_client = len(clients)
    else:
        clients = [httpx.AsyncClient(timeout=None, base_url=f"http://localhost:{args.clients}")]
    print(f"clients: {clients}")
    print(f"len(clients): {len(clients)}")
    router = create_router(args.router, len(clients), args.router_kwargs)
    request_pool = RequestPool(args.window_size, args.timeout, router, clients, args.enable_rerouting, args.enable_rescheduling, args.admission_mode)
    uvicorn.run(app, host=args.host, port=args.port)