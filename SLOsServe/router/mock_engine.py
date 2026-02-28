import asyncio
# from asyncio import Queue
import copy
import time
import random
import json
import logging
from typing import AsyncGenerator, Any
import os 
import ray
from ray.util.queue import Queue as RayQueue

import vllm
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.engine import FinishReason
from vllm.utils import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import init_none_hash, get_request_block_hasher
from vllm.outputs import RequestOutput, CompletionOutput

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


def _task_done(task: asyncio.Task):
    try:
        task.result()  # re-raises exception if task failed
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Background coroutine crashed")

logger = setup_logger("SLOsServe.router.mock_engine", os.getenv("SLOSSERVE_LOG_LEVEL", "INFO"))
DEBUG = False
@ray.remote(max_concurrency=1024)
class MockEngineCore:
    def __init__(self,
                 model_name: str,
                 mock_connector: bool,
                 output_queue: RayQueue,
                 num_gpu_blocks: int = 23949,
                 device_id: int = -1,
                 execplan_bus=None):
        self.model_name = model_name
        self.mock_connector = mock_connector
        self.output_queue = output_queue
        self.device_id = device_id
        self.execplan_bus = execplan_bus
        # NOTE: Attributes such as `engine_id`, `request_block_hasher`,
        # `_profile_events`, `n_arrived`, `n_finished_reqs`, and `n_rejected_reqs`
        # are referenced later but never initialized here. Any code that uses
        # `MockEngineWorker` must set them explicitly or `add_request` will raise.
        
        from vllm.v1.kv_cache_interface import KVCacheConfig
        from vllm.v1.core.sched.scheduler_adm_ctrl import SchedulerAdmCtrl
        from vllm.config import VllmConfig
        from unittest.mock import MagicMock
        from vllm.v1.structured_output import StructuredOutputManager
        from SLOsServe.perf_model import PerfModel
        
        attrs = {}
        import pickle
        for config_name in ['model_config', 'cache_config', 'parallel_config', 'scheduler_config', 'device_config',
                            'load_config', 'lora_config', 'speculative_config', 'decoding_config', 'observability_config',
                            'quant_config', 'kv_transfer_config', 'additional_config', 'kv_events_config', 'additional_config', 'instance_id']:
            with open(f'assets/stub/SchedulerAdmCtrl_{config_name}.pkl', 'rb') as f:
                attrs[config_name] = pickle.load(f)
        
        with open(f'assets/stub/SchedulerAdmCtrl_kv_cache_config.pkl', 'rb') as f:
            kv_cache_configs = pickle.load(f)
            kv_cache_config = KVCacheConfig(
                num_blocks=num_gpu_blocks,
                kv_cache_tensors=[],
                kv_cache_groups=kv_cache_configs['kv_cache_groups'],
            )

        # attrs.update({'model_config': None})
        VllmConfig.__post_init__ = MagicMock()
        vllm_config = VllmConfig(
            **attrs            
        )
        structured_output_manager = StructuredOutputManager(vllm_config)
        
        vllm_config.model_config.model_name = model_name
        
        vllm_config.scheduler_config.scheduler_cls = 'vllm.v1.core.sched.scheduler_adm_ctrl.SchedulerAdmCtrl'
        from vllm.v1.core.sched.scheduler_adm_ctrl import SchedulerAdmCtrl
        self.scheduler = SchedulerAdmCtrl(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            include_finished_set = True
        )
        logger.info(f'Scheduler created: {self.scheduler}')
        self.perf_model = PerfModel.get_perf_model(model_name)
        
        block_size = self.scheduler.vllm_config.cache_config.block_size
        caching_hash_fn = get_hash_fn_by_name(
            self.scheduler.vllm_config.cache_config.prefix_caching_hash_algo)
        init_none_hash(caching_hash_fn)

        self.request_block_hasher = get_request_block_hasher(
            block_size, caching_hash_fn)
        
        self._profile_events = []
        self.loop_task: asyncio.Task | None = None
        
        self.n_arrived_reqs = 0
        self.n_finished_reqs = 0
        self.n_rejected_reqs = 0
        
        self.batch_id = 0
        
        logger.info(f'MockEngineCore initialized: {self.model_name}, {self.mock_connector}')

    def ping(self):
        return True

    async def start_loop(self):
        if self.loop_task is None:
            logger.info('Starting loop task')
            self.loop_task = asyncio.create_task(self.loop())
            self.loop_task.add_done_callback(_task_done)
    async def shutdown(self):
        if self.loop_task is not None:
            self.loop_task.cancel()
            try:
                await self.loop_task
            except asyncio.CancelledError:
                logger.info("Loop task cancelled successfully")
            except Exception as e:
                logger.error(f"Error while shutting down loop task: {e}")
            finally:
                self.loop_task = None
    
    async def health_check(self) -> dict[str, Any]:
        """Return the status of the background loop task for diagnostics.

        When the loop task is not done yet, we still expose useful
        counters to help debug what the mock engine is doing.
        """
        status: dict[str, Any] = {
            "loop_task_created": self.loop_task is not None,
            "loop_task_done": False,
            "loop_task_cancelled": False,
            "loop_task_exception": None,
            # Extra debug fields:
            "n_arrived_reqs": self.n_arrived_reqs,
            "n_finished_reqs": self.n_finished_reqs,
            "n_rejected_reqs": self.n_rejected_reqs,
        }
        loop_task = self.loop_task
        if loop_task is None:
            return status

        status["loop_task_done"] = loop_task.done()
        status["loop_task_cancelled"] = loop_task.cancelled()
        if status["loop_task_done"] and not status["loop_task_cancelled"]:
            exc = loop_task.exception()
            if exc is not None:
                status["loop_task_exception"] = repr(exc)
        return status

    async def update_config(self, config: dict):
        self._profile_events = []
        self.n_arrived_reqs = 0
        self.n_finished_reqs = 0
        self.n_rejected_reqs = 0
        self.batch_id = 0
        for k, v in config.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if hasattr(self.scheduler.scheduler_config, kk):
                        setattr(self.scheduler.scheduler_config, kk, vv)
            elif hasattr(self.scheduler.scheduler_config, k):
                setattr(self.scheduler.scheduler_config, k, v)
        logger.info(f'Scheduler config updated: {self.scheduler.scheduler_config}')
        self.scheduler.reset(self._profile_events)

    @property
    def on(self) -> bool: 
        return (self.n_arrived_reqs - self.n_finished_reqs) > 0

    async def get_load_statistics(self, t: float = 1) -> list[dict[str, Any]]:
        return self.scheduler.get_load_statistics(t)

    async def dump_profile_events(self, filename: str):
        logger.info(f'dumping profile events to {filename}')
        if hasattr(self.scheduler, '_req_cached_tokens'):
            for event in self._profile_events:
                if event['event_type'] == 'arrival':
                    event['num_cached_tokens'] = self.scheduler._req_cached_tokens.get(event['request_id'], 0)
            
        with open(filename, 'w') as f:
            json.dump(self._profile_events, f, indent=4)
        logger.info(f'Saved {len(self._profile_events)} events to {filename}')
        self._profile_events.clear()

    async def abort_request(self, request_id: str):
        self.scheduler.abort_requests([request_id])

    async def loop(self):
        try:
            
            print('mock engine started')
            while True:
                if not self.on:
                    await asyncio.sleep(0.003)
                    continue
                start_time = time.time()
                await asyncio.sleep(0.0)
                if self.batch_id % 100 == 0:
                    stats = self.scheduler.make_stats()
                    print(f'[MockEngineCore] batch_id={self.batch_id}, stats={stats}, n_arrived_reqs={self.n_arrived_reqs}, n_finished_reqs={self.n_finished_reqs}, n_rejected_reqs={self.n_rejected_reqs}')
                self.batch_id += 1
                scheduling_start = time.time()
                scheduler_output = self.scheduler.schedule()
                scheduling_overhead = time.time() - scheduling_start
                publish_start = time.time()
                if self.execplan_bus is not None:
                    self.execplan_bus.publish.remote(
                        self.device_id, time.time(),
                        self.scheduler.get_exec_plan())
                publish_overhead = time.time() - publish_start
                # print(f'[MockEngineCore] batch_id={self.batch_id}, number_scheduled_tokens: {scheduler_output.num_scheduled_tokens}')
                rejs = []
                for req in self.scheduler.get_rejected_requests():
                    self._profile_events.append({
                        "event_type": "finish",
                        "request_id": req.request_id,
                        "timestamp": time.time(),
                        "finish_reason": "rejected",
                        "scheduling_overhead": scheduling_overhead,
                    })
                    
                    rejs.append({
                        'request_id': req.request_id, 
                        'finish_reason': 'rejected'
                    })
                    
                    self.n_finished_reqs += 1 
                    self.n_rejected_reqs += 1 
                    
                    # output_queue = self.output_queues.pop(req.request_id)
                    # output_queue.put_nowait({'finish_reason': 'rejected'})
                    # output_queue.put_nowait(None)
                if len(rejs):
                    self.output_queue.put_nowait(rejs)
                
                req_ids = []
                req_id_to_index = {}
                sampled_token_ids = []
                batch = []

                for req_id in scheduler_output.num_scheduled_tokens:
                    request = self.scheduler.requests[req_id]
                    req_ids.append(req_id)
                    batch.append((request.num_computed_tokens, scheduler_output.num_scheduled_tokens[req_id]))
                    req_id_to_index[req_id] = len(req_ids) - 1
                    if request.num_computed_tokens >= request.num_prompt_tokens:
                        sampled_token_ids.append([random.randint(0, 1000)])
                    else: 
                        sampled_token_ids.append([])
                
                model_runner_output = ModelRunnerOutput(
                    req_ids = req_ids,
                    req_id_to_index = req_id_to_index,
                    sampled_token_ids = sampled_token_ids,
                    logprobs = None,
                    prompt_logprobs_dict = {},
                    pooler_output = [],
                )
                launch_start = time.time()
                batch_time = self.perf_model.get_batch_time(batch)
                
                # we mimic the current runtime by blocking the loop for the batch time.
                await asyncio.sleep(batch_time)
                
                # time.sleep(batch_time)
                output_processing_start = time.time()
                
                outputs = self.scheduler.update_from_output(scheduler_output, model_runner_output)
                responses = []
                for output in outputs.values():
                    for req_output in output.outputs:
                        logger.debug(f"Sending output for request_id={req_output.request_id}")
                        responses.append({
                            'request_id': req_output.request_id,
                            'new_token_ids': req_output.new_token_ids,
                            'finish_reason': str(req_output.finish_reason) if req_output.finish_reason is not None else None,
                            'stop_reason': str(req_output.stop_reason) if req_output.stop_reason is not None else None,
                            'kv_transfer_params': req_output.kv_transfer_params,
                            'num_computed_tokens': req_output.num_computed_tokens,
                            'timestamp': time.time(),
                        })
                        # print(f'Sending output for request_id={req_output.request_id}, putting {to_put}')
                        # output_queue = self.output_queues[req_output.request_id]
                        # output_queue.put_nowait({
                        #     'new_token_ids': req_output.new_token_ids,
                        #     'finish_reason': str(req_output.finish_reason),
                        #     'stop_reason': str(req_output.stop_reason),
                        #     'kv_transfer_params': req_output.kv_transfer_params,
                        #     'num_computed_tokens': req_output.num_computed_tokens,
                        #     'timestamp': time.time(),
                        # })
                        if req_output.finish_reason is not None:
                            # print(f"Request {req_output.request_id} finished batch_id={self.batch_id}")
                            self._profile_events.append({
                                "event_type": "finish",
                                "request_id": req_output.request_id,
                                "timestamp": time.time(),
                                "finish_reason": str(req_output.finish_reason),
                            })
                            # print(f"Request {req_output.request_id} finished {req_output.finish_reason}. Closing output queue.")
                            # self.output_queues.pop(req_output.request_id)
                            # output_queue.put_nowait(None)
                            self.n_finished_reqs += 1
                            if req_output.finish_reason == FinishReason.REJECTED:
                                self.n_rejected_reqs += 1
                if len(responses):
                    self.output_queue.put_nowait(responses)
                
                elapsed = time.time() - start_time
                output_processing_elapsed = time.time() - output_processing_start
                # print(f'waiting {batch_time} seconds, batch takes {elapsed} seconds')
                if scheduler_output.total_num_scheduled_tokens > 0:
                    self._profile_events.append({
                        "event_type": "batch",
                        "batch_id": self.batch_id,
                        "timestamp": time.time(),
                        "elapsed": elapsed,
                        "req_ids": [new_req.req_id for new_req in scheduler_output.scheduled_new_reqs] + scheduler_output.scheduled_cached_reqs.req_ids,
                        "num_computed_tokens": [new_req.num_computed_tokens for new_req in scheduler_output.scheduled_new_reqs] + scheduler_output.scheduled_cached_reqs.num_computed_tokens,
                        "num_scheduled_tokens": scheduler_output.num_scheduled_tokens,
                        "scheduling_overhead": scheduling_overhead,
                        "between_batch_time": elapsed,
                        "output_processing_elapsed": output_processing_elapsed,
                        "estimated_time": batch_time,
                        "rejected_reqs": [r['request_id'] for r in rejs],
                        "publish_overhead": publish_overhead,
                        "extra_args": {
                            'to_schedule': scheduling_start - start_time, 
                            'to_launch': launch_start - start_time, 
                            'to_finish': output_processing_start - start_time,
                            'to_est_finish': launch_start + batch_time - start_time
                        }
                    })
        except Exception as e:
            print(f'Exception in loop: {e}')
            raise

    async def add_request(self, prompt: str | list[int], 
                          request_id: str,
                          sampling_params: SamplingParams,):
        '''
            SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0
        , top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=True, max_tokens=319, min_tokens=0, log
        probs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args={
        'input_length': 41, 'output_length': 319, 'zero_load_ttft': 0.016091, 'slo_ttft': 0.098273, 'profit': 1.0, 'request_id': 'a3cb6618-42bb-435a-bf6f-e488a56f222d', 'router_arrival_time': 1770191392.005625, 'prefill_ddl': 1770191392.103898})
        '''
        # print('dispatch', request_id, time.time(), 'mockenginecore')
        # print(f"Request {request_id} arrived batch_id={self.batch_id}, il={sampling_params.extra_args['input_length']}, ol={sampling_params.extra_args['output_length']}")
        add_request_start = time.time()
        input_length = sampling_params.extra_args.get('input_length', 0)
        decode_only = sampling_params.extra_args.get('decode_only', False)
        prefill_only = sampling_params.extra_args.get('prefill_only', False)
        
        self.n_arrived_reqs += 1
        current_time = time.time()
        request = vllm.v1.request.Request(
            request_id = request_id,
            prompt_token_ids = [random.randint(0, 1000) for _ in range(input_length + (1 if decode_only else 0))],
            sampling_params = sampling_params,
            priority = 0,
            arrival_time = current_time,
            multi_modal_kwargs = None,
            multi_modal_hashes = None,
            multi_modal_placeholders = None,
            pooling_params = None,
            eos_token_id = None,
            client_index = 0,
            block_hasher = self.request_block_hasher 
        )
        
        self._profile_events.append({
            "event_type": "arrival",
            "zero_load_ttft": request.sampling_params.extra_args.get('zero_load_ttft', 0),
            "request_id": request.request_id,
            "prompt_tokens": request.num_prompt_tokens,
            "max_tokens": request.max_tokens,
            "timestamp": time.time(),
            "prefill_ddl": self.scheduler._get_prefill_ddl(request),
            "profit": request.sampling_params.extra_args.get('profit', 1),
            "prefill_only": prefill_only,
            "decode_only": decode_only,
            "add_req_time": request.arrival_time,
        })
        
        if decode_only: request.num_computed_tokens = input_length
        
        # print(f"Adding request: {request}")
        admitted = self.scheduler.add_request(request)
        
        print(f'add_request takes {time.time() - add_request_start}s')
        # print(f"Request {request_id} arrived batch_id={self.batch_id}, il={sampling_params.extra_args['input_length']}, ol={sampling_params.extra_args['output_length']}, admitted = {admitted}")
        if not admitted:
            self.n_finished_reqs += 1
            self.n_rejected_reqs += 1
            scheduler_overhead = time.time() - current_time
            logger.info(f"Request {request.request_id} rejected.")
            self._profile_events.append({
                "event_type": "finish",
                "request_id": request.request_id,
                "timestamp": time.time(),
                "finish_reason": "rejected-arrival",
                "scheduling_overhead": scheduler_overhead,
            })
            return False
        
        logger.debug(f"Request {request.request_id} added to scheduler.")

        # self.output_queues[request.request_id] = out_q
        return True

class MockEngine:
    def __init__(self,
                 model_name: str,
                 mock_connector: bool,
                 num_gpu_blocks: int = 23949,
                 device_id: int = -1,
                 execplan_bus=None):
        self._shared_out_q = RayQueue(maxsize = 8192)
        self._local_queues: dict[str, asyncio.Queue] = {}
        '''
        def __init__(self,
                 model_name: str,
                 mock_connector: bool,
                 output_queue: RayQueue,
                 num_gpu_blocks: int = 23949,
                 device_id: int = -1,
                 execplan_bus=None):
        '''
        self.engine_core = MockEngineCore.remote(
            model_name = model_name,
            mock_connector=mock_connector,
            output_queue = self._shared_out_q,
            num_gpu_blocks = num_gpu_blocks,
            device_id = device_id,
            execplan_bus = execplan_bus,
        )
        self.device_id = device_id
        ray.get(self.engine_core.ping.remote())
        logger.info(f'MockEngineCore created: {self.engine_core}')
        ray.get(self.engine_core.start_loop.remote())
        self._demux_task = asyncio.create_task(self._demux_loop())
        self._demux_task.add_done_callback(_task_done)
        self._profile_events = []
        logger.info(f'MockEngine started: {self.engine_core}')
    
    async def _demux_loop(self):
        while True:
            payload = await self._shared_out_q.get_async()
            if payload:
                self._handle_payload(payload)

            # drain
            while True:
                try:
                    payload = self._shared_out_q.get_nowait()
                except Exception:
                    break
                if payload:
                    self._handle_payload(payload)

    def _handle_payload(self, payload):
        for item in payload:
            rid = item.get("request_id")
            if not rid:
                continue
            q = self._local_queues.get(rid)
            if q is None:
                continue
            q.put_nowait(item)  # ✅ don’t await if you want speed

    def generate(self,
        prompt: str | list[int],
        request_id: str, 
        sampling_params: SamplingParams,
    ) -> AsyncGenerator[RequestOutput, None]:
        
        async def generate_stream():
            if DEBUG:
                self._profile_events.append({
                    'event_type': 'arrive_mock_engine',
                    'request_id': request_id,
                    'device_id': self.device_id, 
                    'timestamp': time.time()
                })
            print('dispatch', request_id, time.time(), 'mockengine')
            _q = asyncio.Queue(maxsize = 512)
            self._local_queues[request_id] = _q
            # TODO: ignore the prompt for now. update later. 
            admitted = await self.engine_core.add_request.remote(
                "", request_id, sampling_params
            )
            
            if DEBUG:
                self._profile_events.append({
                    'event_type': 'mock_engine_get_response',
                    'timestamp': time.time(),
                    'request_id': request_id, 
                    'device_id': self.device_id
                })
            if not admitted:
                self._local_queues.pop(request_id)
                yield RequestOutput(
                    request_id=request_id,
                    prompt=None,
                    prompt_token_ids=None,
                    prompt_logprobs=None,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text="",
                            token_ids=[],
                            cumulative_logprob=None,
                            logprobs=None,
                            finish_reason="rejected",
                            stop_reason=None,
                        )
                    ],
                    finished=True,
                )
                return

            while True:
                chunk = await _q.get()
                yield RequestOutput(
                    request_id=request_id,
                    prompt=None,
                    prompt_token_ids=None,
                    prompt_logprobs=None,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text=chunk.get("text", ""),
                            token_ids=chunk.get("new_token_ids", []),
                            cumulative_logprob=None,
                            logprobs=None,
                            finish_reason=chunk.get("finish_reason", None),
                            stop_reason=chunk.get("stop_reason", None),
                            num_computed_tokens=chunk.get("num_computed_tokens", None)
                        )
                    ],
                    finished=chunk.get("finish_reason", None) is not None,
                    kv_transfer_params=chunk.get('kv_transfer_params', None)
                )
                if chunk.get('finish_reason') is not None: break
            self._local_queues.pop(request_id)
        return generate_stream()

    async def update_config(self, request_json: dict):
        self._profile_events.clear()
        await self.engine_core.update_config.remote(request_json)

    async def dump_profile_events(self, path: str):
        await self.engine_core.dump_profile_events.remote(path)
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        data.extend(self._profile_events)
        with open(path, 'w') as f:
            json.dump(data, f)
    
    async def shutdown(self):
        await self.engine_core.shutdown.remote()
        if self._demux_task is not None:
            self._demux_task.cancel()
            try:
                await self._demux_task
            except asyncio.CancelledError:
                pass
            self._demux_task = None
        
    async def get_load_statistics(self, n: int = 100):
        return await self.engine_core.get_load_statistics.remote(n)

    async def abort(self, request_id: str):
        await self.engine_core.abort_request.remote(request_id)
    
    async def health_check(self) -> dict[str, Any]:
        """Expose the loop task status from the Ray actor."""
        return await self.engine_core.health_check.remote()

if __name__ == "__main__":
    ray.init()
    engine = MockEngine(model_name='Qwen/Qwen2.5-7B-Instruct', mock_connector=True)
    async def test_generate():
        async for chunk in engine.generate(
            prompt = 'Hello, world!',
            request_id = 'test',
            sampling_params = SamplingParams(
                max_tokens = 10,
                extra_args = {
                    'input_length': 100,
                    'output_length': 10,
                    'prefill_ddl': time.time() + 10,
                    'slo_ttft': 10,
                    'profit': 1,
                    'request_id': 'test',
                    'ignore_eos': True,
                },
            ),
        ):
            print(chunk)
            
    asyncio.run(test_generate())
