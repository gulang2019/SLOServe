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
from SLOsServe.router.batchplan_bus import normalize_batchplan

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


logger = setup_logger("SLOsServe.router.mock_engine", os.getenv("SLOSSERVE_LOG_LEVEL", "INFO"))

@ray.remote(max_concurrency=1024)
class MockEngineCore:
    def __init__(self,
                 model_name: str,
                 mock_connector: bool,
                 num_gpu_blocks: int = 23949,
                 device_id: int = -1,
                 batchplan_bus=None):
        self.model_name = model_name
        self.mock_connector = mock_connector
        self.device_id = device_id
        self.batchplan_bus = batchplan_bus
        self.output_queues: dict[str, RayQueue] = {}
        # NOTE: Attributes such as `engine_id`, `request_block_hasher`,
        # `_profile_events`, `n_arrived`, `n_finished_reqs`, and `n_rejected_reqs`
        # are referenced later but never initialized here. Any code that uses
        # `MockEngineWorker` must set them explicitly or `add_request` will raise.
        
        from vllm.v1.kv_cache_interface import KVCacheConfig
        from vllm.v1.core.sched.scheduler_adm_ctrl import SchedulerAdmCtrl
        from vllm.config import VllmConfig
        from unittest.mock import MagicMock
        from vllm.v1.structured_output import StructuredOutputManager
        from motivation.common import PerfModel
        
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
        self._batchplan: list[dict[str, Any]] = []
        self._batchplan_version: int = 0
        self._batchplan_timestamp: float = time.time()
        
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
        for k, v in config.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if hasattr(self.scheduler.scheduler_config, kk):
                        setattr(self.scheduler.scheduler_config, kk, vv)
            elif hasattr(self.scheduler.scheduler_config, k):
                setattr(self.scheduler.scheduler_config, k, v)
        logger.info(f'Scheduler config updated: {self.scheduler.scheduler_config}')
        self.scheduler.reset(self._profile_events)
        self._batchplan = []
        self._batchplan_timestamp = time.time()
        self._batchplan_version += 1
        self._publish_batchplan_to_bus()

    @property
    def on(self) -> bool: 
        return (self.n_arrived_reqs - self.n_finished_reqs) > 0

    async def get_load_statistics(self, t: float = 1) -> list[dict[str, Any]]:
        return self.scheduler.get_load_statistics(t)
    
    async def get_batchplan(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "version": self._batchplan_version,
            "timestamp": self._batchplan_timestamp,
            "batchplan": copy.deepcopy(self._batchplan),
        }

    def _publish_batchplan_to_bus(self):
        if self.batchplan_bus is None:
            return
        self.batchplan_bus.publish.remote(
            self.device_id,
            self._batchplan_version,
            self._batchplan_timestamp,
            self._batchplan,
        )

    def _build_batchplan(self,
                         scheduler_output,
                         first_batch_duration: float) -> list[dict[str, Any]]:
        capacity = getattr(self.scheduler.scheduler_config,
                           "max_num_batched_tokens",
                           None)

        planned_batches: list[dict[str, Any]] = []

        # Try scheduler-provided future plans first (if exposed by this version).
        raw_future = getattr(scheduler_output, "future_batches", None)
        if isinstance(raw_future, list):
            for raw_batch in raw_future:
                if not isinstance(raw_batch, dict):
                    continue
                allocated_tokens = raw_batch.get("allocated_tokens")
                if not isinstance(allocated_tokens, dict):
                    allocated_tokens = raw_batch.get("num_scheduled_tokens", {})
                if not isinstance(allocated_tokens, dict):
                    continue
                batch = {
                    "estimated_time": raw_batch.get("estimated_time", 0.0),
                    "allocated_tokens": allocated_tokens,
                }
                batch_capacity = raw_batch.get("capacity", capacity)
                if batch_capacity is not None:
                    batch["capacity"] = batch_capacity
                planned_batches.append(batch)

        # Fallback: only current executing batch is known.
        if not planned_batches:
            planned_batches.append({
                "estimated_time": 0.0,
                "allocated_tokens": scheduler_output.num_scheduled_tokens,
                "capacity": capacity,
                "duration": first_batch_duration,
            })

        return normalize_batchplan(planned_batches)

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
                    if self._batchplan:
                        self._batchplan = []
                        self._batchplan_timestamp = time.time()
                        self._batchplan_version += 1
                        self._publish_batchplan_to_bus()
                    await asyncio.sleep(0.003)
                    continue
                start_time = time.time()
                await asyncio.sleep(0.0)
                if self.batch_id % 100 == 0:
                    stats = self.scheduler.make_stats()
                    print(f'[MockEngineCore] batch_id={self.batch_id}, stats={stats}, n_arrived_reqs={self.n_arrived_reqs}, n_finished_reqs={self.n_finished_reqs}, n_rejected_reqs={self.n_rejected_reqs}')
                non_loop_time = time.time() - start_time 
                
                self.batch_id += 1
                scheduling_start = time.time()
                scheduler_output = self.scheduler.schedule()
                scheduling_overhead = time.time() - scheduling_start
                
                # print(f'[MockEngineCore] batch_id={self.batch_id}, number_scheduled_tokens: {scheduler_output.num_scheduled_tokens}')
                
                for req in self.scheduler.get_rejected_requests():
                    self._profile_events.append({
                        "event_type": "finish",
                        "request_id": req.request_id,
                        "timestamp": time.time(),
                        "finish_reason": "rejected",
                        "scheduling_overhead": scheduling_overhead,
                    })
                    
                    output_queue = self.output_queues.pop(req.request_id)
                    output_queue.put_nowait({'finish_reason': 'rejected'})
                    output_queue.put_nowait(None)
                
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
                
                batch_time = self.perf_model.get_batch_time(batch)
                if scheduler_output.total_num_scheduled_tokens > 0:
                    self._batchplan = self._build_batchplan(scheduler_output, batch_time)
                    self._batchplan_timestamp = time.time()
                    self._batchplan_version += 1
                    self._publish_batchplan_to_bus()
                elif self._batchplan:
                    self._batchplan = []
                    self._batchplan_timestamp = time.time()
                    self._batchplan_version += 1
                    self._publish_batchplan_to_bus()
                
                # we mimic the current runtime by blocking the loop for the batch time.
                await asyncio.sleep(batch_time)
                
                # time.sleep(batch_time)
                output_processing_start = time.time()
                
                outputs = self.scheduler.update_from_output(scheduler_output, model_runner_output)
                
                for output in outputs.values():
                    for req_output in output.outputs:
                        logger.debug(f"Sending output for request_id={req_output.request_id}")
                        # print(f'Sending output for request_id={req_output.request_id}, putting {to_put}')
                        output_queue = self.output_queues[req_output.request_id]
                        output_queue.put_nowait({
                            'new_token_ids': req_output.new_token_ids,
                            'finish_reason': str(req_output.finish_reason),
                            'stop_reason': str(req_output.stop_reason),
                            'kv_transfer_params': req_output.kv_transfer_params,
                            'num_computed_tokens': req_output.num_computed_tokens,
                            'timestamp': time.time(),
                        })
                        if req_output.finish_reason is not None:
                            # print(f"Request {req_output.request_id} finished batch_id={self.batch_id}")
                            self._profile_events.append({
                                "event_type": "finish",
                                "request_id": req_output.request_id,
                                "timestamp": time.time(),
                                "finish_reason": str(req_output.finish_reason),
                            })
                            # print(f"Request {req_output.request_id} finished {req_output.finish_reason}. Closing output queue.")
                            self.output_queues.pop(req_output.request_id)
                            output_queue.put_nowait(None)
                            self.n_finished_reqs += 1
                            if req_output.finish_reason == FinishReason.REJECTED:
                                self.n_rejected_reqs += 1
                elapsed = time.time() - start_time
                output_processing_elapsed = time.time() - output_processing_start
                print(f'waiting {batch_time} seconds, batch takes {elapsed} seconds')
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
                        "estimated_time": batch_time
                    })
        except Exception as e:
            print(f'Exception in loop: {e}')
            raise

    async def add_request(self, prompt: str | list[int], 
                          request_id: str,
                          sampling_params: SamplingParams,
                          out_q: RayQueue):
        '''
            SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0
        , top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=True, max_tokens=319, min_tokens=0, log
        probs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args={
        'input_length': 41, 'output_length': 319, 'zero_load_ttft': 0.016091, 'slo_ttft': 0.098273, 'profit': 1.0, 'request_id': 'a3cb6618-42bb-435a-bf6f-e488a56f222d', 'router_arrival_time': 1770191392.005625, 'prefill_ddl': 1770191392.103898})
        '''
        print(f"Request {request_id} arrived batch_id={self.batch_id}, il={sampling_params.extra_args['input_length']}, ol={sampling_params.extra_args['output_length']}")
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

        
        self.output_queues[request.request_id] = out_q
        return True

class MockEngine:
    def __init__(self,
                 model_name: str,
                 mock_connector: bool,
                 num_gpu_blocks: int = 23949,
                 device_id: int = -1,
                 batchplan_bus=None):
        self.engine_core = MockEngineCore.remote(
            model_name,
            mock_connector,
            num_gpu_blocks,
            device_id,
            batchplan_bus,
        )
        ray.get(self.engine_core.ping.remote())
        logger.info(f'MockEngineCore created: {self.engine_core}')
        ray.get(self.engine_core.start_loop.remote())
        logger.info(f'MockEngine started: {self.engine_core}')
        
    def generate(self,
        prompt: str | list[int],
        request_id: str, 
        sampling_params: SamplingParams,
    ) -> AsyncGenerator[RequestOutput, None]:
        
        async def generate_stream():
            out_q = RayQueue(maxsize = 512)
            admitted = await self.engine_core.add_request.remote(
                prompt, request_id, sampling_params, out_q
            )
            if not admitted:
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
                chunk = await out_q.get_async()
                if chunk is None:
                    break
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
                )

        return generate_stream()
    
    async def update_config(self, request_json: dict):
        await self.engine_core.update_config.remote(request_json)
        
    async def dump_profile_events(self, path: str):
        await self.engine_core.dump_profile_events.remote(path)
    
    async def shutdown(self):
        await self.engine_core.shutdown.remote()
    
    async def get_load_statistics(self, n: int = 100):
        return await self.engine_core.get_load_statistics.remote(n)
    
    async def get_batchplan(self):
        return await self.engine_core.get_batchplan.remote()
    
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
