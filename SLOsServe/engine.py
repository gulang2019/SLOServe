from typing import List, Optional, Any, Tuple, Dict
import ray.util.collective as collective
from ray.util.queue import Queue
import torch
import logging 
import time 

from vllm.worker.worker import init_worker_distributed_environment
from vllm.config import ParallelConfig, ModelConfig, LoadConfig, DeviceConfig
from vllm.utils import CudaMemoryProfiler
from vllm.model_executor.model_loader import get_model
from vllm.distributed.parallel_state import get_world_group 

from .ops import OpCode, OP_CLASSES
from .object import ObjectImplDict, ObjectDeviceRef
from .backend import Backend 
from .models import ModelImpl, get_model_config
from .programs.batchables import *

logger = logging.getLogger(__name__)

class Engine:
    object_dict: ObjectImplDict
    backend_comm_inited: bool            

    def init_device(self, 
                    *,
            parallel_config: ParallelConfig, 
            rank: int,
            world_size: int, 
            local_gpu_indices = List[int],
            distributed_init_method: Optional[str] = None,
            local_rank: int = -1,
            seed: Optional[int] = None,
            profile: bool = False):
        
        self.parallel_config = parallel_config
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.local_rank = local_rank
        self.profile = profile
        self.profile_logs = []
        self.model_impls: Dict[str, ModelImpl] = {} # Unique Model 
        import os 
        print('torch.cuda_visible_devices', os.environ.get('CUDA_VISIBLE_DEVICES', None), torch.cuda.is_available())
        print('torch.num_gpus: ', torch.cuda.device_count())
        os.environ['LOCAL_RANK'] = str(local_rank)
        print('local_gpu_indices', local_gpu_indices)
        os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        self.device = torch.device(f'cuda:{self.local_rank}')
        torch.cuda.set_device(self.device)
        
        init_worker_distributed_environment(
            self.parallel_config,
            self.rank,
            self.distributed_init_method, 
            self.local_rank
        )

        if seed is not None:
            from SLOsServe.utils import set_random_seed
            set_random_seed(seed)

    def set_profile(self, profile):
        print (f'Engine {self.rank} Update profile: {profile}')
        self.profile = profile

    def _cache_creator(self, 
                       model_tag: str, 
                       num_tokens: int, 
                       cache_tensors: List[torch.Tensor]):
        assert model_tag in self.model_impls 
        model_impl: ModelImpl = self.model_impls[model_tag]
        return model_impl.cache_manager.new(
            num_tokens,
            cache_tensors
        )

    def send_batched(self, dst: int, refs: List[ObjectDeviceRef], keep_olds: List[bool]):
        '''
        To support sending of KVCache,
        we need to find the cache manager located on the other server;
        We need to make model visible to the engine
        '''
        if self.profile:
            torch.cuda.synchronize(device = self.device)
            start_time = time.perf_counter()
        from .utils import object_to_type_meta 
        data = [self.object_dict[obj] if keep_old else self.object_dict.pop(obj) 
                for obj, keep_old in zip(refs, keep_olds)]
        meta, tensors = object_to_type_meta(data)
        
        tensor_dict = {str(_): tensor for _, tensor in enumerate(tensors)}
        tensor_dict['meta'] = meta
        get_world_group().send_tensor_dict(tensor_dict, dst)

        if self.profile:
            torch.cuda.synchronize(device = self.device)
            self.profile_logs.append((start_time, time.perf_counter(), 'SEND', len(refs)))
    
    def recv_batched(self, src: int, refs: List[ObjectDeviceRef]):
        if self.profile:
            torch.cuda.synchronize(device = self.device)
            start_time = time.perf_counter()            

        from .utils import type_meta_to_object 
        data = get_world_group().recv_tensor_dict(src)
        meta = data['meta']
        tensors = [data[str(_)] for _ in range(len(data) - 1)]
        obj_list = type_meta_to_object(meta, tensors, self._cache_creator)
        assert isinstance(obj_list, list) and len(obj_list) == len(refs)
        
        for ref, obj in zip(refs, obj_list):
            self.object_dict[ref] = obj
        
        if self.profile:
            torch.cuda.synchronize(device = self.device)
            self.profile_logs.append((start_time, time.perf_counter(), 'RECV', len(refs)))

    def send(self, obj, dst: int):
        from .utils import object_to_type_meta 
        data = self.object_dict[obj]
        meta, tensors = object_to_type_meta(data)
        tensor_dict = {str(_): tensor for _, tensor in enumerate(tensors)}
        tensor_dict[meta] = meta 
        self.object_dict.pop(obj)
        get_world_group().send_tensor_dict(tensor_dict, dst)

    def recv(self, obj, src: int): 
        from .utils import type_meta_to_object 
        data = get_world_group().recv_tensor_dict(src)
        meta = data['meta']
        tensors = [data[str(_)] for _ in range(len(data) - 1)]
        self.object_dict[obj] = type_meta_to_object(meta, tensors)

    def load_model(self, 
                   model_config: ModelConfig,
                   device_config: DeviceConfig,
                   load_config: LoadConfig,
                   parallel_config: ParallelConfig,
                   ref: ObjectDeviceRef,
                   backend: Backend,
                   block_size: int,
                   use_cuda_graph: bool,
                   cache_type: str):
        with CudaMemoryProfiler() as m:
            if backend == Backend.HUGGINGFACE:
                from transformers import AutoModelForCausalLM
                module = AutoModelForCausalLM.from_pretrained(
                    model_config.model, 
                    torch_dtype = model_config.dtype, 
                    cache_dir = load_config.download_dir)
                module.eval()
                module.cuda()
            elif backend == Backend.VLLM:
                module = get_model(
                    model_config=model_config,
                    device_config=device_config,
                    load_config=load_config,
                    lora_config=None,
                    multimodal_config=None,
                    parallel_config=parallel_config,
                    scheduler_config=None,
                    cache_config=None,
                )
        
        dyserve_model_config = get_model_config(model_config.model)

        assert model_config.model not in self.model_impls
        
        self.model_impls[model_config.model] = self.object_dict[ref] = ModelImpl(
            module = module, 
            name = model_config.model, 
            embed_size = dyserve_model_config.embed_size,
            local_num_heads = dyserve_model_config.num_heads // parallel_config.tensor_parallel_size,
            local_num_layers= dyserve_model_config.n_layer // parallel_config.pipeline_parallel_size,
            backend = backend,  
            dtype = model_config.dtype, 
            memory_usage = m.consumed_memory,
            max_seq_len=dyserve_model_config.max_seq_len,
            head_size = dyserve_model_config.embed_size // dyserve_model_config.num_heads,
            block_size=block_size,
            device = self.device,
            n_params=dyserve_model_config.n_param,
            vocab_size=dyserve_model_config.vocab_size,
            intermidiate_size=dyserve_model_config.intermidiate_size,
            use_cuda_graph=use_cuda_graph,
            pp = parallel_config.pipeline_parallel_size, 
            tp = parallel_config.tensor_parallel_size,
            cache_type=cache_type
        )
        
        self.dtype = model_config.dtype

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model %s weights from %s took %.4f GB",
                    model_config.model, 
                    backend.name, 
                    self.model_memory_usage / float(2**30))
        
    def set_output_queue(self, 
                         output_queue: Queue):
        self.output_queue = output_queue
        from .ops import GetOp
        GetOp.queue = output_queue

    def init_tokenizer(self, model_tag, obj):
        from transformers import AutoTokenizer
        self.object_dict[obj] = AutoTokenizer.from_pretrained(model_tag)

    def setup(self, rank, world_size, seed):
        self.rank = rank 
        self.world_size = world_size 
        collective.init_collective_group(world_size, rank, "nccl", "default")
        if seed is not None:
            from SLOsServe.utils import set_random_seed
            set_random_seed(seed)
        return True

    def set(self, object_ref: ObjectDeviceRef, object):
        if isinstance(object, torch.Tensor):
            object = object.cuda()
        self.object_dict[object_ref] = object

    def _status(self):
        objs = {}
        cache_status = {}
        from .models import ModelImpl 
        for obj_id, obj in self.object_dict.items():
            if isinstance(obj, ModelImpl):
                objs[obj.name] = objs.get(obj.name, 0) + 1
            else: 
                objs[type(obj)] = objs.get(type(obj), 0) + 1
        for model_impl in self.model_impls.values():
            cache_status.update(model_impl.cache_manager.get_status())
        return cache_status, objs

    def print_status(self):
        cache_status, objs, batches = self._status()
        print(f'[Engine {self.rank} cuda:{self.local_rank}] {cache_status} {objs} {batches}')

    def report(self):
        cache_status, objs = self._status()
        return cache_status, objs, self.profile_logs
    
    def __init__(self):
        self.object_dict = ObjectImplDict()

    def execute(self, opcode: OpCode, batched_args: List[Any]):
        if self.profile:
            torch.cuda.synchronize(device = self.device)
            start_time = time.perf_counter()
        op = OP_CLASSES[opcode]
        # TODO(Siyuan): The current impl does not use any object controllor
        job_name = op.get_profile_log(opcode, self.object_dict, batched_args)
        if op.has_batched_impl():
            op.batch_forward(self.object_dict, batched_args)
        else:
            for args in batched_args:
                op.forward(self.object_dict, args)
        if self.profile:
            torch.cuda.synchronize(device = self.device)
            self.profile_logs.append((start_time, time.perf_counter(), job_name, len(batched_args)))

    def reset(self):
        self.profile_logs = []
