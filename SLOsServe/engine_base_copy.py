import torch 
from typing import List, Optional, Dict
import logging 

logger = logging.getLogger(__name__)


from vllm.config import ParallelConfig, ModelConfig, LoadConfig, DeviceConfig, CacheConfig, SchedulerConfig
# from vllm.utils import CudaMemoryProfiler
from vllm.model_executor.model_loader import get_model
from .vllm_utils import is_vllm_v1, CudaMemoryProfiler

from .backend import Backend
from .models import ModelImpl, get_model_config

from ray.util.queue import Queue

# if is_vllm_v1():
#     from vllm.worker.worker import init_worker_distributed_environment_by_para_config
# else:
from vllm.worker.worker import init_worker_distributed_environment
if is_vllm_v1():
    from vllm.config import set_current_vllm_config


class BaseEngine:
    # def set_output_queue(self, 
    #                      output_queue: ):
    #     self.output_queue = output_queue
    def set_queue(self, queue: Queue):
        self.queue = queue
     
    def init_device(self, 
                    *,
            rank: int,
            world_size: int, 
            local_gpu_indices: List[int],
            distributed_init_method: Optional[str] = None,
            local_rank: int = -1,
            seed: Optional[int] = None,
            port: Optional[int] = None,
            profile: bool = False):
        
        self.rank = rank
        self.world_size = world_size
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
        if port is None: port = 29500
        os.environ['MASTER_PORT'] = str(port)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        self.device = torch.device(f'cuda:{self.local_rank}')
        torch.cuda.set_device(self.device)
        
        if seed is not None:
            from SLOsServe.utils import set_random_seed
            set_random_seed(seed)

    def load_model(self, 
                   model_config: ModelConfig,
                   device_config: DeviceConfig,
                   load_config: LoadConfig,
                   parallel_config: ParallelConfig,
                   backend: Backend,
                   block_size: int,
                   use_cuda_graph: bool,
                   cache_type: str,
                   num_blocks: int | None = None,
                   gpu_memory_utilization: float = 0.8):
        
        
        
        if is_vllm_v1():
            cache_config = CacheConfig(
                block_size = block_size,
                gpu_memory_utilization = gpu_memory_utilization
            )
            from vllm.config import CompilationConfig
            compiler_config = CompilationConfig(
                use_cudagraph=use_cuda_graph
            )
            from vllm.config import VllmConfig
            vllm_config = VllmConfig(model_config = model_config, 
                                            cache_config = cache_config,
                                            parallel_config = parallel_config,
                                            scheduler_config = SchedulerConfig(), 
                                            device_config = device_config,
                                            load_config = load_config,
                                            lora_config = None,
                                            multimodal_config = None)
            with set_current_vllm_config(vllm_config):
                init_worker_distributed_environment(
                    vllm_config,
                    self.rank,
                    self.distributed_init_method, 
                    self.local_rank,
                    model_config.model
                )
        else: 
            vllm_config = None
            cache_config = CacheConfig(
                block_size = block_size,
                gpu_memory_utilization = gpu_memory_utilization,
                swap_space = 0,
                cache_dtype = 'auto'
            )
            init_worker_distributed_environment(
                parallel_config,
                self.rank,
                self.distributed_init_method, 
                self.local_rank,
                model_config.model
            )

        from vllm.distributed import get_tp_group, get_pp_group
        print(f'RANK {self.rank}> loading {model_config.model}, TP: {get_tp_group().ranks}, PP: {get_pp_group().ranks}, config: {parallel_config}')

        with CudaMemoryProfiler() as m:
            if backend == Backend.VLLM:
                if is_vllm_v1():
                    module = get_model(vllm_config = vllm_config)
                else:
                    module = get_model(
                        model_config=model_config,
                        device_config=device_config,
                        load_config=load_config,
                        lora_config=None,
                        multimodal_config=None,
                        parallel_config=parallel_config,
                        scheduler_config=None,
                        cache_config=cache_config,
                    )
            else: raise NotImplemented
        
        dyserve_model_config = get_model_config(model_config.model)

        assert model_config.model not in self.model_impls
        
        self.model_impls[model_config.model] = model_impl = ModelImpl(
            module = module, 
            name = model_config.model, 
            embed_size = dyserve_model_config.embed_size,
            local_num_heads = dyserve_model_config.num_heads // parallel_config.tensor_parallel_size,
            local_num_layers= dyserve_model_config.n_layer // parallel_config.pipeline_parallel_size,
            backend = backend,  
            dtype = model_config.dtype, 
            memory_usage = m.consumed_memory,
            max_seq_len=dyserve_model_config.max_seq_len,
            head_size = dyserve_model_config.head_dim,
            block_size=block_size,
            device = self.device,
            n_params=dyserve_model_config.n_param,
            vocab_size=dyserve_model_config.vocab_size,
            intermidiate_size=dyserve_model_config.intermidiate_size,
            use_cuda_graph=use_cuda_graph,
            pp = parallel_config.pipeline_parallel_size, 
            tp = parallel_config.tensor_parallel_size,
            cache_type=cache_type,
            num_blocks = num_blocks,
            vllm_config = vllm_config
        )
        
        self.dtype = model_config.dtype

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model %s weights from %s took %.4f GB",
                    model_config.model, 
                    backend.name, 
                    self.model_memory_usage / float(2**30))
        
        return model_impl.cache_manager.num_blocks