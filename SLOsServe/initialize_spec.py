import torch 
from typing import Dict, Optional, Union, List, Tuple
import ray


from .backend import Backend 
from .initialize import ParallelConfig, get_parallel_config
from .models import get_model_config
from .engine_wrapper import EngineWrapper

from .scheduler import ParaConfig

def init_spec_engines(
    base_model: str,
    spec_model: str|None,
    para_config: ParaConfig,
    dtype: torch.dtype,
    backend: Backend,
    block_size: int, 
    seed: Optional[int] = None,
    profile: bool = False,
    use_cuda_graph: bool = False,
    cache_type: str = 'linear',
    port: int = 29500,
    verbose: bool = False
) -> List[List[ray.ObjectRef]]:

    from vllm.config import ParallelConfig, ModelConfig, LoadConfig, DeviceConfig

    print('initializing devices...', para_config.world_size)
    
    device_config = DeviceConfig('cuda')
    import os
    load_config = LoadConfig(download_dir=f'/mnt/model_weights')
    # ray.init(ignore)
    # initialize_ray_cluster(parallel_config=parallel_config)
    # ray.init(runtime_env={"env_vars": {"RAY_DEBUG": "1"} })
    ray.init()
    
    engines = [ray.remote(
        num_cpus = 0, num_gpus = 1
    )(EngineWrapper).remote('spec_engine') for _ in range(para_config.world_size)]
    
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible_devices is None:
        cuda_visible_devices = ','.join(map(str, range(para_config.world_size)))
    ray.get([engine.update_environment_variables.remote(
        {'CUDA_VISIBLE_DEVICES': cuda_visible_devices}
    ) for engine in engines])

    ray.get([engine.init_engine.remote() for engine in engines])

    ray.get([engine.execute_method.remote('init_device', 
                                   rank=rank, 
                                   world_size=len(engines), 
                                   local_gpu_indices = list(range(para_config.world_size)), 
                                   distributed_init_method='env://', 
                                   local_rank = rank,
                                   seed = seed,
                                   port = port,
                                   profile = profile) 
            for rank, engine in enumerate(engines)])
    
    # exit(0)

    print('loading model...')

    num_blocks = None
    for model_tag, para_config_vllm in [(base_model, ParallelConfig(
                                        tensor_parallel_size=para_config.tp,
                                        pipeline_parallel_size=para_config.pp,
                                        data_parallel_size=para_config.dp,
                                        worker_use_ray=True
                                    )),
                                    (spec_model, ParallelConfig(
                                        tensor_parallel_size=1,
                                        pipeline_parallel_size=1,
                                        data_parallel_size=para_config.world_size,
                                        worker_use_ray=True
                                    ))]:
        if model_tag is None: continue

        model_config_vllm = ModelConfig(
            model = model_tag,
            tokenizer = model_tag,
            tokenizer_mode = 'auto', 
            trust_remote_code = True, 
            dtype = dtype,
            seed = 1234 
        )
        
        num_blockss: List[int] = ray.get([engine.execute_method.remote('load_model',
            model_config=model_config_vllm, 
            device_config=device_config, 
            load_config=load_config,
            parallel_config =para_config_vllm,
            backend=backend,
            block_size=block_size,
            use_cuda_graph = use_cuda_graph,
            cache_type = cache_type,
            num_blocks = num_blocks)
            for engine in engines])
        
        num_blocks = num_blockss[0]

        for num_blocks, engine\
            in zip(num_blockss, engines):
            if not hasattr(engine, 'num_block') or\
            engine.num_block > num_blocks:
                engine.num_block = num_blocks
    
    ray.get([
        engine.execute_method.remote('init', 
                                     base_model_tag = base_model, 
                                     spec_model_tag = spec_model,
                                     verbose = verbose)
    for engine in engines])

    ray.get([
        engine.execute_method.remote('display')
    for engine in engines])

    n_devices_per_replica = para_config.tp * para_config.pp
    return [engines[i:i+n_devices_per_replica] for i \
            in range(0, len(engines), n_devices_per_replica)]

def init_spec(
    base_model: str,
    spec_model: str,
    dtype: Union[str, torch.dtype],
    para_config: ParaConfig,
    seed: Optional[int] = None,
    backend_tag: str = 'vllm', 
    profile: bool = False,
    block_size: int = 16,
    use_cuda_graph: bool = False,
    cache_type: str = 'linear',
    port = 29500,
    verbose: bool = False,
) -> List[List[ray.ObjectRef]]:
    from SLOsServe.utils import get_torch_dtype_from_str
    if isinstance(dtype, str):
        torch_dtype = get_torch_dtype_from_str(dtype)
    else: torch_dtype = dtype
    if backend_tag == 'hf':
        backend = Backend.HUGGINGFACE
    elif backend_tag == 'vllm':
        backend = Backend.VLLM
    else: raise NotImplementedError(f"backend {backend_tag} not supported")
        
    replicas = init_spec_engines(
        para_config=para_config,
        dtype = torch_dtype,
        backend = backend,
        block_size = block_size,
        seed = seed, 
        profile = profile,
        use_cuda_graph=use_cuda_graph,
        cache_type=cache_type,
        base_model = base_model, 
        spec_model = spec_model,
        port = port,
        verbose = verbose
    )
    # exit(0)
        
    return replicas
    
    