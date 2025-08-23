from typing import List, Tuple, Callable, Dict, Optional, Union
import torch
import logging 
from dataclasses import dataclass

from SLOsServe.utils import (
    get_device_config
)
from SLOsServe.models import (
    get_model_config
)
from SLOsServe.object import (
    Placement,
    ObjectStatus,
    ModelRef,
    TokenizerRef 
)
from SLOsServe.context import (
    GlobalContext 
)
from SLOsServe.executor import Executor
from SLOsServe.backend import Backend 

logger = logging.getLogger(__name__)

@dataclass
class ParallelConfig: 
    dp: int 
    tp: int 
    pp: int 
    
def get_parallel_config(
    model_tags,
    element_size: int,
    model_memory_thresh: float = 0.75,
    num_gpus: int = None 
) -> Tuple[int, Dict[str, ParallelConfig]]:
    if not len(model_tags):
        return 1, {}
    model_configs = [get_model_config(model_tag) for model_tag in model_tags]
    device_configs = get_device_config()
    device_configs = device_configs[:num_gpus]
    device_mem = device_configs[0].total_memory 
    '''
    We do parallel for the largest model, and replicate the rest models.
    '''
    largest_model = max(model_configs, key = lambda x: x.n_param)
    mem = largest_model.n_param * element_size 

    tp = 1 
    while mem > model_memory_thresh * tp * device_mem:
        tp *= 2 

    world_size = len(device_configs) // tp * tp    
    
    parallel_configs = {config.tag: ParallelConfig(world_size // tp, tp, 1) 
                        if config.tag == largest_model.tag 
                        else ParallelConfig(world_size, 1, 1) 
                         for config in model_configs} 
    return world_size, parallel_configs

def init_engines(
    world_size: int,
    parallel_configs: Dict[str, ParallelConfig], 
    dtype: torch.dtype,
    backend: Backend,
    block_size: int, 
    seed: Optional[int] = None,
    profile: bool = False,
    use_cuda_graph: bool = False,
    cache_type: str = 'linear'
):
    import ray 
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from collections import defaultdict


    from vllm.config import ParallelConfig, ModelConfig, LoadConfig, DeviceConfig
    from vllm.executor.ray_utils import initialize_ray_cluster
    
    print('initializing devices...', world_size)

    assert all(v.tp == 1 and v.pp == 1 for v in parallel_configs.values()),\
        "distributed inference is not supported"
    
    device_config = DeviceConfig('cuda')
    import os
    load_config = LoadConfig(download_dir=f'{os.getcwd()}/.cache')
    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        worker_use_ray= True, 
        data_parallel_size=world_size)
    # ray.init(ignore)
    # initialize_ray_cluster(parallel_config=parallel_config)
    ray.init()
    from .engine_wrapper import EngineWrapper
    engines = [ray.remote(
        num_cpus = 0, num_gpus = 1 
    )(EngineWrapper).remote('engine') for _ in range(world_size)]
    # engines = [ray.remote(
    #         num_cpus=0,
    #         num_gpus=1,
    #         # scheduling_strategy=PlacementGroupSchedulingStrategy(
    #         #                 placement_group=parallel_config.placement_group,
    #         #                 placement_group_capture_child_tasks=True,
    #         #                 placement_group_bundle_index=bundle_id,
    #                         # ),
    #     )(EngineWrapper).remote() for bundle_id, bundle in enumerate(parallel_config.placement_group.bundle_specs)
    # ]
    
    # engine_node_and_gpu_ids = ray.get([engine.get_node_and_gpu_ids.remote() for engine in engines])

    # node_engines = defaultdict(list)
    # node_gpus = defaultdict(list)

    # for i, (node_id, gpu_ids) in enumerate(engine_node_and_gpu_ids):
    #     node_engines[node_id].append(i)
    #     node_gpus[node_id].extend(gpu_ids)
    # for node_id, gpu_ids in node_gpus.items():
    #     node_gpus[node_id] = sorted(gpu_ids)
    
    # print('update envs...', engines, engine_node_and_gpu_ids, node_gpus)
    # ray.get([engine.update_environment_variables.remote(
    #     {'CUDA_VISIBLE_DEVICES': ','.join(map(str, node_gpus[node_id]))}
    # ) for engine, (node_id, gpu_ids) in zip(engines, engine_node_and_gpu_ids)])
    ray.get([engine.update_environment_variables.remote(
        {'CUDA_VISIBLE_DEVICES': ','.join(map(str, range(world_size)))}
    ) for engine in engines])

    ray.get([engine.init_engine.remote() for engine in engines])

    ray.get([engine.execute_method.remote('init_device', 
                                   parallel_config=parallel_config, 
                                   rank=rank, 
                                   world_size=len(engines), 
                                   local_gpu_indices = list(range(world_size)), 
                                   distributed_init_method='env://', 
                                   local_rank = rank,
                                   seed = seed,
                                   profile = profile) 
            for rank, engine in enumerate(engines)])
    
    model_refs = {}
    tokenizer_refs = {}

    print('loading model...')

    for model_tag, para_config in parallel_configs.items():
        assert world_size == (para_config.dp * para_config.tp * para_config.pp)
        model_placement = Placement(para_config.tp, para_config.pp)
        
        for start_idx in range(0, world_size, para_config.tp * para_config.pp):
            model_placement.add_device_group(range(start_idx,
                    start_idx + para_config.tp * para_config.pp))

        model_config = get_model_config(model_tag)

        model_refs[model_tag] = model_ref = ModelRef(
            placement=model_placement,
            status = ObjectStatus.GLOBAL,
            model_tag = model_tag,
            embed_size = model_config.embed_size,
            num_heads = model_config.num_heads,
            vocab_size = model_config.vocab_size,
            n_layer = model_config.n_layer, 
            tp = para_config.tp,
            pp = para_config.pp, 
            index = 0,
            head_size = model_config.embed_size // model_config.num_heads,
            n_param = model_config.n_param,
            backend = backend ,
            element_size=torch.empty((1,), dtype = dtype).element_size(),
            max_seq_len = model_config.max_seq_len,
            intermidiate_size = model_config.intermidiate_size,
            use_cuda_graph = use_cuda_graph 
        )

        model_config_vllm = ModelConfig(
            model_tag,
            model_tag,
            'auto', 
            True, 
            dtype,
            1234 
        )
        
        ray.get([engine.execute_method.remote('load_model',
            model_config_vllm, 
            device_config, 
            load_config, 
            parallel_config,
            model_ref.device_ref, 
            backend,
            block_size,
            use_cuda_graph,
            cache_type) 
            for engine in engines])
        
        tokenizer_refs[model_tag] = tokenizer_ref = TokenizerRef(
            placement = Placement().add_device_groups(range(world_size)),
            model_tag = model_tag, 
            status = ObjectStatus.GLOBAL,
        )

        ray.get([engine.execute_method.remote(
                'init_tokenizer', 
                model_tag, tokenizer_ref.device_ref) 
                for engine in engines])
    return engines, model_refs, tokenizer_refs

def init(
    model_tags: List[str],
    dtype: Union[str, torch.dtype],
    seed: Optional[int] = None,
    backend_tag: str = 'vllm', 
    debug: bool = False,
    num_gpus: int = None,
    profile: bool = False,
    block_size: int = 16,
    window_size: float = 0.1,
    enable_adaws: bool = False,
    sch_tot_budget: float = 10,
    use_cuda_graph: bool = False,
    cache_type: str = 'linear'
):
    '''
    Initilize the serving system and perform compile time optimizations.
    '''
    print(f'Initialize SLOsServe with {model_tags}, num_gpus: {num_gpus}, dtype: {dtype}, debug: {debug}, profile: {profile}')
    from SLOsServe.utils import get_torch_dtype_from_str
    if isinstance(dtype, str):
        torch_dtype = get_torch_dtype_from_str(dtype)
    else: torch_dtype = dtype
    if backend_tag == 'hf':
        backend = Backend.HUGGINGFACE
    elif backend_tag == 'vllm':
        backend = Backend.VLLM
    else: raise NotImplementedError(f"backend {backend_tag} not supported")

    world_size, parallel_configs = get_parallel_config(model_tags,  
        element_size=torch.empty((1,), dtype = torch_dtype).element_size(), 
        num_gpus = num_gpus)
    
    print('para_config: ', parallel_configs)

    assert cache_type == 'linear' or block_size == 1

    engines, model_refs, tokenizer_refs = init_engines(
        world_size = world_size,
        parallel_configs= parallel_configs,
        dtype = torch_dtype,
        backend = backend,
        seed = seed,
        profile = profile,
        block_size = block_size,
        use_cuda_graph=use_cuda_graph,
        cache_type=cache_type)
    
    executor = Executor(engines, 
                        debug = debug, 
                        window_size = window_size,
                        enable_adaws=enable_adaws,
                        sch_tot_budget=sch_tot_budget)
    
    return GlobalContext(
        executor.dependency_graph,
        model_refs, 
        tokenizer_refs,
        executor
    ), executor