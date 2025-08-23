from dataclasses import dataclass
from typing import List
import torch
import numpy as np 
import logging 

import gc

from .models import RequestCacheManager
logger = logging.getLogger(__name__)

@dataclass
class DeviceConfig:
    id: int
    name: str
    total_memory: int
    current_memory_allocated: int
    current_memory_reserved: int

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device_config(num_gpus: int = None) -> List[DeviceConfig]:
    if torch.cuda.is_available():
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        else: num_gpus = min(torch.cuda.device_count(), num_gpus)
        gpu_details: List[DeviceConfig] = []
        for i in range(num_gpus):
            gpu_details.append(DeviceConfig(
                id = i,
                name = torch.cuda.get_device_name(i),
                total_memory = torch.cuda.get_device_properties(i).total_memory,
                current_memory_allocated = torch.cuda.memory_allocated(i),
                current_memory_reserved = torch.cuda.memory_reserved(i)
            ))
        return gpu_details
    else:
        return []
    
def get_num_gpus():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0 


def get_torch_dtype_from_str(dtype: str) -> torch.dtype:
    dtype_map = {
        'float32': torch.float32, 
        'float16': torch.float16,
        'fp32': torch.float32, 
        'fp16': torch.float16
    }
    return dtype_map.get(dtype, torch.float32)

def object_to_type_meta(obj) -> str:
    tensors = []
    def impl(obj):
        ret = ''
        if isinstance(obj, list):
            ret += f'['
            for item in obj:
                ret += impl(item) + ',' 
            ret += ']'
        elif isinstance(obj, dict):
            ret += '{'
            for k, v in obj.items():
                ret += str(k) + ':' + impl(v) + ','
            ret += '}'
        elif isinstance(obj, tuple):
            ret += '('
            for item in obj:
                ret += impl(item) + ',' 
            ret += ')'
        elif isinstance(obj, torch.Tensor):
            ret += '*'
            tensors.append(obj)
        elif isinstance(obj, RequestCacheManager):
            # we encode the RequestCacheManager by $MODEL_NAME:str,num_tokens:int,ListBlocks:List[Tensor]$
            ret += '$'
            ret += obj.global_manager.model_tag + ','
            ret += str(obj.num_tokens) + ','
            ret += impl(obj.get_aggregated_cache()) + ',' # (2, #blocks, block_size, ...)
            ret += '$'
        elif isinstance(obj, int):
            ret += '@'
            ret += str(obj)
            ret += ',@'
        else:  
            raise TypeError
        return ret 
    meta = impl(obj) + '\\\\'
    return meta, tensors

def type_meta_to_object(meta: str, tensors: List[torch.Tensor], kv_cache_creator: callable):
    meta_iter = iter(meta)
    tensor_iter = iter(tensors)
    def parse():
        token = next(meta_iter)
        if token == '*':
            next(meta_iter)
            return next(tensor_iter)
        elif token == '[':
            ret = []
            while True:
                obj = parse()
                if obj == '':
                    break 
                ret.append(obj)
            next(meta_iter)
            # assert next(meta_iter) == ']'
            return ret
        elif token == '(':
            ret = []
            while True:
                obj = parse()
                if obj == '':
                    break 
                ret.append(obj)
            next(meta_iter)
            # assert next(meta_iter) == ')'
            return tuple(ret)
        elif token == '{':
            ret = {}
            while True: 
                key = parse()
                if key == '': break 
                value = parse()
                ret[key] = value 
            # assert next(meta_iter) == '}'
            next(meta_iter)
            return ret   
        elif token == '$':
            model_tag = parse()
            num_tokens = int(parse())
            tensors = parse()
            assert isinstance(tensors, list) and isinstance(tensors[0], torch.Tensor)
            assert next(meta_iter) == '$'
            next(meta_iter)
            return kv_cache_creator(model_tag, num_tokens, tensors)
        elif token == '@':
            v = int(parse())
            assert next(meta_iter) == '@'
            next(meta_iter)
            return v  
        else: 
            ret = ''
            while token not in ',*:[]{}()\\@$':
                ret += token 
                token = next(meta_iter)
            return ret 
    obj = parse()
    assert next(meta_iter) == '\\'
    return obj

def slice_length(s: slice, l: int) -> int:
    if l is None:
        if s.stop is None and s.start is None: 
            return None
        if s.stop is None:
            return - s.start if s.start < 0 else None
        elif s.start is None:
            return s.stop + 1 if s.stop >= 0 else None
        return s.stop - s.start
    
    start, stop, step = s.start, s.stop, s.step

    # Handle None values for start, stop, and step
    if step is None:
        step = 1
    if start is None:
        start = 0 if step > 0 else l - 1
    if stop is None:
        stop = l if step > 0 else -1

    # Adjust start and stop for negative values and out of bounds
    start = max(min(start, l), -l)
    stop = max(min(stop, l), -l)

    # Calculate the actual indices
    start = l + start if start < 0 else start
    stop = l + stop if stop < 0 else stop

    # Calculate length of slice
    if (step > 0 and start >= stop) or (step < 0 and start <= stop):
        return 0
    return max(0, (stop - start + (step - 1 if step > 0 else step + 1)) // step)

def check_same(a: tuple, b: tuple) -> bool:
    if len(a) != len(b): return False 
    for x, y in zip(a,b):
        if x is None or y is None: continue 
        if x != y: return False 
    return True 

def measure_bandwidth(src, dst, num_iters=10):
    import time 
    
    size_in_bytes = int(1e9)

    size_in_gb = size_in_bytes / (1024 * 1024 * 1024)
    
    # Create a tensor on GPU 0
    tensor = torch.randn(size_in_bytes // 4, device=f'cuda:{src}')
    
    # Warm-up transfer
    _ = tensor.to(f'cuda:{dst}')
    
    # Measure bandwidth
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iters):
        tensor_copy = tensor.to(f'cuda:{dst}')
    torch.cuda.synchronize()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    bandwidth = (size_in_gb * num_iters) / elapsed_time
    
    return bandwidth

def profile_memory_bandwidth(
    p2p = False 
) -> float:
    device_count = torch.cuda.device_count()
    test_cases = [(src, dst) for src in range(device_count) if src < dst for dst in range(device_count)] if p2p else [(0,1)]
    bandwidths = [measure_bandwidth(src, dst) for src, dst in test_cases]
    mean_bw, std_bw = np.mean(bandwidths), np.std(bandwidths)
    logger.info(f'measured bandwidth, {mean_bw} +- {std_bw}')
    return mean_bw