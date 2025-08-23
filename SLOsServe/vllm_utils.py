import vllm 
import torch 
import gc
from typing import Optional 

def is_vllm_v1() -> bool:
    _, v, _ = vllm.__version__.split('.')
    return int(v) >= 8
    

class CudaMemoryProfiler:

    def __init__(self, device: Optional[torch.types.Device] = None):
        self.device = device

    def current_memory_usage(self) -> float:
        # Return the memory usage in bytes.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            mem = torch.cuda.max_memory_allocated(self.device)
        return mem

    def __enter__(self):
        self.initial_memory = self.current_memory_usage()
        # This allows us to call methods of the context manager if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.final_memory = self.current_memory_usage()
        self.consumed_memory = self.final_memory - self.initial_memory

        # Force garbage collection
        gc.collect()