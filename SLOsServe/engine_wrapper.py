import ray 
import logging 
from typing import Tuple, List, Dict 
import os
import importlib

logger = logging.getLogger(__name__)

MODULE_MAP: Dict[str, Tuple[str, str]] = {
    'engine': ('SLOsServe.engine', 'Engine'),
    'spec_engine': ('SLOsServe.spec_decode_engine', 'SpecDecodeEngine')
}

class EngineWrapper:
    # max_n_token: int | None = None 
    # block_size: int | None = None
    
    def __init__(self, engine_type: str):
        self.engine_type = engine_type
        self.job_id = -1
        self.print_period = 1000
        self.cnt = 0

    def init_engine(self, *args, **kwargs):
        module_path, class_name = MODULE_MAP[self.engine_type]
        cls = getattr(importlib.import_module(module_path), class_name)
        self.worker = cls(*args, **kwargs)
    
    def execute_method(self, method, *args, **kwargs):
        if method == 'execute':
            # if self.cnt % self.print_period == 0:
            #     self.worker.print_status()
            self.cnt += 1
        assert self.worker is not None
        executable = getattr(self.worker, method)
        res = executable(*args, **kwargs) 
        return res
    
    def execute_method_with_id(self, id,  method, *args, **kwargs):
        assert id == self.job_id + 1, "Execution order != Submission order"
        self.job_id += 1 
        return self.execute_method(method, *args, **kwargs)
    
    def get_node_and_gpu_ids(self) -> Tuple[str, List[int]]:
        node_id = ray.get_runtime_context().get_node_id()
        gpu_ids = ray.get_gpu_ids()
        return node_id, gpu_ids

    def update_environment_variables(self, envs: Dict[str, str]) -> None:
        logger.info('UPD envs', envs)
        key = 'CUDA_VISIBLE_DEVICES'
        if key in envs and key in os.environ:
            # overwriting CUDA_VISIBLE_DEVICES is desired behavior
            # suppress the warning in `update_environment_variables`
            del os.environ[key]
        for k, v in envs.items():
            if k in os.environ and os.environ[k] != v:
                logger.warning(
                    "Overwriting environment variable %s "
                    "from '%s' to '%s'", k, os.environ[k], v)
            os.environ[k] = v