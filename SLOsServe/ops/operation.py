from abc import ABC, abstractmethod 
from typing import List, Hashable, Any, Callable
from dataclasses import dataclass, field
import enum
import logging
import time 

from SLOsServe.object import (
    ObjectRef,
    OpAllocator,
    ObjectImplDict,
    RequestMeta,
    ObjectStatus
)
from ..comm import Communicator
from ..device import ClusterStatus, DeviceGroup

logger = logging.getLogger(__name__)

class OpCode(enum.Enum):
    # The order here is the order from least prioritized op 
    # to the highest prioritized op. 
    CausalLMInference = enum.auto()
    CONCAT = enum.auto()
    VERIFY = enum.auto()
    DECODE = enum.auto()
    PLACEHOLDER = enum.auto()
    BATCHABLE = enum.auto()
    ENCODE = enum.auto() 
    DELETE = enum.auto() 
    GET = enum.auto()
    
    def __gt__(self, other):
        if isinstance(other, OpCode):
            return self.value > other.value
        return NotImplemented

'''
The class used for recording the remote processing in the backend and their dependency.  
'''
@dataclass 
class Node(ABC):
    op_code: OpCode 
    op_tag: Hashable
    input_refs: List[ObjectRef] # The dependent inputs 
    output_refs: List[ObjectRef] # The output objects 
    args: List[Any]
    output: Any
    placer: Callable[[Communicator, ClusterStatus, Any], DeviceGroup] = field(init = False, default = None)
    device_group: DeviceGroup = field(init = False, default = None)
    request_meta: RequestMeta = field(init = False, default = None)
    load_meta: Any = None
    placed: bool = field(init = False, default = False)
    create_time: float = field(init = False, default = None)

    def place(self, comm: Communicator, status: ClusterStatus):
        assert not self.placed
        self.device_group = self.placer(comm, status, self.output)
        self.placed = True 
    def __post_init__(self):
        if len(self.output_refs):
            self.request_meta = self.output_refs[0].request_meta
        else: 
            for ref in self.input_refs:
                if ref.status != ObjectStatus.GLOBAL:
                    self.request_meta = ref.request_meta 
                    break 
        self.create_time = time.perf_counter()

'''
    The abstract class that implements the interpretor and engine's forward function.
'''
class Operator:
    @staticmethod 
    @abstractmethod
    def create_op(allocator: OpAllocator, *args, **kwargs)->Node:
        raise NotImplementedError
    
    @staticmethod
    def has_batched_impl() -> bool:
        return False 

    @staticmethod  
    @abstractmethod
    def forward(
        obj_impls: ObjectImplDict, 
        args: Any):
        raise NotImplementedError

    @staticmethod 
    def batch_forward(
        obj_impls: ObjectImplDict, 
        batched_args: List[Any]):
        raise NotImplementedError

    @staticmethod 
    def place(
        communicator: Communicator,
        cluster_status: ClusterStatus,
        output: Any,
        *args, **kwargs 
    ) -> DeviceGroup:
        return DeviceGroup()
    
    @staticmethod
    def get_profile_log(
        op_code: OpCode,
        obj_impls: ObjectImplDict, 
        batched_args: List[Any]
    ):
        return str(op_code)

    @staticmethod 
    def estimate_load(
        load_metas: List[Any]  
    ) -> float: # the estimated time in milliseconds
        return 3e-1
