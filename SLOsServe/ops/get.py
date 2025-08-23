from typing import Tuple, Any, List
from ray.util.queue import Queue
import time

from .operation import Node, Operator, OpAllocator, OpCode 
from ..object import ObjectRef, ObjectImplDict, OperationID, ObjectDeviceRef
from ..comm import Communicator 
from ..device import ClusterStatus, DeviceGroup 

class GetOp(Operator):
    queue: Queue = None

    @staticmethod 
    def create_op(allocator: OpAllocator, obj: ObjectRef)->Node:
        return Node(
            OpCode.GET, 
            (OpCode.GET, ),
            [obj],
            [],
            args = [
                (allocator.request_meta.request_id, obj.device_ref)
            ],
            output = None
        )

    @staticmethod 
    def batch_forward(
        obj_impls: ObjectImplDict, 
        batched_args: List[Tuple[OperationID, ObjectDeviceRef]]):
        GetOp.queue.put_nowait_batch([
            (request_id, obj_impls[ref], time.perf_counter()) for request_id, ref in batched_args
        ])

    @staticmethod 
    def place(
        communicator: Communicator,
        cluster_status: ClusterStatus,
        output: Any,
        obj: ObjectRef
    ) -> DeviceGroup:
        assert len(obj.placement.device_groups)
        assert len(obj.placement) == 1, "distributed object is not supported"
        return obj.placement.device_groups[0]
    
    @staticmethod
    def has_batched_impl() -> bool:
        return True