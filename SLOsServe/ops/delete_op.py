from typing import List, Any
import torch 

from .operation import Operator, Node, OpCode
from ..object import (
    ObjectImplDict,
    ObjectRef,
    OpAllocator,
    ObjectDeviceRef)

class DeleteOp(Operator):
    @staticmethod
    def create_op(allocator: OpAllocator, 
                  refs: List[ObjectRef]) -> Node:
        return Node(
            OpCode.DELETE,
            (OpCode.DELETE,),
            input_refs = refs,
            output_refs= [],
            args = refs,
            output = None
        )
    
    @staticmethod
    def has_batched_impl() -> bool:
        return True
    
    @staticmethod
    def batch_forward(obj_impls: ObjectImplDict, 
                    batched_args: List[ObjectDeviceRef]):
        for ref in batched_args:
            obj = obj_impls.pop(ref)
            del obj

    @staticmethod
    def estimate_load(load_metas: List[Any]) -> float:
        return 3e-3 * len(load_metas) + 0.25