from typing import Tuple, Any, List
from dataclasses import dataclass
import torch

from .operation import Operator, Node, OpCode
from ..object import (
    OpAllocator,
    ObjectImplDict,
    TensorRef,
    ObjectDeviceRef
)
from ..comm import Communicator
from ..device import DeviceGroup, ClusterStatus


class ConcatOp(Operator):
    @dataclass 
    class Args:
        tensors: List[ObjectDeviceRef]
        dim: int 
        output: ObjectDeviceRef
    
    @staticmethod 
    def create_op(
        allocator: OpAllocator, 
        tensors: List[TensorRef],
        dim: int = 0
        )->Node:
        example_shape = list(tensors[0].shape)
        example_shape[dim] = None
        from ..utils import check_same
        assert all((check_same(example_shape, t.shape) for t in tensors))
        sizes = [t.shape[dim] for t in tensors]
        sum_size = None if None in sizes else sum(sizes)
        example_shape[dim] = sum_size
        
        new_obj = allocator.new(
            TensorRef, 
            shape = tuple(example_shape)
        )

        return Node(
            op_code = OpCode.CONCAT,
            op_tag = (OpCode.CONCAT,),
            input_refs = tensors,
            output_refs = allocator.allocated, 
            args = [ConcatOp.Args(
                tensors = [x.device_ref for x in tensors],
                dim = dim,
                output = new_obj.device_ref 
            )],
            output = new_obj
        )

    @staticmethod
    def has_batched_impl() -> bool:
        return False 

    @staticmethod
    def forward(obj_impls: ObjectImplDict, 
                args: 'ConcatOp.Args'):
        tensors = [obj_impls[ref] for ref in args.tensors]
        obj_impls[args.output] = torch.concat(tensors, dim = args.dim)

    @staticmethod 
    def place(
        communicator: Communicator,
        cluster_status: ClusterStatus,
        output: TensorRef,
        tensors: List[TensorRef],
        dim: int = 0
    ) -> DeviceGroup:
        # We find one device that minimizes the communication 
        assert all(t.placement.pattern == (1,1) for t in tensors)
        device_counts = {}
        for t in tensors:
            for device_group in t.placement.device_groups:
                device_counts[device_group] = device_counts.get(device_group, 0) + 1
        device_group, _ = max(device_counts.items(), key = lambda x: x[1])
        
        output.placement.add_device_group(device_group)
        
        for t in tensors:
            if device_group not in t.placement:
                communicator.comm(t.placement[0][0], device_group.devices[0], t.device_ref, True)

        return device_group
    
    @staticmethod
    def estimate_load(load_metas: List[Any]) -> float:
        return 3e-2 * len(load_metas)