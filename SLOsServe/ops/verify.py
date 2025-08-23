from typing import Tuple, Any, List
from dataclasses import dataclass
import torch

from .operation import Operator, Node, OpCode
from ..object import (
    OpAllocator,
    ObjectImplDict,
    TensorRef,
    ObjectDeviceRef,
    ObjectRef,
    ConstantRef
)
from ..comm import Communicator
from ..device import DeviceGroup, ClusterStatus

class VerifyOp(Operator):
    @dataclass 
    class Args:
        guessed_tokens: ObjectDeviceRef
        true_tokens: ObjectDeviceRef 
        rewind_size: ObjectDeviceRef 
        tokens: ObjectDeviceRef

    @dataclass 
    class Output:
        rewind_size: ConstantRef
        tokens: TensorRef

    @staticmethod
    def create_op(allocator: OpAllocator, 
                  guessed_tokens: TensorRef,
                  true_tokens: TensorRef
    )->Node:
        assert len(guessed_tokens.shape) == 1 and len(true_tokens.shape) == 1
        assert guessed_tokens.shape[0] == true_tokens.shape[0]
        output = VerifyOp.Output(
            rewind_size = allocator.new(ConstantRef, type = int),
            tokens = allocator.new(TensorRef, shape = (guessed_tokens.shape[0],))
        )

        return Node(
            OpCode.VERIFY,
            (OpCode.VERIFY,),
            input_refs = [guessed_tokens, true_tokens],
            output_refs = allocator.allocated,
            args = [VerifyOp.Args(
                guessed_tokens=guessed_tokens.device_ref,
                true_tokens=true_tokens.device_ref,
                rewind_size=output.rewind_size.device_ref,
                tokens=output.tokens.device_ref
            )],
            output = output
        )

    @staticmethod
    def has_batched_impl() -> bool:
        return False 

    @staticmethod
    def forward(
        obj_impls: ObjectImplDict, 
        args: 'VerifyOp.Args'):
        guessed_tokens: torch.Tensor = obj_impls[args.guessed_tokens]
        true_tokens: torch.Tensor = obj_impls[args.true_tokens]
        spec_length = guessed_tokens.size(0)
        is_correct = (guessed_tokens == true_tokens).tolist() + [False]
        n_acc = min(range(spec_length+1), key = is_correct.__getitem__)
        n_keeped = min(n_acc + 1, spec_length)
        rewind_size = spec_length - n_keeped
        # print('n_acc/spec_length/rewind', n_acc, '/', spec_length, '/', rewind_size)
        obj_impls[args.rewind_size] = rewind_size
        obj_impls[args.tokens] = true_tokens[:n_keeped]

    @staticmethod 
    def place(
        communicator: Communicator,
        cluster_status: ClusterStatus,
        output: 'VerifyOp.Output',
        guessed_tokens: TensorRef,
        true_tokens: TensorRef
    ) -> DeviceGroup:
        assert len(guessed_tokens.placement) == 1 and len(true_tokens.placement) == 1
        device_groups: List[DeviceGroup] = guessed_tokens.placement.device_groups + true_tokens.placement.device_groups
        device_group: DeviceGroup = cluster_status.get_min_load_device_group(device_groups)
        
        for t in [guessed_tokens, true_tokens]:
            if device_group not in t.placement:
                communicator.comm(t.placement[0][0], device_group.devices[0], t.device_ref, True)

        output.rewind_size.placement.add_device_group(device_group)
        output.tokens.placement.add_device_group(device_group)
        return device_group 
    
    @staticmethod 
    def estimate_load(load_metas: List[Any]) -> float:
        return 6e-2 * len(load_metas)