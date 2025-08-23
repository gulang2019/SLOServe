from dataclasses import dataclass
from typing import List
import torch

from ..device import ClusterStatus, DeviceGroup
from .operation import Operator, Node, OpCode
from ..object import (
    ObjectImplDict,
    ObjectRef,
    ObjectDeviceRef,
    TokenizerRef,
    OpAllocator,
    TensorRef, 
    Placement)
from ..comm import Communicator

# walk around for now
from transformers import GPT2TokenizerFast

_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

class TokenizerEncodeOp(Operator):
    @dataclass 
    class Args:
        # Input
        tokenizer_ref: ObjectDeviceRef
        input_str: str 
        # Output  
        input_ids: ObjectDeviceRef

    @staticmethod
    def create_op(
        allocator: OpAllocator,
        tokenizer: TokenizerRef,
        input_str: str 
    ) -> Node:
        input_ids = allocator.new(
            TensorRef,
            # shape = (len(input_str.split()),),
            shape = (len(_tokenizer.tokenize(input_str)),),
        )
        return Node(
            OpCode.ENCODE, 
            op_tag = (OpCode.ENCODE, tokenizer.model_tag),
            input_refs = [tokenizer],
            output_refs = [input_ids],
            args = [TokenizerEncodeOp.Args(
                tokenizer.device_ref, 
                input_str, 
                input_ids.device_ref
            )],
            output = input_ids
        )

    @staticmethod
    def place(communicator: Communicator, 
              cluster_status: ClusterStatus, 
              output: TensorRef, 
              tokenizer: TokenizerRef,
              input_str: str) -> DeviceGroup:
        device_group = cluster_status.get_min_load_device_group(tokenizer.placement.device_groups)
        output.placement.update_device_group(device_group)
        return device_group

    @staticmethod
    def forward(
        obj_impls: ObjectImplDict, 
        args: 'TokenizerEncodeOp.Args' 
    ):
        TokenizerEncodeOp.batch_forward(obj_impls, [args])

    @staticmethod
    def has_batched_impl() -> bool:
        return True 
    
    @staticmethod 
    def batch_forward(obj_impls: ObjectImplDict, 
                      batched_args: List['TokenizerEncodeOp.Args']):
        tokenizer = obj_impls[batched_args[0].tokenizer_ref]
        strs = [args.input_str for args in batched_args]
        for args, input_ids in zip(batched_args, tokenizer(strs).input_ids):
            obj_impls[args.input_ids] = torch.tensor(input_ids, dtype = torch.int32, device = 'cuda')

class TokenizerDecodeOp(Operator):
    @dataclass 
    class Args:
        # Input
        tokenizer_ref: ObjectDeviceRef
        sampled_ids: ObjectDeviceRef 
        # Output
        decoded_text: ObjectDeviceRef 

    @staticmethod
    def create_op(
        allocator: OpAllocator,
        tokenizer: TokenizerRef,
        sampled_ids: TensorRef 
    ) -> Node:
        decoded_text = allocator.new(ObjectRef)

        return Node(
            OpCode.DECODE, 
            op_tag = (OpCode.DECODE, tokenizer.model_tag),
            input_refs = [tokenizer, sampled_ids],
            output_refs = [decoded_text],
            args = [TokenizerDecodeOp.Args(
                tokenizer.device_ref, 
                sampled_ids.device_ref, 
                decoded_text.device_ref
            )],
            output = decoded_text 
        )

    @staticmethod
    def has_batched_impl() -> bool:
        return True 
    
    @staticmethod
    def batch_forward(obj_impls: ObjectImplDict, 
                      batched_args: List['TokenizerDecodeOp.Args']):
        tokenizer = obj_impls[batched_args[0].tokenizer_ref]

        sampled_ids = [obj_impls[args.sampled_ids].tolist() for args in batched_args]

        decoded_texts = tokenizer.batch_decode(sampled_ids)

        for args, decoded_text, sampled_id in zip(batched_args, decoded_texts, sampled_ids):
            is_end = False
            n_generated = 0
            for id in sampled_id:
                if id == tokenizer.eos_token_id: 
                    is_end = True
                    break 
                n_generated += 1
            obj_impls[args.decoded_text] = (
                is_end,
                n_generated,
                decoded_text
            )
            
    @staticmethod
    def place(communicator: Communicator,
               cluster_status: ClusterStatus, 
               output: ObjectRef, 
               tokenizer: TokenizerRef,
               sampled_ids: TensorRef) -> DeviceGroup:
        assert sampled_ids.placement[0][0] in tokenizer.placement
        device_group = sampled_ids.placement.device_groups[0]
        output.placement.add_device_group(device_group)
        return device_group
    
    @staticmethod
    def estimate_load(load_metas: List) -> float:
        return len(load_metas) * 8e-1
    