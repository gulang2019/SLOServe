from typing import List, Tuple, Optional
from dataclasses import dataclass, field 
import torch 
import logging 
import random
from itertools import accumulate

from ..backend import Backend
from ..comm import Communicator
from ..device import ClusterStatus
from ..object import (
    ModelRef,
    TensorRef,
    KVCacheRef,
    OpAllocator,
    ObjectImplDict,
    ObjectDeviceRef,
    Placement,
    ConstantRef
)
from ..models import ModelImpl
from .operation import Node, Operator, OpCode
from .utils import OPTIONAL_REF
import numpy as np
from vllm.attention.backends.flash_attn import FlashAttentionMetadata

logger = logging.getLogger(__name__)


'''
To support the vllm's paged attention, we need to 
1. initialize the per device KVCache manager when loading the model; Done
2. initialize the per request KVCache when doing inference; 
the KVCache manager should come along with the model. 
3. delete the KVCache when the request end;
3. communicate the KVCache when migration;
'''
class CausalLMInferenceOp(Operator):
    @dataclass
    class Args:
        # Input
        model: ObjectDeviceRef
        input_ids: Optional[ObjectDeviceRef] # (seq_len,)
        inputs_embeds: Optional[ObjectDeviceRef] # (seq_len, hs)
        past_key_values: Optional[ObjectDeviceRef] # (#layer, #head, past_len, head_size)
        use_cache: bool
        output_probs: bool
        output_last_hidden_state: bool
        do_sample: bool
        only_sample_last: bool
        max_decode_len: int 
        rewind_size: Optional[ObjectDeviceRef]
        n_iter: int
        temperature: float
        # Output
        last_hidden_state: Optional[ObjectDeviceRef] 
        output_past_key_values: Optional[ObjectDeviceRef]   
        sampled_ids: Optional[ObjectDeviceRef]
        probs: Optional[ObjectDeviceRef]

    @dataclass 
    class Output:
        last_hidden_state: Optional[TensorRef] # (seq_len, hs)
        past_key_values: Optional[KVCacheRef] = field(init = False, default = None) # (#head, past_len, head_size)
        sampled_ids: Optional[TensorRef] # (1 if only_last_logits else seq_len, )
        probs: Optional[TensorRef] # the log prob of 

    @staticmethod 
    def create_op(
        allocator: OpAllocator,
        model_ref: ModelRef, 
        input_ids: Optional[TensorRef] = None,
        inputs_embeds: Optional[TensorRef] = None,
        past_key_values: Optional[KVCacheRef] = None,
        use_cache: bool = False,
        output_probs: bool = False,
        output_last_hidden_state: bool  = False,
        do_sample: bool = False,
        only_sample_last: bool = False,
        max_decode_len: int = None, 
        rewind_size: Optional[ConstantRef] = None,
        n_iter: int = 1,
        temperature: float = 0.0
    ):
        assert isinstance(model_ref, ModelRef)
        past_seq_len = 0 
        cur_seq_len = None 
        if input_ids is not None:
            assert isinstance(input_ids, TensorRef)
            cur_seq_len, = input_ids.shape
        if inputs_embeds is not None:
            assert isinstance(inputs_embeds, TensorRef)
            cur_seq_len, embed_size = inputs_embeds.shape
            assert embed_size == model_ref.embed_size 
        if past_key_values is not None:             
            assert isinstance(past_key_values, KVCacheRef)
        assert n_iter == 1 or cur_seq_len == 1
        cur_seq_len *= n_iter 
        tot_seq_len = None if (None in [cur_seq_len, past_seq_len]) else cur_seq_len + past_seq_len

        output = CausalLMInferenceOp.Output(
            last_hidden_state = allocator.new(TensorRef,
                                             shape = (cur_seq_len, model_ref.embed_size))
                                                  if output_last_hidden_state else None,
            probs = allocator.new(TensorRef,
                                   shape = (cur_seq_len, model_ref.vocab_size)) if output_probs else None,
            sampled_ids = allocator.new(TensorRef,
                                        shape = (1 if only_sample_last else cur_seq_len,)) 
                                        if do_sample else None,
        )

        max_seq_len = min(max_decode_len + cur_seq_len, model_ref.max_seq_len)\
            if (max_decode_len is not None and cur_seq_len is not None) else model_ref.max_seq_len

        if use_cache:
            if model_ref.backend == Backend.VLLM:
                # if past_key_values is None:
                output.past_key_values = allocator.new(KVCacheRef,  
                                        placement = Placement(model_ref.pp, model_ref.tp),
                                        length = tot_seq_len,
                                        reserved = max_seq_len,
                                        backend = model_ref.backend,
                                        n_layer = model_ref.n_layer,
                                        head_size = model_ref.head_size,
                                        num_heads = model_ref.num_heads, 
                                        element_size = model_ref.element_size,
                )
                # else: 
                #     output.past_key_values = past_key_values
                #     output.past_key_values.length = tot_seq_len

            elif model_ref.backend == Backend.HUGGINGFACE:
                output.past_key_values = allocator.new(KVCacheRef, 
                                            placement = Placement(model_ref.pp, model_ref.tp),
                                            length = tot_seq_len,
                                            max_seq_len = max_decode_len,  
                                            backend = model_ref.backend)
            else: raise NotImplementedError

        assert model_ref.pp == 1, "pipeline parallelism is not supported"
        
        args = [CausalLMInferenceOp.Args(
            model_ref.device_ref, 
            OPTIONAL_REF(input_ids),
            OPTIONAL_REF(inputs_embeds), 
            OPTIONAL_REF(past_key_values),
            use_cache, 
            output_probs,
            output_last_hidden_state, 
            do_sample, 
            only_sample_last,
            max_decode_len,
            OPTIONAL_REF(rewind_size),
            n_iter,
            temperature,
            OPTIONAL_REF(output.last_hidden_state),
            OPTIONAL_REF(output.past_key_values),
            OPTIONAL_REF(output.sampled_ids),
            OPTIONAL_REF(output.probs)
        ) for _ in range(model_ref.tp)]

        assert (input_ids is None) ^ (inputs_embeds is None)

        past_seq_len = past_key_values.length if past_key_values is not None else 0
        cur_seq_len = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        return Node(
            op_code = OpCode.CausalLMInference, 
            op_tag = (OpCode.CausalLMInference, 
                      model_ref.model_tag, 
                      model_ref.n_param,
                      False, # past_key_values is None,
                      input_ids is None,
                      inputs_embeds is None,
                      n_iter),
            input_refs = [x for x in [model_ref, input_ids, inputs_embeds, past_key_values] if x is not None],
            output_refs = allocator.allocated,
            args = args,
            output = output,
            load_meta=(model_ref, past_seq_len, cur_seq_len, n_iter)
        )
    
    @staticmethod 
    def estimate_load(load_metas: List[Tuple[ModelRef, int, int, int]]):
        cur_seq_len = sum([x[2] for x in load_metas])
        past_seq_len = np.mean([x[1] for x in load_metas])
        n_iter = np.mean([x[3] for x in load_metas])
        estimated_load = load_metas[0][0].estimate_time(
            bs = len(load_metas), 
            past_seq_len = past_seq_len, cur_seq_len = cur_seq_len) * n_iter
        return estimated_load

    @staticmethod 
    def has_batched_impl() -> bool:
        return True
    
    @staticmethod 
    def place(
            communicator: Communicator, 
            cluster_status: ClusterStatus, 
            output: 'CausalLMInferenceOp.Output',
            model_ref: ModelRef, 
            input_ids: Optional[TensorRef] = None,
            inputs_embeds: Optional[TensorRef] = None,
            past_key_values: Optional[KVCacheRef] = None,
            use_cache: bool = False,
            output_probs: bool = False,
            output_last_hidden_state: bool  = False,
            do_sample: bool = False,
            only_sample_last: bool = False,
            max_decode_len: int = None, 
            rewind_size: Optional[ConstantRef] = None,
            n_iter: int = 1,
            temperature: float = 0.0):
        '''
        1. decide the device group 
        We use a heuristic here for scheduling. 
        We will do the computation in place 
        as long as the current device 
        is not overloaded. 
        compute time: n_layer * (4 * cur_seq_len * hidden_size^2 + hidden_size * cur_seq_len * past_seq_len + 2 * cur_seq_len * hidden_size * intermidiate_size) / #PARA
        memory: n_layer * past_seq_len * hidden_size * 2 / #PARA
        '''
        # if past_key_values is None:
        #     candidates = model_ref.placement.device_groups[:1]
        # else: 
        #     candidates = model_ref.placement.device_groups[-1:]
        candidates = model_ref.placement.device_groups
        device_group = None
        if input_ids is not None:
            for i, candidate_device_group in enumerate(candidates):
                if any((x in input_ids.placement) for x in model_ref.placement[i, 0]):
                    device_group = candidate_device_group
        elif inputs_embeds is not None:
            for i, candidate_device_group in enumerate(candidates):
                if any((x in inputs_embeds.placement) for x in model_ref.placement[i, 0]):
                    device_group = candidate_device_group
        if device_group is None:
            device_group = random.choice(candidates)
        
        cur_load = cluster_status.get_max_load(device_group)
        min_load, min_load_device_group = min((((cluster_status.get_max_load(dg), dg) for dg in candidates)), key = lambda x: x[0])

        if cur_load > min_load + cluster_status.max_load_gap:
            device_group = min_load_device_group
        
        # extra_load = model_ref.n_layer * (past_seq_len + cur_seq_len) * model_ref.embed_size * 2 / len(device_group)
        # cluster_status.add_load(model_ref.estimate_time(), device_group)

        placement = Placement(*model_ref.placement.pattern).add_device_group(device_group)
        
        # 2. Next, we do the communication. 
        if input_ids is not None:
            unavail_devices = [x for x in placement[0, 0, :] if x not in input_ids.placement]
            communicator.broadcast(input_ids.placement[0][0], unavail_devices, input_ids, keep_old = True)
            input_ids.placement.add_device_groups(unavail_devices)
        
        if inputs_embeds is not None:
            unavail_devices = [x for x in placement[0, 0, :] if x not in inputs_embeds.placement]
            communicator.broadcast(inputs_embeds.placement[0][0], unavail_devices, inputs_embeds, keep_old = True)
            inputs_embeds.placement.add_device_groups(unavail_devices)

        if past_key_values is not None:
            assert past_key_values.placement.pattern == placement.pattern
            for src, dst in zip(past_key_values.placement[0], placement[0]):
                communicator.comm(src, dst, past_key_values)
            past_key_values.placement.update_device_group(placement[0])

        # 3. Finally, we update the placement of the output.
        if output.sampled_ids is not None: 
            output.sampled_ids.placement.add_device_groups(placement[0, -1, :])
        if output.probs is not None: 
            output.probs.placement.add_device_groups(placement[0, -1, :])
        if output.past_key_values is not None:
            output.past_key_values.placement.update_device_group(placement[0])
        if output.last_hidden_state is not None:
            output.last_hidden_state.placement.add_device_groups(placement[0, -1, :])

        return device_group

    
    @staticmethod 
    @torch.inference_mode()
    def batch_forward_vllm(
        obj_impls: ObjectImplDict,
        batched_args: List['CausalLMInferenceOp.Args']
    ):
        '''
        hidden_states = self.model(
            input_ids: (seq_len,),
            position_ids: (seq_len,),
            kv_caches: [[() * bs] * layer],
            SeqAttnMetadata(seq_lens, past_seq_lens)
        )
        class Args:
            # Input
            model: ObjectDeviceRef
            input_ids: Optional[ObjectDeviceRef] # (seq_len,)
            inputs_embeds: Optional[ObjectDeviceRef] # (seq_len, hs)
            past_key_values: Optional[ObjectDeviceRef] # (#layer, #head, past_len, head_size)
            use_cache: bool
            output_last_hidden_state: bool
            do_sample: bool
            only_sample_last: bool
            # Output
            last_hidden_state: Optional[ObjectDeviceRef] 
            past_key_values: Optional[ObjectDeviceRef]   
            sampled_ids: Optional[ObjectDeviceRef]
        '''
        assert all(((args.input_ids is not None) and 
                    (args.inputs_embeds is None) and 
                    (not args.output_last_hidden_state) and 
                    (args.output_past_key_values) and
                    args.use_cache) for args in batched_args), NotImplementedError
        assert all(args.n_iter == batched_args[0].n_iter for args in batched_args), NotImplementedError
        model_impl: ModelImpl = obj_impls[batched_args[0].model]
        batched_args = sorted(batched_args, key = lambda args: obj_impls[args.input_ids].size(0), reverse = True)
        for args in batched_args:
            if args.past_key_values is None: 
                obj_impls[args.output_past_key_values] = model_impl.cache_manager.new()
                # args.past_key_values = args.output_past_key_values
            else: 
                obj_impls[args.output_past_key_values] = obj_impls[args.past_key_values].new()
        input_idss: List[torch.Tensor] = [obj_impls[args.input_ids] for args in batched_args]
        assert all(x.ndim == 1 for x in input_idss)
        cur_seq_lens: List[int] = [input_ids.size(0) for input_ids in input_idss]
        input_ids = torch.concat(input_idss, dim = -1)
        req_cache_managers = [obj_impls[args.output_past_key_values] for args in batched_args]
        rewind_sizes = [obj_impls[args.rewind_size] if args.rewind_size is not None else 0 for args in batched_args]
        temperatures = torch.tensor(sum(([args.temperature] * cur_seq_len\
            for args, cur_seq_len in zip(batched_args, cur_seq_lens)), start = []))\
            .clamp_(min=1e-3,max=1).unsqueeze(-1).to(input_ids.device)
        # rewind_sizes = [0] * len(batched_args)
        sampled_ids_lists = [[] for _ in range(len(batched_args))]
        probs_lists = [[] for _ in range(len(batched_args))]
        
        assert torch.any(input_ids < 50272) and torch.any(input_ids >= 0)
        
        for _ in range(batched_args[0].n_iter):
            # [#token, hidden_size]
            hidden_states = model_impl.forward(input_ids, 
                                               req_cache_managers,
                                               cur_seq_lens,
                                               rewind_sizes)
            sampling_indices = sum([
                [acc_seq_len - 1] if args.only_sample_last else list(range(acc_seq_len - seq_len, acc_seq_len))  
                for args, seq_len, acc_seq_len in zip(
                    batched_args, cur_seq_lens, accumulate(cur_seq_lens)
                ) if (args.do_sample or args.output_probs)
            ], start = [])

            from vllm.model_executor.sampling_metadata import SamplingMetadata
            sampling_meta = SamplingMetadata([], sampling_indices, {}, -1)
            logits = model_impl.module.compute_logits(hidden_states, sampling_meta) # [tot_seq_len, vocab_size]
            probs = (logits / temperatures).softmax(dim = -1)
            sampled_ids = torch.multinomial(probs, num_samples = 1).unsqueeze(-1)
            
            n_tokens = [(1 if args.only_sample_last else seq_len) if (args.do_sample or args.output_probs) else 0 for args, seq_len in zip(batched_args, cur_seq_lens)]
            for n_tok, n_tok_acc, args, sampled_ids_list, probs_list in zip(n_tokens, accumulate(n_tokens), batched_args, sampled_ids_lists, probs_lists):
                if args.output_probs:
                    probs_list.append(probs[n_tok_acc - n_tok: n_tok_acc])
                if args.do_sample:
                    sampled_ids_list.append(sampled_ids[n_tok_acc - n_tok: n_tok_acc])

            input_ids = sampled_ids
            cur_seq_lens = [1] * len(batched_args)
            rewind_sizes = [0] * len(batched_args)

        for args, sampled_ids_list, probs_list in zip(batched_args, sampled_ids_lists, probs_lists):
            if args.output_probs:
                assert len(probs_list)
                obj_impls[args.probs] = torch.concat(probs_list) if len(probs_list) else probs_list[0]
            if args.do_sample:
                assert len(sampled_ids_list)
                obj_impls[args.sampled_ids] = torch.concat(sampled_ids_list) if len(sampled_ids_list) else sampled_ids_list[0]
    
    @staticmethod
    def batch_forward(
        obj_impls: ObjectImplDict, 
        batched_args: List['CausalLMInferenceOp.Args']   
    ): 
        example_args = batched_args[0]
        from ..models import ModelImpl 
        model_impl: ModelImpl = obj_impls[example_args.model]
        if model_impl.backend == Backend.VLLM: 
            CausalLMInferenceOp.batch_forward_vllm(
                obj_impls, batched_args)
        elif model_impl.backend == Backend.HUGGINGFACE:
            CausalLMInferenceOp.batch_foward_hf(
                obj_impls, batched_args
            )
        else:
            raise NotImplementedError
    
    @staticmethod 
    def get_profile_log(
        op_code: OpCode, 
        obj_impls: ObjectImplDict, 
        batched_args: List['CausalLMInferenceOp.Args']):
        model_impl: ModelImpl = obj_impls[batched_args[0].model]
        # phase = "prefill" if batched_args[0].past_key_values is None else "decode"
        return model_impl.name # + '-' + phase 
'''
    @staticmethod 
    def _prepare_batch_input(
        obj_impls: ObjectImplDict, 
        inputs: List['CausalLMInferenceOp.Input']
    ):
        use_cache = False 
        do_sample = False
        output_last_hidden_state = False  
        input_ids = None  
        inputs_embeds = None
        past_key_values = None 
        attention_mask = None

        example_input = inputs[0]
        bs = len(inputs)
        for input in inputs:
            use_cache = use_cache or input.use_cache
            do_sample = do_sample or input.do_sample 
            output_last_hidden_state = output_last_hidden_state or input.output_last_hidden_state
        
        if example_input.input_ids is not None: 
            for input in inputs: 
                assert (input.input_ids is not None) and (input.inputs_embeds is None)
            input_idss = [obj_impls[input.input_ids] for input in inputs]
            for input_id in input_idss: 
                assert input_id.ndim == 2 and input_id.size(0) == 1
            input_idss = [input_ids.squeeze(0) for input_ids in input_idss]
            input_ids = torch.nested.nested_tensor(input_idss).to_padded_tensor(padding=0.0)
            cur_lengths = [input_ids.size(0) for input_ids in input_idss]
            cur_max_length = max(cur_lengths)
        else: 
            for input in inputs: assert (input.input_ids is None) and (input.inputs_embeds is not None)
            input_embedss = [obj_impls[input.inputs_embeds] for input in inputs]
            for input_embeds in input_embedss: assert len(input_embeds.size()) == 3 and input_embeds.size(0) == 1
            input_embedss = [input_embeds.squeeze(0) for input_embeds in input_embedss]
            cur_lengths = [inputs_embeds.size(0) for inputs_embeds in input_embedss]
            cur_max_length = max(cur_lengths)
            hs = input_embedss[0].size(-1)
            inputs_embeds = torch.nested.nested_tensor(input_embedss)\
                .to_padded_tensor(padding=0.0, output_size = (bs, cur_max_length, hs))
        
        has_past_key_values = False 
        for input in inputs:
            if input.past_key_values is not None:
                has_past_key_values = True
                past_key_values = obj_impls[input.past_key_values[0][0]]
                _, num_head, __, n_head_embd = past_key_values.size()
                dtype = past_key_values.dtype
                assert _ == 1 
        prev_lengths = [obj_impls[input.past_key_values[0][0]].size(2) if input.past_key_values is not None else 0 
                            for input in inputs]
        if has_past_key_values:
            max_prev_length = max(prev_lengths)
            past_key_values = []
            for l in range(example_input.num_layers):
                kv_caches = []
                for i in range(2):
                    caches = [obj_impls[input.past_key_values[l][i]].squeeze(0) if input.past_key_values is not None 
                                else torch.empty((num_head, max_prev_length, n_head_embd), dtype = dtype, device = 'cuda') for input in inputs]
                    # for cache in caches:
                    #     print(f'{i} CACHE: {cache.size()}')
                    caches = torch.nested.nested_tensor(caches).to_padded_tensor(
                        padding = 0.0, output_size = (bs, num_head, max_prev_length, n_head_embd))
                    kv_caches.append(caches)
                past_key_values.append(tuple(kv_caches))
            attention_mask = (torch.tile(torch.arange(0, max_prev_length), (bs, 1))\
                  < torch.tensor(prev_lengths).view(-1, 1)).cuda()

        cur_attention_mask = (torch.tile(torch.arange(0, cur_max_length), (bs, 1))\
                < torch.tensor(cur_lengths).view(-1, 1)).cuda()
        attention_mask = torch.concat([attention_mask, cur_attention_mask], dim = -1)\
            if attention_mask is not None else cur_attention_mask 
        
        model = obj_impls[example_input.model]

        position_ids = None
        from transformers import GPT2LMHeadModel
        if isinstance(model, GPT2LMHeadModel):
            position_ids = (torch.tile(torch.arange(0, cur_max_length), (bs, 1)) + torch.tensor(prev_lengths).view(-1,1)).cuda() if need_position_ids else None

        return {
            'model': model,
            'input_ids': input_ids,
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'use_cache': use_cache,
            'output_last_hidden_state': output_last_hidden_state,
            'position_ids': position_ids,
            'do_sample': do_sample,
        }, cur_attention_mask, attention_mask
    
    @staticmethod
    def batch_forward(
        obj_impls: ObjectImplDict, 
        inputs: List['CausalLMInferenceOp.Input'],
        outputs: List['CausalLMInferenceOp.Output']):

        assert len(inputs) > 0 and len(inputs) == len(outputs)
        

        batched_inputs, cur_attention_mask, attention_mask = \
            CausalLMInferenceOp._prepare_batch_input(obj_impls, inputs)

        _output = CausalLMInferenceOp._forward(**batched_inputs)
    
        for i, (input, output) in enumerate(zip(inputs, outputs)):
            if input.output_last_hidden_state:
                assert _output.last_hidden_state is not None and\
                      output.last_hidden_state is not None
                # (bs, seq_len, n_embed)
                obj_impls[output.last_hidden_state] =\
                      _output.last_hidden_state[i:i+1, cur_attention_mask[i], :]
            if input.use_cache:
                assert output.past_key_values is not None 
                assert _output.past_key_values is not None
                for l in range(len(output.past_key_values)): 
                    for j in range(2):
                        # (bs, #head, past_seq_len + cur_seq_len, n_embd/head)
                        # print('_output.past_key_values[l][j]', _output.past_key_values[l][j].size())
                        past_key_value = _output.past_key_values[l][j]
                        if not attention_mask[i].size(0) == past_key_value.size(2):
                            print('input_ids:', batched_inputs['input_ids'].size() if batched_inputs['input_ids'] is not None else None)
                            print('iniputs_embeds:', batched_inputs['inputs_embeds'].size() if batched_inputs['inputs_embeds'] is not None else None)
                            print('cur_attention_mask: ', attention_mask.size())
                            print('attention_mask', attention_mask.size())
                            print('past_key_values:', batched_inputs['past_key_values'][0][0].size() if batched_inputs['past_key_values'] is not None else None)
                            print('_output.last_hidden_state', _output.last_hidden_state.size() if _output.last_hidden_state is not None else None)
                            print('past_key_values:', past_key_value.size())
                            print('do_sample:', _output.sampled_ids.size() if _output.sampled_ids is not None else None)
                            raise RuntimeError('mismatched shape')

                        obj_impls[output.past_key_values[l][j]] =\
                            past_key_value[i:i+1, :, attention_mask[i]]
            if input.do_sample:
                assert _output.sampled_ids is not None and output.sampled_ids is not None 
                sampled_ids = _output.sampled_ids[i:i+1, cur_attention_mask[i]]  # (seq_len, )
                if input.only_sample_last: 
                    sampled_ids = sampled_ids[:, -1:]
                obj_impls[output.sampled_ids] = sampled_ids

    @staticmethod
    def forward(
        obj_impls: ObjectImplDict, 
        input: 'CausalLMInferenceOp.Input',
        output: 'CausalLMInferenceOp.Output' 
    ):
        model = obj_impls[input.model]
        input_ids = obj_impls[input.input_ids] if input.input_ids is not None else None
        inputs_embeds = obj_impls[input.inputs_embeds] if input.inputs_embeds is not None else None     
        past_key_values = None 
        if input.past_key_values is not None: 
            past_key_values = [
                (obj_impls[layer_kv[0]], obj_impls[layer_kv[1]])
                  for layer_kv in input.past_key_values
            ]

        outputs = CausalLMInferenceOp._forward(
            model, input_ids, inputs_embeds, 
            past_key_values= past_key_values, 
            use_cache = input.use_cache, 
            output_last_hidden_state=input.output_last_hidden_state,
            do_sample=input.do_sample 
        )
        
        if input.output_last_hidden_state:
            assert output.last_hidden_state is not None 
            obj_impls[output.last_hidden_state] = outputs.last_hidden_state
        if input.use_cache:
            assert output.past_key_values is not None 
            assert outputs.past_key_values is not None 
            for l in range(len(output.past_key_values)): 
                for j in range(2):
                    obj_impls[output.past_key_values[l][j]] =\
                        outputs.past_key_values[l][j]
        if input.do_sample: # (1, seq_len, vocab_size)
            assert output.sampled_ids is not None and outputs.sampled_ids is not None 
            obj_impls[output.sampled_ids] = outputs.sampled_ids[:, -1:, :] if input.only_sample_last\
                  else outputs.sampled_ids
            
    @staticmethod
    def _forward(
        model: PreTrainedModel, 
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_last_hidden_state: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        do_sample: bool = False,
    ) -> 'CausalLMInferenceOp._Output':
        assert (input_ids is not None) ^ (inputs_embeds is not None)
        cur_len = input_ids.size(-1) if input_ids is not None else inputs_embeds.size(1)
        past_len = past_key_values[0][0].size(2) if past_key_values is not None else 0
        if input_ids is not None:
            assert input_ids.ndim == 2 # (bs, seq_len) 
        if inputs_embeds is not None: 
            assert inputs_embeds.ndim == 3 
        if past_key_values is not None: 
            assert past_key_values[0][0].ndim == 4 
        if attention_mask is not None:
            assert attention_mask.ndim == 2 
            assert attention_mask.size(1) == cur_len + past_len 
        from transformers import GPT2LMHeadModel, OPTForCausalLM
        if isinstance(model, GPT2LMHeadModel):
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                past_key_values = past_key_values, 
                inputs_embeds = inputs_embeds, 
                use_cache = use_cache,
                output_hidden_states = output_last_hidden_state,
                position_ids = position_ids
            )
        elif isinstance(model, OPTForCausalLM):
            assert position_ids is None 
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                past_key_values = past_key_values, 
                inputs_embeds = inputs_embeds, 
                use_cache = use_cache,
                output_hidden_states = output_last_hidden_state,
            )
        if outputs.hidden_states is not None:
            assert outputs.hidden_states[-1].ndim == 3
        if outputs.past_key_values is not None: 
            assert outputs.past_key_values[0][0].ndim == 4 
            assert outputs.past_key_values[0][0].size(2) == cur_len + past_len 
        if outputs.logits is not None: 
            assert outputs.logits.ndim == 3 
            assert outputs.logits.size(1) == cur_len

        return CausalLMInferenceOp._Output(
            last_hidden_state = outputs.hidden_states[-1] if output_last_hidden_state else None,
            past_key_values=outputs.past_key_values,
            sampled_ids = torch.argmax(outputs.logits, dim = -1) if do_sample else None 
        )
'''