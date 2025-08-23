from typing import List, Tuple, Optional, Dict, Any, Callable, Coroutine
import networkx as nx
import uuid 
import time 
from asyncio import Queue

from SLOsServe.executor import Executor 
from SLOsServe.object import (
    ObjectRef, 
    RequestMeta,
    ObjectStatus,
    TensorRef, 
    ModelRef, 
    TokenizerRef,
    OpAllocator,
    KVCacheRef,
    ConstantRef,
    OperationID
)
from .ops import (
    OP_CLASSES, 
    Node, 
    OpCode,
    CausalLMInferenceOp,
    VerifyOp
)
from .device import DeviceGroup, ClusterStatus
from .comm import Communicator

def placer_wrapper(
    placer: Callable[[Communicator, ClusterStatus, Any], DeviceGroup],
    *args, 
    **kwargs
) -> Callable[[Communicator, ClusterStatus], DeviceGroup]:
    def impl(comm, engine_status, output):
        return placer(comm, engine_status, output, *args, **kwargs)
    return impl 

class GlobalContext:
    def __init__(
        self,
        graph: nx.Graph,
        models: Dict[str, ModelRef],
        tokenizers: Dict[str, TokenizerRef],
        executor: Executor, 
    ):
        self.dependency_graph = graph 
        self.models = models 
        self.tokenizers = tokenizers
        self.executor = executor # for the get method 
        self.reset()

    def get_model(self, model_tag: str) -> ModelRef:
        if model_tag not in self.models:
            raise RuntimeError(f'model of {model_tag} not registered')
        return self.models[model_tag]
    
    def get_tokenizer(self, model_tag: str) -> TokenizerRef:
        if model_tag not in self.tokenizers:
            raise RuntimeError(f'tokenizer of {model_tag} not registered')
        return self.tokenizers[model_tag]

    def add_node(self, op_id: OperationID, op: Node, input_ops: List[OperationID]) -> Any:            
        self.dependency_graph.add_node(op_id, op=op)
        for input_op in input_ops:
            if self.dependency_graph.has_node(input_op):
                self.dependency_graph.add_edge(input_op, op_id)
    
    def context(self, **kwargs)->'RequestContext': 
        context = RequestContext(self, self.request_cnt, **kwargs)
        self.request_contexts[self.request_cnt] = context
        self.request_cnt += 1
        return context
    
    def reset(self):
        self.request_cnt = 0
        self.request_contexts = {}
        self.executor.set_request_context(self.request_contexts)

class RequestContext:
    def __init__(self, global_context: GlobalContext, request_id: int, **kwargs):
        self.cleaned = False
        self.global_context = global_context
        self.request_output_queue = Queue()
        self.request_meta = RequestMeta(
            request_id = request_id,
            arrive_time = time.perf_counter(),
            **kwargs)
        self.op_ids = []
        self.objs: List[ObjectRef] = []

    def get(self, ref: ObjectRef) -> Coroutine:
        self.create_op(OpCode.GET, ref)
        return self.request_output_queue.get()
    
    def get_nowait(self, ref: ObjectRef) -> Any:
        self.create_op(OpCode.GET, ref)

    def get_output_queue(self) -> Queue:
        return self.request_output_queue
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kwargs):
        if len(self.objs):
            # we can pop unscheduled op from the graph here
            self.global_context.dependency_graph.remove_nodes_from(self.op_ids)
            self.create_op(OpCode.DELETE, self.objs)
        self.global_context.request_contexts.pop(
            self.request_meta.request_id)
    
    def create_op(self, op_code: OpCode, *args, **kwargs):
        op_id = uuid.uuid4()
        op_class = OP_CLASSES[op_code]
        op_tag = kwargs.pop('customized_tag', None)
        op: Node = op_class.create_op(
            OpAllocator(op_id, self.request_meta), *args, **kwargs)
        if op_tag is not None: op.op_tag = op_tag
        op.placer = placer_wrapper(op_class.place, *args, **kwargs)
        self.op_ids.append(op_id)
        self.objs.extend(op.output_refs)
    
        input_ops = set((ref.create_by for ref in op.input_refs 
                         if not ref.status == ObjectStatus.GLOBAL))
        
        self.global_context.add_node(op_id, op, input_ops)

        return op.output

    def forward(self,   model_tag: str, 
                        input_ids: Optional[TensorRef] = None,
                        inputs_embeds: Optional[TensorRef] = None,
                        past_key_values: Optional[KVCacheRef] = None,
                        use_cache: bool = True,
                        output_probs: bool = False,
                        output_last_hidden_state: bool = False,
                        do_sample: bool = True,
                        only_sample_last: bool = False,
                        max_decode_len: int = None, 
                        rewind_size: Optional[ConstantRef] = None,
                        n_iter:int = 1,
                        customized_tag: Tuple = None
        ) -> CausalLMInferenceOp.Output:
        return self.create_op(
                OpCode.CausalLMInference, 
                self.global_context.get_model(model_tag),
                input_ids,
                inputs_embeds,
                past_key_values,
                use_cache,
                output_probs,
                output_last_hidden_state,
                do_sample,
                only_sample_last,
                max_decode_len,
                rewind_size,
                n_iter,
                customized_tag = customized_tag)
    
    def tokenize(self, model_tag: str, prompt: str) -> ObjectRef:
        return self.create_op(
            OpCode.ENCODE, 
            self.global_context.get_tokenizer(model_tag),
            prompt
        )
    
    def decode(self, model_tag: str, sampled_ids: TensorRef) -> ObjectRef:
        return self.create_op(
            OpCode.DECODE, 
            self.global_context.get_tokenizer(model_tag), 
            sampled_ids
        )

    def concat(self, 
               inputs: List[TensorRef], 
               dim: int = 0
    ) -> TensorRef:
        return self.create_op(
            OpCode.CONCAT, 
            inputs, dim 
        )
    
    def verify(
        self,
        guessed_tokens: TensorRef, 
        true_tokens: TensorRef
    ) -> VerifyOp.Output:
        return self.create_op(
            OpCode.VERIFY, 
            guessed_tokens, 
            true_tokens
        )