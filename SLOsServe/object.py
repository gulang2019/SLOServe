from typing import List, Any, Optional, Generic, TypeVar, Tuple, Union
from dataclasses import dataclass, field

from uuid import UUID, uuid1
from enum import Enum

from .backend import Backend
from .device import Placement

import logging 

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Define a type variable

ObjectUUID = UUID
OperationID = UUID

'''
An object that is replicated on multiple devices. 
'''
@dataclass
class ObjectDeviceRef(Generic[T]):
    uuid: ObjectUUID = field(default_factory=uuid1) # A unique identifier relates to a backend object.
    access_key: Any = None # Chain of arguments applied to the object.

    def __eq__(self, other):
        if isinstance(other, ObjectDeviceRef):
            return self.uuid == other.uuid 
        return False 
    
    def __hash__(self):
        return hash(self.uuid)
    
    def __copy__(self):
        raise TypeError("Copying not allowed")

    def __deepcopy__(self, memo):
        raise TypeError("Copying not allowed")

@dataclass 
class RequestMeta:
    request_id: int
    arrive_time: Any = 0.0
    prompt_len: int = field(default = None)
    max_tokens: int = field(default = None)

class ObjectStatus(Enum):
    UNSET = 0
    GLOBAL = 1
    INTERPRETED = 2
    SCHEDULED = 3
    
@dataclass(kw_only=True)
class ObjectRef(Generic[T]):
    placement: Placement = field(default_factory=Placement) # This object is replicated on a list of devices.
    device_ref: ObjectDeviceRef = field(default_factory = ObjectDeviceRef)
    request_meta: Optional[RequestMeta] = None
    create_by: Optional[OperationID] = None # the op create this object, -1 for global object
    # used_by: List[OperationID] = field(default_factory=list) # the ops that use this object. 
    status: ObjectStatus = ObjectStatus.UNSET
    sub_refs: List['ObjectRef'] = field(default_factory=list)

    def __copy__(self):
        raise TypeError("Copying not allowed")

    def __deepcopy__(self, memo):
        raise TypeError("Copying not allowed")


'''
    regresion cost model for ['opt-125m']
    opt-125m 123543552
    figure saved to benchmark/pics/cost_model-opt-125m.png
    Optimized alpha, beta, gamma:
    (0.0006666666666666666, 3.134734137898978, 6.803053420073844)
    Mean Squared Error (MSE): 0.3767447241616227
    Root Mean Squared Error (RMSE): 0.6137953438741799
    Coefficient of Determination (R^2): 0.04078374237519411
    regresion cost model for ['opt-6.7b']
    opt-6.7b 6648365056
    figure saved to benchmark/pics/cost_model-opt-6.7b.png
    Optimized alpha, beta, gamma:
    (0.0006666666666666666, 8.425275974705016, 15.757891495425078)
    Mean Squared Error (MSE): 7.635536722283512
    Root Mean Squared Error (RMSE): 2.763247495662217
    Coefficient of Determination (R^2): 0.9918546407730865
    regresion cost model for ['opt-13b']
    opt-13b 12840304640
    figure saved to benchmark/pics/cost_model-opt-13b.png
    Optimized alpha, beta, gamma:
    (0.0006666666666666666, 8.329930088850096, 22.19457645948364)
    Mean Squared Error (MSE): 13.72942083838038
    Root Mean Squared Error (RMSE): 3.7053233109109898
    Coefficient of Determination (R^2): 0.9965719741816166
    '''
hyper_params_wo_cuda_graph = {
    'facebook/opt-125m': (0, 0, 6), # w/o cuda graph 8.93
    'facebook/opt-2.7b': (0, 0, 29.85),
    'facebook/opt-6.7b': (0, 0, 20), # w/o cuda graph 30
    'facebook/opt-13b': (0.0006666666666666666, 8.329930088850096, 22.19457645948364)
}

hyper_params_w_cuda_graph = {
    'facebook/opt-125m': (0, 0, 2.4339), # w/o cuda graph 8.93
    'facebook/opt-2.7b': (0, 0, 29.85),
    'facebook/opt-6.7b': (0, 0, 30), # w/o cuda graph 30
    'facebook/opt-13b': (0.0006666666666666666, 8.329930088850096, 22.19457645948364)
}

@dataclass 
class ModelRef(ObjectRef):
    model_tag: str
    embed_size: int
    num_heads: int
    vocab_size: int
    n_layer: int
    tp: int
    pp: int
    index: int 
    head_size: int 
    n_param: int
    backend: Backend
    element_size: int
    max_seq_len: int
    intermidiate_size: int
    use_cuda_graph: bool 

    def __post_init__(self):
        if self.use_cuda_graph:
            self.alpha, self.beta, self.gamma = hyper_params_w_cuda_graph[self.model_tag] 
        else: 
            self.alpha, self.beta, self.gamma = hyper_params_wo_cuda_graph[self.model_tag]
 
    def _cal_mem_acc_in_gb(self, bs, past_seq_len, cur_seq_len):
        mem_param = self.n_param
        mem_kvcache = bs * (cur_seq_len + past_seq_len)\
            * self.embed_size\
            * self.n_layer\
            * 2
        mem_vocab = self.vocab_size * self.embed_size 
        return (mem_param + mem_kvcache + mem_vocab) / 1e9

    def _cal_tflops(self, bs, past_seq_len, cur_seq_len):
        mlp_flops = bs * cur_seq_len * self.n_layer * ((self.embed_size*self.embed_size) * 4\
            + self.embed_size * self.intermidiate_size * 2)   # [bs, cur_seq_len, hs] X [hs, 4hs]
        attn_flops = self.n_layer * 2 * bs * cur_seq_len * \
            (cur_seq_len + past_seq_len) * self.embed_size
        return (mlp_flops + attn_flops) / 1e12
    
    def estimate_time(self, bs, past_seq_len, cur_seq_len):        
        return max(self._cal_mem_acc_in_gb(bs, past_seq_len, cur_seq_len) * self.alpha,\
            self._cal_tflops(bs, past_seq_len, cur_seq_len) * self.beta) + self.gamma

@dataclass 
class TensorRef(ObjectRef):
    shape: Tuple[Any, ...] = field(default_factory=tuple)
    def __getitem__(self, key: Union[int, slice, Tuple[Union[int, slice], ...]]):
        assert self.device_ref.access_key is None 
        if isinstance(key, slice):
            from SLOsServe.utils import slice_length
            sub_ref = TensorRef(
                placement=self.placement,
                device_ref=ObjectDeviceRef(
                    self.device_ref.uuid,
                    access_key=key
                ),
                shape = ((slice_length(key, self.shape[0]), ) + self.shape[1:]),
                status = self.status,
                create_by=self.create_by
            )
        else: 
            raise NotImplementedError
        self.sub_refs.append(sub_ref)
        return sub_ref

@dataclass 
class TokenizerRef(ObjectRef):
    model_tag: str 

@dataclass 
class ConstantRef(ObjectRef):
    type: Any

@dataclass 
class KVCacheRef(ObjectRef):
    length: int
    reserved: int 
    backend: Backend
    n_layer: int 
    head_size: int 
    num_heads: int 
    element_size: int 
    memory_usage: int = field(init = False, default = None) # in GB

    def __post_init__(self):
        self.memory_usage = self.n_layer\
        * self.head_size\
        * self.num_heads\
        * self.element_size\
        * self.reserved / 1e9


class ObjectImplDict(dict):
    def __init__(self):
        super().__init__()
        self.deleted = set()

    def __setitem__(self, key: object, value: Any):
        if not isinstance(key, ObjectDeviceRef): 
            raise ValueError("Key must be an ObjectDeviceRef instance")
        assert key.access_key is None 
        super().__setitem__(key.uuid, value)

    def __getitem__(self, ref: object):
        if not isinstance(ref, ObjectDeviceRef):
            raise ValueError("Key must be an ObjectDeviceRef instance")
        try: 
            item = super().__getitem__(ref.uuid)
        except KeyError as e: 
            if ref.uuid in self.deleted:
                raise RuntimeError('Accessing an deleted item')
            else: 
                raise RuntimeError(f"Caught a KeyError with message: {e}")
        if ref.access_key is not None:
            return item[ref.access_key]
        return item
    
    def __contains__(self, key: Any):
        assert isinstance(key, ObjectDeviceRef)            
        return super().__contains__(key.uuid)
    
    def pop(self, ref: object, default=None):
        if not isinstance(ref, ObjectDeviceRef):
            raise ValueError("Key must be an ObjectDeviceRef instance")
        
        if ref in self:
            value = super().pop(ref.uuid)
            self.deleted.add(ref.uuid)
            if ref.access_key is not None:
                return value.get(ref.access_key, default)
            return value
        else:
            if default is not None:
                return default
            elif ref.uuid in self.deleted:
                raise KeyError(f'Delete an deleted key')
            else:
                raise KeyError(f"Key {ref} not found.")

class OpAllocator:
    def __init__(self, 
                 op_id: OperationID, 
                 request_meta: RequestMeta):
        self.allocated = []
        self.op_id = op_id
        self.request_meta = request_meta 
        
    def new(self, ref_class, *args, **kwargs):
        obj = ref_class(
            create_by = self.op_id,
            request_meta = self.request_meta, 
            status = ObjectStatus.INTERPRETED,
            *args, **kwargs
        )
        self.allocated.append(obj)
        return obj

# class ObjectStateStore(dict):
#     def __init__(self):
#         super().__init__()
#     def __setitem__(self, key, value):
#         raise RuntimeError('You cannot set item implicitly,\
#                             you may use the new approach to \
#                            create a obj reference.')
    
#     class OpAllocator:
#         def __init__(self, 
#                     obj_dict: 'ObjectStateStore', 
#                     op_id, 
#                     request_meta):
#             self.obj_dict = obj_dict 
#             self.op_id = op_id
#             self.request_meta = request_meta
#             self.allocated:List[ObjectDeviceRef] = []

#         def get_meta(self, ref: ObjectDeviceRef, type: Optional[ObjectType] = None) -> Any:
#             return self.obj_dict.get_meta(ref, type)
        
#         def new(self, *args, **kwargs):
#             ref = self.obj_dict.new(
#                 *args, **kwargs,
#                 create_by = self.op_id, 
#                 request_meta = self.request_meta,
#                 status = ObjectStatus.INTERPRETED
#             )
#             self.allocated.append(ref)
#             return ref

#     def inc_dec_ref_func(self, global_context, request_meta):
#         from SLOsServe.operation import OpCode 
#         def _inc_impl(obj: ObjectRef):
#             assert isinstance(obj, ObjectRef)
#             self.__getitem__(obj).ref_count += 1
#         def _dec_impl(obj: ObjectRef):
#             assert isinstance(obj, ObjectRef)
#             if not global_context.is_finished(request_meta):
#                 obj_state = self.__getitem__(obj)
#                 obj_state.ref_count -= 1
#                 # We issue delete only before the cleanup is issued
#                 if obj_state.ref_count == 0:
#                     global_context.create_new_op(OpCode.DELETE, request_meta, ObjectDeviceRef(obj_state.uuid, None))
#         return _inc_impl, _dec_impl

#     def allocator(self, op_id, request_meta = None):
#         return self.OpAllocator(self, op_id, request_meta)

#     def new(self, *args, **kwargs) -> ObjectDeviceRef:
#         uuid = uuid1()
#         state = ObjectState(uuid = uuid, *args, **kwargs)
#         ref = ObjectDeviceRef(uuid, None)
#         super().__setitem__(uuid, state)
#         return ref

#     def __getitem__(self, ref: Union[ObjectDeviceRef, ObjectRef]) -> ObjectState:
#         return super().__getitem__(ref.uuid)

#     def get_meta(self, ref: ObjectDeviceRef, type: Optional[ObjectType] = None) -> Any:
#         item = self.__getitem__(ref)
#         if type is not None: 
#             assert item.type == type 
#         if (item.meta is not None) and (ref.access_key is not None): 
#             return item.meta[ref.access_key]
#         return item.meta
