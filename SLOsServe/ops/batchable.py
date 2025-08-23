from typing import Dict, Callable, List, Any, Tuple, get_args, get_origin
import inspect
import torch
from functools import update_wrapper
import ast


from ..object import (
    OpAllocator,
    ObjectImplDict,
    TensorRef,
    ObjectRef,
    ObjectDeviceRef,
    ConstantRef
)
from ..comm import Communicator
from ..device import ClusterStatus, DeviceGroup
from .operation import Node, Operator, OpCode
from .op_classes import OP_CLASSES

class Batchable:
    forward_impl: Callable[[], torch.Tensor]
    batch_forward_impl: Callable[[List[Any]], List[Any]] = None
    # place_impl: Callable[[Communicator, ClusterStatus, Any, Tuple[Any, ...], Dict[str, Any]], DeviceGroup] = Operator.place
    # get_profile_log_impl: Callable[[OpCode, ObjectImplDict, List[Any]], str] = Operator.get_profile_log 
    # estimate_load_impl: Callable[[List[Any]], None] = Operator.estimate_load
    
    def __init__(
        self, 
        forward_func: Callable
    ):
        self.name = forward_func.__name__
        self.forward_impl = forward_func
        # parse and type check
        self.sig: inspect.Signature = inspect.signature(self.forward_impl)
        
        if get_origin(self.sig.return_annotation) is tuple:
            self.return_types = list(get_args(self.sig.return_annotation))
        else:
            self.return_types = [self.sig.return_annotation]
        
        assert all((x in (torch.Tensor, int, float) for x in self.return_types))
        self.n_rets = len(self.return_types)
        
        self._output_names = Batchable._optional_output_names(self.forward_impl)
        if self._output_names is None:
            self._output_names = [f'Out.{i}' for i in range(self.n_rets)]
        assert len(self._output_names) == self.n_rets
        
        assert self.name not in OP_CLASSES
        OP_CLASSES[self.name] = self
    
    @staticmethod
    def _optional_output_names(func: Callable):
        try:
            output_names = []
            stmt = ast.parse(inspect.getsource(func)).body[0].body[-1]
            if isinstance(stmt.value, ast.Tuple):
                for obj in stmt.value.elts:
                    if not isinstance(obj, ast.Name): return None
                    output_names.append(obj.id)
            else: 
                if not isinstance(stmt.value, ast.Name): return None
                output_names.append(obj.id)
            return output_names
        except:
            print('unsucessful parsing outputnames')
            return None
    
    def register_batch_impl(self, batch_impl: Callable):
        if batch_impl is not None: 
            self.batch_forward_impl = self.concatenate_batcher(batch_impl)
        return self 
    
    def register_shape_inference(self, shape_inference: Callable):
        self.shape_inference_impl = shape_inference 
        return self
    
    def has_batched_impl(self) -> bool:
        return self.batch_forward_impl is not None
    
    def create_op(self, 
                  allocator: OpAllocator, 
                  **kwargs):
        input_refs = []
        args = {}
        input_shapes = []
        for param in self.sig.parameters.values():
            if param.name in kwargs:
                param_ref = kwargs[param.name]
            else: 
                assert(param.default is not inspect._empty)
                param_ref = param.default
                
            if param.annotation == torch.Tensor:
                assert(isinstance(param_ref, TensorRef))
                input_refs.append(param_ref)
                args[param.name] = param_ref.device_ref
                input_shapes.append(param_ref.shape)
            else:
                args[param.name] = param_ref
                input_shapes.append(param_ref)

        output_shapes = self.shape_inference_impl(*input_shapes)
        assert isinstance(output_shapes, list)
        assert(len(output_shapes) == self.n_rets)

        outputs = []
        for _name, shape, type in zip(self._output_names, output_shapes, self.return_types):
            if type == torch.Tensor:
                output = allocator.new(TensorRef, shape = shape)
            elif type == int or type == float:
                output = allocator.new(ConstantRef, type = type)
            args[_name] = output.device_ref
            outputs.append(output)
        
        return Node(
            self.name,
            (OpCode.BATCHABLE, self.name),
            input_refs = input_refs,
            output_refs = allocator.allocated,
            args = [args],
            output = outputs if len(outputs) > 1 else outputs[0]
        )
    
    def __call__(self, ctx: 'RequestContext', **kwargs)->ObjectRef:
        return ctx.create_op(
            self.name, **kwargs
        )

    def forward(
        self,
        obj_impls: ObjectImplDict, 
        args: Dict[str, Any]):
        inputs = {param.name: obj_impls[args[param.name]] \
            if isinstance(args[param.name], ObjectDeviceRef) else args[param.name]\
                  for param in self.sig.parameters.values()}
        outputs = self.forward_impl(**inputs)
        if self.n_rets == 1:
            obj_impls[args[self._output_names[0]]] = outputs
        else:
            assert len(outputs) == self.n_rets 
            for _name, out in zip(self._output_names, outputs):
                obj_impls[args[_name]] = out

    def batch_forward(
        self,
        obj_impls: ObjectImplDict, 
        batched_args: List[Dict[str, Any]]):
        
        batched_inputs = [{param.name: (obj_impls[args[param.name]] \
            if isinstance(args[param.name], ObjectDeviceRef) else args[param.name])\
            for param in self.sig.parameters.values()}\
            for args in batched_args]
        
        
        outputs = self.batch_forward_impl(batched_inputs)
        
        for args, out in zip(batched_args, outputs):
            if self.n_rets == 1:
                out = [out]
            assert len(out) == self.n_rets
            for _out_name, _out in zip(self._output_names, out):
                obj_impls[args[_out_name]] = _out
            

    def place(
        self,
        communicator: Communicator,
        cluster_status: ClusterStatus,
        output: Any,
        **kwargs 
    ) -> DeviceGroup:
        device_group = DeviceGroup(devices = (0,))
        if self.n_rets == 1:
            output = [output]
        for out in output:
            assert issubclass(type(out), ObjectRef)
            out.placement.add_device_group(device_group)
        return device_group
    
    def get_profile_log(
        self,
        op_code: OpCode,
        obj_impls: ObjectImplDict, 
        batched_args: List[Any]
    ):
        return self.name

    def estimate_load(
        self, 
        load_metas: List[Any]  
    ) -> float: # the estimated time in milliseconds
        return 1e-3
    
    def concatenate_batcher(
        self,
        batched_function: Callable
    ):
        def impl(inputs: List[Dict[str, Any]]) -> List[torch.Tensor]:
            args = {}
            for param in self.sig.parameters.values():
                _args = [inputs_[param.name] for inputs_ in inputs]
                for _arg in _args:
                    assert isinstance(_arg, param.annotation)
                args[param.name] = torch.stack(_args) if param.annotation == torch.Tensor else _args
            return batched_function(**args)
        return impl 
        
def batchable(
    shape_inference: Callable = None,
    batched_impl: Callable = None):
    def impl(func: Callable) -> Batchable:
        return Batchable(func)\
            .register_batch_impl(batched_impl)\
            .register_shape_inference(shape_inference)
    return impl

if __name__ == '__main__':
    @Batchable
    def func(a: torch.Tensor, b: float, c: Tuple) -> torch.Tensor:
        return a * b
    
    output = func.forward_impl(torch.randn((1,2), 3))
    print(output)