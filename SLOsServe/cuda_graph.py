import torch.nn as nn
import torch 
from typing import Dict, Optional, List, Union, Tuple
import gc

from vllm.distributed import get_pp_group
from vllm.attention import AttentionMetadata
from vllm.distributed.parallel_state import graph_capture
from vllm.sequence import (IntermediateTensors)
import logging 
import time 
logger = logging.getLogger(__name__)

_NUM_WARMUP_ITERS = 2

_PAD_SLOT_ID = -1
_BATCH_SIZE_ALIGNMENT = 16
# Capture graphs for token size 1, 2, 4, 8, 16, 24, 32, 40, ..., 256.
# NOTE: _get_graph_batch_size needs to be updated if this list is changed.
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [
    _BATCH_SIZE_ALIGNMENT * i for i in range(1, 33)
]
_MAX_BATCH_SIZE=max(_BATCH_SIZES_TO_CAPTURE)
def _get_graph_batch_size(batch_size: int) -> int:
    """Returns the padded batch size given actual batch size.

    Batch sizes are 1, 2, 4, _BATCH_SIZE_ALIGNMENT,
    2*_BATCH_SIZE_ALIGNMENT, 3*_BATCH_SIZE_ALIGNMENT...
    """
    assert batch_size <= 256
    if batch_size <= 2:
        return batch_size
    elif batch_size <= 4:
        return 4
    else:
        return ((batch_size + _BATCH_SIZE_ALIGNMENT - 1) //
                _BATCH_SIZE_ALIGNMENT * _BATCH_SIZE_ALIGNMENT)

class CUDAGraphRunner:

    def __init__(self, model: nn.Module):
        self.model = model

        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

        self._graph: Optional[torch.cuda.CUDAGraph] = None

    @property
    def graph(self):
        assert self._graph is not None
        return self._graph

    def capture(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_or_intermediate_states: Optional[Union[IntermediateTensors,
                                                      torch.Tensor]],
        intermediate_inputs: Optional[IntermediateTensors],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        memory_pool: Optional[Tuple[int, int]],
        stream: torch.cuda.Stream,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        assert self._graph is None
        # Run the model a few times without capturing the graph.
        # This is to make sure that the captured graph does not include the
        # kernel launches for initial benchmarking (e.g., Triton autotune).
        # Note one iteration is not enough for torch.jit.script
        for _ in range(_NUM_WARMUP_ITERS):
            self.model(
                input_ids,
                positions,
                kv_caches,
                attn_metadata,
                intermediate_inputs,
                **kwargs,
            )
        torch.cuda.synchronize()

        # Capture the graph.
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=memory_pool, stream=stream):
            output_hidden_or_intermediate_states = self.model(
                input_ids,
                positions,
                kv_caches,
                attn_metadata,
                intermediate_inputs,
                **kwargs,
            )
            if hidden_or_intermediate_states is not None:
                if get_pp_group().is_last_rank:
                    hidden_or_intermediate_states.copy_(
                        output_hidden_or_intermediate_states)
                else:
                    for key in hidden_or_intermediate_states.tensors:
                        hidden_or_intermediate_states[key].copy_(
                            output_hidden_or_intermediate_states[key])
            else:
                hidden_or_intermediate_states = (
                    output_hidden_or_intermediate_states)

            del output_hidden_or_intermediate_states
            # make sure `output_hidden_states` is deleted
            # in the graph's memory pool
            gc.collect()
        torch.cuda.synchronize()

        # Save the input and output buffers.
        self.input_buffers = {
            "input_ids": input_ids,
            "positions": positions,
            "kv_caches": kv_caches,
            "slot_mapping": attn_metadata.slot_mapping,
            "seq_lens_tensor":
            attn_metadata.decode_metadata.seq_lens_tensor,
            "block_tables": attn_metadata.decode_metadata.block_tables,
            **kwargs,
        }
        if intermediate_inputs is not None:
            self.input_buffers.update(intermediate_inputs.tensors)
        if get_pp_group().is_last_rank:
            self.output_buffers = {
                "hidden_states": hidden_or_intermediate_states
            }
        else:
            self.output_buffers = hidden_or_intermediate_states
        return hidden_or_intermediate_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs,
    ) -> torch.Tensor:
        # KV caches are fixed tensors, so we don't need to copy them.
        del kv_caches

        # Copy the input tensors to the input buffers.
        self.input_buffers["input_ids"].copy_(input_ids, non_blocking=True)
        self.input_buffers["positions"].copy_(positions, non_blocking=True)
        self.input_buffers["slot_mapping"].copy_(attn_metadata.slot_mapping,
                                                 non_blocking=True)
        # if self.backend_name != "flashinfer":
        self.input_buffers["seq_lens_tensor"].copy_(
            attn_metadata.decode_metadata.seq_lens_tensor,
            non_blocking=True)
        self.input_buffers["block_tables"].copy_(
            attn_metadata.decode_metadata.block_tables, non_blocking=True)
        if "seqlen_agnostic_capture_inputs" in self.input_buffers:
            self.model.copy_inputs_before_cuda_graphs(self.input_buffers,
                                                      **kwargs)
        if intermediate_tensors is not None:
            for key in intermediate_tensors.tensors:
                self.input_buffers[key].copy_(intermediate_tensors[key],
                                              non_blocking=True)
        # Run the graph.
        self.graph.replay()
        if "seqlen_agnostic_capture_inputs" in self.input_buffers:
            self.model.copy_outputs_after_cuda_graphs(self.input_buffers,
                                                      **kwargs)
        # Return the output tensor.
        if get_pp_group().is_last_rank:
            return self.output_buffers["hidden_states"]

        return self.output_buffers

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)