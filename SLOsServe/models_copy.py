from dataclasses import dataclass, field
import logging 
from typing import List, Dict, Optional, Tuple, Any
import torch 
import numpy as np
import time
from itertools import accumulate

from vllm.distributed import get_pp_group
from vllm.distributed.parallel_state import graph_capture, set_group_tag
from vllm.attention.backends.flash_attn import FlashAttentionMetadata

from .vllm_utils import is_vllm_v1

if is_vllm_v1():
    from vllm.forward_context import set_forward_context
    from  vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
else: 
    from vllm.attention.backends.flash_attn import FlashAttentionMetadata

import math

from .backend import Backend
from .cuda_graph import (
    _PAD_SLOT_ID,
    _BATCH_SIZES_TO_CAPTURE,
    _get_graph_batch_size,
    CUDAGraphRunner,
    _MAX_BATCH_SIZE
)

logger = logging.getLogger(__name__)

MAX_TOT_MEM = 22e9
MAX_NUM_TOKENS = 5e4
MAX_MEM_USE_RATE = 0.60

@dataclass
class PagedKVCacheManager:
    model_tag: str 
    block_size: int 
    num_blocks: int 
    num_kv_heads: int 
    head_size: int 
    num_layer: int 
    dtype: torch.dtype 
    device: torch.device
    cache_type: str = 'linear'
    kv_caches: List[torch.Tensor] = field(init = False)
    free_blocks: List[int] = field(init = False)
    vllm_config: Any | None = None
    
    def __post_init__(self):
        # kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
        assert self.cache_type in ('linear', 'tree')
        self.cache_class = RequestCacheManager if self.cache_type == 'linear' else TreeRequestCacheManager
        self.kv_caches = [torch.empty(
            2, 
            self.num_blocks, 
            self.block_size, 
            self.num_kv_heads, 
            self.head_size, 
            device = self.device, 
            dtype = self.dtype
        ) for _ in range(self.num_layer)]
        self.free_blocks = list(range(self.num_blocks))
        self.max_allocated = 0
        
        ## bind caches to layers
        if is_vllm_v1():
            ctx = self.vllm_config.compilation_config.static_forward_context
            from vllm.model_executor.models.utils import extract_layer_index
            sorted_layer_names = sorted(list(ctx.keys()), key = lambda x: extract_layer_index(x))
            assert len(sorted_layer_names) == self.num_layer
            for kv_cache, layer_name in zip(self.kv_caches, sorted_layer_names):
                ctx[layer_name].kv_cache = [kv_cache]
        

    def _allocate_blocks(self, n = 1) -> List[int]:
        n_free = len(self.free_blocks)
        if n > n_free:
            raise RuntimeError(f'Out of memory for KVCache for {self.model_tag}')
        self.max_allocated = max(self.max_allocated, self.num_blocks - n_free)
        allocated = self.free_blocks[:n]
        self.free_blocks = self.free_blocks[n:]
        return allocated
    
    def new(
        self,
        init_num_tokens:int = 0,
        init_tensors: Optional[List[torch.Tensor]] = None 
    ):
        blocks = []
        if init_num_tokens != 0 and init_tensors is not None:
            assert len(init_tensors) == self.num_layer
            kv, num_blocks, block_size, num_kv_heads, head_size = init_tensors[0].size()
            assert kv == 2 and block_size == self.block_size \
                        and num_kv_heads == self.num_kv_heads \
                        and head_size == self.head_size
            blocks = self._allocate_blocks(num_blocks)
            for init_tensor, kv_cache in zip(init_tensors, self.kv_caches):
                kv_cache.index_copy_(1, torch.tensor(blocks, device = self.device, dtype = torch.long)
                                     , init_tensor)
        return self.cache_class(
            self, 
            init_num_tokens,
            blocks
        )
    
    def get_status(self):
        return {f'KVCache allocated {self.model_tag}': self.num_blocks - len(self.free_blocks), 
                f'KVCache #block {self.model_tag}': self.num_blocks,
                f'KVCache Usage {self.model_tag} (%)': round((1 - len(self.free_blocks) / self.num_blocks) * 100, 2),
                f'Max KVCache Usage {self.model_tag} (%)': round(self.max_allocated / self.num_blocks * 100, 2)}
    
    def get_num_block(self, n_token: int) -> int:
        return int(math.ceil(n_token / self.block_size))
@dataclass 
class RequestCacheManager:
    global_manager: PagedKVCacheManager
    num_tokens: int 
    blocks: List[int] = field(default_factory=list)

    def new(self):
        return self

    def update(self, 
               num_new_token: int,
               num_drop_token: int = 0) -> List[int]:
        assert self.num_tokens >= num_drop_token 
        self.num_tokens -= num_drop_token
        past_seq_len = self.num_tokens
        num_blocks = (self.num_tokens + num_new_token + self.global_manager.block_size - 1)\
                // self.global_manager.block_size
        if num_blocks > len(self.blocks):
            self.blocks.extend(self.global_manager._allocate_blocks(num_blocks - len(self.blocks)))
        slot_mapping = [(self.blocks[(self.num_tokens + idx) // self.global_manager.block_size] * self.global_manager.block_size
                        + (self.num_tokens + idx) % self.global_manager.block_size) for idx in range(num_new_token)]
        self.num_tokens += num_new_token
        return past_seq_len, slot_mapping
    
    def get_aggregated_cache(self) -> List[torch.Tensor]: 
        return [cache[:, self.blocks] for cache in self.global_manager.kv_caches]

    def __del__(self):
        self.global_manager.free_blocks.extend(self.blocks)
        
    def free(self):
        self.global_manager.free_blocks.extend(self.blocks)
        self.blocks = []
        self.num_tokens = 0

@dataclass 
class TreeRequestCacheManager:
    global_manager: PagedKVCacheManager
    num_tokens: int 
    blocks: List[int] = field(default_factory=list)
    parent: 'TreeRequestCacheManager' = None
    frozen: bool = False

    def __post_init__(self):
        assert(self.global_manager.block_size==1)
        
    def new(self) -> 'TreeRequestCacheManager':
        self.frozen = True 
        self.num_tokens_tot = self.num_tokens
        self.slot_mapping_tot = self.blocks
        self.blocks_tot = self.blocks
        
        if self.parent is not None:
            assert isinstance(self.parent, TreeRequestCacheManager) and self.parent.frozen
            self.num_tokens_tot += self.parent.num_tokens_tot
            self.slot_mapping_tot = self.parent.slot_mapping_tot + self.slot_mapping_tot
            self.blocks_tot = self.parent.blocks_tot + self.blocks
            
        return TreeRequestCacheManager(
            global_manager = self.global_manager,
            num_tokens = 0,
            blocks = [],
            parent = self
        )

    def update(self, 
               num_new_token: int,
               num_drop_token: int = 0) -> List[int]:
        assert not self.frozen
        assert self.num_tokens >= num_drop_token 
        self.num_tokens -= num_drop_token
        past_seq_len = self.num_tokens
        num_blocks = (self.num_tokens + num_new_token + self.global_manager.block_size - 1)\
                // self.global_manager.block_size
        if num_blocks > len(self.blocks):
            self.blocks.extend(self.global_manager._allocate_blocks(num_blocks - len(self.blocks)))
        slot_mapping = [(self.blocks[(self.num_tokens + idx) // self.global_manager.block_size] * self.global_manager.block_size
                        + (self.num_tokens + idx) % self.global_manager.block_size) for idx in range(num_new_token)]
        self.num_tokens += num_new_token
        if self.parent is not None:
            return self.parent.num_tokens_tot + past_seq_len, self.parent.slot_mapping_tot+slot_mapping
        return past_seq_len, slot_mapping
    
    def get_aggregated_cache(self) -> List[torch.Tensor]: 
        if self.parent is not None:
            return [cache[:, self.parent.blocks_tot + self.blocks] for cache in self.global_manager.kv_caches]
        return [cache[:, self.blocks] for cache in self.global_manager.kv_caches]
    
    def __del__(self):
        self.global_manager.free_blocks.extend(self.blocks)

@dataclass 
class ModelConfig:
    tag: str 
    embed_size: int
    num_heads: int
    head_dim: int
    vocab_size: int
    n_layer: int
    n_param: int
    max_seq_len: int 
    intermidiate_size: int
    n_elem: int = 2 # TODO(update this for dtypes)
    
    def get_token_cache_mem(self):
        return self.n_layer * self.num_heads * self.head_dim * self.n_elem
        
def get_model_config(
    model_tag: str 
) -> ModelConfig:
    from transformers import AutoConfig, OPTConfig, GPT2Config, Qwen2Config, LlamaConfig
    config = AutoConfig.from_pretrained(model_tag)
    if isinstance(config, GPT2Config):
        intermidiate_size = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        # print('config', config)
        return ModelConfig(
        tag = model_tag,
        embed_size = config.n_embd,
        num_heads = config.n_head,
        head_dim = config.n_embd // config.n_head,
        vocab_size = config.vocab_size,
        n_layer=config.n_layer,
        n_param = config.n_layer * (config.n_embd ** 2 * 4 + config.n_embd * intermidiate_size * 2)+\
        + config.n_embd * config.vocab_size,
        max_seq_len = config.n_ctx,
        intermidiate_size=intermidiate_size 
    )
    elif isinstance(config, OPTConfig):
        return ModelConfig(
            tag = model_tag, 
            embed_size=config.hidden_size,
            num_heads = config.num_attention_heads,
            head_dim = config.hidden_size // config.num_attention_heads,
            vocab_size = config.vocab_size,
            n_layer = config.num_hidden_layers,
            n_param = config.num_hidden_layers * (config.hidden_size ** 2 * 4 + config.hidden_size * config.ffn_dim * 2)\
            + config.hidden_size * config.vocab_size,
            max_seq_len=config.max_position_embeddings,
            intermidiate_size=config.ffn_dim
        )
    elif isinstance(config, Qwen2Config):
        '''
        Qwen2Config {
        "_name_or_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "architectures": [
            "Qwen2ForCausalLM"
        ],
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151643,
        "hidden_act": "silu",
        "hidden_size": 1536,
        "initializer_range": 0.02,
        "intermediate_size": 8960,
        "max_position_embeddings": 131072,
        "max_window_layers": 21,
        "model_type": "qwen2",
        "num_attention_heads": 12,
        "num_hidden_layers": 28,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-06,
        "rope_scaling": null,
        "rope_theta": 10000,
        "sliding_window": null,
        "tie_word_embeddings": false,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.48.3",
        "use_cache": true,
        "use_mrope": false,
        "use_sliding_window": false,
        "vocab_size": 151936
        }
        '''
        return ModelConfig(
            tag = model_tag,
            embed_size = config.hidden_size,
            num_heads = config.num_key_value_heads, 
            head_dim = config.hidden_size // config.num_attention_heads,
            vocab_size = config.vocab_size,
            n_layer = config.num_hidden_layers,
            n_param = config.num_hidden_layers * (config.hidden_size ** 2 * 4 + config.hidden_size * config.intermediate_size * 2)\
            + config.hidden_size * config.vocab_size,
            max_seq_len = 16384,# config.max_position_embeddings,
            intermidiate_size = config.intermediate_size
        )
    elif 'deepseek' in model_tag:
        return ModelConfig(
            tag = model_tag, 
            embed_size = config.hidden_size,
            num_heads = config.num_attention_heads,
            head_dim = config.hidden_size // config.num_attention_heads,
            vocab_size = config.vocab_size,
            n_layer = config.num_hidden_layers,
            n_param = config.num_hidden_layers * (config.hidden_size ** 2 * 4 + config.hidden_size * config.moe_intermediate_size * 2)+\
        + config.hidden_size * config.vocab_size,
            max_seq_len = config.max_position_embeddings,
            intermidiate_size=config.moe_intermediate_size
        )
    elif isinstance(config, LlamaConfig):
        return ModelConfig(
            tag = model_tag,
            embed_size = config.hidden_size,
            num_heads = config.num_key_value_heads, 
            head_dim = config.hidden_size // config.num_attention_heads,
            vocab_size = config.vocab_size,
            n_layer = config.num_hidden_layers,
            n_param = config.num_hidden_layers * (config.hidden_size ** 2 * 4 + config.hidden_size * config.intermediate_size * 2)\
            + config.hidden_size * config.vocab_size,
            max_seq_len = 16384,# config.max_position_embeddings,
            intermidiate_size = config.intermediate_size
        )
    else: raise RuntimeError(f'unknown model {model_tag} type: {type(config)}')
    
@dataclass 
class ModelImpl:
    module: torch.nn.Module 
    name: str
    embed_size: int
    local_num_heads: int
    local_num_layers: int
    backend: Backend
    dtype: torch.dtype 
    memory_usage: int 
    max_seq_len: int
    head_size: int
    intermidiate_size: int 
    n_params: int 
    vocab_size: int
    cache_type: str
    local_embed_size: int = field(init = False)
    device: torch.device
    block_size: int = 16
    cache_manager: PagedKVCacheManager = field(init = False, default = None)
    use_cuda_graph: bool = False
    tp: int = 1 
    pp: int = 1
    num_blocks: int | None = None
    vllm_config: Any | None = None


    def __post_init__(self):
        set_group_tag(self.name)
        per_block_size = self.block_size \
                        * self.local_num_heads \
                        * self.head_size * 2 \
                        * torch.empty(1, dtype = self.dtype).element_size()\
                        * self.local_num_layers

        if self.num_blocks is None:
            total_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            available_memory = MAX_MEM_USE_RATE * total_memory - torch.cuda.memory_allocated()
            # self.num_blocks = int(min(MAX_NUM_TOKENS // self.block_size, MAX_TOT_MEM // per_block_size))
            self.num_blocks = int(available_memory // per_block_size)
        
        self.local_embed_size = self.local_num_heads * self.head_size
        self.kvcache_memory_in_gb = round(per_block_size * self.num_blocks / 1e9, 2)

        self.cache_manager = PagedKVCacheManager(
            model_tag = self.name,
            block_size = self.block_size,
            num_blocks = self.num_blocks,
            num_kv_heads=self.local_num_heads,
            head_size = self.head_size, 
            num_layer = self.local_num_layers,
            dtype = self.dtype, 
            device = self.device,
            cache_type=self.cache_type,
            vllm_config = self.vllm_config
        )
        self.graph_block_tables = np.zeros(
            (_MAX_BATCH_SIZE, (self.max_seq_len + self.block_size - 1) // self.block_size),
            dtype=np.int32)

        print(f"Allocated {self.kvcache_memory_in_gb} GB for KVCache of {self.name} on"
              f" {self.device} for {self.num_blocks * self.block_size} ({self.num_blocks} X {self.block_size}) tokens.")

        if self.use_cuda_graph:
            self.graph_memory_pool: Optional[Tuple[
            int, int]] = None  # Set during graph capture.
            self.capture_model([self.cache_manager.kv_caches])


    @torch.inference_mode()
    def capture_model(self, kv_caches: List[List[torch.Tensor]]) -> None:
        """Cuda graph capture a model.

        Note that CUDA graph's performance gain is negligible if number
        of batched tokens are larger than 200. And since CUDA graph
        requires fixed sized tensors, supporting large/variable batch
        size requires high GPU memory overhead. Thus, vLLM only captures
        decoding requests. Mixed batch (chunked prefill + decoding) or
        prefill requests are not captured.

        Since it is used for decoding-only, it assumes there's only 1 token
        per sequence in the batch.
        """
        logger.info("Capturing the model for CUDA graphs. This may lead to "
                    "unexpected consequences if the model is not static. To "
                    "run the model in eager mode, set 'enforce_eager=True' or "
                    "use '--enforce-eager' in the CLI.")
        logger.info("CUDA graphs can take additional 1~3 GiB memory per GPU. "
                    "If you are running out of memory, consider decreasing "
                    "`gpu_memory_utilization` or enforcing eager mode. "
                    "You can also reduce the `max_num_seqs` as needed "
                    "to decrease memory usage.")
        start_time = time.perf_counter()

        # Prepare dummy inputs. These will be reused for all batch sizes.
        max_batch_size = _MAX_BATCH_SIZE
        input_tokens = torch.zeros(max_batch_size, dtype=torch.long).cuda()
        input_positions = torch.zeros(max_batch_size, dtype=torch.long).cuda()
        slot_mapping = torch.empty(max_batch_size, dtype=torch.long).cuda()
        slot_mapping.fill_(_PAD_SLOT_ID)
        seq_lens = torch.ones(max_batch_size, dtype=torch.int32).cuda()
        block_tables = torch.from_numpy(self.graph_block_tables).cuda()
        intermediate_inputs = None
        assert get_pp_group().is_first_rank

        # Prepare buffer for outputs. These will be reused for all batch sizes.
        # It will be filled after the first graph capture.
        hidden_or_intermediate_states: List[Optional[torch.Tensor]] = [
            None
        ] * self.pp

        batch_size_capture_list = [
            batch_size for batch_size in _BATCH_SIZES_TO_CAPTURE
        ]

        self.graph_runners: List[Dict[int, CUDAGraphRunner]] = [
            {} for _ in range(self.pp)
        ]

        with graph_capture() as graph_capture_context:
            # NOTE: Capturing the largest batch size first may help reduce the
            # memory usage of CUDA graph.
            for virtual_engine in range(
                    self.pp):
                for batch_size in reversed(batch_size_capture_list):
                    if is_vllm_v1():
                        attn_metadata = None
                        with set_forward_context(attn_metadata, 
                                            self.vllm_config,
                                            virtual_engine,
                                            num_tokens = batch_size,
                                            num_tokens_across_dp=None
                                            ):
                            for _ in range(2):
                                self.module(
                                    input_ids = input_tokens[:batch_size],
                                    positions = input_positions[:batch_size]
                                )
                            self.module(
                                input_ids = input_tokens[:batch_size],
                                positions = input_positions[:batch_size]
                            )
                    else:
                        attn_metadata = FlashAttentionMetadata(
                            num_prefills = 0,
                            num_prefill_tokens = 0,
                            num_decode_tokens = batch_size,
                            slot_mapping=slot_mapping[:batch_size],
                            seq_lens=[1] * batch_size,
                            max_query_len=1,
                            max_prefill_seq_len = 0,
                            max_decode_seq_len=self.max_seq_len,
                            query_start_loc=torch.arange(batch_size + 1, device = self.device, dtype = torch.int32),
                            seq_start_loc= torch.arange(batch_size + 1, device = self.device, dtype = torch.int32),
                            context_lens_tensor = torch.zeros(batch_size, device = self.device, dtype = torch.int32),
                            block_tables=block_tables[:batch_size],
                            seq_lens_tensor = seq_lens[:batch_size],
                            use_cuda_graph = True,
                        )

                        graph_runner = CUDAGraphRunner(
                            self.module)

                        capture_inputs = {
                            "input_ids":
                            input_tokens[:batch_size],
                            "positions":
                            input_positions[:batch_size],
                            "hidden_or_intermediate_states":
                            hidden_or_intermediate_states[
                                virtual_engine]  # type: ignore
                            [:batch_size]
                            if hidden_or_intermediate_states[virtual_engine]
                            is not None else None,
                            "intermediate_inputs":
                            intermediate_inputs[:batch_size]
                            if intermediate_inputs is not None else None,
                            "kv_caches":
                            kv_caches[virtual_engine],
                            "attn_metadata":
                            attn_metadata,
                            "memory_pool":
                            self.graph_memory_pool,
                            "stream":
                            graph_capture_context.stream
                        }


                        graph_runner.capture(**capture_inputs)
                        self.graph_memory_pool = graph_runner.graph.pool()
                        self.graph_runners[virtual_engine][batch_size] = (
                            graph_runner)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # This usually takes < 10 seconds.
        print(f"Graph capturing of {self.name} finished in {elapsed_time} secs.")

    def forward(
        self,
        input_ids: torch.Tensor,
        req_cache_managers: List[RequestCacheManager],
        cur_seq_lens: List[int],
        rewind_sizes: List[int]
    ):
        # set_group_tag(self.name)
        batch_size = len(req_cache_managers)
        token_batch_size = sum(cur_seq_lens)
        assert len(cur_seq_lens) == batch_size
        assert len(rewind_sizes) == batch_size
        assert input_ids.size(0) == sum(cur_seq_lens)
        # if batch_size > _MAX_BATCH_SIZE:
        #     print(f'Batchsize {batch_size} > max batch size{_MAX_BATCH_SIZE}')
        #     raise RuntimeError(f'Batchsize {batch_size} > max batch size{_MAX_BATCH_SIZE}')
        past_seq_lens = []
        slot_mappings = []
        block_tables = []

        num_prefills = 0
        for req_cache_manager, cur_seq_len, rewind_size in zip(
            req_cache_managers, cur_seq_lens, rewind_sizes
        ):
            past_seq_len, slot_mapping = req_cache_manager.update(
                num_new_token= cur_seq_len, num_drop_token=rewind_size)
            past_seq_lens.append(past_seq_len)
            slot_mappings.extend(slot_mapping)
            block_tables.append(req_cache_manager.blocks)
            if cur_seq_len != 1:
                num_prefills += 1
        num_prefill_tokens = sum(cur_seq_lens[:num_prefills])
        
        seq_lens = [past_seq_len + cur_seq_len for past_seq_len, cur_seq_len in zip(past_seq_lens, cur_seq_lens)]
        position_ids = sum([list(range(past_seq_len, seq_len)) for past_seq_len, seq_len in zip(past_seq_lens, seq_lens)], start = [])
        for i, block_table in enumerate(block_tables):
            assert self.graph_block_tables.shape[-1] >= len(block_table)
            self.graph_block_tables[i, :len(block_table)] = block_table

        graph_bs = _get_graph_batch_size(token_batch_size)
        use_cuda_graph = (is_vllm_v1() or num_prefills == 0) and self.use_cuda_graph and (graph_bs >= token_batch_size)
        executable = self.module
        if use_cuda_graph:
            if not is_vllm_v1():
                executable = self.graph_runners[get_pp_group().rank_in_group][graph_bs]
            # num_decode_tokens = graph_bs - token_batch_size
            input_ids = torch.concat([input_ids, torch.zeros(size = (graph_bs - token_batch_size,), 
                                   dtype = input_ids.dtype, 
                                   device = input_ids.device)])
            zero_pads = [0] * (graph_bs - token_batch_size)
            ones_pads = [1] * (graph_bs - token_batch_size)
            position_ids += zero_pads
            past_seq_lens += zero_pads
            cur_seq_lens += ones_pads
            seq_lens += ones_pads
            slot_mappings += [_PAD_SLOT_ID] * (graph_bs - token_batch_size)
            block_tables = self.graph_block_tables[:batch_size + graph_bs - token_batch_size]
        else:
            num_decode_tokens =  batch_size - num_prefills
            block_tables = self.graph_block_tables[:batch_size, :max(map(lambda x: len(x), block_tables))]
        # print('cur_seq_lens', cur_seq_lens)
        # print('rewind_sizes', rewind_sizes)
        # print('use_cuda_graph', use_cuda_graph)
        # print('input_ids', input_ids.size())
        # print('position_ids:', len(position_ids), position_ids)
        # print('seq_lens', len(seq_lens), seq_lens)
        # print('cur_seq_lens', len(cur_seq_lens), cur_seq_lens)
        # print('slot_mapping', len(slot_mappings), slot_mappings)
        # print('num_prefills', num_prefills)
        # print('num_prefill_tokens', num_prefill_tokens)
        # print('num_decode_tokens', num_decode_tokens)
        # print('past_seq_lens', len(past_seq_lens), past_seq_lens)
        # print(flush=True)
        if is_vllm_v1():
            # from vllm.v1.attention.backends import FlashAttentionMetadata
            print('cur_seq_lens', cur_seq_lens)
            print('seq_lens', seq_lens)
            print('block_tables', block_tables.shape)
            print('num_actual_tokens', token_batch_size)
            print('max_query_len', max(cur_seq_lens))
            
            attn_metadata = FlashAttentionMetadata(
                num_actual_tokens = token_batch_size,
                max_query_len = max(cur_seq_lens),
                query_start_loc = torch.tensor([0] + list(accumulate(cur_seq_lens)), dtype = torch.int32, device = self.device),
                max_seq_len = max(seq_lens),
                seq_lens = torch.tensor(seq_lens, dtype = torch.int32, device = self.device), 
                block_table = torch.from_numpy(block_tables).to(self.device), 
                slot_mapping = torch.tensor(slot_mappings, device = self.device, dtype = torch.int64),
                use_cascade = False, 
                common_prefix_len = 0,
                cu_prefix_query_lens = None, 
                prefix_kv_lens = None,
                suffix_kv_lens = None
            )
        else: 
            attn_metadata = FlashAttentionMetadata(
                num_prefills = num_prefills,
                num_prefill_tokens = num_prefill_tokens,
                num_decode_tokens = num_decode_tokens,
                slot_mapping = torch.tensor(slot_mappings, device = self.device, dtype = torch.int64),
                seq_lens = seq_lens,
                seq_lens_tensor = torch.tensor(seq_lens, dtype = torch.int32, device = self.device),
                max_query_len = max(cur_seq_lens),
                max_prefill_seq_len = max(seq_lens[:num_prefills], default = 0),
                max_decode_seq_len = max(seq_lens[num_prefills:], default = 0),
                query_start_loc = torch.tensor([0] + list(accumulate(cur_seq_lens)), dtype = torch.int32, device = self.device),
                seq_start_loc = torch.tensor([0] + list(accumulate(seq_lens)), dtype = torch.int32, device = self.device),
                context_lens_tensor = torch.tensor(past_seq_lens, dtype = torch.int32, device = self.device),
                block_tables = torch.from_numpy(block_tables).to(self.device),
                use_cuda_graph=False
            )
        if is_vllm_v1():
            with set_forward_context(attn_metadata, 
                                     self.vllm_config, 
                                     0,
                                     num_tokens = token_batch_size):
                            # graph_runner.capture(**capture_inputs)
                hidden_states = executable(
                    input_ids,
                    torch.tensor(position_ids, device = self.device, dtype = torch.long),
                )
        else: 
            hidden_states = executable(
                input_ids,
                torch.tensor(position_ids, device = self.device, dtype = torch.long),
                self.cache_manager.kv_caches,
                attn_metadata
            )
        return hidden_states if not use_cuda_graph else hidden_states[:batch_size]
