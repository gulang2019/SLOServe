from dataclasses import dataclass
from functools import lru_cache

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
        
@lru_cache(maxsize=None)
def get_model_config(
    model_tag: str 
) -> ModelConfig:
    from transformers import AutoConfig, OPTConfig, GPT2Config, Qwen2Config
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
    else: raise RuntimeError(f'unknown model {model_tag}')
    
