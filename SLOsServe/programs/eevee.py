from ..context import RequestContext
from .api import RequestInput, RequestOutput
from ..ops import *

from .spec_decode_utils import verify

def eevee_spec_decode(
    model_tag: str,
    draft_model_layers: int,
    spec_decode_length: int, 
    temperature: float
):
    async def eevee_spec_impl(
        ctx: RequestContext,
        req_input: RequestInput
    ):
        tokenized = ctx.tokenize(model_tag, 
                                 req_input.prompt)
        
        outputs = ctx.forward(
            model_tag, 
            input_ids = tokenized,
            only_sample_last=True,
            use_cache=True,
            do_sample=True,
        )

        input_ids = outputs.sampled_ids
        rewind_size = None 
        past_key_values = outputs.past_key_values
        
        
        (is_end, n_generated, text), _, _ = ctx.get(ctx.decode(model_tag, generated_tokens))
        n_spec = n_generated
        
        for i in range(req_input.max_new_tokens):
            drafter_output: CausalLMInferenceOp.Output = ctx.forward(
                model_tag, 
                input_ids = input_ids,
                layer_range = slice(0, draft_model_layers),
                rewind_size = rewind_size,
                n_iter = spec_decode_length, 
                use_cache = True,
                past_key_values=past_key_values, 
                output_probs = True, 
                do_sample = False, 
                output_last_hidden_state=True
            )
            
            verifier_output: CausalLMInferenceOp.Output = ctx.forward(
                model_tag, 
                inputs_embeds= drafter_output.last_hidden_state,
                layer_range = slice(draft_model_layers, None),
                use_cache = True, 
                past_key_values = past_key_values, 
                output_probs = True, 
                do_sample = False
            )

            generated_tokens, rewind_size = verify(
                ctx, 
                draft_logits = drafter_output.logits,
                verifier_logits = verifier_output.logits,
                temperature = temperature,
            )
            
            (cur_is_end, cur_n_generated, cur_generated_text), _, _ = ctx.get(ctx.decode(model_tag, generated_tokens))
            text += cur_generated_text
            is_end |= cur_is_end
            n_spec += spec_decode_length
            n_generated += cur_n_generated
            
            if ((not req_input.ignore_eos) and is_end) or \
                n_generated >= req_input.max_new_tokens:
                break
        
        return RequestOutput(
            text,
            is_end, 
            n_generated,
            n_spec 
        )      

    return eevee_spec_impl