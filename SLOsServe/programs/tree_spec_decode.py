from ..context import RequestContext
from .api import RequestInput, RequestOutput
from .spec_decode_utils import verify
from ..ops import batchable

import torch

@batchable()
def multinomial(
    probs: torch.Tensor,
    num_samples: int
):
    return torch.multinomial(probs, num_samples = num_samples)

def tree_decode(
    model_tag: str, 
    draft_model_tag: str, 
    spec_decode_length: int  
):
    def reference(
        ctx: RequestContext,
        input: RequestInput   
    ):
        tokenized = ctx.tokenize(input.prompt)
        model_output = ctx.forward(
            model_tag = model_tag,
            input_ids = tokenized, 
            use_cache = True, 
            only_sample_last = True
        )
        input_ids = model_output.sampled_ids
        verifier_past_key_values = model_output.past_key_values
        
        draft_output = ctx.forward(
            model_tag = draft_model_tag,
            input_ids = tokenized, 
            use_cache = True, 
            do_sample = False,
        )
        
        rewind_size = None
        is_end, n_generated, generated_text = ctx.decode(model_output.sampled_ids)
        n_spec = 0
        while True: 
            draft_output = ctx.forward(
                model_tag = draft_model_tag,
                input_ids = input_ids, 
                past_key_values = draft_output.past_key_values,
                output_probs = True,
                rewind_size = rewind_size
            )
            draft_outputs = [(draft_output, [], [])]
            
            for _ in range(spec_decode_length - 1):
                new_draft_outputs = []
                for draft_output, prior_ids, prior_probs in draft_outputs:
                    sampled_ids = torch.multinomial(draft_output.probs, num_samples = 2)
                    for sampled_id in sampled_ids:
                        draft_output_new = ctx.forward(model_tag = draft_model_tag,
                                    input_ids = sampled_id, 
                                    use_cache = True, 
                                    past_key_values = draft_output.past_key_values)
                        new_draft_outputs.append((draft_output_new,
                                                  prior_ids + [sampled_id],
                                                  prior_probs + [draft_output.probs]))
                draft_outputs = new_draft_outputs
            
            draft_outputs = [(torch.concat(sampled_ids), torch.concat(probs)) 
                             for _, sampled_ids, probs in draft_outputs]
            
            verified_results = []
            for draft_output, sampled_ids, probs in draft_outputs:
                verifier_output = ctx.forward(
                    model_tag = model_tag,
                    input_ids = torch.concat(sampled_ids),
                    use_cache = True, 
                    past_key_values = verifier_past_key_values,
                    output_probs = True,
                    do_sample = False
                )
                generated, rewind_size = verify(ctx, draft_probs = torch.concat(probs, dim = 0), 
                       verifier_probs = verifier_output.probs)
                verified_results.append((verifier_output, draft_output, generated, rewind_size))
            
            verifier_output, draft_output, generated, rewind_size = min(verified_results, lambda x: x[2])
            
            verifier_past_key_values = verifier_output.past_key_values
            input_ids = generated[-1:]
            
            cur_is_end, cur_n_generated, cur_generated_text = ctx.decode(generated)
            is_end |= cur_is_end 
            n_generated += cur_n_generated 
            generated_text += cur_generated_text 
            n_spec += spec_decode_length
            if (not input.ignore_eos and is_end) or\
                (n_generated >= input.max_new_tokens):
                break 
        return RequestOutput(
            generated_text=generated_text,
            is_end = is_end, 
            n_generated=n_generated,
            n_spec = n_spec
        )
    
    async def impl(
        ctx: RequestContext,
        input: RequestInput   
    ):
        tokenized = ctx.tokenize(input.prompt)
        model_output = ctx.forward(
            model_tag = model_tag,
            input_ids = tokenized, 
            use_cache = True, 
            only_sample_last = True
        )
        input_ids = model_output.sampled_ids
        verifier_past_key_values = model_output.past_key_values
        
        draft_output = ctx.forward(
            model_tag = draft_model_tag,
            input_ids = tokenized, 
            use_cache = True, 
            do_sample = False,
        )
        
        rewind_size = None
        is_end, n_generated, generated_text = ctx.decode(model_output.sampled_ids)
        n_spec = 0
        while True: 
            draft_output = ctx.forward(
                model_tag = draft_model_tag,
                input_ids = input_ids, 
                past_key_values = draft_output.past_key_values,
                output_probs = True,
                rewind_size = rewind_size
            )
            draft_outputs = [(draft_output, [], [])]
            
            for _ in range(spec_decode_length - 1):
                new_draft_outputs = []
                for draft_output, prior_ids, prior_probs in draft_outputs:
                    sampled_ids = multinomial(ctx, 
                                              probs = draft_output.probs, 
                                              num_samples = 2)
                    for i in range(2):
                        sampled_id = sampled_ids[i]
                        draft_output_new = ctx.forward(model_tag = draft_model_tag,
                                    input_ids = sampled_id, 
                                    use_cache = True, 
                                    past_key_values = draft_output.past_key_values)
                        new_draft_outputs.append((draft_output_new,
                                                  prior_ids + [sampled_id],
                                                  prior_probs + [draft_output.probs]))
                draft_outputs = new_draft_outputs
            
            verified_results = []
            for draft_output, sampled_ids, probs in draft_outputs:
                verifier_output = ctx.forward(
                    model_tag = model_tag,
                    input_ids = ctx.concat(sampled_ids, dim = -1),
                    use_cache = True, 
                    past_key_values = verifier_past_key_values,
                    output_probs = True,
                    do_sample = False
                )
                generated, rewind_size = verify(ctx, draft_probs = ctx.concat(probs, dim = 0), 
                       verifier_probs = verifier_output.probs)
                rewind_size_value: int = await ctx.get(rewind_size)
                verified_results.append((verifier_output, draft_output, generated, rewind_size_value))
            
            verifier_output, draft_output, generated, rewind_size = min(verified_results, lambda x: x[-1])
            
            verifier_past_key_values = verifier_output.past_key_values
            input_ids = generated[-1:]
            
            cur_is_end, cur_n_generated, cur_generated_text = await ctx.get(ctx.decode(generated))
            is_end |= cur_is_end 
            n_generated += cur_n_generated 
            generated_text += cur_generated_text 
            n_spec += spec_decode_length
            if (not input.ignore_eos and is_end) or\
                (n_generated >= input.max_new_tokens):
                break 
            
        return RequestOutput(
            generated_text=generated_text,
            is_end = is_end, 
            n_generated=n_generated,
            n_spec = n_spec
        )
    return impl