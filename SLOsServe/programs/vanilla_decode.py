from ..context import RequestContext 
from .api import RequestOutput, RequestInput
from ..ops import OpCode
from ..object import ObjectRef, ObjectStatus

def vanilla_decode_streaming(
    model_tag: str
):
    async def impl(
        ctx: RequestContext,
        req_input: RequestInput
    ):        
         # 1. Prefill 
        input_ids = ctx.tokenize(model_tag, req_input.prompt)
        outputs = ctx.forward(
            model_tag, 
            input_ids,
            use_cache = True,
            do_sample=True,
            only_sample_last=True,
            max_decode_len=req_input.max_new_tokens
        )

        # 2. Decode
        past_key_values = outputs.past_key_values
        input_ids = outputs.sampled_ids
        for i in range(req_input.max_new_tokens):
            is_end, decoded_text = await ctx.get(ctx.decode(model_tag, input_ids))
            yield decoded_text
            if not req_input.ignore_eos and is_end: break 
            decoded_outputs = ctx.forward(
                model_tag, 
                input_ids = input_ids,
                use_cache = True,
                past_key_values = past_key_values,
                do_sample = True,
            )
            past_key_values = decoded_outputs.past_key_values
            input_ids = decoded_outputs.sampled_ids

        return
    return impl

def vanilla_decode_with_spec_step(
    model_tag: str,
    spec_step: int 
):
    async def impl(
        ctx: RequestContext,
        req_input: RequestInput
    ):        
         # 1. Prefill 
        input_ids = ctx.tokenize(model_tag, req_input.prompt)
        outputs = ctx.forward(
            model_tag, 
            input_ids,
            use_cache = True,
            do_sample=True,
            only_sample_last=True,
            max_decode_len=req_input.max_new_tokens,
            customized_tag= (
                OpCode.CausalLMInference,
                '1-prefill'
            )
        )

        # 2. Decode
        past_key_values = outputs.past_key_values
        input_ids = outputs.sampled_ids
        n_generated = 0
        n_interpreted = 1
        text = ''
        is_end = False 
        output_queue = ctx.get_output_queue()
        output_refs: list[ObjectRef] = []

        ctx.get_nowait(ctx.decode(model_tag, outputs.sampled_ids))
        for _ in range(2048 // spec_step):
            outputs = ctx.forward(
                model_tag,
                input_ids = input_ids,
                use_cache = True,
                past_key_values = past_key_values,
                do_sample = True,
                n_iter = spec_step,
                customized_tag= (
                    OpCode.CausalLMInference,
                    '0-decode'
                )
            )
            ctx.get_nowait(ctx.decode(model_tag, outputs.sampled_ids))
            output_refs.append(outputs.sampled_ids)
            past_key_values = outputs.past_key_values
            input_ids = outputs.sampled_ids[-1:]
            n_interpreted += spec_step
        
        n_alg_spec = 0
        while not ((not req_input.ignore_eos and is_end) 
                   or n_generated >= req_input.max_new_tokens):
            (cur_is_end, cur_n_generated, cur_text), _, _ = await output_queue.get()
            n_generated += cur_n_generated 
            text += cur_text
            is_end = is_end or cur_is_end
            n_alg_spec += spec_step
        
        n_spec = 1
        for ref in output_refs:
            if ref.status == ObjectStatus.SCHEDULED:
                n_spec += spec_step
        return RequestOutput(
            generated_text=text, 
            is_end = is_end,
            n_generated=n_generated,
            n_spec=n_spec,
            acc_rate = round(n_generated / n_spec, 2),
            n_interpreted=n_interpreted,
            n_alg_spec=n_alg_spec
        )
    return impl

def vanilla_decode(
    model_tag: str
):
    async def impl(
        ctx: RequestContext,
        req_input: RequestInput
    ):        
        # 1. Prefill 
        input_ids = ctx.tokenize(model_tag, req_input.prompt)
        outputs = ctx.forward(
            model_tag, 
            input_ids,
            use_cache = True,
            do_sample=True,
            only_sample_last=True,
            max_decode_len=req_input.max_new_tokens,
            customized_tag=(
                OpCode.CausalLMInference,
                model_tag 
            )
        )

        # 2. Decode
        past_key_values = outputs.past_key_values
        input_ids = outputs.sampled_ids
        generated = [input_ids]
        for i in range(req_input.max_new_tokens):
            decoded_outputs = ctx.forward(
                model_tag, 
                input_ids = input_ids,
                use_cache = True,
                past_key_values = past_key_values,
                do_sample = True,
                customized_tag=(
                    OpCode.CausalLMInference,
                    model_tag 
                )
            )
            past_key_values = decoded_outputs.past_key_values
            input_ids = decoded_outputs.sampled_ids
            generated.append(input_ids)

        (is_end, n_generated, text), _, _ = await ctx.get(ctx.decode(model_tag, ctx.concat(generated)))
        return RequestOutput(
            generated_text=text,
            is_end = is_end,
            n_generated=n_generated,
            n_spec = n_generated, 
            acc_rate = 1,
            n_interpreted=n_generated,
            n_alg_spec=n_generated 
        )
    return impl