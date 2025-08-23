import time
from dataclasses import dataclass 
from typing import Optional, Callable, Union, List
import numpy as np

from ..context import RequestContext 
from .api import RequestInput, RequestOutput
from ..ops import OpCode
import asyncio
from ..object import ObjectRef, ObjectStatus
from .spec_decode_utils import verify

@dataclass 
class SchStrategy: 
    verifier_prefill_tag: str 
    drafter_prefill_tag: str 
    drafter_decode_tag: Union[Callable, str]
    verifier_decode_tag: str 
        
@dataclass
class Strategy: 
    prefill_decode_strategy: str = 'batched'
    is_drafter_first: bool = True 
    align_drafter: bool = True
    def get_strategy(self) -> SchStrategy:
        prefill_tag, decode_tag = (0,0) if self.prefill_decode_strategy == 'batched' else (1,0)
        verifier_prefill_tag = f'{prefill_tag}-{1-self.is_drafter_first}-verifier'
        verifier_decode_tag = f'{decode_tag}-{1-self.is_drafter_first}-verifier'
        drafter_prefill_tag = f'{prefill_tag}-{self.is_drafter_first}-0-drafter'
        if self.align_drafter: 
            drafter_decode_tag = f'{decode_tag}-{self.is_drafter_first}-1-drafter'
        else:
            drafter_decode_tag = lambda i: f'{decode_tag}-{self.is_drafter_first}-{i}-drafter'
        return SchStrategy(
            verifier_prefill_tag,
            drafter_prefill_tag,
            drafter_decode_tag,
            verifier_decode_tag 
        )

def spec_decode_with_spec_step(
    model_tag: str,
    draft_model_tag: str, 
    draft_decode_length: int = 5,
    strategy_dict: dict = {},
    num_spec_step: int = 1,
):
    if strategy_dict is None: 
        strategy_dict = {}
    # align_drafter = strategy_dict.pop('align_drafter', True)
    do_speculate = strategy_dict.pop('speculate', False)
    strategy = Strategy(**strategy_dict) if isinstance(strategy_dict, dict) else Strategy()
    print(f'Choose {strategy} for spec decode, Do Speculate: {do_speculate}')
    _sch_strategy = strategy.get_strategy()
    cases = sorted([('verifier_prefill', _sch_strategy.verifier_prefill_tag), 
             ('verifier_deocode', _sch_strategy.verifier_decode_tag),
             ('drafter_prefill', _sch_strategy.drafter_prefill_tag)] + \
             ([(f'drafter_decode-{i}', _sch_strategy.drafter_decode_tag(i)) for i in range(draft_decode_length)] 
              if not strategy.align_drafter else [('drafter_decode', _sch_strategy.drafter_decode_tag)]),
             key = lambda x: x[1])
    print(f'batch by: {cases[0]}')
    for (name, tag), (next_name, next_tag) in zip(cases[:-1], cases[1:]):
        if tag == next_tag:
            print('=')
        else: print('<')
        print(next_name, next_tag)

    print(f'spec decode with spec step: {num_spec_step}, draft_decode_length: {draft_decode_length}')
    async def spec_infer_impl(
        ctx: RequestContext,
        req_input: RequestInput
    ) -> RequestOutput:
        # 1. Prefill
        start_time = time.perf_counter()
        tokenized = ctx.tokenize(model_tag, req_input.prompt)

        outputs = ctx.forward(
            model_tag = model_tag, 
            input_ids = tokenized, 
            use_cache = True, 
            do_sample = True, 
            only_sample_last = True,
            max_decode_len=req_input.max_new_tokens,
            customized_tag=(
                OpCode.CausalLMInference, 
                _sch_strategy.verifier_prefill_tag # '2-verifier-prefill',
            ))

        verifier_past_key_values = outputs.past_key_values

        draft_past_key_values = ctx.forward(
            model_tag = draft_model_tag, 
            input_ids = tokenized,
            use_cache = True,
            do_sample = False,
            max_decode_len=req_input.max_new_tokens,
            customized_tag=(
                OpCode.CausalLMInference, 
                _sch_strategy.drafter_prefill_tag, #'3-draft-prefill',
            )
        ).past_key_values

        get_time = 0

        # 2. Decode
        input_ids = outputs.sampled_ids
        rewind_size = None
        is_end = False
        generated_text = ''
        n_generated = 0
        # num_spec_step = 1
        n_interpreted = 0
        n_spec = 0
        

        tot_spec_step = (512 + (num_spec_step * draft_decode_length)) // (num_spec_step * draft_decode_length)
        output_refs: List[ObjectRef] = []
        for _ in range(tot_spec_step):
            generated = []
            for _ in range(num_spec_step):
                assert input_ids.shape == (1,)
                if strategy.align_drafter:
                    drafter_outputs = ctx.forward(
                        model_tag = draft_model_tag,
                        input_ids = input_ids,
                        past_key_values = draft_past_key_values,
                        use_cache = True, 
                        do_sample = True,
                        rewind_size = rewind_size,
                        n_iter = draft_decode_length,
                        customized_tag=(
                            OpCode.CausalLMInference, 
                            _sch_strategy.drafter_decode_tag,
                        )
                    )
                    all_sampled_ids = ctx.concat([input_ids, drafter_outputs.sampled_ids], dim = -1)

                else:
                    local_generated = [input_ids]
                    for j in range(draft_decode_length):
                        drafter_output = ctx.forward(
                            model_tag = draft_model_tag,
                            input_ids = input_ids,
                            past_key_values = draft_past_key_values,
                            use_cache = True, 
                            do_sample = True,
                            rewind_size = rewind_size if j == 0 else None,
                            customized_tag=(
                                OpCode.CausalLMInference, 
                                _sch_strategy.drafter_decode_tag(j),
                            )
                        )
                        draft_past_key_values = drafter_output.past_key_values
                        input_ids = drafter_output.sampled_ids
                        local_generated.append(input_ids)
                    all_sampled_ids = ctx.concat(local_generated, dim = -1)

                outputs = ctx.forward(
                    model_tag = model_tag, 
                    input_ids = all_sampled_ids[:-1],
                    past_key_values = verifier_past_key_values,
                    use_cache = True, 
                    do_sample = True,
                    rewind_size = rewind_size,
                    customized_tag=(
                        OpCode.CausalLMInference, 
                        _sch_strategy.verifier_decode_tag, #'3-draft-prefill',
                    )
                )

                verifier_past_key_values = outputs.past_key_values
                assert outputs.sampled_ids.shape == (draft_decode_length,)
                
                verifier_output = ctx.verify(
                    guessed_tokens = all_sampled_ids[1:],
                    true_tokens = outputs.sampled_ids
                )
                '''
                2. speedup the decoding kernel by cuda graph;
                3. reduce the get latency
                1. remove unnecessary compute in the backend;
                '''

                output_refs.append(outputs.sampled_ids)

                input_ids = verifier_output.tokens[-1:]
                rewind_size = verifier_output.rewind_size
                generated.append(verifier_output.tokens)
                n_interpreted += num_spec_step * draft_decode_length

            # cur_is_end, cur_n_generated, cur_generated_text = await ctx.get(ctx.decode(model_tag, ctx.concat(generated)))
            # print('finish spec!')
            decoded = ctx.decode(model_tag, ctx.concat(generated))
            if do_speculate:
                ctx.get_nowait(decoded)
            else: 
                (local_is_end, local_n_generated, local_generated_text), _, _ = await ctx.get(decoded)
                is_end = is_end or local_is_end
                n_generated += local_n_generated
                generated_text += local_generated_text
                n_spec += draft_decode_length * num_spec_step
                if((not req_input.ignore_eos and is_end) \
                    or (n_generated >= req_input.max_new_tokens)):
                    break

        issue_time = time.perf_counter() - start_time
        if not do_speculate:
            return RequestOutput(
                generated_text=generated_text,
                is_end = is_end, 
                n_generated=n_generated, 
                n_spec = n_spec, 
                acc_rate = round(n_generated / n_spec, 2),
                n_interpreted=n_spec, 
                issue_time = issue_time,
                n_alg_spec = n_spec, 
            )

        # We collect the output here.
        executor_get_delays = []
        frontend_get_delays = []
        
        n_get = 0
        n_alg_spec = 0
        output_queue = ctx.get_output_queue()
        for _ in range(tot_spec_step):
            (cur_is_end, cur_n_generated, cur_generated_text), got_time, executor_got_time = await output_queue.get()
            executor_get_delays.append(executor_got_time - got_time)
            frontend_get_delays.append(time.perf_counter() - executor_got_time)
            n_generated += cur_n_generated
            n_alg_spec += num_spec_step * draft_decode_length
            generated_text += cur_generated_text
            # Exit when <EOS> is generated or the generated tokens match the tot tokens
            is_end = is_end or cur_is_end
            n_get += 1
            if((not req_input.ignore_eos and is_end) \
                or (n_generated >= req_input.max_new_tokens)):
                break

        for output_ref in output_refs:
            if output_ref.status == ObjectStatus.SCHEDULED:
                n_spec += draft_decode_length

        acc_rate = round(n_generated / n_spec, 2)
        return RequestOutput(
            generated_text=generated_text,
            is_end = is_end,
            n_generated=n_generated,
            n_spec=n_spec,
            acc_rate = acc_rate,
            get_time = get_time,
            n_interpreted=n_interpreted,
            issue_time=issue_time,
            executor_get_delay=np.mean(executor_get_delays),
            frontend_get_delay=np.mean(frontend_get_delays),
            n_get = n_get,
            n_alg_spec = n_alg_spec
        )
    return spec_infer_impl

def spec_decode(
    model_tag: str,
    draft_model_tag: str, 
    draft_decode_length: int = 4,
    strategy_dict: dict = {},
):
    strategy = Strategy(**strategy_dict) if isinstance(strategy_dict, dict) else Strategy()
    print(f'Choose {strategy} for spec decode')
    _sch_strategy = strategy.get_strategy()
    cases = sorted([('verifier_prefill', _sch_strategy.verifier_prefill_tag), 
             ('verifier_deocode', _sch_strategy.verifier_decode_tag),
             ('drafter_prefill', _sch_strategy.drafter_prefill_tag)] + \
             ([(f'drafter_decode-{i}', _sch_strategy.drafter_decode_tag(i)) for i in range(draft_decode_length)] 
              if not strategy.align_drafter else [('drafter_decode', _sch_strategy.drafter_decode_tag)]),
             key = lambda x: x[1])
    print(f'batch by: {cases[0]}')
    for (name, tag), (next_name, next_tag) in zip(cases[:-1], cases[1:]):
        if tag == next_tag:
            print('=')
        else: print('<')
        print(next_name, next_tag)


    async def spec_infer_impl(
        ctx: RequestContext,
        req_input: RequestInput
    ):
        # 1. Prefill
        tokenized = ctx.tokenize(model_tag, req_input.prompt)

        outputs = ctx.forward(
            model_tag = model_tag, 
            input_ids = tokenized, 
            use_cache = True, 
            do_sample = True, 
            only_sample_last = True,
            max_decode_len=req_input.max_new_tokens,
            customized_tag=(
                OpCode.CausalLMInference, 
                _sch_strategy.verifier_prefill_tag # '2-verifier-prefill',
            ))

        verifier_past_key_values = outputs.past_key_values

        draft_past_key_values = ctx.forward(
            model_tag = draft_model_tag, 
            input_ids = tokenized,
            use_cache = True,
            do_sample = False,
            max_decode_len=req_input.max_new_tokens,
            customized_tag=(
                OpCode.CausalLMInference, 
                _sch_strategy.drafter_prefill_tag, #'3-draft-prefill',
            )
        ).past_key_values

        # 2. Decode
        input_ids = outputs.sampled_ids
        (is_end, n_generated, generated_text), _, _ =\
            await ctx.get(ctx.decode(model_tag, input_ids))
        rewind_size = None
        n_spec = 0
        n_get = 0
        while True:
            assert input_ids.shape == (1,)

            drafter_outputs = ctx.forward(
                model_tag = draft_model_tag,
                input_ids = input_ids,
                past_key_values = draft_past_key_values,
                use_cache = True, 
                output_probs = True,
                do_sample = True,
                rewind_size = rewind_size,
                n_iter = draft_decode_length,
                customized_tag=(
                    OpCode.CausalLMInference, 
                    _sch_strategy.drafter_decode_tag,
                )
            )

            all_sampled_ids = ctx.concat([input_ids, drafter_outputs.sampled_ids], dim = -1)

            verifier_outputs = ctx.forward(
                model_tag = model_tag, 
                input_ids = all_sampled_ids[:-1],
                past_key_values = verifier_past_key_values,
                use_cache = True,
                output_probs = True,
                do_sample = False,
                rewind_size = rewind_size,
                customized_tag=(
                    OpCode.CausalLMInference,
                    _sch_strategy.verifier_decode_tag
                )
            )
            
            generated_tokens, rewind_size = verify(ctx,
                   draft_logits = drafter_outputs.logits,
                   verifier_logits = verifier_outputs.logits,
                   temperature = req_input.temperature)

            # get_start = time.perf_counter()
            (cur_is_end, cur_n_generated, cur_generated_text), _, _ = await ctx.get(ctx.decode(model_tag, generated_tokens))
            input_ids = generated_tokens[-1:]
            n_spec += draft_decode_length
            n_get += 1
            is_end = is_end or cur_is_end 
            n_generated += cur_n_generated 
            generated_text += cur_generated_text
            
            if (not req_input.ignore_eos and is_end) or\
                n_generated >= req_input.max_new_tokens:
                break
            
        return RequestOutput(
            generated_text=generated_text, 
            is_end = is_end, 
            n_generated = n_generated,
            acc_rate = round(n_generated / n_spec, 2),
            n_spec = n_spec,
            n_get = n_get
        )

    return spec_infer_impl