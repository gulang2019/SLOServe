from typing import Callable

from .api import RequestInput, RequestOutput
from .spec_decode import spec_decode, spec_decode_with_spec_step
from .vanilla_decode import vanilla_decode, vanilla_decode_streaming, vanilla_decode_with_spec_step
from ..context import RequestContext
from .beam_search import beam_search_decode

def get_program(
    args
) -> tuple[Callable[[RequestContext, RequestInput], RequestOutput], 
           Callable[[RequestContext, RequestInput], RequestOutput]]:
    def _placeholder(
        ctx: RequestContext,
        req_input: RequestInput 
    ) -> RequestOutput:
        raise NotImplementedError
    assert args.model is not None 
    if args.program == 'spec_decode':
        assert args.draft_model is not None
        return spec_decode_with_spec_step(
            args.model, 
            args.draft_model, 
            args.draft_decode_length,
            args.batch_strategy ,
            args.spec_step) if args.spec_step != 0 else \
            spec_decode(
                args.model, 
                args.draft_model, 
                args.draft_decode_length,
                args.batch_strategy 
            ), _placeholder
    elif args.program == 'vanilla_decode':
        return vanilla_decode_with_spec_step(args.model, args.spec_step)\
            if args.spec_step != 0 else vanilla_decode(args.model), vanilla_decode_streaming(args.model)
    elif args.program == 'beam_search':
        return beam_search_decode(args.model, beam_size = 5), None
    else: raise NotImplementedError

