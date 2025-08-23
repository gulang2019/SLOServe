import torch
import torch.nn as nn
from typing import List, Tuple

from ..context import RequestContext
from .api import RequestInput, RequestOutput
from ..ops import batchable 

@batchable(
    shape_inference=lambda _, __, beam_size: [(beam_size,), (beam_size,), (beam_size,)]
)
def sample_topk(
    probs: torch.Tensor, # [1 or BeamSize, vocab_size]
    scores: torch.Tensor, # [1 or BeamSize]
    beam_size: int       # 
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if probs.ndim == 1: 
        probs = probs.unsqueeze(0) 
    scores = scores.to(probs.device)
    sampled_ids = torch.multinomial(probs, num_samples = beam_size) # [1 or BeamSize, BeamSize]
    sampled_probs = probs[torch.arange(probs.size(0)).unsqueeze(-1), sampled_ids] * scores.unsqueeze(-1) # [1 or BeamSize, BeamSize]
    topk_probs, topk_indices = torch.topk(sampled_probs.view(-1), k = beam_size, dim = -1) # [1 or BeamSize, k], [1 or BeamSize, k]
    topk_sampled_ids = sampled_ids.view(-1)[topk_indices]
    return topk_probs, topk_indices, topk_sampled_ids # [BeamSize], [BeamSize]

@batchable(
    shape_inference=lambda _, shape: [shape]
)
def constant(
    value: float, 
    shape: tuple
) -> torch.Tensor:
    return torch.ones(size = shape, dtype = torch.float16) * value

def beam_search_decode(
    model_tag: str,
    beam_size: int = 5
):
    def reference(ctx: RequestContext,
                input: RequestInput):
        
        tokenized = ctx.tokenize(input.prompt)
        
        output = ctx.forward(
            model_tag = model_tag, 
            input_ids = tokenized, 
            use_cache = True, 
            only_sample_last=True, 
            output_probs = True
        )
        
        outputs = [(output, 1.0, [])]
        for _ in range(input.max_new_tokens):
            sampled_idss = []
            sampled_probss = []
            for output, score, _ in outputs:
                sampled_ids = torch.multinomial(output.probs, num_samples = beam_size) # []
                sampled_probs = output.probs[sampled_ids] * score
                sampled_idss.append(sampled_ids)
                sampled_probss.append(sampled_probs)
            sampled_probs = torch.concat(sampled_probss)
            sampled_ids = torch.concat(sampled_idss)
            probs, indices = torch.topk(sampled_probs, num_samples = beam_size, dim = -1)
            
            new_outputs = []
            for i in range(beam_size):
                index = indices[i]
                prob = probs[i]
                output, _, generated_ids = outputs[index // beam_size]
                sampled_ids = sampled_ids[index % beam_size]
                new_output = ctx.forward(
                    model_tag, 
                    input_ids = sampled_ids,
                    past_key_values = output.past_key_values,
                    use_cache = True,
                    output_probs = True, 
                )
                new_outputs.append((new_output, prob, generated_ids + [sampled_ids]))
            
            outputs = new_outputs
        
        best_output = max(outputs, lambda x: x[1])[0]
        
        return best_output
    
    async def impl(ctx: RequestContext,
                input: RequestInput):
        
        tokenized = ctx.tokenize(model_tag, input.prompt)
        
        output = ctx.forward(
            model_tag = model_tag, 
            input_ids = tokenized, 
            use_cache = True, 
            only_sample_last=True, 
            output_probs = True
        )
        
        outputs = [(output, [])]
        probs = output.probs #[vocab_size]
        scores = constant(ctx, value = 1.0, shape = (1,1)) # [1,]
        
        for _ in range(input.max_new_tokens):
            '''
            probs       (beam_size, vocab_size) or (vocab_size) 
            scores      (beam_size,) or (1)
            indices     (beam_size,),
            sampled_ids (beam_size,)
            '''
            scores, indices, sampled_ids = sample_topk(ctx, 
                                                       probs = probs, 
                                                       scores = scores, 
                                                       beam_size = beam_size)
            indices = await ctx.get(indices)
            
            new_outputs = []
            for i, index in enumerate(indices.tolist()):
                output, generated = outputs[index // beam_size]
                sampled_id = sampled_ids[i]
                new_output = ctx.forward(
                    model_tag,
                    input_ids = sampled_id,
                    past_key_values = output.past_key_values,
                    use_cache = True,
                    output_probs = True, 
                )
                outputs.append((new_output, generated + [sampled_id]))
                
            outputs = new_outputs
        
        best_output = outputs[0]
        
        is_end, n_generated, generated_text = await ctx.get(ctx.decode(model_tag, 
                                                                       ctx.concat(best_output[1])))
        
        return RequestOutput(
            generated_text = generated_text, 
            is_end = is_end, 
            n_generated = n_generated,
            n_spec = n_generated
        )
    return impl


# Create an initial input tensor (use torch.long for input_ids)
# initial_input = torch.randint(0, 5000, (1, 128), dtype=torch.long)  # Assuming vocab_size = 5000

# # Perform beam search
# result = beam_search(model, initial_input)

# print("Best sequence found:", result)

# Example usage:
# Assuming `model` is your sequence model and `initial_input` is the starting token sequence.
# The model should output log probabilities (after softmax or log_softmax) over the vocabulary.
# beam_search = BeamSearch(model, beam_size=5)
# result = beam_search.search(initial_input)
