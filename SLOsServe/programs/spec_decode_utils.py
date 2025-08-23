import torch
from typing import List, Tuple
from SLOsServe.ops import batchable

def batch_verify(
    draft_probs: torch.Tensor, # [P x1,..., P x_gamma] (Batch_Size, SeqLen, vocab_size)
    verifier_probs: torch.Tensor, # [P x1', ..., P x_gamma'] (Batch_Size, seqlen, vocab_size)
):
    # gamma = draft_tokens.size(0)
    bs, gamma = draft_tokens.size()
    draft_tokens = torch.multinomial(draft_probs.view(-1, draft_probs.size(-1)), num_samples = 1).view(bs, -1)
    # draft_p = draft_probs[torch.arange(gamma), draft_tokens]
    draft_p = draft_probs[torch.arange(bs).unsqueeze(1), torch.arange(gamma).unsqueeze(0), draft_tokens]
    verifier_p = verifier_probs[torch.arange(bs).unsqueeze(1), torch.arange(gamma).unsqueeze(0), draft_tokens]
    
    ratio = draft_p / verifier_p
    r_p = torch.rand_like(ratio)
    is_accepted = r_p <= ratio
    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum(dim = -1)  # 0 - n_matches

    '''
    if n_matches < gamma:
        prob = torch.clamp(verifier_probs[n_matches] - draft_probs[n_matches], min = 0)
        t = torch.multinomial(prob, num_samples=1)
        return torch.concat((draft_tokens[:n_matches], t)), n_matches
    return draft_tokens, n_matches
    '''
    results: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(bs):
        n_match = n_matches[i]
        if n_match < gamma:
            prob = torch.clamp(verifier_probs[i, n_match] - draft_probs[i, n_match], min = 0)
            t = torch.multinomial(prob, num_samples=1)
            tokens = torch.concat((draft_tokens[i, :n_match], t))
            results.append((tokens, gamma - n_match.item()))
        else: 
            results.append((draft_tokens[i], 0))
    return results

@batchable(
    shape_inference = lambda _, __, ___: ((None,), None),
    batched_impl = batch_verify
)
def verify(
    draft_probs: torch.Tensor, # [P x1,..., P x_gamma] (SeqLen, vocab_size)
    verifier_probs: torch.Tensor, # [P x1', ..., P x_gamma'] (seqlen, vocab_size)
) -> Tuple[torch.Tensor, int]:
    '''
    verify
    '''
    gamma = draft_probs.size(0)
    draft_tokens = torch.multinomial(draft_probs, num_samples = 1).squeeze(-1)
    draft_p = draft_probs[torch.arange(gamma), draft_tokens]
    verifier_p = verifier_probs[torch.arange(gamma), draft_tokens]
    
    
    ratio = draft_p / verifier_p 
    r_p = torch.rand_like(ratio)
    is_accepted = r_p <= ratio 
    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum(dim = -1)  # 0 - n_matches
    
    if n_matches < gamma:
        prob = torch.clamp(verifier_probs[n_matches] - draft_probs[n_matches], min = 0)
        t = torch.multinomial(prob, num_samples=1)
        generated = torch.concat((draft_tokens[:n_matches], t))
    else: generated = draft_tokens
    
    rewind_sizes = gamma - n_matches.item()
    return generated,  rewind_sizes




if __name__ == '__main__':
    candidate_length = 5
    vocab_size = 256

    draft_logits = torch.randn(size=(candidate_length, vocab_size))

    verifier_logits = torch.randn(size=(candidate_length, vocab_size))

    tokens, rewind_sizes = verify.forward_impl(
        draft_logits, 
        verifier_logits,
        1.0
    )
    print('tokens', tokens.size())
    print('rewind_sizes', rewind_sizes)

    func = torch.jit.script(verify.forward_impl)
    # print('func', func.graph)

    tokens_jit, rewind_sizes_jit = func(
        draft_logits, 
        verifier_logits,
        1.0)
    print('tokens_jit', tokens_jit.size())
    print('rewind_sizes_jit', rewind_sizes_jit)
    
    batch_size = 8

    draft_logits_batch = torch.randn(size=(batch_size, candidate_length, vocab_size))

    verifier_logits_batch = torch.randn(size=(batch_size, candidate_length, vocab_size))

    batch_verify(draft_logits_batch, verifier_logits_batch, [1.0] * batch_size)

    batch_func = torch.jit.script(batch_verify)

    results = batch_func(draft_logits_batch, verifier_logits_batch, [1.0] * batch_size)
    for i, (tokens, rewind_sizes) in enumerate(results):
        assert not torch.any(tokens >= vocab_size) and not torch.any(tokens < 0)
        print('tokens_jit', tokens.size())
        print('rewind_sizes_jit', rewind_sizes)
