"""Pick the next token from the model's logits.

    temperature  < 1 = safer, more repetitive; > 1 = more random; 0 = greedy
    top_k        keep only the k most likely tokens (0 = off)
    top_p        keep the smallest set of tokens whose probability adds up
                 to top_p ("nucleus sampling"; 1.0 = off)
"""

import torch
from torch.nn import functional as F


def sample_next_token(logits, temperature=0.8, top_k=50, top_p=0.95, generator=None):
    """logits: 1-D tensor (vocab_size,) for the last position. Returns a token id."""
    if temperature == 0:
        return int(torch.argmax(logits))  # greedy: always the most likely token

    logits = logits / temperature

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        kth_best = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < kth_best, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_ids = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        # Probability of all tokens BEFORE this one. Remove a token if those
        # already reach top_p. The first token is never removed (its "before" is 0).
        before = torch.cumsum(probs, dim=-1) - probs
        sorted_logits[before >= top_p] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_ids, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator))
