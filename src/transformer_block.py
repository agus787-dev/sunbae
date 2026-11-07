"""one Transformer block.

A block is attention followed by a feed-forward network, each wrapped in a
residual connection, with a LayerNorm applied BEFORE the sub-layer (pre-norm).
Pre-norm is what keeps a deep stack of blocks stable to train - it is the
standard choice since GPT-2.

    x -> LayerNorm -> Attention    -> (+x)
      -> LayerNorm -> FeedForward  -> (+x)
"""

import torch.nn as nn

from src.attention import CausalSelfAttention
from src.mlp import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x):
        # Residual connections: the "+x" lets gradients flow straight through
        # every block, all the way back to the first layer, even in a deep
        # stack. Without them, a 6-layer stack is hard to train at all.
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x