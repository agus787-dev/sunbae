"""Causal self-attention.

Attention lets every token look back at every earlier token and decide, by
itself, which ones matter for predicting the next one. "Causal" means it can
only look backward, never forward - a token must never see the answer before
it predicts it.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """One multi-head causal self-attention layer.

    Input and output are both shaped (batch, time, d_model) - attention mixes
    information between positions, it never changes the shape.
    """

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.head_dim()

        # One linear layer produces Q, K and V together - cheaper than three
        # separate layers, and mathematically identical.
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # The causal mask: position i may attend to positions <= i, never a
        # later one. Registered as a buffer so it moves with .to(device) but
        # is not treated as a trainable parameter.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "causal_mask", mask.view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.shape  # batch, time (sequence length), channels (d_model)

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)

        # Split d_model into n_head smaller heads so each can specialise.
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product: how much should each position attend to each
        # other position? Scaling by sqrt(head_dim) keeps the scores from
        # growing with head size, which would otherwise make softmax too
        # peaked to learn from.
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Block future positions before the softmax by setting their score to
        # -infinity, so softmax turns them into exactly zero probability.
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float("-inf")
        )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v  # weighted sum of values

        # Merge heads back: (B, n_head, T, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.out_proj(out))