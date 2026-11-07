"""The Sunbae language model.

Token + position embeddings -> a stack of TransformerBlocks -> final
LayerNorm -> lm_head, which scores every token in the vocabulary as the
next one. Sizes come from ModelConfig (configs/run_XX.yaml).
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.transformer_block import TransformerBlock


class Sunbae(nn.Module):
    """A small GPT-style, decoder-only Transformer for Uzbek text."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.ln_f = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            # Reuse the embedding table as the output layer - input tokens
            # and output predictions live in the same "meaning space", so
            # sharing the weights is free quality and fewer parameters to
            # train (see ModelConfig.estimate_parameters).
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Small random weights at the start, per the GPT-2 recipe."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def configure_optimizers(self, weight_decay, lr, betas):
        """AdamW with weight decay applied only to matrices.

        A bias or a LayerNorm gain/shift is a single number per output
        channel, not a direction in input space - decaying it toward zero
        does not fight overfitting the way it does for a weight matrix, it
        just fights the norm's ability to rescale. So every tensor with 2+
        dimensions gets weight decay; every 1-D tensor (biases, norm params)
        does not.
        """
        decay, no_decay = [], []
        for param in self.parameters():
            if not param.requires_grad:
                continue
            (decay if param.dim() >= 2 else no_decay).append(param)

        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)

    def num_parameters(self):
        # self.parameters() yields each tensor once, so when weights are tied
        # the shared embedding / lm_head matrix is already counted only once.
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, "sequence longer than block_size"

        positions = torch.arange(T, device=idx.device)

        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Score every position at once: flatten (B, T, vocab) -> (B*T,
            # vocab) and (B, T) -> (B*T,), then one cross-entropy call scores
            # all B*T predictions together.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Greedy generation: always pick the single most likely next token.

        Sampling (temperature, top-k, top-p) comes later in Sunbae - this is
        deliberately the simplest possible version, just to prove the loop
        (encode -> forward -> pick token -> append -> repeat) works end to
        end.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            next_logits = logits[:, -1, :]  # only the last position matters
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)
        return idx