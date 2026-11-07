"""feed-forward network (FFN).

Attention lets tokens exchange information with each other. The feed-forward
network is where the model actually processes what it just gathered - it
runs on each position independently, with no mixing between positions at all.
"""

import torch.nn as nn


class FeedForward(nn.Module):
    """Two linear layers with a GELU activation in between.

    d_model -> mlp_ratio * d_model -> d_model. The hidden layer is wider than the
    input so the network has room to combine features before shrinking back
    down to d_model for the next block.
    """

    def __init__(self, config):
        super().__init__()
        hidden = config.mlp_ratio * config.d_model

        self.fc_in = nn.Linear(config.d_model, hidden)
        self.activation = nn.GELU()
        self.fc_out = nn.Linear(hidden, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.activation(x)
        x = self.fc_out(x)
        return self.dropout(x)