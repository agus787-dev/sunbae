"""The design of the Sunbae model.

There is no model in this file. There are only the DECISIONS, written down,
plus enough arithmetic to find out whether those decisions are affordable.

    Architecture
          |
    Parameter count
          |
    Memory requirement
          |
    Training feasibility

A model that does not fit on the Mac GPU is not a design, it is a wish. The
cheap way to find out is to multiply the numbers before training.

To print the design report, run scripts/design_model.py.
"""

from dataclasses import asdict, dataclass, fields

import yaml


@dataclass
class ModelConfig:
    """The eight numbers that define the model. Values come from configs/run_XX.yaml."""

    # How many different tokens exist. Comes from the tokenizer, not from
    # taste: it must match tokenizer/tokenizer.json.
    vocab_size: int = 8000

    # Context length: how many tokens the model can look back at.
    # Attention cost grows with the SQUARE of this number.
    block_size: int = 256

    # Embedding dimension, the "width" of the model. Parameter count grows
    # with its square, so this is the most expensive dial.
    d_model: int = 384

    # How many transformer blocks are stacked, the "depth" of the model.
    n_layer: int = 6

    # Attention heads per block. d_model must divide evenly by n_head.
    n_head: int = 6

    # Randomly zeroes activations during training to fight overfitting.
    dropout: float = 0.1

    # The feed-forward layer is this many times wider than d_model.
    mlp_ratio: int = 4

    # Reuse the embedding table as the output layer. Saves vocab_size * d_model params.
    tie_weights: bool = True

    def head_dim(self):
        """Size of one attention head."""
        return self.d_model // self.n_head

    def check(self):
        """Catch the mistakes that would break the model on line one."""
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must divide evenly by n_head ({self.n_head})"
            )
        if self.vocab_size > 65536:
            raise ValueError("vocab_size above 65536 does not fit in uint16 token ids")

    def estimate_parameters(self):
        """A rough parameter count. Deliberately rough.

        Per transformer block (with mlp_ratio = 4):
            attention  Q, K, V and output projection  ->  4 * d_model^2
            feed-forward  d_model -> 4*d_model -> d_model  ->  8 * d_model^2

        so the stack is about (4 + 2 * mlp_ratio) * n_layer * d_model^2.
        Biases and LayerNorms add a few thousand more; ignored here.
        """
        per_block = (4 + 2 * self.mlp_ratio) * self.d_model ** 2
        non_embedding = self.n_layer * per_block
        embedding = self.vocab_size * self.d_model
        position = self.block_size * self.d_model  # learned position embeddings

        total = non_embedding + embedding + position
        if not self.tie_weights:
            total += self.vocab_size * self.d_model  # a separate output layer

        return {
            "non_embedding": non_embedding,
            "embedding": embedding,
            "position": position,
            "total": total,
        }

    def estimate_training_memory_mb(self):
        """Very rough memory for the weights during training.

        Each parameter is stored four times: the weight, its gradient, and the
        two running averages AdamW keeps. At 4 bytes each: 16 bytes per param.

        Activations come on top and depend on batch size, so this is a floor,
        not a budget.
        """
        return self.estimate_parameters()["total"] * 16 / (1024 ** 2)

    def to_dict(self):
        return asdict(self)


def load_config(path):
    """Read a ModelConfig from the `model:` section of a run YAML.

    Unknown keys are ignored, so training settings can live in the same file
    and one filename still names the whole experiment.
    """
    with open(path, "r", encoding="utf-8") as f:
        values = yaml.safe_load(f)["model"]

    known = {f.name for f in fields(ModelConfig)}
    config = ModelConfig(**{k: v for k, v in values.items() if k in known})
    config.check()
    return config

