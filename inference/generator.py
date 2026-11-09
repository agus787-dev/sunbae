"""Load the exported Sunbae model and generate text.

    from inference.generator import Generator
    gen = Generator()
    gen.generate("Toshkent shahri", max_tokens=50)
"""

import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from inference.sampling import sample_next_token
from src.model import Sunbae
from src.model_config import ModelConfig

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "sunbae.pt"
TOKENIZER_FILE = ROOT / "tokenizer" / "tokenizer.json"


class Generator:
    def __init__(self, model_path=MODEL_FILE, tokenizer_path=TOKENIZER_FILE, device="cpu"):
        self.device = torch.device(device)

        # weights_only=True: the file can only contain tensors and plain values.
        export = torch.load(model_path, map_location="cpu", weights_only=True)
        self.config = ModelConfig(**export["model_config"])
        self.step = export["step"]
        self.val_loss = export["val_loss"]

        self.model = Sunbae(self.config)
        self.model.load_state_dict(export["model_state"])
        # Stored in float16 to keep the file small; compute in float32 on CPU.
        self.model.float().to(self.device)
        self.model.eval()  # turns off dropout

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.eos_id = self.tokenizer.token_to_id("<eos>")

    def num_parameters(self):
        return self.model.num_parameters()

    @torch.no_grad()
    def generate(self, prompt, max_tokens=100, temperature=0.8, top_k=50, top_p=0.95):
        start = time.perf_counter()
        prompt_ids = self.tokenizer.encode(prompt).ids
        ids = list(prompt_ids)
        new_ids = []

        for _ in range(max_tokens):
            # The model only sees the last block_size tokens.
            context = torch.tensor([ids[-self.config.block_size:]], device=self.device)
            logits, _ = self.model(context)
            next_id = sample_next_token(logits[0, -1], temperature, top_k, top_p)
            if next_id == self.eos_id:
                break  # end of document: stop, and do not show <eos>
            ids.append(next_id)
            new_ids.append(next_id)

        completion = self.tokenizer.decode(new_ids)
        return {
            "text": prompt + completion,
            "completion": completion,
            "tokens": len(new_ids),
            "time_ms": round((time.perf_counter() - start) * 1000),
            # Each token decoded alone (a token can be part of a letter, e.g. "ʻ" or Korean).
            "token_strings": [self.tokenizer.decode([i]) for i in prompt_ids + new_ids],
        }
