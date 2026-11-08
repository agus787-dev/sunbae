"""Turn the final mixture into the files the training loop reads.

    mixture.jsonl  ->  split per document  ->  encode  ->  train.bin / val.bin / meta.json

Based on the teacher's dataset_builder.py, with three ideas:

1. SPLIT BEFORE ENCODING, at document level.
   Each document goes fully to train or fully to val, so val loss measures
   learning, not memory. Here the choice is made per document with a seeded
   RNG while streaming, so the whole corpus never sits in memory.

2. PACKING.
   All documents are glued into one long stream of token ids with <eos>
   between them, stored as a flat uint16 array (2 bytes per token, ids < 65,536).

3. THE INPUT/TARGET SHIFT.
       x = tokens[i     : i + T]
       y = tokens[i + 1 : i + T + 1]
   y is x moved one step left: at every position the model sees x and must
   predict the next token y.
"""

import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

from src.prepare_data import read_jsonl

# uint16 = 2 bytes per token. Valid as long as vocab_size <= 65,536.
DTYPE = np.uint16


class DatasetBuilder:
    def __init__(self, tokenizer, val_ratio=0.01, seed=42, batch_docs=1000):
        self.tokenizer = tokenizer  # a tokenizers.Tokenizer
        self.val_ratio = val_ratio
        self.seed = seed
        self.batch_docs = batch_docs  # documents per encode_batch call
        self.vocab_size = tokenizer.get_vocab_size()
        self.eos_id = tokenizer.token_to_id("<eos>")

    def encode_batch(self, texts, f) -> list[int]:
        """Encode a list of texts, add <eos> after each, append to file `f`.
        Return the number of tokens written for each text (including <eos>)."""
        ids, lengths = [], []
        for encoding in self.tokenizer.encode_batch(texts):
            ids.extend(encoding.ids)
            ids.append(self.eos_id)  # mark where this document ends
            lengths.append(len(encoding.ids) + 1)

        array = np.array(ids, dtype=np.int64)
        # Check BEFORE casting to uint16: a bad id would silently wrap around.
        if array.min() < 0 or array.max() >= self.vocab_size:
            raise ValueError(f"token id out of range: max {array.max()}, vocab {self.vocab_size}")
        array.astype(DTYPE).tofile(f)  # write straight to disk
        return lengths

    def run(self, input_path, train_bin, val_bin, meta_path, block_size=256) -> dict:
        input_path, train_bin, val_bin, meta_path = map(Path, (input_path, train_bin, val_bin, meta_path))
        train_bin.parent.mkdir(parents=True, exist_ok=True)

        # Same seed -> same split every run.
        rng = random.Random(self.seed)
        docs = {"train": 0, "val": 0}
        tokens = {"train": 0, "val": 0}
        train_tokens_by_source = Counter()
        # Waiting texts per split, encoded once a batch is full.
        pending = {"train": [], "val": []}

        with open(train_bin, "wb") as train_f, open(val_bin, "wb") as val_f:
            files = {"train": train_f, "val": val_f}

            def flush(split):
                texts = [text for text, _ in pending[split]]
                lengths = self.encode_batch(texts, files[split])
                tokens[split] += sum(lengths)
                if split == "train":
                    for (_, source), n in zip(pending[split], lengths):
                        train_tokens_by_source[source] += n
                pending[split] = []

            for doc in read_jsonl(input_path):
                split = "val" if rng.random() < self.val_ratio else "train"
                pending[split].append((doc["text"], doc.get("source", input_path.stem)))
                docs[split] += 1
                if len(pending[split]) >= self.batch_docs:
                    flush(split)
                if (docs["train"] + docs["val"]) % 100_000 == 0:
                    print(f"  {docs['train'] + docs['val']:,} docs ...")
            for split in pending:  # the last, not-full batches
                if pending[split]:
                    flush(split)

        for split, path in [("train", train_bin), ("val", val_bin)]:
            print(f"{path.name} -> {docs[split]:,} docs, {tokens[split]:,} tokens")

        # meta.json lets training read the dataset facts without opening the .bin files.
        meta = {
            "vocab_size": self.vocab_size,
            "eos_id": self.eos_id,
            "dtype": np.dtype(DTYPE).name,
            "block_size": block_size,
            "seed": self.seed,
            "val_ratio": self.val_ratio,
            "n_docs_train": docs["train"],
            "n_docs_val": docs["val"],
            "n_tokens_train": tokens["train"],
            "n_tokens_val": tokens["val"],
            "train_tokens_by_source": dict(sorted(train_tokens_by_source.items())),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")
        print("Saved", meta_path.name)
        return meta

    @staticmethod
    def get_batch(bin_path, batch_size, block_size, rng=None):
        """Cut random windows out of the token stream; return (x, y)."""
        rng = rng or np.random.default_rng()
        # memmap: the file stays on disk, only the slices we ask for are read.
        data = np.memmap(bin_path, dtype=DTYPE, mode="r")

        # -1 so that y still has one token left past the end of x.
        starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
        x = np.stack([data[i:i + block_size].astype(np.int64) for i in starts])
        y = np.stack([data[i + 1:i + block_size + 1].astype(np.int64) for i in starts])
        return x, y
