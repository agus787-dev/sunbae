"""Turn the final mixture into dataset/train.bin, val.bin and meta.json.

    python scripts/build_dataset.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from src.dataset_builder import DatasetBuilder  # noqa: E402

INPUT = ROOT / "data" / "final" / "mixture.jsonl"
TOKENIZER_FILE = ROOT / "tokenizer" / "tokenizer.json"

TRAIN_BIN = ROOT / "dataset" / "train.bin"
VAL_BIN = ROOT / "dataset" / "val.bin"
META = ROOT / "dataset" / "meta.json"

BLOCK_SIZE = 256
VAL_RATIO = 0.01  # 1% of documents held out for validation
SEED = 42

tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
builder = DatasetBuilder(tokenizer, val_ratio=VAL_RATIO, seed=SEED)
builder.run(INPUT, TRAIN_BIN, VAL_BIN, META, block_size=BLOCK_SIZE)

# Show the input/target shift on real data. y must be x moved one step left.
rng = np.random.default_rng(SEED)
x, y = builder.get_batch(TRAIN_BIN, batch_size=1, block_size=8, rng=rng)
print()
print("x (input) :", x[0].tolist())
print("y (target):", y[0].tolist())
print("Shift correct:", bool((x[0][1:] == y[0][:-1]).all()))

# Decode one 64-token window from val.bin, to see real text comes back.
x, _ = builder.get_batch(VAL_BIN, batch_size=1, block_size=64, rng=rng)
print()
print("64-token window from val.bin:")
print(repr(tokenizer.decode(x[0].tolist(), skip_special_tokens=False)))
