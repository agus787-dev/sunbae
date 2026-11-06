"""Design our model and check that we can afford it.

    python scripts/design_model.py [configs/run_01.yaml]

This script builds nothing. It reads a run config, multiplies a few numbers,
and tells us whether the design is sensible BEFORE we turn it into code and
spend hours training it.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model_config import load_config  # noqa: E402

CONFIG_FILE = ROOT / "configs" / "run_01.yaml"
DATA_MANIFEST = ROOT / "manifests" / "uz_wiki_summary.json"
TOKENIZER_MANIFEST = ROOT / "manifests" / "tokenizer.json"

config = load_config(CONFIG_FILE)
params = config.estimate_parameters()

print("=== Our LLM Model Design ===")
print()
print("Vocabulary size    :", config.vocab_size)
print("Context length     :", config.block_size)
print("Embedding dimension:", config.d_model)
print("Layers             :", config.n_layer)
print("Attention heads    :", config.n_head, "(head dim", config.head_dim(), ")")
print("Dropout            :", config.dropout)
print("MLP ratio          :", config.mlp_ratio)
print("Tied weights       :", config.tie_weights)
print()

print("Estimated parameters")
print("  non-embedding :  {:>12,}".format(params["non_embedding"]))
print("  embedding     :  {:>12,}".format(params["embedding"]))
print("  position      :  {:>12,}".format(params["position"]))
print("  TOTAL         :  {:>12,}   (~{:.1f}M)".format(
    params["total"], params["total"] / 1e6))
print()

print("Rough training memory for weights + gradients + AdamW: {:.0f} MB".format(
    config.estimate_training_memory_mb()))
print("(activations add more on top, and they grow with batch size)")
print()

# The compute-optimal check. About 20 training tokens per parameter is the
# standard rule of thumb. Fewer means the model is bigger than our data can
# teach; many more means we could have afforded a bigger model.
#
# Corpus tokens are estimated as words * tokenizer fertility (tokens per word),
# both taken from the data pipeline manifests.
if DATA_MANIFEST.exists() and TOKENIZER_MANIFEST.exists():
    words = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))["final_size"]["words"]
    fertility = json.loads(TOKENIZER_MANIFEST.read_text(encoding="utf-8"))["fertility"]
    n_tokens = int(words * fertility)
    ratio = n_tokens / params["total"]

    print("Training tokens available: ~{:,}   ({:,} words x {} tokens/word)".format(
        n_tokens, words, fertility))
    print("Tokens per parameter     : {:.1f}   (about 20 is the target)".format(ratio))

    if ratio < 10:
        print("-> The model is large for this corpus. Train several epochs, shrink it, or get more data.")
    elif ratio > 40:
        print("-> Plenty of data for this model. We could afford to go bigger.")
    else:
        print("-> Reasonably balanced.")
else:
    print("No manifests yet - run the data pipeline and tokenizer first to check")
    print("the tokens-per-parameter ratio against the real corpus.")

print()
print("Why this model?")
print()
print("We selected this configuration based on:")
print("- our dataset size (how many tokens we actually have)")
print("- available compute (Mac GPU via MPS, for how many hours)")
print("- memory limitations (it has to fit while training, not just exist)")
print("- inference cost (a smaller model is cheaper to serve, forever)")
print()
print("A bigger model is not automatically a better model.")
