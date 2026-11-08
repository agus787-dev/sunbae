"""Train, test and save a byte-level BPE tokenizer for Sunbae.

Usage:
    python src/tokenizer.py
    python src/tokenizer.py --input data/final/mixture.jsonl

Trains on a random sample of documents. Round-trip and fertility are measured
on other random documents (per source) that were NOT used for training.
"""

import argparse
import json
import random
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# --- Settings -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = "data/final/mixture.jsonl"  # relative to ROOT
TOKENIZER_DIR = ROOT / "tokenizer"
MANIFESTS = ROOT / "manifests"

VERSION = "v1"
VOCAB_SIZE = 8000
SPECIAL_TOKENS = ["<eos>"]
MIN_FREQUENCY = 2
TRAIN_DOCS = 300_000         # random docs used for training
EVAL_DOCS_PER_SOURCE = 1000  # random docs per source for round-trip + fertility
SEED = 42

EDGE_CASES = [
    "O'zbekiston",
    "  ikki   probel  ",
    "Birinchi qator\nIkkinchi qator\n\n",
    "Telefon: <PHONE>, email: <EMAIL>",
    "1990-2000 yillar, 3.14, 25%",
    "Ўзбекистон",
    "Salom 😊 dunyo",
    "선배 Sunbae",
    "",
]


# --- Helpers ------------------------------------------------------------------

def read_docs(path: Path):
    """Yield documents one by one, so the whole file is never loaded into memory."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def source_of(doc: dict, path: Path) -> str:
    """The doc's source; files without a "source" field (e.g. uz_wiki.jsonl) use the file name."""
    return doc.get("source", path.stem)


def pick_lines(path: Path, n_train: int, n_eval: int, seed: int) -> tuple[set[int], dict[str, set[int]]]:
    """Choose line numbers: n_train random docs for training, and for each
    source n_eval random docs that are NOT in the training sample."""
    sources = [source_of(doc, path) for doc in read_docs(path)]
    rng = random.Random(seed)
    train = set(rng.sample(range(len(sources)), min(n_train, len(sources))))

    # Held-out lines, grouped by source
    held_out: dict[str, list[int]] = {}
    for i, src in enumerate(sources):
        if i not in train:
            held_out.setdefault(src, []).append(i)
    evals = {src: set(rng.sample(lines, min(n_eval, len(lines)))) for src, lines in held_out.items()}
    return train, evals


def texts_at(path: Path, lines: set[int]):
    """Yield the texts of the chosen line numbers (a generator, low memory)."""
    for i, doc in enumerate(read_docs(path)):
        if i in lines:
            yield doc["text"]


# --- Tokenizer ----------------------------------------------------------------

class BPETokenizerBuilder:
    """Builds a byte-level BPE tokenizer: train -> test -> save."""

    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(models.BPE())
        # Byte-level: every text can be encoded, so no <unk> token is needed.
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()

    def train(self, texts) -> None:
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            min_frequency=MIN_FREQUENCY,
        )
        self.tokenizer.train_from_iterator(texts, trainer=trainer)

        # Token IDs are stored as uint16 later, so they must be below 65,536.
        assert self.tokenizer.get_vocab_size() <= 65_536, "Vocab too large for uint16"

    def round_trip(self, texts: list[str]) -> list[str]:
        """Return the texts that do NOT survive encode -> decode unchanged."""
        failures = []
        for text in texts:
            decoded = self.tokenizer.decode(self.tokenizer.encode(text).ids)
            if decoded != text:
                failures.append(text)
        return failures

    def fertility(self, texts: list[str]) -> float:
        """Average number of tokens per word (lower = more efficient)."""
        n_tokens = sum(len(self.tokenizer.encode(t).ids) for t in texts)
        n_words = sum(len(t.split()) for t in texts)
        return n_tokens / n_words

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "tokenizer.json"
        self.tokenizer.save(str(path))
        return path


# --- Main ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Sunbae tokenizer.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="JSONL file, relative to the project root")
    data_file = ROOT / parser.parse_args().input

    print("Choosing training and evaluation docs...")
    train_lines, eval_lines = pick_lines(data_file, TRAIN_DOCS, EVAL_DOCS_PER_SOURCE, SEED)
    print(f"  training docs: {len(train_lines):,}")

    builder = BPETokenizerBuilder()
    print("Training tokenizer...")
    builder.train(texts_at(data_file, train_lines))
    vocab_size = builder.tokenizer.get_vocab_size()
    print("Vocab size:", vocab_size)

    # Held-out texts per source, e.g. {"uz_wiki": [...], "uz_news": [...]}
    eval_texts = {src: list(texts_at(data_file, lines)) for src, lines in sorted(eval_lines.items())}
    all_eval = [t for texts in eval_texts.values() for t in texts]

    print("Round-trip test...")
    edge_failures = builder.round_trip(EDGE_CASES)
    sample_failures = builder.round_trip(all_eval)
    print(f"  edge cases: {len(EDGE_CASES) - len(edge_failures)} / {len(EDGE_CASES)} passed")
    print(f"  corpus:     {len(all_eval) - len(sample_failures)} / {len(all_eval)} passed")
    assert not edge_failures and not sample_failures, "Round-trip test failed"

    print("Fertility (tokens per word, held-out docs):")
    fertility = builder.fertility(all_eval)
    by_source = {src: round(builder.fertility(texts), 3) for src, texts in eval_texts.items()}
    for src, value in by_source.items():
        print(f"  {src:<14}{value:.3f}")
    print(f"  {'overall':<14}{fertility:.3f}")

    path = builder.save(TOKENIZER_DIR)
    print("Saved:", path.relative_to(ROOT))

    manifest = {
        "step": "tokenizer",
        "version": VERSION,
        "type": "byte-level BPE",
        "input": data_file.relative_to(ROOT).as_posix(),
        "output": path.relative_to(ROOT).as_posix(),
        "vocab_size": vocab_size,
        "special_tokens": {t: builder.tokenizer.token_to_id(t) for t in SPECIAL_TOKENS},
        "min_frequency": MIN_FREQUENCY,
        "train_docs": len(train_lines),
        "seed": SEED,
        "eval_docs_per_source": {src: len(texts) for src, texts in eval_texts.items()},
        "round_trip": {"edge_cases": len(EDGE_CASES), "corpus_samples": len(all_eval), "failures": 0},
        "fertility": round(fertility, 3),
        "fertility_by_source": by_source,
        "fertility_note": "held-out docs (not used for training); overall = all eval docs together",
    }
    MANIFESTS.mkdir(exist_ok=True)
    with open(MANIFESTS / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("Manifest saved.")


if __name__ == "__main__":
    main()