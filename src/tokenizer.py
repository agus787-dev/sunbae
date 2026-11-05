"""Train, test and save a byte-level BPE tokenizer for Sunbae.

Usage:
    python src/tokenizer.py
"""

import json
import random
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# --- Settings -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "final" / "uz_wiki.jsonl"
TOKENIZER_DIR = ROOT / "tokenizer"
MANIFESTS = ROOT / "manifests"

VOCAB_SIZE = 8000
SPECIAL_TOKENS = ["<eos>"]
MIN_FREQUENCY = 2
ROUND_TRIP_SAMPLES = 1000
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

def read_texts(path: Path):
    """Yield texts one by one, so the whole file is never loaded into memory."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def sample_texts(path: Path, n: int, seed: int) -> list[str]:
    """Pick n random texts from the file in one pass (reservoir sampling)."""
    rng = random.Random(seed)
    sample = []
    for i, text in enumerate(read_texts(path)):
        if i < n:
            sample.append(text)
        else:
            j = rng.randint(0, i)
            if j < n:
                sample[j] = text
    return sample


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
    builder = BPETokenizerBuilder()

    print("Training tokenizer...")
    builder.train(read_texts(DATA_FILE))
    vocab_size = builder.tokenizer.get_vocab_size()
    print("Vocab size:", vocab_size)

    print("Round-trip test...")
    edge_failures = builder.round_trip(EDGE_CASES)
    sample = sample_texts(DATA_FILE, ROUND_TRIP_SAMPLES, SEED)
    sample_failures = builder.round_trip(sample)
    print(f"  edge cases: {len(EDGE_CASES) - len(edge_failures)} / {len(EDGE_CASES)} passed")
    print(f"  corpus:     {len(sample) - len(sample_failures)} / {len(sample)} passed")
    assert not edge_failures and not sample_failures, "Round-trip test failed"

    fertility = builder.fertility(sample)
    print(f"Fertility: {fertility:.2f} tokens per word")

    path = builder.save(TOKENIZER_DIR)
    print("Saved:", path.relative_to(ROOT))

    manifest = {
        "step": "tokenizer",
        "type": "byte-level BPE",
        "input": str(DATA_FILE.relative_to(ROOT)),
        "output": str(path.relative_to(ROOT)),
        "vocab_size": vocab_size,
        "special_tokens": {t: builder.tokenizer.token_to_id(t) for t in SPECIAL_TOKENS},
        "min_frequency": MIN_FREQUENCY,
        "round_trip": {"edge_cases": len(EDGE_CASES), "corpus_samples": len(sample), "failures": 0},
        "fertility": round(fertility, 3),
    }
    MANIFESTS.mkdir(exist_ok=True)
    with open(MANIFESTS / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("Manifest saved.")


if __name__ == "__main__":
    main()