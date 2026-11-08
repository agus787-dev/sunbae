"""Combine the final files of all "use" sources into one mixture file.

Usage (from the project root):
    python src/mix_data.py

Reads data/final/<id>.jsonl for every source with decision "use" in
configs/sources.yaml, removes exact duplicates ACROSS sources, and writes:
    data/final/mixture.jsonl   one {"id", "source", "text"} per line
    manifests/mixture.json     docs, characters, words and shares per source

If a source has mix_max_chars, only a random sample (seed 42) of its docs is
used, until the character limit is reached. data/final/<id>.jsonl stays complete.
"""

from __future__ import annotations

import random
from pathlib import Path

from check_sources import load_sources
from prepare_data import ExactDeduplicator, JsonlWriter, read_jsonl, relative, write_manifest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = PROJECT_ROOT / "data" / "final"
OUT_PATH = FINAL_DIR / "mixture.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "manifests" / "mixture.json"

# When the same text is in two sources, the copy from the EARLIER source is kept.
# Cleaner sources go first. Sources not listed here come after, in config order.
PRIORITY = ["uz_wiki", "uz_news", "fineweb2_uz"]


def ordered_sources() -> list[dict]:
    """Config entries of all "use" sources, sorted by PRIORITY."""
    used = {src["id"]: src for src in load_sources() if src["decision"] == "use"}
    first = [s for s in PRIORITY if s in used]
    return [used[s] for s in first + [s for s in used if s not in first]]


def sample_lines(path: Path, cap: int, seen: set[str]) -> set[int]:
    """Line numbers of a random sample (seed 42) of non-duplicate docs,
    picked until their characters reach `cap`."""
    candidates = []  # (line number, length) of docs not already in the mixture
    for i, doc in enumerate(read_jsonl(path)):
        if ExactDeduplicator.text_hash(doc["text"]) not in seen:
            candidates.append((i, len(doc["text"])))
    random.Random(42).shuffle(candidates)
    picked, chars = set(), 0
    for i, length in candidates:
        if chars >= cap:
            break
        picked.add(i)
        chars += length
    return picked


def main() -> None:
    sources = ordered_sources()
    seen: set[str] = set()  # hashes of all texts written so far
    stats: dict[str, dict] = {}

    with JsonlWriter(OUT_PATH) as writer:
        for src in sources:
            source, cap = src["id"], src.get("mix_max_chars")  # cap None = no limit
            path = FINAL_DIR / f"{source}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"Final file not found: {relative(path)}")
            picked = sample_lines(path, cap, seen) if cap is not None else None
            docs = chars = words = duplicates = not_sampled = 0
            for i, doc in enumerate(read_jsonl(path)):
                h = ExactDeduplicator.text_hash(doc["text"])
                if h in seen:
                    duplicates += 1  # already kept from a higher-priority source
                    continue
                if picked is not None and i not in picked:
                    not_sampled += 1  # left out because of mix_max_chars
                    continue
                seen.add(h)
                writer.write({"id": doc["id"], "source": source, "text": doc["text"]})
                docs += 1
                chars += len(doc["text"])
                words += len(doc["text"].split())
            stats[source] = {
                "input": relative(path),
                "docs": docs,
                "characters": chars,
                "words": words,
                "cross_source_duplicates_removed": duplicates,
                "mix_max_chars": cap,
                "docs_not_sampled": not_sampled,
            }
            print(f"[{source}] {docs:,} docs kept, {duplicates:,} cross-source duplicates, "
                  f"{not_sampled:,} not sampled")

    total_chars = sum(s["characters"] for s in stats.values())
    for s in stats.values():
        s["char_share"] = round(s["characters"] / total_chars, 4)

    write_manifest({
        "step": "mixture",
        "method": "cross-source exact dedup: SHA-256 of lowercased, whitespace-normalized "
                  "text; first copy kept in priority order",
        "priority": [src["id"] for src in sources],
        "sampling": "sources with mix_max_chars: random sample (seed 42) until the limit",
        "output": relative(OUT_PATH),
        "sources": stats,
        "totals": {
            "docs": writer.count,
            "characters": total_chars,
            "words": sum(s["words"] for s in stats.values()),
            "cross_source_duplicates_removed": sum(
                s["cross_source_duplicates_removed"] for s in stats.values()
            ),
        },
    }, MANIFEST_PATH)

    # Short table
    print(f"\n{'Source':<14}{'Docs':>11}{'Characters':>16}{'Words':>14}{'Share':>8}{'Dups':>8}")
    for source, s in stats.items():
        print(f"{source:<14}{s['docs']:>11,}{s['characters']:>16,}{s['words']:>14,}"
              f"{s['char_share']:>8.1%}{s['cross_source_duplicates_removed']:>8,}")
    t = sum(s["words"] for s in stats.values())
    d = sum(s["cross_source_duplicates_removed"] for s in stats.values())
    print(f"{'TOTAL':<14}{writer.count:>11,}{total_chars:>16,}{t:>14,}{1:>8.1%}{d:>8,}")
    print(f"\nWrote {relative(OUT_PATH)} and {relative(MANIFEST_PATH)}")


if __name__ == "__main__":
    main()
