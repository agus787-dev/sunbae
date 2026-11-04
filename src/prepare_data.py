"""Clean, filter, deduplicate and PII-mask one data source (default: uz_wiki).

Usage (from the project root):
    python src/prepare_data.py
    python src/prepare_data.py --source uz_wiki

Steps: cleaning -> filtering -> exact_dedup -> pii_masking -> summary.
Each step reads the previous step's JSONL file line by line (so the whole
dataset never has to fit in memory), writes its own output file, writes the
removed documents to a separate file for review, and writes a manifest.
The logic, regexes and thresholds are the same as in notebooks/downloads.ipynb.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

# ============================================================
# Settings
# ============================================================

DEFAULT_SOURCE = "uz_wiki"

# Project root = the folder that contains src/. All paths are built from it,
# and manifests store paths RELATIVE to it, so they contain no usernames.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = PROJECT_ROOT / "manifests"

# "{source}" is replaced by the source id, e.g. "uz_wiki".
PATHS = {
    "raw": "data/raw/{source}.jsonl",
    "cleaned": "data/cleaned/{source}_cleaned.jsonl",
    "empty_removed": "data/cleaned/{source}_empty_removed.jsonl",  # docs that became empty
    "filtered": "data/filtered/{source}_filtered.jsonl",
    "filter_removed": "data/filtered/{source}_removed.jsonl",      # docs that failed a quality rule
    "deduped": "data/deduped/{source}_deduped.jsonl",
    "duplicates": "data/deduped/{source}_duplicates.jsonl",        # exact duplicates
    "final": "data/final/{source}.jsonl",                          # output of the last step
}

# Different ways Uzbek texts write the o' / g' / tutuq belgisi apostrophe.
# Note: NFKC runs first and turns "´" into " ́" (space + combining accent),
# so "´" never reaches the apostrophe step. Kept to match the notebook.
APOSTROPHES = ["ʻ", "ʼ", "‘", "’", "`", "´"]

FILTER_CONFIG = {
    "min_chars": 200,                # shorter docs are mostly stubs/redirects
    "min_alpha_ratio": 0.50,         # below this, the doc is mostly numbers/symbols
    "max_repeated_line_ratio": 0.30, # above this, the doc is mostly repeated boilerplate
}

# The ORDER matters: EMAIL runs first, so digits inside an e-mail address are
# already replaced and can't be matched again as a phone number.
# Known limits (kept to match the notebook): IP does not check 0-255,
# and EMAIL can include text glued right after the domain.
PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": re.compile(
        r"(?:\+998[\s-]?\d{2}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2})"   # Uzbekistan
        r"|(?:\+82[\s-]?\d{1,2}[\s-]?\d{3,4}[\s-]?\d{4})"          # Korea, international
        r"|(?:\b01[016789][\s-]?\d{3,4}[\s-]?\d{4}\b)"              # Korea, mobile 010/011/016-019
    ),
    "IP": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


# ============================================================
# File helpers
# ============================================================

def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield documents one by one (a generator, so memory use stays small)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class JsonlWriter:
    """Write documents to a JSONL file and count them. Use with `with`."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.count = 0

    def __enter__(self) -> "JsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", encoding="utf-8")
        return self

    def write(self, doc: dict[str, Any]) -> None:
        # ensure_ascii=False keeps Uzbek/Korean letters readable in the file.
        self._file.write(json.dumps(doc, ensure_ascii=False) + "\n")
        self.count += 1

    def __exit__(self, *exc: object) -> None:
        self._file.close()


def relative(path: Path) -> str:
    """'data/raw/uz_wiki.jsonl' instead of '/Users/<name>/.../data/raw/uz_wiki.jsonl'.

    Manifests are committed to Git, so they must not contain absolute paths.
    """
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")


# ============================================================
# Step 1: cleaning
# ============================================================

class TextNormalizer:
    """Clean raw text. The ORDER of the steps matters and matches the notebook:
    HTML tags -> HTML entities -> NFKC -> apostrophes -> control chars -> whitespace.
    """

    def normalize(self, text: str) -> str:
        """Return the cleaned text ("" if nothing useful is left)."""
        text = self.remove_html(text)
        text = self.decode_entities(text)
        text = self.normalize_unicode(text)
        text = self.normalize_apostrophes(text)
        text = self.remove_control_chars(text)
        text = self.normalize_whitespace(text)
        return text

    def remove_html(self, text: str) -> str:
        # Replace with a space (not "") so "a<br>b" doesn't glue into "ab".
        return re.sub(r"<[^>]+>", " ", text)

    def decode_entities(self, text: str) -> str:
        # Done AFTER tag removal, so "&lt;b&gt;" stays as the text "<b>".
        return html.unescape(text)

    def normalize_unicode(self, text: str) -> str:
        # NFKC maps look-alike characters to one form, e.g. \xa0 -> " ", "ﬁ" -> "fi".
        return unicodedata.normalize("NFKC", text)

    def normalize_apostrophes(self, text: str) -> str:
        # One apostrophe form means "oʻzbek" and "o'zbek" become the same word.
        for symbol in APOSTROPHES:
            text = text.replace(symbol, "'")
        return text

    def remove_control_chars(self, text: str) -> str:
        # Unicode category "C*" = control/format/unassigned characters
        # (e.g. \x00, zero-width space). \n and \t carry structure, so keep them.
        return "".join(
            ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C"
        )

    def normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)    # many spaces -> one space
        text = re.sub(r" *\n *", "\n", text)    # no spaces around newlines
        text = re.sub(r"\n{3,}", "\n\n", text)  # keep at most one blank line
        return text.strip()


# ============================================================
# Step 2: filtering
# ============================================================

class QualityFilter:
    """Decide whether a document is good enough to keep."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def reason(self, doc: dict[str, Any]) -> str | None:
        """Return the name of the first failed rule, or None if the doc passes.

        Only the FIRST failed rule is reported, so every removed doc is
        counted exactly once. Cheap checks go first.
        """
        text = doc["text"]
        if len(text) < self.config["min_chars"]:
            return "min_chars"
        if self.alphabetic_ratio(text) < self.config["min_alpha_ratio"]:
            return "min_alpha_ratio"
        if self.repeated_line_ratio(text) > self.config["max_repeated_line_ratio"]:
            return "max_repeated_line_ratio"
        return None

    @staticmethod
    def alphabetic_ratio(text: str) -> float:
        """Share of characters that are letters (spaces and digits are not)."""
        if not text:
            return 0.0
        return sum(ch.isalpha() for ch in text) / len(text)

    @staticmethod
    def repeated_line_ratio(text: str) -> float:
        """Share of non-empty lines that appear more than once (A, A, B -> 2/3)."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) <= 1:
            return 0.0
        counts = Counter(lines)
        repeated = sum(c for c in counts.values() if c > 1)
        return repeated / len(lines)


# ============================================================
# Step 3: exact deduplication
# ============================================================

class ExactDeduplicator:
    """Keep the first occurrence of each text; later copies are duplicates."""

    def __init__(self) -> None:
        # hash -> title of the first doc with that text. A short hash instead
        # of the full text keeps memory small.
        self.first_title: dict[str, str] = {}

    @staticmethod
    def text_hash(text: str) -> str:
        """SHA-256 of the lowercased text with all whitespace collapsed to one space,
        so texts that differ only in case or spacing count as the same."""
        key = re.sub(r"\s+", " ", text.lower()).strip()
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def check(self, doc: dict[str, Any]) -> str | None:
        """Return the title of the earlier identical doc, or None if `doc` is new."""
        h = self.text_hash(doc["text"])
        if h in self.first_title:
            return self.first_title[h]
        self.first_title[h] = doc["title"]
        return None

    def run(self, docs: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
        """Split a list of docs into (kept, duplicates). Handy for quick checks."""
        kept, duplicates = [], []
        for doc in docs:
            original = self.check(doc)
            if original is None:
                kept.append(doc)
            else:
                duplicates.append({**doc, "duplicate_of": original})
        return kept, duplicates


# ============================================================
# Step 4: PII masking
# ============================================================

class PIIMasker:
    """Replace e-mails, phones and IPs with <EMAIL>, <PHONE>, <IP>.

    Person names are NOT masked: in Wikipedia they are mostly public figures.
    """

    def mask(self, text: str) -> tuple[str, dict[str, int]]:
        """Return (masked_text, {label: number_of_replacements})."""
        counts = {}
        for label, pattern in PII_PATTERNS.items():
            text, n = pattern.subn(f"<{label}>", text)
            counts[label] = n
        return text, counts


# ============================================================
# Pipeline
# ============================================================

class Pipeline:
    """Run all steps for one source, save outputs, and write manifests."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.paths = {name: PROJECT_ROOT / p.format(source=source) for name, p in PATHS.items()}
        self.normalizer = TextNormalizer()
        self.quality_filter = QualityFilter(FILTER_CONFIG)
        self.masker = PIIMasker()
        self.manifests: dict[str, dict[str, Any]] = {}  # step name -> manifest
        self.final_size: dict[str, int] = {}

    def save_manifest(self, step: str, file_suffix: str, manifest: dict[str, Any]) -> None:
        write_manifest(manifest, MANIFESTS_DIR / f"{self.source}_{file_suffix}.json")
        self.manifests[step] = manifest

    def clean(self) -> None:
        """Normalize every text and drop documents that end up empty."""
        src, out, empty_out = self.paths["raw"], self.paths["cleaned"], self.paths["empty_removed"]
        docs_before = chars_before = chars_after = docs_changed = 0

        with JsonlWriter(out) as kept, JsonlWriter(empty_out) as empty:
            for doc in read_jsonl(src):
                docs_before += 1
                chars_before += len(doc["text"])
                text = self.normalizer.normalize(doc["text"])
                if text != doc["text"]:
                    docs_changed += 1
                if not text:
                    empty.write(doc)  # keep the ORIGINAL text, to see what was lost
                    continue
                chars_after += len(text)
                kept.write({**doc, "text": text})

        self.save_manifest("cleaning", "cleaned", {
            "source": self.source,
            "step": "cleaning",
            "input": relative(src),
            "output": relative(out),
            "removed_output": relative(empty_out),
            "docs_before": docs_before,
            "docs_after": kept.count,
            "empty_removed": empty.count,
            "docs_changed": docs_changed,
            "chars_before": chars_before,
            "chars_after": chars_after,
        })

    def filter(self) -> None:
        """Remove low-quality documents and record why each one was removed."""
        src, out, removed_out = self.paths["cleaned"], self.paths["filtered"], self.paths["filter_removed"]
        # Start every reason at 0 so the manifest always lists all rules.
        removed_by_reason = {"min_chars": 0, "min_alpha_ratio": 0, "max_repeated_line_ratio": 0}

        with JsonlWriter(out) as kept, JsonlWriter(removed_out) as removed:
            for doc in read_jsonl(src):
                reason = self.quality_filter.reason(doc)
                if reason is None:
                    kept.write(doc)
                else:
                    removed_by_reason[reason] += 1
                    removed.write({**doc, "filtered": True, "filter_reason": reason})

        self.save_manifest("filtering", "filtered", {
            "source": self.source,
            "step": "filtering",
            "input": relative(src),
            "output": relative(out),
            "removed_output": relative(removed_out),
            "config": FILTER_CONFIG,
            "docs_before": kept.count + removed.count,
            "docs_after": kept.count,
            "removed_total": removed.count,
            "removed_by_reason": removed_by_reason,
        })

    def deduplicate(self) -> None:
        """Drop exact duplicates; keep the first occurrence of each text."""
        src, out, dup_out = self.paths["filtered"], self.paths["deduped"], self.paths["duplicates"]
        dedup = ExactDeduplicator()
        examples = []

        with JsonlWriter(out) as kept, JsonlWriter(dup_out) as dups:
            for doc in read_jsonl(src):
                original = dedup.check(doc)
                if original is None:
                    kept.write(doc)
                    continue
                dups.write({**doc, "duplicate_of": original})
                if len(examples) < 5:
                    examples.append({"removed": doc["title"], "duplicate_of": original})

        self.save_manifest("exact_dedup", "deduped", {
            "source": self.source,
            "step": "exact_dedup",
            "method": "SHA-256 of lowercased, whitespace-normalized text; first occurrence kept",
            "input": relative(src),
            "output": relative(out),
            "duplicates_output": relative(dup_out),
            "docs_before": kept.count + dups.count,
            "docs_after": kept.count,
            "duplicates_removed": dups.count,
            "examples": examples,
        })

    def mask_pii(self) -> None:
        """Mask e-mails, phone numbers and IPs. No documents are removed."""
        src, out = self.paths["deduped"], self.paths["final"]
        masked_counts = Counter({label: 0 for label in PII_PATTERNS})
        docs_with_pii = total_chars = total_words = 0

        with JsonlWriter(out) as writer:
            for doc in read_jsonl(src):
                text, counts = self.masker.mask(doc["text"])
                masked_counts.update(counts)
                if sum(counts.values()) > 0:
                    docs_with_pii += 1
                # Final dataset size, counted here so we don't re-read the file.
                total_chars += len(text)
                total_words += len(text.split())
                writer.write({**doc, "text": text})

        self.final_size = {"characters": total_chars, "words": total_words}
        self.save_manifest("pii_masking", "pii", {
            "source": self.source,
            "step": "pii_masking",
            "method": "regex masking: EMAIL, PHONE (+998, +82, Korean mobile), IP",
            "not_masked": "person names (public figures in Wikipedia)",
            "input": relative(src),
            "output": relative(out),
            "docs": writer.count,
            "docs_with_pii": docs_with_pii,
            "masked_counts": dict(masked_counts),
        })

    def summarize(self) -> None:
        """Combine the step manifests into one summary manifest."""
        m = self.manifests
        raw_docs = m["cleaning"]["docs_before"]
        final_docs = m["pii_masking"]["docs"]
        self.save_manifest("summary", "summary", {
            "source": self.source,
            "pipeline": ["cleaning", "filtering", "exact_dedup", "pii_masking"],
            "raw_docs": raw_docs,
            "final_docs": final_docs,
            "final_output": m["pii_masking"]["output"],
            "kept_ratio": round(final_docs / raw_docs, 4),
            "removed": {
                "empty_after_cleaning": m["cleaning"]["empty_removed"],
                **m["filtering"]["removed_by_reason"],
                "exact_duplicates": m["exact_dedup"]["duplicates_removed"],
            },
            "pii_masked": m["pii_masking"]["masked_counts"],
            "final_size": self.final_size,
        })

    def run(self) -> None:
        if not self.paths["raw"].exists():
            raise FileNotFoundError(f"Raw file not found: {relative(self.paths['raw'])}")
        for name, step in [
            ("cleaning", self.clean),
            ("filtering", self.filter),
            ("exact_dedup", self.deduplicate),
            ("pii_masking", self.mask_pii),
            ("summary", self.summarize),
        ]:
            print(f"[{self.source}] {name} ...")
            step()
        self.print_report()

    def print_report(self) -> None:
        """Print the same tables as the notebook's 'Final statistics' section."""
        m = self.manifests
        raw = m["cleaning"]["docs_before"]
        rows = [
            ("Raw", raw, 0),
            ("Cleaning", m["cleaning"]["docs_after"], m["cleaning"]["empty_removed"]),
            ("Filtering", m["filtering"]["docs_after"], m["filtering"]["removed_total"]),
            ("Exact dedup", m["exact_dedup"]["docs_after"], m["exact_dedup"]["duplicates_removed"]),
            ("PII masking", m["pii_masking"]["docs"], 0),
        ]
        print(f"\n{'Step':<14}{'Docs after':>12}{'Removed':>10}{'Kept of raw':>13}")
        for step, after, removed in rows:
            print(f"{step:<14}{after:>12,}{removed:>10,}{after / raw:>13.1%}")

        print("\nFiltering - removed by reason:")
        for reason, count in m["filtering"]["removed_by_reason"].items():
            print(f"  {reason:<28}{count:>8,}")

        print("\nPII masking:")
        print(f"  {'documents with PII':<28}{m['pii_masking']['docs_with_pii']:>8,}")
        for label, count in m["pii_masking"]["masked_counts"].items():
            print(f"  {label:<28}{count:>8,}")

        print("\nFinal dataset:")
        print(f"  {'characters':<18}{self.final_size['characters']:>14,}")
        print(f"  {'words':<18}{self.final_size['words']:>14,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare one data source for training.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="source id, e.g. uz_wiki")
    Pipeline(parser.parse_args().source).run()
