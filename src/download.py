"""Download one Hugging Face source from configs/sources.yaml to data/raw/<id>.jsonl.

Usage (from the project root):
    python src/download.py --source fineweb2_uz
    python src/download.py --source uz_news --force

The dataset is streamed (streaming=True), so nothing big is cached in memory.
Empty texts are skipped. If the source has max_chars, we stop once the total
number of saved characters reaches it. Each output line looks like:
    {"id": "<source>-<n>", "text": "...", "url": "..."}   # "url" only if the dataset has it
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from check_sources import load_sources

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def find_source(source_id: str) -> dict:
    """Return the config entry for `source_id`, or stop with a clear error."""
    for src in load_sources():
        if src["id"] == source_id:
            return src
    raise SystemExit(f"Unknown source '{source_id}' in configs/sources.yaml")


def download(src: dict, out_path: Path) -> tuple[int, int]:
    """Stream the dataset into `out_path`. Return (docs_saved, chars_saved)."""
    max_chars = src.get("max_chars")  # None = no limit
    dataset = load_dataset(src["repo"], src.get("subset"), split="train", streaming=True)

    docs = chars = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in dataset:
            text = row.get(src["text_field"]) or ""
            if not text.strip():
                continue  # skip empty texts
            doc = {"id": f"{src['id']}-{docs}", "text": text}
            if row.get("url"):
                doc["url"] = row["url"]  # provenance: which website the text came from
            # ensure_ascii=False keeps Uzbek letters readable in the file.
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            docs += 1
            chars += len(text)
            if docs % 100_000 == 0:
                print(f"  {docs:,} docs, {chars:,} chars")
            if max_chars is not None and chars >= max_chars:
                break
    return docs, chars


def main() -> None:
    parser = argparse.ArgumentParser(description="Download one source to data/raw/.")
    parser.add_argument("--source", required=True, help="source id, e.g. fineweb2_uz")
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = parser.parse_args()

    src = find_source(args.source)
    out_path = RAW_DIR / f"{src['id']}.jsonl"
    if out_path.exists() and not args.force:
        print(f"Skip: data/raw/{out_path.name} already exists (use --force to overwrite).")
        return

    print(f"[{src['id']}] downloading {src['repo']} (max_chars={src.get('max_chars')}) ...")
    docs, chars = download(src, out_path)
    print(f"[{src['id']}] saved {docs:,} documents, {chars:,} characters -> data/raw/{out_path.name}")


if __name__ == "__main__":
    main()
