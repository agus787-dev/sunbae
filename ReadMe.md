# Sunbae

An Uzbek-language LLM assistant that helps Uzbek students in Korea with daily life:
housing, visas and documents, part-time work, health insurance, banking and transport.

*Sunbae* (선배) is Korean for a senior who guides those who come after them.

The model is built from scratch, step by step: collect data, clean it, train a
tokenizer, design the model, then train it.

## Pipeline

| Step | What it does | Run | Output |
|---|---|---|---|
| 1. Sources | Lists data sources with license, provenance, decision and mixture weight | `python src/check_sources.py` | — |
| 2. Data preparation | Cleaning → filtering → exact dedup → PII masking | `python src/prepare_data.py --source uz_wiki` | `data/final/uz_wiki.jsonl`, `manifests/uz_wiki_*.json` |
| 3. Tokenizer | Trains a byte-level BPE tokenizer (8,000 tokens) with a round-trip test | `python src/tokenizer.py` | `tokenizer/tokenizer.json`, `manifests/tokenizer.json` |
| 4. Model design | Estimates parameters, memory and tokens per parameter for a run config | `python scripts/design_model.py` | — |
| 5. Training | *Not started* | | |

## Current numbers (uz_wiki)

- **Corpus:** 249,654 documents (83% of raw kept), ~47M words, ~108M tokens
- **Tokenizer:** byte-level BPE, vocab 8,000, ~2.3 tokens per word
- **Model `run_01`:** GPT-style decoder, 6 layers, `d_model` 384, 6 heads, context 256,
  tied embeddings — **~13.8M parameters**, ~210 MB training memory before activations
- **Data budget:** ~7.8 tokens per parameter (target ~20), so plan for 2–3 epochs

## Project structure

- `configs/` – data sources (`sources.yaml`) and run configs (`run_01.yaml`)
- `src/` – data preparation, tokenizer and model config code
- `scripts/` – reports that use the code in `src/`
- `notebooks/` – exploration notebooks for data preparation and the tokenizer
- `tokenizer/` – the trained tokenizer
- `manifests/` – processing logs and statistics for each step
- `data/` – local data (not tracked by Git)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run all commands from the project root. Training targets a Mac GPU (`device: mps`).

## Data and licenses

Each source in `configs/sources.yaml` records its license and our decision
(`use`, `review` or `blocked`). Uzbek Wikipedia text is used under CC BY-SA 4.0.
Emails, phone numbers and IP addresses are masked as `<EMAIL>`, `<PHONE>` and `<IP>`.

## Status

Work in progress: data, tokenizer and model design are done for `uz_wiki`;
next is building and training the model.

