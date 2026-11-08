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
| 2. Download | Streams one Hugging Face source (optional `max_chars` cap) | `python src/download.py --source <id>` | `data/raw/<id>.jsonl` |
| 3. Data preparation | Cleaning → filtering → exact dedup → PII masking, per source | `python src/prepare_data.py --source <id>` | `data/final/<id>.jsonl`, `manifests/<id>_*.json` |
| 4. Mixture | Cross-source exact dedup; random sample of capped sources (`mix_max_chars`) | `python src/mix_data.py` | `data/final/mixture.jsonl`, `manifests/mixture.json` |
| 5. Tokenizer v1 | Byte-level BPE (8,000 tokens) on 300k mixture docs; round-trip test and held-out fertility | `python src/tokenizer.py` | `tokenizer/tokenizer.json`, `manifests/tokenizer.json` |
| 6. Dataset | Document-level train/val split, packing with `<eos>`, uint16 | `python scripts/build_dataset.py` | `dataset/train.bin`, `val.bin`, `meta.json` |
| 7. Model design | Parameters, memory and tokens per parameter; 4 model sanity checks | `python scripts/design_model.py`, `python tests/test_model.py` | — |
| 8. Training | Warmup + cosine lr, grad accumulation, AMP on CUDA, resumable checkpoints | `python scripts/train.py --config configs/run_01.yaml` | `checkpoints/last.pt`, `best.pt`, `train_log.jsonl` |

## Current numbers

- **Corpus (mixture):** 570,222 documents, ~1.23B characters, ~148M words —
  fineweb2_uz 36%, uz_news 33% (capped at 400M characters), uz_wiki 31%
- **Tokenizer v1:** byte-level BPE, vocab 8,000, ~2.24 tokens per word
  (uz_news 2.09, fineweb2_uz 2.29, uz_wiki 2.33; held-out docs)
- **Dataset:** train 328.3M tokens (564,546 docs), val 3.2M tokens (5,676 docs)
- **Model `run_01`:** GPT-style decoder, 6 layers, `d_model` 384, 6 heads, context 256,
  tied embeddings — **~13.8M parameters**, ~210 MB training memory before activations
- **Data budget:** ~23.8 tokens per parameter (target ~20)
- **Training plan:** 8,000 steps × 32,768 tokens = ~262M tokens (~0.8 pass over train)

## Project structure

- `configs/` – data sources (`sources.yaml`) and run configs (`run_01.yaml`: run, model, data, training)
- `src/` – data preparation, tokenizer, dataset, model and training code
- `scripts/` – entry points that use the code in `src/`
- `notebooks/` – exploration notebooks and `train_colab.ipynb`
- `tests/` – model sanity checks
- `tokenizer/` – the trained tokenizer
- `manifests/` – processing logs and statistics for each step
- `data/`, `dataset/*.bin`, `checkpoints/` – local files (not tracked by Git)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run all commands from the project root.

## Training

Local (Mac GPU, short test):

```bash
python tests/test_model.py
python scripts/train.py --config configs/run_01.yaml --max_steps 30 --out_dir checkpoints/smoke
python scripts/train.py --config configs/run_01.yaml --resume checkpoints/smoke/last.pt --max_steps 50 --out_dir checkpoints/smoke
```

Options: `--resume`, `--data_dir`, `--out_dir`, `--max_steps`, `--device` (auto, cuda, mps, cpu).

### On Google Colab

1. Upload `dataset/train.bin`, `val.bin` and `meta.json` to Google Drive: `MyDrive/sunbae/dataset/`.
2. Open `notebooks/train_colab.ipynb` in Colab and select a GPU runtime.
3. Run all cells. The notebook clones this repo (branch `asadbek`), installs the
   requirements, runs `tests/test_model.py`, copies the dataset to `/content/dataset/`
   (reading random batches from the Drive mount is slow) and trains with checkpoints in
   `MyDrive/sunbae/checkpoints/`.
4. After a disconnect, run all cells again: training resumes from `last.pt`.

## Data and licenses

Each source in `configs/sources.yaml` records its license and our decision
(`use`, `review` or `blocked`). Uzbek Wikipedia text is used under CC BY-SA 4.0.
Emails, phone numbers and IP addresses are masked as `<EMAIL>`, `<PHONE>` and `<IP>`.

## Known limitations

- Instagram embed text and some site boilerplate remain in uz_news.
- Exact dedup runs before PII masking, so a few duplicates survive.
- No fuzzy dedup yet (near-duplicate news articles).
- uz_news is capped at 400M characters in the mixture; the other ~307k documents are not used.
- Tokenizer v1 is slightly less efficient on Wikipedia than v0 (2.33 vs 2.27 tokens per word).

## Status

Work in progress: data, tokenizer v1, dataset and training code are done;
next is the full `run_01` training on Colab, then evaluation and generation.
