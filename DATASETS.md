# Datasets and data redistribution

This repository contains **code** for training/evaluating R3M models. It does **not** grant any rights to
redistribute third-party datasets.

## Policy

- **Do not commit or redistribute** raw dataset exports, pretokenized shards, or other derived data unless you
  have verified you are allowed to do so under the dataset's license/terms.
- This repo may include scripts to **download/stream** datasets from their original sources. You are responsible
  for complying with the original licenses and any gated-access requirements.

## Local caches (ignored)

The following directories are intended to be **local-only** and are ignored by `.gitignore`:
- `data_cache/` (JSONL exports)
- `data_cache_tok*/` (pretokenized / tokbin-style caches)
- `outputs/` (checkpoints, logs, training runs)

## Hugging Face access

Some datasets require an access token. Set:

- `HF_TOKEN` (see `env.example`)

## Export utilities

See:
- `scripts/export_hf_to_jsonl.py` for exporting HF datasets to JSONL (`{"text": ...}` rows)

