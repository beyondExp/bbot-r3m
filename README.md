![R3M architecture diagram](docs/images/r3m.png)

## R3M (standalone bundle)

This folder is intended to be a **self-contained mini-repo** for the R3M work:

- **Architecture**: Hybrid SSM/Attention ~50M base model (Nemotron-H style)
- **Training**: single-GPU-friendly pretrain script + dataset mixture configs
- **Data**: utilities to flatten Hugging Face datasets into a simple `{"text": ...}` JSONL format

### Quickstart

From inside the `R3M/` folder:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/pretrain_hybrid_ssm.py --data-mode tiny --steps 50 --device cuda --amp bf16
```

### Dataset mixes (commercial-safe)

Mix configs live in `R3M/configs/mixes/`.

Notes:
- Some datasets may require an HF token (gated access). If needed, set `HF_TOKEN` in a `.env`.
  (This bundle will also read `research_architectures/.env` for backward compatibility.)
- Some datasets may require attribution/compliance (e.g. ODC-BY / CC-BY-4.0). Track your sources.

See `R3M/env.example` for the expected environment variable name.

### Open source / publishing notes

- **License**: Apache-2.0 (see `LICENSE` and `NOTICE`)
- **Trademarks**: see `TRADEMARKS.md` (B‑Bot naming/branding guidelines)
- **Citation**: see `CITATION.cff`
- **Datasets**: this repo does not grant redistribution rights for third-party datasets; see `DATASETS.md`
