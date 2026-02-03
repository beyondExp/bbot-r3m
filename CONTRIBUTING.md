# Contributing

Thanks for your interest in contributing to **B-Bot R3M**.

## Ground rules

- Keep changes focused and easy to review.
- Do not commit large artifacts (datasets, tokbin shards, checkpoints). See `.gitignore` and `DATASETS.md`.
- Be explicit about licenses/attribution if you add new external resources.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Quick smoke tests

CPU-only minimal smoke:

```bash
python scripts/smoke_test_r3m.py
python scripts/smoke_test_r3m_recurrent.py
```

## Submitting a PR

- Describe what changed and why.
- Include repro steps for training/eval changes.
- If behavior changes, include before/after samples or metrics.

