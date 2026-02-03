# Ensure R3M/ is on sys.path\nimport os\nimport sys\nR3M_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))\nif R3M_ROOT not in sys.path:\n    sys.path.insert(0, R3M_ROOT)\n\n#!/usr/bin/env python3
"""
RÂ³M smoke test:
- instantiate a tiny model
- run one forward + one backward step
- generate a short continuation
"""

import os
import sys

import torch
from transformers import AutoTokenizer

# Ensure repo root is on sys.path so `import src...` works when running from /scripts
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.r3m import R3MConfig, R3MModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    cfg = R3MConfig(
        vocab_size=len(tok),
        d_model=256,
        max_seq_len=128,
        n_layers=2,
        n_heads=4,
        k_train=4,
        k_max=16,
        episodic_enabled=True,
        episodic_slots=32,
        halting_enabled=False,
    )
    model = R3MModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    prompt = "Human: Hello!\nAssistant:"
    enc = tok(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = out["loss"]
    print("loss", float(loss.item()), "avg_steps", float(out["steps"].float().mean().item()))

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    gen = model.generate(input_ids[:, : len(tok.encode(prompt))], max_new_tokens=40, k_steps=cfg.k_train)
    print(tok.decode(gen[0].tolist()))


if __name__ == "__main__":
    main()



