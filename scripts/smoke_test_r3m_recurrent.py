# Ensure R3M/ is on sys.path\nimport os\nimport sys\nR3M_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))\nif R3M_ROOT not in sys.path:\n    sys.path.insert(0, R3M_ROOT)\n\n#!/usr/bin/env python3
"""
Smoke test for Fix-B RÂ³M recurrent model:
- forward/backward
- short generation
"""

import os
import sys

import torch
from transformers import AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from r3m.models.recurrent import R3MRecurrentConfig, R3MRecurrentLM


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    cfg = R3MRecurrentConfig(
        vocab_size=len(tok),
        d_model=256,
        max_seq_len=64,
        k_train=2,
        k_max=8,
        episodic_enabled=True,
        episodic_slots=32,
    )
    model = R3MRecurrentLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    prompt = "Human: What is 2+2?\nAssistant:"
    enc = tok(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=48)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, k_steps=cfg.k_train)
    loss = out["loss"]
    print("loss", float(loss.item()))

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    ids = torch.tensor([tok.encode(prompt)], device=device)
    gen = model.generate(ids, max_new_tokens=40, k_steps=cfg.k_train, eos_token_id=tok.eos_token_id, repetition_penalty=1.1)
    # Print an ASCII-safe representation (avoid Windows cp1252 console encoding issues)
    text = tok.decode(gen[0].tolist())
    print(text.encode("ascii", errors="backslashreplace").decode("ascii"))


if __name__ == "__main__":
    main()



