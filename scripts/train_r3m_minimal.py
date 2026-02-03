# Ensure R3M/ is on sys.path\nimport os\nimport sys\nR3M_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))\nif R3M_ROOT not in sys.path:\n    sys.path.insert(0, R3M_ROOT)\n\n#!/usr/bin/env python3
"""
Minimal RÂ³M trainer (single GPU friendly).

Reads a JSONL or JSON file similar to existing scripts:
- If items have "text": uses that directly
- If items have "messages": concatenates role/content lines

Key correctness features:
- causal LM loss (shifted)
- padding labels masked to -100
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Ensure repo root is on sys.path so `import src...` works when running from /scripts
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.r3m import R3MConfig, R3MModel


class TextJsonDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 256, limit: int | None = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[Dict[str, Any]] = []

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)

        if p.suffix.lower() == ".jsonl":
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.examples.append(json.loads(line))
                    if limit and len(self.examples) >= limit:
                        break
        else:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.examples = data[:limit] if limit else data
            else:
                raise ValueError("Expected a list JSON file")

    def __len__(self):
        return len(self.examples)

    def _to_text(self, ex: Dict[str, Any]) -> str:
        if isinstance(ex, dict) and "text" in ex and isinstance(ex["text"], str):
            return ex["text"]

        if isinstance(ex, dict) and "messages" in ex and isinstance(ex["messages"], list):
            parts = []
            for m in ex["messages"]:
                role = (m.get("role") or "").strip()
                content = (m.get("content") or "").strip()
                if not content:
                    continue
                if role:
                    parts.append(f"{role}: {content}")
                else:
                    parts.append(content)
            return "\n".join(parts)

        return str(ex)

    def __getitem__(self, idx):
        text = self._to_text(self.examples[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # critical: do not learn on pad

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to JSON or JSONL dataset")
    ap.add_argument("--out", default="outputs/r3m_minimal", help="Output directory")
    ap.add_argument("--steps", type=int, default=200, help="Training steps")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (r3m_step_*.pt)")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size")
    ap.add_argument("--max-len", type=int, default=256, help="Max sequence length")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    ap.add_argument("--k-train", type=int, default=8, help="Thinking steps during training")
    ap.add_argument("--model-dim", type=int, default=512, help="Model width")
    ap.add_argument("--layers", type=int, default=6, help="Transformer layers")
    ap.add_argument("--heads", type=int, default=8, help="Attention heads")
    ap.add_argument("--limit", type=int, default=5000, help="Max examples to load")
    ap.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    ap.add_argument("--save-every", type=int, default=200, help="Save checkpoint every N steps")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ds = TextJsonDataset(args.data, tokenizer, max_length=args.max_len, limit=args.limit)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    cfg = R3MConfig(
        vocab_size=len(tokenizer),
        d_model=args.model_dim,
        max_seq_len=args.max_len,
        n_layers=args.layers,
        n_heads=args.heads,
        k_train=args.k_train,
        k_max=max(64, args.k_train),
        episodic_enabled=True,
        episodic_slots=128,
        halting_enabled=False,
    )
    model = R3MModel(cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    model.train()
    it = iter(dl)
    start_step = 0

    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        if "model_state_dict" not in ck:
            raise ValueError("Resume checkpoint missing model_state_dict")
        model.load_state_dict(ck["model_state_dict"])
        if "step" in ck:
            start_step = int(ck["step"])
        print(f"Resumed from {args.resume} at step {start_step}")

    for step in range(start_step, args.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, k_steps=cfg.k_train)
        loss = out["loss"]

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % max(1, args.log_every) == 0:
            avg_steps = float(out["steps"].float().mean().item())
            avg_write = float(out["avg_write_strength"].mean().item())
            print(f"step {step+1}/{args.steps} | loss {loss.item():.4f} | avg_steps {avg_steps:.2f} | avg_write {avg_write:.4f}", flush=True)

        if (step + 1) % max(1, args.save_every) == 0:
            ckpt = {
                "step": step + 1,
                "config": cfg.__dict__,
                "model_state_dict": model.state_dict(),
            }
            torch.save(ckpt, out_dir / f"r3m_step_{step+1}.pt")

    # Final checkpoint
    torch.save(
        {"step": args.steps, "config": cfg.__dict__, "model_state_dict": model.state_dict()},
        out_dir / "r3m_final.pt",
    )

    # Quick generation sanity
    prompt = "Human: Hello!\nAssistant:"
    ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    gen = model.generate(
        ids,
        max_new_tokens=60,
        temperature=0.9,
        top_k=50,
        k_steps=cfg.k_train,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.15,
    )[0].tolist()
    text = tokenizer.decode(gen)
    (out_dir / "sample.txt").write_text(text, encoding="utf-8")
    print("\nSample saved to:", out_dir / "sample.txt")


if __name__ == "__main__":
    main()



