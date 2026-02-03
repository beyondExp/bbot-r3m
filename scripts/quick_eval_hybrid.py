#!/usr/bin/env python3
"""Quick eval for the Hybrid SSM model.

- Loads a checkpoint (.pt) produced by R3M/scripts/pretrain_hybrid_ssm.py
- Computes average loss / ppl on a small packed slice of a jsonl text dataset
- Runs tiny greedy generation for a few prompts
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Ensure R3M/ is on sys.path
import sys
R3M_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if R3M_ROOT not in sys.path:
    sys.path.insert(0, R3M_ROOT)

from r3m.models.hybrid_ssm import R3MHybridSSMConfig, R3MHybridSSMLM


def load_texts_from_jsonl(path: str, take: int) -> List[str]:
    texts: List[str] = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        t = r.get("text", None)
        if t is None:
            continue
        t = str(t).strip()
        if not t:
            continue
        texts.append(t)
        if take and len(texts) >= int(take):
            break
    return texts


class PackedTextDataset(Dataset):
    def __init__(self, *, tokenizer, texts: List[str], block_size: int, max_tokens: int, seed: int):
        super().__init__()
        self.block_size = int(block_size)
        eos = tokenizer.eos_token_id
        if eos is None:
            raise RuntimeError("Tokenizer has no eos_token_id; cannot pack dataset.")

        rnd = random.Random(int(seed))
        texts = list(texts)
        rnd.shuffle(texts)

        ids: List[int] = []
        i = 0
        while len(ids) < int(max_tokens):
            if not texts:
                break
            t = texts[i % len(texts)]
            i += 1
            tid = tokenizer(t, add_special_tokens=False).input_ids
            if not tid:
                continue
            ids.extend(tid)
            ids.append(int(eos))
            if len(ids) >= int(max_tokens):
                break

        if len(ids) < self.block_size + 2:
            raise RuntimeError("Not enough tokens to build eval dataset; decrease --seq-len or provide more text.")

        n_blocks = (len(ids) - 1) // self.block_size
        usable = n_blocks * self.block_size + 1
        ids = ids[:usable]
        self._ids = torch.tensor(ids, dtype=torch.long)
        self._n = int(n_blocks)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        i = int(idx) * self.block_size
        x = self._ids[i : i + self.block_size]
        y = self._ids[i : i + self.block_size]
        attn = torch.ones_like(x)
        return {"input_ids": x, "labels": y, "attention_mask": attn}


def collate(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


@torch.no_grad()
def greedy_generate(model: R3MHybridSSMLM, tok, prompt: str, *, max_new_tokens: int) -> str:
    model.eval()
    device = next(model.parameters()).device

    # Avoid long-text warnings by letting tokenizer accept long prompts
    tok.model_max_length = 10**9

    ids = tok(prompt, add_special_tokens=False).input_ids
    if not ids:
        ids = [tok.eos_token_id]

    for _ in range(int(max_new_tokens)):
        ctx = ids[-int(model.cfg.max_seq_len) :]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        attn = torch.ones_like(x)
        out = model(input_ids=x, attention_mask=attn, labels=None)
        logits = out["logits"]  # [1,T,V]
        next_id = int(torch.argmax(logits[0, -1, :]).item())
        ids.append(next_id)

    return tok.decode(ids)


@torch.no_grad()
def eval_loss(model: R3MHybridSSMLM, dl: DataLoader, *, device: torch.device, amp: str) -> float:
    model.eval()
    use_amp = device.type == "cuda" and amp != "off"
    amp_dtype = torch.float16 if amp == "fp16" else (torch.bfloat16 if amp == "bf16" else None)

    total = 0.0
    n = 0
    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"), dtype=amp_dtype, enabled=bool(use_amp)):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out["loss"]
        total += float(loss.item())
        n += 1
    return total / max(1, n)


@torch.no_grad()
def eval_loss_and_acc(model: R3MHybridSSMLM, dl: DataLoader, *, device: torch.device, amp: str) -> tuple[float, float]:
    """Returns (avg_loss, next_token_accuracy). Accuracy is computed on shifted labels, ignoring -100."""
    model.eval()
    use_amp = device.type == "cuda" and amp != "off"
    amp_dtype = torch.float16 if amp == "fp16" else (torch.bfloat16 if amp == "bf16" else None)

    loss_total = 0.0
    n_batches = 0
    correct = 0
    total = 0

    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"), dtype=amp_dtype, enabled=bool(use_amp)):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = out["logits"]
            loss = out["loss"]

        # Shift: position t predicts label at t+1
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        pred = torch.argmax(shift_logits, dim=-1)
        mask = shift_labels.ne(-100)
        correct += int((pred.eq(shift_labels) & mask).sum().item())
        total += int(mask.sum().item())

        loss_total += float(loss.item())
        n_batches += 1

    avg_loss = loss_total / max(1, n_batches)
    acc = float(correct) / float(max(1, total))
    return avg_loss, acc


def main() -> None:
    # Windows terminals often default to cp1252; ensure we can print arbitrary model text safely.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="R3M/outputs/stageA_fineweb_100k_10k/model_final.pt")
    ap.add_argument("--tokenizer", type=str, default="gpt2")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")

    ap.add_argument("--eval-jsonl", type=str, default="R3M/data_cache/fineweb_100k.jsonl")
    ap.add_argument("--eval-take-texts", type=int, default=2000)
    ap.add_argument("--eval-max-pack-tokens", type=int, default=400000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)

    ap.add_argument("--gen-max-new", type=int, default=80)
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    ckpt = torch.load(str(args.ckpt), map_location="cpu")
    cfg_d = ckpt.get("config", None)
    if cfg_d is None:
        raise RuntimeError("Checkpoint missing config")
    cfg = R3MHybridSSMConfig(**cfg_d)
    model = R3MHybridSSMLM(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)

    tok = AutoTokenizer.from_pretrained(str(args.tokenizer), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.model_max_length = 10**9

    # Eval loss
    if str(args.eval_jsonl).strip() and Path(str(args.eval_jsonl)).exists():
        texts = load_texts_from_jsonl(str(args.eval_jsonl), take=int(args.eval_take_texts))
        ds = PackedTextDataset(
            tokenizer=tok,
            texts=texts,
            block_size=int(args.seq_len),
            max_tokens=int(args.eval_max_pack_tokens),
            seed=0,
        )
        dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, drop_last=False, collate_fn=collate)
        avg_loss, acc = eval_loss_and_acc(model, dl, device=device, amp=str(args.amp))
        ppl = math.exp(min(20.0, avg_loss))
        print(f"[eval] avg_loss={avg_loss:.8f} ppl~{ppl:.4f} acc={acc:.6f} | blocks={len(ds)}", flush=True)
    else:
        print("[eval] skipped (no eval jsonl found)", flush=True)

    # Generation
    prompts = [
        "User: Hello!\nAssistant:",
        "User: What is 2+2?\nAssistant:",
        "User: Write a short haiku about winter.\nAssistant:",
        "User: Explain what a transformer is.\nAssistant:",
    ]
    for p in prompts:
        out = greedy_generate(model, tok, p, max_new_tokens=int(args.gen_max_new))
        print("\n--- prompt ---\n" + p)
        print("--- gen ---\n" + out)


if __name__ == "__main__":
    main()

