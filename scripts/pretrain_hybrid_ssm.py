#!/usr/bin/env python3
"""Pretrain R3M Hybrid SSM Base LM (from scratch).

Modes:
- tiny: built-in tiny corpus (smoke tests)
- jsonl: load ALL texts from a jsonl {"text": ...} (only for small files)
- jsonl_stream: stream a large jsonl line-by-line (recommended for millions of rows)

Notes:
- Uses bf16/fp16 AMP on CUDA.
- Prints CUDA memory usage at log intervals so you can tune batch/seq to fill VRAM.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, List

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoTokenizer

# Ensure R3M/ is on sys.path
R3M_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if R3M_ROOT not in sys.path:
    sys.path.insert(0, R3M_ROOT)

from r3m.models.hybrid_ssm import R3MHybridSSMConfig, R3MHybridSSMLM


def build_tiny_corpus() -> List[str]:
    return [
        "User: Hello!\nAssistant: Hi! How can I help you today?\n",
        "User: What is 2+2?\nAssistant: 2+2 is 4.\n",
        "User: Explain what a transformer is.\nAssistant: A transformer is a neural network that uses attention to process sequences.\n",
        "In the beginning, there was only a quiet page and the promise of words.\n",
        "The quick brown fox jumps over the lazy dog.\n",
        "To be, or not to be, that is the question.\n",
        "Numbers can be added, multiplied, and compared.\n",
    ]


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
    """Packs texts into a token stream (in-memory); good for tiny/small jsonl."""

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
        # Cycle through texts if needed (tiny corpora)
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
            raise RuntimeError("Not enough tokens to build dataset; decrease --seq-len or provide more text.")

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
        return {"input_ids": x, "labels": y}


class PackedJsonlStream(IterableDataset):
    """Streams a jsonl file line-by-line and yields fixed-length token blocks."""

    def __init__(
        self,
        *,
        jsonl_path: str,
        tokenizer,
        block_size: int,
        seed: int,
        repeat: bool,
        max_lines: int,
    ):
        super().__init__()
        self.path = str(jsonl_path)
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.repeat = bool(repeat)
        self.max_lines = int(max_lines)

        eos = tokenizer.eos_token_id
        if eos is None:
            raise RuntimeError("Tokenizer has no eos_token_id; cannot stream pack dataset.")
        self.eos = int(eos)

    def _iter_texts(self) -> Iterator[str]:
        rng = random.Random(self.seed)
        _ = rng  # reserved for future shuffling

        path = Path(self.path)
        if not path.exists():
            raise RuntimeError(f"jsonl not found: {path}")

        while True:
            n = 0
            with path.open("r", encoding="utf-8") as f:
                for ln in f:
                    if self.max_lines > 0 and n >= self.max_lines:
                        break
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        r = json.loads(ln)
                    except Exception:
                        # tolerate partially-written lines if the file is being appended
                        continue
                    t = r.get("text", None)
                    if not t:
                        continue
                    t = str(t).strip()
                    if not t:
                        continue
                    yield t
                    n += 1
            if not self.repeat:
                break

    def __iter__(self):
        buf: List[int] = []
        for t in self._iter_texts():
            tid = self.tokenizer(t, add_special_tokens=False).input_ids
            if not tid:
                continue
            buf.extend(tid)
            buf.append(self.eos)
            while len(buf) >= (self.block_size + 1):
                x = torch.tensor(buf[: self.block_size], dtype=torch.long)
                y = torch.tensor(buf[: self.block_size], dtype=torch.long)
                buf = buf[self.block_size :]
                yield {"input_ids": x, "labels": y}


def collate(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    return {"input_ids": input_ids, "labels": labels}


def parse_int_list(s: str) -> List[int]:
    s = str(s).strip()
    if not s:
        return []
    return [int(p.strip()) for p in s.split(",") if p.strip()]



@torch.no_grad()
def eval_loss(model, dl, *, device: torch.device, amp: str) -> float:
    model.eval()
    use_amp = device.type == 'cuda' and amp != 'off'
    amp_dtype = torch.float16 if amp == 'fp16' else (torch.bfloat16 if amp == 'bf16' else None)
    total = 0.0
    n = 0
    for batch in dl:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        with torch.autocast(device_type=('cuda' if device.type=='cuda' else 'cpu'), dtype=amp_dtype, enabled=bool(use_amp)):
            out = model(input_ids=input_ids, attention_mask=None, labels=labels)
            loss = out['loss']
        total += float(loss.item())
        n += 1
    model.train()
    return total / max(1, n)


def cuda_mem_str(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return ""
    try:
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserv = torch.cuda.memory_reserved() / (1024**3)
        peak = torch.cuda.max_memory_reserved() / (1024**3)
        return f" | cuda_gb alloc={alloc:.2f} reserv={reserv:.2f} peak_reserv={peak:.2f}"
    except Exception:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="R3M/outputs/r3m_hybrid_ssm_50m")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--data-mode", choices=["tiny", "jsonl", "jsonl_stream"], default="tiny")
    ap.add_argument("--tokenizer", type=str, default="gpt2")
    ap.add_argument("--jsonl", type=str, default="")
    ap.add_argument("--take-texts", type=int, default=20000, help="jsonl mode only")
    ap.add_argument("--max-pack-tokens", type=int, default=2_000_000, help="jsonl/tiny mode only")
    ap.add_argument("--stream-repeat", action="store_true", help="jsonl_stream: loop the file forever")
    ap.add_argument("--stream-max-lines", type=int, default=0, help="jsonl_stream: cap lines per pass (0=all)")

    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)

    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=200)

    # Eval (optional)
    ap.add_argument("--eval-jsonl", type=str, default="")
    ap.add_argument("--eval-every", type=int, default=0, help="Run eval every N steps (0=off)")
    ap.add_argument("--eval-take-texts", type=int, default=2000)
    ap.add_argument("--eval-max-pack-tokens", type=int, default=400000)
    ap.add_argument("--eval-seq-len", type=int, default=256)
    ap.add_argument("--eval-batch-size", type=int, default=8)

    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--attn-layers", type=str, default="2,7")
    ap.add_argument("--ssm-expand", type=int, default=2)
    ap.add_argument("--ssm-kernel", type=int, default=3)
    ap.add_argument("--ffn-mult", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and str(args.amp) != "off"
    amp_dtype = torch.float16 if str(args.amp) == "fp16" else (torch.bfloat16 if str(args.amp) == "bf16" else None)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    tok = AutoTokenizer.from_pretrained(str(args.tokenizer), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.model_max_length = 10**9

    if str(args.data_mode) == "jsonl":
        if not str(args.jsonl).strip():
            raise RuntimeError("--jsonl is required when --data-mode jsonl")
        texts = load_texts_from_jsonl(str(args.jsonl), take=int(args.take_texts))
        if not texts:
            raise RuntimeError("No texts loaded from jsonl.")
        ds = PackedTextDataset(
            tokenizer=tok,
            texts=texts,
            block_size=int(args.seq_len),
            max_tokens=int(args.max_pack_tokens),
            seed=int(args.seed),
        )
        dl = DataLoader(
            ds,
            batch_size=int(args.batch_size),
            shuffle=True,
            drop_last=False,
            collate_fn=collate,
            pin_memory=(device.type == "cuda"),
        )
    elif str(args.data_mode) == "jsonl_stream":
        if not str(args.jsonl).strip():
            raise RuntimeError("--jsonl is required when --data-mode jsonl_stream")
        ds = PackedJsonlStream(
            jsonl_path=str(args.jsonl),
            tokenizer=tok,
            block_size=int(args.seq_len),
            seed=int(args.seed),
            repeat=bool(args.stream_repeat),
            max_lines=int(args.stream_max_lines),
        )
        dl = DataLoader(
            ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            drop_last=False,
            collate_fn=collate,
            pin_memory=(device.type == "cuda"),
        )
    else:
        texts = build_tiny_corpus()
        ds = PackedTextDataset(
            tokenizer=tok,
            texts=texts,
            block_size=int(args.seq_len),
            max_tokens=int(args.max_pack_tokens),
            seed=int(args.seed),
        )
        dl = DataLoader(
            ds,
            batch_size=int(args.batch_size),
            shuffle=True,
            drop_last=False,
            collate_fn=collate,
            pin_memory=(device.type == "cuda"),
        )

    cfg = R3MHybridSSMConfig(
        vocab_size=int(tok.vocab_size),
        max_seq_len=int(args.max_seq_len),
        d_model=int(args.d_model),
        n_layers=int(args.layers),
        attention_layers=tuple(parse_int_list(str(args.attn_layers))),
        n_heads=int(args.heads),
        ssm_expand=int(args.ssm_expand),
        ssm_conv_kernel=int(args.ssm_kernel),
        ffn_mult=int(args.ffn_mult),
        dropout=float(args.dropout),
    )

    model = R3MHybridSSMLM(cfg).to(device)
    (out_dir / "model_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(
        f"[model] params: {model.num_parameters()/1e6:.2f}M | vocab={cfg.vocab_size} d={cfg.d_model} L={cfg.n_layers}",
        flush=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    step = 0
    t0 = time.time()
    dl_iter = iter(dl)
    while step < int(args.steps):
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(int(args.grad_accum)):
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dl)
                batch = next(dl_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=("cuda" if device.type == "cuda" else "cpu"),
                dtype=amp_dtype,
                enabled=bool(use_amp),
            ):
                out = model(input_ids=input_ids, attention_mask=None, labels=labels)
                loss = out["loss"]
                if loss is None:
                    raise RuntimeError("Model did not return loss.")
                loss = loss / max(1, int(args.grad_accum))

            loss.backward()
            total_loss += float(loss.item())

        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        opt.step()

        step += 1
        if step % max(1, int(args.log_every)) == 0:
            dt = time.time() - t0
            sps = step / max(1e-9, dt)
            msg = f"step {step}/{args.steps} | loss {total_loss:.4f} | {sps:.2f} steps/s{cuda_mem_str(device)}"
            print(msg, flush=True)

            # Optional held-out eval
            if int(args.eval_every) > 0 and str(args.eval_jsonl).strip() and (step % int(args.eval_every) == 0):
                try:
                    ev_texts = load_texts_from_jsonl(str(args.eval_jsonl), take=int(args.eval_take_texts))
                    ev_ds = PackedTextDataset(tokenizer=tok, texts=ev_texts, block_size=int(args.eval_seq_len), max_tokens=int(args.eval_max_pack_tokens), seed=123)
                    ev_dl = DataLoader(ev_ds, batch_size=int(args.eval_batch_size), shuffle=False, drop_last=False, collate_fn=collate, pin_memory=(device.type=='cuda'))
                    ev_loss = eval_loss(model, ev_dl, device=device, amp=str(args.amp))
                    ev_ppl = math.exp(min(20.0, ev_loss))
                    print(f"[eval] step={step} loss={ev_loss:.4f} ppl~={ev_ppl:.2f} blocks={len(ev_ds)}", flush=True)
                except Exception as e:
                    print(f"[eval] failed: {{e}}", flush=True)
            if device.type == "cuda":
                # reset so peak reflects the next interval
                torch.cuda.reset_peak_memory_stats()

        if int(args.save_every) > 0 and step % int(args.save_every) == 0:
            ckpt = {
                "global_step": int(step),
                "config": asdict(cfg),
                "tokenizer": str(args.tokenizer),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
            }
            ckpt_path = out_dir / f"ckpt_step_{step}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"[ckpt] wrote {ckpt_path}", flush=True)

    final_path = out_dir / "model_final.pt"
    torch.save(
        {"global_step": int(step), "config": asdict(cfg), "tokenizer": str(args.tokenizer), "model_state_dict": model.state_dict()},
        final_path,
    )
    print(f"wrote {final_path}", flush=True)


if __name__ == "__main__":
    main()

