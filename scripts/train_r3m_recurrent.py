# Ensure R3M/ is on sys.path\nimport os\nimport sys\nR3M_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))\nif R3M_ROOT not in sys.path:\n    sys.path.insert(0, R3M_ROOT)\n\n#!/usr/bin/env python3
"""
Train Fix-B RÂ³M recurrent LM (causal-by-construction).

This avoids the previous leakage mode by predicting token_{t+1} from a state
that only saw tokens <= t.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Ensure repo root is on sys.path so `import src...` works when running from /scripts
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from r3m.models.recurrent import R3MRecurrentConfig, R3MRecurrentLM


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
                parts.append(f"{role}: {content}" if role else content)
            return "\n".join(parts)
        return str(ex)

    def __getitem__(self, idx):
        # Add an explicit end-of-example token so the model learns to stop.
        # Without this, small models often "chain" into other memorized templates.
        text = self._to_text(self.examples[idx]).strip() + (self.tokenizer.eos_token or "")
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
        labels[attention_mask == 0] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs/r3m_recurrent")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--k-train", type=int, default=4)
    ap.add_argument("--model-dim", type=int, default=512)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--write-rate-target", type=float, default=None, help="Target average episodic write strength (e.g. 0.2). If set, adds a penalty.")
    ap.add_argument("--write-rate-lambda", type=float, default=0.0, help="Weight for write-rate penalty (e.g. 1.0).")
    ap.add_argument("--write-topk", type=int, default=0, help="Episodic write routing: 0=soft, 1=top-1, k=top-k")
    ap.add_argument("--no-write-straight-through", action="store_true", help="Disable straight-through gradients for top-1 routing")
    ap.add_argument("--adaptive-halting", action="store_true", help="Enable ACT-style adaptive halting in thinking loop")
    ap.add_argument("--halting-epsilon", type=float, default=1e-2)
    ap.add_argument("--ponder-lambda", type=float, default=0.0, help="Weight for ponder cost (encourages fewer thinking steps)")
    ap.add_argument("--moe-enabled", action="store_true", help="Enable MoE feedforward in the recurrent core")
    ap.add_argument("--moe-num-experts", type=int, default=4)
    ap.add_argument("--moe-router", type=str, default="topk", choices=["topk", "sinkhorn"], help="MoE router type; sinkhorn is mHC-like balanced routing")
    ap.add_argument("--moe-top-k", type=int, default=1)
    ap.add_argument("--moe-sinkhorn-iters", type=int, default=8)
    ap.add_argument("--moe-temperature", type=float, default=1.0)
    ap.add_argument("--moe-aux-lambda", type=float, default=0.0, help="Weight for MoE load-balancing auxiliary loss")
    ap.add_argument("--mix-entropy-lambda", type=float, default=0.0, help="Encourage non-degenerate base/mem convex mixing (maximize entropy).")
    ap.add_argument("--mem-utility-lambda", type=float, default=0.0, help="Penalty if memory makes loss worse than no-memory baseline (hinge).")
    ap.add_argument("--mem-utility-margin", type=float, default=0.0, help="Hinge margin for mem-utility loss.")
    ap.add_argument("--mem-utility-every", type=int, default=0, help="Compute no-memory baseline every N steps (0=off).")
    ap.add_argument("--mhc-enabled", action="store_true", help="Enable minimal mHC stream mixing inside thinking loop (not MoE)")
    ap.add_argument("--mhc-n", type=int, default=4, help="Number of mHC streams for thinking-loop residual")
    ap.add_argument("--mhc-alpha-init", type=float, default=0.01, help="Initial mix strength toward DS matrix; small preserves identity mapping")
    ap.add_argument("--mhc-sinkhorn-iters", type=int, default=12)
    ap.add_argument("--mhc-temperature", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    ds = TextJsonDataset(args.data, tok, max_length=args.max_len, limit=args.limit)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    cfg = R3MRecurrentConfig(
        vocab_size=len(tok),
        d_model=args.model_dim,
        max_seq_len=args.max_len,
        k_train=args.k_train,
        k_max=max(32, args.k_train),
        episodic_enabled=True,
        episodic_slots=128,
        neuromodulator_enabled=True,
        write_topk=int(args.write_topk),
        write_straight_through=not bool(args.no_write_straight_through),
        adaptive_halting=bool(args.adaptive_halting),
        halting_epsilon=float(args.halting_epsilon),
        ponder_lambda=float(args.ponder_lambda),
        moe_enabled=bool(args.moe_enabled),
        moe_num_experts=int(args.moe_num_experts),
        moe_router=str(args.moe_router),
        moe_top_k=int(args.moe_top_k),
        moe_sinkhorn_iters=int(args.moe_sinkhorn_iters),
        moe_temperature=float(args.moe_temperature),
        mix_entropy_lambda=float(args.mix_entropy_lambda),
        mhc_enabled=bool(args.mhc_enabled),
        mhc_n=int(args.mhc_n),
        mhc_alpha_init=float(args.mhc_alpha_init),
        mhc_sinkhorn_iters=int(args.mhc_sinkhorn_iters),
        mhc_temperature=float(args.mhc_temperature),
    )
    model = R3MRecurrentLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        start_step = int(ck.get("step", 0))
        print(f"Resumed from {args.resume} at step {start_step}", flush=True)

    model.train()
    it = iter(dl)

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
        base_loss = loss.detach()
        write_reg = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        ponder_reg = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        moe_reg = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        mix_reg = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        mem_util_reg = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        mem_adv = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

        # Optional: discourage write-gate saturation (avg_write -> 1.0)
        if args.write_rate_target is not None and float(args.write_rate_lambda) > 0.0:
            avg_write = out["avg_write_strength"].mean()
            tgt = torch.tensor(float(args.write_rate_target), device=avg_write.device, dtype=avg_write.dtype)
            write_reg = float(args.write_rate_lambda) * (avg_write - tgt).pow(2)
            loss = loss + write_reg

        # Optional: encourage fewer adaptive-halting steps (ponder cost)
        if cfg.adaptive_halting and float(cfg.ponder_lambda) > 0.0:
            ponder_reg = float(cfg.ponder_lambda) * out["ponder_cost"].mean()
            loss = loss + ponder_reg

        if cfg.moe_enabled and float(args.moe_aux_lambda) > 0.0:
            moe_reg = float(args.moe_aux_lambda) * out["moe_load_balance"]
            loss = loss + moe_reg

        # Optional: discourage degenerate mixing (always base or always mem)
        if float(args.mix_entropy_lambda) > 0.0:
            mix_reg = -float(args.mix_entropy_lambda) * out.get("mix_entropy", torch.tensor(0.0, device=loss.device, dtype=loss.dtype))
            loss = loss + mix_reg

        # Optional: memory utility hinge (compute a no-memory baseline occasionally)
        if float(args.mem_utility_lambda) > 0.0 and int(args.mem_utility_every) > 0 and ((step + 1) % int(args.mem_utility_every) == 0):
            out_no = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                k_steps=cfg.k_train,
                episodic_enabled_override=False,
            )
            loss_no = out_no["loss"].detach()
            mem_adv = (loss_no - out["loss"]).detach()
            margin = torch.tensor(float(args.mem_utility_margin), device=loss.device, dtype=loss.dtype)
            mem_util_reg = float(args.mem_utility_lambda) * F.relu(out["loss"] - loss_no + margin)
            loss = loss + mem_util_reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % max(1, args.log_every) == 0:
            avg_write = float(out["avg_write_strength"].mean().item())
            ponder = float(out.get("ponder_cost", torch.tensor(0.0)).mean().item())
            moe_lb = float(out.get("moe_load_balance", torch.tensor(0.0)).item())
            moe_ent = float(out.get("moe_entropy", torch.tensor(0.0)).item())
            mix_ent = float(out.get("mix_entropy", torch.tensor(0.0)).item())
            mhc_alpha = float(out.get("mhc_alpha", torch.tensor(0.0)).item())
            print(
                f"step {step+1}/{args.steps} | loss {loss.item():.4f} "
                f"(base {base_loss.item():.4f} + write_reg {write_reg.item():.4f} + ponder_reg {ponder_reg.item():.4f} + moe_reg {moe_reg.item():.4f} + mix_reg {mix_reg.item():.4f} + mem_util {mem_util_reg.item():.4f}) "
                f"| avg_write {avg_write:.4f} | ponder {ponder:.4f} | moe_lb {moe_lb:.4f} | moe_ent {moe_ent:.4f} | mix_ent {mix_ent:.4f} | mem_adv {mem_adv.item():.4f} | mhc_alpha {mhc_alpha:.4f}",
                flush=True,
            )

        if (step + 1) % max(1, args.save_every) == 0:
            torch.save(
                {"step": step + 1, "config": cfg.__dict__, "model_state_dict": model.state_dict()},
                out_dir / f"r3m_rec_step_{step+1}.pt",
            )

    # final
    torch.save({"step": args.steps, "config": cfg.__dict__, "model_state_dict": model.state_dict()}, out_dir / "r3m_rec_final.pt")

    # UTF-8 sample write (avoid console encoding issues)
    prompt = "Human: What is 12 - 1?\nAssistant:"
    ids = torch.tensor([tok.encode(prompt)], device=device)
    gen = model.generate(
        ids,
        max_new_tokens=80,
        temperature=0.8,
        top_k=50,
        k_steps=cfg.k_train,
        eos_token_id=tok.eos_token_id,
        repetition_penalty=1.15,
    )[0].tolist()
    text = tok.decode(gen)
    (out_dir / "sample.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()



