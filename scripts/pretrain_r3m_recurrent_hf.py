# Ensure R3M/ is on sys.path\nimport os\nimport sys\nR3M_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))\nif R3M_ROOT not in sys.path:\n    sys.path.insert(0, R3M_ROOT)\n\n#!/usr/bin/env python3
"""
HF-streaming pretraining for R3MRecurrentLM.

Why this script exists:
- Your current `combined_conversational_dataset.jsonl` is mostly tiny, templated Q/A.
  Small models learn to "template-chain" (sequence -> train distance -> shoes...) which looks odd.
- For real language ability, we want large-scale pretraining text (Apertus / Nemotron / etc.).
- These corpora are huge; we must support *streaming* so you can start training without
  downloading terabytes.

Usage (examples):
  python scripts\\pretrain_r3m_recurrent_hf.py ^
    --dataset <hf_dataset_id> --split train ^
    --text-field text --streaming ^
    --out outputs\\r3m_rec_pretrain_nemotron_sample ^
    --steps 2000 --batch-size 8 --max-len 256 --model-dim 512 --k-train 2

Notes:
- This is "continued pretraining" from scratch weights by default; you can resume from a ckpt.
- Tokenizer defaults to GPT-2 for now (keeps vocab stable with the rest of this repo).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

# Ensure repo root is on sys.path so `import src...` works when running from /scripts
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from r3m.models.recurrent import R3MRecurrentConfig, R3MRecurrentLM


def _try_import_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Hugging Face `datasets` is required for HF streaming.\n"
            "Install with: pip install datasets\n"
            f"Import error: {e}"
        )
    return load_dataset


def _extract_text(example: Dict[str, Any], text_field: str) -> Optional[str]:
    if text_field != "auto":
        v = example.get(text_field)
        return v if isinstance(v, str) and v.strip() else None

    # Common pretraining field names
    for k in ["text", "content", "document", "doc", "article", "completion"]:
        v = example.get(k)
        if isinstance(v, str) and v.strip():
            return v

    # Chat-style messages
    msgs = example.get("messages")
    if isinstance(msgs, list) and msgs:
        parts: List[str] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "").strip()
            content = str(m.get("content") or "").strip()
            if not content:
                continue
            if role:
                parts.append(f"{role}: {content}")
            else:
                parts.append(content)
        joined = "\n".join(parts).strip()
        return joined if joined else None

    return None


class PackedTokenStream(IterableDataset):
    """
    Streams text from an HF dataset and packs it into fixed-length token blocks.
    Produces *already causal* examples:
      input_ids: [T]
      attention_mask: [T] (all ones)
      labels: [T] (identical to input_ids; no padding)
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer,
        max_len: int,
        streaming: bool,
        text_field: str = "auto",
        dataset_config: Optional[str] = None,
        seed: int = 0,
        shuffle_buffer: int = 10_000,
        add_eos_between_docs: bool = True,
        take_examples: Optional[int] = None,
        hf_read_timeout_s: Optional[float] = None,
        hf_connect_timeout_s: Optional[float] = None,
        hf_retries: int = 10,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.streaming = bool(streaming)
        self.text_field = text_field
        self.seed = int(seed)
        self.shuffle_buffer = int(shuffle_buffer)
        self.add_eos_between_docs = bool(add_eos_between_docs)
        self.take_examples = int(take_examples) if take_examples is not None else None
        self.hf_read_timeout_s = float(hf_read_timeout_s) if hf_read_timeout_s is not None else None
        self.hf_connect_timeout_s = float(hf_connect_timeout_s) if hf_connect_timeout_s is not None else None
        self.hf_retries = int(hf_retries)

        self._load_dataset = _try_import_datasets()

        if self.max_len < 8:
            raise ValueError("max_len is too small; use at least 8.")

    def _iter_examples(self) -> Iterable[Dict[str, Any]]:
        # Configure HF hub timeouts via env vars (huggingface_hub respects these).
        if self.hf_read_timeout_s is not None and self.hf_read_timeout_s > 0:
            os.environ["HF_HUB_READ_TIMEOUT"] = str(float(self.hf_read_timeout_s))
        if self.hf_connect_timeout_s is not None and self.hf_connect_timeout_s > 0:
            os.environ["HF_HUB_CONNECT_TIMEOUT"] = str(float(self.hf_connect_timeout_s))

        # Robust load_dataset with retries/backoff (HF streaming sometimes times out on metadata fetch).
        last_err: Optional[Exception] = None
        retries = max(1, int(self.hf_retries))
        for attempt in range(retries):
            try:
                print(
                    f"[hf] loading dataset={self.dataset_name}"
                    + (f" config={self.dataset_config}" if self.dataset_config else "")
                    + f" split={self.split} streaming={self.streaming} (attempt {attempt+1}/{retries})",
                    flush=True,
                )
                if self.dataset_config:
                    ds = self._load_dataset(self.dataset_name, self.dataset_config, split=self.split, streaming=self.streaming)
                else:
                    ds = self._load_dataset(self.dataset_name, split=self.split, streaming=self.streaming)
                last_err = None
                print("[hf] dataset loaded", flush=True)
                break
            except Exception as e:
                last_err = e
                wait_s = min(60.0, 2.0 ** attempt)
                print(f"[hf] load_dataset failed (attempt {attempt+1}/{retries}): {e} | retrying in {wait_s:.1f}s", flush=True)
                time.sleep(wait_s)
        if last_err is not None:
            raise last_err

        # Streaming datasets support shuffle(buffer_size=...)
        try:
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        except Exception:
            pass

        n = 0
        for ex in ds:
            yield ex
            n += 1
            if self.take_examples is not None and n >= self.take_examples:
                break

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rng = random.Random(self.seed + int(torch.utils.data.get_worker_info().id) if torch.utils.data.get_worker_info() else self.seed)  # type: ignore

        buf: List[int] = []
        eos_id = self.tokenizer.eos_token_id

        for ex in self._iter_examples():
            text = _extract_text(ex, text_field=self.text_field)
            if not text:
                continue

            # Lightweight cleanup (avoid huge whitespace runs)
            text = " ".join(text.split())
            if not text:
                continue

            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue

            buf.extend(ids)
            if self.add_eos_between_docs and eos_id is not None:
                buf.append(int(eos_id))

            # Emit as many full blocks as possible
            while len(buf) >= self.max_len:
                block = buf[: self.max_len]
                buf = buf[self.max_len :]

                input_ids = torch.tensor(block, dtype=torch.long)
                attn = torch.ones(self.max_len, dtype=torch.long)
                labels = input_ids.clone()

                yield {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

            # Occasionally drop buffer if it grows too large (defensive)
            if len(buf) > 10 * self.max_len:
                # keep a tail to preserve continuity a bit
                keep = rng.randint(self.max_len, 2 * self.max_len)
                buf = buf[-keep:]


def preview_dataset(
    dataset_name: str,
    split: str,
    dataset_config: Optional[str],
    text_field: str,
    streaming: bool,
    n: int,
) -> None:
    load_dataset = _try_import_datasets()
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)
    else:
        ds = load_dataset(dataset_name, split=split, streaming=streaming)

    print(f"Preview: dataset={dataset_name} split={split} streaming={streaming} text_field={text_field}")
    shown = 0
    for ex in ds:
        text = _extract_text(ex, text_field=text_field)
        if not text:
            continue
        text = text.strip().replace("\n", "\\n")
        print(f"- {text[:240]}{'...' if len(text) > 240 else ''}")
        shown += 1
        if shown >= n:
            break
    if shown == 0:
        print("No usable text examples found. Try --text-field <fieldname> instead of auto.")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", required=True, help="HF dataset ID (e.g., org/name)")
    ap.add_argument("--dataset-config", type=str, default=None, help="Optional HF dataset config name")
    ap.add_argument("--split", type=str, default="train", help="Split name (e.g., train)")
    ap.add_argument("--text-field", type=str, default="auto", help="Text field name or 'auto'")
    ap.add_argument("--streaming", action="store_true", help="Use streaming mode (recommended for large datasets)")
    ap.add_argument("--take-examples", type=int, default=None, help="Limit number of HF examples (debug/smoke)")

    ap.add_argument("--out", type=str, default="outputs/r3m_rec_pretrain_hf")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--k-train", type=int, default=2)
    ap.add_argument("--model-dim", type=int, default=512)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--max-hours", type=float, default=None, help="Optional wall-clock limit; stop after this many hours.")
    ap.add_argument("--hf-read-timeout", type=float, default=60.0, help="HF hub read timeout seconds (helps avoid ReadTimeout)")
    ap.add_argument("--hf-connect-timeout", type=float, default=10.0, help="HF hub connect timeout seconds")
    ap.add_argument("--hf-retries", type=int, default=10, help="Retry count for HF load_dataset() on transient failures")
    ap.add_argument("--write-rate-target", type=float, default=None, help="Target average episodic write strength (e.g. 0.2). If set, adds a penalty.")
    ap.add_argument("--write-rate-lambda", type=float, default=0.0, help="Weight for write-rate penalty (e.g. 0.5).")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-buffer", type=int, default=10_000)
    ap.add_argument("--no-eos-between-docs", action="store_true")

    ap.add_argument("--tokenizer", type=str, default="gpt2")
    ap.add_argument("--amp", type=str, default="off", choices=["off", "fp16", "bf16"], help="Mixed precision autocast mode (CUDA only).")

    # Architecture knobs
    ap.add_argument("--write-topk", type=int, default=0, help="Episodic write routing: 0=soft, 1=top-1, k=top-k")
    ap.add_argument("--no-write-straight-through", action="store_true", help="Disable straight-through gradients for top-1 routing")
    ap.add_argument("--adaptive-halting", action="store_true", help="Enable ACT-style adaptive halting in thinking loop")
    ap.add_argument("--halting-epsilon", type=float, default=1e-2)
    ap.add_argument("--ponder-lambda", type=float, default=0.0, help="Weight for ponder cost (encourages fewer thinking steps)")
    ap.add_argument("--moe-enabled", action="store_true", help="Enable MoE feedforward in the recurrent core")
    ap.add_argument("--moe-num-experts", type=int, default=4)
    ap.add_argument("--moe-router", type=str, default="topk", choices=["topk", "sinkhorn"], help="MoE router type; sinkhorn is mHC-like balanced routing")
    ap.add_argument("--moe-top-k", type=int, default=1, help="Experts per token (sparsification)")
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

    ap.add_argument("--preview", action="store_true", help="Only preview N samples and exit")
    ap.add_argument("--preview-n", type=int, default=5)

    args = ap.parse_args()

    if args.preview:
        preview_dataset(
            dataset_name=args.dataset,
            split=args.split,
            dataset_config=args.dataset_config,
            text_field=args.text_field,
            streaming=bool(args.streaming),
            n=int(args.preview_n),
        )
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and str(args.amp) != "off"
    amp_dtype = torch.float16 if str(args.amp) == "fp16" else (torch.bfloat16 if str(args.amp) == "bf16" else None)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and str(args.amp) == "fp16"))

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # We intentionally tokenize long documents and then pack into max_len blocks.
    # Suppress the confusing transformers warning about exceeding model_max_length.
    tok.model_max_length = int(10**9)

    ds = PackedTokenStream(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        tokenizer=tok,
        max_len=args.max_len,
        streaming=bool(args.streaming),
        text_field=args.text_field,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        add_eos_between_docs=not bool(args.no_eos_between_docs),
        take_examples=args.take_examples,
        hf_read_timeout_s=float(args.hf_read_timeout),
        hf_connect_timeout_s=float(args.hf_connect_timeout),
        hf_retries=int(args.hf_retries),
    )

    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    cfg = R3MRecurrentConfig(
        vocab_size=len(tok),
        d_model=int(args.model_dim),
        max_seq_len=int(args.max_len),
        k_train=int(args.k_train),
        k_max=max(32, int(args.k_train)),
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
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.1)

    print(
        f"Starting pretrain: dataset={args.dataset}"
        + (f" config={args.dataset_config}" if args.dataset_config else "")
        + f" split={args.split} streaming={bool(args.streaming)} text_field={args.text_field}",
        flush=True,
    )
    print(
        f"Model: d_model={cfg.d_model} max_len={cfg.max_seq_len} k_train={cfg.k_train} batch_size={int(args.batch_size)} device={device}",
        flush=True,
    )
    if use_amp:
        print(f"AMP: enabled ({args.amp})", flush=True)
    print(
        f"Arch: mhc={bool(cfg.mhc_enabled)} moe={bool(cfg.moe_enabled)} adaptive_halting={bool(cfg.adaptive_halting)} write_topk={int(cfg.write_topk)}",
        flush=True,
    )

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        start_step = int(ck.get("step", 0))
        print(f"Resumed from {args.resume} at step {start_step}", flush=True)

    model.train()
    it = iter(dl)

    start_time = time.time()
    last_step = int(start_step)
    max_hours = None
    if args.max_hours is not None and float(args.max_hours) > 0.0:
        max_hours = float(args.max_hours)

    first_batch_logged = False
    for step in range(start_step, int(args.steps)):
        if max_hours is not None and (time.time() - start_time) >= max_hours * 3600.0:
            print(f"Reached max-hours={max_hours:.2f}. Stopping at step {step}.", flush=True)
            break
        try:
            if not first_batch_logged:
                print("Fetching first batch...", flush=True)
            batch = next(it)
            if not first_batch_logged:
                print("First batch fetched.", flush=True)
                first_batch_logged = True
        except StopIteration:
            # IterableDataset (especially streaming) can be exhausted.
            # Restarting the iterator gives us "infinite" training for a fixed step budget.
            it = iter(dl)
            batch = next(it)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=bool(use_amp)):
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
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=bool(use_amp)):
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
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        last_step = int(step + 1)

        if (step + 1) % max(1, int(args.log_every)) == 0:
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

        if (step + 1) % max(1, int(args.save_every)) == 0:
            torch.save(
                {"step": step + 1, "config": cfg.__dict__, "model_state_dict": model.state_dict()},
                out_dir / f"r3m_rec_step_{step+1}.pt",
            )

    torch.save({"step": int(last_step), "config": cfg.__dict__, "model_state_dict": model.state_dict()}, out_dir / "r3m_rec_final.pt")


if __name__ == "__main__":
    main()



