# Ensure R3M/ is on sys.path\nimport os\nimport sys\nR3M_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))\nif R3M_ROOT not in sys.path:\n    sys.path.insert(0, R3M_ROOT)\n\n#!/usr/bin/env python3
"""
Generate samples from an R3MRecurrentLM checkpoint and write UTF-8 output to disk.

This avoids giant `python -c ...` commands and uses conservative defaults so it
finishes quickly even on CPU.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoTokenizer

# Ensure repo root is on sys.path so `import src...` works when running from /scripts
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from r3m.models.recurrent import R3MRecurrentConfig, R3MRecurrentLM


DEFAULT_PROMPTS: List[str] = [
    "truth is, that the stage after the Restoration reflects only too\nfaithfully the manners and the sentiments",
    "CHAPTER I.\n\nIt was in the year",
    "In the beginning of the nineteenth century,",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--k-steps", type=int, default=1, help="Thinking steps during generation (keep small for speed)")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=None, help="Nucleus sampling (e.g. 0.9). If set, applied after top-k.")
    ap.add_argument("--repetition-penalty", type=float, default=1.12)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0)
    ap.add_argument("--no-eos-stop", action="store_true")
    ap.add_argument("--prompt-file", type=str, default=None, help="Optional UTF-8 file with prompts separated by blank lines")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location=device)
    cfg = R3MRecurrentConfig(**ck["config"])

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = R3MRecurrentLM(cfg).to(device)
    # Some optional modules/params (e.g. mHC stream params, MoE experts) may be created lazily.
    # Ensure mHC params exist before loading, and fall back to strict=False if needed.
    if bool(getattr(cfg, "mhc_enabled", False)) and int(getattr(cfg, "mhc_n", 0)) >= 2:
        try:
            model.core._mhc_init_if_needed(int(cfg.mhc_n), float(getattr(cfg, "mhc_alpha_init", 0.01)))
        except Exception:
            pass
    try:
        model.load_state_dict(ck["model_state_dict"], strict=True)
    except RuntimeError:
        model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()

    if args.prompt_file:
        p = Path(args.prompt_file)
        raw = p.read_text(encoding="utf-8")
        prompts = [blk.strip() for blk in raw.split("\n\n") if blk.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    eos_token_id: Optional[int] = None if args.no_eos_stop else tok.eos_token_id

    outs: List[str] = []
    for i, prompt in enumerate(prompts):
        ids = torch.tensor([tok.encode(prompt)], device=device)
        gen = model.generate(
            ids,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=None if args.top_p is None else float(args.top_p),
            k_steps=int(args.k_steps),
            eos_token_id=eos_token_id,
            repetition_penalty=float(args.repetition_penalty),
            no_repeat_ngram_size=int(args.no_repeat_ngram_size),
        )[0].tolist()
        cont = tok.decode(gen[len(ids[0]) :])
        outs.append(f"=== SAMPLE {i+1} ===\n\n=== PROMPT ===\n{prompt}\n\n=== CONTINUATION ===\n{cont}\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(outs), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()



