# Ensure R3M/ is on sys.path\nimport os\nimport sys\nR3M_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))\nif R3M_ROOT not in sys.path:\n    sys.path.insert(0, R3M_ROOT)\n\n#!/usr/bin/env python3
"""
Evaluate R3MRecurrentLM checkpoints with a "one-line answer" extraction.

Why:
- Our dataset is chat-ish and often contains many short Q/A templates.
- Small models can answer correctly but then drift into other memorized snippets.
- This eval clips at the first newline after the final "Assistant:" marker,
  making checkpoint-to-checkpoint improvements much easier to see.

Output:
- JSONL report with prompt, decoded text, extracted one-line answer, and settings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

# Ensure repo root is on sys.path so `import src...` works when running from /scripts
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from r3m.models.recurrent import R3MRecurrentConfig, R3MRecurrentLM


DEFAULT_PROMPTS: List[str] = [
    "Human: Hello!\nAssistant:",
    "Human: What is 12 - 1?\nAssistant:",
    "Human: List three colors.\nAssistant:",
    "Human: What do you call a baby horse?\nAssistant:",
    "Human: Explain gravity in one sentence.\nAssistant:",
]


def _infer_step_from_ckpt_name(path: str) -> Optional[int]:
    m = re.search(r"step_(\d+)\.pt$", os.path.basename(path))
    if not m:
        return None
    return int(m.group(1))


def extract_one_line_answer(decoded: str, prompt: str) -> str:
    """
    Take the decoded full text (prompt + completion) and extract the first line
    of the completion *immediately following the prompt*.

    Important:
    - The model can sometimes emit extra "Assistant:" / "Human:" markers.
    - Using the *last* "Assistant:" can hide a correct first answer.
    """
    # Prefer slicing based on the exact prompt prefix.
    if prompt and decoded.startswith(prompt):
        tail = decoded[len(prompt) :]
    else:
        # Fallback: find prompt inside decoded (tokenizer may change whitespace)
        pos = decoded.find(prompt) if prompt else -1
        if pos != -1:
            tail = decoded[pos + len(prompt) :]
        else:
            # Last resort: take content after the FIRST occurrence of Assistant:
            marker = "Assistant:"
            idx = decoded.find(marker)
            if idx == -1:
                return decoded.strip().splitlines()[0].strip() if decoded.strip() else ""
            tail = decoded[idx + len(marker) :]

    # Cut off if the model starts a new turn marker
    # (common failure: it begins "Human:" / "Assistant:" again).
    for turn_marker in ["\nHuman:", "\nAssistant:"]:
        j = tail.find(turn_marker)
        if j != -1:
            tail = tail[:j]

    # clip at first newline
    return tail.split("\n", 1)[0].strip()


@torch.no_grad()
def generate_one(
    model: R3MRecurrentLM,
    tok,
    prompt: str,
    max_new_tokens: int,
    k_steps: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    eos_token_id: Optional[int],
    device: torch.device,
) -> Tuple[str, str]:
    ids = torch.tensor([tok.encode(prompt)], device=device)
    gen = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        k_steps=k_steps,
        eos_token_id=eos_token_id,
        repetition_penalty=repetition_penalty,
    )[0].tolist()
    decoded = tok.decode(gen)
    one_line = extract_one_line_answer(decoded, prompt=prompt)
    return decoded, one_line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    ap.add_argument("--out", required=True, help="Path to write JSONL report")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--k-list", type=str, default="1,2,4", help="Comma-separated k_steps values")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=1, help="Use 1 for deterministic (greedy-like) decoding")
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--no-eos-stop", action="store_true", help="Do not stop on eos token")
    ap.add_argument("--prompts-file", type=str, default=None, help="Optional text file with one prompt per line")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location=device)
    cfg = R3MRecurrentConfig(**ck["config"])

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    model = R3MRecurrentLM(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    if args.prompts_file:
        prompts_path = Path(args.prompts_file)
        prompts = [ln.rstrip("\n") for ln in prompts_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    k_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    eos_token_id = None if args.no_eos_stop else tok.eos_token_id

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    step = int(ck.get("step", 0)) or (_infer_step_from_ckpt_name(args.ckpt) or 0)

    rows: List[Dict[str, object]] = []
    for k_steps in k_list:
        for prompt in prompts:
            decoded, one_line = generate_one(
                model=model,
                tok=tok,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                k_steps=k_steps,
                temperature=args.temperature,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=eos_token_id,
                device=device,
            )
            rows.append(
                {
                    "ckpt": os.path.normpath(args.ckpt),
                    "step": step,
                    "k_steps": int(k_steps),
                    "temperature": float(args.temperature),
                    "top_k": int(args.top_k),
                    "repetition_penalty": float(args.repetition_penalty),
                    "prompt": prompt,
                    "decoded": decoded,
                    "one_line_answer": one_line,
                }
            )

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {out_path} ({len(rows)} rows) on device={device.type}")


if __name__ == "__main__":
    main()



