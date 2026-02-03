import argparse
import os
import sys

import torch
from transformers import AutoTokenizer

# Ensure R3M/ is on sys.path\nR3M_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))\nif R3M_ROOT not in sys.path:\n    sys.path.insert(0, R3M_ROOT)\n\nfrom r3m.models.adapter_hf import R3MHFAdapterConfig, R3MHFAdapterLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = R3MHFAdapterConfig(base_model_name=str(args.base), freeze_base=True, episodic_slots=64)
    model = R3MHFAdapterLM(cfg, torch_dtype=(torch.bfloat16 if device.type == "cuda" else None)).to(device)

    prompts = [
        "Hello!",
        "What is 2+2?",
        "What is the capital of France?",
        "Write a short haiku about winter.",
    ]
    for p in prompts:
        text = f"User: {p}\nAssistant:"
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        out = model.generate(
            ids,
            max_new_tokens=int(args.max_new),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            eos_token_id=tok.eos_token_id,
        )
        gen = tok.decode(out[0].tolist(), skip_special_tokens=True)
        print("\n--- PROMPT ---")
        print(p)
        print("--- OUT ---")
        print(gen)


if __name__ == "__main__":
    main()




