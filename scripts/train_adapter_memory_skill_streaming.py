import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Ensure R3M/ is on sys.path\nR3M_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))\nif R3M_ROOT not in sys.path:\n    sys.path.insert(0, R3M_ROOT)\n\nfrom r3m.models.adapter_hf import R3MHFAdapterConfig, R3MHFAdapterLM


CODE_RE = re.compile(r"\b([A-Z0-9]{10})\b")


def load_recall_rows(path: str, take: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        ans = str(r.get("teacher_answer", "")).strip()
        user = str(r.get("user", "")).strip()
        # Recall rows have teacher_answer as the 10-char code
        if CODE_RE.fullmatch(ans):
            rows.append({"user": user, "code": ans})
    if take and take > 0:
        rows = rows[: int(take)]
    if not rows:
        raise RuntimeError("No recall rows found in dataset (teacher_answer should be a 10-char code).")
    return rows


def parse_turns(user_text: str) -> Tuple[str, List[str], str, str]:
    """
    Parse memory-skill v2 user text:
      Remember... (contains code)
      Filler question i...
      What is the secret code? ...
    Returns (store_turn, distractor_turns[], query_turn, code)
    """
    lines = [ln.strip() for ln in user_text.split("\n") if ln.strip()]
    if len(lines) < 2:
        raise ValueError("not enough lines")
    store = lines[0]
    query = lines[-1]
    distract = lines[1:-1]
    m = CODE_RE.search(store.upper())
    if not m:
        raise ValueError("code not found in store turn")
    code = m.group(1)
    return store, distract, query, code


def trunc(ids: torch.Tensor, max_ctx: int) -> torch.Tensor:
    if ids.size(1) <= int(max_ctx):
        return ids
    return ids[:, -int(max_ctx) :]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="outputs/adapter_memoryskill_stream")
    ap.add_argument("--base", type=str, default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--dataset", type=str, default="outputs/memory_skill_v2/teacher_rows.jsonl")
    ap.add_argument("--take-rows", type=int, default=5000)
    ap.add_argument("--resume-adapter-ckpt", type=str, default=None)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")

    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)

    ap.add_argument("--max-ctx-tokens", type=int, default=64, help="Visible context window at each turn (forces memory).")
    ap.add_argument("--max-code-len", type=int, default=12, help="Max code tokens to train (typically 10 chars, but tokenizer may split).")

    # Adapter stability knobs
    ap.add_argument("--episodic-slots", type=int, default=64)
    ap.add_argument("--write-topk", type=int, default=1)
    ap.add_argument("--write-gate-max", type=float, default=0.15)
    ap.add_argument("--mem-scale", type=float, default=0.25)
    ap.add_argument("--mem-delta-clip", type=float, default=0.6)

    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and str(args.amp) != "off"
    amp_dtype = torch.float16 if str(args.amp) == "fp16" else (torch.bfloat16 if str(args.amp) == "bf16" else None)

    tok = AutoTokenizer.from_pretrained(str(args.base), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = load_recall_rows(str(args.dataset), take=int(args.take_rows))
    print(f"[data] recall rows: {len(rows)}", flush=True)

    cfg = R3MHFAdapterConfig(
        base_model_name=str(args.base),
        freeze_base=True,
        episodic_slots=int(args.episodic_slots),
        write_topk=int(args.write_topk),
        write_gate_max=float(args.write_gate_max),
        mem_scale=float(args.mem_scale),
        mem_delta_clip=float(args.mem_delta_clip),
        detach_memory_across_tokens=False,  # IMPORTANT: allow gradients across turns
        detach_every_tokens=0,
    )
    model = R3MHFAdapterLM(cfg, torch_dtype=(amp_dtype if use_amp else None)).to(device)
    model.train()

    # Optimizer: adapter-only
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))

    if args.resume_adapter_ckpt:
        ck = torch.load(str(args.resume_adapter_ckpt), map_location=device)
        if "adapter_state_dict" in ck:
            model.load_adapter_state_dict(ck["adapter_state_dict"], strict=False)
            print(f"[resume] loaded adapter weights from {args.resume_adapter_ckpt}", flush=True)

    losses: List[float] = []
    for step in range(int(args.steps)):
        r = random.choice(rows)
        store, distract, query, code = parse_turns(r["user"])

        # Initialize persistent memory state for this episode (batch=1).
        mem_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = model.init_memory_state(batch_size=1, device=device)

        # 1) Store turn: update memory (visible context truncated)
        prompt_store = f"User: {store}\nAssistant:"
        ids = tok(prompt_store, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        ids = trunc(ids, int(args.max_ctx_tokens))
        attn = torch.ones_like(ids, device=device)
        _, mem_state = model.encode_with_memory_state(ids, attn, mem_state, update_memory=True)

        # 2) Distractors: update memory (but no code in visible tokens)
        for d in distract:
            prompt_d = f"User: {d}\nAssistant:"
            ids_d = tok(prompt_d, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            ids_d = trunc(ids_d, int(args.max_ctx_tokens))
            attn_d = torch.ones_like(ids_d, device=device)
            _, mem_state = model.encode_with_memory_state(ids_d, attn_d, mem_state, update_memory=True)

        # 3) Query: teacher-force the code tokens WITHOUT including the code in the prompt context.
        prompt_q = f"User: {query}\nAssistant:"
        ctx = tok(prompt_q, add_special_tokens=False).input_ids
        code_ids = tok(code, add_special_tokens=False).input_ids
        code_ids = code_ids[: int(args.max_code_len)]
        if len(code_ids) == 0:
            # Extremely unlikely, but avoid backward() on a constant.
            continue

        total = torch.tensor(0.0, device=device)
        n = 0
        ctx_ids = torch.tensor([ctx], dtype=torch.long, device=device)
        for target_id in code_ids:
            ctx_ids = trunc(ctx_ids, int(args.max_ctx_tokens))
            attn_q = torch.ones_like(ctx_ids, device=device)
            logits, _ = model.encode_with_memory_state(ctx_ids, attn_q, mem_state, update_memory=False)
            next_logits = logits[:, -1, :].float()
            loss_t = F.cross_entropy(next_logits, torch.tensor([target_id], device=device))
            total = total + loss_t
            n += 1
            # teacher forcing: append correct token
            ctx_ids = torch.cat([ctx_ids, torch.tensor([[target_id]], device=device)], dim=1)

        loss = total / max(1, n)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        losses.append(float(loss.item()))
        if (step + 1) % max(1, int(args.log_every)) == 0:
            print(f"step {step+1}/{args.steps} | loss {loss.item():.4f}", flush=True)

        if int(args.save_every) > 0 and ((step + 1) % int(args.save_every) == 0):
            ckpt_path = out_dir / f"adapter_ckpt_step_{step+1}.pt"
            torch.save(
                {
                    "adapter_config": asdict(cfg),
                    "adapter_state_dict": model.adapter_state_dict(),
                    "global_step": int(step + 1),
                },
                ckpt_path,
            )
            print(f"[ckpt] wrote {ckpt_path}", flush=True)

    (out_dir / "losses.json").write_text(json.dumps({"losses": losses[-200:]}, indent=2), encoding="utf-8")
    final_path = out_dir / "adapter_final.pt"
    torch.save(
        {"adapter_config": asdict(cfg), "adapter_state_dict": model.adapter_state_dict(), "global_step": int(len(losses))},
        final_path,
    )
    print(f"wrote {final_path}", flush=True)


if __name__ == "__main__":
    main()




