import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure R3M/ is on sys.path\nR3M_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))\nif R3M_ROOT not in sys.path:\n    sys.path.insert(0, R3M_ROOT)\n\nfrom r3m.models.adapter_hf import R3MHFAdapterConfig, R3MHFAdapterLM


_CTRL_TOKEN_RE = re.compile(r"<\|[^>]+\|>")
_ROLE_LINE_RE = re.compile(r"^\s*(system|user|human|assistant)\s*:?\s*$", re.IGNORECASE)


def _clean_teacher_answer(text: str) -> str:
    t = text.replace("\r\n", "\n").strip()
    t = _CTRL_TOKEN_RE.sub("", t)
    for marker in [
        "\nHuman:",
        "\nUser:",
        "\nAssistant:",
        "\nSystem:",
        "\nhuman:",
        "\nuser:",
        "\nassistant:",
        "\nsystem:",
        "\n###",
        "\n\nHuman:",
        "\n\nUser:",
        "\n\nAssistant:",
        "\n\nSystem:",
        "\n\nhuman:",
        "\n\nuser:",
        "\n\nassistant:",
        "\n\nsystem:",
    ]:
        idx = t.find(marker)
        if idx != -1:
            t = t[:idx].strip()
            break
    if "\n" in t:
        lines = []
        for ln in t.split("\n"):
            if _ROLE_LINE_RE.match(ln):
                continue
            # Drop common meta-instructions that sometimes leak into teacher answers
            lnl = ln.strip().lower()
            if lnl.startswith("you are a helpful assistant") or lnl.startswith("you are an ai assistant"):
                continue
            if "provide a detailed answer" in lnl and "assistant" in lnl:
                continue
            lines.append(ln)
        t = "\n".join(lines).strip()
    if "\n\n" in t:
        t = t.split("\n\n", 1)[0].strip()
    return t.strip()


def _strip_prompt_echo(user_prompt: str, answer: str) -> str:
    u = (user_prompt or "").strip()
    a = (answer or "").lstrip()
    if not u or not a:
        return a.strip()
    u_low = u.lower()
    a_low = a.lower()
    # Drop echoed first line
    if "\n" in a:
        first, rest = a.split("\n", 1)
        first_s = first.strip()
        if 0 < len(first_s) <= 80 and first_s.lower() in u_low and ("?" in first_s or first_s.endswith("?")):
            a = rest.lstrip()
            a_low = a.lower()
    # Strip longest prefix of answer that appears in prompt
    max_k = min(120, len(a_low))
    for k in range(max_k, 17, -1):
        pref = a_low[:k]
        if pref and pref in u_low:
            a = a[k:].lstrip(" \t\n:,-")
            return a.strip()
    return a.strip()


def load_teacher_rows(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    rows: List[Dict[str, Any]] = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rows.append(json.loads(ln))
    return rows


def build_student_examples(
    rows: List[Dict[str, Any]],
    tok,
    max_len: int,
) -> List[Dict[str, Any]]:
    exs: List[Dict[str, Any]] = []
    for r in rows:
        user = str(r.get("user", "")).strip()
        ans_raw = str(r.get("teacher_answer", r.get("answer", ""))).strip()
        ans = _strip_prompt_echo(user, _clean_teacher_answer(ans_raw))
        if not user or not ans:
            continue
        prompt = f"User: {user}\nAssistant:"
        text = prompt + " " + ans
        # Prompt token length for masking
        prompt_len = len(tok(prompt, add_special_tokens=False).input_ids)
        ids = tok(text, add_special_tokens=False).input_ids
        if len(ids) < 4:
            continue
        # Pre-truncate very long examples early (avoid wasting time later)
        if len(ids) > int(max_len):
            # keep tail supervision by trimming from the front, but ensure we keep full prompt region
            # If prompt is already too long, drop.
            if prompt_len >= int(max_len) - 2:
                continue
        exs.append({"text": text, "prompt_len": int(prompt_len), "user": user, "answer": ans})
    random.shuffle(exs)
    return exs


def batch_tokenize(
    tok,
    batch: List[Dict[str, Any]],
    max_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    texts = [b["text"] for b in batch]
    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_len),
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    keep_rows: List[int] = []
    for i, b in enumerate(batch):
        pl = int(b["prompt_len"])
        pl = max(0, min(pl, labels.size(1)))
        labels[i, :pl] = -100
        if (labels[i] != -100).any().item():
            keep_rows.append(i)
    if len(keep_rows) == 0:
        raise ValueError("All sampled rows had zero supervised tokens after prompt masking; resample batch.")
    if len(keep_rows) != labels.size(0):
        idx = torch.tensor(keep_rows, device=input_ids.device, dtype=torch.long)
        input_ids = input_ids.index_select(0, idx)
        attention_mask = attention_mask.index_select(0, idx)
        labels = labels.index_select(0, idx)
    return input_ids, attention_mask, labels


@torch.no_grad()
def generate_eval_samples(
    model: R3MHFAdapterLM,
    tok,
    prompts: List[str],
    device: torch.device,
    out_path: Path,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: Optional[float],
) -> None:
    model.eval()
    outs: List[str] = []
    for i, p in enumerate(prompts):
        prompt = f"User: {p}\nAssistant:"
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        gen = model.generate(
            ids,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=None if top_p is None else float(top_p),
            eos_token_id=tok.eos_token_id,
        )[0].tolist()
        cont = tok.decode(gen[len(ids[0]) :], skip_special_tokens=True)
        outs.append(f"=== SAMPLE {i+1} ===\nPROMPT: {p}\n\nCONTINUATION:\n{cont}\n")
    out_path.write_text("\n\n".join(outs), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="outputs/distill_adapter_hf")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--reuse-teacher-rows", type=str, required=True, help="Path to existing teacher_rows.jsonl")
    ap.add_argument("--take-rows", type=int, default=2000, help="How many teacher rows to use (subset).")

    ap.add_argument("--student-base", type=str, default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--student-max-len", type=int, default=256)
    ap.add_argument("--student-steps", type=int, default=2000)
    ap.add_argument("--student-batch-size", type=int, default=8)
    ap.add_argument("--student-lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--amp", type=str, default="bf16", choices=["off", "fp16", "bf16"])
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--resume-adapter-ckpt", type=str, default=None, help="Resume adapter weights (and optimizer if present) from a previous adapter checkpoint.")

    # Adapter/memory knobs
    ap.add_argument("--episodic-slots", type=int, default=64)
    ap.add_argument("--write-topk", type=int, default=1)
    ap.add_argument("--write-gate-max", type=float, default=0.2)
    ap.add_argument("--mem-scale", type=float, default=0.5)
    ap.add_argument("--mem-delta-clip", type=float, default=2.0, help="Clamp adapter memory residual to avoid repetition/collapse.")
    ap.add_argument("--no-detach-mem", action="store_true")
    ap.add_argument("--detach-every-tokens", type=int, default=0, help="If >0 and --no-detach-mem, detach memory state every N tokens (truncated BPTT).")

    # Periodic outputs
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--eval-prompt-file", type=str, default=None)
    ap.add_argument("--sample-max-new", type=int, default=120)
    ap.add_argument("--sample-temperature", type=float, default=0.7)
    ap.add_argument("--sample-top-k", type=int, default=50)
    ap.add_argument("--sample-top-p", type=float, default=0.92)

    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and str(args.amp) != "off"
    amp_dtype = torch.float16 if str(args.amp) == "fp16" else (torch.bfloat16 if str(args.amp) == "bf16" else None)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and str(args.amp) == "fp16"))

    # Tokenizer (student/base tokenizer)
    tok = AutoTokenizer.from_pretrained(str(args.student_base), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load cached teacher rows
    teacher_rows = load_teacher_rows(str(args.reuse_teacher_rows))
    if not teacher_rows:
        raise RuntimeError("teacher_rows.jsonl was empty")
    if int(args.take_rows) > 0:
        teacher_rows = teacher_rows[: int(args.take_rows)]
    print(f"[data] loaded teacher rows: {len(teacher_rows)}", flush=True)

    examples = build_student_examples(rows=teacher_rows, tok=tok, max_len=int(args.student_max_len))
    if not examples:
        raise RuntimeError("No usable examples after cleaning/building.")
    (out_dir / "student_dataset_preview.jsonl").write_text(
        "\n".join(json.dumps({"text": e["text"], "prompt_len": e["prompt_len"]}, ensure_ascii=False) for e in examples[:10]) + "\n",
        encoding="utf-8",
    )

    # Eval prompt suite
    eval_prompts = [
        "Hello!",
        "What is 2+2?",
        "What is 13*7?",
        "What is the capital of France?",
        "Write a short haiku about winter.",
    ]
    if args.eval_prompt_file:
        raw = Path(str(args.eval_prompt_file)).read_text(encoding="utf-8").splitlines()
        loaded = [ln.strip() for ln in raw if ln.strip()]
        if loaded:
            eval_prompts = loaded

    # Model
    cfg = R3MHFAdapterConfig(
        base_model_name=str(args.student_base),
        episodic_slots=int(args.episodic_slots),
        write_topk=int(args.write_topk),
        write_gate_max=float(args.write_gate_max),
        mem_scale=float(args.mem_scale),
        mem_delta_clip=float(args.mem_delta_clip),
        detach_memory_across_tokens=(not bool(args.no_detach_mem)),
        detach_every_tokens=int(args.detach_every_tokens),
        freeze_base=True,
    )
    model = R3MHFAdapterLM(cfg, torch_dtype=(amp_dtype if use_amp else None)).to(device)
    model.train()

    # Optimizer only on trainable params (adapter)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=float(args.student_lr), weight_decay=float(args.weight_decay))

    # Optional resume
    if args.resume_adapter_ckpt:
        rp = Path(str(args.resume_adapter_ckpt))
        ckpt = torch.load(rp, map_location=device)
        if "adapter_state_dict" in ckpt:
            model.load_adapter_state_dict(ckpt["adapter_state_dict"], strict=False)
            print(f"[resume] loaded adapter weights from {rp}", flush=True)
        if "optimizer_state_dict" in ckpt:
            try:
                opt.load_state_dict(ckpt["optimizer_state_dict"])
                print("[resume] loaded optimizer state", flush=True)
            except Exception as e:
                print(f"[resume] optimizer state load failed: {e}", flush=True)
        if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None and scaler.is_enabled():
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
                print("[resume] loaded scaler state", flush=True)
            except Exception as e:
                print(f"[resume] scaler state load failed: {e}", flush=True)

    # Before-training samples
    generate_eval_samples(
        model=model,
        tok=tok,
        prompts=eval_prompts[: min(12, len(eval_prompts))],
        device=device,
        out_path=out_dir / "samples_before.txt",
        max_new_tokens=int(args.sample_max_new),
        temperature=float(args.sample_temperature),
        top_k=int(args.sample_top_k),
        top_p=float(args.sample_top_p) if args.sample_top_p is not None else None,
    )

    losses: List[float] = []
    for step in range(int(args.student_steps)):
        # Resample if masking drops all supervision
        for _try in range(10):
            batch = random.sample(examples, k=min(int(args.student_batch_size), len(examples)))
            try:
                input_ids, attention_mask, labels = batch_tokenize(tok, batch, int(args.student_max_len), device)
                break
            except ValueError:
                continue
        else:
            raise RuntimeError("Could not sample a batch with supervised tokens after 10 tries.")

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=bool(use_amp)):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
            loss = out["loss"]

        if loss is None or (not torch.isfinite(loss).all()):
            print(f"step {step+1}/{args.student_steps} | loss non-finite. stopping.", flush=True)
            break

        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

        losses.append(float(loss.item()))
        if (step + 1) % max(1, int(args.log_every)) == 0:
            print(f"step {step+1}/{args.student_steps} | loss {loss.item():.4f}", flush=True)

        global_step = step + 1
        if int(args.save_every) > 0 and (global_step % int(args.save_every) == 0):
            ckpt_path = out_dir / f"adapter_ckpt_step_{global_step}.pt"
            torch.save(
                {
                    "adapter_config": asdict(cfg),
                    "adapter_state_dict": model.adapter_state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scaler_state_dict": (scaler.state_dict() if scaler is not None else None),
                    "global_step": int(global_step),
                },
                ckpt_path,
            )
            print(f"[ckpt] wrote {ckpt_path}", flush=True)
        if int(args.sample_every) > 0 and (global_step % int(args.sample_every) == 0):
            sample_path = out_dir / f"samples_step_{global_step}.txt"
            generate_eval_samples(
                model=model,
                tok=tok,
                prompts=eval_prompts[: min(12, len(eval_prompts))],
                device=device,
                out_path=sample_path,
                max_new_tokens=int(args.sample_max_new),
                temperature=float(args.sample_temperature),
                top_k=int(args.sample_top_k),
                top_p=float(args.sample_top_p) if args.sample_top_p is not None else None,
            )
            print(f"[eval] wrote {sample_path}", flush=True)

    (out_dir / "losses.json").write_text(json.dumps({"losses": losses[-200:]}, indent=2), encoding="utf-8")

    # Final save
    final_path = out_dir / "adapter_final.pt"
    torch.save(
        {
            "adapter_config": asdict(cfg),
            "adapter_state_dict": model.adapter_state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scaler_state_dict": (scaler.state_dict() if scaler is not None else None),
            "global_step": int(len(losses)),
        },
        final_path,
    )
    print(f"wrote {final_path}", flush=True)

    generate_eval_samples(
        model=model,
        tok=tok,
        prompts=eval_prompts[: min(12, len(eval_prompts))],
        device=device,
        out_path=out_dir / "samples_after.txt",
        max_new_tokens=int(args.sample_max_new),
        temperature=float(args.sample_temperature),
        top_k=int(args.sample_top_k),
        top_p=float(args.sample_top_p) if args.sample_top_p is not None else None,
    )


if __name__ == "__main__":
    main()




