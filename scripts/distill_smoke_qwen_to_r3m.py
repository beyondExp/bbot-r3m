# Ensure R3M/ is on sys.path\nimport os\nimport sys\nR3M_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))\nif R3M_ROOT not in sys.path:\n    sys.path.insert(0, R3M_ROOT)\n\n#!/usr/bin/env python3
"""
Distillation smoke test:
- Use a small fast teacher (Qwen/Qwen2.5-3B-Instruct) to generate instruction-style targets
- Train R3MRecurrentLM (student) on those targets for a few hundred steps

This is not meant to produce a great model; it's a pipeline sanity test to confirm:
  teacher -> dataset -> student training -> improved short-form responses

Writes UTF-8 outputs to disk (Windows-friendly).
"""

from __future__ import annotations

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

# Ensure repo root is on sys.path so `import src...` works when running from /scripts
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from r3m.models.recurrent import R3MRecurrentConfig, R3MRecurrentLM


DEFAULT_PROMPTS: List[str] = [
    "What is 2+2?",
    "What is 13*7?",
    "Explain in one sentence what a transformer model is.",
    "Explain what overfitting is in one sentence.",
    "Write a short haiku about winter.",
    "Give three tips for staying focused while studying.",
    "Summarize: Cats are small domesticated carnivores.",
    "Translate to German: 'The sky is blue.'",
    "Write a polite email asking for a meeting tomorrow.",
    "What is the capital of France?",
    "Explain the difference between a list and a tuple in Python.",
    "Give a simple example of a for-loop in Python.",
]


_CTRL_TOKEN_RE = re.compile(r"<\|[^>]+\|>")

_ROLE_LINE_RE = re.compile(r"^\s*(system|user|human|assistant)\s*:?\s*$", re.IGNORECASE)


def _clean_teacher_answer(text: str) -> str:
    """
    Normalize teacher output into plain assistant text:
    - remove special control tokens like <|im_start|> / <|endoftext|>
    - stop at the first sign of a new prompt/role
    """
    t = text.replace("\r\n", "\n").strip()
    t = _CTRL_TOKEN_RE.sub("", t)
    # Hard stop if the model starts a new conversation/prompt (handle case variants too)
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
    # Drop standalone role lines like "assistant" that often leak in from chat templates
    if "\n" in t:
        lines = []
        for ln in t.split("\n"):
            if _ROLE_LINE_RE.match(ln):
                continue
            lines.append(ln)
        t = "\n".join(lines).strip()
    # Also cut at first double-blank if it starts rambling into another topic
    if "\n\n" in t:
        t = t.split("\n\n", 1)[0].strip()
    # Avoid empty answers
    return t.strip()


def _strip_prompt_echo(user_prompt: str, answer: str) -> str:
    """
    Teachers sometimes echo parts of the user prompt at the start of the answer,
    which teaches the student to emit "Assistant:" / repeated fragments.
    Heuristic: if answer starts with a long substring of the user prompt, drop it.
    """
    u = (user_prompt or "").strip()
    a = (answer or "").lstrip()
    if not u or not a:
        return a.strip()
    u_low = u.lower()
    a_low = a.lower()
    # 1) Drop an echoed first line like "right now?" that appears verbatim in the prompt.
    if "\n" in a:
        first, rest = a.split("\n", 1)
        first_s = first.strip()
        if 0 < len(first_s) <= 80 and first_s.lower() in u_low and ("?" in first_s or first_s.endswith("?")):
            a = rest.lstrip()
            a_low = a.lower()

    # 2) Strip the longest prefix of the answer that appears anywhere inside the user prompt.
    # This catches cases like answer starting with a clause from the prompt:
    # "looking for milk alternatives..." / "some good foods..." / etc.
    max_k = min(120, len(a_low))
    for k in range(max_k, 17, -1):
        pref = a_low[:k]
        if pref and pref in u_low:
            a = a[k:].lstrip(" \t\n:,-")
            return a.strip()

    # 3) Fallback: direct prefix match (rare after step 2 but cheap).
    max_check = min(len(u_low), len(a_low), 240)
    n = 0
    while n < max_check and a_low[n] == u_low[n]:
        n += 1
    if n >= 18:
        a = a[n:].lstrip(" \t\n:,-")
    return a.strip()


def _try_import_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Hugging Face `datasets` is required for --prompt-dataset.\n"
            "Install with: pip install datasets\n"
            f"Import error: {e}"
        )
    return load_dataset


def _load_state_dict_shape_safe(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """
    Load checkpoint weights but skip keys whose tensor shapes don't match.
    Useful when we override config dimensions like episodic_slots.
    """
    model_sd = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for k, v in state_dict.items():
        if k not in model_sd:
            continue
        try:
            if hasattr(model_sd[k], "shape") and hasattr(v, "shape") and model_sd[k].shape != v.shape:
                skipped.append(k)
                continue
        except Exception:
            pass
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if skipped:
        print(f"[ckpt] skipped {len(skipped)} keys due to shape mismatch (e.g. episodic_slots override).", flush=True)
    if unexpected:
        print(f"[ckpt] unexpected keys: {len(unexpected)}", flush=True)
    if missing:
        # missing is expected when we skip shape-mismatched params
        print(f"[ckpt] missing keys: {len(missing)}", flush=True)


def _extract_user_prompt_from_messages(
    messages: Any,
    mode: str = "last_user",
    require_single_turn: bool = False,
    user_index: int = 0,
) -> Optional[str]:
    """
    Extract a user prompt from a Nemotron-style messages list.
    messages: List[{"role": "...", "content": "..."}]
    By default we take the last user message content. Optionally, to avoid "mid conversation"
    prompts that rely on missing context, you can:
      - mode="first_user" to take the first user message
      - require_single_turn=True to only accept conversations with exactly one user message
    """
    try:
        if not isinstance(messages, list):
            return None
        user_msgs: List[str] = []
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content", None)
                if isinstance(c, str) and c.strip():
                    user_msgs.append(c.strip())
        if not user_msgs:
            return None
        if require_single_turn and len(user_msgs) != 1:
            return None
        mode = str(mode or "last_user").strip().lower()
        if mode == "first_user":
            return user_msgs[0]
        if mode == "user_index":
            idx = int(user_index)
            if idx < 0:
                idx = len(user_msgs) + idx
            if 0 <= idx < len(user_msgs):
                return user_msgs[idx]
            return None
        # default: last_user
        return user_msgs[-1]
    except Exception:
        return None


def _is_good_prompt(p: str, max_chars: int, banned_regex: Optional[re.Pattern]) -> bool:
    if not isinstance(p, str):
        return False
    p = p.strip()
    if len(p) < 3:
        return False
    if int(max_chars) > 0 and len(p) > int(max_chars):
        return False
    if banned_regex is not None and banned_regex.search(p) is not None:
        return False
    return True


def load_prompts_from_hf(
    dataset_name: str,
    split: str,
    dataset_config: Optional[str],
    streaming: bool,
    take: int,
    seed: int,
    max_chars: int = 280,
    banned_regex: Optional[re.Pattern] = None,
    scan_max: int = 200_000,
    prompt_mode: str = "last_user",
    require_single_turn: bool = False,
    prompt_user_index: int = 0,
) -> List[str]:
    """
    Load prompts from a Hugging Face dataset (streaming recommended).
    Currently supports Nemotron-style `messages` datasets.
    """
    load_dataset = _try_import_datasets()
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)
    else:
        ds = load_dataset(dataset_name, split=split, streaming=streaming)

    prompts: List[str] = []
    rng = random.Random(int(seed))

    # Streaming datasets: we just iterate and collect up to `take`.
    scanned = 0
    for ex in ds:
        scanned += 1
        if int(scan_max) > 0 and scanned > int(scan_max):
            break
        p = None
        if isinstance(ex, dict):
            if "messages" in ex:
                p = _extract_user_prompt_from_messages(
                    ex.get("messages"),
                    mode=str(prompt_mode),
                    require_single_turn=bool(require_single_turn),
                    user_index=int(prompt_user_index),
                )
            # fallback: common fields
            if p is None:
                for k in ["prompt", "instruction", "question", "query", "user"]:
                    v = ex.get(k, None)
                    if isinstance(v, str) and v.strip():
                        p = v.strip()
                        break
        if p is None:
            continue
        # light cleanup
        p = " ".join(p.split())
        if not _is_good_prompt(p, max_chars=int(max_chars), banned_regex=banned_regex):
            continue
        prompts.append(p)
        if len(prompts) >= int(take):
            break

    # Shuffle locally for variety (we're not relying on dataset shuffle)
    rng.shuffle(prompts)
    return prompts

def _teacher_build_text(tok, user_prompt: str, style_prefix: str = "") -> str:
    # Prefer chat template if present; otherwise fallback to plain prompt.
    if style_prefix:
        user_prompt = f"{style_prefix}\n{user_prompt}".strip()
    if hasattr(tok, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"You are a helpful assistant.\n\nUser: {user_prompt}\nAssistant:"


@torch.no_grad()
def teacher_generate(
    model,
    tok,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int = 4,
    style_prefix: str = "",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    bs = max(1, int(batch_size))
    texts = [_teacher_build_text(tok, p, style_prefix=style_prefix) for p in prompts]

    for start in range(0, len(prompts), bs):
        chunk_texts = texts[start : start + bs]
        chunk_prompts = prompts[start : start + bs]

        enc = tok(chunk_texts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(device)
            input_lens = attn.sum(dim=-1).tolist()
        else:
            input_lens = [input_ids.size(1)] * input_ids.size(0)

        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

        for i in range(gen_ids.size(0)):
            gen = gen_ids[i].tolist()
            full = tok.decode(gen)
            new_text = tok.decode(gen[int(input_lens[i]) :])
            ans = _clean_teacher_answer(new_text)
            rows.append({"user": chunk_prompts[i], "teacher_answer": ans, "teacher_full": full})
        # Progress log (helps long runs not look "stuck")
        done = min(start + bs, len(prompts))
        if done % max(1, 10 * bs) == 0 or done == len(prompts):
            print(f"[teacher] generated {done}/{len(prompts)}", flush=True)
    return rows


def build_student_examples(gpt2_tok, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build student training examples in a stable text format and store prompt token length
    so we can mask labels for prompt tokens.
    """
    exs: List[Dict[str, Any]] = []
    for r in rows:
        prompt = f"Human: {r['user']}\nAssistant:"
        # Keep answer clean; strip leading roles if teacher included them
        ans = _clean_teacher_answer(str(r["teacher_answer"]))
        ans = _strip_prompt_echo(str(r.get("user", "")), ans)
        for prefix in ["Assistant:", "assistant:", "Answer:", "A:"]:
            if ans.startswith(prefix):
                ans = ans[len(prefix) :].lstrip()
        if not ans:
            ans = "I don't know."
        text = prompt + " " + ans
        prompt_len = len(gpt2_tok.encode(prompt))
        exs.append({"text": text, "prompt_len": int(prompt_len), "user": r["user"], "answer": ans})
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
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    # mask prompt tokens (note forward() shifts labels internally)
    keep_rows: List[int] = []
    for i, b in enumerate(batch):
        pl = int(b["prompt_len"])
        pl = max(0, min(pl, labels.size(1)))
        labels[i, :pl] = -100
        # If prompt consumes the whole sequence (or leaves no supervised tokens),
        # the loss can become unstable/non-informative. Drop these rows.
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


def _cuda_peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated()) / (1024.0**3)


def _autotune_student_batch_size(
    student: R3MRecurrentLM,
    tok,
    examples: List[Dict[str, Any]],
    device: torch.device,
    max_len: int,
    k_steps: int,
    lr: float,
    target_gb: float,
    max_batch: int,
    start_batch: int,
    safety_margin_gb: float = 3.5,
) -> int:
    """
    Increase batch size until we approach target VRAM (or hit OOM), then return best batch.
    This is a quick heuristic so we can "use ~20GB" on a 24GB card while leaving headroom.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return int(start_batch)
    if len(examples) == 0:
        return int(start_batch)

    target = float(target_gb)
    max_b = max(1, int(max_batch))
    b = max(1, int(start_batch))
    best_b = b
    best_gb = 0.0

    # Use a fresh optimizer for the probe so we don't pollute the real run.
    opt = torch.optim.AdamW(student.parameters(), lr=float(lr), weight_decay=0.0)
    # cuDNN RNN backward (nn.GRU) requires training mode.
    prev_training = bool(student.training)
    student.train(True)

    while b <= max_b:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            batch = random.sample(examples, k=min(b, len(examples)))
            input_ids, attention_mask, labels = batch_tokenize(tok=tok, batch=batch, max_len=max_len, device=device)
            out = student(input_ids=input_ids, attention_mask=attention_mask, labels=labels, k_steps=int(k_steps))
            loss = out["loss"]
            if not torch.isfinite(loss).all():
                # If the probe already produces NaNs/Infs, treat this as unstable and stop.
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()

            peak = _cuda_peak_gb()
            # Keep headroom for kernels/fragmentation
            if peak <= target and (24.0 - peak) >= float(safety_margin_gb):
                best_b = b
                best_gb = peak
                # grow quickly at first, then slower
                b = b * 2 if b < 16 else b + 4
                continue
            # If we overshot target or used too much, stop.
            break
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda out of memory" in msg:
                break
            raise

    # Restore original mode
    student.train(prev_training)
    # Avoid non-ASCII chars (Windows cp1252 consoles can crash on characters like 'â‰ˆ').
    print(f"[autotune] student batch_size={best_b} peak_alloc_gb~{best_gb:.2f} (target {target_gb:.1f})", flush=True)
    return int(best_b)


@torch.no_grad()
def student_generate_samples(
    model: R3MRecurrentLM,
    tok,
    prompts: List[str],
    device: torch.device,
    out_path: Path,
    max_new_tokens: int = 120,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: Optional[float] = 0.92,
    repetition_penalty: float = 1.12,
    no_repeat_ngram_size: int = 4,
    k_steps: int = 1,
) -> None:
    # Temporarily switch to eval for deterministic-ish generation, then restore mode.
    prev_training = bool(model.training)
    model.eval()
    outs: List[str] = []
    try:
        for i, p in enumerate(prompts):
            prompt = f"Human: {p}\nAssistant:"
            ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
            # Avoid any persistent generation caches affecting eval.
            if hasattr(model, "reset_generation_memory"):
                try:
                    model.reset_generation_memory()
                except Exception:
                    pass
            gen_ids = model.generate(
                ids,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=None if top_p is None else float(top_p),
                repetition_penalty=float(repetition_penalty),
                no_repeat_ngram_size=int(no_repeat_ngram_size),
                eos_token_id=tok.eos_token_id,
                k_steps=int(k_steps),
            )[0].tolist()
            cont = tok.decode(gen_ids[len(ids[0]) :])
            outs.append(f"=== SAMPLE {i+1} ===\nPROMPT: {p}\n\nCONTINUATION:\n{cont}\n")
        out_path.write_text("\n\n".join(outs), encoding="utf-8")
    finally:
        model.train(prev_training)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="outputs/distill_smoke_qwen2_5_3b")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--teacher-model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--teacher-max-new", type=int, default=160)
    ap.add_argument("--teacher-temperature", type=float, default=0.7)
    ap.add_argument("--teacher-top-p", type=float, default=0.9)
    ap.add_argument("--teacher-batch-size", type=int, default=4, help="Teacher generation batch size (speed/VRAM tradeoff).")
    ap.add_argument("--teacher-style", type=str, default="short", choices=["short", "normal"], help="Hint to avoid boilerplate in teacher answers.")
    ap.add_argument(
        "--reuse-teacher-rows",
        type=str,
        default=None,
        help="Path to an existing teacher_rows.jsonl to skip teacher generation and reuse cached teacher answers.",
    )

    ap.add_argument("--num-prompts", type=int, default=60)
    ap.add_argument("--prompt-file", type=str, default=None, help="Optional file: one prompt per line")
    ap.add_argument("--prompt-dataset", type=str, default=None, help="HF dataset ID to source prompts from (e.g. nvidia/Nemotron-Instruction-Following-Chat-v1)")
    ap.add_argument("--prompt-dataset-config", type=str, default=None)
    ap.add_argument("--prompt-split", type=str, default="train")
    ap.add_argument("--prompt-streaming", action="store_true")
    ap.add_argument("--prompt-max-chars", type=int, default=280)
    ap.add_argument("--prompt-scan-max", type=int, default=200000)
    ap.add_argument(
        "--prompt-mode",
        type=str,
        default="last_user",
        choices=["last_user", "first_user", "user_index"],
        help="How to extract prompts from chat-style `messages` datasets.",
    )
    ap.add_argument(
        "--prompt-user-index",
        type=int,
        default=1,
        help="When --prompt-mode=user_index, which user-message index to use (0=first user msg, 1=second, etc.). "
        "For everyday-conversations, 1 is often the first 'real question' after the greeting.",
    )
    ap.add_argument(
        "--prompt-require-single-turn",
        action="store_true",
        help="Only accept examples with exactly one user message (avoids mid-conversation prompts).",
    )
    ap.add_argument(
        "--prompt-ban-pattern",
        type=str,
        default=r"(ends exactly|nothing after|verbatim|all caps|ALL CAPITAL|lowercase word|no more than twice|exactly\\s+\\d+|wrap.*double quotation|include nothing else|copy the exact phrase|do not add)",
        help="Regex to drop constraint-heavy prompts that often teach boilerplate.",
    )

    ap.add_argument("--student-ckpt", type=str, default="outputs/r3m_rec_pretrain_8h_apertus_gutenberg_mhc_v2/r3m_rec_final.pt")
    ap.add_argument("--student-steps", type=int, default=300)
    ap.add_argument("--student-batch-size", type=int, default=4)
    ap.add_argument("--student-lr", type=float, default=3e-4)
    ap.add_argument("--student-max-len", type=int, default=256)
    ap.add_argument("--k-train", type=int, default=2)
    ap.add_argument("--student-disable-episodic", action="store_true", help="Disable episodic memory during student training (stability/debug).")
    ap.add_argument("--student-disable-mhc", action="store_true", help="Disable mHC stream mixing during student training (stability/debug).")
    ap.add_argument("--student-episodic-slots", type=int, default=None, help="Override episodic memory slot count (e.g. 32 for faster/safer early training).")
    ap.add_argument("--student-write-topk", type=int, default=None, help="Override episodic write routing (0=soft, 1=top-1, k=top-k).")
    ap.add_argument("--student-write-rate-target", type=float, default=None, help="Target avg write strength; adds a penalty if set (e.g. 0.1).")
    ap.add_argument("--student-write-rate-lambda", type=float, default=0.0, help="Weight for write-rate penalty (e.g. 0.5).")
    ap.add_argument("--student-mhc-alpha-init", type=float, default=None, help="Override mHC alpha init (e.g. 0.001-0.005 for gentle start).")
    ap.add_argument("--student-mhc-sinkhorn-iters", type=int, default=None, help="Override mHC Sinkhorn iters (e.g. 2-4 for speed/stability).")
    ap.add_argument("--student-mhc-temperature", type=float, default=None, help="Override mHC temperature (e.g. 2.0 for stability).")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=0, help="If >0, save student checkpoint every N steps.")
    ap.add_argument("--sample-every", type=int, default=0, help="If >0, write eval samples every N steps.")
    ap.add_argument("--eval-prompt-file", type=str, default=None, help="Optional UTF-8 file: one eval prompt per line.")
    ap.add_argument("--sample-max-new", type=int, default=160)
    ap.add_argument("--sample-temperature", type=float, default=0.8)
    ap.add_argument("--sample-top-k", type=int, default=50)
    ap.add_argument("--sample-top-p", type=float, default=0.92)
    ap.add_argument("--sample-repetition-penalty", type=float, default=1.12)
    ap.add_argument("--sample-no-repeat-ngram-size", type=int, default=4)
    ap.add_argument("--sample-k-steps", type=int, default=1)
    ap.add_argument("--target-vram-gb", type=float, default=None, help="If set (CUDA only), auto-tune student batch size to approach this VRAM usage.")
    ap.add_argument("--max-batch-autotune", type=int, default=32, help="Upper bound for student batch size autotune.")
    ap.add_argument("--vram-safety-margin-gb", type=float, default=3.5, help="Keep at least this much VRAM free to avoid fragmentation/OOM.")

    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--amp", type=str, default="off", choices=["off", "fp16", "bf16"], help="Mixed precision autocast mode for student training (CUDA only).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and str(args.amp) != "off"
    amp_dtype = torch.float16 if str(args.amp) == "fp16" else (torch.bfloat16 if str(args.amp) == "bf16" else None)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and str(args.amp) == "fp16"))

    # 1) Build prompt list
    prompts: List[str]
    if args.prompt_dataset:
        banned = re.compile(str(args.prompt_ban_pattern), flags=re.IGNORECASE) if args.prompt_ban_pattern else None
        prompts = load_prompts_from_hf(
            dataset_name=str(args.prompt_dataset),
            dataset_config=args.prompt_dataset_config,
            split=str(args.prompt_split),
            streaming=bool(args.prompt_streaming),
            take=int(args.num_prompts),
            seed=int(args.seed),
            max_chars=int(args.prompt_max_chars),
            banned_regex=banned,
            scan_max=int(args.prompt_scan_max),
            prompt_mode=str(args.prompt_mode),
            require_single_turn=bool(args.prompt_require_single_turn),
            prompt_user_index=int(args.prompt_user_index),
        )
    elif args.prompt_file:
        raw = Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
        prompts = [ln.strip() for ln in raw if ln.strip()]
    else:
        prompts = DEFAULT_PROMPTS[:]
    if len(prompts) == 0:
        raise RuntimeError("No prompts found. Check --prompt-dataset/--prompt-split or provide --prompt-file.")
    while len(prompts) < int(args.num_prompts):
        prompts.append(random.choice(prompts))
    prompts = prompts[: int(args.num_prompts)]
    (out_dir / "prompts_used.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")

    # 2) Teacher rows: generate or reuse cache
    teacher_rows: List[Dict[str, Any]] = []
    if args.reuse_teacher_rows:
        reuse_path = Path(str(args.reuse_teacher_rows))
        if not reuse_path.exists():
            raise FileNotFoundError(f"--reuse-teacher-rows not found: {reuse_path}")
        for ln in reuse_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            teacher_rows.append(json.loads(ln))
        if not teacher_rows:
            raise RuntimeError(f"--reuse-teacher-rows was empty: {reuse_path}")
        print(f"[teacher] reusing cached rows from {reuse_path} (n={len(teacher_rows)})", flush=True)
    else:
        # Teacher generation (GPU, fp16)
        t0 = time.time()
        teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model, use_fast=True)
        # Decoder-only models should use left padding for correct batched generation.
        teacher_tok.padding_side = "left"
        if teacher_tok.pad_token is None and teacher_tok.eos_token is not None:
            teacher_tok.pad_token = teacher_tok.eos_token
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map=None,
        ).to(device)
        teacher.eval()

        teacher_rows = teacher_generate(
            model=teacher,
            tok=teacher_tok,
            prompts=prompts,
            device=device,
            max_new_tokens=int(args.teacher_max_new),
            temperature=float(args.teacher_temperature),
            top_p=float(args.teacher_top_p),
            batch_size=int(args.teacher_batch_size),
            style_prefix=("Answer directly. No preamble. Keep it short." if str(args.teacher_style) == "short" else ""),
        )
        (out_dir / "teacher_rows.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in teacher_rows) + "\n",
            encoding="utf-8",
        )
        (out_dir / "teacher_time_sec.txt").write_text(f"{time.time() - t0:.2f}\n", encoding="utf-8")

        # Free teacher before loading student to avoid VRAM pressure.
        del teacher
        del teacher_tok
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 3) Student load
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    if gpt2_tok.pad_token is None:
        gpt2_tok.pad_token = gpt2_tok.eos_token

    ck = torch.load(args.student_ckpt, map_location=device)
    cfg = R3MRecurrentConfig(**ck["config"])
    cfg.k_train = int(args.k_train)
    cfg.max_seq_len = int(args.student_max_len)
    if bool(args.student_disable_episodic):
        cfg.episodic_enabled = False
    if bool(args.student_disable_mhc):
        cfg.mhc_enabled = False
    if args.student_episodic_slots is not None and int(args.student_episodic_slots) > 0:
        cfg.episodic_slots = int(args.student_episodic_slots)
    if args.student_write_topk is not None and int(args.student_write_topk) >= 0:
        cfg.write_topk = int(args.student_write_topk)
    if args.student_mhc_alpha_init is not None and float(args.student_mhc_alpha_init) > 0.0:
        cfg.mhc_alpha_init = float(args.student_mhc_alpha_init)
    if args.student_mhc_sinkhorn_iters is not None and int(args.student_mhc_sinkhorn_iters) > 0:
        cfg.mhc_sinkhorn_iters = int(args.student_mhc_sinkhorn_iters)
    if args.student_mhc_temperature is not None and float(args.student_mhc_temperature) > 0.0:
        cfg.mhc_temperature = float(args.student_mhc_temperature)
    student = R3MRecurrentLM(cfg).to(device)
    _load_state_dict_shape_safe(student, ck["model_state_dict"])
    student.train()

    examples = build_student_examples(gpt2_tok, teacher_rows)
    (out_dir / "student_dataset_preview.jsonl").write_text(
        "\n".join(json.dumps({"text": e["text"], "prompt_len": e["prompt_len"]}, ensure_ascii=False) for e in examples[:10]) + "\n",
        encoding="utf-8",
    )

    # Eval prompts (stable across checkpoints)
    eval_prompts = [
        "Hello!",
        "What is 2+2?",
        "What is the capital of France?",
        "Explain in one sentence what a transformer model is.",
        "Explain what overfitting is in one sentence.",
        "Write a short haiku about winter.",
        "Give three tips for staying focused while studying.",
        "Translate to German: 'The sky is blue.'",
        "Write a polite email asking for a meeting tomorrow.",
    ]
    if args.eval_prompt_file:
        raw = Path(str(args.eval_prompt_file)).read_text(encoding="utf-8").splitlines()
        loaded = [ln.strip() for ln in raw if ln.strip()]
        if loaded:
            eval_prompts = loaded
    student_generate_samples(
        model=student,
        tok=gpt2_tok,
        prompts=eval_prompts,
        device=device,
        out_path=out_dir / "student_greedy_before.txt",
        max_new_tokens=120,
    )

    opt = torch.optim.AdamW(student.parameters(), lr=float(args.student_lr), weight_decay=0.1)

    # 4) Student training loop
    losses: List[float] = []
    # Optional: increase batch size to better utilize VRAM for the student phase.
    if args.target_vram_gb is not None and float(args.target_vram_gb) > 0.0:
        tuned_bs = _autotune_student_batch_size(
            student=student,
            tok=gpt2_tok,
            examples=examples,
            device=device,
            max_len=int(args.student_max_len),
            k_steps=int(cfg.k_train),
            lr=float(args.student_lr),
            target_gb=float(args.target_vram_gb),
            max_batch=int(args.max_batch_autotune),
            start_batch=int(args.student_batch_size),
            safety_margin_gb=float(args.vram_safety_margin_gb),
        )
        args.student_batch_size = int(tuned_bs)

    had_nonfinite = False
    for step in range(int(args.student_steps)):
        # Resample if a batch ends up with no supervised tokens after masking.
        for _try in range(10):
            batch = random.sample(examples, k=min(int(args.student_batch_size), len(examples)))
            try:
                input_ids, attention_mask, labels = batch_tokenize(
                    tok=gpt2_tok, batch=batch, max_len=int(args.student_max_len), device=device
                )
                break
            except ValueError:
                continue
        else:
            raise RuntimeError("Could not sample a batch with supervised tokens after 10 tries.")
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=bool(use_amp)):
            out = student(input_ids=input_ids, attention_mask=attention_mask, labels=labels, k_steps=int(cfg.k_train))
            loss = out["loss"]
        if not torch.isfinite(loss).all():
            print(f"step {step+1}/{args.student_steps} | loss is non-finite (nan/inf). Stopping.", flush=True)
            had_nonfinite = True
            break
        # Optional write-rate regularization (discourage gate collapse / stabilize learning)
        if args.student_write_rate_target is not None and float(args.student_write_rate_lambda) > 0.0:
            avg_write = out.get("avg_write_strength", torch.tensor(0.0, device=loss.device, dtype=loss.dtype)).mean()
            tgt = torch.tensor(float(args.student_write_rate_target), device=avg_write.device, dtype=avg_write.dtype)
            loss = loss + float(args.student_write_rate_lambda) * (avg_write - tgt).pow(2)
        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()

        losses.append(float(loss.item()))
        if (step + 1) % max(1, int(args.log_every)) == 0:
            print(f"step {step+1}/{args.student_steps} | loss {loss.item():.4f}", flush=True)

        global_step = step + 1
        if int(args.save_every) > 0 and (global_step % int(args.save_every) == 0):
            ckpt_path = out_dir / f"ckpt_step_{global_step}.pt"
            torch.save(
                {
                    "config": asdict(cfg),
                    "model_state_dict": student.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scaler_state_dict": (scaler.state_dict() if scaler is not None else None),
                    "global_step": int(global_step),
                },
                ckpt_path,
            )
            print(f"[ckpt] wrote {ckpt_path}", flush=True)
        if int(args.sample_every) > 0 and (global_step % int(args.sample_every) == 0):
            sample_path = out_dir / f"samples_step_{global_step}.txt"
            student_generate_samples(
                model=student,
                tok=gpt2_tok,
                prompts=eval_prompts[: min(24, len(eval_prompts))],
                device=device,
                out_path=sample_path,
                max_new_tokens=int(args.sample_max_new),
                temperature=float(args.sample_temperature),
                top_k=int(args.sample_top_k),
                top_p=None if args.sample_top_p is None else float(args.sample_top_p),
                repetition_penalty=float(args.sample_repetition_penalty),
                no_repeat_ngram_size=int(args.sample_no_repeat_ngram_size),
                k_steps=int(args.sample_k_steps),
            )
            print(f"[eval] wrote {sample_path}", flush=True)

    (out_dir / "student_losses.json").write_text(json.dumps({"losses": losses[-200:]}, indent=2), encoding="utf-8")

    if had_nonfinite:
        print("Stopping early due to non-finite loss; skipping after-training sampling + checkpoint write.", flush=True)
        return

    # After-training samples
    student_generate_samples(
        model=student,
        tok=gpt2_tok,
        prompts=eval_prompts,
        device=device,
        out_path=out_dir / "student_greedy_after.txt",
        max_new_tokens=160,
        temperature=float(args.sample_temperature),
        top_k=int(args.sample_top_k),
        top_p=None if args.sample_top_p is None else float(args.sample_top_p),
        repetition_penalty=float(args.sample_repetition_penalty),
        no_repeat_ngram_size=int(args.sample_no_repeat_ngram_size),
        k_steps=int(args.sample_k_steps),
    )

    # Save checkpoint
    torch.save(
        {
            "config": asdict(cfg),
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scaler_state_dict": (scaler.state_dict() if scaler is not None else None),
            "global_step": int(args.student_steps),
        },
        out_dir / "student_distilled_smoke.pt",
    )
    print(f"wrote {out_dir / 'student_distilled_smoke.pt'}", flush=True)


if __name__ == "__main__":
    main()



