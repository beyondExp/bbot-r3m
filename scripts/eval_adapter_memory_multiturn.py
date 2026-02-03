import argparse
import os
import random
import re
import string
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

# Ensure R3M/ is on sys.path\nR3M_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))\nif R3M_ROOT not in sys.path:\n    sys.path.insert(0, R3M_ROOT)\n\nfrom r3m.models.adapter_hf import R3MHFAdapterConfig, R3MHFAdapterLM


def _rand_key(n: int = 10) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(n))


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _extract_code(s: str) -> Optional[str]:
    m = re.search(r"\b[A-Z0-9]{10}\b", s.upper())
    return m.group(0) if m else None


def _hard_stop_on_user(text: str) -> str:
    # If the model starts inventing a new user turn, cut it.
    for marker in ["\nUser:", "\nuser:", "\nHuman:", "\nhuman:"]:
        idx = text.find(marker)
        if idx != -1:
            return text[:idx].strip()
    return text.strip()


@torch.no_grad()
def answer_one_turn(
    model: R3MHFAdapterLM,
    tok,
    user_text: str,
    *,
    device: torch.device,
    mem_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    max_ctx_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    memory_enabled: bool,
) -> Tuple[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    One conversation turn with *truncated visible context* but persistent mem_state.
    """
    if not memory_enabled:
        mem_state = None

    prompt = f"User: {user_text}\nAssistant:"
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if ids.size(1) > int(max_ctx_tokens):
        ids = ids[:, -int(max_ctx_tokens) :]
    attn = torch.ones_like(ids, device=device)

    # Generate with persistent memory state
    gen_ids, mem_state2 = model.generate_with_memory_state(
        ids,
        attention_mask=attn,
        mem_state=mem_state if memory_enabled else None,
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        eos_token_id=tok.eos_token_id,
        update_memory=True,
    )
    gen = gen_ids[0].tolist()
    cont = tok.decode(gen[len(ids[0]) :], skip_special_tokens=True)
    cont = _hard_stop_on_user(cont)
    return cont.strip(), (mem_state2 if memory_enabled else model.init_memory_state(batch_size=int(ids.size(0)), device=device))


@torch.no_grad()
def answer_one_turn_base(
    base_model,
    tok,
    user_text: str,
    *,
    device: torch.device,
    max_ctx_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    prompt = f"User: {user_text}\nAssistant:"
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if ids.size(1) > int(max_ctx_tokens):
        ids = ids[:, -int(max_ctx_tokens) :]
    attn = torch.ones_like(ids, device=device)
    # generation with sampling
    out = base_model.generate(
        input_ids=ids,
        attention_mask=attn,
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )[0].tolist()
    cont = tok.decode(out[len(ids[0]) :], skip_special_tokens=True)
    return _hard_stop_on_user(cont).strip()


@torch.no_grad()
def kl_at_query(
    model: R3MHFAdapterLM,
    tok,
    query: str,
    *,
    device: torch.device,
    mem_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    max_ctx_tokens: int,
) -> float:
    """
    Measure whether memory changes the model's distribution at the query prompt.
    Returns KL( p_mem || p_empty ) at the last position.
    """
    prompt = f"User: {query}\nAssistant:"
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if ids.size(1) > int(max_ctx_tokens):
        ids = ids[:, -int(max_ctx_tokens) :]
    attn = torch.ones_like(ids, device=device)

    logits_mem, _ = model.encode_with_memory_state(ids, attn, mem_state, update_memory=False)
    logits_empty, _ = model.encode_with_memory_state(ids, attn, None, update_memory=False)
    p = torch.log_softmax(logits_mem[:, -1, :].float(), dim=-1)
    q = torch.log_softmax(logits_empty[:, -1, :].float(), dim=-1)
    kl = torch.sum(torch.exp(p) * (p - q), dim=-1).mean()
    return float(kl.item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--adapter-ckpt", type=str, required=True, help="Path to adapter_final.pt or adapter_ckpt_step_*.pt")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--max-ctx-tokens", type=int, default=64, help="Visible context window (simulate forgetting).")
    ap.add_argument("--distractor-turns", type=int, default=6, help="How many filler turns between store and query.")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reset-each-trial", action="store_true", help="Reset memory state at the start of every trial.")
    ap.add_argument("--print-examples", type=int, default=3, help="How many example transcripts to print.")
    args = ap.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(args.base), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = R3MHFAdapterConfig(base_model_name=str(args.base), freeze_base=True, episodic_slots=64)
    model = R3MHFAdapterLM(cfg, torch_dtype=(torch.bfloat16 if device.type == "cuda" else None)).to(device)
    ck = torch.load(str(args.adapter_ckpt), map_location=device)
    if "adapter_state_dict" not in ck:
        raise RuntimeError("Adapter checkpoint missing adapter_state_dict")
    model.load_adapter_state_dict(ck["adapter_state_dict"], strict=False)
    model.eval()

    # Base model (no adapter/memory)
    base_model = AutoModelForCausalLM.from_pretrained(
        str(args.base),
        torch_dtype=(torch.bfloat16 if device.type == "cuda" else torch.float32),
        low_cpu_mem_usage=True,
        device_map=None,
    ).to(device)
    base_model.eval()

    def run(mode: str) -> Tuple[float, float]:
        """
        Returns (exact_match_accuracy, avg_kl_at_query).
        """
        correct = 0
        kl_vals: List[float] = []
        mem_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        printed = 0
        for ti in range(int(args.trials)):
            if bool(args.reset_each_trial):
                mem_state = None
            key = _rand_key(10)
            # Store turn: use non-guessable secret.
            store = f"Remember this secret code exactly: {key}. Reply only with OK."
            if mode == "base":
                ans_store = answer_one_turn_base(
                    base_model,
                    tok,
                    store,
                    device=device,
                    max_ctx_tokens=int(args.max_ctx_tokens),
                    max_new_tokens=int(args.max_new),
                    temperature=float(args.temperature),
                    top_k=int(args.top_k),
                    top_p=float(args.top_p),
                )
            else:
                # Toggle episodic on/off for adapter runs
                prev = bool(model.cfg.episodic_enabled)
                model.cfg.episodic_enabled = (mode == "mem_on")
                ans_store, mem_state = answer_one_turn(
                    model,
                    tok,
                    store,
                    device=device,
                    mem_state=mem_state,
                    max_ctx_tokens=int(args.max_ctx_tokens),
                    max_new_tokens=int(args.max_new),
                    temperature=float(args.temperature),
                    top_k=int(args.top_k),
                    top_p=float(args.top_p),
                    memory_enabled=(mode == "mem_on"),
                )
                model.cfg.episodic_enabled = prev

            # Distractors (should not contain the key)
            for j in range(int(args.distractor_turns)):
                distract = f"Filler question {j+1}: tell me a random fruit."
                if mode == "base":
                    _ = answer_one_turn_base(
                        base_model,
                        tok,
                        distract,
                        device=device,
                        max_ctx_tokens=int(args.max_ctx_tokens),
                        max_new_tokens=int(args.max_new),
                        temperature=float(args.temperature),
                        top_k=int(args.top_k),
                        top_p=float(args.top_p),
                    )
                else:
                    prev = bool(model.cfg.episodic_enabled)
                    model.cfg.episodic_enabled = (mode == "mem_on")
                    _, mem_state = answer_one_turn(
                        model,
                        tok,
                        distract,
                        device=device,
                        mem_state=mem_state,
                        max_ctx_tokens=int(args.max_ctx_tokens),
                        max_new_tokens=int(args.max_new),
                        temperature=float(args.temperature),
                        top_k=int(args.top_k),
                        top_p=float(args.top_p),
                        memory_enabled=(mode == "mem_on"),
                    )
                    model.cfg.episodic_enabled = prev

            # Query: ask for the secret code
            query = "What is the secret code? Reply with just the code."
            if mode == "base":
                out = answer_one_turn_base(
                    base_model,
                    tok,
                    query,
                    device=device,
                    max_ctx_tokens=int(args.max_ctx_tokens),
                    max_new_tokens=int(args.max_new),
                    temperature=float(args.temperature),
                    top_k=int(args.top_k),
                    top_p=float(args.top_p),
                )
            else:
                # KL diagnostic: does memory affect logits at the query?
                if mode == "mem_on":
                    kl_vals.append(
                        kl_at_query(
                            model,
                            tok,
                            query,
                            device=device,
                            mem_state=mem_state,
                            max_ctx_tokens=int(args.max_ctx_tokens),
                        )
                    )
                # For the query step, use GREEDY decoding to avoid sampling noise.
                prev = bool(model.cfg.episodic_enabled)
                model.cfg.episodic_enabled = (mode == "mem_on")
                prompt = f"User: {query}\nAssistant:"
                ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                if ids.size(1) > int(args.max_ctx_tokens):
                    ids = ids[:, -int(args.max_ctx_tokens) :]
                attn = torch.ones_like(ids, device=device)
                gen_ids, mem_state = model.generate_with_memory_state(
                    ids,
                    attention_mask=attn,
                    mem_state=mem_state if (mode == "mem_on") else None,
                    max_new_tokens=int(args.max_new),
                    do_sample=False,
                    temperature=1.0,
                    top_k=1,
                    top_p=None,
                    eos_token_id=tok.eos_token_id,
                    update_memory=True,
                )
                gen = gen_ids[0].tolist()
                out = _hard_stop_on_user(tok.decode(gen[len(ids[0]) :], skip_special_tokens=True)).strip()
                model.cfg.episodic_enabled = prev
            got = _extract_code(out) or ""
            if got == key:
                correct += 1
            if printed < int(args.print_examples):
                printed += 1
                print(f"\n--- TRIAL {ti+1} ({mode}) ---")
                print(f"secret: {key}")
                if mode == "base":
                    print(f"store_ans: {ans_store}")
                else:
                    print(f"store_ans: {ans_store}")
                print(f"query_out: {out}")

        acc = correct / float(args.trials)
        avg_kl = float(sum(kl_vals) / max(1, len(kl_vals)))
        return acc, avg_kl

    acc_base, _ = run("base")
    acc_memoff, _ = run("mem_off")
    acc_memon, kl_memon = run("mem_on")
    print(f"\nbase_only accuracy: {acc_base:.3f}")
    print(f"adapter_mem_off accuracy: {acc_memoff:.3f}")
    print(f"adapter_mem_on accuracy: {acc_memon:.3f}")
    print(f"adapter_mem_on avg_kl_at_query: {kl_memon:.6f}")


if __name__ == "__main__":
    main()




