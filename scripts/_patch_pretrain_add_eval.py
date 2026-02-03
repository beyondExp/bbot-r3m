from pathlib import Path

p = Path(r"R3M/scripts/pretrain_hybrid_ssm.py")
lines = p.read_text(encoding="utf-8-sig").splitlines(True)

# 1) ensure import math
if not any(l.strip() == "import math" for l in lines):
    for i,l in enumerate(lines):
        if l.strip() == "import json":
            lines.insert(i+1, "import math\n")
            break

# 2) add eval_loss helper before cuda_mem_str
if not any(l.startswith("def eval_loss") or "def eval_loss" in l for l in lines):
    idx = None
    for i,l in enumerate(lines):
        if l.startswith("def cuda_mem_str"):
            idx = i
            break
    if idx is None:
        raise SystemExit("failed to find def cuda_mem_str")

    inject = [
        "\n",
        "@torch.no_grad()\n",
        "def eval_loss(model, dl, *, device: torch.device, amp: str) -> float:\n",
        "    model.eval()\n",
        "    use_amp = device.type == 'cuda' and amp != 'off'\n",
        "    amp_dtype = torch.float16 if amp == 'fp16' else (torch.bfloat16 if amp == 'bf16' else None)\n",
        "    total = 0.0\n",
        "    n = 0\n",
        "    for batch in dl:\n",
        "        input_ids = batch['input_ids'].to(device, non_blocking=True)\n",
        "        labels = batch['labels'].to(device, non_blocking=True)\n",
        "        with torch.autocast(device_type=('cuda' if device.type=='cuda' else 'cpu'), dtype=amp_dtype, enabled=bool(use_amp)):\n",
        "            out = model(input_ids=input_ids, attention_mask=None, labels=labels)\n",
        "            loss = out['loss']\n",
        "        total += float(loss.item())\n",
        "        n += 1\n",
        "    model.train()\n",
        "    return total / max(1, n)\n",
        "\n",
        "\n",
    ]
    lines[idx:idx] = inject

# 3) add eval argparse args after --save-every
if not any("--eval-jsonl" in l for l in lines):
    ins_idx = None
    for i,l in enumerate(lines):
        if "ap.add_argument(\"--save-every\"" in l or "ap.add_argument(\"--save-every\"," in l or "ap.add_argument(\"--save-every\"" in l:
            ins_idx = i+1
            break
    if ins_idx is None:
        raise SystemExit("failed to find --save-every argparse")

    inject = [
        "\n",
        "    # Eval (optional)\n",
        "    ap.add_argument(\"--eval-jsonl\", type=str, default=\"\")\n",
        "    ap.add_argument(\"--eval-every\", type=int, default=0, help=\"Run eval every N steps (0=off)\")\n",
        "    ap.add_argument(\"--eval-take-texts\", type=int, default=2000)\n",
        "    ap.add_argument(\"--eval-max-pack-tokens\", type=int, default=400000)\n",
        "    ap.add_argument(\"--eval-seq-len\", type=int, default=256)\n",
        "    ap.add_argument(\"--eval-batch-size\", type=int, default=8)\n",
    ]
    lines[ins_idx:ins_idx] = inject

# 4) insert eval hook after print(msg, flush=True)
if not any("[eval] step=" in l for l in lines):
    idx = None
    for i,l in enumerate(lines):
        if "print(msg, flush=True)" in l:
            idx = i+1
            indent = l.split("print")[0]
            break
    if idx is None:
        raise SystemExit("failed to find print(msg, flush=True)")

    # indent inside the log-every block
    ind = indent
    hook = [
        "\n",
        f"{ind}# Optional held-out eval\n",
        f"{ind}if int(args.eval_every) > 0 and str(args.eval_jsonl).strip() and (step % int(args.eval_every) == 0):\n",
        f"{ind}    try:\n",
        f"{ind}        ev_texts = load_texts_from_jsonl(str(args.eval_jsonl), take=int(args.eval_take_texts))\n",
        f"{ind}        ev_ds = PackedTextDataset(tokenizer=tok, texts=ev_texts, block_size=int(args.eval_seq_len), max_tokens=int(args.eval_max_pack_tokens), seed=123)\n",
        f"{ind}        ev_dl = DataLoader(ev_ds, batch_size=int(args.eval_batch_size), shuffle=False, drop_last=False, collate_fn=collate, pin_memory=(device.type=='cuda'))\n",
        f"{ind}        ev_loss = eval_loss(model, ev_dl, device=device, amp=str(args.amp))\n",
        f"{ind}        ev_ppl = math.exp(min(20.0, ev_loss))\n",
        f"{ind}        print(f\"[eval] step={{{{step}}}} loss={{{{ev_loss:.4f}}}} ppl~={{{{ev_ppl:.2f}}}} blocks={{{{len(ev_ds)}}}}\", flush=True)\n",
        f"{ind}    except Exception as e:\n",
        f"{ind}        print(f\"[eval] failed: {{{{e}}}}\", flush=True)\n",
    ]
    lines[idx:idx] = hook

p.write_text("".join(lines), encoding="utf-8")
print("[ok] patched", p)
