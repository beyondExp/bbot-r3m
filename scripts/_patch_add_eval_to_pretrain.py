import json
import math
import re
from pathlib import Path

p = Path(r"R3M/scripts/pretrain_hybrid_ssm.py")
s = p.read_text(encoding="utf-8")

# add eval args after training args
if "--eval-jsonl" not in s:
    # insert into argparse section after --save-every
    pat = r"(\s*ap\.add_argument\(\"--save-every\"[^\n]*\)\n)"
    m = re.search(pat, s)
    if not m:
        raise SystemExit("failed to find --save-every argparse line")
    insert = m.group(1) + (
        "\n"
        "    # Eval (optional)\n"
        "    ap.add_argument(\"--eval-jsonl\", type=str, default=\"\")\n"
        "    ap.add_argument(\"--eval-every\", type=int, default=0, help=\"Run eval every N steps (0=off)\")\n"
        "    ap.add_argument(\"--eval-take-texts\", type=int, default=2000)\n"
        "    ap.add_argument(\"--eval-max-pack-tokens\", type=int, default=400000)\n"
        "    ap.add_argument(\"--eval-seq-len\", type=int, default=256)\n"
        "    ap.add_argument(\"--eval-batch-size\", type=int, default=8)\n"
    )
    s = s[: m.start(1)] + insert + s[m.end(1) :]

# ensure we have helper to load texts (already present) and PackedTextDataset (already)
# add eval helpers if absent
if "def eval_loss(" not in s:
    anchor = "def cuda_mem_str(device: torch.device) -> str:"
    i = s.find(anchor)
    if i == -1:
        raise SystemExit("failed to find cuda_mem_str")
    # inject before cuda_mem_str
    inject = (
        "@torch.no_grad()\n"
        "def eval_loss(model, dl, *, device: torch.device, amp: str) -> float:\n"
        "    model.eval()\n"
        "    use_amp = device.type == 'cuda' and amp != 'off'\n"
        "    amp_dtype = torch.float16 if amp == 'fp16' else (torch.bfloat16 if amp == 'bf16' else None)\n"
        "    total = 0.0\n"
        "    n = 0\n"
        "    for batch in dl:\n"
        "        input_ids = batch['input_ids'].to(device, non_blocking=True)\n"
        "        labels = batch['labels'].to(device, non_blocking=True)\n"
        "        with torch.autocast(device_type=('cuda' if device.type=='cuda' else 'cpu'), dtype=amp_dtype, enabled=bool(use_amp)):\n"
        "            out = model(input_ids=input_ids, attention_mask=None, labels=labels)\n"
        "            loss = out['loss']\n"
        "        total += float(loss.item())\n"
        "        n += 1\n"
        "    model.train()\n"
        "    return total / max(1, n)\n\n\n"
    )
    s = s[:i] + inject + s[i:]

# In the training loop, after logging, add optional eval
if "[eval]" not in s:
    # locate the logging print block
    pat = r"(\n\s*if step % max\(1, int\(args\.log-every\)\) == 0:\n[\s\S]*?torch\.cuda\.reset_peak_memory_stats\(\)\n)"
    m = re.search(pat, s)
    if not m:
        # older version might not reset peak each interval; fall back to after print(msg)
        pat2 = r"(\n\s*if step % max\(1, int\(args\.log-every\)\) == 0:[\s\S]*?print\(msg, flush=True\)\n)"
        m = re.search(pat2, s)
    if not m:
        raise SystemExit("failed to find log-every block")

    block = m.group(1)
    add = (
        "\n"
        "        # Optional held-out eval\n"
        "        if int(args.eval_every) > 0 and str(args.eval_jsonl).strip() and (step % int(args.eval_every) == 0):\n"
        "            try:\n"
        "                ev_texts = load_texts_from_jsonl(str(args.eval_jsonl), take=int(args.eval_take_texts))\n"
        "                ev_ds = PackedTextDataset(tokenizer=tok, texts=ev_texts, block_size=int(args.eval_seq_len), max_tokens=int(args.eval_max_pack_tokens), seed=123)\n"
        "                ev_dl = DataLoader(ev_ds, batch_size=int(args.eval_batch_size), shuffle=False, drop_last=False, collate_fn=collate, pin_memory=(device.type=='cuda'))\n"
        "                ev_loss = eval_loss(model, ev_dl, device=device, amp=str(args.amp))\n"
        "                ev_ppl = math.exp(min(20.0, ev_loss))\n"
        "                print(f\"[eval] step={step} loss={ev_loss:.4f} ppl~{ev_ppl:.2f} blocks={len(ev_ds)}\", flush=True)\n"
        "            except Exception as e:\n"
        "                print(f\"[eval] failed: {e}\", flush=True)\n"
    )

    s = s.replace(block, block + add, 1)

# ensure imports: math is needed
if "import math" not in s:
    s = s.replace("import json\n", "import json\nimport math\n", 1)

p.write_text(s, encoding="utf-8")
print("[ok] patched", p)
