from pathlib import Path
import re

p = Path(r"R3M\scripts\pretrain_hybrid_ssm.py")
s = p.read_text(encoding="utf-8")

def require_contains(needle: str, name: str):
    if needle not in s:
        raise SystemExit(f"missing pattern for {name}")

# 1) PackedTextDataset __getitem__
require_contains('attn = torch.ones_like(x)', 'attn creation (PackedTextDataset)')
s = s.replace(
    '        attn = torch.ones_like(x)\n        return {"input_ids": x, "labels": y, "attention_mask": attn}\n',
    '        return {"input_ids": x, "labels": y}\n',
)

# 2) PackedJsonlStream yield
require_contains('yield {"input_ids": x, "labels": y, "attention_mask": attn}', 'attn yield (PackedJsonlStream)')
s = s.replace(
    '                attn = torch.ones_like(x)\n                yield {"input_ids": x, "labels": y, "attention_mask": attn}\n',
    '                yield {"input_ids": x, "labels": y}\n',
)

# 3) collate
require_contains('attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)', 'collate attention_mask stack')
s = s.replace(
    '    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)\n    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}\n',
    '    return {"input_ids": input_ids, "labels": labels}\n',
)

# 4) training loop: remove attention_mask to(device)
pat = r"\n\s*attention_mask\s*=\s*batch\[\"attention_mask\"\]\.to\(device[^\)]*\)\n"
s, n = re.subn(pat, "\n", s, count=1)
if n != 1:
    raise SystemExit("failed to remove attention_mask device line")

# Replace forward call
if 'out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)' not in s:
    raise SystemExit('missing model call with attention_mask=attention_mask')
s = s.replace(
    'out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)',
    'out = model(input_ids=input_ids, attention_mask=None, labels=labels)'
)

p.write_text(s, encoding="utf-8")
print('[ok] patched', p)
