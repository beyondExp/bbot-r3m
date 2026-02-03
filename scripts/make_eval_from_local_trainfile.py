import json
import random
from pathlib import Path

src = Path(r"R3M/data_cache/fineweb_1000000.jsonl")
out = Path(r"R3M/data_cache/fineweb_eval_local_50k.jsonl")

rnd = random.Random(999)

# reservoir sample from the local 1M file
k = 50000
sample = []
with src.open("r", encoding="utf-8") as f:
    for i, ln in enumerate(f, 1):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        t = r.get("text", None)
        if not t:
            continue
        if len(sample) < k:
            sample.append({"text": str(t)})
        else:
            j = rnd.randrange(i)
            if j < k:
                sample[j] = {"text": str(t)}

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for r in sample:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[ok] wrote {len(sample)} rows -> {out}")
