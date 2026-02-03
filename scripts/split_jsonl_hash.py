import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-eval", required=True)
    ap.add_argument("--eval-pct", type=float, default=0.02)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--salt", type=str, default="r3m-split-v1")
    args = ap.parse_args()

    inp = Path(args.in_jsonl)
    out_tr = Path(args.out_train)
    out_ev = Path(args.out_eval)
    out_tr.parent.mkdir(parents=True, exist_ok=True)
    out_ev.parent.mkdir(parents=True, exist_ok=True)

    eval_pct = float(args.eval_pct)
    if not (0.0 < eval_pct < 1.0):
        raise SystemExit("--eval-pct must be in (0,1)")

    n = 0
    n_tr = 0
    n_ev = 0

    with inp.open("r", encoding="utf-8") as fi, out_tr.open("w", encoding="utf-8") as ftr, out_ev.open(
        "w", encoding="utf-8"
    ) as fev:
        for ln in fi:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            t = r.get("text", None)
            if not t:
                continue
            t = str(t)

            h = hashlib.sha1((args.salt + "\n" + t).encode("utf-8", errors="ignore")).digest()
            # map first 8 bytes to [0,1)
            x = int.from_bytes(h[:8], "big") / float(2**64)
            out = fev if x < eval_pct else ftr
            out.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            if x < eval_pct:
                n_ev += 1
            else:
                n_tr += 1
            n += 1
            if args.max_rows and n >= int(args.max_rows):
                break

    print(f"[ok] split rows={n} train={n_tr} eval={n_ev} eval_pct={eval_pct}")
    print(f"[train] {out_tr}")
    print(f"[eval ] {out_ev}")


if __name__ == "__main__":
    main()
