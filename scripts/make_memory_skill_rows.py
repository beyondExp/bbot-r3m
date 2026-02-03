import argparse
import json
import random
import string
from pathlib import Path
from typing import Dict, List


def rand_code(n: int = 10) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(n))


def build_rows_for_code(code: str, distractors: int, include_ok_row: bool) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if include_ok_row:
        rows.append({"user": f"Remember this secret code exactly: {code}. Reply only with OK.", "teacher_answer": "OK"})
    convo: List[str] = []
    convo.append(f"Remember this secret code exactly: {code}. Reply only with OK.")
    for i in range(distractors):
        convo.append(f"Filler question {i+1}: say a random fruit.")
    convo.append("What is the secret code? Reply with just the code.")
    rows.append({"user": "\n".join(convo), "teacher_answer": code})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="R3M/outputs/memory_skill_v2/teacher_rows.jsonl")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--distractors", type=int, default=6)
    ap.add_argument("--include-ok-row", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(int(args.seed))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for _ in range(int(args.n)):
        code = rand_code(10)
        rows.extend(build_rows_for_code(code=code, distractors=int(args.distractors), include_ok_row=bool(args.include_ok_row)))

    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {out_path} (n={len(rows)})")


if __name__ == "__main__":
    main()

