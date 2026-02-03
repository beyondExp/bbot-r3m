#!/usr/bin/env python3
"""Export (stream) Hugging Face datasets into a jsonl: {"text": "..."}."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from datasets import load_dataset

R3M_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if R3M_ROOT not in sys.path:
    sys.path.insert(0, R3M_ROOT)

from r3m.utils.env import load_hf_token_from_default_locations


def _safe_write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_fineweb_text(ds_name: str, split: str) -> Iterator[str]:
    ds = load_dataset(ds_name, split=split, streaming=True)
    for r in ds:
        t = r.get("text", None)
        if t:
            yield str(t)


def iter_ultrachat_text(ds_name: str, split: str) -> Iterator[str]:
    ds = load_dataset(ds_name, split=split, streaming=True)
    for r in ds:
        msgs = r.get("messages", None)
        if not isinstance(msgs, list) or not msgs:
            continue
        parts: List[str] = []
        for m in msgs:
            role = str(m.get("role", "")).strip().lower()
            content = str(m.get("content", "")).strip()
            if not content:
                continue
            if role in ("user", "human", "prompter"):
                parts.append(f"User: {content}")
            elif role in ("assistant", "gpt", "assistant_response"):
                parts.append(f"Assistant: {content}")
            else:
                parts.append(content)
        if parts:
            yield "\n".join(parts).strip() + "\n"


def iter_oasst1_text(ds_name: str, split: str) -> Iterator[str]:
    ds = load_dataset(ds_name, split=split, streaming=True)
    for r in ds:
        role = str(r.get("role", "")).strip().lower()
        text = str(r.get("text", "")).strip()
        if not text:
            continue
        if role == "prompter":
            yield f"User: {text}\n"
        elif role == "assistant":
            yield f"Assistant: {text}\n"
        else:
            yield text + "\n"


def iter_generic_text(ds_name: str, split: str, text_field: str = "text") -> Iterator[str]:
    ds = load_dataset(ds_name, split=split, streaming=True)
    for r in ds:
        t = r.get(text_field, None)
        if t:
            yield str(t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--skip", type=int, default=0, help="Skip the first N source rows before exporting (streaming).")
    ap.add_argument("--take", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-buffer", type=int, default=0)
    args = ap.parse_args()

    load_hf_token_from_default_locations(override=False)

    ds_name = str(args.dataset).strip()
    split = str(args.split).strip()
    out = Path(args.out)

    if ds_name in ("HuggingFaceFW/fineweb", "HuggingFaceFW/fineweb-edu"):
        it = iter_fineweb_text(ds_name, split)
    elif ds_name == "HuggingFaceH4/ultrachat_200k":
        it = iter_ultrachat_text(ds_name, split)
    elif ds_name == "OpenAssistant/oasst1":
        it = iter_oasst1_text(ds_name, split)
    else:
        it = iter_generic_text(ds_name, split)

    def rows() -> Iterator[Dict[str, Any]]:
        n = 0
        skip = max(0, int(args.skip))
        buf_n = int(args.shuffle_buffer)
        if buf_n > 0:
            rnd = random.Random(int(args.seed))
            buf: List[str] = []
            if skip > 0:
                skipped = 0
                for _t in it:
                    skipped += 1
                    if skipped >= skip:
                        break
            for t in it:
                buf.append(t)
                if len(buf) >= buf_n:
                    rnd.shuffle(buf)
                    while buf:
                        yield_t = str(buf.pop()).strip()
                        if not yield_t:
                            continue
                        yield {"text": yield_t}
                        n += 1
                        if int(args.take) > 0 and n >= int(args.take):
                            return
            rnd.shuffle(buf)
            for t in buf:
                yield_t = str(t).strip()
                if not yield_t:
                    continue
                yield {"text": yield_t}
                n += 1
                if int(args.take) > 0 and n >= int(args.take):
                    return
            return

        if skip > 0:
            skipped = 0
            for _t in it:
                skipped += 1
                if skipped >= skip:
                    break

        for t in it:
            t = str(t).strip()
            if not t:
                continue
            yield {"text": t}
            n += 1
            if int(args.take) > 0 and n >= int(args.take):
                return

    n = _safe_write_jsonl(out, rows())
    print(f"[ok] wrote {n} rows -> {out}")


if __name__ == "__main__":
    main()

