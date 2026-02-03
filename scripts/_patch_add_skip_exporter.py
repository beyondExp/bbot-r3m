from pathlib import Path
import re

p = Path(r"R3M/scripts/export_hf_to_jsonl.py")
s = p.read_text(encoding="utf-8")

if "--skip" in s:
    print("[ok] already has --skip")
    raise SystemExit(0)

m = re.search(r"(ap\.add_argument\(\"--out\"[^\n]*\)\n)", s)
if not m:
    raise SystemExit("failed to find --out argparse line")

insert = m.group(1) + "    ap.add_argument(\"--skip\", type=int, default=0, help=\"Skip the first N source rows before exporting (streaming).\")\n"
s = s[: m.start(1)] + insert + s[m.end(1) :]

m2 = re.search(
    r"def rows\(\) -> Iterator\[Dict\[str, Any\]\]:\n\s*n = 0\n\s*buf_n = int\(args\.shuffle_buffer\)\n",
    s,
)
if not m2:
    raise SystemExit("failed to find rows() header")

rep = "def rows() -> Iterator[Dict[str, Any]]:\n        n = 0\n        skip = max(0, int(args.skip))\n        buf_n = int(args.shuffle_buffer)\n"
s = s[: m2.start()] + rep + s[m2.end() :]

needle = "            buf: List[str] = []\n"
idx = s.find(needle)
if idx == -1:
    raise SystemExit("failed to find buf init")

ins = (
    needle
    + "            if skip > 0:\n"
    + "                skipped = 0\n"
    + "                for _t in it:\n"
    + "                    skipped += 1\n"
    + "                    if skipped >= skip:\n"
    + "                        break\n"
)

s = s.replace(needle, ins, 1)

needle2 = "\n        for t in it:\n"
idx2 = s.find(needle2)
if idx2 == -1:
    raise SystemExit("failed to find no-shuffle for loop")

ins2 = (
    "\n        if skip > 0:\n"
    "            skipped = 0\n"
    "            for _t in it:\n"
    "                skipped += 1\n"
    "                if skipped >= skip:\n"
    "                    break\n"
    "\n        for t in it:\n"
)

s = s.replace(needle2, ins2, 1)

p.write_text(s, encoding="utf-8")
print("[ok] patched", p)
