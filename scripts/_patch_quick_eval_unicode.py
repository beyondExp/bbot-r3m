import io
from pathlib import Path

p = Path(r"R3M/scripts/quick_eval_hybrid.py")
raw = p.read_text(encoding="utf-8")

# ensure stdout/stderr utf-8 reconfigure
if "sys.stdout.reconfigure" not in raw:
    raw = raw.replace(
        "import random\n",
        "import random\nimport sys\n",
        1,
    )
    marker = "def main() -> None:\n"
    i = raw.find(marker)
    if i == -1:
        raise SystemExit("failed to find main()")
    j = i + len(marker)
    inject = (
        "    # Windows terminals often default to cp1252; ensure we can print arbitrary model text safely.\n"
        "    try:\n"
        "        sys.stdout.reconfigure(encoding=\"utf-8\", errors=\"replace\")\n"
        "        sys.stderr.reconfigure(encoding=\"utf-8\", errors=\"replace\")\n"
        "    except Exception:\n"
        "        pass\n\n"
    )
    raw = raw[:j] + inject + raw[j:]

# bump eval print precision
raw = raw.replace(
    'print(f"[eval] avg_loss={avg_loss:.4f} ppl~{ppl:.2f} | blocks={len(ds)}", flush=True)\n',
    'print(f"[eval] avg_loss={avg_loss:.8f} ppl~{ppl:.4f} | blocks={len(ds)}", flush=True)\n',
)

p.write_text(raw, encoding="utf-8")
print("[ok] patched", p)
