from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


def _parse_env_text(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in (text or "").splitlines():
        ln = raw.strip()
        if not ln or ln.startswith("#"):
            continue
        if "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            out[k] = v
    return out


def load_env_file(path: str | Path, *, override: bool = False) -> Dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    parsed = _parse_env_text(p.read_text(encoding="utf-8"))
    for k, v in parsed.items():
        if override or (k not in os.environ):
            os.environ[k] = v
    return parsed


def load_hf_token_from_default_locations(*, override: bool = False) -> Optional[str]:
    here = Path(__file__).resolve()
    r3m_root = here.parents[2]  # R3M/
    repo_root = r3m_root.parent

    candidates = [
        r3m_root / ".env",
        repo_root / "research_architectures" / ".env",
        repo_root / ".env",
    ]
    for p in candidates:
        load_env_file(p, override=override)
        tok = os.environ.get("HF_TOKEN", "").strip()
        if tok:
            return tok
    return None
