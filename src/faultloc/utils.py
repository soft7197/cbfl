from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def safe_relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()

def is_probably_noise_dir(p: Path) -> bool:
    parts = {x.lower() for x in p.parts}
    noise = {".git", "__pycache__", ".venv", "venv", "env", "build", "dist", ".mypy_cache", ".pytest_cache"}
    return any(x in parts for x in noise)

def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))
