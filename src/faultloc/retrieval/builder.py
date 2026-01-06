from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .text_units import TextUnit

def build_units(repo_root: Path, repo_index: Dict[str, Any], max_file_chars: int = 8000) -> List[TextUnit]:
    # Build FILE units and FUNCTION units (with small body excerpts) from symbol index.
    out: List[TextUnit] = []
    file_index = repo_index["file_index"]
    sym_index = repo_index["symbol_index"]["files"]
    path_mod = file_index.get("path_module_map", {})

    for rel, fentry in sym_index.items():
        p = repo_root / rel
        try:
            src = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # FILE unit: take header + imports + docstring-ish chunk
        head = src[:max_file_chars]
        out.append(TextUnit(
            uid=f"FILE::{rel}",
            kind="FILE",
            file_path=rel,
            qualname=None,
            text=head,
            meta={"module": path_mod.get(rel, "")},
        ))

        # FUNCTION units: include signature lines and a bounded excerpt
        for fn in fentry["symbols"]["functions"]:
            sp = fn["span"]
            start = max(1, int(sp["start_line"]))
            end = min(int(sp["end_line"]), start + 120)  # bounded excerpt
            lines = src.splitlines()
            excerpt = "\n".join(lines[start-1:end])
            out.append(TextUnit(
                uid=f"FUNC::{rel}::{fn.get('qualname') or fn.get('name')}",
                kind="FUNCTION",
                file_path=rel,
                qualname=fn.get("qualname"),
                text=excerpt,
                meta={"span": sp, "name": fn.get("name")},
            ))
    return out
