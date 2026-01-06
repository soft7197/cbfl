from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

def summarize_symbol(source: str, symbol: Dict[str, Any], max_lines: int = 80) -> Dict[str, Any]:
    sp = symbol["span"]
    start = int(sp["start_line"]); end = int(sp["end_line"])
    lines = source.splitlines()
    excerpt = "\n".join(lines[start-1: min(end, start-1+max_lines)])
    # quick features
    calls = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", excerpt)
    uniq_calls = []
    seen=set()
    for c in calls:
        if c not in seen:
            seen.add(c); uniq_calls.append(c)
        if len(uniq_calls) >= 20:
            break
    return {
        "qualname": symbol.get("qualname"),
        "name": symbol.get("name"),
        "kind": symbol.get("kind"),
        "span": sp,
        "calls": uniq_calls,
        "excerpt": excerpt[:5000],
    }

def build_candidate_summaries(repo_root: Path, symbols_by_file: Dict[str, List[Dict[str, Any]]], max_candidates: int = 30) -> List[Dict[str, Any]]:
    # Flatten candidates and create short summaries for LLM reasoning.
    out: List[Dict[str, Any]] = []
    cid = 1
    for rel, syms in symbols_by_file.items():
        p = repo_root / rel
        try:
            src = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for s in syms:
            out.append({"candidate_id": cid, "file": rel, "summary": summarize_symbol(src, s)})
            cid += 1
            if len(out) >= max_candidates:
                return out
    return out
