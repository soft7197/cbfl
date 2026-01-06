from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .utils import clamp

def resolve_file_candidates(file_candidates: List[str], file_index: Dict[str, Any]) -> List[str]:
    all_files = set(file_index.get("all_py_files", []))
    basename_map = file_index.get("basename_map", {})
    suffix_map = file_index.get("suffix_map", {})
    module_map = file_index.get("module_path_map", {})

    out: List[str] = []
    def add(p: str):
        if p not in out:
            out.append(p)

    for cand in file_candidates:
        c = cand.strip().replace("\\", "/")
        if not c:
            continue

        # direct match
        if c in all_files:
            add(c); continue

        # module candidate (dotted -> exact file path if known)
        if c in module_map:
            add(module_map[c]); continue

        bn = c.split("/")[-1]

        # IMPORTANT: prevent exploding to all packages
        if bn == "__init__.py":
            # Only accept if caller specified a path/suffix (e.g., "pkg/__init__.py")
            if "/" in c and c in suffix_map:
                for p in suffix_map[c]:
                    add(p)
            # Otherwise ignore plain "__init__.py"
            continue

        # basename expansion (safe for normal files)
        if bn in basename_map:
            for p in basename_map[bn]:
                add(p)

        # suffix match
        if c in suffix_map:
            for p in suffix_map[c]:
                add(p)

        # add .py if missing
        if not c.endswith(".py") and "/" in c:
            c2 = c + ".py"
            if c2 in suffix_map:
                for p in suffix_map[c2]:
                    add(p)

    return out


def score_symbol_match(problem_statement: str, file_path: str, qualname: Optional[str], extra: Dict[str, Any]) -> float:
    # Lightweight heuristic scoring for ambiguity:
    # - presence of name tokens in problem statement
    # - file path token overlap
    text = problem_statement.lower()
    score = 0.0
    if qualname:
        for tok in re.split(r"[^a-zA-Z0-9_]+", qualname.lower()):
            if tok and tok in text:
                score += 0.15
    for tok in re.split(r"[^a-zA-Z0-9_]+", file_path.lower()):
        if tok and tok in text:
            score += 0.05
    # boost if error names in evidence overlap
    return score

def select_best_k(scored: List[Tuple[float, Any]], k: int) -> List[Any]:
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:k]]

def neighbors_in_file(symbols: List[Dict[str, Any]], target_span: Dict[str, int], window: int = 80) -> List[Dict[str, Any]]:
    # Return symbols whose spans overlap target span +/- window lines
    ts = target_span["start_line"]
    te = target_span["end_line"]
    lo = max(1, ts - window)
    hi = te + window
    out = []
    for s in symbols:
        sp = s["span"]
        if sp["end_line"] < lo or sp["start_line"] > hi:
            continue
        out.append(s)
    # keep stable ordering by start line
    out.sort(key=lambda s: (s["span"]["start_line"], s["span"]["end_line"]))
    return out
