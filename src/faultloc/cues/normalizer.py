from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from ..types import CueBundle, NormalizedCues

_LINE_RE = re.compile(r"(\d{1,7})")

def normalize_cues(cues: CueBundle, problem_statement: str) -> NormalizedCues:
    # 1) Normalize function mentions (strip () and whitespace)
    funcs = []
    for f in cues.function_mentions:
        x = f.strip()
        x = re.sub(r"\(\)$", "", x)
        x = x.replace("()", "")
        if x:
            funcs.append(x)

    # 2) Normalize module mentions (strip)
    mods = [m.strip() for m in cues.module_mentions if m and m.strip()]

    # 3) Normalize class mentions
    clss = [c.strip() for c in cues.class_mentions if c and c.strip()]

    # 4) Normalize file mentions; keep as-is but trim
    files = [p.strip() for p in cues.file_mentions if p and p.strip()]

    # 5) Extract line numbers
    lines: List[int] = []
    for ln in cues.line_mentions:
        for m in _LINE_RE.findall(str(ln)):
            try:
                lines.append(int(m))
            except Exception:
                pass

    # 6) Derive variants:
    # - module -> path candidate
    file_variants: List[str] = []
    for m in mods:
        # dotted module -> path
        file_variants.append(m.replace(".", "/") + ".py")

    # - if a file mention looks like just basename, keep it; resolver will match via basename_map
    file_candidates = files + file_variants

    # - derive symbol variants: Class.method splitting
    func_candidates: List[str] = []
    class_candidates: List[str] = list(clss)
    for f in funcs:
        func_candidates.append(f)
        if "." in f:
            parts = f.split(".")
            # if looks like Class.method or module.func
            if len(parts) == 2 and parts[0] and parts[1]:
                # treat as possible class+method
                class_candidates.append(parts[0])
                func_candidates.append(parts[1])

    # - also derive from other clues: backticks `name`
    bt = re.findall(r"`([^`]{1,120})`", problem_statement)
    for t in bt:
        if "/" in t and t.endswith(".py"):
            file_candidates.append(t)
        elif "." in t:
            # module or qualname candidate
            mods.append(t)

    # de-dup preserve order
    def uniq(xs: List[str]) -> List[str]:
        seen=set(); out=[]
        for x in xs:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    return NormalizedCues(
        file_candidates=uniq(file_candidates),
        module_candidates=uniq(mods),
        class_candidates=uniq(class_candidates),
        function_candidates=uniq(func_candidates),
        line_numbers=sorted(set(lines)),
        raw=cues,
    )
