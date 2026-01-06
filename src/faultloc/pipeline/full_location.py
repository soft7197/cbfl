from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..types import BuggyLocation, CandidateGroup, LocationType, Span
from ..resolve import resolve_file_candidates, score_symbol_match, neighbors_in_file
from ..repo_index.ast_indexer import find_enclosing_symbol

def run_full_location(problem_statement: str, normalized: Dict[str, Any], repo_index: Dict[str, Any], topk: int, max_locs: int) -> List[CandidateGroup]:
    fi = repo_index["file_index"]
    si = repo_index["symbol_index"]["files"]

    # resolve file candidates
    files = resolve_file_candidates(normalized["file_candidates"], fi)
    # if none, allow all files but cap
    if not files:
        files = fi["all_py_files"][:500]

    # resolve symbols by function/class mention
    func_names = normalized["function_candidates"]
    class_names = normalized["class_candidates"]
    anchor_lines = normalized["line_numbers"]

    scored_candidates: List[Tuple[float, BuggyLocation]] = []

    for rel in files:
        fentry = si.get(rel)
        if not fentry:
            continue
        symbols = fentry["symbols"]
        # anchor line mapping
        if anchor_lines:
            for ln in anchor_lines:
                enc = find_enclosing_symbol(fentry, ln)
                if enc:
                    lt = LocationType.FUNCTION if enc["kind"] == "FUNCTION" else (LocationType.CLASS_ELEMENT if enc["kind"] == "CLASS_ELEMENT" else LocationType.MODULE_SYMBOL)
                    bl = BuggyLocation(
                        location_type=lt,
                        file_path=rel,
                        qualname=enc.get("qualname"),
                        span=Span(enc["span"]["start_line"], enc["span"]["end_line"]),
                        anchor_line=ln,
                        evidence={"anchor_line": ln, "method": "enclosing_symbol"},
                        confidence=0.6,
                    )
                    scored_candidates.append((0.8, bl))

        # function matches
        for fn in symbols["functions"]:
            qn = fn.get("qualname") or ""
            base = qn.split(".")[-1]
            ok = False
            if any((base == f or qn.endswith(f) or f in qn) for f in func_names if f):
                ok = True
            if class_names and ". ":
                # if class mention exists, boost matches containing it
                pass
            if ok:
                s = 0.9 + score_symbol_match(problem_statement, rel, qn, fn.get("extra", {}))
                scored_candidates.append((s, BuggyLocation(
                    location_type=LocationType.FUNCTION,
                    file_path=rel,
                    qualname=qn,
                    span=Span(fn["span"]["start_line"], fn["span"]["end_line"]),
                    evidence={"matched_function": func_names},
                    confidence=min(1.0, s),
                )))

        # class element matches: if class mentioned but no method
        for ce in symbols["class_elements"]:
            qn = ce.get("qualname") or ""
            if class_names and any(c in qn for c in class_names):
                s = 0.75 + score_symbol_match(problem_statement, rel, qn, ce.get("extra", {}))
                scored_candidates.append((s, BuggyLocation(
                    location_type=LocationType.CLASS_ELEMENT,
                    file_path=rel,
                    qualname=qn,
                    span=Span(ce["span"]["start_line"], ce["span"]["end_line"]),
                    evidence={"matched_class": class_names},
                    confidence=min(1.0, s),
                )))

    # If no matches, fall back to module symbols in resolved files
    if not scored_candidates:
        for rel in files[:80]:
            fentry = si.get(rel)
            if not fentry:
                continue
            for ms in fentry["symbols"]["module_symbols"][:5]:
                qn = ms.get("qualname")
                s = 0.3 + score_symbol_match(problem_statement, rel, qn, ms.get("extra", {}))
                scored_candidates.append((s, BuggyLocation(
                    location_type=LocationType.MODULE_SYMBOL,
                    file_path=rel,
                    qualname=qn,
                    span=Span(ms["span"]["start_line"], ms["span"]["end_line"]),
                    evidence={"fallback": True},
                    confidence=min(1.0, s),
                )))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # Build top-k groups with nearby expansion
    groups: List[CandidateGroup] = []
    used = set()
    rank = 1
    for score, loc in scored_candidates:
        key = (loc.file_path, loc.qualname, loc.span.start_line, loc.span.end_line, loc.location_type.value)
        if key in used:
            continue
        used.add(key)

        # nearby expansion: include neighbors in same file
        fentry = si.get(loc.file_path)
        expanded = [loc]
        if fentry:
            all_syms = fentry["symbols"]["functions"] + fentry["symbols"]["class_elements"] + fentry["symbols"]["module_symbols"]
            neigh = neighbors_in_file(all_syms, {"start_line": loc.span.start_line, "end_line": loc.span.end_line}, window=60)
            for s in neigh:
                if len(expanded) >= max_locs:
                    break
                lt = LocationType.FUNCTION if s["kind"] == "FUNCTION" else (LocationType.CLASS_ELEMENT if s["kind"] == "CLASS_ELEMENT" else LocationType.MODULE_SYMBOL)
                expanded.append(BuggyLocation(
                    location_type=lt,
                    file_path=loc.file_path,
                    qualname=s.get("qualname"),
                    span=Span(s["span"]["start_line"], s["span"]["end_line"]),
                    evidence={"nearby_of": loc.qualname},
                    confidence=max(0.2, loc.confidence - 0.3),
                ))

        groups.append(CandidateGroup(rank=rank, locations=expanded[:max_locs], score=float(score), rationale={"mode": "FULL_LOCATION"}))
        rank += 1
        if rank > topk:
            break

    return groups
