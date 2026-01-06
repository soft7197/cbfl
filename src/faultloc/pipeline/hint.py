from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..types import BuggyLocation, CandidateGroup, LocationType, Span
from ..resolve import score_symbol_match
from ..repo_index.ast_indexer import find_enclosing_symbol

def run_hint_or_nohint(
    problem_statement: str,
    repo_root: Path,
    repo_index: Dict[str, Any],
    retrieval: Dict[str, Any],
    topk: int,
    max_locs: int,
) -> List[CandidateGroup]:
    # Build candidate groups from retrieval hits (files + functions).
    file_hits = retrieval.get("file_hits") or []
    func_hits = retrieval.get("function_hits") or []
    si = repo_index["symbol_index"]["files"]

    # Bucket by file, merge file+function signals
    file_scores: Dict[str, float] = {}
    for h in file_hits:
        file_scores[h["file"]] = max(file_scores.get(h["file"], 0.0), float(h["score"]))
    for h in func_hits:
        file_scores[h["file"]] = max(file_scores.get(h["file"], 0.0), float(h["score"]))

    ranked_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[: max(30, topk*5)]

    groups: List[CandidateGroup] = []
    rank = 1
    for rel, fs in ranked_files:
        fentry = si.get(rel)
        if not fentry:
            continue
        # Prefer top functions from func_hits in this file
        locs: List[BuggyLocation] = []
        for fh in func_hits:
            if fh["file"] != rel:
                continue
            qn = fh.get("qualname")
            if qn:
                # locate in symbol index
                for fn in fentry["symbols"]["functions"]:
                    if fn.get("qualname") == qn:
                        locs.append(BuggyLocation(
                            location_type=LocationType.FUNCTION,
                            file_path=rel,
                            qualname=qn,
                            span=Span(fn["span"]["start_line"], fn["span"]["end_line"]),
                            evidence={"retrieval": fh},
                            confidence=min(1.0, float(fh["score"])),
                        ))
                        break
            if len(locs) >= max_locs:
                break

        # If not enough, add top few functions/class elements
        if len(locs) < 2:
            for fn in fentry["symbols"]["functions"][: min(6, max_locs - len(locs))]:
                locs.append(BuggyLocation(
                    location_type=LocationType.FUNCTION,
                    file_path=rel,
                    qualname=fn.get("qualname"),
                    span=Span(fn["span"]["start_line"], fn["span"]["end_line"]),
                    evidence={"fallback_in_file": True},
                    confidence=max(0.2, fs - 0.2),
                ))

        if locs:
            groups.append(CandidateGroup(rank=rank, locations=locs[:max_locs], score=float(fs), rationale={"mode": "RETRIEVAL"}))
            rank += 1
        if rank > topk:
            break
    return groups
