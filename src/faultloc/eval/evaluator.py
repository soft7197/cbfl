from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from ..types import CandidateGroup
from ..repo_index.ast_indexer import find_enclosing_symbol
from .patch_parser import parse_unified_diff

def _gt_locations_from_patch(repo_index: Dict[str, Any], patch_text: str) -> Set[Tuple[str, str]]:
    # Returns a set of (file, qualname_or_kind_anchor) ground-truth symbols from patch changed lines.
    si = repo_index["symbol_index"]["files"]
    hunks = parse_unified_diff(patch_text)
    gt: Set[Tuple[str, str]] = set()
    for h in hunks:
        rel = h.file_path
        fentry = si.get(rel)
        if not fentry:
            continue
        changed = sorted(set(h.added_lines + h.removed_lines))
        for ln in changed:
            enc = find_enclosing_symbol(fentry, ln)
            if not enc:
                continue
            qn = enc.get("qualname") or enc.get("kind") or "UNKNOWN"
            gt.add((rel, qn))
    return gt

def _pred_group_key(g: CandidateGroup) -> Set[Tuple[str, str]]:
    s: Set[Tuple[str, str]] = set()
    for l in g.locations:
        s.add((l.file_path, l.qualname or l.location_type.value))
    return s

def evaluate_against_patch(repo_root, repo_index: Dict[str, Any], candidates: List[CandidateGroup], patch_text: str) -> Dict[str, Any]:
    gt = _gt_locations_from_patch(repo_index, patch_text)
    best_rank = None
    exact_hits = []
    for g in candidates:
        pred = _pred_group_key(g)
        if pred == gt:
            best_rank = g.rank
            exact_hits.append(g.rank)
            break
    return {
        "gt_symbol_set": sorted(list(gt)),
        "exact_match_rank": best_rank,
        "exact_match": best_rank is not None,
        "checked_groups": len(candidates),
    }
