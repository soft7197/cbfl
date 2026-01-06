from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

from ..ast_reasoning import build_candidate_summaries
from ..openai_client import OpenAIClient
from ..prompts import AST_REASON_SYSTEM
from ..types import BuggyLocation, CandidateGroup, LocationType, Span
from ..resolve import resolve_file_candidates, score_symbol_match
from ..repo_index.ast_indexer import find_enclosing_symbol

def _collect_scoped_symbols(repo_index: Dict[str, Any], files: List[str], function_candidates: List[str], class_candidates: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    si = repo_index["symbol_index"]["files"]
    scoped: Dict[str, List[Dict[str, Any]]] = {}
    for rel in files:
        fentry = si.get(rel)
        if not fentry:
            continue
        syms = []
        # include functions that name-match; else include top few functions
        for fn in fentry["symbols"]["functions"]:
            qn = fn.get("qualname") or ""
            base = qn.split(".")[-1]
            if function_candidates and any(base == f or qn.endswith(f) or f in qn for f in function_candidates):
                syms.append(fn)
        if class_candidates:
            for ce in fentry["symbols"]["class_elements"]:
                qn = ce.get("qualname") or ""
                if any(c in qn for c in class_candidates):
                    syms.append(ce)
        if not syms:
            syms = (fentry["symbols"]["functions"][:8] + fentry["symbols"]["class_elements"][:4] + fentry["symbols"]["module_symbols"][:2])
        scoped[rel] = syms
    return scoped

def run_partial_location(
    client: OpenAIClient,
    model: str,
    problem_statement: str,
    normalized: Dict[str, Any],
    repo_root: Path,
    repo_index: Dict[str, Any],
    topk: int,
    max_locs: int,
    allow_ast_reasoning: bool,
) -> Tuple[List[CandidateGroup], Optional[Dict[str, Any]]]:
    fi = repo_index["file_index"]
    files = resolve_file_candidates(normalized["file_candidates"], fi)
    if not files:
        # if no file cue, pick by basename matches from function mention heuristics
        files = fi["all_py_files"][:200]

    scoped = _collect_scoped_symbols(repo_index, files[:60], normalized["function_candidates"], normalized["class_candidates"])
    # Create candidate summaries for LLM reasoning
    summaries = build_candidate_summaries(repo_root, scoped, max_candidates=30)

    if not allow_ast_reasoning or not summaries:
        # heuristic-only fallback
        groups: List[CandidateGroup] = []
        rank = 1
        for rel, syms in list(scoped.items())[:topk]:
            locs: List[BuggyLocation] = []
            for s in syms[:max_locs]:
                lt = LocationType.FUNCTION if s["kind"] == "FUNCTION" else (LocationType.CLASS_ELEMENT if s["kind"] == "CLASS_ELEMENT" else LocationType.MODULE_SYMBOL)
                locs.append(BuggyLocation(
                    location_type=lt,
                    file_path=rel,
                    qualname=s.get("qualname"),
                    span=Span(s["span"]["start_line"], s["span"]["end_line"]),
                    evidence={"mode": "PARTIAL_HEURISTIC"},
                    confidence=0.4,
                ))
            groups.append(CandidateGroup(rank=rank, locations=locs, score=0.4, rationale={"mode": "PARTIAL_HEURISTIC"}))
            rank += 1
        return groups, None

    # LLM reasoning over summaries
    user = {
        "problem_statement": problem_statement,
        "candidates": summaries,
    }
    resp = client.chat_json(model, AST_REASON_SYSTEM, json_dumps(user), temperature=0.1)

    decision = str(resp.content.get("decision", "OK")).strip().upper()
    why = resp.content.get("why", "")
    hint_terms = resp.content.get("hint_terms") or []
    groups_out = resp.content.get("groups") or []

    if decision == "HINT" or not isinstance(groups_out, list) or len(groups_out) == 0:
        # Signal the orchestrator to switch to HINT pipeline (retrieval)
        fallback = {
            "fallback_to": "HINT",
            "why": why,
            "hint_terms": hint_terms,
            "raw_model_output": resp.content,
        }
        return [], fallback


    # Map candidate_ids back to symbols
    id_map = {c["candidate_id"]: c for c in summaries}
    groups: List[CandidateGroup] = []
    rank = 1
    for g in groups_out[:topk]:
        cids = g.get("candidate_ids") or []
        locs: List[BuggyLocation] = []
        for cid in cids:
            c = id_map.get(cid)
            if not c:
                continue
            rel = c["file"]
            summ = c["summary"]
            kind = summ.get("kind")
            lt = LocationType.FUNCTION if kind == "FUNCTION" else (LocationType.CLASS_ELEMENT if kind == "CLASS_ELEMENT" else LocationType.MODULE_SYMBOL)
            sp = summ.get("span") or {}
            locs.append(BuggyLocation(
                location_type=lt,
                file_path=rel,
                qualname=summ.get("qualname"),
                span=Span(int(sp.get("start_line", 1)), int(sp.get("end_line", 1))),
                evidence={"mode": "PARTIAL_AST", "why": g.get("why")},
                confidence=float(g.get("score", 0.5)),
            ))
            if len(locs) >= max_locs:
                break
        if locs:
            groups.append(CandidateGroup(rank=rank, locations=locs, score=float(g.get("score", 0.5)), rationale={"mode": "PARTIAL_AST", "why": g.get("why")}))
            rank += 1
    if not groups:
        # fallback
        return run_partial_location(client, model, problem_statement, normalized, repo_root, repo_index, topk, max_locs, allow_ast_reasoning=False)
    return groups, None

def json_dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, indent=2)
