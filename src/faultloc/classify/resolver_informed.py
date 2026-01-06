from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from ..types import Category
from ..resolve import resolve_file_candidates

def _match_function(qualname: str, fn: str) -> bool:
    # fn may be "name" or "Class.name" or "module.name"
    q = qualname or ""
    base = q.split(".")[-1]
    return base == fn or q.endswith(fn) or fn in q

def _match_class(qualname: str, cls: str) -> bool:
    # Qualname contains ".Class." for methods, or ".Class.<class_element>..."
    q = qualname or ""
    return f".{cls}." in q or q.endswith(f".{cls}")

def _collect_function_matches(
    symbol_index_files: Dict[str, Any],
    files_scope: List[str],
    function_candidates: List[str],
    class_candidates: List[str],
    max_scan_files: int = 800,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Returns list of (file, function_symbol_entry) matches.
    """
    matches: List[Tuple[str, Dict[str, Any]]] = []

    scope = files_scope if files_scope else list(symbol_index_files.keys())[:max_scan_files]
    for rel in scope:
        fentry = symbol_index_files.get(rel)
        if not fentry:
            continue
        for fn in fentry["symbols"]["functions"]:
            qn = fn.get("qualname") or ""
            if function_candidates and any(_match_function(qn, f) for f in function_candidates):
                # if class candidates exist, filter/boost by class containment
                if class_candidates:
                    if any(_match_class(qn, c) for c in class_candidates):
                        matches.append((rel, fn))
                else:
                    matches.append((rel, fn))
    return matches

def _collect_class_element_matches(
    symbol_index_files: Dict[str, Any],
    files_scope: List[str],
    class_candidates: List[str],
    max_scan_files: int = 800,
) -> List[Tuple[str, Dict[str, Any]]]:
    matches: List[Tuple[str, Dict[str, Any]]] = []
    scope = files_scope if files_scope else list(symbol_index_files.keys())[:max_scan_files]
    if not class_candidates:
        return matches
    for rel in scope:
        fentry = symbol_index_files.get(rel)
        if not fentry:
            continue
        for ce in fentry["symbols"]["class_elements"]:
            qn = ce.get("qualname") or ""
            if any(_match_class(qn, c) for c in class_candidates):
                matches.append((rel, ce))
    return matches

def resolver_informed_classify(
    norm_dict: Dict[str, Any],
    repo_index: Dict[str, Any],
) -> Tuple[Category, Dict[str, Any]]:
    """
    Deterministically classify using:
    - extracted/normalized cues
    - existence + resolvability in repo index
    """
    fi = repo_index["file_index"]
    si_files = repo_index["symbol_index"]["files"]

    # Resolve files from candidates (basename/suffix/module → actual repo paths)
    resolved_files = resolve_file_candidates(norm_dict.get("file_candidates", []), fi)
    unique_file = resolved_files[0] if len(resolved_files) == 1 else None

    function_candidates = norm_dict.get("function_candidates", []) or []
    class_candidates = norm_dict.get("class_candidates", []) or []
    line_numbers = norm_dict.get("line_numbers", []) or []
    module_candidates = norm_dict.get("module_candidates", []) or []

    # Match functions / class elements (existence in AST index)
    fn_matches = _collect_function_matches(
        symbol_index_files=si_files,
        files_scope=resolved_files,  # if empty, scan across repo (capped)
        function_candidates=function_candidates,
        class_candidates=class_candidates,
    )

    ce_matches = _collect_class_element_matches(
        symbol_index_files=si_files,
        files_scope=resolved_files,
        class_candidates=class_candidates,
    )

    # --- FULL_LOCATION rules (resolvability-based) ---
    full_reason = None

    # Rule 1: file + line anchors to a concrete enclosing symbol in that file (and file is uniquely resolved)
    if unique_file and line_numbers:
        fentry = si_files.get(unique_file)
        if fentry:
            # if at least one anchor line exists and maps to some enclosing symbol, treat as FULL
            from ..repo_index.ast_indexer import find_enclosing_symbol
            for ln in line_numbers:
                enc = find_enclosing_symbol(fentry, int(ln))
                if enc:
                    full_reason = {"rule": "file+line", "file": unique_file, "line": ln, "enclosing": enc.get("qualname")}
                    return Category.FULL_LOCATION, _details(resolved_files, fn_matches, ce_matches, full_reason, module_candidates)

    # Rule 2: file + function uniquely resolves inside that file (and file is uniquely resolved)
    if unique_file and function_candidates:
        in_file = [(f, fn) for (f, fn) in fn_matches if f == unique_file]
        # consider "unique" if only one match in that file after class filtering
        if len(in_file) == 1:
            full_reason = {"rule": "file+function_unique", "file": unique_file, "function": in_file[0][1].get("qualname")}
            return Category.FULL_LOCATION, _details(resolved_files, fn_matches, ce_matches, full_reason, module_candidates)

    # Rule 3: class + function uniquely resolves across repo (even without file mention)
    if function_candidates and class_candidates:
        if len(fn_matches) == 1:
            full_reason = {"rule": "class+function_unique", "file": fn_matches[0][0], "function": fn_matches[0][1].get("qualname")}
            return Category.FULL_LOCATION, _details(resolved_files, fn_matches, ce_matches, full_reason, module_candidates)

    # --- PARTIAL rules ---
    has_location_parts = bool(
        norm_dict.get("file_candidates") or module_candidates or function_candidates or class_candidates or line_numbers
    )

    # If there are explicit location parts but not resolvable as FULL, this is PARTIAL
    if has_location_parts:
        return Category.PARTIAL, _details(resolved_files, fn_matches, ce_matches, {"rule": "has_parts_not_full"}, module_candidates)

    # --- HINT / NO_HINT ---
    # If nothing location-like, caller should decide based on other_clues; but other_clues are not in norm_dict.
    # Pipeline will set HINT if extractor produced other_clues, else NO_HINT.
    return Category.NO_HINT, _details(resolved_files, fn_matches, ce_matches, {"rule": "no_parts"}, module_candidates)

def _details(
    resolved_files: List[str],
    fn_matches: List[Tuple[str, Dict[str, Any]]],
    ce_matches: List[Tuple[str, Dict[str, Any]]],
    decision: Dict[str, Any],
    module_candidates: List[str],
) -> Dict[str, Any]:
    # compact + serializable
    return {
        "decision": decision,
        "resolved_files": resolved_files[:50],
        "module_candidates": module_candidates[:50],
        "function_matches": [
            {"file": f, "qualname": (fn.get("qualname") or fn.get("name")), "span": fn.get("span")}
            for (f, fn) in fn_matches[:50]
        ],
        "class_element_matches": [
            {"file": f, "qualname": (ce.get("qualname") or ce.get("name")), "span": ce.get("span")}
            for (f, ce) in ce_matches[:50]
        ],
        "counts": {
            "resolved_files": len(resolved_files),
            "function_matches": len(fn_matches),
            "class_element_matches": len(ce_matches),
        },
    }
