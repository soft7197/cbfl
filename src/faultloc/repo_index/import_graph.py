from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

def build_import_graph(repo_root: Path, file_index: Dict[str, Any]) -> Dict[str, Any]:
    # Best-effort static import graph at module granularity.
    path_to_mod: Dict[str, str] = file_index.get("path_module_map", {})
    mod_to_path: Dict[str, str] = file_index.get("module_path_map", {})

    edges: Dict[str, Set[str]] = {}
    for rel in file_index.get("all_py_files", []):
        mod = path_to_mod.get(rel)
        if not mod:
            continue
        p = repo_root / rel
        try:
            src = p.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except Exception:
            continue

        deps: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name:
                        deps.add(a.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    deps.add(node.module)

        # Keep only deps that look like repo modules (best-effort prefix match)
        repo_mods = set(mod_to_path.keys())
        filtered: Set[str] = set()
        for d in deps:
            # accept exact module or any prefix that exists
            if d in repo_mods:
                filtered.add(d)
            else:
                # try shortening
                parts = d.split(".")
                for k in range(len(parts)-1, 0, -1):
                    cand = ".".join(parts[:k])
                    if cand in repo_mods:
                        filtered.add(cand)
                        break

        edges[mod] = filtered

    # invert for in-edges
    rev: Dict[str, Set[str]] = {}
    for a, bs in edges.items():
        for b in bs:
            rev.setdefault(b, set()).add(a)

    return {
        "nodes": sorted(set(edges.keys()) | set(rev.keys())),
        "out_edges": {k: sorted(list(v)) for k, v in edges.items()},
        "in_edges": {k: sorted(list(v)) for k, v in rev.items()},
    }
