from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set

def expand_files_with_import_graph(file_hits: List[Dict[str, Any]], repo_index: Dict[str, Any], max_expand: int = 80) -> List[str]:
    # Expand starting from retrieved files by 1-hop import neighbors.
    file_index = repo_index["file_index"]
    import_graph = repo_index["import_graph"]
    path_to_mod = file_index.get("path_module_map", {})
    mod_to_path = file_index.get("module_path_map", {})
    out_edges = import_graph.get("out_edges", {})
    in_edges = import_graph.get("in_edges", {})

    seeds = [h["file"] for h in file_hits if "file" in h]
    mods = [path_to_mod.get(f) for f in seeds if path_to_mod.get(f)]
    expanded: Set[str] = set(seeds)

    for m in mods:
        for nb in out_edges.get(m, []):
            p = mod_to_path.get(nb)
            if p:
                expanded.add(p)
        for nb in in_edges.get(m, []):
            p = mod_to_path.get(nb)
            if p:
                expanded.add(p)
        if len(expanded) >= max_expand:
            break
    return list(expanded)
