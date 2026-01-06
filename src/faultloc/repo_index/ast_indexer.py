from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils import safe_relpath

@dataclass
class SymbolEntry:
    kind: str                  # FUNCTION | CLASS | CLASS_ELEMENT | MODULE_SYMBOL
    name: str                  # short name (func/class) or synthetic for elements
    qualname: Optional[str]    # module.Class.func etc
    span: Tuple[int, int]      # (start_line, end_line)
    extra: Dict[str, Any]

def _node_span(node: ast.AST) -> Tuple[int, int]:
    start = getattr(node, "lineno", 1) or 1
    end = getattr(node, "end_lineno", start) or start
    return int(start), int(end)

def index_file(repo_root: Path, rel_path: str, source: str, module_name: str) -> Dict[str, Any]:
    tree = ast.parse(source)
    symbols: Dict[str, List[Dict[str, Any]]] = {"functions": [], "classes": [], "class_elements": [], "module_symbols": []}

    def add_entry(bucket: str, e: SymbolEntry) -> None:
        symbols[bucket].append({
            "kind": e.kind,
            "name": e.name,
            "qualname": e.qualname,
            "span": {"start_line": e.span[0], "end_line": e.span[1]},
            "extra": e.extra,
        })

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            span = _node_span(node)
            qn = f"{module_name}.{node.name}" if module_name else node.name
            add_entry("functions", SymbolEntry("FUNCTION", node.name, qn, span, {"type": type(node).__name__}))
        elif isinstance(node, ast.ClassDef):
            span = _node_span(node)
            qn = f"{module_name}.{node.name}" if module_name else node.name
            add_entry("classes", SymbolEntry("CLASS", node.name, qn, span, {"bases": [getattr(b, "id", ast.dump(b)) for b in node.bases]}))

            # Inside class: methods vs class-level elements
            elem_idx = 0
            for cnode in node.body:
                if isinstance(cnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    cspan = _node_span(cnode)
                    cqn = f"{qn}.{cnode.name}"
                    add_entry("functions", SymbolEntry("FUNCTION", cnode.name, cqn, cspan, {"owner_class": node.name, "type": type(cnode).__name__}))
                else:
                    cspan = _node_span(cnode)
                    synth = f"<class_element>#{type(cnode).__name__}#{elem_idx}"
                    cqn = f"{qn}.{synth}"
                    add_entry("class_elements", SymbolEntry("CLASS_ELEMENT", synth, cqn, cspan, {"owner_class": node.name, "node_type": type(cnode).__name__}))
                    elem_idx += 1
        else:
            # module-level symbol (imports, assignments, expr)
            span = _node_span(node)
            synth = f"<module_symbol>#{type(node).__name__}#{len(symbols['module_symbols'])}"
            qn = f"{module_name}.{synth}" if module_name else synth
            add_entry("module_symbols", SymbolEntry("MODULE_SYMBOL", synth, qn, span, {"node_type": type(node).__name__}))

    return {"file": rel_path, "module": module_name, "symbols": symbols}

def build_symbol_index(repo_root: Path, file_index: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"files": {}}
    path_module = file_index.get("path_module_map", {})
    for rel in file_index["all_py_files"]:
        p = repo_root / rel
        try:
            source = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        module_name = path_module.get(rel, "")
        out["files"][rel] = index_file(repo_root, rel, source, module_name)
    return out

def find_enclosing_symbol(symbol_file_entry: Dict[str, Any], line_no: int) -> Optional[Dict[str, Any]]:
    # Given an indexed file entry (with symbols), return the smallest span that encloses line_no.
    candidates: List[Dict[str, Any]] = []
    sym = symbol_file_entry["symbols"]
    all_entries = sym["functions"] + sym["class_elements"] + sym["module_symbols"] + sym["classes"]
    for e in all_entries:
        sp = e["span"]
        if sp["start_line"] <= line_no <= sp["end_line"]:
            candidates.append(e)
    if not candidates:
        return None
    # smallest span wins
    candidates.sort(key=lambda e: (e["span"]["end_line"] - e["span"]["start_line"], e["span"]["start_line"]))
    return candidates[0]
