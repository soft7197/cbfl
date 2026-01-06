from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .file_scanner import scan_py_files, build_file_index
from .ast_indexer import build_symbol_index
from .import_graph import build_import_graph

def build_repo_index(repo_root: Path) -> Dict[str, Any]:
    py_files = scan_py_files(repo_root)
    file_index = build_file_index(repo_root, py_files)
    symbol_index = build_symbol_index(repo_root, file_index)
    import_graph = build_import_graph(repo_root, file_index)
    return {"file_index": file_index, "symbol_index": symbol_index, "import_graph": import_graph}
