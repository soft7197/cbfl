from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from ..utils import is_probably_noise_dir, safe_relpath

def scan_py_files(repo_root: Path) -> List[Path]:
    out: List[Path] = []
    for p in repo_root.rglob("*.py"):
        if not p.is_file():
            continue
        if is_probably_noise_dir(p):
            continue
        out.append(p)
    return out

def build_file_index(repo_root: Path, py_files: List[Path]) -> Dict[str, object]:
    all_files = [safe_relpath(p, repo_root) for p in py_files]
    basename_map: Dict[str, List[str]] = {}
    suffix_map: Dict[str, List[str]] = {}
    for rel in all_files:
        bn = Path(rel).name
        basename_map.setdefault(bn, []).append(rel)
        # suffixes of increasing length
        parts = rel.split("/")
        for i in range(len(parts)):
            suf = "/".join(parts[i:])
            suffix_map.setdefault(suf, []).append(rel)

    # module map: best-effort path -> dotted module
    module_path_map: Dict[str, str] = {}
    path_module_map: Dict[str, str] = {}
    for rel in all_files:
        if rel.endswith("__init__.py"):
            mod = rel[:-12].replace("/", ".")
        else:
            mod = rel[:-3].replace("/", ".")
        path_module_map[rel] = mod
        module_path_map[mod] = rel

    return {
        "repo_root": repo_root.as_posix(),
        "all_py_files": all_files,
        "basename_map": basename_map,
        "suffix_map": suffix_map,
        "module_path_map": module_path_map,
        "path_module_map": path_module_map,
    }
