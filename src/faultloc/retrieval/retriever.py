from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..budget import BudgetManager
from ..embeddings import Embedder
from .builder import build_units
from .store import EmbeddingStore, SearchHit

class HybridRetriever:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._store_files: Optional[EmbeddingStore] = None
        self._store_funcs: Optional[EmbeddingStore] = None
        self._dim: Optional[int] = None
        self._units_files = []
        self._units_funcs = []

    def build(
        self,
        repo_root: Path,
        repo_index: Dict[str, Any],
        budget: BudgetManager,
        max_files: int = 4000,
        max_funcs: int = 20000,
    ) -> Dict[str, Any]:
        units = build_units(repo_root, repo_index)
        files = [u for u in units if u.kind == "FILE"]
        funcs = [u for u in units if u.kind == "FUNCTION"]

        # Cap to control cost/time
        files = files[: min(len(files), max_files)]
        funcs = funcs[: min(len(funcs), max_funcs)]

        # Embed in batches
        meta = {"files": len(files), "functions": len(funcs), "embedded": {"files": 0, "functions": 0}}
        if not files and not funcs:
            return meta

        # Determine dim from first embedding
        def embed_units(units_list):
            if not units_list:
                return [], None
            texts = [u.text for u in units_list]
            vecs = self.embedder.embed(texts)
            return vecs, None

        # Embed files
        if files:
            vecs, usage = embed_units(files)
            budget.add_cost_from_usage(usage, 0.0, 0.0)  # embedding pricing is unknown here; user can override externally if desired
            self._dim = len(vecs[0])
            self._store_files = EmbeddingStore(self._dim)
            self._store_files.add([u.uid for u in files], files, vecs)
            self._units_files = files
            meta["embedded"]["files"] = len(files)
            budget.ensure_within_budget()

        # Embed funcs
        if funcs:
            vecs, usage = embed_units(funcs)
            budget.add_cost_from_usage(usage, 0.0, 0.0)
            if self._dim is None:
                self._dim = len(vecs[0])
            self._store_funcs = EmbeddingStore(self._dim)
            self._store_funcs.add([u.uid for u in funcs], funcs, vecs)
            self._units_funcs = funcs
            meta["embedded"]["functions"] = len(funcs)
            budget.ensure_within_budget()

        return meta

    def query(self, text: str, top_files: int = 40, top_funcs: int = 80) -> Dict[str, Any]:
        qvecs = self.embedder.embed([text])
        q = qvecs[0]
        file_hits: List[SearchHit] = self._store_files.search(q, top_files) if self._store_files else []
        func_hits: List[SearchHit] = self._store_funcs.search(q, top_funcs) if self._store_funcs else []
        return {
            "file_hits": [{"uid": h.uid, "score": h.score, "file": h.unit.file_path} for h in file_hits],
            "function_hits": [{"uid": h.uid, "score": h.score, "file": h.unit.file_path, "qualname": h.unit.qualname} for h in func_hits],
        }
