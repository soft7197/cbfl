from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

from .text_units import TextUnit

def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

@dataclass
class SearchHit:
    uid: str
    score: float
    unit: TextUnit

class EmbeddingStore:
    def __init__(self, dim: int, use_faiss: bool = True) -> None:
        self.dim = dim
        self.use_faiss = use_faiss and _HAS_FAISS
        self._uids: List[str] = []
        self._units: List[TextUnit] = []
        self._mat: Optional[np.ndarray] = None
        self._index = None

    def add(self, uids: List[str], units: List[TextUnit], vectors: List[List[float]]) -> None:
        X = np.asarray(vectors, dtype="float32")
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"Embedding dim mismatch: got {X.shape}, expected (*,{self.dim})")
        X = _l2norm(X)
        if self._mat is None:
            self._mat = X
        else:
            self._mat = np.vstack([self._mat, X])
        self._uids.extend(uids)
        self._units.extend(units)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        if self._mat is None:
            return
        if self.use_faiss:
            self._index = faiss.IndexFlatIP(self.dim)
            self._index.add(self._mat)
        else:
            self._index = None

    def search(self, qvec: List[float], topn: int = 20) -> List[SearchHit]:
        if self._mat is None:
            return []
        q = np.asarray(qvec, dtype="float32").reshape(1, -1)
        q = _l2norm(q)
        if self.use_faiss and self._index is not None:
            scores, idxs = self._index.search(q, topn)
            hits: List[SearchHit] = []
            for s, i in zip(scores[0].tolist(), idxs[0].tolist()):
                if i < 0:
                    continue
                hits.append(SearchHit(uid=self._uids[i], score=float(s), unit=self._units[i]))
            return hits
        # numpy fallback
        sims = (self._mat @ q.T).reshape(-1)
        idxs = np.argsort(-sims)[:topn]
        return [SearchHit(uid=self._uids[int(i)], score=float(sims[int(i)]), unit=self._units[int(i)]) for i in idxs]
