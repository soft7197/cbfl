from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray:  # shape: (n, dim)
        ...


def _auto_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=8)
def _load_codebert(model_name: str, device: str):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "CodeBERT embeddings require 'torch' and 'transformers'. "
            "Install them via `pip install -r requirements.txt`."
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(torch.device(device))
    return tokenizer, model


def _mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@dataclass
class CodeBERTEmbedder:
    model_name: str = "microsoft/codebert-base"
    device: str = "auto"  # cpu|cuda|mps|auto
    batch_size: int = 16
    max_length: int = 256

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.max_length <= 0:
            raise ValueError("max_length must be > 0")
        if self.device == "auto":
            self.device = _auto_device()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        import torch

        tokenizer, model = _load_codebert(self.model_name, self.device)

        chunks: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                emb = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            emb_cpu = emb.detach().cpu().to(dtype=torch.float32)
            try:
                chunks.append(emb_cpu.numpy())
            except Exception:
                chunks.append(np.asarray(emb_cpu.tolist(), dtype=np.float32))

        return np.vstack(chunks) if len(chunks) > 1 else chunks[0]
