from __future__ import annotations

import os
from dataclasses import dataclass

def _env(key: str, default: str) -> str:
    v = os.getenv(key)
    return default if v is None or v.strip() == "" else v.strip()

def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

@dataclass(frozen=True)
class FLConfig:
    # LLM
    llm_model: str = _env("FL_LLM_MODEL", "gpt-4o-2024-05-13")
    # Embeddings (CodeBERT by default; local/offline)
    embed_model: str = _env("FL_EMBED_MODEL", "microsoft/codebert-base")
    embed_device: str = _env("FL_EMBED_DEVICE", "auto")  # cpu|cuda|mps|auto
    embed_batch_size: int = _env_int("FL_EMBED_BATCH_SIZE", 16)
    embed_max_length: int = _env_int("FL_EMBED_MAX_LENGTH", 256)

    # Determinism
    classification_temperature: float = _env_float("FL_CLASSIFY_TEMP", 0.0)
    extraction_temperature: float = _env_float("FL_EXTRACT_TEMP", 0.0)
    reasoning_temperature: float = _env_float("FL_REASON_TEMP", 0.1)

    # Budgets (per bug)
    time_limit_sec: int = _env_int("FL_TIME_LIMIT_SEC", 3000)  # 5 minutes
    cost_limit_usd: float = _env_float("FL_COST_LIMIT_USD", 0.5)

    # Retrieval/build limits
    max_files_for_embedding: int = _env_int("FL_MAX_FILES_EMBED", 4000)  # safety cap
    max_functions_for_embedding: int = _env_int("FL_MAX_FUNCS_EMBED", 20000)

    # Candidate limits
    default_topk: int = _env_int("FL_TOPK", 2)
    max_locations_per_group: int = _env_int("FL_MAX_LOCS_PER_GROUP", 20)

    # Optional pricing (approx; can be overridden)
    # Provide your own if you want more accurate cost enforcement.
    price_input_per_1k: float = _env_float("FL_PRICE_IN_1K", 0.0)
    price_output_per_1k: float = _env_float("FL_PRICE_OUT_1K", 0.0)

CONFIG = FLConfig()
