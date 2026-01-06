from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import write_json, now_iso

def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj

class ArtifactLogger:
    def __init__(self, root: Path, instance_id: str) -> None:
        self.root = root / instance_id
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, rel: str, obj: Any) -> None:
        p = self.root / rel
        write_json(p, _to_jsonable(obj))

    def write_text(self, rel: str, text: str) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def meta(self) -> Dict[str, Any]:
        return {"timestamp": now_iso(), "root": self.root.as_posix()}
