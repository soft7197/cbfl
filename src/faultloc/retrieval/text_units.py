from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class TextUnit:
    uid: str              # stable id
    kind: str             # FILE | FUNCTION
    file_path: str
    qualname: Optional[str]
    text: str
    meta: Dict[str, Any]
