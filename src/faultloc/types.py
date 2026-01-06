from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

class Category(str, Enum):
    FULL_LOCATION = "FULL_LOCATION"
    PARTIAL = "PARTIAL"
    HINT = "HINT"
    NO_HINT = "NO_HINT"

class LocationType(str, Enum):
    FUNCTION = "FUNCTION"
    CLASS_ELEMENT = "CLASS_ELEMENT"
    MODULE_SYMBOL = "MODULE_SYMBOL"

@dataclass(frozen=True)
class Span:
    start_line: int
    end_line: int

@dataclass
class CueBundle:
    file_mentions: List[str] = field(default_factory=list)
    module_mentions: List[str] = field(default_factory=list)
    class_mentions: List[str] = field(default_factory=list)
    function_mentions: List[str] = field(default_factory=list)
    line_mentions: List[str] = field(default_factory=list)  # keep as raw strings; normalize later
    other_clues: List[str] = field(default_factory=list)

@dataclass
class NormalizedCues:
    # Derived variants and normalized candidates for resolution
    file_candidates: List[str] = field(default_factory=list)
    module_candidates: List[str] = field(default_factory=list)
    class_candidates: List[str] = field(default_factory=list)
    function_candidates: List[str] = field(default_factory=list)
    line_numbers: List[int] = field(default_factory=list)

    # Keep original/raw cues for evidence
    raw: CueBundle = field(default_factory=CueBundle)

@dataclass
class BuggyLocation:
    location_type: LocationType
    file_path: str              # repo-relative, POSIX
    qualname: Optional[str]      # fully qualified name when applicable
    span: Span
    anchor_line: Optional[int] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

@dataclass
class CandidateGroup:
    rank: int
    locations: List[BuggyLocation]
    score: float
    rationale: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BudgetState:
    time_limit_sec: int
    cost_limit_usd: float
    started_at: float
    spent_usd: float = 0.0

    # internal flags to disable optional steps
    allow_llm_refine: bool = True
    allow_ast_reasoning: bool = True
    allow_retrieval_expand: bool = True

@dataclass
class RepoIndex:
    repo_root: str
    file_index: Dict[str, Any]
    symbol_index: Dict[str, Any]
    import_graph: Dict[str, Any]

@dataclass
class FLResult:
    instance_id: str
    category: Category
    cues: CueBundle
    normalized: NormalizedCues
    candidates: List[CandidateGroup]
