from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class PatchHunk:
    file_path: str
    added_lines: List[int]
    removed_lines: List[int]

_DIFF_FILE_RE = re.compile(r"^\+\+\+\s+b/(.+)$")
_HUNK_RE = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")

def parse_unified_diff(patch_text: str) -> List[PatchHunk]:
    # Minimal unified diff parser sufficient for SWE-bench patches.
    lines = patch_text.splitlines()
    cur_file: Optional[str] = None
    hunks: List[PatchHunk] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _DIFF_FILE_RE.match(line)
        if m:
            cur_file = m.group(1).strip()
            i += 1
            continue
        m = _HUNK_RE.match(line)
        if m and cur_file:
            old_start = int(m.group(1))
            new_start = int(m.group(3))
            i += 1
            old_line = old_start
            new_line = new_start
            added: List[int] = []
            removed: List[int] = []
            while i < len(lines) and not lines[i].startswith("@@"):
                l = lines[i]
                if l.startswith("+") and not l.startswith("+++"):
                    added.append(new_line)
                    new_line += 1
                elif l.startswith("-") and not l.startswith("---"):
                    removed.append(old_line)
                    old_line += 1
                else:
                    # context
                    old_line += 1
                    new_line += 1
                i += 1
            hunks.append(PatchHunk(file_path=cur_file, added_lines=added, removed_lines=removed))
            continue
        i += 1
    return hunks
