from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from .types import BudgetState

class BudgetManager:
    def __init__(self, state: BudgetState) -> None:
        self.s = state

    def elapsed(self) -> float:
        return time.monotonic() - self.s.started_at

    def remaining_time(self) -> float:
        return max(0.0, self.s.time_limit_sec - self.elapsed())

    def remaining_cost(self) -> float:
        return max(0.0, self.s.cost_limit_usd - self.s.spent_usd)

    def add_cost_from_usage(self, usage: Optional[Dict[str, Any]], price_in_per_1k: float, price_out_per_1k: float) -> None:
        if not usage:
            return
        in_tok = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        out_tok = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        self.s.spent_usd += (in_tok / 1000.0) * price_in_per_1k + (out_tok / 1000.0) * price_out_per_1k
        self._apply_degradation()

    def _apply_degradation(self) -> None:
        # Disable optional steps progressively
        if self.remaining_time() < 60 or self.remaining_cost() < 0.10:
            self.s.allow_llm_refine = False
        if self.remaining_time() < 30 or self.remaining_cost() < 0.05:
            self.s.allow_ast_reasoning = False
            self.s.allow_retrieval_expand = False

    def ensure_within_budget(self) -> None:
        # Raise if hard exceeded
        if self.elapsed() > self.s.time_limit_sec:
            raise TimeoutError(f"Time budget exceeded: {self.elapsed():.1f}s > {self.s.time_limit_sec}s")
        if self.s.spent_usd > self.s.cost_limit_usd:
            raise RuntimeError(f"Cost budget exceeded: ${self.s.spent_usd:.3f} > ${self.s.cost_limit_usd:.3f}")

    def snapshot(self) -> Dict[str, Any]:
        d = asdict(self.s)
        d["elapsed_sec"] = round(self.elapsed(), 3)
        d["remaining_time_sec"] = round(self.remaining_time(), 3)
        d["remaining_cost_usd"] = round(self.remaining_cost(), 6)
        return d
