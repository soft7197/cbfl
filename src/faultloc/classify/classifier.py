from __future__ import annotations

from typing import Any, Dict

from ..openai_client import OpenAIClient
from ..prompts import CLASSIFIER_SYSTEM
from ..types import Category

class IssueClassifier:
    def __init__(self, client: OpenAIClient, model: str, temperature: float) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature

    def classify(self, problem_statement: str) -> Dict[str, Any]:
        user = "Issue description:\n\n" + problem_statement
        resp = self.client.chat_json(self.model, CLASSIFIER_SYSTEM, user, temperature=self.temperature)
        cat = str(resp.content.get("category", "")).strip()
        if cat not in {c.value for c in Category}:
            # fallback: map loosely
            cat = "HINT"
        return {"category": cat, "reason": resp.content.get("reason", ""), "usage": resp.usage, "raw_text": resp.raw_text}
