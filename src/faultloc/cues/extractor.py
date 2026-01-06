from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from ..openai_client import OpenAIClient
from ..prompts import EXTRACTOR_SYSTEM
from ..types import CueBundle

class CueExtractor:
    def __init__(self, client: OpenAIClient, model: str, temperature: float) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature

    def extract(self, problem_statement: str) -> Dict[str, Any]:
        user = "Issue description:\n\n" + problem_statement
        resp = self.client.chat_json(self.model, EXTRACTOR_SYSTEM, user, temperature=self.temperature)
        return {"result": resp.content, "usage": resp.usage, "raw_text": resp.raw_text}

    @staticmethod
    def to_bundle(obj: Dict[str, Any]) -> CueBundle:
        return CueBundle(
            file_mentions=list(obj.get("file_mentions") or []),
            module_mentions=list(obj.get("module_mentions") or []),
            class_mentions=list(obj.get("class_mentions") or []),
            function_mentions=list(obj.get("function_mentions") or []),
            line_mentions=list(obj.get("line_mentions") or []),
            other_clues=list(obj.get("other_clues") or []),
        )
