from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import OpenAI

@dataclass
class LLMResponse:
    content: Dict[str, Any]
    usage: Optional[Dict[str, Any]] = None
    raw_text: Optional[str] = None

class OpenAIClient:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None or api_key.strip() == "":
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        self._client = OpenAI(api_key=api_key)

    def chat_json(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_retries: int = 1,
        timeout_sec: int = 120,
    ) -> LLMResponse:
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    top_p=1,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                    timeout=timeout_sec,
                )
                text = resp.choices[0].message.content
                usage_obj = getattr(resp, "usage", None)
                usage = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else usage_obj
                usage = usage if isinstance(usage, dict) else None
                return LLMResponse(content=json.loads(text), usage=usage, raw_text=text)
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"OpenAI chat_json failed after retries: {last_err}")
