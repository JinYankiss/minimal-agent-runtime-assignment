from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .errors import LLMProtocolError


class LLMClient(Protocol):
    def complete(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]]
    ) -> str:
        """Return one JSON decision from the model."""


@dataclass
class OpenAICompatibleClient:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 60.0

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleClient":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise ValueError("OPENAI_API_KEY is required")
        return cls(
            api_key=key,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip(
                "/"
            ),
        )

    def complete(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]]
    ) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMProtocolError(f"LLM request failed: {exc}") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProtocolError("LLM response does not contain message content") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMProtocolError("LLM response content is empty")
        return content


@dataclass
class ScriptedLLMClient:
    """Deterministic client used by tests and local demonstrations."""

    responses: list[str]

    def __post_init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], list[dict[str, Any]]]] = []

    def complete(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]]
    ) -> str:
        self.calls.append((messages, tools))
        if not self.responses:
            raise LLMProtocolError("scripted response queue is empty")
        return self.responses.pop(0)
