from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping


class ToolEffect(str, Enum):
    READ_ONLY = "read_only"
    IDEMPOTENT_WRITE = "idempotent_write"
    NON_IDEMPOTENT_WRITE = "non_idempotent_write"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Message:
    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None

    def as_llm_message(self) -> dict[str, str]:
        message: dict[str, str] = {"role": self.role.value, "content": self.content}
        if self.name:
            message["name"] = self.name
        if self.tool_call_id:
            message["tool_call_id"] = self.tool_call_id
        return message


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    thought: str = ""


@dataclass(frozen=True)
class FinalAnswer:
    answer: str
    thought: str = ""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, Any]
    effect: ToolEffect
    handler: Callable[["ToolContext", Mapping[str, Any]], Any]
    timeout_seconds: float = 5.0
    max_retries: int = 1

    def schema_for_prompt(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "effect": self.effect.value,
            "retry_policy": {
                "max_retries": self.max_retries,
                "idempotency_key_required": self.effect
                != ToolEffect.READ_ONLY,
            },
        }


@dataclass(frozen=True)
class ToolContext:
    user_id: str
    session_id: str
    run_id: str
    call_id: str
    idempotency_key: str


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    tool_name: str
    ok: bool
    output: Any = None
    error: str | None = None
    attempts: int = 1
    idempotency_key: str | None = None
    replayed: bool = False
    retryable: bool = False
    requires_status_query: bool = False

    def as_context(self) -> str:
        if self.ok:
            return f"tool={self.tool_name} result={self.output!r}"
        return (
            f"tool={self.tool_name} error={self.error!r} "
            f"retryable={self.retryable} requires_status_query={self.requires_status_query}"
        )


@dataclass
class Session:
    user_id: str
    session_id: str
    messages: list[Message] = field(default_factory=list)
    summary: str = ""


@dataclass(frozen=True)
class RunResult:
    run_id: str
    user_id: str
    session_id: str
    answer: str
    rounds: int
    trace_path: str | None
