from __future__ import annotations

import json
from typing import Any

from .errors import LLMProtocolError
from .models import FinalAnswer, ToolCall


def parse_decision(raw: str) -> FinalAnswer | ToolCall:
    payload = _parse_json_object(raw)
    thought = _optional_string(payload, "thought")
    action = payload.get("action")
    if not isinstance(action, dict):
        raise LLMProtocolError("decision.action must be an object")
    action_type = action.get("type")
    if action_type == "final":
        answer = action.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise LLMProtocolError("final action requires a non-empty answer")
        return FinalAnswer(answer=answer.strip(), thought=thought)
    if action_type == "tool_call":
        name = action.get("tool_name")
        arguments = action.get("arguments", {})
        call_id = action.get("call_id")
        if not isinstance(name, str) or not name.strip():
            raise LLMProtocolError("tool_call requires tool_name")
        if not isinstance(arguments, dict):
            raise LLMProtocolError("tool_call.arguments must be an object")
        if not isinstance(call_id, str) or not call_id.strip():
            raise LLMProtocolError("tool_call requires call_id")
        return ToolCall(
            call_id=call_id.strip(),
            tool_name=name.strip(),
            arguments=arguments,
            thought=thought,
        )
    raise LLMProtocolError("action.type must be final or tool_call")


def _parse_json_object(raw: str) -> dict[str, Any]:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`").strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMProtocolError(f"LLM output is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise LLMProtocolError("LLM output must be a JSON object")
    return value


def _optional_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key, "")
    if not isinstance(value, str):
        raise LLMProtocolError(f"{key} must be a string")
    return value.strip()
