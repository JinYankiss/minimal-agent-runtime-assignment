from __future__ import annotations

import json
from typing import Any

from .errors import ToolNotFoundError, ToolValidationError
from .models import ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"unknown tool: {name}") from exc

    def schemas(self) -> list[dict[str, Any]]:
        return [self._tools[name].schema_for_prompt() for name in sorted(self._tools)]

    def validate(self, spec: ToolSpec, arguments: dict[str, Any]) -> None:
        schema = spec.parameters
        if schema.get("type") != "object":
            return
        required = schema.get("required", [])
        missing = [name for name in required if name not in arguments]
        if missing:
            raise ToolValidationError(
                f"tool {spec.name} missing required arguments: {', '.join(missing)}"
            )
        properties = schema.get("properties", {})
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            raise ToolValidationError(
                f"tool {spec.name} received unknown arguments: {', '.join(unknown)}"
            )

    def prompt_json(self) -> str:
        return json.dumps(self.schemas(), ensure_ascii=False, separators=(",", ":"))
