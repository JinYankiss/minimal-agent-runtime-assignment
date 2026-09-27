from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import operator
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Mapping

from .errors import ToolExecutionError, ToolTimeoutError
from .models import ToolContext, ToolEffect, ToolResult, ToolSpec
from .registry import ToolRegistry


@dataclass
class LedgerEntry:
    status: str
    result: ToolResult | None = None


class ToolExecutionLedger:
    """Stores the result under one stable key so retries replay, not reapply."""

    def __init__(self) -> None:
        self._entries: dict[str, LedgerEntry] = {}
        self._lock = Lock()

    def get(self, key: str) -> LedgerEntry | None:
        with self._lock:
            return self._entries.get(key)

    def reserve(self, key: str) -> bool:
        with self._lock:
            if key in self._entries:
                return False
            self._entries[key] = LedgerEntry(status="running")
            return True

    def complete(self, key: str, result: ToolResult) -> None:
        with self._lock:
            self._entries[key] = LedgerEntry(status="completed", result=result)


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, ledger: ToolExecutionLedger | None = None):
        self.registry = registry
        self.ledger = ledger or ToolExecutionLedger()

    def execute(
        self,
        context: ToolContext,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        spec = self.registry.get(tool_name)
        self.registry.validate(spec, arguments)
        key = context.idempotency_key
        previous = self.ledger.get(key)
        if previous and previous.result:
            return ToolResult(
                **{**previous.result.__dict__, "replayed": True}
            )
        if spec.effect == ToolEffect.NON_IDEMPOTENT_WRITE and spec.max_retries > 0:
            return ToolResult(
                call_id=context.call_id,
                tool_name=tool_name,
                ok=False,
                error=(
                    f"tool {tool_name} is non-idempotent; automatic retry is blocked "
                    "until the external operation can be queried safely"
                ),
                attempts=0,
                idempotency_key=key,
                requires_status_query=True,
            )
        attempts = 0
        max_attempts = spec.max_retries + 1
        while attempts < max_attempts:
            attempts += 1
            if spec.effect != ToolEffect.READ_ONLY and not self.ledger.reserve(key):
                previous = self.ledger.get(key)
                if previous and previous.result:
                    return ToolResult(**{**previous.result.__dict__, "replayed": True})
            try:
                output = _invoke_with_timeout(spec, context, arguments)
                result = ToolResult(
                    call_id=context.call_id,
                    tool_name=tool_name,
                    ok=True,
                    output=output,
                    attempts=attempts,
                    idempotency_key=key,
                )
                if spec.effect != ToolEffect.READ_ONLY:
                    self.ledger.complete(key, result)
                return result
            except Exception as exc:  # handler boundary: normalize and retry by policy
                if attempts >= max_attempts:
                    result = ToolResult(
                        call_id=context.call_id,
                        tool_name=tool_name,
                        ok=False,
                        error=str(exc),
                        attempts=attempts,
                        idempotency_key=key,
                        retryable=False,
                        requires_status_query=isinstance(exc, ToolTimeoutError),
                    )
                    if spec.effect != ToolEffect.READ_ONLY:
                        self.ledger.complete(key, result)
                    return result
                time.sleep(0)
        raise ToolExecutionError("unreachable tool execution state")


def _invoke_with_timeout(
    spec: ToolSpec, context: ToolContext, arguments: Mapping[str, Any]
) -> Any:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-tool")
    future = executor.submit(spec.handler, context, arguments)
    try:
        return future.result(timeout=spec.timeout_seconds)
    except FutureTimeoutError as exc:
        future.cancel()
        raise ToolTimeoutError(
            f"tool {spec.name} timed out after {spec.timeout_seconds}s; "
            "retries reuse the same idempotency key"
        ) from exc
    finally:
        # A running handler cannot be force-killed. The key contract prevents a
        # later retry from applying the same write twice at the tool boundary.
        executor.shutdown(wait=False, cancel_futures=True)


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="calculator",
            description="Evaluate a basic arithmetic expression.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
                "additionalProperties": False,
            },
            effect=ToolEffect.READ_ONLY,
            handler=_calculator,
        )
    )
    registry.register(
        ToolSpec(
            name="search",
            description="Search a small deterministic mock index.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            effect=ToolEffect.READ_ONLY,
            handler=_search,
        )
    )
    registry.register(
        ToolSpec(
            name="weather",
            description="Return mock weather for a city.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
            effect=ToolEffect.READ_ONLY,
            handler=_weather,
        )
    )
    registry.register(
        ToolSpec(
            name="todo_add",
            description="Add one todo item. The same idempotency key is safe to replay.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            effect=ToolEffect.IDEMPOTENT_WRITE,
            handler=_todo_add,
        )
    )
    registry.register(
        ToolSpec(
            name="todo_list",
            description="List current todo items.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            effect=ToolEffect.READ_ONLY,
            handler=_todo_list,
        )
    )
    return registry


_TODO_ITEMS: dict[tuple[str, str], set[str]] = {}
_TODO_LOCK = Lock()


def _calculator(_: ToolContext, arguments: Mapping[str, Any]) -> Any:
    expression = str(arguments["expression"])
    tree = ast.parse(expression, mode="eval")
    return _eval_math(tree.body)


def _eval_math(node: ast.AST) -> float | int:
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
        return operators[type(node.op)](_eval_math(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in operators:
        return operators[type(node.op)](_eval_math(node.left), _eval_math(node.right))
    raise ToolExecutionError("calculator only supports basic arithmetic")


def _search(_: ToolContext, arguments: Mapping[str, Any]) -> Any:
    query = str(arguments["query"]).lower()
    documents = [
        "Idempotency keys prevent duplicate side effects during retries.",
        "A session is isolated by the pair user_id and session_id.",
        "Context compaction keeps recent messages and a bounded summary.",
    ]
    return [item for item in documents if query in item.lower()] or ["No mock result found."]


def _weather(_: ToolContext, arguments: Mapping[str, Any]) -> Any:
    city = str(arguments["city"])
    return {"city": city, "condition": "晴", "temperature_c": 25}


def _todo_add(context: ToolContext, arguments: Mapping[str, Any]) -> Any:
    text = str(arguments["text"]).strip()
    if not text:
        raise ToolExecutionError("todo text cannot be empty")
    with _TODO_LOCK:
        _TODO_ITEMS.setdefault((context.user_id, context.session_id), set()).add(text)
    return {"added": text, "idempotency_key": context.idempotency_key}


def _todo_list(context: ToolContext, _: Mapping[str, Any]) -> Any:
    with _TODO_LOCK:
        return sorted(_TODO_ITEMS.get((context.user_id, context.session_id), set()))
