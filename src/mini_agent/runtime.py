from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any

from .errors import AgentError, LLMProtocolError, MaxRoundsExceeded
from .llm import LLMClient
from .models import (
    FinalAnswer,
    Message,
    MessageRole,
    RunResult,
    Session,
    ToolCall,
    ToolContext,
)
from .parser import parse_decision
from .registry import ToolRegistry
from .session import SessionStore
from .tools import ToolExecutor
from .trace import TraceLogger


SYSTEM_PROMPT = """You are a minimal tool-using agent.
Return exactly one JSON object and no markdown:
{"thought":"brief decision rationale","action":{"type":"final","answer":"..."}}
or
{"thought":"brief decision rationale","action":{"type":"tool_call","call_id":"stable-id","tool_name":"name","arguments":{}}}
Use a tool when it is needed. After a tool result, either call another tool or return final.
The thought must be brief and operational; do not expose private chain-of-thought.
"""


@dataclass
class AgentRuntime:
    llm: LLMClient
    registry: ToolRegistry
    sessions: SessionStore
    executor: ToolExecutor
    tracer: TraceLogger
    max_rounds: int = 8

    @classmethod
    def with_defaults(cls, llm: LLMClient, trace_dir: str = "traces") -> "AgentRuntime":
        from .tools import default_registry

        registry = default_registry()
        return cls(
            llm=llm,
            registry=registry,
            sessions=SessionStore(),
            executor=ToolExecutor(registry),
            tracer=TraceLogger(trace_dir),
        )

    def run(self, user_id: str, session_id: str, user_input: str) -> RunResult:
        if not user_input.strip():
            raise ValueError("user_input cannot be empty")
        run_id = uuid.uuid4().hex
        session = self.sessions.get_or_create(user_id, session_id)
        self.sessions.append(session, Message(MessageRole.USER, user_input.strip()))
        trace_path = self.tracer.event(
            run_id, "run_started", user_id=user_id, session_id=session_id
        )
        try:
            answer, rounds = self._loop(run_id, session)
        except AgentError as exc:
            self.tracer.event(run_id, "run_failed", error=str(exc))
            raise
        self.sessions.append(session, Message(MessageRole.ASSISTANT, answer))
        self.tracer.event(run_id, "run_finished", rounds=rounds)
        return RunResult(run_id, user_id, session_id, answer, rounds, trace_path)

    def _loop(self, run_id: str, session: Session) -> tuple[str, int]:
        for round_number in range(1, self.max_rounds + 1):
            decision = self._ask_llm(session, run_id, round_number)
            if isinstance(decision, FinalAnswer):
                self.tracer.event(run_id, "final_answer", round=round_number)
                return decision.answer, round_number
            result = self._run_tool(run_id, session, decision, round_number)
            self.sessions.append(
                session,
                Message(
                    MessageRole.ASSISTANT,
                    json.dumps(
                        {
                            "action": "tool_call",
                            "tool_name": decision.tool_name,
                            "arguments": decision.arguments,
                        },
                        ensure_ascii=False,
                    ),
                ),
                Message(
                    MessageRole.TOOL,
                    result.as_context(),
                    name=decision.tool_name,
                    tool_call_id=decision.call_id,
                ),
            )
        raise MaxRoundsExceeded(f"agent exceeded max rounds: {self.max_rounds}")

    def _ask_llm(self, session: Session, run_id: str, round_number: int):
        messages = [
            Message(MessageRole.SYSTEM, SYSTEM_PROMPT).as_llm_message(),
            Message(
                MessageRole.SYSTEM,
                f"Available tools JSON: {self.registry.prompt_json()}",
            ).as_llm_message(),
        ]
        messages.extend(message.as_llm_message() for message in self.sessions.context(session))
        raw = self.llm.complete(messages, self.registry.schemas())
        self.tracer.event(run_id, "llm_decision", round=round_number, raw=raw)
        try:
            return parse_decision(raw)
        except LLMProtocolError as exc:
            self.tracer.event(
                run_id,
                "llm_format_retry",
                round=round_number,
                error=str(exc),
            )
            repair_messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "Your previous output was invalid. Repair only the JSON format and "
                        "return exactly one decision object matching the required schema. "
                        f"Invalid output: {raw}"
                    ),
                }
            ]
            repaired = self.llm.complete(repair_messages, self.registry.schemas())
            self.tracer.event(
                run_id,
                "llm_repaired_decision",
                round=round_number,
                raw=repaired,
            )
            return parse_decision(repaired)

    def _run_tool(
        self, run_id: str, session: Session, decision: ToolCall, round_number: int
    ):
        key = _idempotency_key(run_id, decision)
        context = ToolContext(
            user_id=session.user_id,
            session_id=session.session_id,
            run_id=run_id,
            call_id=decision.call_id,
            idempotency_key=key,
        )
        self.tracer.event(
            run_id,
            "tool_started",
            round=round_number,
            tool=decision.tool_name,
            call_id=decision.call_id,
            idempotency_key=key,
        )
        result = self.executor.execute(context, decision.tool_name, dict(decision.arguments))
        self.tracer.event(
            run_id,
            "tool_finished",
            tool=decision.tool_name,
            ok=result.ok,
            attempts=result.attempts,
            replayed=result.replayed,
        )
        return result


def _idempotency_key(run_id: str, decision: ToolCall) -> str:
    canonical = json.dumps(
        {
            "run_id": run_id,
            "call_id": decision.call_id,
            "tool": decision.tool_name,
            "arguments": decision.arguments,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
