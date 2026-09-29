from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

from .models import Message, MessageRole, Session


class SessionStore:
    def __init__(
        self,
        max_messages: int = 20,
        summary_trigger: int = 24,
        storage_path: str | Path | None = None,
    ) -> None:
        self.max_messages = max_messages
        self.summary_trigger = summary_trigger
        self.storage_path = Path(storage_path) if storage_path else None
        self._sessions: dict[tuple[str, str], Session] = {}
        self._lock = RLock()
        self._load()

    def get_or_create(self, user_id: str, session_id: str) -> Session:
        key = (user_id, session_id)
        with self._lock:
            if key not in self._sessions:
                self._sessions[key] = Session(user_id=user_id, session_id=session_id)
                self._save()
            return self._sessions[key]

    def append(self, session: Session, *messages: Message) -> None:
        with self._lock:
            session.messages.extend(messages)
            self._compact_if_needed(session)
            self._save()

    def context(self, session: Session) -> list[Message]:
        with self._lock:
            messages: list[Message] = []
            if session.summary:
                messages.append(
                    Message(
                        MessageRole.SYSTEM,
                        f"Conversation summary from earlier turns: {session.summary}",
                    )
                )
            messages.extend(session.messages[-self.max_messages :])
            return list(messages)

    def _compact_if_needed(self, session: Session) -> None:
        if len(session.messages) <= self.summary_trigger:
            return
        keep = max(4, self.max_messages // 2)
        old = session.messages[:-keep]
        session.summary = _merge_summary(session.summary, old)
        session.messages = session.messages[-keep:]

    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        sessions = payload.get("sessions", [])
        for item in sessions:
            session = _session_from_dict(item)
            self._sessions[(session.user_id, session.session_id)] = session

    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "sessions": [
                _session_to_dict(session)
                for session in sorted(
                    self._sessions.values(),
                    key=lambda item: (item.user_id, item.session_id),
                )
            ],
        }
        tmp_path = self.storage_path.with_name(f"{self.storage_path.name}.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self.storage_path)


def _merge_summary(existing: str, messages: list[Message]) -> str:
    lines = [existing] if existing else []
    for message in messages:
        if message.role in {MessageRole.USER, MessageRole.TOOL}:
            text = message.content.replace("\n", " ").strip()
            if len(text) > 240:
                text = text[:237] + "..."
            lines.append(f"{message.role.value}: {text}")
    summary = " | ".join(lines)
    return summary[-2000:]


def _session_to_dict(session: Session) -> dict[str, Any]:
    return {
        "user_id": session.user_id,
        "session_id": session.session_id,
        "summary": session.summary,
        "messages": [_message_to_dict(message) for message in session.messages],
    }


def _session_from_dict(item: dict[str, Any]) -> Session:
    return Session(
        user_id=str(item["user_id"]),
        session_id=str(item["session_id"]),
        summary=str(item.get("summary", "")),
        messages=[
            _message_from_dict(message)
            for message in item.get("messages", [])
            if isinstance(message, dict)
        ],
    )


def _message_to_dict(message: Message) -> dict[str, Any]:
    return {
        "role": message.role.value,
        "content": message.content,
        "name": message.name,
        "tool_call_id": message.tool_call_id,
    }


def _message_from_dict(item: dict[str, Any]) -> Message:
    return Message(
        role=MessageRole(str(item["role"])),
        content=str(item.get("content", "")),
        name=item.get("name"),
        tool_call_id=item.get("tool_call_id"),
    )
