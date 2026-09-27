from __future__ import annotations

from threading import RLock

from .models import Message, MessageRole, Session


class SessionStore:
    def __init__(self, max_messages: int = 20, summary_trigger: int = 24) -> None:
        self.max_messages = max_messages
        self.summary_trigger = summary_trigger
        self._sessions: dict[tuple[str, str], Session] = {}
        self._lock = RLock()

    def get_or_create(self, user_id: str, session_id: str) -> Session:
        key = (user_id, session_id)
        with self._lock:
            if key not in self._sessions:
                self._sessions[key] = Session(user_id=user_id, session_id=session_id)
            return self._sessions[key]

    def append(self, session: Session, *messages: Message) -> None:
        with self._lock:
            session.messages.extend(messages)
            self._compact_if_needed(session)

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
