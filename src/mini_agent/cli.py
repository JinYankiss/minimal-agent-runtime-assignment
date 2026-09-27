from __future__ import annotations

import os
import sys

from .llm import OpenAICompatibleClient
from .runtime import AgentRuntime


def main() -> int:
    try:
        runtime = AgentRuntime.with_defaults(
            OpenAICompatibleClient.from_environment(),
            trace_dir=os.getenv("AGENT_TRACE_DIR", "traces"),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    user_id = input("user_id> ").strip()
    session_id = input("session_id> ").strip()
    print("输入 exit 退出")
    while True:
        try:
            user_input = input("you> ")
        except EOFError:
            return 0
        if user_input.strip().lower() == "exit":
            return 0
        try:
            result = runtime.run(user_id, session_id, user_input)
            print(f"agent> {result.answer}")
            print(f"trace> {result.trace_path}")
        except Exception as exc:  # CLI boundary: show failure and keep interactive loop alive
            print(f"agent_error> {exc}", file=sys.stderr)
