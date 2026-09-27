import tempfile
import time
import unittest
from pathlib import Path

from mini_agent.errors import MaxRoundsExceeded
from mini_agent.llm import ScriptedLLMClient
from mini_agent.models import Message, MessageRole, ToolContext, ToolEffect, ToolSpec
from mini_agent.registry import ToolRegistry
from mini_agent.runtime import AgentRuntime
from mini_agent.session import SessionStore
from mini_agent.tools import ToolExecutionLedger, ToolExecutor, default_registry
from mini_agent.trace import TraceLogger


def runtime_for(responses: list[str], trace_dir: Path) -> AgentRuntime:
    registry = default_registry()
    return AgentRuntime(
        llm=ScriptedLLMClient(responses),
        registry=registry,
        sessions=SessionStore(max_messages=6),
        executor=ToolExecutor(registry),
        tracer=TraceLogger(trace_dir),
        max_rounds=4,
    )


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_direct_reply_and_trace(self):
        runtime = runtime_for(['{"action":{"type":"final","answer":"你好"}}'], self.tmp_path)
        result = runtime.run("u1", "s1", "你好")
        self.assertEqual(result.answer, "你好")
        self.assertEqual(result.rounds, 1)
        self.assertTrue((self.tmp_path / f"{result.run_id}.jsonl").exists())

    def test_tool_loop_and_follow_up(self):
        responses = [
            '{"thought":"calculate","action":{"type":"tool_call","call_id":"c1","tool_name":"calculator","arguments":{"expression":"2+3"}}}',
            '{"action":{"type":"final","answer":"结果是 5"}}',
        ]
        runtime = runtime_for(responses, self.tmp_path)
        result = runtime.run("u1", "s1", "算 2+3")
        self.assertEqual(result.answer, "结果是 5")
        session = runtime.sessions.get_or_create("u1", "s1")
        self.assertTrue(any("tool=calculator" in item.content for item in session.messages))

    def test_sessions_are_isolated(self):
        runtime = runtime_for(
            ['{"action":{"type":"final","answer":"ok"}}'] * 2, self.tmp_path
        )
        runtime.run("u1", "window-1", "记住 A")
        runtime.run("u1", "window-2", "记住 B")
        first = runtime.sessions.get_or_create("u1", "window-1")
        second = runtime.sessions.get_or_create("u1", "window-2")
        self.assertTrue(any("记住 A" in item.content for item in first.messages))
        self.assertFalse(any("记住 B" in item.content for item in first.messages))
        self.assertTrue(any("记住 B" in item.content for item in second.messages))

    def test_context_compaction_keeps_summary_and_recent_messages(self):
        store = SessionStore(max_messages=4, summary_trigger=5)
        session = store.get_or_create("u", "s")
        for index in range(6):
            store.append(session, Message(MessageRole.USER, f"message-{index}"))
        self.assertTrue(session.summary)
        self.assertIn("message-0", session.summary)
        self.assertLessEqual(len(store.context(session)), 5)

    def test_max_rounds_is_enforced(self):
        loop = '{"action":{"type":"tool_call","call_id":"same","tool_name":"weather","arguments":{"city":"厦门"}}}'
        runtime = runtime_for([loop] * 4, self.tmp_path)
        with self.assertRaises(MaxRoundsExceeded):
            runtime.run("u", "s", "天气")

    def test_non_idempotent_tool_is_not_retried(self):
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="charge",
                description="charge card",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                effect=ToolEffect.NON_IDEMPOTENT_WRITE,
                handler=lambda _ctx, _args: "charged",
                max_retries=1,
            )
        )
        executor = ToolExecutor(registry, ToolExecutionLedger())
        context = ToolContext("u", "s", "r", "c", "stable-key")
        result = executor.execute(context, "charge", {})
        self.assertFalse(result.ok)
        self.assertTrue(result.requires_status_query)

    def test_idempotent_write_replays_ledger_result(self):
        count = {"value": 0}

        def write(_ctx, _args):
            count["value"] += 1
            return "written"

        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="write",
                description="write once",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                effect=ToolEffect.IDEMPOTENT_WRITE,
                handler=write,
                max_retries=1,
            )
        )
        executor = ToolExecutor(registry)
        context = ToolContext("u", "s", "r", "c", "stable-key")
        first = executor.execute(context, "write", {})
        second = executor.execute(context, "write", {})
        self.assertTrue(first.ok and second.ok)
        self.assertTrue(second.replayed)
        self.assertEqual(count["value"], 1)

    def test_timeout_is_retried_with_the_same_key_for_idempotent_tool(self):
        keys = []

        def slow_write(context, _args):
            keys.append(context.idempotency_key)
            time.sleep(0.02)
            return "done"

        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="slow_write",
                description="slow idempotent write",
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                effect=ToolEffect.IDEMPOTENT_WRITE,
                handler=slow_write,
                timeout_seconds=0.001,
                max_retries=1,
            )
        )
        executor = ToolExecutor(registry)
        context = ToolContext("u", "s", "r", "c", "stable-key")
        result = executor.execute(context, "slow_write", {})
        self.assertFalse(result.ok)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(keys, ["stable-key", "stable-key"])
