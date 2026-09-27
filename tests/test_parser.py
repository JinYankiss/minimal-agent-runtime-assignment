import unittest

from mini_agent.errors import LLMProtocolError
from mini_agent.models import FinalAnswer, ToolCall
from mini_agent.parser import parse_decision


class ParserTest(unittest.TestCase):
    def test_parse_tool_call_and_final(self):
        tool = parse_decision(
            '{"thought":"need math","action":{"type":"tool_call","call_id":"c1","tool_name":"calculator","arguments":{"expression":"2+3"}}}'
        )
        final = parse_decision(
            '{"thought":"done","action":{"type":"final","answer":"5"}}'
        )
        self.assertIsInstance(tool, ToolCall)
        self.assertEqual(tool.arguments["expression"], "2+3")
        self.assertIsInstance(final, FinalAnswer)
        self.assertEqual(final.answer, "5")

    def test_parse_rejects_invalid_shape(self):
        with self.assertRaises(LLMProtocolError):
            parse_decision(
                '{"action":{"type":"tool_call","tool_name":"calculator"}}'
            )
