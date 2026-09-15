import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graph.core.router import route_chatbot, route_judge, turn_messages
from app.graph.judge.critique import CRITIQUE_NAME, build_critique, is_critique


def _verdict(**overrides):
    base = {
        "original_response": "x",
        "passed": False,
        "grounded": False,
        "needs_more_info": False,
        "issues": ["Invented a price."],
        "error": None,
    }
    return {**base, **overrides}


class RouteChatbotTests(unittest.TestCase):
    def test_tool_calls_go_to_tools(self) -> None:
        ai = AIMessage(content="", tool_calls=[{"name": "retrieve_information", "args": {"query": "x"}, "id": "1"}])
        self.assertEqual(route_chatbot({"messages": [HumanMessage("q"), ai]}), "tools")

    def test_short_no_tool_reply_skips_judge(self) -> None:
        state = {"messages": [HumanMessage("hi!"), AIMessage("Hello nya~")]}
        self.assertEqual(route_chatbot(state), "finalize")

    def test_long_no_tool_reply_is_judged(self) -> None:
        state = {"messages": [HumanMessage("explain"), AIMessage("word " * 100)]}
        self.assertEqual(route_chatbot(state), "judge")

    def test_tool_backed_reply_is_judged_even_if_short(self) -> None:
        state = {"messages": [
            HumanMessage("VRAM?"),
            AIMessage(content="", tool_calls=[{"name": "retrieve_information", "args": {"query": "x"}, "id": "1"}]),
            ToolMessage(content="ctx", tool_call_id="1"),
            AIMessage("32 GB nya~"),
        ]}
        self.assertEqual(route_chatbot(state), "judge")

    def test_tools_from_previous_turn_do_not_count(self) -> None:
        state = {"messages": [
            HumanMessage("VRAM?"),
            AIMessage(content="", tool_calls=[{"name": "retrieve_information", "args": {"query": "x"}, "id": "1"}]),
            ToolMessage(content="ctx", tool_call_id="1"),
            AIMessage("32 GB nya~"),
            HumanMessage("thanks!"),
            AIMessage("Anytime nya~"),
        ]}
        self.assertEqual(route_chatbot(state), "finalize")


class RouteJudgeTests(unittest.TestCase):
    def test_pass_finalizes(self) -> None:
        self.assertEqual(route_judge({"messages": [], "judge": _verdict(passed=True), "judge_attempts": 0}), "finalize")

    def test_missing_verdict_finalizes(self) -> None:
        self.assertEqual(route_judge({"messages": []}), "finalize")

    def test_first_failure_revises(self) -> None:
        self.assertEqual(route_judge({"messages": [], "judge": _verdict(), "judge_attempts": 0}), "revise")

    def test_failure_after_retry_finalizes(self) -> None:
        self.assertEqual(route_judge({"messages": [], "judge": _verdict(), "judge_attempts": 1}), "finalize")


class CritiqueTests(unittest.TestCase):
    def test_critique_is_tagged_and_lists_issues(self) -> None:
        msg = build_critique(_verdict(issues=["Wrong VRAM.", "Invented price."]))
        self.assertEqual(msg.name, CRITIQUE_NAME)
        self.assertTrue(is_critique(msg))
        self.assertIn("- Wrong VRAM.", msg.content)
        self.assertIn("- Invented price.", msg.content)
        self.assertIn("only the retrieved context", msg.content)

    def test_needs_more_info_asks_for_another_search(self) -> None:
        msg = build_critique(_verdict(needs_more_info=True, grounded=True))
        self.assertIn("Search again", msg.content)

    def test_critique_does_not_start_a_new_turn(self) -> None:
        critique = build_critique(_verdict())
        messages = [HumanMessage("q"), AIMessage("bad"), critique, AIMessage("better")]
        self.assertEqual(turn_messages({"messages": messages}), messages[1:])
        self.assertFalse(is_critique(HumanMessage("q")))


if __name__ == "__main__":
    unittest.main()
