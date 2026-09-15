import unittest
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.graph.core import state as state_module
from app.graph.core.nodes import finalize, prepare_turn, trim_history
from app.graph.core.state import bounded_retrievals


def _entry(tag: str) -> dict:
    return {"retrieval_status": "FOUND", "retrieval_query": tag, "retrieval_stage": 0,
            "retrieval_next_stage": 1, "retrieval_documents": []}


class BoundedRetrievalsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig = state_module.RETRIEVAL_MAX_ENTRIES
        state_module.RETRIEVAL_MAX_ENTRIES = 2

    def tearDown(self) -> None:
        state_module.RETRIEVAL_MAX_ENTRIES = self._orig

    def test_appends_and_keeps_newest(self) -> None:
        out = bounded_retrievals([_entry("a")], [_entry("b"), _entry("c")])
        self.assertEqual([e["retrieval_query"] for e in out], ["b", "c"])

    def test_none_resets(self) -> None:
        self.assertEqual(bounded_retrievals([_entry("a")], None), [])

    def test_empty_left(self) -> None:
        self.assertEqual(len(bounded_retrievals(None, [_entry("a")])), 1)

    def test_reducer_works_inside_langgraph(self) -> None:
        class S(TypedDict):
            retrieval: Annotated[list, bounded_retrievals]

        g = StateGraph(S)
        g.add_node("add", lambda s: {"retrieval": [_entry("x"), _entry("y"), _entry("z")]})
        g.add_node("reset", lambda s: {"retrieval": None})
        g.add_edge(START, "add")
        g.add_edge("add", "reset")
        g.add_edge("reset", END)
        result = g.compile().invoke({"retrieval": [_entry("old")]})
        self.assertEqual(result["retrieval"], [])

        g2 = StateGraph(S)
        g2.add_node("add", lambda s: {"retrieval": [_entry("x"), _entry("y"), _entry("z")]})
        g2.add_edge(START, "add")
        g2.add_edge("add", END)
        result = g2.compile().invoke({"retrieval": [_entry("old")]})
        self.assertEqual([e["retrieval_query"] for e in result["retrieval"]], ["y", "z"])


class TrimHistoryTests(unittest.TestCase):
    def test_short_history_untouched(self) -> None:
        msgs = [HumanMessage("hi"), AIMessage("hello"), HumanMessage("q")]
        self.assertEqual(trim_history(msgs), msgs)

    def test_old_turns_dropped_and_window_starts_on_human(self) -> None:
        old = []
        for i in range(40):
            old += [HumanMessage(f"question {i} " + "x" * 800), AIMessage("answer " + "y" * 800)]
        current = [HumanMessage("latest question")]
        trimmed = trim_history(old + current)
        self.assertLess(len(trimmed), len(old) + 1)
        self.assertEqual(trimmed[0].type, "human")
        self.assertIs(trimmed[-1], current[0])

    def test_tool_pairs_are_not_split(self) -> None:
        # ~7.5k approximate tokens each, so the old turn cannot fit in the 6k budget next to the current one.
        old = [HumanMessage("a " + "x" * 30000), AIMessage("b " + "y" * 30000)]
        turn = [
            HumanMessage("VRAM?"),
            AIMessage(content="", tool_calls=[{"name": "retrieve_information", "args": {"query": "x"}, "id": "1"}]),
            ToolMessage(content="ctx " + "z" * 500, tool_call_id="1"),
        ]
        trimmed = trim_history(old + turn)
        self.assertEqual(trimmed, turn)

    def test_oversized_current_turn_is_kept(self) -> None:
        turn = [
            HumanMessage("big question"),
            AIMessage(content="", tool_calls=[{"name": "read_webpage", "args": {"url": "u"}, "id": "1"}]),
            ToolMessage(content="page " * 20000, tool_call_id="1"),  # far above the budget
        ]
        self.assertEqual(trim_history([HumanMessage("old"), AIMessage("old answer")] + turn), turn)


class FinalizeTests(unittest.TestCase):
    def test_renders_last_answer(self) -> None:
        state = {"messages": [HumanMessage("q"), AIMessage("**bold** answer")]}
        self.assertEqual(finalize(state).update["final_response"], "<p><strong>bold</strong> answer</p>")

    def test_empty_revision_falls_back_to_original(self) -> None:
        state = {
            "messages": [HumanMessage("q"), AIMessage("first answer"), AIMessage("")],
            "judge": {"original_response": "first answer", "passed": False},
        }
        self.assertEqual(finalize(state).update["final_response"], "<p>first answer</p>")

    def test_empty_with_no_original_stays_empty(self) -> None:
        state = {"messages": [HumanMessage("q"), AIMessage("")]}
        self.assertEqual(finalize(state).update["final_response"], "")


class PrepareTurnTests(unittest.TestCase):
    def test_resets_judge_state_and_detects_language(self) -> None:
        state = {
            "messages": [HumanMessage("Hola, ¿me puedes explicar cómo funciona la memoria de una tarjeta gráfica moderna?")],
            "judge": {"passed": False},
            "judge_attempts": 1,
            "final_response": "<p>old</p>",
            "retrieval": [_entry("keep me")],
        }
        update = prepare_turn(state).update
        self.assertEqual(update["user_language"], "Spanish")
        self.assertIsNone(update["judge"])
        self.assertEqual(update["judge_attempts"], 0)
        self.assertIsNone(update["final_response"])
        self.assertNotIn("retrieval", update)


if __name__ == "__main__":
    unittest.main()
