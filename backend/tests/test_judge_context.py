import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graph.core.router import turn_retrievals, web_searched_this_turn
from app.graph.judge.critique import build_critique
from app.graph.judge.node import _format_context, _last_message_text, _recent_conversation, revise


def _retrieval(title: str, content: str, tool_call_id: str = "1", stage: int = 1) -> dict:
    return {
        "retrieval_status": "FOUND", "retrieval_query": "q", "retrieval_stage": stage, "retrieval_next_stage": 2,
        "retrieval_documents": [{"title": title, "content": content, "source": "src", "stage": "rag",
                                 "confidence": 0.5, "metadata": {}}],
        "tool_call_id": tool_call_id,
    }


def _search_turn(question: str, answer: str, call_id: str) -> list:
    return [
        HumanMessage(question),
        AIMessage(content="", tool_calls=[{"name": "retrieve_information", "args": {"query": "x"}, "id": call_id}]),
        ToolMessage(content="Retrieval context: 1 document retrieved for query 'x' (stage 1). They are listed in the retrieved context for the current question.", tool_call_id=call_id),
        AIMessage(answer),
    ]


class FormatContextTests(unittest.TestCase):
    def test_includes_this_turns_tool_outputs(self) -> None:
        state = {"messages": [
            HumanMessage("What time is it?"),
            AIMessage(content="", tool_calls=[{"name": "get_current_time", "args": {}, "id": "1"}]),
            ToolMessage(content='{"Date": "2026-09-14", "Time": "22:54:00"}', tool_call_id="1", name="get_current_time"),
            AIMessage("It is 22:54 nya~"),
        ]}
        context = _format_context(state)
        self.assertIn("[tool output: get_current_time]", context)
        self.assertIn("22:54:00", context)

    def test_retrieval_acknowledgements_are_not_duplicated(self) -> None:
        state = {"messages": _search_turn("VRAM?", "32 GB", "1"), "retrieval": [_retrieval("RTX 5090", "VRAM: 32 GB GDDR7")]}
        context = _format_context(state)
        self.assertIn("VRAM: 32 GB GDDR7", context)
        self.assertNotIn("Retrieval context:", context)
        self.assertNotIn("[tool output", context)

    def test_earlier_turn_documents_are_titles_only(self) -> None:
        state = {
            "messages": _search_turn("what is expedition 33", "an RPG", "old") + _search_turn("best teams for her?", "Furina teams", "new"),
            "retrieval": [_retrieval("Expedition 33 entry", "long stale content", tool_call_id="old"),
                          _retrieval("Furina team guide", "Furina pairs well with Neuvillette", tool_call_id="new")],
        }
        context = _format_context(state)
        self.assertIn("Furina pairs well with Neuvillette", context)
        self.assertNotIn("long stale content", context)
        self.assertIn("background only", context)
        self.assertIn("Expedition 33 entry", context)
        self.assertEqual([r["tool_call_id"] for r in turn_retrievals(state)], ["new"])

    def test_recent_conversation_and_web_flag(self) -> None:
        critique = build_critique({"issues": ["x"], "grounded": False, "needs_more_info": False})
        messages = (
            _search_turn("what is expedition 33", "It is an RPG nya~", "a")
            + [HumanMessage("Who is Furina?"), AIMessage("Furina is the Hydro Archon.")]
            + _search_turn("what are the best teams for her?", "bad answer", "c")
            + [critique, AIMessage("better answer")]
        )
        state = {"messages": messages, "retrieval": [_retrieval("guide", "c", tool_call_id="c", stage=2)]}
        convo = _recent_conversation(state)
        self.assertIn("User: what is expedition 33", convo)
        self.assertIn("Assistant: Furina is the Hydro Archon.", convo)
        self.assertNotIn("best teams", convo)          # the current question is not part of the history
        self.assertNotIn("Internal review", convo)     # critiques never appear
        self.assertTrue(web_searched_this_turn(state))
        self.assertEqual(_recent_conversation({"messages": [HumanMessage("first")]}), "(this is the first question of the conversation)")

    def test_chatbot_and_judge_agree_on_this_turns_documents(self) -> None:
        from app.graph.core.nodes import build_retrieval_context

        state = {
            "messages": _search_turn("cars?", "car answer", "cars") + _search_turn("food?", "food answer", "food"),
            "retrieval": [_retrieval("Car page", "car content", tool_call_id="cars"),
                          _retrieval("Food page", "food content", tool_call_id="food")],
        }
        judge_text = _format_context(state)
        chatbot_text = build_retrieval_context(state).content
        for text in (judge_text, chatbot_text):
            self.assertIn("food content", text)
            self.assertNotIn("car content", text)
            self.assertIn("Car page", text)  # earlier turn: title only, for both readers

    def test_previous_turn_tool_outputs_are_excluded(self) -> None:
        state = {"messages": [
            HumanMessage("time?"),
            AIMessage(content="", tool_calls=[{"name": "get_current_time", "args": {}, "id": "1"}]),
            ToolMessage(content="old tool output", tool_call_id="1", name="get_current_time"),
            AIMessage("It is late"),
            HumanMessage("thanks"),
            AIMessage("welcome"),
        ]}
        self.assertEqual(_format_context(state), "(no documents or tool outputs for this turn)")

    def test_question_lookup_skips_critique(self) -> None:
        critique = build_critique({"issues": ["x"], "grounded": False, "needs_more_info": False})
        state = {"messages": [HumanMessage("real question"), AIMessage("bad"), critique, AIMessage("better")]}
        self.assertEqual(_last_message_text(state, "human"), "real question")
        self.assertEqual(_last_message_text(state, "ai"), "better")


class ReviseTests(unittest.TestCase):
    def test_revise_appends_critique_and_counts(self) -> None:
        state = {"messages": [HumanMessage("q"), AIMessage("a")], "judge_attempts": 0,
                 "judge": {"passed": False, "grounded": False, "needs_more_info": False, "issues": ["Wrong."]}}
        update = revise(state).update
        self.assertEqual(update["judge_attempts"], 1)
        self.assertIn("- Wrong.", update["messages"][0].content)

    def test_revise_without_verdict_is_noop(self) -> None:
        self.assertEqual(revise({"messages": []}).update, {})


if __name__ == "__main__":
    unittest.main()
