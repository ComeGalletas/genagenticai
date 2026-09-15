import unittest

from langchain_core.messages import AIMessage, AIMessageChunk

from app.graph.core.graph import translate_stream_events


def _chunk(text: str, node: str = "chatbot"):
    return ("messages", (AIMessageChunk(content=text), {"langgraph_node": node}))


def _update(node: str, update: dict | None):
    return ("updates", {node: update})


class TranslateStreamEventsTests(unittest.TestCase):
    def test_full_turn_with_tool_call_and_final(self) -> None:
        tool_call = AIMessage(content="", tool_calls=[{"name": "retrieve_information", "args": {"query": "x"}, "id": "1"}])
        raw = [
            _update("prepare_turn", {"user_language": "English"}),
            _chunk(""),                                   # empty tool-call chunk, ignored
            _update("chatbot", {"messages": [tool_call]}),
            _update("tools", {"messages": []}),
            _chunk("The RTX"), _chunk(" 5090 has 32 GB"),
            _update("chatbot", {"messages": [AIMessage("The RTX 5090 has 32 GB")]}),
            _chunk('{"passed": true}', node="judge"),      # judge tokens never forwarded
            _update("judge", {"judge": {"passed": True}}),
            _update("finalize", {"final_response": "<p>The RTX 5090 has 32 GB</p>"}),
        ]
        events = list(translate_stream_events(raw))
        self.assertEqual(events, [
            {"event": "status", "text": "Thinking…"},
            {"event": "reset"},
            {"event": "status", "text": "Searching…"},
            {"event": "delta", "text": "The RTX"},
            {"event": "delta", "text": " 5090 has 32 GB"},
            {"event": "status", "text": "Reviewing the answer…"},
            {"event": "final", "html": "<p>The RTX 5090 has 32 GB</p>"},
        ])

    def test_revise_resets_the_draft(self) -> None:
        raw = [
            _chunk("bad answer"),
            _update("chatbot", {"messages": [AIMessage("bad answer")]}),
            _update("judge", {"judge": {"passed": False}}),
            _update("revise", {"judge_attempts": 1}),
            _chunk("better answer"),
            _update("chatbot", {"messages": [AIMessage("better answer")]}),
            _update("finalize", {"final_response": "<p>better answer</p>"}),
        ]
        events = [e["event"] for e in translate_stream_events(raw)]
        self.assertEqual(events, ["delta", "status", "reset", "status", "delta", "status", "final"])

    def test_unknown_tool_gets_generic_status(self) -> None:
        tool_call = AIMessage(content="", tool_calls=[{"name": "mystery_tool", "args": {}, "id": "1"}])
        events = list(translate_stream_events([_update("chatbot", {"messages": [tool_call]})]))
        self.assertEqual(events[-1], {"event": "status", "text": "Working…"})

    def test_none_update_and_other_modes_are_ignored(self) -> None:
        self.assertEqual(list(translate_stream_events([_update("tools", None), ("values", {})])), [])


if __name__ == "__main__":
    unittest.main()
