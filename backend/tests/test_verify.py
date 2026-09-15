import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graph.core.nodes import build_retrieval_context, finalize
from app.graph.judge.critique import build_critique
from app.graph.judge.verify import (
    FetchedPage,
    VerificationPlan,
    VerifyTarget,
    build_plan,
    extract_links,
    fetch_pages,
    format_verification,
    normalize_url,
    strip_links,
    unreachable_cited_links,
    unverified_sources,
    unverified_sources_note,
)
from app.retrieval.schemas import RetrievalStage


def _doc(stage: str, source: str, title: str = "t") -> dict:
    return {"title": title, "content": "c", "source": source, "stage": stage, "confidence": 0.9, "metadata": {}}


def _retrieval(*docs: dict, tool_call_id: str = "call-now", stage: int = 0) -> dict:
    return {"retrieval_status": "FOUND", "retrieval_query": "q", "retrieval_stage": stage, "retrieval_next_stage": 1,
            "retrieval_documents": list(docs), "tool_call_id": tool_call_id}


def _turn(*extra_messages, call_id: str = "call-now"):
    """A current turn whose retrieval tool call has id `call_id`."""
    return [
        HumanMessage("current question"),
        AIMessage(content="", tool_calls=[{"name": "retrieve_information", "args": {"query": "x"}, "id": call_id}]),
        ToolMessage(content="Retrieval context: 1 document retrieved for query 'x' (stage 0). They are listed in the retrieved context for the current question.", tool_call_id=call_id),
        *extra_messages,
    ]


class ExtractAndNormalizeTests(unittest.TestCase):
    def test_extracts_markdown_html_and_bare_links_once(self) -> None:
        text = ("See [wiki](https://en.wikipedia.org/wiki/X). Also https://en.wikipedia.org/wiki/X, "
                "and <a href=\"https://example.com/a/\">a</a> plus https://example.com/b.")
        self.assertEqual(extract_links(text), ["https://en.wikipedia.org/wiki/X", "https://example.com/a/", "https://example.com/b"])

    def test_markdown_punctuation_glued_to_urls_is_dropped(self) -> None:
        text = "see `https://wroters.com/` and **https://a.example/x** or https://b.example/y."
        self.assertEqual(extract_links(text), ["https://wroters.com/", "https://a.example/x", "https://b.example/y"])

    def test_normalize(self) -> None:
        self.assertEqual(normalize_url("HTTPS://Example.com/Path/#frag"), "https://example.com/Path")
        self.assertEqual(normalize_url("https://example.com/x."), "https://example.com/x")


class BuildPlanTests(unittest.TestCase):
    def test_origins(self) -> None:
        state = {
            "messages": [
                HumanMessage("q"),
                AIMessage(content="", tool_calls=[{"name": "read_webpage", "args": {"url": "https://read.me/page"}, "id": "1"}]),
                ToolMessage(content='[{"job_url": "https://www.linkedin.com/jobs/view/1"}]', tool_call_id="1", name="retrieve_job_postings"),
                AIMessage("answer"),
            ],
            "retrieval": [_retrieval(
                _doc(RetrievalStage.WEB, "https://en.wikipedia.org/wiki/Real"),
                _doc(RetrievalStage.STATIC, "https://wroters.com/", title="Writers Revenge"),
                _doc(RetrievalStage.RAG, "gpu_knowledge.md"),
            )],
        }
        answer = ("[a](https://www.linkedin.com/jobs/view/1) [b](https://read.me/page) [c](https://en.wikipedia.org/wiki/Real) "
                  "[d](https://wroters.com) [e](https://made.up/page)")
        plan = build_plan(state, answer, max_retrievals=3)
        self.assertEqual(sorted(t.origin for t in plan.trusted), ["tool", "tool", "web"])
        targets = {t.url: t for t in plan.targets}
        self.assertEqual(targets["https://wroters.com"].origin, "static")
        self.assertEqual(targets["https://wroters.com"].entry_title, "Writers Revenge")
        self.assertTrue(targets["https://wroters.com"].cited)
        self.assertEqual(targets["https://made.up/page"].origin, "unknown")

    def test_uncited_static_entry_from_this_turn_is_fetched(self) -> None:
        state = {"messages": _turn(AIMessage("no links")),
                 "retrieval": [_retrieval(_doc(RetrievalStage.STATIC, "https://wroters.com/", title="WR"))]}
        plan = build_plan(state, "no links here", max_retrievals=3)
        self.assertEqual(len(plan.targets), 1)
        self.assertFalse(plan.targets[0].cited)
        self.assertEqual(plan.targets[0].origin, "static")

    def test_stale_static_entry_from_earlier_turn_is_ignored_unless_cited(self) -> None:
        stale = _retrieval(_doc(RetrievalStage.STATIC, "https://wroters.com/", title="WR"), tool_call_id="call-old")
        state = {"messages": _turn(AIMessage("about something else")), "retrieval": [stale]}
        self.assertEqual(build_plan(state, "about something else", max_retrievals=3).targets, [])
        cited = build_plan(state, "see https://wroters.com/", max_retrievals=3)
        self.assertEqual([(t.origin, t.cited) for t in cited.targets], [("static", True)])

    def test_cap(self) -> None:
        answer = " ".join(f"https://x{i}.example/p" for i in range(8))
        plan = build_plan({"messages": [HumanMessage("q"), AIMessage(answer)]}, answer, max_retrievals=3, max_targets=3)
        self.assertEqual(len(plan.targets), 3)

    def test_no_links_no_static(self) -> None:
        plan = build_plan({"messages": [HumanMessage("q"), AIMessage("plain")]}, "plain", max_retrievals=3)
        self.assertEqual((plan.targets, plan.trusted), ([], []))
        self.assertIn("cites no links", format_verification(plan, {}))


class FetchAndFormatTests(unittest.TestCase):
    def test_fetch_pages_uses_fetcher_and_keys_by_normalized_url(self) -> None:
        pages = fetch_pages(["https://A.com/x/", "https://a.com/x"], fetcher=lambda u: FetchedPage(u, "ok", text="hello"))
        self.assertEqual(list(pages), ["https://a.com/x"])

    def test_unreachable_cited_links_and_capped(self) -> None:
        plan = VerificationPlan(targets=[
            VerifyTarget("https://dead.example", "unknown", cited=True),
            VerifyTarget("https://ok.example", "static", cited=True, entry_title="E"),
            VerifyTarget("https://uncited.example", "static", cited=False, entry_title="U"),
            VerifyTarget("https://blocked.example", "unknown", cited=True),
            VerifyTarget("https://capped.example", "unknown", cited=True),
        ])
        pages = {
            "https://dead.example": FetchedPage("https://dead.example", "unreachable", detail="DNS"),
            "https://ok.example": FetchedPage("https://ok.example", "ok", text="E is real"),
            "https://uncited.example": FetchedPage("https://uncited.example", "no_text"),
            "https://blocked.example": FetchedPage("https://blocked.example", "blocked", detail="HTTP 403"),
        }
        # blocked and no_text are inconclusive; missing pages and capped links are not
        self.assertEqual(unreachable_cited_links(plan, pages), ["https://dead.example", "https://capped.example"])
        text = format_verification(plan, pages)
        self.assertIn("status: unreachable (DNS)", text)
        self.assertIn('source of curated entry "E"', text)
        self.assertIn("excerpt: E is real", text)
        self.assertIn("status: skipped", text)
        self.assertIn("status: blocked (HTTP 403)", text)

    def test_unverified_sources_are_curated_entries_with_missing_or_empty_pages(self) -> None:
        plan = VerificationPlan(targets=[
            VerifyTarget("https://wroters.com/", "static", cited=False, entry_title="Writers Revenge"),
            VerifyTarget("https://en.wikipedia.org/wiki/Real", "static", cited=False, entry_title="Real"),
            VerifyTarget("https://gone.example", "unknown", cited=True),
        ])
        pages = {
            "https://wroters.com": FetchedPage("https://wroters.com/", "no_text"),
            "https://en.wikipedia.org/wiki/Real": FetchedPage("https://en.wikipedia.org/wiki/Real", "blocked", detail="HTTP 403"),
            "https://gone.example": FetchedPage("https://gone.example", "unreachable"),
        }
        out = unverified_sources(plan, pages)
        self.assertEqual([(u.url, u.title, u.status) for u in out], [("https://wroters.com/", "Writers Revenge", "no_text")])


class DownstreamTests(unittest.TestCase):
    def test_context_marks_unverified_sources(self) -> None:
        state = {
            "messages": _turn(),
            "retrieval": [_retrieval(_doc(RetrievalStage.STATIC, "https://wroters.com/", title="WR"),
                                     _doc(RetrievalStage.STATIC, "https://ok.example/x", title="OK"))],
            "judge": {"passed": False, "unverified_sources": ["https://wroters.com"]},
        }
        text = build_retrieval_context(state).content
        self.assertIn("Source: https://wroters.com/  [UNVERIFIED: source page missing or empty]", text)
        self.assertIn("Source: https://ok.example/x\n", text)

    def test_critique_tells_chatbot_to_drop_unverified_claims_not_search(self) -> None:
        msg = build_critique({"issues": ["Spinoff unsupported."], "grounded": False, "needs_more_info": True,
                              "unverified_sources": ["https://wroters.com/"], "unsupported_links": ["https://wroters.com/x"]})
        self.assertIn("- https://wroters.com/", msg.content)
        self.assertIn("Do not search for them again", msg.content)
        self.assertNotIn("Search again with your retrieval tool", msg.content)
        self.assertIn("These links could not be verified", msg.content)

    def test_finalize_strips_links_and_names_unverified_source(self) -> None:
        state = {
            "messages": _turn(AIMessage("Spinoff: [WR](https://wroters.com/x). Sequel: none.")),
            "retrieval": [_retrieval(_doc(RetrievalStage.STATIC, "https://wroters.com/", title="Writers Revenge"))],
            "judge": {"passed": False, "original_response": "x", "unsupported_links": ["https://wroters.com/x"],
                      "unverified_sources": ["https://wroters.com/"],
                      "issues": ["The answer invents a request for clarification."]},
        }
        html = finalize(state).update["final_response"]
        self.assertNotIn("wroters.com", html)
        self.assertIn("Spinoff: WR.", html)
        self.assertIn("1 reference could not be verified", html)
        self.assertIn("“Writers Revenge” could not be verified", html)
        self.assertNotIn("request for clarification", html)  # judge prose never reaches the user

    def test_finalize_note_needs_a_source_from_this_turn(self) -> None:
        state = {
            "messages": [HumanMessage("q"), AIMessage("plain answer")],
            "retrieval": [_retrieval(_doc(RetrievalStage.STATIC, "https://wroters.com/", title="WR"), tool_call_id="call-old")],
            "judge": {"passed": False, "unsupported_links": [], "unverified_sources": ["https://wroters.com/"], "issues": ["x"]},
        }
        self.assertNotIn("Note:", finalize(state).update["final_response"])
        self.assertEqual(unverified_sources_note(state, ["https://wroters.com/"]), "")

    def test_finalize_passed_answer_untouched(self) -> None:
        state = {"messages": [HumanMessage("q"), AIMessage("fine [w](https://ok.example)")],
                 "judge": {"passed": True, "unsupported_links": [], "unverified_sources": [], "issues": []}}
        html = finalize(state).update["final_response"]
        self.assertIn('href="https://ok.example"', html)
        self.assertNotIn("Note:", html)

    def test_critique_web_searched_suppresses_search_again(self) -> None:
        verdict = {"issues": ["Thin."], "grounded": True, "needs_more_info": True}
        self.assertIn("Search again with your retrieval tool", build_critique(verdict).content)
        again = build_critique(verdict, web_searched=True).content
        self.assertNotIn("Search again with your retrieval tool", again)
        self.assertIn("already searched the web", again)
        self.assertIn("do not ask the user to clarify", again)


class StripLinksTests(unittest.TestCase):
    def test_strips_only_listed_links_keeping_labels(self) -> None:
        text = "See [spinoff](https://wroters.com/) and [wiki](https://en.wikipedia.org/wiki/X) or https://wroters.com/ now."
        out, removed = strip_links(text, ["https://wroters.com"])
        self.assertEqual(removed, 2)
        self.assertEqual(out, "See spinoff and [wiki](https://en.wikipedia.org/wiki/X) or now.")

    def test_bare_url_in_parentheses_leaves_no_empty_parens(self) -> None:
        out, removed = strip_links("The source page (https://wroters.com/) could not be verified.", ["https://wroters.com/"])
        self.assertEqual((out, removed), ("The source page could not be verified.", 1))

    def test_source_note_lists_titles_only(self) -> None:
        state = {"messages": _turn(AIMessage("a")),
                 "retrieval": [_retrieval(_doc(RetrievalStage.STATIC, "https://wroters.com/", title="Writers Revenge"))]}
        note = unverified_sources_note(state, ["https://wroters.com"])
        self.assertIn("“Writers Revenge” could not be verified", note)
        self.assertNotIn("wroters", note)

    def test_nothing_to_strip(self) -> None:
        self.assertEqual(strip_links("plain text", ["https://x.example"]), ("plain text", 0))


if __name__ == "__main__":
    unittest.main()
