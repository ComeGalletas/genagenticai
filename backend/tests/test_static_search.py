import unittest

from app.search.static import query_static_search


class StaticSearchTests(unittest.TestCase):
    def test_stopword_alone_does_not_match(self) -> None:
        # "the" appears in "the matrix movie ..." keywords; a single generic hit must not match.
        self.assertEqual(query_static_search("What is the TDP of the RTX 5090?"), [])

    def test_single_meaningful_token_is_not_enough(self) -> None:
        self.assertEqual(query_static_search("tell me about python"), [])

    def test_phrase_match_is_decisive(self) -> None:
        results = query_static_search("Which studio made Hollow Knight?")
        self.assertTrue(results)
        self.assertIn("Hollow Knight", results[0].title)

    def test_two_token_match(self) -> None:
        results = query_static_search("python langgraph")
        self.assertTrue(results)
        self.assertIn("LangGraph", results[0].title)

    def test_ranking_by_hits(self) -> None:
        results = query_static_search("elden ring open world rpg by fromsoftware, souls like")
        self.assertIn("Elden Ring", results[0].title)
        self.assertGreaterEqual(results[0].metadata["keyword_hits"], 3)

    def test_full_name_is_decisive(self) -> None:
        for query in ("Which studio made Hollow Knight?", "what is clair obscur: expedition 33?", "What is LangGraph used for in Python?"):
            results = query_static_search(query)
            self.assertTrue(results, query)
            self.assertTrue(results[0].metadata["decisive"], query)
            self.assertEqual(results[0].confidence, 0.95)

    def test_partial_name_is_tentative(self) -> None:
        results = query_static_search("what is expedition 33")
        self.assertTrue(results)
        self.assertIn("Expedition 33", results[0].title)
        self.assertFalse(results[0].metadata["decisive"])
        self.assertEqual(results[0].confidence, 0.6)

    def test_decisive_entries_rank_first(self) -> None:
        results = query_static_search("compare clair obscur: expedition 33 with clair obscur: writers revenge")
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.metadata["decisive"] for r in results))

    def test_case_insensitive_and_punctuation(self) -> None:
        self.assertTrue(query_static_search("who DEVELOPED elden-ring?"))

    def test_empty_query(self) -> None:
        self.assertEqual(query_static_search(""), [])


if __name__ == "__main__":
    unittest.main()
