import unittest

from app.retrieval.engine import RetrievalEngine
from app.retrieval.schemas import RetrievalResult


def _r(tag: str) -> RetrievalResult:
    return RetrievalResult(title=tag, content=tag, source=tag, stage="test")


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Stage:
    """Callable stage that records how many times it ran."""

    def __init__(self, name: str, results: list[RetrievalResult]) -> None:
        self.__name__ = name
        self.results = results
        self.calls = 0

    def __call__(self, query: str) -> list[RetrievalResult]:
        self.calls += 1
        return list(self.results)


class RunPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.s0 = _Stage("static", [])
        self.s1 = _Stage("rag", [_r("rag-doc")])
        self.s2 = _Stage("web", [_r("web-1"), _r("web-2")])
        self.engine = RetrievalEngine([self.s0, self.s1, self.s2], stop_min_results=1, cache_ttl=0)

    def test_stops_at_first_stage_with_results(self) -> None:
        results, last, nxt = self.engine.run_pipeline("q")
        self.assertEqual([r.title for r in results], ["rag-doc"])
        self.assertEqual((last, nxt), (1, 2))
        self.assertEqual((self.s0.calls, self.s1.calls, self.s2.calls), (1, 1, 0))

    def test_start_stage_skips_cheaper_stages(self) -> None:
        results, last, nxt = self.engine.run_pipeline("q", start_stage=2)
        self.assertEqual(len(results), 2)
        self.assertEqual((last, nxt), (2, -1))
        self.assertEqual((self.s0.calls, self.s1.calls), (0, 0))

    def test_all_stages_empty_runs_to_the_end(self) -> None:
        engine = RetrievalEngine([_Stage("a", []), _Stage("b", [])], stop_min_results=1, cache_ttl=0)
        results, last, nxt = engine.run_pipeline("q")
        self.assertEqual((results, last, nxt), ([], 1, -1))

    def test_higher_threshold_accumulates_across_stages(self) -> None:
        engine = RetrievalEngine([self.s0, self.s1, self.s2], stop_min_results=3, cache_ttl=0)
        results, last, nxt = engine.run_pipeline("q")
        self.assertEqual(len(results), 3)
        self.assertEqual((last, nxt), (2, -1))

    def test_tentative_results_do_not_stop_the_pipeline(self) -> None:
        tentative = RetrievalResult(title="game", content="", source="", stage="static", metadata={"decisive": False})
        engine = RetrievalEngine([_Stage("static", [tentative]), _Stage("rag", []), self.s2], stop_min_results=1, cache_ttl=0)
        results, last, nxt = engine.run_pipeline("q")
        self.assertEqual([r.title for r in results], ["game", "web-1", "web-2"])
        self.assertEqual((last, nxt), (2, -1))

    def test_start_beyond_pipeline(self) -> None:
        self.assertEqual(self.engine.run_pipeline("q", start_stage=9), ([], 9, -1))

    def test_run_stage_out_of_range(self) -> None:
        self.assertEqual(self.engine.run_stage("q", 5), ([], -1))


class ResolveStartStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = _Clock()
        self.s0, self.s1, self.s2 = _Stage("static", []), _Stage("rag", []), _Stage("web", [_r("w")])
        self.engine = RetrievalEngine([self.s0, self.s1, self.s2], cache_ttl=60, clock=self.clock)

    def test_requested_higher_stage_ignored_on_first_call(self) -> None:
        self.assertEqual(self.engine.resolve_start_stage("q", 2), 0)
        self.assertEqual(self.engine.resolve_start_stage("q", 0), 0)

    def test_honored_after_lower_stages_were_tried(self) -> None:
        self.engine.run_pipeline("q", start_stage=0)  # runs 0, 1, 2 (only web has results)
        self.assertEqual(self.engine.tried_stages("q"), {0, 1, 2})
        self.assertEqual(self.engine.resolve_start_stage("Q ", 2), 2)  # normalised query

    def test_partial_coverage_is_not_enough(self) -> None:
        self.engine.run_stage("q", 0)
        self.assertEqual(self.engine.resolve_start_stage("q", 2), 0)
        self.assertEqual(self.engine.resolve_start_stage("q", 1), 1)

    def test_expired_cache_resets_escalation(self) -> None:
        self.engine.run_pipeline("q", start_stage=0)
        self.clock.now += 61
        self.assertEqual(self.engine.tried_stages("q"), set())
        self.assertEqual(self.engine.resolve_start_stage("q", 2), 0)

    def test_escalation_bypasses_weak_early_exit(self) -> None:
        # Stage 1 returns something weak, so a plain run stops there; an explicit retry reaches the web.
        weak = RetrievalEngine([_Stage("static", []), _Stage("rag", [_r("weak")]), self.s2], cache_ttl=60, clock=self.clock)
        first, last, _ = weak.run_pipeline("q", weak.resolve_start_stage("q", 2))
        self.assertEqual(([r.title for r in first], last), (["weak"], 1))
        second, last, _ = weak.run_pipeline("q", weak.resolve_start_stage("q", 2))
        self.assertEqual(([r.title for r in second], last), (["w"], 2))


class CacheTests(unittest.TestCase):
    def test_hit_within_ttl_and_expiry(self) -> None:
        clock = _Clock()
        stage = _Stage("web", [_r("x")])
        engine = RetrievalEngine([stage], cache_ttl=60, clock=clock)

        engine.run_stage("Hello  World", 0)
        engine.run_stage("hello world", 0)  # normalised key: same query
        self.assertEqual(stage.calls, 1)

        clock.now += 61
        engine.run_stage("hello world", 0)
        self.assertEqual(stage.calls, 2)

    def test_cache_is_per_stage_and_bounded(self) -> None:
        clock = _Clock()
        a, b = _Stage("a", [_r("a")]), _Stage("b", [_r("b")])
        engine = RetrievalEngine([a, b], cache_ttl=60, cache_size=2, clock=clock)
        engine.run_stage("q", 0)
        engine.run_stage("q", 1)
        engine.run_stage("other", 0)  # evicts the oldest entry ("q", 0)
        engine.run_stage("q", 0)
        self.assertEqual(a.calls, 3)
        self.assertEqual(b.calls, 1)

    def test_cached_results_are_copies(self) -> None:
        stage = _Stage("a", [_r("a")])
        engine = RetrievalEngine([stage], cache_ttl=60)
        first, _ = engine.run_stage("q", 0)
        first.append(_r("mutated"))
        second, _ = engine.run_stage("q", 0)
        self.assertEqual(len(second), 1)


if __name__ == "__main__":
    unittest.main()
