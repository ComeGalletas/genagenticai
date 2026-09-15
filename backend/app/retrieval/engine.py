import logging
import time
from collections import OrderedDict
from typing import Callable

from ..config import RETRIEVAL_CACHE_SIZE, RETRIEVAL_CACHE_TTL, RETRIEVAL_STOP_MIN_RESULTS
from .schemas import RetrievalResult

logger = logging.getLogger(__name__)

StageFn = Callable[[str], list[RetrievalResult]]


class RetrievalEngine:
    """Runs an ordered list of retrieval strategies (cheap first, expensive last).

    Each strategy is a callable taking a query string and returning RetrievalResult objects.
    `run_pipeline` walks the stages from a starting point and stops at the first stage that
    yields enough results, so a question the static index or the local RAG store can answer
    never reaches the web. `run_stage` executes a single stage and is what node mode uses.
    Results are cached per (query, stage) for a short TTL. Knows nothing about LangGraph.
    """

    def __init__(
        self,
        pipeline: list[StageFn],
        stop_min_results: int = RETRIEVAL_STOP_MIN_RESULTS,
        cache_ttl: float = RETRIEVAL_CACHE_TTL,
        cache_size: int = RETRIEVAL_CACHE_SIZE,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.pipeline = pipeline
        self.stop_min_results = stop_min_results
        self.cache_ttl = cache_ttl
        self.cache_size = cache_size
        self._clock = clock
        self._cache: OrderedDict[tuple[str, int], tuple[float, list[RetrievalResult]]] = OrderedDict()
        logger.debug("RetrievalEngine initialized with %d stage(s): %s", len(pipeline), [fn.__name__ for fn in pipeline])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tried_stages(self, query: str) -> set[int]:
        """Stages whose result for this query is still in the cache (i.e. were run recently)."""
        if self.cache_ttl <= 0:
            return set()
        normalized = self._key(query, 0)[0]
        now = self._clock()
        return {
            stage for (q, stage), (stored_at, _) in self._cache.items()
            if q == normalized and now - stored_at <= self.cache_ttl
        }

    def resolve_start_stage(self, query: str, requested: int) -> int:
        """Honor a requested higher stage only if every cheaper stage was already tried for this query.

        LLMs tend to ask for the most powerful stage every time; this keeps the cheap local sources
        first while still allowing escalation ("search again at a higher stage") on a retry.
        """
        requested = max(0, requested)
        if requested == 0:
            return 0
        tried = self.tried_stages(query)
        if all(stage in tried for stage in range(requested)):
            return requested
        logger.info("Requested stage %d ignored: lower stages not yet tried for %r; starting at 0", requested, query)
        return 0

    def run_pipeline(self, query: str, start_stage: int = 0) -> tuple[list[RetrievalResult], int, int]:
        """Run stages from `start_stage` until enough results are found or the pipeline ends.

        Returns (results, last_stage_run, next_stage). `next_stage` is -1 when no stage remains,
        so a caller can escalate later by calling again with `start_stage=next_stage`.
        """
        stage = max(0, start_stage)
        if stage >= len(self.pipeline):
            logger.warning("run_pipeline start_stage=%d beyond pipeline of %d stage(s)", stage, len(self.pipeline))
            return [], stage, -1

        started = self._clock()
        all_results: list[RetrievalResult] = []
        last_stage = stage
        next_stage = -1
        while True:
            results, next_stage = self.run_stage(query, stage)
            last_stage = stage
            all_results.extend(results)
            if self._enough_information(all_results):
                logger.info("Enough information after stage %d (%d result(s))", stage, len(all_results))
                break
            if next_stage == -1:
                logger.info("Reached end of retrieval pipeline with %d result(s)", len(all_results))
                break
            stage = next_stage

        logger.info(
            "run_pipeline | query=%r stages %d..%d | %d result(s) in %.2fs",
            query, max(0, start_stage), last_stage, len(all_results), self._clock() - started,
        )
        return all_results, last_stage, next_stage

    def run_stage(self, query: str, stage: int) -> tuple[list[RetrievalResult], int]:
        """Execute exactly one stage. Returns (results, next_stage); next_stage is -1 after the last stage."""
        if stage < 0 or stage >= len(self.pipeline):
            logger.warning("run_stage called with stage=%d but pipeline has %d stage(s)", stage, len(self.pipeline))
            return [], -1

        next_stage = stage + 1 if stage + 1 < len(self.pipeline) else -1
        cached = self._cache_get(query, stage)
        if cached is not None:
            logger.info("Run Stage | stage: %d cache hit | query: %r | %d result(s)", stage, query, len(cached))
            return cached, next_stage

        search = self.pipeline[stage]
        logger.info("Run Stage | stage: %d (%s) | query: %r", stage, search.__name__, query)
        started = self._clock()
        results = search(query)
        logger.info("Stage %d (%s) returned %d result(s) in %.2fs", stage, search.__name__, len(results), self._clock() - started)

        self._cache_put(query, stage, results)
        return results, next_stage

    def clear_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _enough_information(self, results: list[RetrievalResult]) -> bool:
        """Stop once at least `stop_min_results` *decisive* results have been gathered.

        A result with metadata["decisive"] False (a partial-name hit in the curated index) is kept
        as context but does not end the search, so other meanings can still be found.
        """
        decisive = sum(1 for r in results if r.metadata.get("decisive", True))
        return decisive >= self.stop_min_results

    @staticmethod
    def _key(query: str, stage: int) -> tuple[str, int]:
        return (" ".join(query.casefold().split()), stage)

    def _cache_get(self, query: str, stage: int) -> list[RetrievalResult] | None:
        if self.cache_ttl <= 0:
            return None
        key = self._key(query, stage)
        entry = self._cache.get(key)
        if entry is None:
            return None
        stored_at, results = entry
        if self._clock() - stored_at > self.cache_ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return list(results)

    def _cache_put(self, query: str, stage: int, results: list[RetrievalResult]) -> None:
        if self.cache_ttl <= 0:
            return
        self._cache[self._key(query, stage)] = (self._clock(), list(results))
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
