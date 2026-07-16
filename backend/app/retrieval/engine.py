
import logging
from typing import Callable

from .schemas import RetrievalResult

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Executes one or more retrieval strategies. Knows nothing about LangGraph, ToolMessages, or graph state."""

    def __init__(self, pipeline: list[Callable[[str], list[RetrievalResult]]]):
        self.pipeline = pipeline
        logger.debug("RetrievalEngine initialized with %d stage(s): %s", len(pipeline), [fn.__name__ for fn in pipeline])

    def run_pipeline(self, query: str) -> list[RetrievalResult]:
        """Run the retrieval pipeline until enough information is found or all sources have been searched."""
        logger.info("run_pipeline | query: %r | stages: %d", query, len(self.pipeline))

        all_results: list[RetrievalResult] = []
        stage = 0

        while True:
            results, next_stage = self.run_stage(query, stage)
            if results:
                all_results.extend(results)

            if self._enough_information(all_results):
                logger.info("Enough information found after stage %d", stage)
                break

            if next_stage == -1:
                logger.info("Reached end of retrieval pipeline.")
                break

            stage = next_stage

        logger.info("run_pipeline complete | total results: %d", len(all_results))
        return all_results

    def run_stage(self, query: str, stage: int) -> tuple[list[RetrievalResult], int]:
        """Execute exactly one retrieval stage. Returns (results, next_stage); next_stage is -1 when the pipeline has finished."""
        if stage >= len(self.pipeline):
            logger.warning("run_stage called with stage=%d but pipeline has only %d stage(s)", stage, len(self.pipeline))
            return [], -1

        search = self.pipeline[stage]
        logger.info("run_stage | stage: %d (%s) | query: %r", stage, search.__name__, query)

        results = search(query)
        logger.debug("Stage %d (%s) returned %d result(s)", stage, search.__name__, len(results))

        next_stage = stage + 1 if stage + 1 < len(self.pipeline) else -1
        return results, next_stage

    def _enough_information(self, results: list[RetrievalResult]) -> bool:
        """Decide whether retrieval should stop. Replace this with whatever heuristic you like."""
        return len(results) >= 4