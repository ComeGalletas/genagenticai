
import logging
from typing import Callable

from .retrieval import RetrievalResult

logger = logging.getLogger(__name__)

class RetrievalEngine:
    """
    Executes one or more retrieval strategies.

    The engine knows nothing about LangGraph, ToolMessages,
    or graph state.
    """

    def __init__(self, pipeline: list[Callable[[str], list[RetrievalResult]]]):
        self.pipeline = pipeline
        logger.debug("RetrievalEngine initialised with %d stage(s): %s", len(pipeline), [fn.__name__ for fn in pipeline])


    def run_pipeline(self, query: str) -> list[RetrievalResult]:
        """
        Run the retrieval pipeline until enough information
        is found or all sources have been searched.
        """
        logger.info("run_pipeline | query: %r | stages: %d", query, len(self.pipeline))
        all_results: list[RetrievalResult] = []
        for search in self.pipeline:
            logger.debug("Running stage: %s", search.__name__)
            results = search(query)
            if not results:
                logger.debug("Stage %s returned no results", search.__name__)
                continue
            logger.debug("Stage %s returned %d result(s)", search.__name__, len(results))
            all_results.extend(results)
            if self._enough_information(results):
                logger.info("Enough information found after stage %s — stopping pipeline", search.__name__)
                break

        logger.info("run_pipeline complete | total results: %d", len(all_results))
        return all_results


    def run_stage(self, query: str, stage: int) -> tuple[list[RetrievalResult], int | None]:
        """
        Execute exactly one retrieval stage.
        Returns:
            (results, next_stage)
        next_stage is None when the pipeline has finished.
        """
        if stage >= len(self.pipeline):
            logger.warning("run_stage called with stage=%d but pipeline has only %d stage(s)", stage, len(self.pipeline))
            return [], None

        search = self.pipeline[stage]
        logger.info("run_stage | stage: %d (%s) | query: %r", stage, search.__name__, query)
        results = search(query)
        next_stage = stage + 1 if stage + 1 < len(self.pipeline) else None
        logger.debug("run_stage complete | results: %d | next_stage: %s", len(results), next_stage)
        return results, next_stage


    def _enough_information(self, results: list[RetrievalResult]) -> bool:
        """
        Decide whether retrieval should stop.
        Replace this with whatever heuristic you like.
        """
        return len(results) >= 3