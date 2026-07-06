from __future__ import annotations
import logging
from ddgs import DDGS
from ..retrieval.retrieval import RetrievalResult, RetrievalStage

logger = logging.getLogger(__name__)
logging.getLogger("primp").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def google_to_results(google_results) -> list[RetrievalResult]:
    retrieval_results = []
    for result in google_results:
        retrieval_results.append(
            RetrievalResult(
                title=result["title"],
                content=result["snippet"],
                source=result["url"],
                stage=RetrievalStage.GOOGLE,
            )
        )
    return retrieval_results


def query_ddu_google_search(query: str, max_results: int = 5) -> list[RetrievalResult]:
    """
    Search DuckDuckGo and return structured retrieval results.
    """
    logger.debug("Google query: %r", query)
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        logger.exception("DuckDuckGo search failed")
        return []
    
    retrieval_results: list[RetrievalResult] = []
    for result in results:
        retrieval_results.append(
            RetrievalResult(
                title=result.get("title", "Untitled"),
                content=result.get("body", ""),
                source=result.get("href", ""),
                stage=RetrievalStage.GOOGLE,
                score=None,
                confidence=None,
                metadata={"engine": "DuckDuckGo"},
            )
        )
    logger.info("Returning %d Google result(s) for query %r", len(retrieval_results), query) 
    return retrieval_results
