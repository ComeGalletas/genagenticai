
from __future__ import annotations

import logging

from typing import Dict
from langgraph.types import Command

from ..retrieval.schemas import RetrievalResult, RetrievalStage
from .static_data import STATIC_RESULTS

logger = logging.getLogger(__name__)


def query_static_search(query: str) -> list[RetrievalResult]:
    """Return a fixed set of results for certain queries and returns a structured list, without making external network requests.
    Args:
        query (str): The search query string.
    Returns:
        list[RetrievalResult]: A list of RetrievalResult objects containing the search results.
    """
    normalized = query.lower().strip() if query else ""
    logger.info("Google static search function called — query: %r", normalized)
    
    results: list[Dict[str, str]] = []    
    for entry in STATIC_RESULTS:
        keywords = entry["keywords"].split()
        if any(word.lower() in normalized for word in keywords):
            results.append({
                "query": str(query),
                "title": entry["title"],
                "url": entry["url"],
                "snippet": entry["snippet"],
            })

    retrieval_results: list[RetrievalResult] = []
    for result in results:
        retrieval_results.append(
            RetrievalResult(
                title=result.get("title", "Untitled"),
                content=result.get("snippet", ""),
                source=result.get("url", ""),
                stage=RetrievalStage.STATIC,
                confidence=0.95,  # Static results are considered highly relevant
            )
        )
    
    if not results:
        logger.info("No static entry found for %r — Search results were insufficient to answer the question.", normalized)
    else:
        logger.info("Static entry found for %r: %r", normalized, results[0]["title"])
    
    return retrieval_results
