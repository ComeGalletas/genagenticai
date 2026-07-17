
from __future__ import annotations

import logging

from typing import Dict
from langgraph.types import Command

from .data import STATIC_RESULTS
from ..retrieval.schemas import RetrievalResult, RetrievalStage

logger = logging.getLogger(__name__)


def query_static_search(query: str) -> list[RetrievalResult]:
    """Return a fixed set of results for certain queries and returns a structured list, without making external network requests."""
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
                score=100,
                confidence=None,
            )
        )
    
    """
    if not results:
        logger.info("No static entry found for %r — Search results were insufficient to answer the question.", normalized)
        return [RetrievalResult(
                title="No relevant information found",
                content="No relevant information was found to answer the user's question.",
                source="",
                stage=RetrievalStage.STATIC,
                score=None,
                confidence=None,
            )]
    else:
        logger.info("Static entry found for %r: %r", normalized, results[0]["title"])

    """
    
    return retrieval_results

