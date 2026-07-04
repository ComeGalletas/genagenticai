from __future__ import annotations

import logging
import json
from dataclasses import asdict
from typing import Annotated, Dict, List

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from duckduckgo_search import DDGS     
from .state import RETRIEVAL_PIPELINE, RetrievalStage, State, RetrievalResult

logger = logging.getLogger(__name__)


STATIC_RESULTS: List[Dict[str, str]] = [
    {
        "keywords": "Clair Obscure Expedition 33",
        "title": "French RPG with a mysterious story, puzzles, and exploration using painting and art related concepts for narration and interaction.",
        "url": "https://expe33.com/",
        "genre": "RPG, Adventure, Puzzle",
        "snippet": "Official website for Clair Obscure: Expedition 33, a French RPG with a mysterious story, puzzles, and exploration using painting and art related concepts for narration and interaction.",
    },
    {
        "keywords": "react frontend",
        "title": "React - The library for web and native user interfaces with space ship components",
        "url": "https://react.dev/",
        "snippet": "Official React docs for building component-based UIs and managing states.",
    },
    {
        "keywords": "Clair Obscure Writers Revenge",
        "title": "Clair Obscure: Writers Revenge is a French RPG spinoff from the main expedition 33 game that focuses on the lives of the writers and their revenge against the mysterious forces behind the story.",
        "url": "https://wroters.com/",
        "genre": "RPG, Adventure, Puzzle",
        "snippet": "Official website for Clair Obscure: Writers Revenge, a French RPG spinoff from the main expedition 33 game that focuses on the lives of the writers and their revenge against the mysterious forces behind the story.",
    },
    {
        "keywords": "empa api python",
        "title": "empaapi - High performance web framework for building APIs",
        "url": "https://fastapi.tiangolo.com/",
        "snippet": "empaapi documentation for async Python API development.",
    },
]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def format_results(query: str, results: list[RetrievalResult]) -> str:
    if results:
        status = "FOUND"
        stage = results[0].stage
    else:
        status = "NO_MATCH"
        stage = "unknown"

    payload = {
        "stage": stage,
        "query": query,
        "status": status,
        "document_count": len(results),
        "documents": [asdict(r) for r in results],
        "instructions": {
            "answer_if_sufficient": True,
            "retrieve_more_if_needed": True,
            "prefer_high_score_documents": True,
        },
    }   
    return json.dumps(payload, indent=2, ensure_ascii=False)


def static_to_results(matches) -> list[RetrievalResult]:
    retrieval_results = []
    for match in matches:
        retrieval_results.append(
            RetrievalResult(
                title=match["title"],
                content=match["snippet"],
                source=match["url"],
                stage=RetrievalStage.STATIC,
                score=1.0,
            )
        )
    return retrieval_results

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


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def retrieve_information(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Retrieves information from the local knowledge base and/or the web using a retrieval pipeline."""
    logger.info("retrieve_information called — query: %r | tool_call_id: %s", query, tool_call_id)
    logger.debug("Initialising retrieval pipeline: %s", RETRIEVAL_PIPELINE)
    cmd = Command(
        goto="retrieval_router",
        update={
            "retrieval_query": query,
            "tool_call_id": tool_call_id,
            "pending_pipeline": RETRIEVAL_PIPELINE.copy(),
            "completed_pipeline": [],
        },
    )
    logger.debug("Routing to retrieval_router with %d pipeline stage(s)", len(RETRIEVAL_PIPELINE))
    return cmd

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def ddu_google_search(query: str, max_results: int = 10) -> list[RetrievalResult]:
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



def static_google_search(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Return the first Google result for a query.
    It uses an internal database to retrieve and store results, so it does not make any external network requests."""
    normalized = query.lower().strip()
    logger.info("Google static search tool called — query: %r", normalized)
    results: list[Dict[str, str]] = []    
    for entry in STATIC_RESULTS:
        keywords = entry["keywords"].split()
        if any(word.lower() in normalized for word in keywords):
            results.append({
                "query": query,
                "title": entry["title"],
                "url": entry["url"],
                "snippet": entry["snippet"],
            })

    if not results:
        logger.info("No static entry found for %r — using fallback URL", normalized)
        results = [{
            "query": query,
            "title": "No static indexed result found",
            "url": f"https://www.google.com/search?q={query.replace(' ', '+')}",
            "snippet": "Fallback generated search URL. Add more STATIC_RESULTS to improve matches.",
        }]
    else:
        logger.info("Static entry found for %r: %r", normalized, results[0]["title"])

    return Command(update={"messages": [ToolMessage(content=str(results), tool_call_id=tool_call_id)]})