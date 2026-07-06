
from __future__ import annotations
import logging
from typing import List, Dict, Annotated
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from ..retrieval.retrieval import RetrievalResult, RetrievalStage

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
# Functions
# ---------------------------------------------------------------------------


def query_static_search(query: str) -> list[RetrievalResult]:
    normalized = query.lower().strip() if query else ""
    logger.debug("Google static search tool called — query: %r", normalized)
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
                score=None,
                confidence=None,
            )
        )
    
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

    return retrieval_results

@tool
def static_google_search(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """ Returns a fixed set of results for certain queries, without making external network requests. 
        It is fast to access but it has very few data and it is not updated. 
        Use it if you need to know the answer to a question that requires external knowledge or recent information.
    """
    results = query_static_search(str(query))

    return Command(update={"messages": [ToolMessage(content=str(results), tool_call_id=tool_call_id)]})