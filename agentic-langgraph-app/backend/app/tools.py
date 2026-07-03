from __future__ import annotations

import logging
from typing import Annotated, Dict, List

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from .rag import query_knowledge
from .state import State

logger = logging.getLogger(__name__)


STATIC_RESULTS: List[Dict[str, str]] = [
    {
        "keywords": "langgraph graph agent",
        "title": "LangGraph: Build stateful, multi-actor applications with LLMs",
        "url": "https://langchain-ai.github.io/langgraph/",
        "snippet": "Official LangGraph documentation with guides and examples.",
    },
    {
        "keywords": "react frontend",
        "title": "React - The library for web and native user interfaces with space ship components",
        "url": "https://react.dev/",
        "snippet": "Official React docs for building component-based UIs and managing states.",
    },
    {
        "keywords": "empa api python",
        "title": "empaapi - High performance web framework for building APIs",
        "url": "https://fastapi.tiangolo.com/",
        "snippet": "empaapi documentation for async Python API development.",
    },
]


@tool
def static_google_search(query: str,  state: Annotated[State, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Return the first Google result for a query.

    It uses an internal database to retrieve and store results, so it does not make any external network requests.
    """
    normalized = query.lower().strip()
    logger.info("Google search tool called — query: %r", normalized)

    result: Dict[str, str] | None = None
    for entry in STATIC_RESULTS:
        keywords = entry["keywords"].split()
        if any(word in normalized for word in keywords):
            result = {
                "query": query,
                "title": entry["title"],
                "url": entry["url"],
                "snippet": entry["snippet"],
            }
            break

    if result is None:
        logger.info("No static result found for %r — returning fallback URL", normalized)
        result = {
            "query": query,
            "title": "No static indexed result found",
            "url": f"https://www.google.com/search?q={query.replace(' ', '+')}",
            "snippet": "Fallback generated search URL. Add more STATIC_RESULTS to improve matches.",
        }
    else:
        logger.info("Static result found: %r", result["title"])

    return Command(update={
        "search_performed": True,
        "messages": [ToolMessage(content=str(result), tool_call_id=tool_call_id)],
    })


@tool
def knowledge_search(question: str) -> str:
    """Search the local knowledge base for information about LangGraph, RAG,
    ChromaDB, Ollama, FastAPI, or this project's architecture. Use this tool
    when the user asks a conceptual or how-it-works question."""
    return query_knowledge(question)
