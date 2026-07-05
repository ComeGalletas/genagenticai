from __future__ import annotations

import logging
import json
from dataclasses import asdict
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command

from .retrieval.retrieval import RetrievalResult
from .retrieval.retrieval_config import RETRIEVAL_PIPELINE
from .retrieval.retrieval_engine import RetrievalEngine

logger = logging.getLogger(__name__)

retrieval_engine = RetrievalEngine(pipeline=RETRIEVAL_PIPELINE)


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


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def retrieve_information(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Retrieve information from the retrieval pipeline and return structured results."""
    results = retrieval_engine.run_pipeline(query)
    formatted = format_results(
        query=query,
        results=results,
    )

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=formatted,
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )
