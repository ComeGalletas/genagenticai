from __future__ import annotations

import logging
import json
from dataclasses import asdict
from typing import Annotated
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command

from ..retrieval.retrieval import RetrievalResult
from ..retrieval.retrieval_config import RETRIEVAL_PIPELINE
from ..retrieval.retrieval_engine import RetrievalEngine

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
    """Retrieves information from a knowledge base and web search functions. It includes a knowledge base and web search functions.
        It is a very very slow tool that may take a few seconds to return results, but it is very comprehensive and it is updated.
        Use it if you need to know the answer to a question that requires external knowledge or recent information.
    """
    logger.info("retrieve_information TOOL called")
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

@tool
def get_current_time() -> str:
    """Return the current local date, time, and timezone of the machine running the application.
        Use it if you need to know the current date and time for any reason or queries.
    """

    now = datetime.now().astimezone()

    return (
        f"Date: {now:%Y-%m-%d}\n"
        f"Time: {now:%H:%M:%S}\n"
        f"Timezone: {now.tzname()}"
    )