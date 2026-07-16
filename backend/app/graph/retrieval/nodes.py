import json
import logging
from dataclasses import asdict

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from ..core.state import State
from ...retrieval.service import retrieval_engine
from ...retrieval.schemas import RetrievalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_results(query: str, results: list[RetrievalResult], stage: int = 0) -> str:
    """Format the retrieval results into a structured JSON string."""
    if results:
        status = "FOUND"
        stage = int(results[0].stage) if stage == -1 else stage
    else:
        status = "NO_MATCH"

    payload = {
        "stage": stage,
        "query": query,
        "status": status,
        "document_count": len(results),
        "documents": [asdict(r) for r in results]
    }   
    
    return json.dumps(payload, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def retrieve_information_node(state: State):
    """ Retrieves information from a knowledge base and web search functions. It includes a knowledge base and web search functions.
        It is a very very slow tool that may take a few seconds to return results, but it is very comprehensive and it is updated.
        Use it if you need to know the answer to a question that requires external knowledge or recent information.
    """
    logger.info("Retrieve_information node called")
    # First time entering the node
    if state.get("retrieval_query") is None:
        ai_message = state["messages"][-1]
        tool_call = ai_message.tool_calls[0]
        query = tool_call["args"]["query"]
        tool_call_id = tool_call["id"]

        stage = 0
        accumulated_results = []
    # Returning from a previous stage
    else:
        query = state.get("retrieval_query")
        stage = state.get("retrieval_stage")
        accumulated_results = state.get("retrieval_results")
        tool_call_id = state.get("tool_call_id")

    if stage is None:
        raise RuntimeError("retrieve_information_node called after retrieval completed")
    stage_results, next_stage = retrieval_engine.run_stage(query=str(query), stage=stage)
    if accumulated_results is not None:
        accumulated_results.extend(stage_results)

    return Command(
        update={
            "retrieval_query": query,
            "retrieval_stage": next_stage,
            "retrieval_results": accumulated_results,
            "tool_call_id": tool_call_id,
        }
    )


def finish_retrieval_node(state: State):
    """Finish the retrieval process and format the results then go back to the chatbot node."""
    formatted = format_results(
        query=str(state.get("retrieval_query")),
        results=state.get("retrieval_results", []),
    )
    return Command(update={"messages": [ToolMessage(content=formatted, tool_call_id=state.get("tool_call_id"))
            ],
            "retrieval_query": None,
            "retrieval_stage": None,
            "retrieval_results": [],
            "tool_call_id": None,
        }
    )
