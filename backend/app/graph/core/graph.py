from __future__ import annotations

import logging
import time
from typing import Any, Hashable

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from .state import State
from .nodes import chatbot, detect_language, tools as llm_tools
from ..judge.node import judge_response
from ..retrieval.nodes import retrieve_information_node, finish_retrieval_node
from .router import route_chatbot, retrieval_router

from ...config import USE_RETRIEVAL_PIPELINE_TOOL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        if parts:
            return "\n".join(parts)
    return str(content)

# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

# CV Analysis Subgraph
#cv_builder = StateGraph(CVAnalysisState)


# ---------------------------------------------------------------------------
# Main Graph Construction
# ---------------------------------------------------------------------------

# Use USE_RETRIEVAL_PIPELINE_TOOL to switch between using the retrieval pipeline tool or not. If set to False, the retrieve_information_node will be used instead. 
memory = MemorySaver()
builder = StateGraph(State)
tool_node = ToolNode(llm_tools)  # same list bound to the LLM, so every advertised tool is executable
# ---------------------------------------------------------------------------
# Nodes ---------------------------------------------------------------------
# ---------------------------------------------------------------------------
builder.add_node("detect_language", detect_language)
builder.add_node("chatbot", chatbot)
builder.add_node("judge", judge_response)
builder.add_node("tools", tool_node)
#builder.add_node("cv_analysis", cv_analysis_node)
if not USE_RETRIEVAL_PIPELINE_TOOL:
    builder.add_node("retrieve_information", retrieve_information_node)
    builder.add_node("finish_retrieval", finish_retrieval_node)
# ---------------------------------------------------------------------------
# Edges ---------------------------------------------------------------------
# ---------------------------------------------------------------------------
builder.add_edge(START, "detect_language")
builder.add_edge("detect_language", "chatbot")
builder.add_edge("tools", "chatbot")
builder.add_edge("judge", END)
# ---------------------------------------------------------------------------
if USE_RETRIEVAL_PIPELINE_TOOL:
    builder.add_conditional_edges("chatbot", route_chatbot, {"tools": "tools", "judge": "judge"})
else:
    routes: dict[Hashable, str] = {"tools": "tools", "retrieve_information": "retrieve_information", "judge": "judge"}
    #builder.add_conditional_edges("judge", tool_node, {"tools": "tools", "retrieve_information": "retrieve_information"})
    builder.add_conditional_edges("chatbot", route_chatbot, routes)
    builder.add_conditional_edges("retrieve_information", retrieval_router, {"retrieve_information": "retrieve_information", "finish_retrieval": "finish_retrieval"})
    builder.add_edge("finish_retrieval", "chatbot")
# ---------------------------------------------------------------------------

graph = builder.compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# Visual Test - generates a PNG of the graph and opens it in the default image viewer.
# ---------------------------------------------------------------------------

from pathlib import Path
import webbrowser

png = graph.get_graph().draw_mermaid_png()

path = Path("graph.png")
path.write_bytes(png)

#webbrowser.open(path.resolve().as_uri())

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_agent(user_message: str, thread_id: str = "default") -> str:
    """Run one conversation turn through the graph.
    Args:
        user_message: The message from the user to process.
        thread_id: A unique identifier for the conversation thread. Defaults to "default".
    Returns:
        The response from the agent / LLM as a string for the user.
    """
    logger.info("Run Agent | thread=%s | message=%r", thread_id, user_message[:120])

    config = RunnableConfig(configurable={"thread_id": thread_id})
    inputs: State = {"messages": [HumanMessage(content=user_message)]}
    start = time.perf_counter()

    try:
        for event in graph.stream(inputs, config=config, stream_mode="updates"):
            for node, update in event.items():
                #print(f"Node: {node}, Update: {update}")
                logger.info("Finished executed node: %s", node)
                if update is not None:
                    if "messages" in update:
                        logger.info("Produced %d message(s)", len(update["messages"]))
                    if "pending_pipeline" in update:
                        logger.info("Pending pipeline: %s", update["pending_pipeline"])
                else:
                    logger.info("Unavailable update for node: %s", node)
    except Exception:
        logger.exception("Graph execution failed")
        raise

    logger.info("Finished in %.2fs", time.perf_counter() - start)
    state = graph.get_state(config).values
    #logger.info("Agent full RETRIEVAL response: %s", state.get("retrieval", []))
    messages = state.get("messages", [])
    logger.debug("Final pipeline | pending=%s completed=%s", state.get("pending_pipeline"), state.get("completed_pipeline"))

    for message in reversed(messages):
        if getattr(message, "type", "") == "ai":
            text = _extract_text(message.content)
            if not text or not text.strip():
                logger.warning("Last AI message has empty content.")
                return "I could not produce a response - empty message -."
            return text

    logger.warning("No AI message found in the conversation.")
    return "I could not produce a response - no AI message -."