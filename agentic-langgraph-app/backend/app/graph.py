from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from .state import State
from .nodes import chatbot, rag_search
from .searches.static_search import static_google_search
from .tools import retrieve_information

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
# Graph
# ---------------------------------------------------------------------------

memory = MemorySaver()
builder = StateGraph(State)
tool_node = ToolNode([static_google_search, retrieve_information])

builder.add_node("chatbot", chatbot)
builder.add_node("tools", tool_node)

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

graph = builder.compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# Visual Test - UNCOMMENT the entire block to generate a PNG of the graph and open it in the default image viewer.
# ---------------------------------------------------------------------------
"""
from pathlib import Path
import webbrowser

png = graph.get_graph().draw_mermaid_png()

path = Path("graph.png")
path.write_bytes(png)

webbrowser.open(path.resolve().as_uri())
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_agent(user_message: str, thread_id: str = "default") -> str:
    """Run one conversation turn through the graph."""
    logger.info("run_agent | thread=%s | message=%r", thread_id, user_message[:120])

    config = RunnableConfig(configurable={"thread_id": thread_id})
    inputs = State(messages=[{"role": "user", "content": user_message}])
    start = time.perf_counter()

    try:
        for event in graph.stream(inputs, config=config, stream_mode="updates"):
            for node, update in event.items():
                logger.info("Executed node: %s", node)
                if "messages" in update:
                    logger.info("Produced %d message(s)", len(update["messages"]))
                if "pending_pipeline" in update:
                    logger.info("Pending pipeline: %s", update["pending_pipeline"])
    except Exception:
        logger.exception("Graph execution failed")
        raise

    logger.info("Finished in %.2fs", time.perf_counter() - start)

    state = graph.get_state(config).values
    messages = state.get("messages", [])
    logger.debug("Final pipeline | pending=%s completed=%s", state.get("pending_pipeline"), state.get("completed_pipeline"))

    for message in reversed(messages):
        if getattr(message, "type", "") == "ai":
            text = _extract_text(message.content)
            if not text or not text.strip():
                logger.warning("Last AI message has empty content.")
                #logger.info("full %s text(s)", message)
                return "I could not produce a response."
            return text

    return "I could not produce a response."