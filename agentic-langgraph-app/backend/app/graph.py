from __future__ import annotations

import logging
import time
import pprint
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from .state import State
from .nodes import chatbot
from .tools import static_google_search, retrieve_information

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant with access to a local knowledge base and a search tool.

Rules you must always follow:
1. For EVERY user question, call knowledge_search first to check the local knowledge base before answering.
2. Only call static_google_search if knowledge_search returned no useful result. ALWAYS.
3. Never answer from memory alone when a tool could provide a better answer.
4. Extract all proper nouns and identifiers exactly as written — do not modify them.
5. Add "nya" before any comma or period, like an anime character.
"""

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



def route_after_search(state):
    if state["search_performed"]:
        return "chatbot"
    if state.get("search_failed"):
        return "error_handler"

    return END

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

memory = MemorySaver()
builder = StateGraph(State)
tool_node = ToolNode([static_google_search])

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

def run_agent(
    user_message: str,
    thread_id: str = "default",
) -> str:
    """Run one conversation turn through the graph."""

    logger.info(
        "run_agent | thread=%s | message=%r",
        thread_id,
        user_message[:120],
    )

    config = RunnableConfig(
        configurable={
            "thread_id": thread_id,
        }
    )

    inputs = State(messages=[{"role": "user", "content": user_message}])
    start = time.perf_counter()
    last_state = None
    try:
        for event in graph.stream(
            inputs,
            config=config,
            stream_mode="updates",
        ):
            for node, update in event.items():
                logger.info("Executed node: %s", node)
                if "messages" in update:
                    logger.info("Produced %d message(s)", len(update["messages"]))
                    #logger.info("Last message: %r", update["messages"][-1])
                if "pending_pipeline" in update:
                    logger.info("Pending pipeline: %s", update["pending_pipeline"])

    except Exception:
        logger.exception("Graph execution failed")
        raise
    logger.info("Finished in %.2fs", time.perf_counter() - start)

    #
    # Read the final graph state.
    #
    state = graph.get_state(config).values
    messages = state.get("messages", [])
    logger.debug(
        "Final pipeline | pending=%s completed=%s",
        state.get("pending_pipeline"),
        state.get("completed_pipeline"),
    )

    #
    # Return the final AI message.
    #
    for message in reversed(messages):
        if getattr(message, "type", "") == "ai":
            return _extract_text(message.content)

    return "I could not produce a response."