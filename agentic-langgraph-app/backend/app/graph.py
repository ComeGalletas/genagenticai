from __future__ import annotations

import logging
import time
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from .state import State
from .tools import static_google_search, knowledge_search

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Before answering:
Extract all proper nouns and identifiers exactly as written.
Do not modify them.
Then answer. If you don't know the answer, use the knowledge base and if it was already done use the google search tool internally, use the  do not assume anything if you don't have at least 80%/ certainty.

Add a "nya" before any comma or period, similar to an anime character.
"""

# ---------------------------------------------------------------------------
# LLM + tools
# ---------------------------------------------------------------------------

tools = [static_google_search, knowledge_search]

_llm = ChatOllama(
    model="qwen3",
    temperature=0,
)
llm_with_tools = _llm.bind_tools(tools)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def chatbot(state: State) -> dict:
    """Core node: send the current message list to the LLM and return its reply."""
    search_done = state.get("search_performed", False)
    logger.debug(
        "Chatbot node — %d message(s) in state, search_performed=%s",
        len(state["messages"]),
        search_done,
    )
    response = llm_with_tools.invoke(state["messages"])
    tool_calls = getattr(response, "tool_calls", [])
    if tool_calls:
        logger.info("LLM requested tool call(s): %s", [tc["name"] for tc in tool_calls])
    else:
        logger.debug("LLM produced a direct response (no tool calls)")
    return {"messages": [SystemMessage(content=SYSTEM_PROMPT), response]}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

memory = MemorySaver()

builder = StateGraph(State)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

graph = builder.compile(checkpointer=memory)


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
# Public API
# ---------------------------------------------------------------------------

def run_agent(user_message: str, thread_id: str = "default") -> str:
    """Run the graph for one user turn.  thread_id scopes the MemorySaver
    so each browser session keeps its own conversation history."""
    logger.info("run_agent — thread_id=%r message=%r", thread_id, user_message[:120])
    config = RunnableConfig(configurable={"thread_id": thread_id})
    start = time.perf_counter()
    try:
        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config,
            options={
                "num_predict": 100
            }
        )
    except Exception as exc:
        logger.error("Graph invocation failed — thread_id=%r: %s", thread_id, exc)
        raise
    elapsed = time.perf_counter() - start
    messages = result.get("messages", [])
    search_done = result.get("search_performed", False)
    logger.info(
        "Graph finished in %.2fs — %d message(s), search_performed=%s",
        elapsed,
        len(messages),
        search_done,
    )
    for message in reversed(messages):
        if getattr(message, "type", "") == "ai":
            return _extract_text(getattr(message, "content", ""))
    logger.warning("No AI message found in final state — thread_id=%r", thread_id)
    return "I could not produce a response."
