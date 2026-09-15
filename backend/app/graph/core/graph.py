from __future__ import annotations

import logging
import time
from typing import Any, Hashable, Iterable, Iterator

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from .state import State
from .nodes import chatbot, finalize, prepare_turn, tools as llm_tools
from .formatting import render_reply
from ..judge.node import judge_response, revise
from ..retrieval.nodes import retrieve_information_node, finish_retrieval_node
from .router import route_chatbot, route_judge, retrieval_router

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
# Main Graph Construction
# ---------------------------------------------------------------------------

# Use USE_RETRIEVAL_PIPELINE_TOOL to switch between using the retrieval pipeline tool or not. If set to False, the retrieve_information_node will be used instead. 
memory = MemorySaver()
builder = StateGraph(State)
tool_node = ToolNode(llm_tools)  # same list bound to the LLM, so every advertised tool is executable
# ---------------------------------------------------------------------------
# Nodes ---------------------------------------------------------------------
# ---------------------------------------------------------------------------
builder.add_node("prepare_turn", prepare_turn)
builder.add_node("chatbot", chatbot)
builder.add_node("judge", judge_response)
builder.add_node("revise", revise)
builder.add_node("finalize", finalize)
builder.add_node("tools", tool_node)
if not USE_RETRIEVAL_PIPELINE_TOOL:
    builder.add_node("retrieve_information", retrieve_information_node)
    builder.add_node("finish_retrieval", finish_retrieval_node)
# ---------------------------------------------------------------------------
# Edges ---------------------------------------------------------------------
# ---------------------------------------------------------------------------
builder.add_edge(START, "prepare_turn")
builder.add_edge("prepare_turn", "chatbot")
builder.add_edge("tools", "chatbot")
builder.add_conditional_edges("judge", route_judge, {"finalize": "finalize", "revise": "revise"})
builder.add_edge("revise", "chatbot")  # critique loop: the chatbot rewrites, and may call tools again first
builder.add_edge("finalize", END)
# ---------------------------------------------------------------------------
chatbot_routes: dict[Hashable, str] = {"tools": "tools", "judge": "judge", "finalize": "finalize"}
if USE_RETRIEVAL_PIPELINE_TOOL:
    builder.add_conditional_edges("chatbot", route_chatbot, chatbot_routes)
else:
    builder.add_conditional_edges("chatbot", route_chatbot, {**chatbot_routes, "retrieve_information": "retrieve_information"})
    builder.add_conditional_edges("retrieve_information", retrieval_router, {"retrieve_information": "retrieve_information", "finish_retrieval": "finish_retrieval"})
    builder.add_edge("finish_retrieval", "chatbot")
# ---------------------------------------------------------------------------

graph = builder.compile(checkpointer=memory)


def save_graph_png(path: str = "graph.png") -> str:
    """Render the compiled graph to a PNG (used by `python -m app.main graph`)."""
    from pathlib import Path

    target = Path(path)
    target.write_bytes(graph.get_graph().draw_mermaid_png())
    return str(target.resolve())


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
    return _reply_from_state(graph.get_state(config).values)


def _reply_from_state(state: dict) -> str:
    """The HTML reply for the client: finalize's output, or the last AI message rendered directly."""
    final_response = state.get("final_response")
    if final_response:
        return final_response

    for message in reversed(state.get("messages", [])):
        if getattr(message, "type", "") == "ai":
            text = _extract_text(message.content)
            if not text or not text.strip():
                logger.warning("Last AI message has empty content.")
                return "I could not produce a response - empty message -."
            return render_reply(text) or text

    logger.warning("No AI message found in the conversation.")
    return "I could not produce a response - no AI message -."


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

_TOOL_STATUS = {
    "retrieve_information": "Searching…",
    "retrieve_job_postings": "Searching job postings…",
    "retrieve_baloto_results": "Looking up Baloto results…",
    "suggest_baloto_numbers": "Crunching Baloto numbers…",
    "read_webpage": "Reading the page…",
    "get_current_time": "Checking the time…",
}


def translate_stream_events(raw: Iterable[tuple[str, Any]]) -> Iterator[dict]:
    """Turn LangGraph's combined ("messages" + "updates") stream into client events.

    Events: {"event": "status", "text"} progress hints; {"event": "delta", "text"} chatbot tokens;
    {"event": "reset"} discard the draft (the chatbot turn was a tool call, or the judge sent the
    answer back); {"event": "final", "html"} the sanitized reply from the finalize node.
    Judge tokens are never forwarded; only the chatbot node's text is.
    """
    for mode, data in raw:
        if mode == "messages":
            chunk, metadata = data
            if metadata.get("langgraph_node") != "chatbot":
                continue
            text = _extract_text(getattr(chunk, "content", ""))
            if text:
                yield {"event": "delta", "text": text}
            continue

        if mode != "updates":
            continue
        for node, update in data.items():
            update = update or {}
            if node == "prepare_turn":
                yield {"event": "status", "text": "Thinking…"}
            elif node == "chatbot":
                messages = update.get("messages") or []
                last = messages[-1] if messages else None
                tool_calls = getattr(last, "tool_calls", None) or []
                if tool_calls:
                    yield {"event": "reset"}
                    yield {"event": "status", "text": _TOOL_STATUS.get(tool_calls[0]["name"], "Working…")}
                else:
                    yield {"event": "status", "text": "Reviewing the answer…"}
            elif node == "revise":
                yield {"event": "reset"}
                yield {"event": "status", "text": "Improving the answer…"}
            elif node == "finalize":
                yield {"event": "final", "html": update.get("final_response") or ""}


def stream_agent_events(user_message: str, thread_id: str = "default") -> Iterator[dict]:
    """Run one turn and yield client events as the graph progresses (see translate_stream_events).

    Always ends with {"event": "final"} (computed from state if finalize produced nothing) and
    then {"event": "done"}, or {"event": "error"} if the graph failed.
    """
    logger.info("Stream Agent | thread=%s | message=%r", thread_id, user_message[:120])
    config = RunnableConfig(configurable={"thread_id": thread_id})
    inputs: State = {"messages": [HumanMessage(content=user_message)]}
    start = time.perf_counter()
    sent_final = False

    try:
        raw = graph.stream(inputs, config=config, stream_mode=["messages", "updates"])
        for event in translate_stream_events(raw):
            if event["event"] == "final":
                if not event["html"]:
                    continue
                sent_final = True
            yield event
        if not sent_final:
            yield {"event": "final", "html": _reply_from_state(graph.get_state(config).values)}
    except Exception as exc:
        logger.exception("Streamed graph execution failed")
        yield {"event": "error", "message": str(exc)}
        return

    logger.info("Stream finished in %.2fs", time.perf_counter() - start)
    yield {"event": "done"}