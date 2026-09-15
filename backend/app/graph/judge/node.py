from __future__ import annotations

import logging
import time
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.types import Command

from ...config import JUDGE_KEEP_ALIVE, JUDGE_MAX_DOC_CHARS, JUDGE_MAX_RETRIEVALS, JUDGE_MODEL
from ..core.state import State
from .schema import JudgeVerdict
from .state import JudgeState
from .system_prompt import JUDGE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_MESSAGE = SystemMessage(content=JUDGE_SYSTEM_PROMPT)

# Small, non-thinking model: the judge only has to emit a short JSON verdict.
_judge_llm = ChatOllama(
    model=JUDGE_MODEL,
    temperature=0.0,
    reasoning=False,
    keep_alive=JUDGE_KEEP_ALIVE,
)
_judge = _judge_llm.with_structured_output(JudgeVerdict, method="json_schema")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [str(item["text"]) for item in content if isinstance(item, dict) and item.get("text")]
        if parts:
            return "\n".join(parts)
    return str(content)


def _last_message_text(state: State, message_type: str) -> str:
    message = next((m for m in reversed(state["messages"]) if getattr(m, "type", "") == message_type), None)
    return _extract_text(getattr(message, "content", "")) if message is not None else ""


def _format_context(state: State) -> str:
    """Render the most recent retrievals as plain text so the judge can check groundedness."""
    retrievals = state.get("retrieval") or []
    lines: list[str] = []
    for retrieval in retrievals[-JUDGE_MAX_RETRIEVALS:]:
        if retrieval.get("retrieval_status") != "FOUND":
            continue
        for doc in retrieval.get("retrieval_documents") or []:
            if doc is None:
                continue
            lines.append(f"[{doc.get('stage', '?')}] {doc.get('title', 'Untitled')} ({doc.get('source', 'unknown')})")
            lines.append(str(doc.get("content", ""))[:JUDGE_MAX_DOC_CHARS])
            lines.append("")
    return "\n".join(lines).strip() or "(no documents were retrieved for this turn)"


def _build_prompt(question: str, context: str, answer: str) -> HumanMessage:
    return HumanMessage(
        content=(
            "## USER QUESTION\n"
            f"{question}\n\n"
            "## RETRIEVED CONTEXT\n"
            f"{context}\n\n"
            "## CANDIDATE ANSWER\n"
            f"{answer}"
        )
    )


def _fallback_verdict(answer: str, error: str) -> JudgeState:
    """When the judge cannot run, let the answer through rather than blocking the user."""
    return JudgeState(
        original_response=answer,
        passed=True,
        grounded=True,
        needs_more_info=False,
        issues=[],
        error=error,
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def judge_response(state: State) -> Command:
    """Evaluate the chatbot's final answer and store a structured verdict. Never rewrites the answer."""
    logger.debug("Entering judge node.")

    answer = _last_message_text(state, "ai")
    if not answer.strip():
        logger.warning("Judge node skipped: chatbot response was empty.")
        return Command(update={})

    question = _last_message_text(state, "human")
    context = _format_context(state)

    start_time = time.perf_counter()
    try:
        verdict = cast(JudgeVerdict, _judge.invoke([JUDGE_SYSTEM_MESSAGE, _build_prompt(question, context, answer)]))
    except Exception as exc:
        logger.error("Judge failed after %.3f seconds, passing answer through: %s", time.perf_counter() - start_time, exc)
        return Command(update={"judge": _fallback_verdict(answer, str(exc))})

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Judge verdict in %.3f seconds | passed=%s grounded=%s needs_more_info=%s issues=%d",
        elapsed,
        verdict.passed,
        verdict.grounded,
        verdict.needs_more_info,
        len(verdict.issues),
    )
    for issue in verdict.issues:
        logger.info("Judge issue: %s", issue)

    return Command(
        update={
            "judge": JudgeState(
                original_response=answer,
                passed=verdict.passed,
                grounded=verdict.grounded,
                needs_more_info=verdict.needs_more_info,
                issues=list(verdict.issues),
                error=None,
            )
        }
    )
