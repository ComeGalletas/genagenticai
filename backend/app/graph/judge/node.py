from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.types import Command

from ...config import (
    JUDGE_KEEP_ALIVE,
    JUDGE_MAX_DOC_CHARS,
    JUDGE_MAX_RETRIEVALS,
    JUDGE_MAX_TOOL_CHARS,
    JUDGE_MODEL,
    JUDGE_NUM_CTX,
    JUDGE_VERIFY_LINKS,
    OLLAMA_BASE_URL,
)
from ..core.router import turn_messages, turn_retrievals, web_searched_this_turn
from ..core.state import State
from ..retrieval.state import RETRIEVAL_ACK_PREFIX
from .critique import build_critique, is_critique
from .schema import JudgeVerdict
from .state import JudgeState
from .system_prompt import JUDGE_SYSTEM_PROMPT
from .verify import (
    build_plan,
    extract_links,
    fetch_pages,
    format_verification,
    normalize_url,
    unreachable_cited_links,
    unverified_sources,
)

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_MESSAGE = SystemMessage(content=JUDGE_SYSTEM_PROMPT)

# Small, non-thinking model: the judge only has to emit a short JSON verdict.
_judge_llm = ChatOllama(
    model=JUDGE_MODEL,
    temperature=0.0,
    reasoning=False,
    num_ctx=JUDGE_NUM_CTX,
    keep_alive=JUDGE_KEEP_ALIVE,
    base_url=OLLAMA_BASE_URL,
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
    """Text of the newest message of a type, skipping judge critiques so 'human' means the real user."""
    message = next(
        (m for m in reversed(state["messages"]) if getattr(m, "type", "") == message_type and not is_critique(m)),
        None,
    )
    return _extract_text(getattr(message, "content", "")) if message is not None else ""


RECENT_EXCHANGES = 3          # user/assistant pairs shown to the judge before the current question
RECENT_MESSAGE_CHARS = 400    # per message


def _recent_conversation(state: State) -> str:
    """The last few user/assistant exchanges before this turn, so the judge can resolve "her", "it", "that game"."""
    messages = state["messages"]
    turn_start = len(messages) - len(turn_messages(state)) - 1  # index of the current real user message
    if turn_start <= 0:
        return "(this is the first question of the conversation)"

    lines: list[str] = []
    for message in messages[:turn_start]:
        kind = getattr(message, "type", "")
        if kind == "human" and not is_critique(message):
            text = _extract_text(message.content).strip()
            if text:
                lines.append(f"User: {text[:RECENT_MESSAGE_CHARS]}")
        elif kind == "ai" and not (getattr(message, "tool_calls", None) or []):
            text = _extract_text(message.content).strip()
            if text:
                lines.append(f"Assistant: {text[:RECENT_MESSAGE_CHARS]}")
    lines = lines[-2 * RECENT_EXCHANGES:]
    return "\n".join(lines) if lines else "(no earlier exchanges)"


def _format_context(state: State) -> str:
    """Render what the assistant actually had: this turn's retrieval documents in full, earlier
    turns' documents as titles only (background), plus this turn's raw tool outputs.

    Tools such as get_current_time, suggest_baloto_numbers and retrieve_job_postings return their
    data only as ToolMessages, so without them the judge would wrongly call those answers ungrounded.
    """
    lines: list[str] = []
    current = turn_retrievals(state)
    for retrieval in current:
        if retrieval.get("retrieval_status") != "FOUND":
            continue
        for doc in retrieval.get("retrieval_documents") or []:
            if doc is None:
                continue
            lines.append(f"[{doc.get('stage', '?')}] {doc.get('title', 'Untitled')} ({doc.get('source', 'unknown')})")
            lines.append(str(doc.get("content", ""))[:JUDGE_MAX_DOC_CHARS])
            lines.append("")

    earlier_titles: list[str] = []
    for retrieval in (state.get("retrieval") or [])[-JUDGE_MAX_RETRIEVALS:]:
        if retrieval in current or retrieval.get("retrieval_status") != "FOUND":
            continue
        earlier_titles.extend(str(doc.get("title", "Untitled")) for doc in (retrieval.get("retrieval_documents") or []) if doc)
    if earlier_titles:
        lines.append("Retrieved in earlier turns (background only, may be unrelated to this question): "
                     + "; ".join(dict.fromkeys(earlier_titles)))
        lines.append("")

    for message in turn_messages(state):
        if getattr(message, "type", "") != "tool":
            continue
        content = _extract_text(getattr(message, "content", "")).strip()
        if not content or content.startswith(RETRIEVAL_ACK_PREFIX):
            continue  # retrieval tools only acknowledge; their documents are listed above
        lines.append(f"[tool output: {getattr(message, 'name', None) or 'tool'}]")
        if len(content) > JUDGE_MAX_TOOL_CHARS:
            content = content[:JUDGE_MAX_TOOL_CHARS] + "\n... (tool output truncated; treat later items as present)"
        lines.append(content)
        lines.append("")

    return "\n".join(lines).strip() or "(no documents or tool outputs for this turn)"


def _build_prompt(conversation: str, question: str, context: str, verification: str, answer: str) -> HumanMessage:
    # Small models assume their training-cutoff date and then call real recent dates "future".
    today = datetime.now().astimezone().strftime("%Y-%m-%d (%A)")
    return HumanMessage(
        content=(
            f"## TODAY\n{today}\n\n"
            "## RECENT CONVERSATION\n"
            f"{conversation}\n\n"
            "## USER QUESTION\n"
            f"{question}\n\n"
            "## RETRIEVED CONTEXT\n"
            f"{context}\n\n"
            "## CITED PAGES\n"
            f"{verification}\n\n"
            "## CANDIDATE ANSWER\n"
            f"{answer}"
        )
    )


def _verify_links(state: State, answer: str) -> tuple[str, list[str], list[str]]:
    """Fetch the pages behind the answer's links and curated sources.

    Returns the prompt section, the cited links that are unsupported before the model even looks
    (pages that do not exist, or links beyond the fetch cap), and the curated sources whose page
    does not exist or has no readable content.
    """
    if not JUDGE_VERIFY_LINKS:
        return "(link verification disabled)", [], []
    plan = build_plan(state, answer, JUDGE_MAX_RETRIEVALS)
    if not plan.targets:
        return format_verification(plan, {}), [], []
    start = time.perf_counter()
    pages = fetch_pages(t.url for t in plan.targets)
    bad = unreachable_cited_links(plan, pages)
    unverified = unverified_sources(plan, pages)
    statuses = {p.url: p.status for p in pages.values()}
    logger.info(
        "Link verification | %d page(s) in %.2fs | %s | unreachable cited: %s | unverified curated: %s",
        len(pages), time.perf_counter() - start, statuses, bad, [u.url for u in unverified],
    )
    return format_verification(plan, pages), bad, [u.url for u in unverified]


def _merge_unsupported(model_links: list[str], deterministic_links: list[str], answer: str) -> list[str]:
    """Union of the judge's unsupported links and the deterministic ones, kept only if actually in the answer."""
    present = {normalize_url(u): u for u in extract_links(answer)}
    merged: dict[str, str] = {}
    for url in deterministic_links + list(model_links):
        key = normalize_url(url)
        if key in present:
            merged.setdefault(key, present[key])
    return list(merged.values())


def _fallback_verdict(answer: str, error: str) -> JudgeState:
    """When the judge cannot run, let the answer through rather than blocking the user."""
    return JudgeState(
        original_response=answer,
        passed=True,
        grounded=True,
        needs_more_info=False,
        issues=[],
        unsupported_links=[],
        unverified_sources=[],
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
    conversation = _recent_conversation(state)
    context = _format_context(state)
    try:
        verification, unreachable, unverified = _verify_links(state, answer)
    except Exception as exc:  # verification must never block the judge
        logger.error("Link verification failed, judging without it: %s", exc)
        verification, unreachable, unverified = "(link verification failed)", [], []

    start_time = time.perf_counter()
    try:
        verdict = cast(JudgeVerdict, _judge.invoke([JUDGE_SYSTEM_MESSAGE, _build_prompt(conversation, question, context, verification, answer)]))
    except Exception as exc:
        logger.error("Judge failed after %.3f seconds, passing answer through: %s", time.perf_counter() - start_time, exc)
        return Command(update={"judge": _fallback_verdict(answer, str(exc))})

    unsupported = _merge_unsupported(verdict.unsupported_links, unreachable, answer)
    issues = list(verdict.issues)
    passed = verdict.passed
    if unsupported:
        passed = False
        flagged = ", ".join(unsupported)
        if not any(url in issue for issue in issues for url in unsupported):
            issues.append(f"Unverifiable link(s) cited: {flagged}")
    if unverified and passed:
        # The model passed an answer that rests on a curated entry whose page does not exist.
        passed = False
        issues.append("The answer relies on a curated index entry whose source page could not be verified: " + ", ".join(unverified))

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Judge verdict in %.3f seconds | passed=%s grounded=%s needs_more_info=%s issues=%d unsupported_links=%d",
        elapsed, passed, verdict.grounded, verdict.needs_more_info, len(issues), len(unsupported),
    )
    for issue in issues:
        logger.info("Judge issue: %s", issue)

    return Command(
        update={
            "judge": JudgeState(
                original_response=answer,
                passed=passed,
                grounded=verdict.grounded,
                needs_more_info=verdict.needs_more_info,
                issues=issues,
                unsupported_links=unsupported,
                unverified_sources=unverified,
                error=None,
            )
        }
    )


def revise(state: State) -> Command:
    """Send a failed verdict back to the chatbot as a critique message and count the attempt."""
    verdict = state.get("judge")
    if verdict is None:
        logger.warning("Revise node called without a verdict; nothing to do.")
        return Command(update={})

    attempts = state.get("judge_attempts", 0) + 1
    web_done = web_searched_this_turn(state)
    logger.info(
        "Revise | attempt %d | needs_more_info=%s grounded=%s issues=%d web_searched=%s",
        attempts,
        verdict.get("needs_more_info"),
        verdict.get("grounded"),
        len(verdict.get("issues") or []),
        web_done,
    )
    return Command(update={"messages": [build_critique(verdict, web_searched=web_done)], "judge_attempts": attempts})
