import logging
import re
import time
from typing import cast

from babel import Locale
from langdetect import DetectorFactory, LangDetectException, detect_langs
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.types import Command

from ...config import CHAT_HISTORY_MAX_TOKENS, CHAT_KEEP_ALIVE, CHAT_MODEL, CHAT_NUM_CTX
from ..judge.verify import normalize_url, strip_links, unverified_note, unverified_sources_note
from .formatting import render_reply
from .router import turn_messages
from .state import State
from .tools import get_current_time, read_webpage, retrieve_baloto_results, retrieve_information, retrieve_job_postings, suggest_baloto_numbers
from .system_prompt import SYSTEM_PROMPT

DetectorFactory.seed = 0  # langdetect samples randomly by default; a fixed seed makes it deterministic
MIN_DETECTABLE_CHARS = 12
MIN_LANGUAGE_CONFIDENCE = 0.85
_LANGUAGE_NOISE = re.compile(r"```.*?```|`[^`]*`|https?://\S+|[\d_/\\<>{}\[\]]+", re.DOTALL)

SYSTEM_MESSAGE = SystemMessage(content=SYSTEM_PROMPT)

logger = logging.getLogger(__name__)

tools = [get_current_time, read_webpage, retrieve_information, retrieve_baloto_results, retrieve_job_postings, suggest_baloto_numbers]
_llm = ChatOllama(
    model=CHAT_MODEL,  # set CHAT_MODEL in .env to switch (e.g. granite4.1:8b or qwen3:14b fit a 16 GB GPU entirely)
    temperature=0.1,
    num_ctx=CHAT_NUM_CTX,
    keep_alive=CHAT_KEEP_ALIVE,
)
llm_with_tools = _llm.bind_tools(tools)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _extract_text(content) -> str:
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


def _language_name(code: str | None) -> str:
    """Turn a langdetect code ('es', 'zh-cn') into an English language name, falling back to the code."""
    if not code:
        return "unknown"
    try:
        return Locale.parse(code.replace("-", "_")).get_display_name("en") or code
    except Exception:
        return code


def build_retrieval_context(state: State) -> SystemMessage | None:
    """Converts the retrieval state into a temporary system message.

    Documents whose source the judge found unverifiable (page missing or empty) are labelled so
    the chatbot does not present them as fact on a revision.
    """
    retrievals = state.get("retrieval", [])
    if not retrievals:
        return None

    unverified = {normalize_url(u) for u in ((state.get("judge") or {}).get("unverified_sources") or [])}

    parts = [
        "## Retrieved Context",
        "",
        "Use this information as the primary factual context when answering.",
        "If the retrieved information is sufficient, prefer it over your internal knowledge.",
        "Entries marked UNVERIFIED have a source page that does not exist or is empty: do not state their content as fact.",
        "",
    ]

    for retrieval in retrievals:
        if retrieval["retrieval_status"] != "FOUND":
            continue

        query = retrieval.get("retrieval_query", "")
        parts.append(f"### Retrieval Query")
        parts.append(query)
        parts.append("")


        for doc in retrieval.get("retrieval_documents", []):
            if doc is not None:
                source = str(doc.get("source", ""))
                flagged = source.startswith("http") and normalize_url(source) in unverified
                parts.append(f"Source: {source}" + ("  [UNVERIFIED: source page missing or empty]" if flagged else ""))
                parts.append(f"Title: {doc['title']}")
                parts.append(f"Stage: {doc['stage']}")
                parts.append(f"Confidence: {doc['confidence']}")
                parts.append("")
                parts.append(doc['content'])
                parts.append("")
                parts.append("-" * 40)
                parts.append("")

    if len(parts) < 1:
        return None

    return SystemMessage(content="\n".join(parts))


def trim_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Keep the newest messages that fit the token budget, always starting on a human message so
    tool-call/tool-result pairs stay intact. The current turn is never dropped, even if it alone
    exceeds the budget. Only what is sent to the LLM is trimmed; the checkpoint keeps everything.
    """
    trimmed = trim_messages(
        messages,
        max_tokens=CHAT_HISTORY_MAX_TOKENS,
        token_counter=count_tokens_approximately,
        strategy="last",
        start_on="human",
        include_system=False,
        allow_partial=False,
    )
    if not trimmed:
        # Nothing fit: fall back to the current turn (last real user message onward).
        turn = turn_messages({"messages": messages})
        trimmed = messages[max(0, len(messages) - len(turn) - 1):]
    if len(trimmed) < len(messages):
        logger.info("History trimmed | %d -> %d messages (~%d tokens budget)", len(messages), len(trimmed), CHAT_HISTORY_MAX_TOKENS)
    return trimmed

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def prepare_turn(state: State) -> Command:
    """First node of every turn: detect the user's language and reset per-turn judge state.

    `retrieval` is deliberately not reset; its reducer keeps a bounded window so follow-up
    questions can reuse the previous search.
    """
    return Command(update={
        "user_language": detect_language(state),
        "judge": None,
        "judge_attempts": 0,
        "final_response": None,
    })


def detect_language(state: State) -> str:
    """Deterministically name the language of the latest user message (English name, e.g. 'Spanish')."""
    last_human = next((m for m in reversed(state["messages"]) if getattr(m, "type", "") == "human"), None)
    raw = _extract_text(last_human.content) if last_human is not None else ""
    text = _LANGUAGE_NOISE.sub(" ", raw).strip()  # urls, code and digits push langdetect towards English

    previous = state.get("user_language")
    language = previous
    if len(text) >= MIN_DETECTABLE_CHARS:
        try:
            candidates = detect_langs(text)
            best = candidates[0]
            logger.debug("Language candidates: %s", candidates)
            if best.prob >= MIN_LANGUAGE_CONFIDENCE:
                language = best.lang
            else:
                logger.info("Language guess %s rejected at %.2f confidence, keeping %s", best.lang, best.prob, previous)
        except LangDetectException:
            logger.warning("Language detection failed for %r", text[:60])

    name = _language_name(language)
    logger.info("Detected user language: %s", name)
    return name


def chatbot(state: State) -> Command:
    """Invoke the LLM and append its response to the conversation.
    Args:
        state: The current state of the conversation, including messages and any relevant context.
    Returns:
        A Command object for the LLM containing the updated conversation state with the LLMs response.
    """
    logger.debug("Entering chatbot node.")

    messages: list = []
    retrieval_context = build_retrieval_context(state)
    if retrieval_context:
        messages.append(retrieval_context)

    messages.extend(trim_history(state["messages"]))

    start_time = time.perf_counter()
    response = cast(AIMessage, llm_with_tools.invoke([SYSTEM_MESSAGE, *messages]))
    elapsed = time.perf_counter() - start_time

    logger.info("LLM response completed in %.3f seconds.", elapsed)
    if response.tool_calls:
        logger.info("LLM requested tool: %s", response.tool_calls[0]["name"])
    else:
        logger.info("LLM produced final response.")
        
    #print("LLM response:", response)
    return Command(update={"messages": [response]})


def finalize(state: State) -> Command:
    """Convert the chatbot's final markdown answer into sanitized HTML for the client.

    The conversation history keeps the markdown so the LLM never sees HTML in its own past turns.
    """
    last_ai = next((m for m in reversed(state["messages"]) if getattr(m, "type", "") == "ai"), None)
    text = _extract_text(getattr(last_ai, "content", "")) if last_ai is not None else ""

    verdict = state.get("judge") or {}
    if not text.strip():
        # A revision came back empty (seen with a thinking model that spent its budget thinking).
        # The judged original answer beats an empty reply.
        original = verdict.get("original_response") or ""
        if original.strip():
            logger.warning("Finalize | last answer is empty; falling back to the original answer the judge reviewed")
            text = original

    # Last line of defense: links the judge could not verify never reach the user, even when the
    # revision kept them or the retry budget ran out.
    unsupported = verdict.get("unsupported_links") or []
    if unsupported:
        text, removed = strip_links(text, unsupported)
        if removed:
            logger.warning("Finalize | stripped %d unverified link(s): %s", removed, unsupported)
            text += unverified_note(removed)

    # Claims cannot be cut out mechanically, so when the final answer still rests on a curated
    # entry whose page is missing or empty, the reply names that source as unverified. The note is
    # built from verification data, never from the judge's prose.
    if verdict and not verdict.get("passed", True) and verdict.get("unverified_sources"):
        note = unverified_sources_note(state, verdict["unverified_sources"])
        if note:
            logger.warning("Finalize | final answer still rests on unverified sources; appending note")
            text += note

    rendered = render_reply(text)
    logger.info("Finalize | %d chars markdown -> %d chars html", len(text), len(rendered))
    return Command(update={"final_response": rendered})
