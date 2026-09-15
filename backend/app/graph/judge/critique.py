"""The message that carries a failed verdict back to the chatbot for revision."""
from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage

from .state import JudgeState

# Critiques are HumanMessages so every chat template accepts them mid-conversation, but they are
# tagged by name so the judge, routers and language detector can tell them apart from the user.
CRITIQUE_NAME = "judge_critique"


def is_critique(message: BaseMessage) -> bool:
    return getattr(message, "name", None) == CRITIQUE_NAME


def build_critique(verdict: JudgeState, web_searched: bool = False) -> HumanMessage:
    """Turn a failed verdict into instructions for the chatbot.

    `web_searched` says the web stage already ran this turn; then "search again" is impossible and
    the chatbot is told to answer from what it found instead.
    """
    lines = [
        "[Internal review of your last answer. The user did not write this and cannot see it.]",
        "Keep answering the user's question as you understood it from the conversation; do not ask the "
        "user to clarify something the conversation already makes clear.",
        "Your answer was rejected for these reasons:",
    ]
    lines.extend(f"- {issue}" for issue in verdict.get("issues") or ["The answer did not meet the quality bar."])
    lines.append("")

    unsupported = verdict.get("unsupported_links") or []
    if unsupported:
        lines.append("These links could not be verified. Remove them, and remove or clearly mark as unverified "
                     "every claim that rests only on them. Do not replace them with other links unless the link "
                     "appears in the retrieved context:")
        lines.extend(f"- {url}" for url in unsupported)
        lines.append("")

    unverified = verdict.get("unverified_sources") or []
    if unverified:
        lines.append("These curated sources could not be verified: their page does not exist or has no readable "
                     "content, so nothing from them counts as fact. Remove every claim that rests on them, or state "
                     "plainly that it is unverified. Do not search for them again; a repeated hit on the same entry "
                     "proves nothing:")
        lines.extend(f"- {url}" for url in unverified)
        lines.append("")

    if verdict.get("needs_more_info") and (unverified or web_searched):
        lines.append(
            "You have already searched the web this turn; searching again will not help. Answer from "
            "the sources that were verified, keep only claims they support, and say plainly what is not known."
        )
    elif verdict.get("needs_more_info"):
        lines.append(
            "The information you had was insufficient. Search again with your retrieval tool, "
            "using a higher stage than before, and only then answer."
        )
    elif not verdict.get("grounded", True):
        lines.append(
            "Fix the answer using only the retrieved context you already have. Remove or correct "
            "every claim that the context does not support."
        )
    else:
        lines.append("Rewrite the answer so that every listed problem is fixed.")

    lines.append(
        "Then write the complete corrected answer for the user, in the same language as their "
        "question. Do not mention this review."
    )
    return HumanMessage(content="\n".join(lines), name=CRITIQUE_NAME)
