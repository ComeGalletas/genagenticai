from typing_extensions import NotRequired, TypedDict


class JudgeState(TypedDict):
    """Verdict stored in graph state after the judge evaluates the chatbot's final answer."""

    original_response: str
    passed: bool
    grounded: bool
    needs_more_info: bool
    issues: list[str]
    # Set when the judge model failed and the verdict fell back to "pass".
    error: NotRequired[str | None]
