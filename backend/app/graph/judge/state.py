from typing_extensions import NotRequired, TypedDict


class JudgeState(TypedDict):
    """Verdict stored in graph state after the judge evaluates the chatbot's final answer."""

    original_response: str
    passed: bool
    grounded: bool
    needs_more_info: bool
    issues: list[str]
    # Links in the answer that failed verification (unreachable, unsupported, or not in any source).
    unsupported_links: NotRequired[list[str]]
    # Source URLs of curated index entries whose page does not exist or has no readable content.
    # The chatbot's context marks documents from these sources as unverified on the revision.
    unverified_sources: NotRequired[list[str]]
    # Set when the judge model failed and the verdict fell back to "pass".
    error: NotRequired[str | None]
