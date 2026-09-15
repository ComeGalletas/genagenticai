"""Planned CV-analysis subgraph. Only the state schema exists; nothing is wired into the main graph yet."""
from typing_extensions import NotRequired, TypedDict


class CVAnalysisState(TypedDict):
    """State for matching a CV against job postings."""

    cv: str
    job_postings: list[str]
    analysis: NotRequired[str]
    match_score: NotRequired[float]
    missing_skills: NotRequired[list[str]]
    strengths: NotRequired[list[str]]
    weaknesses: NotRequired[list[str]]
