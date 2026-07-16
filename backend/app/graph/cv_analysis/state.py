class CVAnalysisState(TypedDict):
    """State used only by the CV analysis graph."""

    messages: list

    cv: str
    job_postings: list[str]
    analysis: NotRequired[str]
    match_score: NotRequired[float]
    missing_skills: NotRequired[list[str]]
    strengths: NotRequired[list[str]]
    weaknesses: NotRequired[list[str]]
