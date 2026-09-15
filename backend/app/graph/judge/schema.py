from __future__ import annotations

from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    """Structured verdict returned by the judge model. Routing reads these fields, never prose."""

    passed: bool = Field(description="True when the answer is acceptable to send to the user as-is.")
    grounded: bool = Field(
        description="True when every factual claim is supported by the provided context or is common knowledge. "
        "False when the answer invents facts or contradicts the context."
    )
    needs_more_info: bool = Field(
        description="True when the answer admits it could not find information or the context is insufficient "
        "to answer the question, so another search would help."
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Short, concrete problems with the answer. Empty when the answer passes.",
    )
