from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

from ..settings import KNOWLEDGE_DIR

BALOTO_COLLECTION = "baloto"

_KEYWORDS_FILE = KNOWLEDGE_DIR / "baloto_keywords.json"
_DATE_PATTERNS = (
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
)


# ---------------------------------------------------------------------------
# Query routing
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_baloto_keywords() -> dict[str, int]:
    with _KEYWORDS_FILE.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return {str(key).casefold(): int(value) for key, value in payload.items()}


def score_baloto_query(question: str) -> int:
    """Return a deterministic score for routing a query to the Baloto collection."""
    lowered = question.casefold()
    score = sum(points for keyword, points in load_baloto_keywords().items() if keyword in lowered)

    if any(pattern.search(question) for pattern in _DATE_PATTERNS):
        score += 3

    if re.search(r"\b(what|which)\s+numbers\b", lowered):
        score += 2

    if re.search(r"\b(last|latest|recent)\s+(draw|result|results)\b", lowered):
        score += 2

    return score


# ---------------------------------------------------------------------------
# Collection source
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BalotoSource:
    """Baloto draw results, one document per month."""

    name: str = BALOTO_COLLECTION
    file: Path = KNOWLEDGE_DIR / "baloto_results.json"

    def load(self) -> list[Document]:
        with self.file.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)

        grouped_rows: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            parsed_date = datetime.strptime(row["date"], "%Y-%m-%d")
            grouped_rows[f"{parsed_date.year:04d}-{parsed_date.month:02d}"].append({
                **row,
                "year": parsed_date.year,
                "month": parsed_date.month,
                "day": parsed_date.day,
            })

        documents: list[Document] = []
        for month_key, month_rows in sorted(grouped_rows.items()):
            ordered_rows = sorted(month_rows, key=lambda item: item["date"], reverse=True)
            sample_row = ordered_rows[0]

            lines = [f"Baloto results for {month_key}"]
            for item in ordered_rows:
                lines.append(" | ".join([
                    f"Date: {item['date']}",
                    f"Year: {item['year']}",
                    f"Month: {item['month']}",
                    f"Day: {item['day']}",
                    f"Numbers: {item['raw_numbers']}",
                ]))

            documents.append(Document(
                page_content="\n".join(lines),
                metadata={
                    "category": self.file.parent.name,
                    "filename": self.file.name,
                    "source": str(self.file),
                    "source_type": "json",
                    "dataset": "baloto_results",
                    "title": f"Baloto results {month_key}",
                    "year": sample_row["year"],
                    "month": sample_row["month"],
                    "days": [item["day"] for item in ordered_rows],
                    "month_key": month_key,
                    "record_count": len(ordered_rows),
                },
            ))

        return documents

    def score(self, question: str) -> int:
        return score_baloto_query(question)
