from __future__ import annotations

import logging
import re

from ..retrieval.schemas import RetrievalResult, RetrievalStage
from .static_data import STATIC_RESULTS

logger = logging.getLogger(__name__)

# Keyword tokens that carry no topical signal. Entries list things like "the matrix movie", and a
# single hit on "the" or "game" must never count as a match.
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "with", "is", "it",
    "game", "games", "movie", "movies", "film", "app", "api", "ai", "types", "static",
    "open", "world", "space", "time", "memory", "safe", "language", "programming",
})
MIN_TOKEN_HITS = 2  # distinct keyword tokens that must appear, unless an adjacent pair matches as a phrase
DECISIVE_CONFIDENCE = 0.95  # the entry's full name is in the query
TENTATIVE_CONFIDENCE = 0.6  # only keywords or a partial phrase matched; other meanings may exist


def _normalize(text: str) -> str:
    """Lowercase, punctuation to spaces, single spaces: "Clair Obscur: Expedition 33" -> "clair obscur expedition 33"."""
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").casefold()))


def _entry_name(entry: dict) -> str:
    return entry.get("name") or entry.get("title", "").split(" - ")[0]


def _name_matches(normalized_query: str, entry: dict) -> bool:
    name = _normalize(_entry_name(entry))
    return bool(name) and re.search(rf"\b{re.escape(name)}\b", normalized_query) is not None


def _score(normalized_query: str, query_words: set[str], keywords: str) -> int:
    """Number of meaningful keyword tokens found in the query; adjacent-pair phrase hits count as a full match."""
    tokens = keywords.casefold().split()
    meaningful = [t for t in tokens if t not in _STOPWORDS]
    hits = sum(1 for t in set(meaningful) if t in query_words)

    # A phrase such as "hollow knight" or "expedition 33" is enough to match, though not to be decisive.
    for first, second in zip(tokens, tokens[1:]):
        if first in _STOPWORDS or second in _STOPWORDS:
            continue
        if re.search(rf"\b{re.escape(first)}\s+{re.escape(second)}\b", normalized_query):
            return max(hits, MIN_TOKEN_HITS)
    return hits


def query_static_search(query: str) -> list[RetrievalResult]:
    """Return curated entries that match the query, without any network request.

    A match needs at least MIN_TOKEN_HITS distinct meaningful keyword tokens, or an adjacent
    keyword pair appearing as a phrase. A match is *decisive* (metadata["decisive"] True,
    confidence 0.95) only when the entry's full name appears in the query; otherwise it is
    *tentative* (confidence 0.6) and the retrieval pipeline keeps searching other sources, so a
    partial name like "Expedition 33" can still surface its other meanings.
    """
    normalized = _normalize(query)
    query_words = set(normalized.split())
    logger.info("Static search called — query: %r", normalized)

    scored: list[tuple[bool, int, dict]] = []
    for entry in STATIC_RESULTS:
        decisive = _name_matches(normalized, entry)
        hits = _score(normalized, query_words, entry["keywords"])
        if decisive or hits >= MIN_TOKEN_HITS:
            scored.append((decisive, hits, entry))
    scored.sort(key=lambda item: (not item[0], -item[1]))

    retrieval_results = [
        RetrievalResult(
            title=entry.get("title", "Untitled"),
            content=entry.get("snippet", ""),
            source=entry.get("url", ""),
            stage=RetrievalStage.STATIC,
            confidence=DECISIVE_CONFIDENCE if decisive else TENTATIVE_CONFIDENCE,
            metadata={"keyword_hits": hits, "decisive": decisive, "match": "name" if decisive else "keywords"},
        )
        for decisive, hits, entry in scored
    ]

    if retrieval_results:
        logger.info(
            "Static entry found for %r: %r (%s)",
            normalized, retrieval_results[0].title, "decisive" if retrieval_results[0].metadata["decisive"] else "tentative",
        )
    else:
        logger.info("No static entry found for %r", normalized)
    return retrieval_results
