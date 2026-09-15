from __future__ import annotations

import re
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import trafilatura
from bs4 import BeautifulSoup
from ddgs import DDGS

from ..config import WEB_FETCH_TIMEOUT, WEB_FETCH_WORKERS
from ..retrieval.schemas import RetrievalResult, RetrievalStage

logger = logging.getLogger(__name__)
logging.getLogger("primp").setLevel(logging.WARNING)
logging.getLogger("trafilatura").setLevel(logging.ERROR)

MAX_RESULTS = 5        # Maximum number of search results to return per query
MAX_CONTENT_CHARS = 1000  # Hard cap on extracted text to avoid oversized payloads
MIN_WORDS = 40
BLOCKED_STATUS_CODES = frozenset({401, 403, 429, 503})  # "you look like a bot"; retried with trafilatura's fetcher


def fetch_webpage(
    url: str,
    fallback_content: str = "",
    timeout: float = WEB_FETCH_TIMEOUT,
    max_chars: int = MAX_CONTENT_CHARS,
) -> tuple[str, dict[str, Any]]:
    """Download a webpage and extract its readable plain-text content.

    Attempts to fetch the URL with ``httpx``, then runs ``trafilatura`` to
    extract the main body text from the HTML.  If trafilatura cannot extract
    anything meaningful (e.g. heavy JavaScript pages), ``BeautifulSoup`` is
    used as a secondary fallback to get all visible text.  If the HTTP request
    itself fails for any reason, ``fallback_content`` (typically the search
    snippet) is returned instead.

    Args:
        url: The URL to download.  An empty string skips the request entirely.
        fallback_content: Text to return when the page cannot be fetched or
            parsed. Defaults to ``""``.
        timeout: HTTP request timeout in seconds.  Defaults to ``WEB_FETCH_TIMEOUT``.
        max_chars: Cap on the extracted text. Defaults to ``MAX_CONTENT_CHARS``.
    Returns:
        A two-element tuple, (text, metadata) for the webpage.
    """
    metadata: dict[str, Any] = {"content_source": "snippet"}
    if not url:
        return fallback_content, metadata

    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True,
            headers={"User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/138.0 Safari/537.36"
            )},
        )

        html = ""
        if response.status_code in BLOCKED_STATUS_CODES:
            # Some sites (Wikipedia among them) refuse httpx but accept trafilatura's fetcher.
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                html = downloaded
                metadata["fetched_via"] = "trafilatura"
        if not html:
            if response.status_code != 200:
                metadata["status_code"] = response.status_code
                return fallback_content, metadata
            if "text/html" not in response.headers.get("Content-Type", ""):
                metadata["content_type"] = response.headers.get("Content-Type")
                return fallback_content, metadata
            html = response.text
        text = trafilatura.extract(html, include_comments=False, include_tables=True, include_formatting=False)

        # Secondary fallback: BeautifulSoup when trafilatura yields nothing, html source gets extracted as plain text
        if not text:
            text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
        # Add constant metadata regardless if there were search results or not and limits the text if there is, so the LLM can reason about the source of the content.
        if text:
            text = re.sub(r"\s+", " ", text).strip()
            word_count = len(re.findall(r"\b\w+\b", text))

            if word_count >= MIN_WORDS:
                metadata["content_source"] = "webpage"
                return text[:max_chars], metadata

    except Exception as e:
        metadata["error"] = str(e)

    return fallback_content, metadata


def query_web_search(query: str, max_results: int = MAX_RESULTS) -> list[RetrievalResult]:
    """Search DuckDuckGo and enrich each result with full webpage content.

    Issues a DuckDuckGo text search via ``ddgs.DDGS``, then calls
    :func:`fetch_webpage` for every result URL to replace the short snippet
    with the actual page body (up to ``MAX_CONTENT_CHARS`` characters).  If
    the DuckDuckGo request fails entirely, an empty list is returned and the
    exception is logged.

    Args:
        query: The search query string to send to DuckDuckGo.
        max_results: Maximum number of results to request.
    Returns:
        A list of ``RetrievalResult`` objects. Returns an empty list on search failure.
    """
    logger.debug("DuckDuckGo query: %r", query)

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        logger.exception("DuckDuckGo search failed")
        return []

    if not results:
        logger.info("DuckDuckGo returned no results for %r", query)
        return []

    # Fetch every result page concurrently; the stage then costs about one slow page, not the sum.
    def _enrich(result: dict) -> tuple[str, dict]:
        return fetch_webpage(url=result.get("href", ""), fallback_content=result.get("body", ""), timeout=WEB_FETCH_TIMEOUT)

    with ThreadPoolExecutor(max_workers=max(1, min(WEB_FETCH_WORKERS, len(results)))) as pool:
        enriched = list(pool.map(_enrich, results))

    retrieval_results: list[RetrievalResult] = []
    for result, (content, meta) in zip(results, enriched):
        retrieval_results.append(RetrievalResult(
            title=result.get("title", "Untitled"),
            content=content,
            source=result.get("href", ""),
            stage=RetrievalStage.WEB,
            metadata=meta,
            confidence=0.92,  # DuckDuckGo results are considered highly relevant
        ))

    logger.info("Returning %d DuckDuckGo result(s) for query %r", len(retrieval_results), query)

    return retrieval_results
