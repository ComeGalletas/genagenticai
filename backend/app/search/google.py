from __future__ import annotations
from importlib.metadata import metadata
import logging
import httpx
from bs4 import BeautifulSoup

from typing import Any

from ddgs import DDGS
from copy import deepcopy
from trafilatura.settings import DEFAULT_CONFIG
import trafilatura

from urllib.parse import urlparse


from ..retrieval.schemas import RetrievalResult, RetrievalStage

logger = logging.getLogger(__name__)
logging.getLogger("primp").setLevel(logging.WARNING)
logging.getLogger("trafilatura").setLevel(logging.ERROR)

MAX_RESULTS = 7  # Maximum number of Google search results to return
MAX_CONTENT_CHARS = 5000

config = deepcopy(DEFAULT_CONFIG)
config['DEFAULT']['DOWNLOAD_TIMEOUT'] = '3'
config['DEFAULT']['MAX_REDIRECTS'] = '1'
config['DEFAULT']['SLEEP_TIME'] = '0'

def fetch_webpage(url: str, fallback_content: str = "", timeout: int = 15) -> tuple[str, dict[str, Any]]:
    """
    Download a webpage and extract its readable content.

    Falls back to the search snippet if downloading or extraction fails.
    """
    metadata: dict[str, Any] = {"content_source": "snippet"}
    if not url:
        return fallback_content, metadata

    try:
        response = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/138.0 Safari/537.36"
                )
            },
        )

        if response.status_code != 200:
            metadata["status_code"] = response.status_code
            return fallback_content, metadata

        if "text/html" not in response.headers.get("Content-Type", ""):
            metadata["content_type"] = response.headers.get("Content-Type")
            return fallback_content, metadata

        html = response.text

        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            include_formatting=False,
        )

        # Fallback if Trafilatura couldn't extract anything
        if not text:
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(" ", strip=True)

        if text:
            metadata["content_source"] = "webpage"
            return text[:MAX_CONTENT_CHARS], metadata

    except Exception as e:
        metadata["error"] = str(e)

    return fallback_content, metadata


def query_ddu_google_search(query: str, max_results: int = MAX_RESULTS) -> list[RetrievalResult]:
    """
    Search DuckDuckGo and enrich each result with webpage content.
    """

    logger.debug("DuckDuckGo query: %r", query)

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        logger.exception("DuckDuckGo search failed")
        return []

    retrieval_results: list[RetrievalResult] = []

    for result in results:
        title = result.get("title", "Untitled")
        url = result.get("href", "")
        snippet = result.get("body", "")

        content, metadata = fetch_webpage(url=url, fallback_content=snippet)

        retrieval_results.append(
            RetrievalResult(
                title=title,
                content=content,
                source=url,
                stage=RetrievalStage.GOOGLE,
                metadata=metadata,
            )
        )

    logger.info("Returning %d Google result(s) for query %r", len(retrieval_results), query)

    return retrieval_results


def query_ddu_google_search_oldi(query: str, max_results: int = MAX_RESULTS) -> list[RetrievalResult]:
    """Search DuckDuckGo and return extracted webpage content."""
    logger.debug("Google query: %r", query)

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        logger.exception("DuckDuckGo search failed")
        return []

    retrieval_results: list[RetrievalResult] = []
    for result in results:
        title = result.get("title", "Untitled")
        url = result.get("href", "")
        snippet = result.get("body", "")
        content = snippet
        metadata: dict[str, str | None] = {"engine": "DuckDuckGo", "content_source": "snippet",}

        if url:
            try:
                downloaded = trafilatura.fetch_url(url, config=config)
                if downloaded:
                    extracted_text = trafilatura.extract(
                        downloaded,
                        include_comments=False,
                        include_tables=True,
                        include_formatting=False,
                    )
                    extracted_metadata = trafilatura.extract_metadata(downloaded)
                    if extracted_text:
                        content = extracted_text[:MAX_CONTENT_CHARS]
                        metadata["content_source"] = "trafilatura"

                    if extracted_metadata:
                        if extracted_metadata.sitename:
                            metadata["site"] = extracted_metadata.sitename
                        if extracted_metadata.author:
                            metadata["author"] = extracted_metadata.author
                        if extracted_metadata.date:
                            metadata["date"] = extracted_metadata.date
                        if extracted_metadata.language:
                            metadata["language"] = extracted_metadata.language
                        if extracted_metadata.hostname:
                            metadata["hostname"] = extracted_metadata.hostname
                else:
                    logger.debug("Trafilatura could not download %s", url)

            except Exception as e:
                logger.warning("Failed to extract %s: %s", url, e)
        
        retrieval_results.append(
            RetrievalResult(
                title=title,
                content=content,
                source=url,
                stage=RetrievalStage.GOOGLE,
                metadata=metadata,
            )
        )

    logger.info("Returning %d Google result(s) for query %r", len(retrieval_results), query)

    return retrieval_results


def query_ddu_google_search_old(query: str, max_results: int = MAX_RESULTS) -> list[RetrievalResult]:
    """Search DuckDuckGo and return structured retrieval results."""
    logger.debug("Google query: %r", query)
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        logger.exception("DuckDuckGo search failed")
        return []
    
    retrieval_results: list[RetrievalResult] = []
    for result in results:
        retrieval_results.append(
            RetrievalResult(
                title=result.get("title", "Untitled"),
                content=result.get("body", ""),
                source=result.get("href", ""),
                stage=RetrievalStage.GOOGLE,
                metadata={"engine": "DuckDuckGo"},
            )
        )
    logger.info("Returning %d Google result(s) for query %r", len(retrieval_results), query) 

    return retrieval_results
