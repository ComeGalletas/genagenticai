"""Link verification for the judge: which links an answer cites, where they came from, and what
their pages actually say.

Origins of a link, from most to least trusted:
- "tool":    appears in a tool output or tool-call argument this turn (job listings, read_webpage)
- "web":     the source of a web-stage document already fetched into the context
- "static":  the source of a curated index entry; hand-written, so its page is fetched and checked
- "unknown": appears nowhere in the context; the model produced it, so its page is fetched and checked

Curated entries the answer relied on are also fetched even when not cited, because an entry
whose page is unreachable or silent is not evidence for the claims built on it.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ...config import JUDGE_MAX_VERIFY_LINKS, JUDGE_VERIFY_PAGE_CHARS, JUDGE_VERIFY_TIMEOUT
from ...retrieval.schemas import RetrievalStage
from ...search.ddgo import fetch_webpage
from ..core.router import turn_messages, turn_retrievals

logger = logging.getLogger(__name__)

_URL = re.compile(r"https?://[^\s<>\"'`()\[\]]+")
_MD_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^\s)`]+)\)")
_HTML_LINK = re.compile(r"<a\s+[^>]*href=[\"'](https?://[^\"']+)[\"'][^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
_TRAILING = ".,;:!?`*_"  # punctuation and markdown emphasis that models glue onto a URL

TRUSTED_ORIGINS = frozenset({"tool", "web"})


def normalize_url(url: str) -> str:
    url = url.strip().rstrip(_TRAILING)
    url = url.split("#", 1)[0]
    scheme, _, rest = url.partition("://")
    host, slash, path = rest.partition("/")
    return f"{scheme.lower()}://{host.lower()}{slash}{path}".rstrip("/")


def extract_links(text: str) -> list[str]:
    """Every http(s) link in markdown/plain/HTML text, in order of first appearance, deduplicated."""
    seen: dict[str, str] = {}
    for match in _MD_LINK.finditer(text or ""):
        seen.setdefault(normalize_url(match.group(2)), match.group(2))
    for match in _HTML_LINK.finditer(text or ""):
        seen.setdefault(normalize_url(match.group(1)), match.group(1))
    for match in _URL.finditer(text or ""):
        seen.setdefault(normalize_url(match.group(0)), match.group(0).rstrip(_TRAILING))
    return list(seen.values())


@dataclass
class VerifyTarget:
    url: str
    origin: str                      # tool | web | static | unknown
    cited: bool                      # appears in the answer
    entry_title: str | None = None   # for static: the curated entry it backs


@dataclass
class FetchedPage:
    url: str
    status: str                      # ok | unreachable | blocked | no_text | skipped
    text: str = ""
    detail: str = ""


# A page that refuses automated clients exists but cannot be read; that is inconclusive, not missing.
BLOCKED_STATUSES = frozenset({401, 403, 429, 503})
INCONCLUSIVE = frozenset({"blocked", "no_text"})


@dataclass
class VerificationPlan:
    targets: list[VerifyTarget] = field(default_factory=list)   # to fetch
    trusted: list[VerifyTarget] = field(default_factory=list)   # cited, already backed by a tool/web doc


def _docs(entries: Iterable[dict]) -> list[dict]:
    docs: list[dict] = []
    for retrieval in entries:
        if retrieval.get("retrieval_status") != "FOUND":
            continue
        docs.extend(d for d in (retrieval.get("retrieval_documents") or []) if d is not None)
    return docs


def _context_docs(state: dict, max_retrievals: int) -> list[dict]:
    return _docs((state.get("retrieval") or [])[-max_retrievals:])


def _turn_docs(state: dict) -> list[dict]:
    return _docs(turn_retrievals(state))


def _tool_urls(state: dict) -> set[str]:
    urls: set[str] = set()
    for message in turn_messages(state):
        kind = getattr(message, "type", "")
        if kind == "tool":
            urls.update(normalize_url(u) for u in extract_links(str(getattr(message, "content", ""))))
        elif kind == "ai":
            for call in getattr(message, "tool_calls", None) or []:
                for value in (call.get("args") or {}).values():
                    if isinstance(value, str):
                        urls.update(normalize_url(u) for u in extract_links(value))
    return urls


def build_plan(state: dict, answer: str, max_retrievals: int, max_targets: int = JUDGE_MAX_VERIFY_LINKS) -> VerificationPlan:
    """Decide which links to fetch for this answer and which are already trusted.

    Links are trusted if backed by any web document or tool output still in the window (an answer
    may legitimately reuse an earlier source). Curated entries are fetched uncited only when they
    were retrieved *this turn*; entries left over from earlier turns are ignored unless the answer
    cites them, so a stale entry can never fail or annotate an unrelated question.
    """
    window_docs = _context_docs(state, max_retrievals)
    web_sources = {normalize_url(d["source"]) for d in window_docs if d.get("stage") == RetrievalStage.WEB and str(d.get("source", "")).startswith("http")}
    static_any = {normalize_url(d["source"]): d for d in window_docs if d.get("stage") == RetrievalStage.STATIC and str(d.get("source", "")).startswith("http")}
    static_entries = {normalize_url(d["source"]): d for d in _turn_docs(state) if d.get("stage") == RetrievalStage.STATIC and str(d.get("source", "")).startswith("http")}
    tool_urls = _tool_urls(state)

    plan = VerificationPlan()
    planned: set[str] = set()
    for link in extract_links(answer):
        key = normalize_url(link)
        if key in tool_urls:
            plan.trusted.append(VerifyTarget(link, "tool", cited=True))
        elif key in web_sources:
            plan.trusted.append(VerifyTarget(link, "web", cited=True))
        elif key in static_any:
            plan.targets.append(VerifyTarget(link, "static", cited=True, entry_title=static_any[key].get("title")))
            planned.add(key)
        else:
            plan.targets.append(VerifyTarget(link, "unknown", cited=True))
            planned.add(key)

    # Curated entries retrieved this turn, even when the answer did not cite them.
    for key, doc in static_entries.items():
        if key not in planned:
            plan.targets.append(VerifyTarget(doc["source"], "static", cited=False, entry_title=doc.get("title")))
            planned.add(key)

    if len(plan.targets) > max_targets:
        logger.info("Verification capped at %d of %d links", max_targets, len(plan.targets))
        plan.targets = plan.targets[:max_targets]
    return plan


def _fetch_one(url: str) -> FetchedPage:
    text, meta = fetch_webpage(url, fallback_content="", timeout=JUDGE_VERIFY_TIMEOUT, max_chars=JUDGE_VERIFY_PAGE_CHARS)
    if meta.get("error"):
        return FetchedPage(url, "unreachable", detail=str(meta["error"])[:120])
    code = meta.get("status_code")
    if code and code != 200:
        if code in BLOCKED_STATUSES:
            return FetchedPage(url, "blocked", detail=f"HTTP {code}, page refuses automated clients")
        return FetchedPage(url, "unreachable", detail=f"HTTP {code}")
    if meta.get("content_source") != "webpage" or not text:
        return FetchedPage(url, "no_text", detail=str(meta.get("content_type") or "no readable text"))
    return FetchedPage(url, "ok", text=text)


def fetch_pages(urls: Iterable[str], fetcher: Callable[[str], FetchedPage] = _fetch_one) -> dict[str, FetchedPage]:
    urls = list(dict.fromkeys(urls))
    if not urls:
        return {}
    with ThreadPoolExecutor(max_workers=min(5, len(urls))) as pool:
        pages = list(pool.map(fetcher, urls))
    return {normalize_url(p.url): p for p in pages}


def format_verification(plan: VerificationPlan, pages: dict[str, FetchedPage]) -> str:
    """Section for the judge prompt: each fetched page with role, status and excerpt."""
    if not plan.targets and not plan.trusted:
        return "(the answer cites no links and relies on no curated entries)"

    lines: list[str] = []
    for target in plan.targets:
        page = pages.get(normalize_url(target.url))
        roles = []
        if target.cited:
            roles.append("cited in the answer")
        if target.origin == "static":
            roles.append(f'source of curated entry "{target.entry_title or "?"}"')
        if target.origin == "unknown":
            roles.append("not present in any retrieved source")
        status = page.status if page else "skipped"
        detail = f" ({page.detail})" if page and page.detail else ""
        lines.append(f"- {target.url} | {'; '.join(roles)} | status: {status}{detail}")
        if page and page.text:
            lines.append(f"  excerpt: {page.text}")
    if plan.trusted:
        lines.append("")
        lines.append("Already supported (from tool outputs or fetched web documents; do not doubt these):")
        lines.extend(f"- {t.url}" for t in plan.trusted)
    return "\n".join(lines)


def unreachable_cited_links(plan: VerificationPlan, pages: dict[str, FetchedPage]) -> list[str]:
    """Cited links whose page does not exist, failed to load, or was capped out of verification.

    Blocked and no_text pages are inconclusive and left to the judge's reading of the excerpt.
    """
    bad: list[str] = []
    for target in plan.targets:
        if not target.cited:
            continue
        page = pages.get(normalize_url(target.url))
        if page is None or page.status == "unreachable":
            bad.append(target.url)
    return bad


@dataclass
class UnverifiedSource:
    url: str
    title: str
    status: str
    detail: str = ""


def unverified_sources(plan: VerificationPlan, pages: dict[str, FetchedPage]) -> list[UnverifiedSource]:
    """Curated index entries whose source page does not exist or has no readable content.

    Such an entry cannot back any claim. Blocked pages are excluded: the page exists, it just
    refuses automated clients, so the entry is neither confirmed nor refuted.
    """
    out: list[UnverifiedSource] = []
    for target in plan.targets:
        if target.origin != "static":
            continue
        page = pages.get(normalize_url(target.url))
        if page is None:
            continue
        if page.status in {"unreachable", "no_text"}:
            out.append(UnverifiedSource(target.url, target.entry_title or "?", page.status, page.detail))
    return out


def strip_links(text: str, urls: Iterable[str]) -> tuple[str, int]:
    """Remove the given links from markdown/plain text, keeping link labels. Returns (text, removed)."""
    keys = {normalize_url(u) for u in urls}
    if not keys or not text:
        return text, 0
    removed = 0

    def _md(match: re.Match) -> str:
        nonlocal removed
        if normalize_url(match.group(2)) in keys:
            removed += 1
            return match.group(1)
        return match.group(0)

    def _html(match: re.Match) -> str:
        nonlocal removed
        if normalize_url(match.group(1)) in keys:
            removed += 1
            return match.group(2)
        return match.group(0)

    def _bare(match: re.Match) -> str:
        nonlocal removed
        if normalize_url(match.group(0)) in keys:
            removed += 1
            return ""
        return match.group(0)

    text = _MD_LINK.sub(_md, text)
    text = _HTML_LINK.sub(_html, text)
    text = _URL.sub(_bare, text)
    if removed:
        text = re.sub(r"\(\s*\)", "", text)          # "(https://x)" -> "()" -> ""
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r" +([.,;:])", r"\1", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip(), removed


def unverified_note(count: int) -> str:
    return f"\n\n*Note: {count} reference{'s' if count != 1 else ''} could not be verified and {'were' if count != 1 else 'was'} removed.*"


def unverified_sources_note(state: dict, unverified_urls: Iterable[str]) -> str:
    """Appended when the final answer still rests on curated entries whose page is missing or empty.

    Built from the verification data itself, never from the judge's prose, so nothing unrelated
    to source verification reaches the user. Entry titles come from this turn's retrievals.
    """
    keys = {normalize_url(u) for u in unverified_urls if u}
    if not keys:
        return ""
    titles: list[str] = []
    for doc in _turn_docs(state):
        source = str(doc.get("source", ""))
        if source.startswith("http") and normalize_url(source) in keys:
            title = str(doc.get("title") or "").strip()
            if title and title not in titles:
                titles.append(title)
    if not titles:
        return ""
    listed = "; ".join(f"“{t}”" for t in titles[:3])
    return (f"\n\n*Note: the curated source{'s' if len(titles) != 1 else ''} {listed} could not be verified "
            f"(page missing or empty). Anything above that rests only on {'them' if len(titles) != 1 else 'it'} may be inaccurate.*")
