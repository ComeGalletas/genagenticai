"""Deterministic reply formatting: markdown from the LLM -> sanitized HTML for the frontend.

Doing this in code instead of asking an LLM to emit HTML removes a whole generation pass and
guarantees well-formed markup. The frontend sanitizes again with DOMPurify; this layer is
defense in depth and also keeps the API response clean for any other client.
"""
from __future__ import annotations

import html
import re
from html.parser import HTMLParser

from markdown_it import MarkdownIt

# html=True lets any HTML the model still emits pass through the parser; the sanitizer below
# decides what survives. Tables and strikethrough are GFM extras the models use often.
_md = MarkdownIt("commonmark", {"html": True, "breaks": False}).enable("table").enable("strikethrough")

# Thinking models occasionally leak their scratchpad into the visible content.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

ALLOWED_TAGS = frozenset({
    "p", "br", "hr", "strong", "b", "em", "i", "u", "s", "del", "sup", "sub", "span", "div",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "blockquote", "code", "pre",
    "a", "table", "thead", "tbody", "tr", "th", "td",
})
ALLOWED_ATTRS: dict[str, frozenset[str]] = {
    "a": frozenset({"href", "title"}),
    "ol": frozenset({"start"}),
    "th": frozenset({"align"}),
    "td": frozenset({"align"}),
    "code": frozenset({"class"}),  # language-xyz from fenced blocks
}
VOID_TAGS = frozenset({"br", "hr"})
# Contents of these are dropped entirely, not just unwrapped.
DROP_CONTENT_TAGS = frozenset({"script", "style", "iframe", "object", "embed", "noscript", "template"})
_SAFE_HREF = re.compile(r"^(https?:|mailto:|#|/)", re.IGNORECASE)


class _Sanitizer(HTMLParser):
    """Allowlist HTML filter. Unknown tags are unwrapped (text kept); dangerous ones are dropped."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.open: list[str] = []
        self._drop_depth = 0

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _clean_attrs(tag: str, attrs: list[tuple[str, str | None]]) -> str:
        allowed = ALLOWED_ATTRS.get(tag, frozenset())
        parts: list[str] = []
        for name, value in attrs:
            name = name.lower()
            if name not in allowed or value is None:
                continue
            value = value.strip()
            if name == "href" and not _SAFE_HREF.match(value):
                continue
            parts.append(f' {name}="{html.escape(value, quote=True)}"')
        return "".join(parts)

    # -- parser callbacks --------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self._drop_depth:
            if tag in DROP_CONTENT_TAGS and tag not in VOID_TAGS:
                self._drop_depth += 1
            return
        if tag in DROP_CONTENT_TAGS:
            self._drop_depth = 1
            return
        if tag not in ALLOWED_TAGS:
            return  # unwrap: keep children, drop the tag itself
        if tag in VOID_TAGS:
            self.out.append(f"<{tag}>")
            return
        self.out.append(f"<{tag}{self._clean_attrs(tag, attrs)}>")
        self.open.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in VOID_TAGS and not self._drop_depth:
            self.out.append(f"<{tag}>")
        elif tag in ALLOWED_TAGS and not self._drop_depth:
            self.out.append(f"<{tag}{self._clean_attrs(tag, attrs)}></{tag}>")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._drop_depth:
            if tag in DROP_CONTENT_TAGS:
                self._drop_depth -= 1
            return
        if tag not in ALLOWED_TAGS or tag in VOID_TAGS or tag not in self.open:
            return
        # Close everything opened after this tag so the output stays balanced.
        while self.open:
            opened = self.open.pop()
            self.out.append(f"</{opened}>")
            if opened == tag:
                break

    def handle_data(self, data: str) -> None:
        if not self._drop_depth:
            self.out.append(html.escape(data, quote=False))

    def handle_comment(self, data: str) -> None:
        return  # comments never reach the client

    def result(self) -> str:
        while self.open:
            self.out.append(f"</{self.open.pop()}>")
        return "".join(self.out)


def sanitize_html(raw: str) -> str:
    """Keep only allowlisted tags and attributes; drop scripts, styles, handlers and unsafe URLs."""
    parser = _Sanitizer()
    parser.feed(raw)
    parser.close()
    return parser.result()


def markdown_to_html(text: str) -> str:
    """Render markdown (with tables and strikethrough) to HTML. Raw HTML in the input passes through unsanitized."""
    return _md.render(text)


def render_reply(text: str) -> str:
    """Turn the chatbot's final text into clean, safe HTML for the client."""
    cleaned = _THINK_BLOCK.sub("", text or "").strip()
    if not cleaned:
        return ""
    return sanitize_html(markdown_to_html(cleaned)).strip()
