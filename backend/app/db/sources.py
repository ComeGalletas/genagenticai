from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from .baloto import BalotoSource
from .chroma_store import register_source
from .settings import DEFAULT_COLLECTION, KNOWLEDGE_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KnowledgeSource:
    """Markdown/text files from the knowledge directory, split on level-2 headers."""

    name: str = DEFAULT_COLLECTION

    def load(self) -> list[Document]:
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "product")])

        chunks: list[Document] = []
        for file in KNOWLEDGE_DIR.rglob("*"):
            if file.suffix.lower() not in {".txt", ".md"}:
                continue

            try:
                loaded = TextLoader(str(file), encoding="utf-8").load()
            except Exception as exc:
                logger.warning("Skipping %s: %s", file, exc)
                continue

            metadata = {"category": file.parent.name, "filename": file.name}
            for doc in loaded:
                for chunk in splitter.split_text(doc.page_content):
                    chunk.metadata = {**metadata, **chunk.metadata}
                    chunks.append(chunk)

        if not chunks:
            raise ValueError(f"No .txt or .md documents found in {KNOWLEDGE_DIR}.")
        return chunks

    def score(self, question: str) -> int:
        return 0


register_source(KnowledgeSource())
register_source(BalotoSource())
