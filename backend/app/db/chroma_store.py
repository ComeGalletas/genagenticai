from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_CHROMA_DIR = _DATA_DIR / "chroma_db"
_KNOWLEDGE_DIR = _DATA_DIR / "knowledge"

SIMPLE_KNOWLEDGE_DIR = "data/knowledge"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60

_embeddings = OllamaEmbeddings(model="nomic-embed-text")
#_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # TODO: make this configurable
_vectorstore: Chroma | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _open_existing() -> Chroma:
    return Chroma(
        persist_directory=str(_CHROMA_DIR),
        embedding_function=_embeddings,
        collection_name="knowledge",
    )


def _build_from_knowledge() -> Chroma:
    """Embed all .txt/.md files in the knowledge directory and persist to ChromaDB."""
    if not _KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge directory not found at {_KNOWLEDGE_DIR}. "
            "Add .txt or .md files before building the vector store."
        )
    
    docs = []
    for file in _KNOWLEDGE_DIR.rglob("*"):
        if file.suffix.lower() not in {".txt", ".md"}:
            continue
        try:
            loaded = TextLoader(str(file), encoding="utf-8").load()
        except Exception as exc:
            logger.warning("Skipping %s: %s", file, exc)
            continue
        for doc in loaded:
            doc.metadata = {
                "category": file.parent.name,
                "filename": file.name
            }
        docs.extend(loaded)

    if not docs:
        raise ValueError(
            "No documents found in the knowledge directory. "
            "Add .txt or .md files to backend/data/knowledge/."
        )
    else:
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("##", "product")
            ]
        )

        chunks = []
        for doc in docs:
            split_docs = splitter.split_text(doc.page_content)

            for chunk in split_docs:
                chunk.metadata = {
                    **doc.metadata,
                    **chunk.metadata
                }

                chunks.append(chunk)


    chunks_2 = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    ).split_documents(docs)

    logger.info("Building vector store — %d chunk(s) from %d file(s)", len(chunks), len(docs))
    return Chroma.from_documents(
        chunks,
        embedding=_embeddings,
        persist_directory=str(_CHROMA_DIR),
        collection_name="knowledge",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_vectorstore() -> Chroma | None:
    """Return the in-memory singleton. None if not yet loaded."""
    return _vectorstore


def load_vectorstore() -> Chroma | None:
    """
    Load the existing ChromaDB into the singleton.
    If the database does not exist, logs a warning and returns None — the
    application continues running without a knowledge base.
    Call load_or_build_vectorstore() to create the DB from knowledge files.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    if not _CHROMA_DIR.exists():
        logger.warning(
            "ChromaDB not found at %s — knowledge base unavailable. "
            "Run load_or_build_vectorstore() to create it.",
            _CHROMA_DIR,
        )
        return None

    try:
        _vectorstore = _open_existing()
        logger.info("Vector store loaded from %s", _CHROMA_DIR)
        return _vectorstore
    except Exception as exc:
        logger.error("Failed to load ChromaDB: %s", exc)
        return None


def load_or_build_vectorstore() -> Chroma:
    """
    On-demand loader: returns the singleton if already loaded, tries to open an
    existing ChromaDB, and falls back to building one from the knowledge files.
    Raises on unrecoverable errors.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    if _CHROMA_DIR.exists():
        try:
            _vectorstore = _open_existing()
            logger.info("Vector store loaded from %s", _CHROMA_DIR)
            return _vectorstore
        except Exception as exc:
            logger.warning("Could not open existing ChromaDB (%s) — rebuilding.", exc)

    _vectorstore = _build_from_knowledge()
    logger.info("Vector store built and saved to %s", _CHROMA_DIR)
    return _vectorstore


def rebuild_vectorstore() -> Chroma:
    """
    Drop the current singleton and ChromaDB on disk, then rebuild from the
    knowledge files. Call this after adding or editing .md / .txt files.
    """
    global _vectorstore
    import shutil

    if _vectorstore is not None:
        try:
            _vectorstore._client.reset()  # release Chroma's file handles
        except Exception:
            pass
        _vectorstore = None

    if _CHROMA_DIR.exists():
        shutil.rmtree(_CHROMA_DIR)
        logger.info("Deleted old ChromaDB at %s", _CHROMA_DIR)

    _vectorstore = _build_from_knowledge()
    logger.info("Vector store rebuilt and saved to %s", _CHROMA_DIR)
    return _vectorstore
