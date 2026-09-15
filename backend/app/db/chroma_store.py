from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from ..config import OLLAMA_BASE_URL
from .settings import CHROMA_DIR, DEFAULT_COLLECTION, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
_vectorstores: dict[str, Chroma] = {}
_sources: dict[str, "CollectionSource"] = {}


@runtime_checkable
class CollectionSource(Protocol):
    """One named Chroma collection: how to build it and how well it fits a question."""

    name: str

    def load(self) -> list[Document]:
        """Return the documents to index for this collection."""
        ...

    def score(self, question: str) -> int:
        """Routing score for a question; 0 means "no opinion"."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def register_source(source: CollectionSource) -> None:
    """Make a collection buildable and routable. Call once per source at import time."""
    _sources[source.name] = source


def registered_sources() -> list[CollectionSource]:
    return list(_sources.values())


def registered_collections() -> list[str]:
    return sorted(_sources)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_client() -> Any:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _open_existing(collection: str) -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=_embeddings,
        collection_name=collection,
    )


def _build_collection(collection: str) -> Chroma:
    source = _sources.get(collection)
    if source is None:
        raise ValueError(
            f"No source registered for collection {collection!r}. Registered: {registered_collections()}"
        )

    chunks = source.load()
    if not chunks:
        raise ValueError(f"Source {collection!r} produced no documents.")

    logger.info("Building %s collection — %d chunk(s)", collection, len(chunks))
    return Chroma.from_documents(
        chunks,
        embedding=_embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=collection,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_vectorstore(collection: str = DEFAULT_COLLECTION) -> Chroma | None:
    """Return an already-loaded collection without any I/O, or None."""
    return _vectorstores.get(collection)


def load_vectorstore(collection: str = DEFAULT_COLLECTION) -> Chroma | None:
    """Load a persisted collection into the cache. Returns None if it does not exist yet."""
    if collection in _vectorstores:
        return _vectorstores[collection]

    if not CHROMA_DIR.exists():
        logger.warning("ChromaDB not found at %s — collection %s unavailable.", CHROMA_DIR, collection)
        return None

    try:
        if collection not in {item.name for item in _get_client().list_collections()}:
            logger.warning("Collection %s not found in ChromaDB at %s", collection, CHROMA_DIR)
            return None

        _vectorstores[collection] = _open_existing(collection)
        logger.info("Vector store collection %s loaded from %s", collection, CHROMA_DIR)
        return _vectorstores[collection]
    except Exception as exc:
        logger.error("Failed to load ChromaDB collection %s: %s", collection, exc)
        return None


def load_or_build_vectorstore(collection: str = DEFAULT_COLLECTION) -> Chroma:
    """Return a collection, loading it from disk or building it from its registered source."""
    loaded = load_vectorstore(collection)
    if loaded is not None:
        return loaded

    _vectorstores[collection] = _build_collection(collection)
    logger.info("Vector store collection %s built and saved to %s", collection, CHROMA_DIR)
    return _vectorstores[collection]


def list_vectorstore_collections() -> list[str]:
    """Return the collection names currently persisted on disk."""
    if not CHROMA_DIR.exists():
        return []

    try:
        return sorted(item.name for item in _get_client().list_collections())
    except Exception as exc:
        logger.error("Failed to list ChromaDB collections: %s", exc)
        return []


def rebuild_vectorstore(collection: str | None = None) -> Chroma:
    """Rebuild one registered collection, or every registered collection when None."""
    targets = tuple(registered_collections()) if collection is None else (collection,)
    if not targets:
        raise ValueError("No collection sources registered.")

    for target in targets:
        cached = _vectorstores.pop(target, None)
        if cached is not None:
            try:
                cached._client.reset()  # release Chroma file handles
            except Exception:
                pass

    if CHROMA_DIR.exists():
        client = _get_client()
        for target in targets:
            try:
                client.delete_collection(name=target)
                logger.info("Deleted old ChromaDB collection %s at %s", target, CHROMA_DIR)
            except Exception:
                logger.debug("Collection %s did not exist before rebuild", target)

    for target in targets:
        _vectorstores[target] = _build_collection(target)
        logger.info("Vector store collection %s rebuilt and saved to %s", target, CHROMA_DIR)

    return _vectorstores[targets[0]]
