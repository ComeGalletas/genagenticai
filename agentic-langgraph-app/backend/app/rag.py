from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent / "data"
_CHROMA_DIR = _DATA_DIR / "chroma_db"
_KNOWLEDGE_DIR = _DATA_DIR / "knowledge"

# ---------------------------------------------------------------------------
# Embeddings (uses Ollama — run `ollama pull nomic-embed-text` once)
# ---------------------------------------------------------------------------

_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Module-level singleton so the DB is only opened once per process
_vectorstore: Chroma | None = None


def _build_vectorstore() -> Chroma:
    """Load the knowledge directory, chunk all files, embed them, and persist to ChromaDB."""
    if not _KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge dir not found at {_KNOWLEDGE_DIR}. "
        )

    logger.info("Building vector store from %s", _KNOWLEDGE_DIR)
    docs = []

    for file in Path(_KNOWLEDGE_DIR).rglob("*"):
        if file.suffix in [".txt", ".md"]:
            try:
                loader = TextLoader(str(file), encoding="utf-8")
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["category"] = file.parent.name
                    doc.metadata["filename"] = file.name
                docs.extend(loaded_docs)
                logger.debug("Loaded %d doc(s) from %s", len(loaded_docs), file.name)
            except Exception as exc:
                logger.warning("Skipping %s — could not load: %s", file, exc)

    try:
        loader = TextLoader(str(_KNOWLEDGE_DIR), encoding="utf-8")
        docs.extend(loader.load())
    except Exception as exc:
        logger.warning("Skipping directory-level TextLoader — %s", exc)

    try:
        gpu_loader = TextLoader(str(_DATA_DIR / "gpu_knowledge.md"), encoding="utf-8")
        gpu_docs = gpu_loader.load()
        docs.extend(gpu_docs)
        logger.debug("Loaded %d doc(s) from gpu_knowledge.md", len(gpu_docs))
    except FileNotFoundError:
        logger.warning("gpu_knowledge.md not found — skipping.")
    except Exception as exc:
        logger.error("Failed to load gpu_knowledge.md: %s", exc)

    if not docs:
        raise ValueError(
            "No documents were loaded from the knowledge directory. "
            "Add .txt or .md files to backend/data/knowledge/."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
    )
    chunks = splitter.split_documents(docs)
    logger.info("Splitting produced %d chunks — embedding and persisting", len(chunks))

    try:
        store = Chroma.from_documents(
            documents=chunks,
            embedding=_embeddings,
            persist_directory=str(_CHROMA_DIR),
            collection_name="knowledge",
        )
        logger.info("Vector store built and saved to %s", _CHROMA_DIR)
        return store
    except Exception as exc:
        logger.error("Failed to build Chroma vector store: %s", exc)
        raise


def get_vectorstore() -> Chroma:
    """Return the singleton vector store, building it on first call."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    if _CHROMA_DIR.exists():
        logger.info("Loading existing vector store from %s", _CHROMA_DIR)
        try:
            _vectorstore = Chroma(
                persist_directory=str(_CHROMA_DIR),
                embedding_function=_embeddings,
                collection_name="knowledge",
            )
            logger.info("Vector store loaded successfully")
        except Exception as exc:
            logger.error(
                "Failed to load existing vector store, rebuilding from scratch: %s", exc
            )
            _vectorstore = _build_vectorstore()
    else:
        logger.info("No existing vector store found — building from knowledge files")
        _vectorstore = _build_vectorstore()

    return _vectorstore


def query_knowledge(question: str, k: int = 3) -> str:
    """Return the top-k most relevant passages from the local knowledge base."""
    logger.debug("RAG query: %r", question)
    try:
        vs = get_vectorstore()
    except FileNotFoundError as exc:
        logger.warning("Knowledge directory missing: %s", exc)
        return str(exc)
    except Exception as exc:
        logger.error("Could not load vector store: %s", exc)
        return f"Knowledge base unavailable: {exc}"

    try:
        results = vs.similarity_search(question, k=k)
    except Exception as exc:
        logger.error("Similarity search failed: %s", exc)
        return f"Search failed: {exc}"

    if not results:
        logger.info("No results found for query: %r", question)
        return "No relevant information found in the knowledge base."

    logger.info("Returning %d chunk(s) for query: %r", len(results), question)
    return "\n\n---\n\n".join(doc.page_content for doc in results)
