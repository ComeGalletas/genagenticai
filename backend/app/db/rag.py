from __future__ import annotations

import logging

from .chroma_store import get_vectorstore
from ..retrieval.schemas import RetrievalStage, RetrievalResult

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

K_VALUE = 3
SCORE_THRESHOLD = 0.45


def query_knowledge(question: str, k: int = K_VALUE, score_threshold: float = SCORE_THRESHOLD) -> list[RetrievalResult]:
    """Return relevant passages from the local vector store."""
    logger.debug("RAG query: %r", question)

    vs = get_vectorstore()
    if vs is None:
        logger.warning("Vector store not loaded.")
        return []

    try:
        logger.debug("Similarity search question: %r (k=%d, threshold=%.2f)", question, k, score_threshold)
        results = vs.similarity_search_with_relevance_scores(question, k=k)
        logger.debug("Similarity search raw results: %d doc(s) returned", len(results))
        #results = vs.similarity_search(question, k=k)
    except Exception:
        logger.exception("Similarity search failed")
        return []

    retrieval_results: list[RetrievalResult] = []
    for doc, score in results:
        logger.debug("Score: %.3f | Source: %s", score, doc.metadata.get("filename", "unknown"))
        if score < score_threshold:
            continue
        retrieval_results.append(RetrievalResult(
            title=doc.metadata.get("title", "Untitled"),
            content=doc.page_content,
            source=doc.metadata.get("filename", doc.metadata.get("source", "Unknown")),
            stage=RetrievalStage.RAG,
            score=score,
            confidence=score,
            metadata=doc.metadata,
        ))

    logger.info("Returning %d relevant chunk(s)", len(retrieval_results))
    return retrieval_results