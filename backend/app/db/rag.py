from __future__ import annotations

import logging

from .chroma_store import get_vectorstore
from ..retrieval.schemas import RetrievalStage, RetrievalResult

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

K_VALUE = 3 # Top 3 values retrieved from the vector store, can be adjusted based on the use case
SCORE_THRESHOLD = 0.62 # Minimum score threshold for a result to be considered relevant, can be adjusted based on the use case


def query_knowledge(question: str, k: int = K_VALUE, score_threshold: float = SCORE_THRESHOLD) -> list[RetrievalResult]:
    """Return relevant passages from the local vector store.
    Args:
        question: The query string to search for.
        k: The number of top results to return.
        score_threshold: The minimum score required for a result to be included.
    Returns:
        A list of RetrievalResult objects containing the relevant passages.
    """
    logger.info("RAG query: %r", question)

    vs = get_vectorstore()
    if vs is None:
        logger.warning("Vector store not loaded.")
        return []

    try:
        logger.info("Similarity search question k=%d, threshold=%.2f", k, score_threshold)
        results = vs.similarity_search_with_relevance_scores(question, k=k)
        logger.debug("Similarity search raw results: %d doc(s) returned", len(results))
    except Exception:
        logger.exception("Similarity with relevance search failed")
        return []

    retrieval_results: list[RetrievalResult] = []
    for doc, score in results:
        logger.debug("Score: %.3f | Source: %s", score, doc.metadata.get("filename", "unknown"))
        if score > score_threshold:
            retrieval_results.append(RetrievalResult(
                title=doc.metadata.get("title", "Untitled"),
                content=doc.page_content,
                source=doc.metadata.get("filename", doc.metadata.get("source", "Unknown")),
                stage=RetrievalStage.RAG,
                confidence=round(score, 3),
                metadata=doc.metadata,
            ))

    logger.info("Returning %d relevant chunk(s)", len(retrieval_results))
    return retrieval_results