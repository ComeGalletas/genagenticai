from __future__ import annotations

import logging

from .chroma_store import get_vectorstore, list_vectorstore_collections, load_vectorstore, registered_sources
from .settings import DEFAULT_COLLECTION, RAG_K, RAG_ROUTING_MIN_SCORE, RAG_SCORE_THRESHOLD
from ..retrieval.schemas import RetrievalStage, RetrievalResult

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

K_VALUE = RAG_K
SCORE_THRESHOLD = RAG_SCORE_THRESHOLD
AUTO_COLLECTION = "auto"


def select_knowledge_collection(question: str, collections: list[str] | None = None) -> str:
    """Deterministically route a question to the best-scoring registered collection."""
    available = collections if collections is not None else list_knowledge_collections()
    if not available:
        return DEFAULT_COLLECTION

    fallback = DEFAULT_COLLECTION if DEFAULT_COLLECTION in available else available[0]
    scored = sorted(
        ((source.score(question), source.name) for source in registered_sources() if source.name in available),
        reverse=True,
    )
    best_score, best_name = scored[0] if scored else (0, fallback)

    selected = best_name if best_score >= RAG_ROUTING_MIN_SCORE else fallback
    logger.info(
        "Deterministic selector chose %s | best=%s score=%d | collections=%s",
        selected,
        best_name,
        best_score,
        available,
    )
    return selected


def list_knowledge_collections() -> list[str]:
    """Return the persisted Chroma collection names available for deterministic RAG queries."""
    return list_vectorstore_collections()


def query_knowledge(
    question: str,
    k: int = K_VALUE,
    score_threshold: float = SCORE_THRESHOLD,
    collection: str = DEFAULT_COLLECTION,
) -> list[RetrievalResult]:
    """Return relevant passages from one local vector store collection.
    Args:
        question: The query string to search for.
        k: The number of top results to return.
        score_threshold: The minimum score required for a result to be included.
        collection: The Chroma collection name to query.
    Returns:
        A list of RetrievalResult objects containing the relevant passages.
    """
    selected_collection = select_knowledge_collection(question) if collection == AUTO_COLLECTION else collection
    logger.info("RAG query: %r | collection=%s", question, selected_collection)

    vs = get_vectorstore(selected_collection)
    if vs is None:
        vs = load_vectorstore(selected_collection)
    if vs is None:
        logger.warning("Vector store collection %s not loaded.", selected_collection)
        return []

    try:
        logger.info("Similarity search collection=%s k=%d, threshold=%.2f", selected_collection, k, score_threshold)
        results = vs.similarity_search_with_relevance_scores(question, k=k)
        logger.info("Similarity search raw results for %s: %d doc(s) returned", selected_collection, len(results))
    except Exception:
        logger.exception("Similarity with relevance search failed")
        return []

    retrieval_results: list[RetrievalResult] = []
    for doc, score in results:
        logger.info("Score: %.3f | Source: %s", score, doc.metadata.get("filename", "unknown"))
        if score > score_threshold:
            metadata = {**doc.metadata, "collection": selected_collection}
            retrieval_results.append(RetrievalResult(
                title=doc.metadata.get("title", "Untitled"),
                content=doc.page_content,
                source=doc.metadata.get("filename", doc.metadata.get("source", "Unknown")),
                stage=RetrievalStage.RAG,
                confidence=round(score, 3),
                metadata=metadata,
            ))

    logger.info("Returning %d relevant chunk(s)", len(retrieval_results))

    #for result in retrieval_results:
    #    logger.info("Retrieved chunk: %s", str(result.content))

    return retrieval_results