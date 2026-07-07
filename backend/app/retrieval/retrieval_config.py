from ..db.rag import query_knowledge
from ..search.google import query_ddu_google_search
from ..search.static import query_static_search

RETRIEVAL_PIPELINE = [
    query_static_search,
    query_knowledge,
    query_ddu_google_search,
]