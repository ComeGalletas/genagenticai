from ..searches.rag import query_knowledge
from ..searches.google_search import query_ddu_google_search
from ..searches.static_search import query_static_search

RETRIEVAL_PIPELINE = [
    query_static_search,
    query_knowledge,
    query_ddu_google_search,
]