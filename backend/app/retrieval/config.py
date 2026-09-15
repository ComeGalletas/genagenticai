from ..db.rag import query_knowledge
from ..search.ddgo import query_web_search
from ..search.static import query_static_search

RETRIEVAL_PIPELINE = [
    query_static_search,
    query_knowledge,
    query_web_search,
]