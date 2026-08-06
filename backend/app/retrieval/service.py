from .engine import RetrievalEngine
from .config import RETRIEVAL_PIPELINE

"""Singleton instance for the engine to be used throughout the application.
    Args:
        RETRIEVAL_PIPELINE: Simple list with the retrieval pipeline functions available to use. The order corresponds to the stages of retrieval.
        It is recommended to keep low latency sources first, and more expensive sources later in the pipeline.
"""
retrieval_engine = RetrievalEngine(RETRIEVAL_PIPELINE)