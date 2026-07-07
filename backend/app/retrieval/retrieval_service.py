from .retrieval_engine import RetrievalEngine
from .retrieval_config import RETRIEVAL_PIPELINE

"""Singleton instance for the engine to be used throughout the application."""
retrieval_engine = RetrievalEngine(RETRIEVAL_PIPELINE)