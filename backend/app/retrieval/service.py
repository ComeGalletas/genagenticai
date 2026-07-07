from .engine import RetrievalEngine
from .config import RETRIEVAL_PIPELINE

"""Singleton instance for the engine to be used throughout the application."""
retrieval_engine = RetrievalEngine(RETRIEVAL_PIPELINE)