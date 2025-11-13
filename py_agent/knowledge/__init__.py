"""
Knowledge module for RAG (Retrieval-Augmented Generation).

Provides example-based recommendations through similarity search
over a library of forecasting scenarios.
"""

from py_agent.knowledge.example_library import (
    ForecastingExample,
    ExampleLibrary,
    create_foundational_examples,
    DEFAULT_LIBRARY_PATH
)
from py_agent.knowledge.rag_retrieval import (
    RAGRetriever,
    RetrievalResult
)

__all__ = [
    'ForecastingExample',
    'ExampleLibrary',
    'create_foundational_examples',
    'DEFAULT_LIBRARY_PATH',
    'RAGRetriever',
    'RetrievalResult',
]
