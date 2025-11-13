"""
RAG retrieval system using vector similarity search.

This module provides similarity-based retrieval of forecasting examples
using sentence embeddings and approximate nearest neighbor search.

Uses sentence-transformers for embedding generation and simple cosine
similarity for retrieval (can be upgraded to ChromaDB/FAISS for scale).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pickle
import os

# Try to import sentence-transformers, fallback to simple TF-IDF
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

from py_agent.knowledge.example_library import ExampleLibrary, ForecastingExample


@dataclass
class RetrievalResult:
    """
    Result from RAG retrieval.

    Attributes:
        example: The retrieved forecasting example
        similarity_score: Similarity score (0-1, higher is more similar)
        rank: Rank in retrieval results (1 = most similar)
    """
    example: ForecastingExample
    similarity_score: float
    rank: int


class RAGRetriever:
    """
    Retrieval-Augmented Generation system for forecasting examples.

    Embeds forecasting examples and user queries into vector space,
    then retrieves most similar examples using cosine similarity.
    """

    def __init__(
        self,
        example_library: ExampleLibrary,
        model_name: str = 'all-MiniLM-L6-v2',
        cache_embeddings: bool = True
    ):
        """
        Initialize RAG retriever.

        Args:
            example_library: Library of forecasting examples
            model_name: Sentence transformer model name (if available)
            cache_embeddings: Whether to cache embeddings to disk
        """
        self.example_library = example_library
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings

        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer(model_name)
            self.use_sentence_transformers = True
        else:
            # Fallback to TF-IDF
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            self.use_sentence_transformers = False

        # Embeddings cache
        self.example_embeddings: Optional[np.ndarray] = None
        self.embeddings_cache_path = os.path.join(
            os.path.dirname(__file__),
            'embeddings_cache.pkl'
        )

        # Build embeddings for all examples
        self._build_embeddings()

    def _build_embeddings(self) -> None:
        """Build embeddings for all examples in library."""
        # Try to load from cache
        if self.cache_embeddings and os.path.exists(self.embeddings_cache_path):
            try:
                with open(self.embeddings_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)

                # Verify cache is for same examples
                if cache_data['num_examples'] == len(self.example_library):
                    self.example_embeddings = cache_data['embeddings']
                    if not self.use_sentence_transformers:
                        self.vectorizer = cache_data['vectorizer']
                    return
            except Exception:
                pass  # Cache invalid, rebuild

        # Build embeddings
        texts = [ex.get_embedding_text() for ex in self.example_library.examples]

        if self.use_sentence_transformers:
            # Use sentence transformers
            self.example_embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
        else:
            # Use TF-IDF
            self.example_embeddings = self.vectorizer.fit_transform(texts).toarray()

        # Cache embeddings
        if self.cache_embeddings:
            cache_data = {
                'num_examples': len(self.example_library),
                'embeddings': self.example_embeddings
            }
            if not self.use_sentence_transformers:
                cache_data['vectorizer'] = self.vectorizer

            with open(self.embeddings_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        domain_filter: Optional[str] = None,
        difficulty_filter: Optional[str] = None,
        min_similarity: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Retrieve most similar examples for a query.

        Args:
            query: User's problem description or data characteristics
            top_k: Number of examples to retrieve (default: 3)
            domain_filter: Filter by domain (e.g., 'retail', 'finance')
            difficulty_filter: Filter by difficulty ('easy', 'medium', 'hard')
            min_similarity: Minimum similarity score (0-1)

        Returns:
            List of RetrievalResult objects sorted by similarity

        Example:
            >>> retriever = RAGRetriever(example_library)
            >>> results = retriever.retrieve(
            ...     "Daily sales data with weekly seasonality",
            ...     top_k=3
            ... )
            >>> for result in results:
            ...     print(f"{result.rank}. {result.example.title} (score: {result.similarity_score:.3f})")
        """
        # Embed query
        if self.use_sentence_transformers:
            query_embedding = self.embedding_model.encode(
                [query],
                show_progress_bar=False,
                convert_to_numpy=True
            )[0]
        else:
            query_embedding = self.vectorizer.transform([query]).toarray()[0]

        # Calculate similarities
        similarities = self._cosine_similarity(
            query_embedding,
            self.example_embeddings
        )

        # Apply filters
        filtered_indices = []
        for idx in range(len(self.example_library)):
            example = self.example_library[idx]

            # Domain filter
            if domain_filter and example.domain != domain_filter:
                continue

            # Difficulty filter
            if difficulty_filter and example.difficulty != difficulty_filter:
                continue

            # Similarity threshold
            if similarities[idx] < min_similarity:
                continue

            filtered_indices.append(idx)

        # Sort by similarity
        if not filtered_indices:
            return []

        filtered_similarities = similarities[filtered_indices]
        sorted_indices = np.argsort(filtered_similarities)[::-1][:top_k]

        # Create results
        results = []
        for rank, idx in enumerate(sorted_indices, 1):
            original_idx = filtered_indices[idx]
            results.append(RetrievalResult(
                example=self.example_library[original_idx],
                similarity_score=float(similarities[original_idx]),
                rank=rank
            ))

        return results

    def retrieve_by_data_characteristics(
        self,
        data_chars: Dict,
        top_k: int = 3
    ) -> List[RetrievalResult]:
        """
        Retrieve examples based on data characteristics.

        Constructs a query from data characteristics and retrieves similar examples.

        Args:
            data_chars: Dict with frequency, seasonality, trend, etc.
            top_k: Number of examples to retrieve

        Returns:
            List of RetrievalResult objects

        Example:
            >>> data_chars = {
            ...     'frequency': 'daily',
            ...     'seasonality': {'detected': True, 'strength': 0.8},
            ...     'trend': {'direction': 'increasing', 'strength': 0.5}
            ... }
            >>> results = retriever.retrieve_by_data_characteristics(data_chars)
        """
        # Construct query from data characteristics
        query_parts = []

        if 'frequency' in data_chars:
            query_parts.append(f"Frequency: {data_chars['frequency']}")

        if 'seasonality' in data_chars:
            seasonality = data_chars['seasonality']
            if isinstance(seasonality, dict):
                if seasonality.get('detected'):
                    strength = seasonality.get('strength', 0.5)
                    query_parts.append(f"Strong seasonality (strength={strength:.2f})")
                else:
                    query_parts.append("No seasonality")
            else:
                query_parts.append(f"Seasonality: {seasonality}")

        if 'trend' in data_chars:
            trend = data_chars['trend']
            if isinstance(trend, dict):
                direction = trend.get('direction', 'unknown')
                query_parts.append(f"Trend: {direction}")
            else:
                query_parts.append(f"Trend: {trend}")

        if 'n_observations' in data_chars:
            n_obs = data_chars['n_observations']
            if n_obs < 50:
                query_parts.append("Limited data (<50 observations)")
            elif n_obs > 1000:
                query_parts.append("Large dataset (>1000 observations)")

        if 'n_features' in data_chars:
            n_features = data_chars['n_features']
            if n_features > 20:
                query_parts.append("High-dimensional data (>20 features)")

        query = ". ".join(query_parts)
        return self.retrieve(query, top_k=top_k)

    def get_model_recommendations_from_examples(
        self,
        results: List[RetrievalResult],
        top_n: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Extract model recommendations from retrieved examples.

        Aggregates recommended models from similar examples,
        weighted by similarity scores.

        Args:
            results: List of RetrievalResult objects
            top_n: Number of top models to return

        Returns:
            List of (model_type, confidence_score) tuples

        Example:
            >>> results = retriever.retrieve("Daily sales with seasonality")
            >>> models = retriever.get_model_recommendations_from_examples(results)
            >>> print(models)
            [('prophet_reg', 0.85), ('arima_reg', 0.72), ('linear_reg', 0.68)]
        """
        # Count model occurrences weighted by similarity
        model_scores = {}

        for result in results:
            for model in result.example.recommended_models:
                if model not in model_scores:
                    model_scores[model] = 0.0
                model_scores[model] += result.similarity_score

        # Normalize scores
        if model_scores:
            max_score = max(model_scores.values())
            model_scores = {m: s / max_score for m, s in model_scores.items()}

        # Sort by score
        sorted_models = sorted(
            model_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_models[:top_n]

    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between vectors.

        Args:
            vec1: Query vector (1D)
            vec2: Example vectors (2D, shape: [n_examples, embedding_dim])

        Returns:
            Similarity scores (1D, shape: [n_examples])
        """
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + 1e-8)

        # Compute dot product
        similarities = np.dot(vec2_norm, vec1_norm)

        return similarities

    def get_preprocessing_insights(
        self,
        results: List[RetrievalResult]
    ) -> List[str]:
        """
        Extract preprocessing insights from retrieved examples.

        Args:
            results: List of RetrievalResult objects

        Returns:
            List of preprocessing recommendations

        Example:
            >>> results = retriever.retrieve("Hourly energy data")
            >>> insights = retriever.get_preprocessing_insights(results)
            >>> for insight in insights:
            ...     print(f"- {insight}")
        """
        insights = []

        for result in results:
            strategy = result.example.preprocessing_strategy
            if strategy and strategy not in insights:
                insights.append(strategy)

        return insights

    def get_key_lessons(
        self,
        results: List[RetrievalResult]
    ) -> List[str]:
        """
        Extract key lessons from retrieved examples.

        Args:
            results: List of RetrievalResult objects

        Returns:
            List of key lessons learned

        Example:
            >>> results = retriever.retrieve("Limited data, yearly")
            >>> lessons = retriever.get_key_lessons(results)
            >>> for lesson in lessons:
            ...     print(f"💡 {lesson}")
        """
        lessons = []

        for result in results:
            for lesson in result.example.key_lessons:
                if lesson not in lessons:
                    lessons.append(lesson)

        return lessons
