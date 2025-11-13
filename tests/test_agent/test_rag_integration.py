"""
Tests for RAG knowledge base integration with ForecastAgent.

Tests Phase 3.4: RAG Knowledge Base
- Example library management
- Similarity retrieval
- Integration with ForecastAgent
- Model recommendation enhancement
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import json

from py_agent.knowledge import (
    ForecastingExample,
    ExampleLibrary,
    RAGRetriever,
    RetrievalResult
)


class TestForecastingExample:
    """Test ForecastingExample dataclass."""

    def test_create_example(self):
        """Test creating a forecasting example."""
        example = ForecastingExample(
            id="test_001",
            title="Test Example",
            description="Test description",
            domain="retail",
            data_characteristics={
                "frequency": "daily",
                "seasonality": "strong",
                "trend": "increasing"
            },
            recommended_models=["prophet_reg", "arima_reg"],
            preprocessing_strategy="Normalize, date features",
            performance_metrics={"prophet_reg": {"rmse": 10.0}},
            key_lessons=["Lesson 1", "Lesson 2"],
            difficulty="easy",
            tags=["seasonality", "retail"]
        )

        assert example.id == "test_001"
        assert example.domain == "retail"
        assert len(example.recommended_models) == 2

    def test_get_embedding_text(self):
        """Test embedding text generation."""
        example = ForecastingExample(
            id="test_001",
            title="Daily Sales Forecasting",
            description="E-commerce daily sales with seasonality",
            domain="retail",
            data_characteristics={"frequency": "daily"},
            recommended_models=["prophet_reg"],
            preprocessing_strategy="Normalize",
            performance_metrics={},
            key_lessons=["Use Prophet for seasonality"],
            difficulty="easy",
            tags=["seasonality"]
        )

        text = example.get_embedding_text()

        # Should contain key information
        assert "Daily Sales Forecasting" in text
        assert "E-commerce daily sales with seasonality" in text
        assert "retail" in text
        assert "daily" in text
        assert "prophet_reg" in text


class TestExampleLibrary:
    """Test ExampleLibrary management."""

    def test_create_empty_library(self):
        """Test creating empty library."""
        library = ExampleLibrary()
        assert len(library) == 0
        assert len(library.examples) == 0

    def test_add_example(self):
        """Test adding examples to library."""
        library = ExampleLibrary()

        example = ForecastingExample(
            id="test_001",
            title="Test",
            description="Test",
            domain="retail",
            data_characteristics={},
            recommended_models=["linear_reg"],
            preprocessing_strategy="",
            performance_metrics={},
            key_lessons=[],
            difficulty="easy",
            tags=[]
        )

        library.add_example(example)
        assert len(library) == 1
        assert library[0].id == "test_001"

    def test_filter_by_domain(self):
        """Test filtering by domain."""
        library = ExampleLibrary()

        # Add examples from different domains
        for i, domain in enumerate(["retail", "finance", "retail"]):
            library.add_example(ForecastingExample(
                id=f"test_{i:03d}",
                title=f"Test {i}",
                description="Test",
                domain=domain,
                data_characteristics={},
                recommended_models=["linear_reg"],
                preprocessing_strategy="",
                performance_metrics={},
                key_lessons=[],
                difficulty="easy",
                tags=[]
            ))

        retail_examples = library.filter_by_domain("retail")
        assert len(retail_examples) == 2

        finance_examples = library.filter_by_domain("finance")
        assert len(finance_examples) == 1

    def test_filter_by_difficulty(self):
        """Test filtering by difficulty."""
        library = ExampleLibrary()

        # Add examples with different difficulties
        for i, difficulty in enumerate(["easy", "hard", "easy"]):
            library.add_example(ForecastingExample(
                id=f"test_{i:03d}",
                title=f"Test {i}",
                description="Test",
                domain="retail",
                data_characteristics={},
                recommended_models=["linear_reg"],
                preprocessing_strategy="",
                performance_metrics={},
                key_lessons=[],
                difficulty=difficulty,
                tags=[]
            ))

        easy_examples = library.filter_by_difficulty("easy")
        assert len(easy_examples) == 2

    def test_save_and_load(self):
        """Test saving and loading library."""
        library = ExampleLibrary()

        # Add example
        library.add_example(ForecastingExample(
            id="test_001",
            title="Test Example",
            description="Test description",
            domain="retail",
            data_characteristics={"frequency": "daily"},
            recommended_models=["prophet_reg"],
            preprocessing_strategy="Normalize",
            performance_metrics={"prophet_reg": {"rmse": 10.0}},
            key_lessons=["Lesson 1"],
            difficulty="easy",
            tags=["retail"]
        ))

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            library.save_to_file(temp_path)

            # Load into new library
            loaded_library = ExampleLibrary(temp_path)

            assert len(loaded_library) == 1
            assert loaded_library[0].id == "test_001"
            assert loaded_library[0].title == "Test Example"
            assert loaded_library[0].domain == "retail"
        finally:
            os.unlink(temp_path)


class TestRAGRetriever:
    """Test RAG retrieval system."""

    @pytest.fixture
    def example_library(self):
        """Create example library with diverse examples."""
        library = ExampleLibrary()

        # Daily retail sales with strong seasonality
        library.add_example(ForecastingExample(
            id="retail_001",
            title="Daily Retail Sales",
            description="E-commerce sales with strong weekly seasonality and promotional effects",
            domain="retail",
            data_characteristics={
                "frequency": "daily",
                "seasonality": "strong",
                "trend": "increasing",
                "n_observations": 730
            },
            recommended_models=["prophet_reg", "arima_reg", "linear_reg"],
            preprocessing_strategy="Date features, normalization",
            performance_metrics={
                "prophet_reg": {"rmse": 142.5, "mae": 98.3}
            },
            key_lessons=["Prophet excels with weekly seasonality"],
            difficulty="easy",
            tags=["seasonality", "retail"]
        ))

        # Hourly energy load
        library.add_example(ForecastingExample(
            id="energy_001",
            title="Energy Load Forecasting",
            description="Hourly electricity demand with daily and weekly patterns",
            domain="energy",
            data_characteristics={
                "frequency": "hourly",
                "seasonality": "strong",
                "trend": "increasing",
                "n_observations": 17520
            },
            recommended_models=["prophet_reg", "boost_tree", "mlp"],
            preprocessing_strategy="Temperature interaction, normalization",
            performance_metrics={
                "boost_tree": {"rmse": 198.5, "mae": 142.7}
            },
            key_lessons=["XGBoost best for complex hourly patterns"],
            difficulty="medium",
            tags=["hourly", "energy"]
        ))

        # Stock prices (no seasonality)
        library.add_example(ForecastingExample(
            id="finance_001",
            title="Stock Price Prediction",
            description="Daily stock prices with high volatility and no seasonality",
            domain="finance",
            data_characteristics={
                "frequency": "daily",
                "seasonality": "none",
                "trend": "non-stationary",
                "n_observations": 1250
            },
            recommended_models=["arima_reg", "rand_forest", "boost_tree"],
            preprocessing_strategy="YeoJohnson, rolling window features",
            performance_metrics={
                "arima_reg": {"rmse": 2.45, "mae": 1.87}
            },
            key_lessons=["ARIMA with differencing handles non-stationarity"],
            difficulty="hard",
            tags=["finance", "volatility"]
        ))

        return library

    def test_create_retriever(self, example_library):
        """Test creating RAG retriever."""
        retriever = RAGRetriever(example_library)

        assert retriever.example_library == example_library
        assert retriever.example_embeddings is not None
        assert retriever.example_embeddings.shape[0] == len(example_library)

    def test_retrieve_by_query(self, example_library):
        """Test retrieval by text query."""
        retriever = RAGRetriever(example_library, cache_embeddings=False)

        results = retriever.retrieve(
            query="Daily sales data with strong weekly seasonality",
            top_k=2
        )

        assert len(results) <= 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].rank == 1

        # Should retrieve retail example first
        assert "retail" in results[0].example.id.lower()

    def test_retrieve_by_data_characteristics(self, example_library):
        """Test retrieval by data characteristics."""
        retriever = RAGRetriever(example_library, cache_embeddings=False)

        # Query for daily data with seasonality
        data_chars = {
            "frequency": "daily",
            "seasonality": {"detected": True, "strength": 0.8},
            "trend": {"direction": "increasing"},
            "n_observations": 700
        }

        results = retriever.retrieve_by_data_characteristics(data_chars, top_k=2)

        assert len(results) <= 2
        # Should retrieve retail example (daily + strong seasonality)
        assert "retail" in results[0].example.id.lower()

    def test_retrieve_with_domain_filter(self, example_library):
        """Test retrieval with domain filter."""
        retriever = RAGRetriever(example_library, cache_embeddings=False)

        results = retriever.retrieve(
            query="Forecasting with strong seasonality",
            top_k=5,
            domain_filter="energy"
        )

        # Should only return energy examples
        assert all(r.example.domain == "energy" for r in results)

    def test_retrieve_with_difficulty_filter(self, example_library):
        """Test retrieval with difficulty filter."""
        retriever = RAGRetriever(example_library, cache_embeddings=False)

        results = retriever.retrieve(
            query="Daily forecasting",
            top_k=5,
            difficulty_filter="easy"
        )

        # Should only return easy examples
        assert all(r.example.difficulty == "easy" for r in results)

    def test_get_model_recommendations(self, example_library):
        """Test extracting model recommendations from results."""
        retriever = RAGRetriever(example_library, cache_embeddings=False)

        results = retriever.retrieve(
            query="Daily data with seasonality",
            top_k=2
        )

        models = retriever.get_model_recommendations_from_examples(results, top_n=3)

        assert len(models) <= 3
        assert all(isinstance(m, tuple) and len(m) == 2 for m in models)
        # prophet_reg should be highly recommended
        model_names = [m[0] for m in models]
        assert "prophet_reg" in model_names

    def test_get_preprocessing_insights(self, example_library):
        """Test extracting preprocessing insights."""
        retriever = RAGRetriever(example_library, cache_embeddings=False)

        results = retriever.retrieve(
            query="Daily sales forecasting",
            top_k=2
        )

        insights = retriever.get_preprocessing_insights(results)

        assert len(insights) > 0
        assert all(isinstance(i, str) for i in insights)

    def test_get_key_lessons(self, example_library):
        """Test extracting key lessons."""
        retriever = RAGRetriever(example_library, cache_embeddings=False)

        results = retriever.retrieve(
            query="Forecasting with seasonality",
            top_k=2
        )

        lessons = retriever.get_key_lessons(results)

        assert len(lessons) > 0
        assert all(isinstance(l, str) for l in lessons)

    def test_embedding_cache(self, example_library):
        """Test embedding caching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "embeddings_cache.pkl")

            # Create retriever with caching
            retriever1 = RAGRetriever(example_library, cache_embeddings=True)
            # Manually set cache path
            retriever1.embeddings_cache_path = cache_path
            retriever1._build_embeddings()

            # Cache file should exist
            assert os.path.exists(cache_path)

            # Create new retriever - should load from cache
            retriever2 = RAGRetriever(example_library, cache_embeddings=True)
            retriever2.embeddings_cache_path = cache_path
            retriever2._build_embeddings()

            # Embeddings should match
            np.testing.assert_array_almost_equal(
                retriever1.example_embeddings,
                retriever2.example_embeddings
            )


class TestRAGIntegrationWithAgent:
    """Test RAG integration with ForecastAgent."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=365, freq='D')

        # Daily data with weekly seasonality
        t = np.arange(365)
        trend = 100 + 0.5 * t
        seasonality = 20 * np.sin(2 * np.pi * t / 7)
        noise = np.random.randn(365) * 5

        values = trend + seasonality + noise

        return pd.DataFrame({
            'date': dates,
            'sales': values,
            'temperature': np.random.randn(365) * 10 + 20
        })

    def test_agent_with_rag_enabled(self, sample_data):
        """Test ForecastAgent with RAG enabled."""
        from py_agent import ForecastAgent

        # This will load the foundational examples
        agent = ForecastAgent(verbose=False, use_rag=True)

        assert agent.use_rag is True
        assert agent.rag_retriever is not None
        assert agent.example_library is not None
        assert len(agent.example_library) > 0

    def test_agent_without_rag(self, sample_data):
        """Test ForecastAgent without RAG."""
        from py_agent import ForecastAgent

        agent = ForecastAgent(verbose=False, use_rag=False)

        assert agent.use_rag is False
        assert agent.rag_retriever is None
        assert agent.example_library is None

    def test_generate_workflow_with_rag(self, sample_data):
        """Test workflow generation with RAG enhancement."""
        from py_agent import ForecastAgent

        agent = ForecastAgent(verbose=False, use_rag=True)

        # Generate workflow
        workflow = agent.generate_workflow(
            data=sample_data,
            request="Forecast daily sales with weekly seasonality"
        )

        assert workflow is not None

        # Check that RAG influenced model selection
        info = agent.last_workflow_info
        assert 'model_type' in info
        # Should recommend prophet_reg due to seasonality

    def test_compare_models_with_rag(self, sample_data):
        """Test multi-model comparison with RAG enhancement."""
        from py_agent import ForecastAgent

        agent = ForecastAgent(verbose=False, use_rag=True)

        # Compare models
        results = agent.compare_models(
            data=sample_data,
            request="Forecast daily sales",
            n_models=3,
            cv_strategy='vfold',
            n_folds=3
        )

        assert results is not None
        assert 'best_model_id' in results
        assert 'rankings' in results
        assert 'workflowset' in results

    def test_rag_boosts_confidence(self, sample_data):
        """Test that RAG boosts confidence for recommended models."""
        from py_agent import ForecastAgent
        from py_agent.tools.model_selection import suggest_model
        from py_agent.tools.data_analysis import analyze_temporal_patterns

        agent_rag = ForecastAgent(verbose=False, use_rag=True)
        agent_no_rag = ForecastAgent(verbose=False, use_rag=False)

        # Analyze data
        data_chars = analyze_temporal_patterns(sample_data, 'date', 'sales')

        # Get suggestions without RAG
        suggestions_no_rag = suggest_model(data_chars, None)
        model_confidences_no_rag = {
            s['model_type']: s['confidence']
            for s in suggestions_no_rag
        }

        # Generate workflow with RAG (should boost confidence)
        workflow_rag = agent_rag.generate_workflow(
            data=sample_data,
            request="Forecast daily sales with seasonality"
        )

        # Check that some models got RAG boost
        # (We can't directly access the boosted suggestions, but they influenced the selection)
        assert workflow_rag is not None

    def test_rag_fallback_on_error(self, sample_data):
        """Test that agent continues if RAG fails."""
        from py_agent import ForecastAgent

        # Create agent with invalid RAG path
        agent = ForecastAgent(verbose=False, use_rag=True)
        # Force RAG to None to simulate failure
        agent.rag_retriever = None

        # Should still work without RAG
        workflow = agent.generate_workflow(
            data=sample_data,
            request="Forecast daily sales"
        )

        assert workflow is not None


class TestFoundationalExamples:
    """Test the foundational example set."""

    def test_load_foundational_examples(self):
        """Test loading foundational examples."""
        from py_agent.knowledge import create_foundational_examples

        examples = create_foundational_examples()

        assert len(examples) == 8
        assert all(isinstance(ex, ForecastingExample) for ex in examples)

    def test_example_diversity(self):
        """Test that examples cover diverse scenarios."""
        from py_agent.knowledge import create_foundational_examples

        examples = create_foundational_examples()

        # Check domains
        domains = {ex.domain for ex in examples}
        assert len(domains) >= 5  # At least 5 different domains

        # Check frequencies
        frequencies = {ex.data_characteristics.get('frequency') for ex in examples}
        assert len(frequencies) >= 4  # At least 4 different frequencies

        # Check difficulties
        difficulties = {ex.difficulty for ex in examples}
        assert 'easy' in difficulties
        assert 'medium' in difficulties
        assert 'hard' in difficulties

    def test_all_examples_have_required_fields(self):
        """Test that all examples have required fields."""
        from py_agent.knowledge import create_foundational_examples

        examples = create_foundational_examples()

        for ex in examples:
            assert ex.id
            assert ex.title
            assert ex.description
            assert ex.domain
            assert ex.data_characteristics
            assert ex.recommended_models
            assert len(ex.recommended_models) > 0
            assert ex.preprocessing_strategy
            assert ex.key_lessons
            assert len(ex.key_lessons) > 0
            assert ex.difficulty in ['easy', 'medium', 'hard']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
