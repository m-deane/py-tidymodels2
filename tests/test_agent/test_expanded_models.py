"""
Tests for expanded model support (23 models).

Verifies that all 23 py-tidymodels models are supported in py_agent.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from py_agent.tools.model_selection import get_model_profiles, suggest_model
from py_agent.agents.forecast_agent import ForecastAgent


class TestModelProfiles:
    """Tests for get_model_profiles with all 23 models."""

    def test_all_23_models_present(self):
        """Test that all 23 models are in profiles."""
        profiles = get_model_profiles()

        expected_models = [
            # Baseline (2)
            'null_model', 'naive_reg',
            # Linear & Generalized (3)
            'linear_reg', 'poisson_reg', 'gen_additive_mod',
            # Tree-Based (3)
            'decision_tree', 'rand_forest', 'boost_tree',
            # SVM (2)
            'svm_rbf', 'svm_linear',
            # Instance-Based & Adaptive (3)
            'nearest_neighbor', 'mars', 'mlp',
            # Time Series (5)
            'arima_reg', 'prophet_reg', 'exp_smoothing', 'seasonal_reg', 'varmax_reg',
            # Hybrid Time Series (2)
            'arima_boost', 'prophet_boost',
            # Recursive (1)
            'recursive_reg',
            # Hybrid & Manual (2)
            'hybrid_model', 'manual_reg'
        ]

        assert len(profiles) == 23, f"Expected 23 models, got {len(profiles)}"

        for model in expected_models:
            assert model in profiles, f"Missing model: {model}"

    def test_all_profiles_have_required_fields(self):
        """Test that all profiles have required fields."""
        profiles = get_model_profiles()

        required_fields = [
            'train_time_per_1k',
            'predict_time_per_1k',
            'memory_per_feature',
            'interpretability',
            'accuracy_tier',
            'strengths',
            'weaknesses',
            'good_for_seasonality',
            'good_for_trend',
            'good_for_interactions'
        ]

        for model_type, profile in profiles.items():
            for field in required_fields:
                assert field in profile, f"{model_type} missing field: {field}"

    def test_interpretability_values(self):
        """Test that interpretability values are valid."""
        profiles = get_model_profiles()
        valid_interp = ['low', 'medium', 'high']

        for model_type, profile in profiles.items():
            assert profile['interpretability'] in valid_interp, \
                f"{model_type} has invalid interpretability: {profile['interpretability']}"

    def test_accuracy_tier_values(self):
        """Test that accuracy tier values are valid."""
        profiles = get_model_profiles()
        valid_tiers = ['low', 'medium', 'medium-high', 'high', 'very_high', 'varies']

        for model_type, profile in profiles.items():
            assert profile['accuracy_tier'] in valid_tiers, \
                f"{model_type} has invalid accuracy_tier: {profile['accuracy_tier']}"

    def test_boolean_flags(self):
        """Test that boolean flags are actually booleans."""
        profiles = get_model_profiles()

        for model_type, profile in profiles.items():
            assert isinstance(profile['good_for_seasonality'], bool), \
                f"{model_type} good_for_seasonality must be bool"
            assert isinstance(profile['good_for_trend'], bool), \
                f"{model_type} good_for_trend must be bool"
            assert isinstance(profile['good_for_interactions'], bool), \
                f"{model_type} good_for_interactions must be bool"


class TestModelRecommendations:
    """Tests for model recommendations with expanded model set."""

    def test_baseline_model_recommended_for_simple_data(self):
        """Test that baseline models can be recommended."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False, 'strength': 0.1},
            'trend': {'significant': False, 'direction': 'stable', 'strength': 0.05},
            'autocorrelation': {1: 0.1, 7: 0.05, 30: 0.02},
            'n_observations': 30,  # Very small sample
            'n_features': 1,
            'missing_rate': 0.0,
            'outlier_rate': 0.0
        }

        recommendations = suggest_model(data_chars)

        # Should get some recommendations
        assert len(recommendations) > 0

        # Check if baseline or simple models are prioritized for small samples
        model_types = [r['model_type'] for r in recommendations]
        # For small samples, simpler models should appear
        simple_models = ['null_model', 'naive_reg', 'linear_reg', 'exp_smoothing']
        assert any(m in model_types for m in simple_models)

    def test_prophet_recommended_for_strong_seasonality(self):
        """Test that prophet/seasonal models recommended for strong seasonality."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': True, 'strength': 0.9},
            'trend': {'significant': True, 'direction': 'increasing', 'strength': 0.6},
            'autocorrelation': {1: 0.8, 7: 0.75, 30: 0.5},
            'n_observations': 730,
            'n_features': 5,
            'missing_rate': 0.05,
            'outlier_rate': 0.02
        }

        recommendations = suggest_model(data_chars)

        # Prophet should be top recommendation
        model_types = [r['model_type'] for r in recommendations]
        seasonal_models = ['prophet_reg', 'naive_reg', 'exp_smoothing', 'seasonal_reg']

        # At least one seasonal model should be in top 3
        assert any(m in model_types[:3] for m in seasonal_models)

    def test_boost_tree_recommended_for_complex_patterns(self):
        """Test that advanced models recommended for complex data."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': True, 'strength': 0.5},
            'trend': {'significant': True, 'direction': 'increasing', 'strength': 0.7},
            'autocorrelation': {1: 0.6, 7: 0.4, 30: 0.3},
            'n_observations': 2000,  # Large sample
            'n_features': 20,  # Many features
            'missing_rate': 0.02,
            'outlier_rate': 0.03
        }

        recommendations = suggest_model(data_chars)

        # Should get high-capacity models for large, complex data
        model_types = [r['model_type'] for r in recommendations]
        complex_models = ['boost_tree', 'rand_forest', 'mlp', 'hybrid_model']

        # At least one complex model should be recommended
        assert any(m in model_types for m in complex_models)

    def test_interpretable_constraint_filters_models(self):
        """Test that interpretability constraint works."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False, 'strength': 0.1},
            'trend': {'significant': True, 'direction': 'increasing', 'strength': 0.5},
            'autocorrelation': {1: 0.4, 7: 0.2, 30: 0.1},
            'n_observations': 500,
            'n_features': 10,
            'missing_rate': 0.0,
            'outlier_rate': 0.0
        }

        constraints = {'interpretability': 'high'}

        recommendations = suggest_model(data_chars, constraints)

        # All recommendations should have high interpretability
        profiles = get_model_profiles()
        for rec in recommendations:
            model_type = rec['model_type']
            assert profiles[model_type]['interpretability'] == 'high', \
                f"{model_type} doesn't meet high interpretability constraint"

    def test_train_time_constraint_filters_models(self):
        """Test that training time constraint works."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': True, 'strength': 0.6},
            'trend': {'significant': True, 'direction': 'increasing', 'strength': 0.5},
            'autocorrelation': {1: 0.5, 7: 0.3, 30: 0.2},
            'n_observations': 1000,
            'n_features': 10,
            'missing_rate': 0.01,
            'outlier_rate': 0.01
        }

        constraints = {'max_train_time': 1.0}  # Very strict: 1 second

        recommendations = suggest_model(data_chars, constraints)

        # All recommended models should be fast
        for rec in recommendations:
            assert rec['train_time_estimate'] <= 1.0, \
                f"{rec['model_type']} exceeds time constraint: {rec['train_time_estimate']}"

    def test_all_model_types_can_be_recommended(self):
        """Test that all 23 models can potentially be recommended."""
        # Create scenarios that favor each model type
        scenarios = [
            # Baseline models
            {'n_observations': 20, 'seasonality': {'detected': False, 'strength': 0}},
            {'n_observations': 50, 'seasonality': {'detected': True, 'strength': 0.8}},

            # Linear & Generalized
            {'n_observations': 200, 'seasonality': {'detected': False, 'strength': 0}},
            {'n_observations': 200, 'seasonality': {'detected': False, 'strength': 0}},  # Poisson for count data
            {'n_observations': 300, 'seasonality': {'detected': True, 'strength': 0.6}},

            # Tree-based
            {'n_observations': 500, 'seasonality': {'detected': False, 'strength': 0}},
            {'n_observations': 1000, 'seasonality': {'detected': False, 'strength': 0}},
            {'n_observations': 2000, 'seasonality': {'detected': False, 'strength': 0}},

            # SVM
            {'n_observations': 800, 'seasonality': {'detected': False, 'strength': 0}},
            {'n_observations': 600, 'seasonality': {'detected': False, 'strength': 0}},

            # Instance-based
            {'n_observations': 400, 'seasonality': {'detected': False, 'strength': 0}},
            {'n_observations': 700, 'seasonality': {'detected': False, 'strength': 0}},
            {'n_observations': 1500, 'seasonality': {'detected': False, 'strength': 0}},

            # Time series
            {'n_observations': 500, 'seasonality': {'detected': False, 'strength': 0}},
            {'n_observations': 730, 'seasonality': {'detected': True, 'strength': 0.9}},
            {'n_observations': 365, 'seasonality': {'detected': True, 'strength': 0.7}},
            {'n_observations': 500, 'seasonality': {'detected': True, 'strength': 0.8}},
            {'n_observations': 1000, 'seasonality': {'detected': False, 'strength': 0}},

            # Hybrid
            {'n_observations': 1000, 'seasonality': {'detected': True, 'strength': 0.8}},
            {'n_observations': 1200, 'seasonality': {'detected': True, 'strength': 0.85}},

            # Recursive
            {'n_observations': 800, 'seasonality': {'detected': True, 'strength': 0.6}},

            # Generic hybrid & manual
            {'n_observations': 1500, 'seasonality': {'detected': True, 'strength': 0.7}},
            {'n_observations': 100, 'seasonality': {'detected': False, 'strength': 0}}
        ]

        all_recommended = set()

        for scenario in scenarios:
            data_chars = {
                'frequency': 'daily',
                'seasonality': scenario['seasonality'],
                'trend': {'significant': True, 'direction': 'increasing', 'strength': 0.5},
                'autocorrelation': {1: 0.5, 7: 0.3, 30: 0.2},
                'n_observations': scenario['n_observations'],
                'n_features': 10,
                'missing_rate': 0.01,
                'outlier_rate': 0.01
            }

            recommendations = suggest_model(data_chars)
            for rec in recommendations:
                all_recommended.add(rec['model_type'])

        # We should see a good variety of models recommended
        # (not necessarily all 23, but at least 15-18 across different scenarios)
        assert len(all_recommended) >= 15, \
            f"Only {len(all_recommended)} model types recommended across all scenarios"


class TestDynamicModelCreation:
    """Tests for _create_model_spec dynamic model creation."""

    def test_create_all_23_models(self):
        """Test that all 23 models can be dynamically created."""
        agent = ForecastAgent(verbose=False)

        all_models = [
            # Baseline (2)
            'null_model', 'naive_reg',
            # Linear & Generalized (3)
            'linear_reg', 'poisson_reg', 'gen_additive_mod',
            # Tree-Based (3)
            'decision_tree', 'rand_forest', 'boost_tree',
            # SVM (2)
            'svm_rbf', 'svm_linear',
            # Instance-Based & Adaptive (3)
            'nearest_neighbor', 'mars', 'mlp',
            # Time Series (5)
            'arima_reg', 'prophet_reg', 'exp_smoothing', 'seasonal_reg', 'varmax_reg',
            # Hybrid Time Series (2)
            'arima_boost', 'prophet_boost',
            # Recursive (1)
            'recursive_reg',
            # Hybrid & Manual (2)
            'hybrid_model', 'manual_reg'
        ]

        for model_type in all_models:
            spec = agent._create_model_spec(model_type)
            assert spec is not None, f"Failed to create {model_type}"
            # Check it's a model spec object (has model_type attribute)
            assert hasattr(spec, 'model_type'), f"{model_type} doesn't have model_type attribute"

    def test_unknown_model_falls_back_to_linear_reg(self):
        """Test that unknown models fall back to linear_reg."""
        agent = ForecastAgent(verbose=False)

        spec = agent._create_model_spec('nonexistent_model')

        assert spec is not None
        assert spec.model_type == 'linear_reg'

    def test_model_spec_has_correct_type(self):
        """Test that created specs have correct model_type."""
        agent = ForecastAgent(verbose=False)

        test_models = ['linear_reg', 'prophet_reg', 'rand_forest', 'boost_tree', 'arima_reg']

        for model_type in test_models:
            spec = agent._create_model_spec(model_type)
            assert spec.model_type == model_type, \
                f"Created spec has wrong model_type: {spec.model_type} (expected {model_type})"


class TestEndToEndModelVariety:
    """Integration tests with different model recommendations."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        values = (
            np.sin(np.arange(365) * 2 * np.pi / 7) * 10 +  # Weekly seasonality
            np.arange(365) * 0.1 +  # Trend
            np.random.randn(365) * 2  # Noise
        )
        return pd.DataFrame({'date': dates, 'sales': values})

    def test_generate_workflow_with_various_models(self, sample_data):
        """Test that generate_workflow works with different model recommendations."""
        agent = ForecastAgent(verbose=False)

        # This will use the default recommendation logic
        workflow = agent.generate_workflow(
            data=sample_data,
            request="Forecast sales with weekly patterns"
        )

        assert workflow is not None
        assert agent.last_workflow_info['model_type'] in get_model_profiles()

    def test_workflow_info_contains_model_details(self, sample_data):
        """Test that workflow info includes complete model details."""
        agent = ForecastAgent(verbose=False)

        workflow = agent.generate_workflow(
            data=sample_data,
            request="Forecast next month sales"
        )

        info = agent.last_workflow_info

        assert 'model_type' in info
        assert 'data_characteristics' in info
        assert 'recipe_code' in info
        assert 'workflow_code' in info

        # Model type should be one of the 23 supported models
        assert info['model_type'] in get_model_profiles()


def test_model_count_constant():
    """Test that model count is exactly 23 (catches accidental additions/removals)."""
    profiles = get_model_profiles()
    assert len(profiles) == 23, \
        f"Model count changed! Expected 23, got {len(profiles)}. Update tests if intentional."
