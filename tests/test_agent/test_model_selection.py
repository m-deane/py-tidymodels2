"""
Tests for model selection tools.

Tests the recommendation system that matches data characteristics
to appropriate py-tidymodels models.
"""

import pytest
from py_agent.tools.model_selection import (
    suggest_model,
    get_model_profiles,
    _meets_constraints,
    _calculate_suitability_score
)


class TestGetModelProfiles:
    """Tests for model profile retrieval."""

    def test_returns_dict(self):
        """Test that get_model_profiles returns a dictionary."""
        profiles = get_model_profiles()

        assert isinstance(profiles, dict)
        assert len(profiles) > 0

    def test_contains_mvp_models(self):
        """Test that MVP models are included."""
        profiles = get_model_profiles()

        # MVP supports these 3 models
        assert 'linear_reg' in profiles
        assert 'prophet_reg' in profiles
        assert 'rand_forest' in profiles

    def test_profile_structure(self):
        """Test that each profile has required fields."""
        profiles = get_model_profiles()

        for model_type, profile in profiles.items():
            assert 'train_time_per_1k' in profile
            assert 'interpretability' in profile
            assert 'accuracy_tier' in profile
            assert 'strengths' in profile
            assert 'weaknesses' in profile
            assert 'good_for_seasonality' in profile
            assert 'good_for_trend' in profile

    def test_interpretability_values(self):
        """Test that interpretability values are valid."""
        profiles = get_model_profiles()
        valid_levels = ['low', 'medium', 'high']

        for profile in profiles.values():
            assert profile['interpretability'] in valid_levels


class TestSuggestModel:
    """Tests for model suggestion function."""

    def test_suggest_for_seasonal_data(self):
        """Test suggestion for data with strong seasonality."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': True, 'strength': 0.8, 'period': 7},
            'trend': {'significant': True, 'direction': 'increasing', 'strength': 0.6},
            'autocorrelation': {'lag_1': 0.7, 'lag_7': 0.5},
            'missing_rate': 0.02,
            'outlier_rate': 0.01,
            'n_observations': 365
        }

        suggestions = suggest_model(data_chars)

        assert len(suggestions) > 0
        # Prophet should be top recommendation for seasonal data
        assert suggestions[0]['model_type'] == 'prophet_reg'
        assert suggestions[0]['confidence'] > 0.5

    def test_suggest_for_linear_trend(self):
        """Test suggestion for data with linear trend, no seasonality."""
        data_chars = {
            'frequency': 'monthly',
            'seasonality': {'detected': False, 'strength': 0.1},
            'trend': {'significant': True, 'direction': 'increasing', 'strength': 0.9},
            'autocorrelation': {'lag_1': 0.3},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_observations': 100
        }

        suggestions = suggest_model(data_chars)

        assert len(suggestions) > 0
        # Linear regression or ARIMA should be recommended
        suggested_models = [s['model_type'] for s in suggestions]
        assert 'linear_reg' in suggested_models or 'arima_reg' in suggested_models

    def test_with_time_constraint(self):
        """Test that time constraint filters slow models."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False, 'strength': 0.1},
            'trend': {'significant': True, 'direction': 'stable', 'strength': 0.2},
            'autocorrelation': {'lag_1': 0.5},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_observations': 10000  # Large dataset
        }

        constraints = {
            'max_train_time': 10  # Only 10 seconds
        }

        suggestions = suggest_model(data_chars, constraints)

        # Should only include fast models
        for suggestion in suggestions:
            assert suggestion['train_time_estimate'] <= 10

    def test_with_interpretability_constraint(self):
        """Test that interpretability constraint is respected."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False, 'strength': 0.1},
            'trend': {'significant': True, 'direction': 'stable', 'strength': 0.2},
            'autocorrelation': {'lag_1': 0.5},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_observations': 365
        }

        constraints = {
            'interpretability': 'high'
        }

        suggestions = suggest_model(data_chars, constraints)

        # All suggestions should be highly interpretable
        for suggestion in suggestions:
            assert suggestion['interpretability'] in ['high', 'medium']

    def test_returns_multiple_suggestions(self):
        """Test that multiple models are suggested."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': True, 'strength': 0.6},
            'trend': {'significant': True, 'direction': 'increasing', 'strength': 0.5},
            'autocorrelation': {'lag_1': 0.6},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_observations': 365
        }

        suggestions = suggest_model(data_chars)

        # Should return multiple options (up to 5)
        assert len(suggestions) >= 2
        assert len(suggestions) <= 5

    def test_suggestions_sorted_by_confidence(self):
        """Test that suggestions are sorted by confidence."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': True, 'strength': 0.6},
            'trend': {'significant': True, 'direction': 'increasing', 'strength': 0.5},
            'autocorrelation': {'lag_1': 0.6},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_observations': 365
        }

        suggestions = suggest_model(data_chars)

        # Confidence should be descending
        confidences = [s['confidence'] for s in suggestions]
        assert confidences == sorted(confidences, reverse=True)

    def test_suggestion_has_reasoning(self):
        """Test that each suggestion includes reasoning."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': True, 'strength': 0.8},
            'trend': {'significant': True, 'direction': 'increasing', 'strength': 0.5},
            'autocorrelation': {'lag_1': 0.6},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_observations': 365
        }

        suggestions = suggest_model(data_chars)

        for suggestion in suggestions:
            assert 'reasoning' in suggestion
            assert len(suggestion['reasoning']) > 0
            assert isinstance(suggestion['reasoning'], str)


class TestConstraintFiltering:
    """Tests for constraint filtering logic."""

    def test_meets_time_constraint(self):
        """Test time constraint checking."""
        profile = {'train_time_per_1k': 0.05}  # 50ms per 1k samples
        constraints = {'max_train_time': 1.0}  # 1 second max
        data_chars = {'n_observations': 10000}  # 10k samples

        # 10k samples * 0.05ms/1k = 0.5 seconds < 1 second
        meets = _meets_constraints(profile, constraints, data_chars)

        assert meets == True

    def test_fails_time_constraint(self):
        """Test rejection when time constraint not met."""
        profile = {'train_time_per_1k': 5.0}  # 5 seconds per 1k samples
        constraints = {'max_train_time': 10.0}  # 10 seconds max
        data_chars = {'n_observations': 10000}  # 10k samples

        # 10k samples * 5s/1k = 50 seconds > 10 seconds
        meets = _meets_constraints(profile, constraints, data_chars)

        assert meets == False

    def test_meets_interpretability_constraint(self):
        """Test interpretability constraint checking."""
        profile = {'interpretability': 'high'}
        constraints = {'interpretability': 'medium'}
        data_chars = {'n_observations': 1000}

        # High interpretability meets medium requirement
        meets = _meets_constraints(profile, constraints, data_chars)

        assert meets == True

    def test_fails_interpretability_constraint(self):
        """Test rejection when interpretability too low."""
        profile = {'interpretability': 'low'}
        constraints = {'interpretability': 'high'}
        data_chars = {'n_observations': 1000}

        # Low interpretability doesn't meet high requirement
        meets = _meets_constraints(profile, constraints, data_chars)

        assert meets == False


class TestSuitabilityScoring:
    """Tests for suitability score calculation."""

    def test_seasonal_data_boosts_prophet(self):
        """Test that seasonal data increases Prophet's score."""
        prophet_profile = get_model_profiles()['prophet_reg']

        data_chars_seasonal = {
            'seasonality': {'detected': True, 'strength': 0.9},
            'trend': {'significant': True, 'strength': 0.5},
            'autocorrelation': {'lag_1': 0.6, 'lag_7': 0.4},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_observations': 365
        }

        data_chars_no_seasonal = {
            'seasonality': {'detected': False, 'strength': 0.1},
            'trend': {'significant': True, 'strength': 0.5},
            'autocorrelation': {'lag_1': 0.6, 'lag_7': 0.4},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_observations': 365
        }

        score_seasonal = _calculate_suitability_score(prophet_profile, data_chars_seasonal)
        score_no_seasonal = _calculate_suitability_score(prophet_profile, data_chars_no_seasonal)

        # Prophet should score higher with seasonal data
        assert score_seasonal > score_no_seasonal

    def test_score_bounded(self):
        """Test that scores are between 0 and 1."""
        profiles = get_model_profiles()

        data_chars = {
            'seasonality': {'detected': True, 'strength': 0.8},
            'trend': {'significant': True, 'strength': 0.7},
            'autocorrelation': {'lag_1': 0.8, 'lag_7': 0.5},
            'missing_rate': 0.1,
            'outlier_rate': 0.05,
            'n_observations': 500
        }

        for profile in profiles.values():
            score = _calculate_suitability_score(profile, data_chars)
            assert 0.0 <= score <= 1.0

    def test_missing_data_penalty(self):
        """Test that missing data reduces score for non-robust models."""
        linear_profile = get_model_profiles()['linear_reg']

        data_chars_clean = {
            'seasonality': {'detected': False, 'strength': 0.1},
            'trend': {'significant': True, 'strength': 0.5},
            'autocorrelation': {'lag_1': 0.3},
            'missing_rate': 0.0,  # No missing data
            'outlier_rate': 0.0,
            'n_observations': 365
        }

        data_chars_missing = data_chars_clean.copy()
        data_chars_missing['missing_rate'] = 0.2  # 20% missing

        score_clean = _calculate_suitability_score(linear_profile, data_chars_clean)
        score_missing = _calculate_suitability_score(linear_profile, data_chars_missing)

        # Score should be lower with missing data
        assert score_missing < score_clean


def test_integration_suggest_for_retail_sales():
    """Integration test: Suggest model for retail sales scenario."""
    # Typical retail sales characteristics
    data_chars = {
        'frequency': 'daily',
        'seasonality': {
            'detected': True,
            'strength': 0.75,  # Strong weekly pattern
            'period': 7
        },
        'trend': {
            'significant': True,
            'direction': 'increasing',
            'strength': 0.6
        },
        'autocorrelation': {
            'lag_1': 0.7,
            'lag_7': 0.6,
            'lag_30': 0.3
        },
        'missing_rate': 0.05,
        'outlier_rate': 0.02,
        'n_observations': 730  # 2 years
    }

    suggestions = suggest_model(data_chars)

    # Prophet should be top choice for retail with strong seasonality
    assert suggestions[0]['model_type'] == 'prophet_reg'
    assert suggestions[0]['confidence'] > 0.7
    assert 'seasonality' in suggestions[0]['reasoning'].lower()
