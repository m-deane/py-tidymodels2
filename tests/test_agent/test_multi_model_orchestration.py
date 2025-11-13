"""
Tests for multi-model orchestration tools (Phase 3.3).

Validates WorkflowSet generation, cross-validation, model ranking,
and ensemble recommendation capabilities.
"""

import pytest
import pandas as pd
import numpy as np
from py_agent.tools.multi_model_orchestration import (
    generate_workflowset,
    compare_models_cv,
    select_best_models,
    recommend_ensemble,
    _create_model_spec,
    _get_model_families,
    _extract_model_type
)


# Fixtures

@pytest.fixture
def sample_time_series_data():
    """Generate sample time series data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=365, freq='D')

    data = pd.DataFrame({
        'date': dates,
        'sales': 100 + np.cumsum(np.random.randn(365)) + 10 * np.sin(np.arange(365) * 2 * np.pi / 7),
        'temperature': 20 + 10 * np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.randn(365),
        'promotion': np.random.choice([0, 1], size=365, p=[0.8, 0.2])
    })

    return data


@pytest.fixture
def sample_model_recommendations():
    """Generate sample model recommendations."""
    return [
        {
            'model_type': 'prophet_reg',
            'reasoning': 'Strong seasonality detected',
            'confidence': 0.85,
            'expected_performance': 'high',
            'train_time_estimate': 5.0,
            'interpretability': 'medium'
        },
        {
            'model_type': 'linear_reg',
            'reasoning': 'Significant trend',
            'confidence': 0.75,
            'expected_performance': 'medium',
            'train_time_estimate': 0.5,
            'interpretability': 'high'
        },
        {
            'model_type': 'rand_forest',
            'reasoning': 'Complex patterns',
            'confidence': 0.70,
            'expected_performance': 'high',
            'train_time_estimate': 10.0,
            'interpretability': 'low'
        },
        {
            'model_type': 'arima_reg',
            'reasoning': 'Temporal autocorrelation',
            'confidence': 0.68,
            'expected_performance': 'medium',
            'train_time_estimate': 8.0,
            'interpretability': 'medium'
        },
        {
            'model_type': 'boost_tree',
            'reasoning': 'Nonlinear relationships',
            'confidence': 0.65,
            'expected_performance': 'high',
            'train_time_estimate': 15.0,
            'interpretability': 'low'
        }
    ]


@pytest.fixture
def sample_recipe_code():
    """Generate sample recipe code."""
    return """from py_recipes import recipe
from py_recipes.selectors import all_numeric, all_nominal

rec = (recipe(data, formula)
    .step_impute_median(all_numeric())
    .step_normalize(all_numeric())
    .step_dummy(all_nominal()))
"""


# Tests for generate_workflowset()

class TestGenerateWorkflowSet:
    """Tests for WorkflowSet generation."""

    def test_basic_workflowset_creation(self, sample_model_recommendations, sample_recipe_code):
        """Test basic WorkflowSet creation with multiple models."""
        wf_set = generate_workflowset(
            model_recommendations=sample_model_recommendations,
            recipe_code=sample_recipe_code,
            formula='sales ~ temperature + promotion',
            max_models=5
        )

        assert len(wf_set.workflows) == 5
        assert 'prophet_reg_1' in [wf_id for wf_id, _ in wf_set.workflows]
        assert 'linear_reg_2' in [wf_id for wf_id, _ in wf_set.workflows]

    def test_workflowset_respects_max_models(self, sample_model_recommendations, sample_recipe_code):
        """Test that max_models parameter limits workflows created."""
        wf_set = generate_workflowset(
            model_recommendations=sample_model_recommendations,
            recipe_code=sample_recipe_code,
            formula='sales ~ .',
            max_models=3
        )

        assert len(wf_set.workflows) == 3

    def test_workflowset_with_single_model(self, sample_model_recommendations, sample_recipe_code):
        """Test WorkflowSet creation with single model."""
        wf_set = generate_workflowset(
            model_recommendations=sample_model_recommendations[:1],
            recipe_code=sample_recipe_code,
            formula='sales ~ temperature',
            max_models=1
        )

        assert len(wf_set.workflows) == 1

    def test_workflowset_handles_empty_recommendations(self, sample_recipe_code):
        """Test that empty recommendations list is handled."""
        wf_set = generate_workflowset(
            model_recommendations=[],
            recipe_code=sample_recipe_code,
            formula='sales ~ .',
            max_models=5
        )

        assert len(wf_set.workflows) == 0


# Tests for select_best_models()

class TestSelectBestModels:
    """Tests for model selection from rankings."""

    @pytest.fixture
    def sample_rankings(self):
        """Generate sample ranking results."""
        return pd.DataFrame({
            'rank': [1, 2, 3, 4, 5],
            'wflow_id': ['prophet_reg_1', 'arima_reg_4', 'linear_reg_2', 'rand_forest_3', 'boost_tree_5'],
            'mean': [10.5, 12.3, 15.2, 16.8, 18.1],
            'std_err': [0.8, 1.2, 1.5, 1.8, 2.0],
            '.metric': ['rmse'] * 5
        })

    def test_select_best_strategy(self, sample_rankings):
        """Test 'best' selection strategy."""
        selected = select_best_models(
            sample_rankings,
            selection_strategy='best',
            n_models=3
        )

        assert len(selected) == 3
        assert selected[0] == 'prophet_reg_1'
        assert selected[1] == 'arima_reg_4'
        assert selected[2] == 'linear_reg_2'

    def test_select_single_best_model(self, sample_rankings):
        """Test selecting only the best model."""
        selected = select_best_models(
            sample_rankings,
            selection_strategy='best',
            n_models=1
        )

        assert len(selected) == 1
        assert selected[0] == 'prophet_reg_1'

    def test_select_within_1se_strategy(self, sample_rankings):
        """Test 'within_1se' selection strategy."""
        selected = select_best_models(
            sample_rankings,
            selection_strategy='within_1se'
        )

        # Best RMSE: 10.5, Best SE: 0.8
        # Threshold: 10.5 + 0.8 = 11.3
        # Should select: prophet (10.5) only (arima 12.3 > 11.3)
        assert len(selected) >= 1
        assert 'prophet_reg_1' in selected

    def test_select_threshold_strategy(self, sample_rankings):
        """Test 'threshold' selection strategy."""
        selected = select_best_models(
            sample_rankings,
            selection_strategy='threshold',
            performance_threshold=13.0
        )

        # Should select models with mean <= 13.0
        # prophet (10.5), arima (12.3)
        assert len(selected) == 2
        assert 'prophet_reg_1' in selected
        assert 'arima_reg_4' in selected

    def test_threshold_strategy_requires_threshold(self, sample_rankings):
        """Test that threshold strategy requires performance_threshold."""
        with pytest.raises(ValueError, match="performance_threshold required"):
            select_best_models(
                sample_rankings,
                selection_strategy='threshold'
            )

    def test_unknown_strategy_raises_error(self, sample_rankings):
        """Test that unknown strategy raises error."""
        with pytest.raises(ValueError, match="Unknown selection strategy"):
            select_best_models(
                sample_rankings,
                selection_strategy='unknown_strategy'
            )


# Tests for recommend_ensemble()

class TestRecommendEnsemble:
    """Tests for ensemble recommendation."""

    @pytest.fixture
    def mock_workflowset(self):
        """Create mock WorkflowSet for testing."""
        # We'll use a minimal mock since actual WorkflowSet creation requires models
        from py_workflowsets import WorkflowSet
        return None  # Placeholder

    @pytest.fixture
    def sample_rankings(self):
        """Generate sample rankings for ensemble testing."""
        return pd.DataFrame({
            'rank': [1, 2, 3, 4, 5],
            'wflow_id': ['prophet_reg_1', 'linear_reg_2', 'rand_forest_3', 'arima_reg_4', 'boost_tree_5'],
            'mean': [10.5, 12.3, 13.2, 14.8, 15.1],
            'std_err': [0.8, 1.2, 1.3, 1.5, 1.6],
            '.metric': ['rmse'] * 5
        })

    def test_basic_ensemble_recommendation(self, mock_workflowset, sample_rankings):
        """Test basic ensemble recommendation."""
        ensemble_rec = recommend_ensemble(
            wf_set=mock_workflowset,
            ranked_results=sample_rankings,
            ensemble_size=3
        )

        assert 'model_ids' in ensemble_rec
        assert 'expected_performance' in ensemble_rec
        assert 'diversity_score' in ensemble_rec
        assert 'reasoning' in ensemble_rec
        assert 'ensemble_type' in ensemble_rec

    def test_ensemble_size_respected(self, mock_workflowset, sample_rankings):
        """Test that ensemble_size parameter is respected."""
        ensemble_rec = recommend_ensemble(
            wf_set=mock_workflowset,
            ranked_results=sample_rankings,
            ensemble_size=3
        )

        assert len(ensemble_rec['model_ids']) == 3

    def test_ensemble_selects_diverse_models(self, mock_workflowset, sample_rankings):
        """Test that ensemble selects diverse model families."""
        ensemble_rec = recommend_ensemble(
            wf_set=mock_workflowset,
            ranked_results=sample_rankings,
            ensemble_size=3
        )

        # Should select prophet, linear_reg, rand_forest (different families)
        model_ids = ensemble_rec['model_ids']
        assert 'prophet_reg_1' in model_ids or 'arima_reg_4' in model_ids  # Time series
        assert 'linear_reg_2' in model_ids  # Linear
        # May include rand_forest or boost_tree for diversity

    def test_ensemble_diversity_score(self, mock_workflowset, sample_rankings):
        """Test that diversity score is calculated correctly."""
        ensemble_rec = recommend_ensemble(
            wf_set=mock_workflowset,
            ranked_results=sample_rankings,
            ensemble_size=3
        )

        # Diversity score should be between 0 and 1
        assert 0 <= ensemble_rec['diversity_score'] <= 1

    def test_ensemble_expected_performance(self, mock_workflowset, sample_rankings):
        """Test that expected performance is estimated."""
        ensemble_rec = recommend_ensemble(
            wf_set=mock_workflowset,
            ranked_results=sample_rankings,
            ensemble_size=3
        )

        # Expected performance should be better than average (5% improvement)
        top_3_avg = sample_rankings.head(3)['mean'].mean()
        expected = top_3_avg * 0.95

        assert ensemble_rec['expected_performance'] == pytest.approx(expected, abs=0.1)


# Tests for helper functions

class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_create_model_spec_linear_reg(self):
        """Test creating linear_reg model spec."""
        spec = _create_model_spec('linear_reg')

        assert spec is not None
        assert hasattr(spec, 'model_type')

    def test_create_model_spec_prophet(self):
        """Test creating prophet_reg model spec."""
        spec = _create_model_spec('prophet_reg')

        assert spec is not None

    def test_create_model_spec_all_23_models(self):
        """Test that all 23 model types can be created."""
        model_types = [
            'null_model', 'naive_reg',
            'linear_reg', 'poisson_reg', 'gen_additive_mod',
            'decision_tree', 'rand_forest', 'boost_tree',
            'svm_rbf', 'svm_linear',
            'nearest_neighbor', 'mars', 'mlp',
            'arima_reg', 'prophet_reg', 'exp_smoothing', 'seasonal_reg', 'varmax_reg',
            'arima_boost', 'prophet_boost',
            'recursive_reg',
            'hybrid_model', 'manual_reg'
        ]

        for model_type in model_types:
            spec = _create_model_spec(model_type)
            assert spec is not None, f"Failed to create spec for {model_type}"

    def test_create_model_spec_unknown_fallback(self):
        """Test that unknown model types fall back to linear_reg."""
        spec = _create_model_spec('unknown_model_type')

        # Should fallback to linear_reg
        assert spec is not None

    def test_get_model_families_mapping(self):
        """Test model family mapping."""
        model_types = ['linear_reg', 'prophet_reg', 'rand_forest', 'boost_tree']
        families = _get_model_families(model_types)

        assert families['linear_reg'] == 'Linear'
        assert families['prophet_reg'] == 'Time Series (Prophet)'
        assert families['rand_forest'] == 'Tree Ensemble'
        assert families['boost_tree'] == 'Boosting'

    def test_get_model_families_all_types(self):
        """Test that all model types have family mappings."""
        all_model_types = [
            'null_model', 'naive_reg',
            'linear_reg', 'poisson_reg', 'gen_additive_mod',
            'decision_tree', 'rand_forest', 'boost_tree',
            'svm_rbf', 'svm_linear',
            'nearest_neighbor', 'mars', 'mlp',
            'arima_reg', 'prophet_reg', 'exp_smoothing', 'seasonal_reg', 'varmax_reg',
            'arima_boost', 'prophet_boost',
            'recursive_reg', 'hybrid_model', 'manual_reg'
        ]

        families = _get_model_families(all_model_types)

        for model_type in all_model_types:
            assert model_type in families
            assert families[model_type] != 'Unknown'

    def test_extract_model_type_basic(self):
        """Test extracting model type from workflow ID."""
        assert _extract_model_type('prophet_reg_1') == 'prophet_reg'
        assert _extract_model_type('linear_reg_2') == 'linear_reg'
        assert _extract_model_type('rand_forest_3') == 'rand_forest'

    def test_extract_model_type_multi_word(self):
        """Test extracting multi-word model types."""
        assert _extract_model_type('boost_tree_1') == 'boost_tree'
        assert _extract_model_type('arima_boost_2') == 'arima_boost'
        assert _extract_model_type('prophet_boost_3') == 'prophet_boost'

    def test_extract_model_type_edge_cases(self):
        """Test edge cases in model type extraction."""
        # No numeric suffix
        assert _extract_model_type('prophet_reg') == 'prophet_reg'

        # Multiple underscores
        assert _extract_model_type('gen_additive_mod_1') == 'gen_additive_mod'


# Integration tests (require actual data)

class TestIntegrationMultiModel:
    """Integration tests for multi-model orchestration."""

    def test_end_to_end_comparison(self, sample_time_series_data, sample_model_recommendations, sample_recipe_code):
        """Test end-to-end multi-model comparison workflow."""
        # This is a placeholder for full integration test
        # In practice, would run actual CV and compare models

        # Generate WorkflowSet
        wf_set = generate_workflowset(
            model_recommendations=sample_model_recommendations[:3],
            recipe_code=sample_recipe_code,
            formula='sales ~ temperature + promotion',
            max_models=3
        )

        assert len(wf_set.workflows) == 3

    def test_compare_models_cv_time_series_strategy(self, sample_time_series_data):
        """Test CV with time_series strategy."""
        # This would require actual model fitting and CV
        # Placeholder for integration test
        pass

    def test_compare_models_cv_vfold_strategy(self, sample_time_series_data):
        """Test CV with vfold strategy."""
        # This would require actual model fitting and CV
        # Placeholder for integration test
        pass


# Edge case tests

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_workflowset_with_no_recommendations(self, sample_recipe_code):
        """Test WorkflowSet generation with no recommendations."""
        wf_set = generate_workflowset(
            model_recommendations=[],
            recipe_code=sample_recipe_code,
            formula='y ~ x',
            max_models=5
        )

        assert len(wf_set.workflows) == 0

    def test_select_more_models_than_available(self):
        """Test selecting more models than available."""
        rankings = pd.DataFrame({
            'rank': [1, 2],
            'wflow_id': ['model_1', 'model_2'],
            'mean': [10.0, 12.0],
            'std_err': [0.5, 0.7],
            '.metric': ['rmse', 'rmse']
        })

        selected = select_best_models(
            rankings,
            selection_strategy='best',
            n_models=5  # More than available
        )

        # Should return all available models
        assert len(selected) == 2

    def test_ensemble_with_single_model(self):
        """Test ensemble recommendation with only one model."""
        rankings = pd.DataFrame({
            'rank': [1],
            'wflow_id': ['prophet_reg_1'],
            'mean': [10.5],
            'std_err': [0.8],
            '.metric': ['rmse']
        })

        ensemble_rec = recommend_ensemble(
            wf_set=None,
            ranked_results=rankings,
            ensemble_size=3  # Request 3 but only 1 available
        )

        assert len(ensemble_rec['model_ids']) == 1

    def test_ensemble_diversity_with_same_family(self):
        """Test ensemble diversity when all models from same family."""
        rankings = pd.DataFrame({
            'rank': [1, 2, 3],
            'wflow_id': ['linear_reg_1', 'linear_reg_2', 'poisson_reg_3'],
            'mean': [10.0, 11.0, 12.0],
            'std_err': [0.5, 0.6, 0.7],
            '.metric': ['rmse', 'rmse', 'rmse']
        })

        ensemble_rec = recommend_ensemble(
            wf_set=None,
            ranked_results=rankings,
            ensemble_size=3
        )

        # All from Linear family, so diversity should be 1/3
        assert ensemble_rec['diversity_score'] == pytest.approx(1/3, abs=0.01)
