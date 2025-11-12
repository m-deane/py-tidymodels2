"""
Tests for parallel evaluation in genetic algorithm feature selection.
"""

import pytest
import numpy as np
import pandas as pd
from py_recipes.steps.genetic_selection import StepSelectGeneticAlgorithm
from py_recipes.recipe import recipe
from py_parsnip import linear_reg


@pytest.fixture
def feature_selection_data():
    """Create dataset for testing parallel evaluation."""
    np.random.seed(42)
    n = 200

    # Important features
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)

    # Noise features
    x4 = np.random.randn(n)
    x5 = np.random.randn(n)
    x6 = np.random.randn(n)

    # Outcome: y = 3*x1 + 2*x2 + x3 + noise
    y = 3 * x1 + 2 * x2 + x3 + np.random.randn(n) * 0.5

    return pd.DataFrame({
        'y': y,
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,
        'x5': x5,
        'x6': x6
    })


class TestParallelEvaluation:
    """Tests for parallel fitness evaluation."""

    def test_sequential_evaluation_default(self, feature_selection_data):
        """Test that sequential evaluation is default (n_jobs=1)."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Should complete successfully
        assert prepared._is_prepared
        assert len(prepared._selected_features) >= 1
        assert len(prepared._selected_features) <= 3

    def test_sequential_evaluation_explicit(self, feature_selection_data):
        """Test explicit sequential evaluation (n_jobs=1)."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            n_jobs=1,  # Explicit sequential
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        assert prepared._is_prepared
        assert len(prepared._selected_features) <= 3

    def test_parallel_evaluation_two_jobs(self, feature_selection_data):
        """Test parallel evaluation with 2 workers."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            n_jobs=2,  # 2 parallel workers
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Should complete successfully with parallel evaluation
        assert prepared._is_prepared
        assert len(prepared._selected_features) >= 1
        assert len(prepared._selected_features) <= 3

    def test_parallel_evaluation_all_cores(self, feature_selection_data):
        """Test parallel evaluation with all cores (n_jobs=-1)."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            n_jobs=-1,  # Use all cores
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Should complete successfully using all cores
        assert prepared._is_prepared
        assert len(prepared._selected_features) <= 3

    def test_parallel_results_deterministic(self, feature_selection_data):
        """Test that parallel evaluation produces deterministic results with same seed."""
        # Sequential
        step_seq = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            n_jobs=1,
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        # Parallel
        step_par = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            n_jobs=2,
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prep_seq = step_seq.prep(feature_selection_data)
        prep_par = step_par.prep(feature_selection_data)

        # Both should select the same features (deterministic with same seed)
        assert set(prep_seq._selected_features) == set(prep_par._selected_features)

    def test_parallel_with_constraints(self, feature_selection_data):
        """Test parallel evaluation with statistical constraints."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            constraints={'p_value': {'max': 0.05}},
            n_jobs=2,
            population_size=20,
            generations=15,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Should handle constraints correctly in parallel
        assert prepared._is_prepared
        assert len(prepared._selected_features) >= 1

    def test_parallel_with_adaptive_params(self, feature_selection_data):
        """Test parallel evaluation with adaptive GA parameters."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            adaptive_mutation=True,
            adaptive_crossover=True,
            n_jobs=2,
            population_size=20,
            generations=15,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Should combine adaptive parameters with parallel evaluation
        assert prepared._is_prepared
        assert len(prepared._selected_features) <= 3

    def test_parallel_with_warm_start(self, feature_selection_data):
        """Test parallel evaluation with warm start initialization."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            warm_start='importance',
            n_jobs=2,
            population_size=20,
            generations=15,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Should combine warm start with parallel evaluation
        assert prepared._is_prepared
        assert len(prepared._selected_features) >= 1

    def test_parallel_via_recipe_api(self, feature_selection_data):
        """Test parallel evaluation via recipe convenience function."""
        from py_recipes.steps import step_select_genetic_algorithm

        rec = recipe(feature_selection_data)
        rec = step_select_genetic_algorithm(
            rec,
            outcome='y',
            model=linear_reg(),
            top_n=3,
            n_jobs=2,  # Parallel evaluation
            population_size=15,
            generations=10,
            cv_folds=1,
            random_state=42
        )

        prepped = rec.prep(feature_selection_data)
        baked = prepped.bake(feature_selection_data)

        # Should complete successfully
        assert 'y' in baked.columns
        assert baked.shape[1] >= 2  # y + at least 1 feature
        assert baked.shape[1] <= 4  # y + at most 3 features


class TestParallelWithAllEnhancements:
    """Tests for parallel evaluation combined with all other enhancements."""

    def test_all_enhancements_together(self, feature_selection_data):
        """Test all enhancements including parallel evaluation together."""
        costs = {'x1': 1.0, 'x2': 1.0, 'x3': 2.0, 'x4': 3.0, 'x5': 5.0, 'x6': 5.0}

        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=2,
            # Enhancement #1: Mandatory/Forbidden
            mandatory_features=['x1'],
            # Enhancement #2: Feature Costs
            feature_costs=costs,
            max_total_cost=8.0,
            cost_weight=0.3,
            # Enhancement #3: Sparsity
            sparsity_weight=0.2,
            # Enhancement #4: Warm Start
            warm_start='importance',
            # Enhancement #5: Adaptive Parameters
            adaptive_mutation=True,
            adaptive_crossover=True,
            # Enhancement #6: Constraint Relaxation
            constraints={'p_value': {'max': 0.05}},
            relax_constraints_after=8,
            relaxation_rate=0.1,
            # Enhancement #7: Parallel Evaluation
            n_jobs=2,
            population_size=20,
            generations=20,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Should handle all enhancements together
        assert prepared._is_prepared
        assert 'x1' in prepared._selected_features  # Mandatory
        assert len(prepared._selected_features) <= 2


class TestParallelEvaluationValidation:
    """Tests for parallel evaluation parameter validation."""

    def test_invalid_n_jobs_zero(self, feature_selection_data):
        """Test that n_jobs=0 raises error."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            n_jobs=0,  # Invalid
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        # Should raise ValueError during prep
        with pytest.raises(ValueError, match="n_jobs must be -1"):
            step.prep(feature_selection_data)

    def test_invalid_n_jobs_negative_not_minus_one(self, feature_selection_data):
        """Test that n_jobs < -1 raises error."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            n_jobs=-2,  # Invalid
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        # Should raise ValueError during prep
        with pytest.raises(ValueError, match="n_jobs must be -1"):
            step.prep(feature_selection_data)
