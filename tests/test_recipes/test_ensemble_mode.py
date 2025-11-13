"""
Tests for ensemble mode in genetic algorithm feature selection.
"""

import pytest
import numpy as np
import pandas as pd
from py_recipes.steps.genetic_selection import StepSelectGeneticAlgorithm
from py_recipes.recipe import recipe
from py_parsnip import linear_reg


@pytest.fixture
def feature_selection_data():
    """Create dataset for testing ensemble mode."""
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


class TestEnsembleMode:
    """Tests for ensemble mode functionality."""

    def test_single_run_default(self, feature_selection_data):
        """Test that single run is default (n_ensemble=1)."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            n_ensemble=1,  # Default
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Should complete successfully
        assert prepared._is_prepared
        assert len(prepared._selected_features) <= 3
        assert len(prepared._ensemble_results) == 0  # No ensemble results

    def test_ensemble_voting_strategy(self, feature_selection_data):
        """Test ensemble with voting strategy."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            n_ensemble=5,  # 5 GA runs
            ensemble_strategy='voting',
            ensemble_threshold=0.6,  # Need 60%+ (3+ runs)
            population_size=15,
            generations=8,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Should complete successfully
        assert prepared._is_prepared
        assert len(prepared._ensemble_results) == 5
        assert len(prepared._feature_frequencies) > 0

        # Check that selected features appear in majority of runs
        for feat in prepared._selected_features:
            freq = prepared._feature_frequencies[feat]
            assert freq >= 3  # 60% of 5 = 3

    def test_ensemble_frequency_strategy(self, feature_selection_data):
        """Test ensemble with frequency strategy (same as voting)."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            n_ensemble=4,
            ensemble_strategy='frequency',
            ensemble_threshold=0.5,  # Need 50%+ (2+ runs)
            population_size=15,
            generations=8,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        assert prepared._is_prepared
        assert len(prepared._ensemble_results) == 4

        # Check that selected features appear in at least 50% of runs
        for feat in prepared._selected_features:
            freq = prepared._feature_frequencies[feat]
            assert freq >= 2  # 50% of 4 = 2

    def test_ensemble_union_strategy(self, feature_selection_data):
        """Test ensemble with union strategy (any run)."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            n_ensemble=3,
            ensemble_strategy='union',  # All features from any run
            population_size=15,
            generations=8,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        assert prepared._is_prepared
        assert len(prepared._ensemble_results) == 3

        # Union includes all features that appeared in at least one run
        for feat in prepared._selected_features:
            assert prepared._feature_frequencies[feat] >= 1

    def test_ensemble_intersection_strategy(self, feature_selection_data):
        """Test ensemble with intersection strategy (all runs)."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            n_ensemble=3,
            ensemble_strategy='intersection',  # Only features in ALL runs
            population_size=15,
            generations=8,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        assert prepared._is_prepared
        assert len(prepared._ensemble_results) == 3

        # Intersection only includes features that appeared in ALL runs
        for feat in prepared._selected_features:
            assert prepared._feature_frequencies[feat] == 3

    def test_ensemble_different_seeds(self, feature_selection_data):
        """Test that ensemble uses different seeds for each run."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            n_ensemble=4,
            population_size=15,
            generations=8,
            cv_folds=2,
            random_state=100,  # Base seed
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Check that each run used a different seed
        seeds = [result['seed'] for result in prepared._ensemble_results]
        assert len(set(seeds)) == 4  # All unique
        assert seeds == [100, 101, 102, 103]  # Sequential from base seed

    def test_ensemble_results_stored(self, feature_selection_data):
        """Test that individual run results are stored."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            n_ensemble=3,
            population_size=15,
            generations=8,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Check ensemble results structure
        assert len(prepared._ensemble_results) == 3
        for i, result in enumerate(prepared._ensemble_results):
            assert result['run_idx'] == i
            assert 'seed' in result
            assert 'chromosome' in result
            assert 'fitness' in result
            assert 'history' in result
            assert 'features' in result
            assert 'converged' in result
            assert 'n_generations' in result

    def test_ensemble_via_recipe_api(self, feature_selection_data):
        """Test ensemble mode via recipe convenience function."""
        from py_recipes.steps import step_select_genetic_algorithm

        rec = recipe(feature_selection_data)
        rec = step_select_genetic_algorithm(
            rec,
            outcome='y',
            model=linear_reg(),
            top_n=3,
            n_ensemble=3,  # Ensemble mode
            ensemble_strategy='voting',
            ensemble_threshold=0.6,
            population_size=15,
            generations=8,
            cv_folds=1,
            random_state=42
        )

        prepped = rec.prep(feature_selection_data)
        baked = prepped.bake(feature_selection_data)

        # Should complete successfully
        assert 'y' in baked.columns
        assert baked.shape[1] >= 2  # y + at least 1 feature


class TestEnsembleValidation:
    """Tests for ensemble parameter validation."""

    def test_invalid_n_ensemble_zero(self, feature_selection_data):
        """Test that n_ensemble=0 raises error."""
        with pytest.raises(ValueError, match="n_ensemble must be >= 1"):
            StepSelectGeneticAlgorithm(
                outcome='y',
                model=linear_reg(),
                metric='rmse',
                n_ensemble=0  # Invalid
            )

    def test_invalid_n_ensemble_negative(self, feature_selection_data):
        """Test that negative n_ensemble raises error."""
        with pytest.raises(ValueError, match="n_ensemble must be >= 1"):
            StepSelectGeneticAlgorithm(
                outcome='y',
                model=linear_reg(),
                metric='rmse',
                n_ensemble=-1  # Invalid
            )

    def test_invalid_ensemble_strategy(self, feature_selection_data):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError, match="ensemble_strategy must be one of"):
            StepSelectGeneticAlgorithm(
                outcome='y',
                model=linear_reg(),
                metric='rmse',
                n_ensemble=3,
                ensemble_strategy='invalid'  # Invalid
            )

    def test_invalid_ensemble_threshold_zero(self, feature_selection_data):
        """Test that threshold=0 raises error."""
        with pytest.raises(ValueError, match="ensemble_threshold must be in"):
            StepSelectGeneticAlgorithm(
                outcome='y',
                model=linear_reg(),
                metric='rmse',
                n_ensemble=3,
                ensemble_threshold=0.0  # Invalid
            )

    def test_invalid_ensemble_threshold_above_one(self, feature_selection_data):
        """Test that threshold>1 raises error."""
        with pytest.raises(ValueError, match="ensemble_threshold must be in"):
            StepSelectGeneticAlgorithm(
                outcome='y',
                model=linear_reg(),
                metric='rmse',
                n_ensemble=3,
                ensemble_threshold=1.1  # Invalid
            )


class TestEnsembleWithOtherEnhancements:
    """Tests for ensemble mode combined with other enhancements."""

    def test_ensemble_with_constraints(self, feature_selection_data):
        """Test ensemble with statistical constraints."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            constraints={'p_value': {'max': 0.05}},
            n_ensemble=3,
            ensemble_strategy='voting',
            ensemble_threshold=0.6,
            population_size=15,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Should handle constraints in ensemble mode
        assert prepared._is_prepared
        assert len(prepared._ensemble_results) == 3

    def test_ensemble_with_adaptive_params(self, feature_selection_data):
        """Test ensemble with adaptive GA parameters."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            adaptive_mutation=True,
            adaptive_crossover=True,
            n_ensemble=3,
            population_size=15,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Should combine adaptive params with ensemble
        assert prepared._is_prepared
        assert len(prepared._ensemble_results) == 3

    def test_ensemble_with_warm_start(self, feature_selection_data):
        """Test ensemble with warm start initialization."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            warm_start='importance',
            n_ensemble=3,
            population_size=15,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Should combine warm start with ensemble
        assert prepared._is_prepared
        assert len(prepared._ensemble_results) == 3

    def test_all_enhancements_with_ensemble(self, feature_selection_data):
        """Test all enhancements including ensemble together."""
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
            # Enhancement #8: Ensemble Mode
            n_ensemble=3,
            ensemble_strategy='voting',
            ensemble_threshold=0.6,
            population_size=15,
            generations=15,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Should handle all enhancements together
        assert prepared._is_prepared
        assert 'x1' in prepared._selected_features  # Mandatory
        assert len(prepared._selected_features) <= 2
        assert len(prepared._ensemble_results) == 3
