"""
Tests for NSGA-II multi-objective optimization in genetic algorithm feature selection.
"""

import pytest
import numpy as np
import pandas as pd
from py_recipes.steps.genetic_selection import StepSelectGeneticAlgorithm
from py_recipes.recipe import recipe
from py_parsnip import linear_reg


@pytest.fixture
def feature_selection_data():
    """Create dataset for testing NSGA-II mode."""
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


class TestNSGA2Basic:
    """Tests for basic NSGA-II functionality."""

    def test_nsga2_two_objectives_performance_sparsity(self, feature_selection_data):
        """Test NSGA-II with performance and sparsity objectives."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            use_nsga2=True,
            nsga2_objectives=['performance', 'sparsity'],
            nsga2_selection_method='knee_point',
            population_size=20,  # Even number required for NSGA-II
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Should complete successfully
        assert prepared._is_prepared
        assert len(prepared._selected_features) >= 1
        assert prepared._pareto_front is not None
        assert prepared._pareto_objectives is not None
        assert len(prepared._pareto_front) > 0

    def test_nsga2_three_objectives_with_cost(self, feature_selection_data):
        """Test NSGA-II with three objectives including cost."""
        costs = {'x1': 1.0, 'x2': 1.0, 'x3': 2.0, 'x4': 3.0, 'x5': 5.0, 'x6': 5.0}

        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            use_nsga2=True,
            nsga2_objectives=['performance', 'sparsity', 'cost'],
            nsga2_selection_method='knee_point',
            feature_costs=costs,
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Should complete successfully
        assert prepared._is_prepared
        assert prepared._pareto_objectives.shape[1] == 3  # 3 objectives
        assert len(prepared._selected_features) >= 1

    def test_nsga2_selection_method_knee_point(self, feature_selection_data):
        """Test knee point selection method."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            use_nsga2=True,
            nsga2_selection_method='knee_point',
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        assert prepared._is_prepared
        assert len(prepared._selected_features) >= 1

    def test_nsga2_selection_method_min_features(self, feature_selection_data):
        """Test minimum features selection method."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            use_nsga2=True,
            nsga2_selection_method='min_features',
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        assert prepared._is_prepared
        # Should select solution with fewest features
        assert len(prepared._selected_features) >= 1

    def test_nsga2_selection_method_best_performance(self, feature_selection_data):
        """Test best performance selection method."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            use_nsga2=True,
            nsga2_selection_method='best_performance',
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        assert prepared._is_prepared
        assert len(prepared._selected_features) >= 1

    def test_nsga2_selection_method_index(self, feature_selection_data):
        """Test index-based selection method."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            use_nsga2=True,
            nsga2_selection_method='index',
            nsga2_selection_index=0,
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        assert prepared._is_prepared
        assert len(prepared._selected_features) >= 1


class TestNSGA2Validation:
    """Tests for NSGA-II parameter validation."""

    def test_invalid_less_than_two_objectives(self, feature_selection_data):
        """Test that less than 2 objectives raises error."""
        with pytest.raises(ValueError, match="NSGA-II requires at least 2 objectives"):
            StepSelectGeneticAlgorithm(
                outcome='y',
                model=linear_reg(),
                metric='rmse',
                use_nsga2=True,
                nsga2_objectives=['performance']  # Only 1 objective
            )

    def test_invalid_objective_name(self, feature_selection_data):
        """Test that invalid objective name raises error."""
        with pytest.raises(ValueError, match="nsga2_objectives must be from"):
            StepSelectGeneticAlgorithm(
                outcome='y',
                model=linear_reg(),
                metric='rmse',
                use_nsga2=True,
                nsga2_objectives=['performance', 'invalid_objective']
            )

    def test_cost_objective_without_feature_costs(self, feature_selection_data):
        """Test that cost objective without feature_costs raises error."""
        with pytest.raises(ValueError, match="nsga2_objectives includes 'cost' but feature_costs is not provided"):
            StepSelectGeneticAlgorithm(
                outcome='y',
                model=linear_reg(),
                metric='rmse',
                use_nsga2=True,
                nsga2_objectives=['performance', 'cost']
            )

    def test_invalid_selection_method(self, feature_selection_data):
        """Test that invalid selection method raises error."""
        with pytest.raises(ValueError, match="nsga2_selection_method must be one of"):
            StepSelectGeneticAlgorithm(
                outcome='y',
                model=linear_reg(),
                metric='rmse',
                use_nsga2=True,
                nsga2_selection_method='invalid_method'
            )

    def test_negative_selection_index(self, feature_selection_data):
        """Test that negative selection index raises error."""
        with pytest.raises(ValueError, match="nsga2_selection_index must be >= 0"):
            StepSelectGeneticAlgorithm(
                outcome='y',
                model=linear_reg(),
                metric='rmse',
                use_nsga2=True,
                nsga2_selection_method='index',
                nsga2_selection_index=-1
            )

    def test_nsga2_incompatible_with_ensemble(self, feature_selection_data):
        """Test that NSGA-II and ensemble mode are incompatible."""
        with pytest.raises(ValueError, match="Ensemble mode .* is not compatible with NSGA-II"):
            StepSelectGeneticAlgorithm(
                outcome='y',
                model=linear_reg(),
                metric='rmse',
                use_nsga2=True,
                n_ensemble=3  # Ensemble mode
            )


class TestNSGA2Integration:
    """Tests for NSGA-II integration with other features."""

    def test_nsga2_with_top_n(self, feature_selection_data):
        """Test NSGA-II with top_n constraint."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            use_nsga2=True,
            top_n=3,
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        assert prepared._is_prepared
        assert len(prepared._selected_features) <= 3

    def test_nsga2_via_recipe_api(self, feature_selection_data):
        """Test NSGA-II via recipe convenience function."""
        from py_recipes.steps import step_select_genetic_algorithm

        rec = recipe(feature_selection_data)
        rec = step_select_genetic_algorithm(
            rec,
            outcome='y',
            model=linear_reg(),
            use_nsga2=True,
            nsga2_objectives=['performance', 'sparsity'],
            population_size=20,
            generations=10,
            cv_folds=1,
            random_state=42
        )

        prepped = rec.prep(feature_selection_data)
        baked = prepped.bake(feature_selection_data)

        # Should complete successfully
        assert 'y' in baked.columns
        assert baked.shape[1] >= 2  # y + at least 1 feature

    def test_nsga2_pareto_front_storage(self, feature_selection_data):
        """Test that Pareto front is properly stored."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            use_nsga2=True,
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Check Pareto front structure
        assert prepared._pareto_front is not None
        assert prepared._pareto_objectives is not None
        assert isinstance(prepared._pareto_indices, list)

        # Pareto front should have solutions
        assert len(prepared._pareto_front) > 0
        assert prepared._pareto_objectives.shape[0] == len(prepared._pareto_front)
        assert prepared._pareto_objectives.shape[1] == 2  # 2 objectives (performance, sparsity)

    def test_nsga2_maximization_metric(self, feature_selection_data):
        """Test NSGA-II with maximization metric (R²)."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='r_squared',
            maximize=True,  # Maximize R²
            use_nsga2=True,
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        assert prepared._is_prepared
        assert len(prepared._selected_features) >= 1


class TestNSGA2Output:
    """Tests for NSGA-II output structure."""

    def test_pareto_front_is_diverse(self, feature_selection_data):
        """Test that Pareto front contains diverse solutions."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            use_nsga2=True,
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Pareto front should have multiple solutions
        assert len(prepared._pareto_front) >= 2

        # Solutions should have varying numbers of features
        feature_counts = np.sum(prepared._pareto_front, axis=1)
        assert len(np.unique(feature_counts)) >= 2  # At least 2 different feature counts

    def test_knee_point_finding(self, feature_selection_data):
        """Test knee point finding algorithm."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            use_nsga2=True,
            nsga2_selection_method='knee_point',
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Knee point should be found
        assert len(prepared._selected_features) >= 1

        # Should be a reasonable trade-off (not extremes)
        # This is a heuristic check
        n_features = len(prepared._selected_features)
        total_features = len([c for c in feature_selection_data.columns if c != 'y'])

        # Knee point shouldn't select all or zero features (in most cases)
        assert n_features > 0
        assert n_features < total_features
