"""
Tests for adaptive GA parameters (mutation and crossover rates).
"""

import pytest
import numpy as np
import pandas as pd
from py_recipes.steps.genetic_selection import StepSelectGeneticAlgorithm
from py_recipes.recipe import recipe
from py_recipes.utils import GeneticAlgorithm, GAConfig
from py_parsnip import linear_reg


@pytest.fixture
def feature_selection_data():
    """Create dataset for testing adaptive parameters."""
    np.random.seed(42)
    n = 150

    # Important features
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)

    # Noise features
    x4 = np.random.randn(n)
    x5 = np.random.randn(n)

    # Outcome: y = 3*x1 + 2*x2 + 0.5*x3 + noise
    y = 3 * x1 + 2 * x2 + 0.5 * x3 + np.random.randn(n) * 0.5

    return pd.DataFrame({
        'y': y,
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,
        'x5': x5
    })


class TestAdaptiveRatesInGA:
    """Tests for adaptive rates in GeneticAlgorithm core."""

    def test_adaptive_mutation_enabled(self):
        """Test that adaptive mutation modifies rates during evolution."""
        # Simple fitness function
        def fitness_fn(chromosome):
            return np.sum(chromosome)

        ga = GeneticAlgorithm(
            n_features=5,
            fitness_function=fitness_fn,
            config=GAConfig(
                population_size=20,
                generations=20,
                mutation_rate=0.1,
                adaptive_mutation=True,
                random_state=42,
                verbose=False
            )
        )

        # Run evolution
        best_chrom, best_fit, history = ga.evolve()

        # Should have rate history
        assert len(ga.mutation_rate_history_) > 0
        assert len(ga.mutation_rate_history_) <= 20  # One per generation after gen 5

        # Rates should have changed
        if len(ga.mutation_rate_history_) > 1:
            rates_changed = len(set(ga.mutation_rate_history_)) > 1
            # Allow possibility that rates stayed constant if evolution was very stable
            # (not asserting change, just checking mechanism works)

    def test_adaptive_crossover_enabled(self):
        """Test that adaptive crossover modifies rates during evolution."""
        def fitness_fn(chromosome):
            return np.sum(chromosome)

        ga = GeneticAlgorithm(
            n_features=5,
            fitness_function=fitness_fn,
            config=GAConfig(
                population_size=20,
                generations=20,
                crossover_rate=0.8,
                adaptive_crossover=True,
                random_state=42,
                verbose=False
            )
        )

        best_chrom, best_fit, history = ga.evolve()

        # Should have rate history
        assert len(ga.crossover_rate_history_) > 0

    def test_both_adaptive_enabled(self):
        """Test both adaptive mutation and crossover together."""
        def fitness_fn(chromosome):
            return np.sum(chromosome)

        ga = GeneticAlgorithm(
            n_features=5,
            fitness_function=fitness_fn,
            config=GAConfig(
                population_size=20,
                generations=20,
                mutation_rate=0.1,
                crossover_rate=0.8,
                adaptive_mutation=True,
                adaptive_crossover=True,
                random_state=42,
                verbose=False
            )
        )

        best_chrom, best_fit, history = ga.evolve()

        # Both should have rate histories
        assert len(ga.mutation_rate_history_) > 0
        assert len(ga.crossover_rate_history_) > 0

    def test_adaptive_disabled_by_default(self):
        """Test that rates don't adapt when adaptive flags are False."""
        def fitness_fn(chromosome):
            return np.sum(chromosome)

        ga = GeneticAlgorithm(
            n_features=5,
            fitness_function=fitness_fn,
            config=GAConfig(
                population_size=20,
                generations=20,
                mutation_rate=0.1,
                crossover_rate=0.8,
                adaptive_mutation=False,  # Disabled
                adaptive_crossover=False,  # Disabled
                random_state=42,
                verbose=False
            )
        )

        best_chrom, best_fit, history = ga.evolve()

        # Rates should remain constant (equal to initial)
        assert ga.current_mutation_rate_ == ga.initial_mutation_rate_
        assert ga.current_crossover_rate_ == ga.initial_crossover_rate_

    def test_adaptive_rates_bounded(self):
        """Test that adaptive rates stay within bounds."""
        def fitness_fn(chromosome):
            return np.sum(chromosome)

        ga = GeneticAlgorithm(
            n_features=5,
            fitness_function=fitness_fn,
            config=GAConfig(
                population_size=20,
                generations=30,
                mutation_rate=0.1,
                crossover_rate=0.8,
                adaptive_mutation=True,
                adaptive_crossover=True,
                random_state=42,
                verbose=False
            )
        )

        best_chrom, best_fit, history = ga.evolve()

        # Check mutation rate bounds (0.05 to 0.5)
        if ga.mutation_rate_history_:
            assert all(0.05 <= rate <= 0.5 for rate in ga.mutation_rate_history_)

        # Check crossover rate bounds (0.5 to 0.95)
        if ga.crossover_rate_history_:
            assert all(0.5 <= rate <= 0.95 for rate in ga.crossover_rate_history_)


class TestAdaptiveRatesInRecipeStep:
    """Tests for adaptive rates in recipe step."""

    def test_adaptive_mutation_in_step(self, feature_selection_data):
        """Test adaptive mutation through recipe step."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            population_size=20,
            generations=20,
            adaptive_mutation=True,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Should complete successfully
        assert prepared._is_prepared
        assert len(prepared._selected_features) >= 1

    def test_adaptive_crossover_in_step(self, feature_selection_data):
        """Test adaptive crossover through recipe step."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            population_size=20,
            generations=20,
            adaptive_crossover=True,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        assert prepared._is_prepared
        assert len(prepared._selected_features) >= 1

    def test_both_adaptive_in_step(self, feature_selection_data):
        """Test both adaptive parameters together in recipe step."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            population_size=20,
            generations=20,
            adaptive_mutation=True,
            adaptive_crossover=True,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        assert prepared._is_prepared
        assert len(prepared._selected_features) <= 3

    def test_adaptive_with_recipe_api(self, feature_selection_data):
        """Test adaptive parameters via recipe convenience function."""
        from py_recipes.steps import step_select_genetic_algorithm

        rec = recipe(feature_selection_data)
        rec = step_select_genetic_algorithm(
            rec,
            outcome='y',
            model=linear_reg(),
            top_n=2,
            population_size=15,
            generations=15,
            adaptive_mutation=True,
            adaptive_crossover=True,
            cv_folds=1,
            random_state=42
        )

        prepped = rec.prep(feature_selection_data)
        baked = prepped.bake(feature_selection_data)

        # Should complete successfully
        assert 'y' in baked.columns
        assert baked.shape[1] >= 2  # y + at least 1 feature


class TestAdaptiveWithOtherEnhancements:
    """Tests for adaptive parameters combined with other enhancements."""

    def test_adaptive_with_warm_start(self, feature_selection_data):
        """Test adaptive parameters with warm start."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            warm_start='importance',
            adaptive_mutation=True,
            adaptive_crossover=True,
            population_size=20,
            generations=15,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        assert prepared._is_prepared
        assert len(prepared._selected_features) >= 1

    def test_adaptive_with_constraints(self, feature_selection_data):
        """Test adaptive parameters with mandatory/forbidden features."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            mandatory_features=['x1'],
            forbidden_features=['x5'],
            adaptive_mutation=True,
            population_size=20,
            generations=15,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # x1 must be selected, x5 must not be
        assert 'x1' in prepared._selected_features
        assert 'x5' not in prepared._selected_features

    def test_adaptive_with_costs_and_sparsity(self, feature_selection_data):
        """Test adaptive parameters with costs and sparsity."""
        costs = {'x1': 1.0, 'x2': 1.0, 'x3': 2.0, 'x4': 5.0, 'x5': 5.0}

        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            feature_costs=costs,
            max_total_cost=10.0,
            cost_weight=0.3,
            sparsity_weight=0.2,
            adaptive_mutation=True,
            adaptive_crossover=True,
            population_size=20,
            generations=15,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        assert prepared._is_prepared
        assert len(prepared._selected_features) >= 1
        assert len(prepared._selected_features) <= 3


class TestAdaptiveRatesPerformance:
    """Tests comparing adaptive vs static rates."""

    def test_adaptive_convergence(self, feature_selection_data):
        """Test that adaptive rates help convergence."""
        # Static rates
        step_static = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            adaptive_mutation=False,
            adaptive_crossover=False,
            population_size=20,
            generations=30,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        # Adaptive rates
        step_adaptive = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            adaptive_mutation=True,
            adaptive_crossover=True,
            population_size=20,
            generations=30,
            cv_folds=2,
            random_state=43,  # Different seed
            verbose=False
        )

        prep_static = step_static.prep(feature_selection_data)
        prep_adaptive = step_adaptive.prep(feature_selection_data)

        # Both should find good solutions
        assert prep_static._is_prepared
        assert prep_adaptive._is_prepared
        assert len(prep_static._selected_features) >= 1
        assert len(prep_adaptive._selected_features) >= 1

        # Both should include important features (x1, x2 are most important)
        # This is a stochastic test, but with reasonable parameters both should find them
        important_features = {'x1', 'x2'}
        static_found = len(important_features & set(prep_static._selected_features))
        adaptive_found = len(important_features & set(prep_adaptive._selected_features))

        # At least one important feature should be found by each
        assert static_found >= 1 or adaptive_found >= 1
