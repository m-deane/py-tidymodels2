"""
Tests for genetic algorithm warm start initialization.
"""

import pytest
import numpy as np
import pandas as pd
from py_recipes.steps.genetic_selection import StepSelectGeneticAlgorithm
from py_recipes.recipe import recipe
from py_recipes.utils import (
    create_importance_based_seeds,
    create_low_correlation_seeds,
    GeneticAlgorithm,
    GAConfig
)
from py_parsnip import linear_reg


@pytest.fixture
def feature_selection_data():
    """Create dataset for testing warm start."""
    np.random.seed(42)
    n = 150

    # Important features (correlated with outcome)
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)

    # Moderately important
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


class TestSeedGeneration:
    """Tests for seed chromosome generation functions."""

    def test_create_importance_based_seeds(self, feature_selection_data):
        """Test importance-based seed creation."""
        seeds = create_importance_based_seeds(
            data=feature_selection_data,
            outcome_col='y',
            feature_names=['x1', 'x2', 'x3', 'x4', 'x5'],
            n_seeds=3,
            top_n=3
        )

        # Should return binary array
        assert seeds.shape == (3, 5)
        assert np.all(np.isin(seeds, [0, 1]))

        # Should have different numbers of features selected
        feature_counts = np.sum(seeds, axis=1)
        assert len(np.unique(feature_counts)) > 1  # Not all the same

    def test_create_low_correlation_seeds(self, feature_selection_data):
        """Test low-correlation seed creation."""
        seeds = create_low_correlation_seeds(
            data=feature_selection_data,
            feature_names=['x1', 'x2', 'x3', 'x4', 'x5'],
            n_seeds=3,
            max_corr_threshold=0.5
        )

        # Should return binary array
        assert seeds.shape == (3, 5)
        assert np.all(np.isin(seeds, [0, 1]))

        # Should have some features selected
        assert np.any(seeds)

    def test_seed_generation_with_small_dataset(self):
        """Test seed generation with minimal data."""
        data = pd.DataFrame({
            'y': [1, 2, 3, 4, 5],
            'x1': [1, 2, 3, 4, 5],
            'x2': [5, 4, 3, 2, 1]
        })

        seeds = create_importance_based_seeds(
            data=data,
            outcome_col='y',
            feature_names=['x1', 'x2'],
            n_seeds=2,
            top_n=2
        )

        assert seeds.shape == (2, 2)
        assert np.all(np.isin(seeds, [0, 1]))


class TestWarmStartInGA:
    """Tests for warm start in GeneticAlgorithm core."""

    def test_ga_with_custom_seeds(self):
        """Test GA accepts and uses custom seed chromosomes."""
        # Create simple fitness function
        def fitness_fn(chromosome):
            return np.sum(chromosome)  # Maximize number of features

        # Create custom seeds
        seeds = np.array([
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]
        ])

        ga = GeneticAlgorithm(
            n_features=3,
            fitness_function=fitness_fn,
            config=GAConfig(population_size=10, generations=5, random_state=42),
            seed_chromosomes=seeds
        )

        population = ga.initialize_population()

        # First 3 individuals should match seeds
        assert np.array_equal(population[:3], seeds)

    def test_ga_seed_validation(self):
        """Test that GA validates seed chromosomes."""
        def fitness_fn(chromosome):
            return np.sum(chromosome)

        # Wrong shape
        with pytest.raises(ValueError, match="must be 2D array"):
            ga = GeneticAlgorithm(
                n_features=3,
                fitness_function=fitness_fn,
                seed_chromosomes=np.array([1, 0, 1])  # 1D instead of 2D
            )

        # Wrong number of features
        with pytest.raises(ValueError, match="must have 3 features"):
            ga = GeneticAlgorithm(
                n_features=3,
                fitness_function=fitness_fn,
                seed_chromosomes=np.array([[1, 0]])  # Only 2 features
            )

        # Non-binary values
        with pytest.raises(ValueError, match="must be binary"):
            ga = GeneticAlgorithm(
                n_features=3,
                fitness_function=fitness_fn,
                seed_chromosomes=np.array([[1, 0, 2]])  # Has value 2
            )


class TestWarmStartInRecipeStep:
    """Tests for warm start in recipe step."""

    def test_importance_based_warm_start(self, feature_selection_data):
        """Test using importance-based warm start."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            warm_start='importance',
            population_size=20,
            generations=15,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Should complete successfully
        assert prepared._is_prepared
        assert len(prepared._selected_features) <= 3
        assert len(prepared._selected_features) >= 1

    def test_low_correlation_warm_start(self, feature_selection_data):
        """Test using low-correlation warm start."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            warm_start='low_correlation',
            population_size=20,
            generations=15,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Should complete successfully
        assert prepared._is_prepared
        assert len(prepared._selected_features) <= 3

    def test_custom_seeds_warm_start(self, feature_selection_data):
        """Test using custom seed chromosomes."""
        # Create custom seeds favoring x1 and x2
        custom_seeds = np.array([
            [1, 1, 0, 0, 0],  # x1, x2
            [1, 0, 1, 0, 0],  # x1, x3
            [0, 1, 1, 0, 0]   # x2, x3
        ])

        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=2,
            warm_start=custom_seeds,
            population_size=20,
            generations=15,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Should use custom seeds
        assert prepared._is_prepared
        assert len(prepared._selected_features) <= 2

    def test_invalid_warm_start_string(self, feature_selection_data):
        """Test that invalid warm_start string raises error."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            warm_start='invalid_method'
        )

        with pytest.raises(ValueError, match="Invalid warm_start method"):
            step.prep(feature_selection_data)

    def test_invalid_warm_start_type(self, feature_selection_data):
        """Test that invalid warm_start type raises error."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            warm_start=123  # Invalid type
        )

        with pytest.raises(ValueError, match="warm_start must be"):
            step.prep(feature_selection_data)


class TestWarmStartWithRecipe:
    """Tests for warm start via recipe API."""

    def test_warm_start_with_recipe_api(self, feature_selection_data):
        """Test warm start through recipe convenience function."""
        from py_recipes.steps import step_select_genetic_algorithm

        rec = recipe(feature_selection_data)
        rec = step_select_genetic_algorithm(
            rec,
            outcome='y',
            model=linear_reg(),
            top_n=2,
            warm_start='importance',
            population_size=15,
            generations=10,
            cv_folds=1,
            random_state=42
        )

        prepped = rec.prep(feature_selection_data)
        baked = prepped.bake(feature_selection_data)

        # Should complete successfully
        assert 'y' in baked.columns
        assert baked.shape[1] <= 3  # y + up to 2 features


class TestWarmStartPerformance:
    """Tests comparing warm start vs random initialization."""

    def test_warm_start_convergence(self, feature_selection_data):
        """Test that warm start may improve convergence speed."""
        # Run without warm start
        step_random = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            warm_start=None,
            population_size=20,
            generations=50,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        # Run with warm start
        step_warm = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            warm_start='importance',
            population_size=20,
            generations=50,
            cv_folds=2,
            random_state=43,  # Different seed
            verbose=False
        )

        prep_random = step_random.prep(feature_selection_data)
        prep_warm = step_warm.prep(feature_selection_data)

        # Both should complete successfully
        assert prep_random._is_prepared
        assert prep_warm._is_prepared

        # Both should find good solutions (not testing which is better,
        # as that's stochastic and depends on many factors)
        assert len(prep_random._selected_features) >= 1
        assert len(prep_warm._selected_features) >= 1
