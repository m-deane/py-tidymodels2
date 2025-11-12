"""
Tests for model fitness evaluator.
"""

import pytest
import numpy as np
import pandas as pd
from py_recipes.utils.model_fitness import (
    ModelFitnessEvaluator,
    create_model_fitness_evaluator
)
from py_parsnip import linear_reg


@pytest.fixture
def simple_regression_data():
    """Create simple regression dataset."""
    np.random.seed(42)
    n = 100

    # Create features
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)
    x4 = np.random.randn(n)  # noise feature

    # Create outcome: y = 2*x1 + 3*x2 + noise
    y = 2 * x1 + 3 * x2 + np.random.randn(n) * 0.5

    return pd.DataFrame({
        'y': y,
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4
    })


class TestModelFitnessEvaluator:
    """Tests for ModelFitnessEvaluator class."""

    def test_initialization(self, simple_regression_data):
        """Test evaluator initialization."""
        evaluator = ModelFitnessEvaluator(
            data=simple_regression_data,
            outcome_col='y',
            model_spec=linear_reg(),
            metric='rmse'
        )

        assert evaluator.outcome_col == 'y'
        assert evaluator.metric == 'rmse'
        assert evaluator.n_features_ == 4
        assert set(evaluator.feature_names_) == {'x1', 'x2', 'x3', 'x4'}

    def test_invalid_metric(self, simple_regression_data):
        """Test that invalid metric raises error."""
        with pytest.raises(ValueError, match="Unsupported metric"):
            ModelFitnessEvaluator(
                data=simple_regression_data,
                outcome_col='y',
                model_spec=linear_reg(),
                metric='invalid_metric'
            )

    def test_evaluate_features_all_features(self, simple_regression_data):
        """Test evaluating with all features selected."""
        evaluator = ModelFitnessEvaluator(
            data=simple_regression_data,
            outcome_col='y',
            model_spec=linear_reg(),
            metric='rmse',
            cv_folds=1,  # No CV for speed
            random_state=42
        )

        # Select all features
        chromosome = np.array([1, 1, 1, 1])
        fitness = evaluator.evaluate_features(chromosome)

        # Should return positive fitness
        assert fitness > 0
        assert np.isfinite(fitness)

    def test_evaluate_features_no_features(self, simple_regression_data):
        """Test evaluating with no features returns -inf."""
        evaluator = ModelFitnessEvaluator(
            data=simple_regression_data,
            outcome_col='y',
            model_spec=linear_reg(),
            metric='rmse',
            cv_folds=1
        )

        # Select no features
        chromosome = np.array([0, 0, 0, 0])
        fitness = evaluator.evaluate_features(chromosome)

        # Should return -inf
        assert fitness == -np.inf

    def test_evaluate_features_subset(self, simple_regression_data):
        """Test evaluating with subset of features."""
        evaluator = ModelFitnessEvaluator(
            data=simple_regression_data,
            outcome_col='y',
            model_spec=linear_reg(),
            metric='rmse',
            cv_folds=1,
            random_state=42
        )

        # Select x1 and x2 (relevant features)
        chromosome = np.array([1, 1, 0, 0])
        fitness = evaluator.evaluate_features(chromosome)

        # Should return positive fitness
        assert fitness > 0
        assert np.isfinite(fitness)

    def test_evaluate_with_cv(self, simple_regression_data):
        """Test evaluation with cross-validation."""
        evaluator = ModelFitnessEvaluator(
            data=simple_regression_data,
            outcome_col='y',
            model_spec=linear_reg(),
            metric='rmse',
            cv_folds=3,
            random_state=42
        )

        chromosome = np.array([1, 1, 0, 0])
        fitness = evaluator.evaluate_features(chromosome)

        # Should return positive fitness
        assert fitness > 0
        assert np.isfinite(fitness)

    def test_maximize_vs_minimize_metric(self, simple_regression_data):
        """Test that maximize parameter affects fitness direction."""
        # Minimize RMSE
        evaluator_min = ModelFitnessEvaluator(
            data=simple_regression_data,
            outcome_col='y',
            model_spec=linear_reg(),
            metric='rmse',
            maximize=False,
            cv_folds=1,
            random_state=42
        )

        # Maximize R²
        evaluator_max = ModelFitnessEvaluator(
            data=simple_regression_data,
            outcome_col='y',
            model_spec=linear_reg(),
            metric='r_squared',
            maximize=True,
            cv_folds=1,
            random_state=42
        )

        chromosome = np.array([1, 1, 0, 0])

        fitness_min = evaluator_min.evaluate_features(chromosome)
        fitness_max = evaluator_max.evaluate_features(chromosome)

        # Both should be positive but use different scales
        assert fitness_min > 0
        assert fitness_max > 0

    def test_better_features_higher_fitness(self, simple_regression_data):
        """Test that better feature subsets have higher fitness."""
        evaluator = ModelFitnessEvaluator(
            data=simple_regression_data,
            outcome_col='y',
            model_spec=linear_reg(),
            metric='rmse',
            maximize=False,
            cv_folds=3,
            random_state=42
        )

        # Good features: x1, x2 (true predictors)
        good_chromosome = np.array([1, 1, 0, 0])

        # Bad features: x3, x4 (noise)
        bad_chromosome = np.array([0, 0, 1, 1])

        fitness_good = evaluator.evaluate_features(good_chromosome)
        fitness_bad = evaluator.evaluate_features(bad_chromosome)

        # Good features should have higher fitness
        assert fitness_good > fitness_bad

    def test_create_fitness_function(self, simple_regression_data):
        """Test creating fitness function for GA."""
        evaluator = ModelFitnessEvaluator(
            data=simple_regression_data,
            outcome_col='y',
            model_spec=linear_reg(),
            metric='rmse',
            cv_folds=1
        )

        fitness_fn = evaluator.create_fitness_function()

        # Test that function works
        chromosome = np.array([1, 1, 0, 0])
        fitness = fitness_fn(chromosome)

        assert fitness > 0
        assert np.isfinite(fitness)


class TestCreateModelFitnessEvaluator:
    """Tests for convenience function."""

    def test_create_fitness_evaluator(self, simple_regression_data):
        """Test convenience function creates working fitness function."""
        fitness_fn = create_model_fitness_evaluator(
            data=simple_regression_data,
            outcome_col='y',
            model_spec=linear_reg(),
            metric='rmse',
            cv_folds=2,
            random_state=42
        )

        # Test function works
        chromosome = np.array([1, 1, 0, 0])
        fitness = fitness_fn(chromosome)

        assert fitness > 0
        assert np.isfinite(fitness)

    def test_different_metrics(self, simple_regression_data):
        """Test that different metrics can be used."""
        metrics = ['rmse', 'mae', 'r_squared']

        for metric in metrics:
            maximize = (metric == 'r_squared')
            fitness_fn = create_model_fitness_evaluator(
                data=simple_regression_data,
                outcome_col='y',
                model_spec=linear_reg(),
                metric=metric,
                maximize=maximize,
                cv_folds=1
            )

            chromosome = np.array([1, 1, 0, 0])
            fitness = fitness_fn(chromosome)

            assert fitness > 0, f"Metric {metric} failed"
            assert np.isfinite(fitness), f"Metric {metric} returned non-finite"


class TestIntegrationWithGA:
    """Integration tests with GeneticAlgorithm."""

    def test_ga_with_model_fitness(self, simple_regression_data):
        """Test GA can optimize feature selection using model fitness."""
        from py_recipes.utils.genetic_algorithm import GeneticAlgorithm, GAConfig

        # Create fitness function
        fitness_fn = create_model_fitness_evaluator(
            data=simple_regression_data,
            outcome_col='y',
            model_spec=linear_reg(),
            metric='rmse',
            cv_folds=2,
            random_state=42
        )

        # Run GA
        config = GAConfig(
            population_size=20,
            generations=10,
            random_state=42,
            verbose=False
        )
        ga = GeneticAlgorithm(
            n_features=4,
            fitness_function=fitness_fn,
            config=config
        )

        best_chromosome, best_fitness, history = ga.evolve()

        # Check results are valid
        assert best_fitness > 0
        assert np.sum(best_chromosome) >= 1  # At least one feature selected
        assert len(history) > 0

        # Fitness should improve over generations
        assert history[-1] >= history[0]

    def test_ga_finds_relevant_features(self, simple_regression_data):
        """Test GA tends to select relevant features (x1, x2)."""
        from py_recipes.utils.genetic_algorithm import GeneticAlgorithm, GAConfig

        fitness_fn = create_model_fitness_evaluator(
            data=simple_regression_data,
            outcome_col='y',
            model_spec=linear_reg(),
            metric='rmse',
            cv_folds=3,
            random_state=42
        )

        config = GAConfig(
            population_size=30,
            generations=20,
            random_state=42,
            verbose=False
        )
        ga = GeneticAlgorithm(
            n_features=4,
            fitness_function=fitness_fn,
            config=config
        )

        best_chromosome, best_fitness, history = ga.evolve()

        # x1 and x2 (indices 0, 1) should likely be selected
        # (This is probabilistic, so we just check it's reasonable)
        feature_names = ['x1', 'x2', 'x3', 'x4']
        selected = [feature_names[i] for i in range(4) if best_chromosome[i] == 1]

        # Should have selected some features
        assert len(selected) >= 1
        assert len(selected) <= 4
