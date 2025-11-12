"""
Tests for genetic algorithm feature selection recipe step.
"""

import pytest
import numpy as np
import pandas as pd
from py_recipes.steps.genetic_selection import (
    StepSelectGeneticAlgorithm,
    step_select_genetic_algorithm
)
from py_recipes.recipe import recipe
from py_parsnip import linear_reg


@pytest.fixture
def simple_regression_data():
    """Create simple regression dataset with relevant and noise features."""
    np.random.seed(42)
    n = 150

    # Relevant features
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)

    # Noise features
    x3 = np.random.randn(n)
    x4 = np.random.randn(n)
    x5 = np.random.randn(n)

    # Outcome: y = 2*x1 + 3*x2 + noise
    y = 2 * x1 + 3 * x2 + np.random.randn(n) * 0.5

    return pd.DataFrame({
        'y': y,
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,
        'x5': x5
    })


class TestStepSelectGeneticAlgorithm:
    """Tests for StepSelectGeneticAlgorithm class."""

    def test_initialization(self):
        """Test step initialization with valid parameters."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            population_size=20,
            generations=10
        )

        assert step.outcome == 'y'
        assert step.metric == 'rmse'
        assert step.top_n == 3
        assert step.population_size == 20
        assert step.generations == 10
        assert not step._is_prepared

    def test_invalid_model_parameter(self):
        """Test that missing model raises error."""
        with pytest.raises(ValueError, match="model parameter is required"):
            StepSelectGeneticAlgorithm(
                outcome='y',
                model=None
            )

    def test_invalid_mutation_rate(self):
        """Test that mutation_rate out of range raises error."""
        with pytest.raises(ValueError, match="mutation_rate must be in"):
            StepSelectGeneticAlgorithm(
                outcome='y',
                model=linear_reg(),
                mutation_rate=1.5
            )

    def test_invalid_top_n(self):
        """Test that top_n < 1 raises error."""
        with pytest.raises(ValueError, match="top_n must be >= 1"):
            StepSelectGeneticAlgorithm(
                outcome='y',
                model=linear_reg(),
                top_n=0
            )

    def test_prep_basic(self, simple_regression_data):
        """Test prep() method with basic parameters."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            population_size=20,
            generations=15,
            cv_folds=2,  # Fast CV
            random_state=42,
            verbose=False
        )

        prepared = step.prep(simple_regression_data)

        # Check prepared state
        assert prepared._is_prepared
        assert len(prepared._selected_features) == 3
        assert len(prepared._ga_history) > 0
        assert prepared._final_fitness > 0
        assert prepared._best_chromosome is not None

    def test_prep_selects_relevant_features(self, simple_regression_data):
        """Test that GA tends to select relevant features (x1, x2)."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=2,
            population_size=30,
            generations=20,
            cv_folds=3,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(simple_regression_data)
        selected = prepared._selected_features

        # x1 and x2 should likely be selected (probabilistic test)
        assert len(selected) == 2
        # At least one relevant feature should be selected
        assert 'x1' in selected or 'x2' in selected

    def test_bake(self, simple_regression_data):
        """Test bake() method filters to selected features."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            top_n=2,
            population_size=15,
            generations=10,
            cv_folds=1,  # No CV for speed
            random_state=42
        )

        prepared = step.prep(simple_regression_data)
        baked = prepared.bake(simple_regression_data)

        # Should have outcome + 2 features
        assert baked.shape[1] == 3
        assert 'y' in baked.columns
        assert len([c for c in baked.columns if c != 'y']) == 2

    def test_bake_without_prep_raises_error(self, simple_regression_data):
        """Test that bake() without prep() raises error."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg()
        )

        with pytest.raises(RuntimeError, match="must be prepared before baking"):
            step.bake(simple_regression_data)

    def test_get_selected_features(self, simple_regression_data):
        """Test get_selected_features() returns feature list."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            top_n=2,
            population_size=15,
            generations=10,
            cv_folds=1,
            random_state=42
        )

        prepared = step.prep(simple_regression_data)
        features = prepared.get_selected_features()

        assert isinstance(features, list)
        assert len(features) == 2

    def test_get_fitness_history(self, simple_regression_data):
        """Test get_fitness_history() returns history."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            top_n=2,
            population_size=15,
            generations=10,
            cv_folds=1,
            random_state=42
        )

        prepared = step.prep(simple_regression_data)
        history = prepared.get_fitness_history()

        assert isinstance(history, list)
        assert len(history) > 0
        assert all(isinstance(f, float) for f in history)

    def test_convergence(self, simple_regression_data):
        """Test that GA can converge early."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            top_n=2,
            population_size=20,
            generations=100,  # Large number
            convergence_threshold=0.001,
            convergence_patience=5,
            cv_folds=1,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(simple_regression_data)

        # Should converge before max generations
        assert prepared._converged
        assert prepared._n_generations < 100

    def test_with_constraints_p_value(self, simple_regression_data):
        """Test with p-value constraint."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=2,
            constraints={
                'p_value': {'max': 0.05, 'method': 'none'}
            },
            population_size=20,
            generations=15,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(simple_regression_data)

        # Should complete without errors
        assert prepared._is_prepared
        assert len(prepared._selected_features) == 2

    def test_maximize_metric(self, simple_regression_data):
        """Test with maximize=True for R²."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='r_squared',
            maximize=True,
            top_n=2,
            population_size=15,
            generations=10,
            cv_folds=1,
            random_state=42
        )

        prepared = step.prep(simple_regression_data)

        assert prepared._is_prepared
        assert prepared._final_fitness > 0


class TestStepSelectGeneticAlgorithmWithRecipe:
    """Tests for integration with py_recipes."""

    def test_convenience_function(self, simple_regression_data):
        """Test step_select_genetic_algorithm() convenience function."""
        rec = recipe(simple_regression_data)
        rec = step_select_genetic_algorithm(
            rec,
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=2,
            population_size=15,
            generations=10,
            cv_folds=1,
            random_state=42
        )

        assert len(rec.steps) == 1
        assert isinstance(rec.steps[0], StepSelectGeneticAlgorithm)

    def test_with_recipe_prep_and_bake(self, simple_regression_data):
        """Test full recipe workflow: prep + bake."""
        rec = recipe(simple_regression_data)
        rec = step_select_genetic_algorithm(
            rec,
            outcome='y',
            model=linear_reg(),
            top_n=2,
            population_size=15,
            generations=10,
            cv_folds=1,
            random_state=42
        )

        # Prep recipe
        prepped = rec.prep(simple_regression_data)

        # Bake on same data
        baked = prepped.bake(simple_regression_data)

        # Should have outcome + 2 features
        assert baked.shape[1] == 3
        assert 'y' in baked.columns

        # Bake on new data (simulated)
        test_data = simple_regression_data.copy()
        baked_test = prepped.bake(test_data)

        assert baked_test.shape[1] == 3
        assert 'y' in baked_test.columns

    def test_chained_with_other_steps(self, simple_regression_data):
        """Test chaining with other recipe steps."""
        from py_recipes.steps import StepNormalize

        rec = recipe(simple_regression_data)

        # Add normalization before GA selection
        rec = rec.step_normalize()

        # Add GA selection
        rec = step_select_genetic_algorithm(
            rec,
            outcome='y',
            model=linear_reg(),
            top_n=2,
            population_size=15,
            generations=10,
            cv_folds=1,
            random_state=42
        )

        assert len(rec.steps) == 2
        assert isinstance(rec.steps[0], StepNormalize)
        assert isinstance(rec.steps[1], StepSelectGeneticAlgorithm)

        # Prep and bake
        prepped = rec.prep(simple_regression_data)
        baked = prepped.bake(simple_regression_data)

        # Should have outcome + 2 normalized features
        assert baked.shape[1] == 3


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_outcome_column(self, simple_regression_data):
        """Test that missing outcome raises error."""
        step = StepSelectGeneticAlgorithm(
            outcome='nonexistent',
            model=linear_reg(),
            population_size=10,
            generations=5
        )

        with pytest.raises(ValueError, match="Outcome .* not found"):
            step.prep(simple_regression_data)

    def test_no_numeric_columns(self):
        """Test that non-numeric data raises error."""
        data = pd.DataFrame({
            'y': [1, 2, 3],
            'x1': ['a', 'b', 'c'],
            'x2': ['d', 'e', 'f']
        })

        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            population_size=10,
            generations=5
        )

        with pytest.raises(ValueError, match="No numeric columns"):
            step.prep(data)

    def test_skip_parameter(self, simple_regression_data):
        """Test that skip=True bypasses prep."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            skip=True
        )

        result = step.prep(simple_regression_data)

        # Should return self without preparation
        assert not result._is_prepared
        assert result is step


class TestVerboseOutput:
    """Tests for verbose output."""

    def test_verbose_prints_progress(self, simple_regression_data, capsys):
        """Test that verbose=True prints progress."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            top_n=2,
            population_size=10,
            generations=5,
            cv_folds=1,
            random_state=42,
            verbose=True
        )

        step.prep(simple_regression_data)

        captured = capsys.readouterr()
        assert "Genetic Algorithm Feature Selection" in captured.out
        assert "Candidate features:" in captured.out
        assert "GA Complete:" in captured.out
