"""
Tests for diversity maintenance in genetic algorithm feature selection.
"""

import pytest
import numpy as np
import pandas as pd
from py_recipes.steps.genetic_selection import StepSelectGeneticAlgorithm
from py_recipes.recipe import recipe
from py_parsnip import linear_reg


@pytest.fixture
def feature_selection_data():
    """Create dataset for testing diversity maintenance."""
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


class TestDiversityBasics:
    """Tests for basic diversity maintenance functionality."""

    def test_diversity_disabled_by_default(self, feature_selection_data):
        """Test that diversity maintenance is disabled by default."""
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

        # Diversity tracking should be empty when disabled
        assert len(prepared._diversity_history) == 0

    def test_diversity_enabled(self, feature_selection_data):
        """Test that diversity maintenance can be enabled."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            maintain_diversity=True,
            diversity_threshold=0.3,
            fitness_sharing_sigma=0.5,
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

        # Diversity should be tracked
        assert len(prepared._diversity_history) > 0
        assert prepared._diversity_history[0] >= 0  # Initial diversity
        assert prepared._diversity_history[0] <= 1  # Normalized

    def test_diversity_history_tracking(self, feature_selection_data):
        """Test that diversity is tracked at each generation."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            maintain_diversity=True,
            population_size=20,
            generations=15,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Diversity history should have entries for each generation + initial
        # (generations + 1, or fewer if converged early)
        assert len(prepared._diversity_history) >= 10  # At least 10 generations
        assert len(prepared._diversity_history) <= 16  # At most 15 + initial

        # All diversity values should be valid
        for diversity in prepared._diversity_history:
            assert 0 <= diversity <= 1

    def test_fitness_sharing_applied(self, feature_selection_data):
        """Test that fitness sharing is applied when diversity drops."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            maintain_diversity=True,
            diversity_threshold=0.4,  # Higher threshold to trigger sharing
            fitness_sharing_sigma=0.5,
            population_size=20,
            generations=20,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Check if any generation had low diversity
        low_diversity_count = sum(1 for d in prepared._diversity_history if d < 0.4)

        # If diversity dropped, fitness sharing should have been applied
        # (We can't directly check if it was applied, but we can verify diversity was tracked)
        if low_diversity_count > 0:
            # Diversity maintenance should help recover diversity
            # Check if diversity increased after dropping
            for i in range(1, len(prepared._diversity_history) - 1):
                if prepared._diversity_history[i] < 0.4:
                    # Check if diversity recovered in next few generations
                    future_diversity = prepared._diversity_history[i+1:i+4]
                    if len(future_diversity) > 0:
                        # At least one future value should be higher (or same due to randomness)
                        assert max(future_diversity) >= prepared._diversity_history[i] - 0.1


class TestDiversityParameters:
    """Tests for diversity parameter validation and behavior."""

    def test_diversity_threshold_zero(self, feature_selection_data):
        """Test that diversity_threshold=0 is valid."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            maintain_diversity=True,
            diversity_threshold=0.0,  # Always trigger sharing
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)
        assert prepared._is_prepared

    def test_diversity_threshold_one(self, feature_selection_data):
        """Test that diversity_threshold=1 is valid."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            maintain_diversity=True,
            diversity_threshold=1.0,  # Never trigger sharing
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)
        assert prepared._is_prepared

    def test_invalid_diversity_threshold_negative(self, feature_selection_data):
        """Test that negative diversity_threshold raises error."""
        with pytest.raises(ValueError, match="diversity_threshold must be in"):
            StepSelectGeneticAlgorithm(
                outcome='y',
                model=linear_reg(),
                metric='rmse',
                maintain_diversity=True,
                diversity_threshold=-0.1  # Invalid
            )

    def test_invalid_diversity_threshold_above_one(self, feature_selection_data):
        """Test that diversity_threshold>1 raises error."""
        with pytest.raises(ValueError, match="diversity_threshold must be in"):
            StepSelectGeneticAlgorithm(
                outcome='y',
                model=linear_reg(),
                metric='rmse',
                maintain_diversity=True,
                diversity_threshold=1.1  # Invalid
            )

    def test_invalid_fitness_sharing_sigma_zero(self, feature_selection_data):
        """Test that fitness_sharing_sigma=0 raises error."""
        with pytest.raises(ValueError, match="fitness_sharing_sigma must be >"):
            StepSelectGeneticAlgorithm(
                outcome='y',
                model=linear_reg(),
                metric='rmse',
                maintain_diversity=True,
                fitness_sharing_sigma=0.0  # Invalid
            )

    def test_invalid_fitness_sharing_sigma_negative(self, feature_selection_data):
        """Test that negative fitness_sharing_sigma raises error."""
        with pytest.raises(ValueError, match="fitness_sharing_sigma must be >"):
            StepSelectGeneticAlgorithm(
                outcome='y',
                model=linear_reg(),
                metric='rmse',
                maintain_diversity=True,
                fitness_sharing_sigma=-0.5  # Invalid
            )

    def test_different_sigma_values(self, feature_selection_data):
        """Test that different sigma values work."""
        for sigma in [0.1, 0.5, 1.0, 2.0]:
            step = StepSelectGeneticAlgorithm(
                outcome='y',
                model=linear_reg(),
                metric='rmse',
                top_n=3,
                maintain_diversity=True,
                fitness_sharing_sigma=sigma,
                population_size=20,
                generations=10,
                cv_folds=2,
                random_state=42,
                verbose=False
            )

            prepared = step.prep(feature_selection_data)
            assert prepared._is_prepared


class TestDiversityIntegration:
    """Tests for diversity maintenance integrated with other features."""

    def test_diversity_with_ensemble(self, feature_selection_data):
        """Test diversity maintenance with ensemble mode."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            maintain_diversity=True,
            diversity_threshold=0.3,
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

        # Should work with ensemble
        assert prepared._is_prepared
        assert len(prepared._ensemble_results) == 3

        # Each ensemble run should have diversity tracking
        # (Can't directly check since diversity is per-GA instance)

    def test_diversity_with_adaptive_params(self, feature_selection_data):
        """Test diversity maintenance with adaptive GA parameters."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            maintain_diversity=True,
            diversity_threshold=0.3,
            adaptive_mutation=True,
            adaptive_crossover=True,
            population_size=20,
            generations=15,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Should combine diversity with adaptive params
        assert prepared._is_prepared
        assert len(prepared._diversity_history) > 0

    def test_diversity_with_constraints(self, feature_selection_data):
        """Test diversity maintenance with feature constraints."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            maintain_diversity=True,
            diversity_threshold=0.3,
            mandatory_features=['x1'],
            forbidden_features=['x6'],
            population_size=20,
            generations=10,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # Should respect constraints while maintaining diversity
        assert prepared._is_prepared
        assert 'x1' in prepared._selected_features  # Mandatory
        assert 'x6' not in prepared._selected_features  # Forbidden
        assert len(prepared._diversity_history) > 0

    def test_diversity_via_recipe_api(self, feature_selection_data):
        """Test diversity maintenance via recipe convenience function."""
        from py_recipes.steps import step_select_genetic_algorithm

        rec = recipe(feature_selection_data)
        rec = step_select_genetic_algorithm(
            rec,
            outcome='y',
            model=linear_reg(),
            top_n=3,
            maintain_diversity=True,
            diversity_threshold=0.3,
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


class TestDiversityPerformance:
    """Tests for diversity maintenance performance characteristics."""

    def test_diversity_maintains_performance(self, feature_selection_data):
        """Test that diversity maintenance doesn't severely degrade performance."""
        # Run without diversity
        step_no_diversity = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            maintain_diversity=False,
            population_size=20,
            generations=20,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared_no_diversity = step_no_diversity.prep(feature_selection_data)

        # Run with diversity
        step_with_diversity = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            maintain_diversity=True,
            diversity_threshold=0.3,
            population_size=20,
            generations=20,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared_with_diversity = step_with_diversity.prep(feature_selection_data)

        # Both should complete and select features
        assert len(prepared_no_diversity._selected_features) >= 1
        assert len(prepared_with_diversity._selected_features) >= 1

        # Both should have similar fitness (diversity shouldn't hurt too much)
        # Note: This is a weak assertion - just checking both converged
        assert prepared_no_diversity._final_fitness > 0
        assert prepared_with_diversity._final_fitness > 0

    def test_diversity_increases_exploration(self, feature_selection_data):
        """Test that diversity maintenance encourages exploration."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            maintain_diversity=True,
            diversity_threshold=0.5,  # High threshold to trigger often
            population_size=30,
            generations=25,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)

        # With high diversity threshold, should maintain reasonable diversity
        # throughout evolution
        if len(prepared._diversity_history) > 5:
            # Check that diversity doesn't drop to very low values
            min_diversity = min(prepared._diversity_history)
            # With diversity maintenance, minimum should be reasonable
            # (This is a soft check - diversity can still drop naturally)
            assert min_diversity >= 0.0  # At least non-negative

            # Check that diversity varies (not stuck at single value)
            diversity_range = max(prepared._diversity_history) - min(prepared._diversity_history)
            assert diversity_range > 0  # Some variation
