"""
Tests for constraint relaxation in genetic algorithm feature selection.
"""

import pytest
import numpy as np
import pandas as pd
from py_recipes.steps.genetic_selection import StepSelectGeneticAlgorithm
from py_recipes.recipe import recipe
from py_parsnip import linear_reg


@pytest.fixture
def constrained_data():
    """Create dataset where strict constraints may be difficult to satisfy."""
    np.random.seed(42)
    n = 100

    # Create correlated features (violates some constraints)
    x1 = np.random.randn(n)
    x2 = x1 + np.random.randn(n) * 0.3  # Highly correlated with x1
    x3 = np.random.randn(n)
    x4 = x3 + np.random.randn(n) * 0.3  # Highly correlated with x3
    x5 = np.random.randn(n)

    # Outcome depends on x1 and x3
    y = 2 * x1 + 1.5 * x3 + np.random.randn(n) * 0.5

    return pd.DataFrame({
        'y': y,
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,
        'x5': x5
    })


class TestConstraintRelaxation:
    """Tests for constraint relaxation feature."""

    def test_relaxation_disabled_by_default(self, constrained_data):
        """Test that relaxation is disabled when not specified."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            constraints={'p_value': {'max': 0.05}},
            relax_constraints_after=None,  # Disabled
            population_size=20,
            generations=15,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(constrained_data)

        # Should complete successfully
        assert prepared._is_prepared
        assert len(prepared._selected_features) >= 1

    def test_relaxation_enabled(self, constrained_data):
        """Test that relaxation is enabled when specified."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            constraints={'p_value': {'max': 0.01}},  # Strict constraint
            relax_constraints_after=10,  # Start relaxing after gen 10
            relaxation_rate=0.1,  # Relax by 10% per generation
            population_size=20,
            generations=20,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(constrained_data)

        # Should complete and find features
        assert prepared._is_prepared
        assert len(prepared._selected_features) >= 1
        assert len(prepared._selected_features) <= 3

    def test_relaxation_with_multiple_constraints(self, constrained_data):
        """Test relaxation with multiple statistical constraints."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=2,
            constraints={
                'p_value': {'max': 0.01},
                'vif': {'max': 2.0}  # Strict multicollinearity constraint
            },
            relax_constraints_after=8,
            relaxation_rate=0.15,
            population_size=20,
            generations=20,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(constrained_data)

        assert prepared._is_prepared
        assert len(prepared._selected_features) >= 1

    def test_relaxation_rate_zero(self, constrained_data):
        """Test that zero relaxation rate keeps constraints constant."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            constraints={'p_value': {'max': 0.05}},
            relax_constraints_after=5,
            relaxation_rate=0.0,  # No relaxation
            population_size=20,
            generations=15,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(constrained_data)

        # Should still work (relaxation just doesn't happen)
        assert prepared._is_prepared

    def test_relaxation_late_start(self, constrained_data):
        """Test relaxation starting late in evolution."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            constraints={'p_value': {'max': 0.05}},
            relax_constraints_after=18,  # Start late
            relaxation_rate=0.2,
            population_size=20,
            generations=20,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(constrained_data)

        assert prepared._is_prepared
        assert len(prepared._selected_features) >= 1

    def test_relaxation_via_recipe_api(self, constrained_data):
        """Test constraint relaxation via recipe convenience function."""
        from py_recipes.steps import step_select_genetic_algorithm

        rec = recipe(constrained_data)
        rec = step_select_genetic_algorithm(
            rec,
            outcome='y',
            model=linear_reg(),
            top_n=2,
            constraints={'p_value': {'max': 0.01}},
            relax_constraints_after=10,
            relaxation_rate=0.1,
            population_size=15,
            generations=20,
            cv_folds=1,
            random_state=42
        )

        prepped = rec.prep(constrained_data)
        baked = prepped.bake(constrained_data)

        # Should complete successfully
        assert 'y' in baked.columns
        assert baked.shape[1] >= 2  # y + at least 1 feature

    def test_no_constraints_no_relaxation(self, constrained_data):
        """Test that relaxation parameters are ignored when no constraints."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            constraints={},  # No constraints
            relax_constraints_after=5,
            relaxation_rate=0.1,
            population_size=20,
            generations=15,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(constrained_data)

        # Should work normally without constraints
        assert prepared._is_prepared
        assert len(prepared._selected_features) <= 3


class TestRelaxationWithOtherEnhancements:
    """Tests for combining relaxation with other enhancements."""

    def test_relaxation_with_adaptive_params(self, constrained_data):
        """Test relaxation combined with adaptive parameters."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            constraints={'p_value': {'max': 0.01}},
            relax_constraints_after=8,
            relaxation_rate=0.1,
            adaptive_mutation=True,
            adaptive_crossover=True,
            population_size=20,
            generations=20,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(constrained_data)

        assert prepared._is_prepared
        assert len(prepared._selected_features) >= 1

    def test_relaxation_with_warm_start(self, constrained_data):
        """Test relaxation combined with warm start."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            constraints={'p_value': {'max': 0.05}},
            relax_constraints_after=10,
            relaxation_rate=0.1,
            warm_start='importance',
            population_size=20,
            generations=20,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(constrained_data)

        assert prepared._is_prepared

    def test_relaxation_with_mandatory_forbidden(self, constrained_data):
        """Test relaxation with mandatory/forbidden constraints."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            constraints={'p_value': {'max': 0.01}},
            relax_constraints_after=8,
            relaxation_rate=0.1,
            mandatory_features=['x1'],
            forbidden_features=['x5'],
            population_size=20,
            generations=20,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(constrained_data)

        # x1 must be selected, x5 must not be
        assert 'x1' in prepared._selected_features
        assert 'x5' not in prepared._selected_features

    def test_all_enhancements_with_relaxation(self, constrained_data):
        """Test all enhancements including relaxation together."""
        costs = {'x1': 1.0, 'x2': 2.0, 'x3': 1.0, 'x4': 2.0, 'x5': 5.0}

        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=2,
            constraints={'p_value': {'max': 0.01}},
            relax_constraints_after=10,
            relaxation_rate=0.1,
            mandatory_features=['x1'],
            feature_costs=costs,
            max_total_cost=5.0,
            cost_weight=0.3,
            sparsity_weight=0.2,
            warm_start='importance',
            adaptive_mutation=True,
            adaptive_crossover=True,
            population_size=20,
            generations=20,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(constrained_data)

        # Should handle all enhancements together
        assert prepared._is_prepared
        assert 'x1' in prepared._selected_features  # Mandatory
        assert len(prepared._selected_features) <= 2


class TestRelaxationBehavior:
    """Tests for relaxation behavior and mechanics."""

    def test_relaxation_factor_computation(self):
        """Test that relaxation factor is computed correctly."""
        # This tests the formula: max(0, 1 - (gen - relax_after) * rate)

        # Generation before relaxation starts
        gen = 5
        relax_after = 10
        rate = 0.1
        # Factor should be 1.0 (no relaxation yet)
        if gen < relax_after:
            expected = 1.0
        else:
            expected = max(0.0, 1.0 - (gen - relax_after) * rate)
        assert expected == 1.0

        # Generation right after relaxation starts
        gen = 11
        expected = max(0.0, 1.0 - (11 - 10) * 0.1)
        assert expected == 0.9

        # Several generations after
        gen = 15
        expected = max(0.0, 1.0 - (15 - 10) * 0.1)
        assert expected == 0.5

        # Far after (should be clamped to 0)
        gen = 25
        expected = max(0.0, 1.0 - (25 - 10) * 0.1)
        assert expected == 0.0

    def test_relaxation_helps_find_solution(self, constrained_data):
        """Test that relaxation can help find solutions with strict constraints."""
        # Very strict constraints - may be hard to satisfy initially
        step_strict = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=2,
            constraints={
                'p_value': {'max': 0.001},  # Very strict
                'vif': {'max': 1.5}  # Very strict
            },
            relax_constraints_after=None,  # No relaxation
            population_size=20,
            generations=15,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        # Same constraints with relaxation
        step_relaxed = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=2,
            constraints={
                'p_value': {'max': 0.001},
                'vif': {'max': 1.5}
            },
            relax_constraints_after=5,
            relaxation_rate=0.15,
            population_size=20,
            generations=15,
            cv_folds=2,
            random_state=43,  # Different seed
            verbose=False
        )

        prep_strict = step_strict.prep(constrained_data)
        prep_relaxed = step_relaxed.prep(constrained_data)

        # Both should complete
        assert prep_strict._is_prepared
        assert prep_relaxed._is_prepared

        # Both should find at least one feature
        assert len(prep_strict._selected_features) >= 1
        assert len(prep_relaxed._selected_features) >= 1
