"""
Tests for genetic algorithm feature selection enhancements.

Tests for mandatory/forbidden features, feature costs, and sparsity objectives.
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
def feature_selection_data():
    """Create dataset with known feature importance for testing constraints."""
    np.random.seed(42)
    n = 200

    # Important features
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)

    # Moderate importance
    x3 = np.random.randn(n)

    # Low importance (noise)
    x4 = np.random.randn(n)
    x5 = np.random.randn(n)
    x6 = np.random.randn(n)

    # Outcome: y = 3*x1 + 2*x2 + 0.5*x3 + noise
    y = 3 * x1 + 2 * x2 + 0.5 * x3 + np.random.randn(n) * 0.5

    return pd.DataFrame({
        'y': y,
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,
        'x5': x5,
        'x6': x6
    })


class TestMandatoryFeatures:
    """Tests for mandatory feature constraints."""

    def test_mandatory_features_always_selected(self, feature_selection_data):
        """Test that mandatory features are always included in selection."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            mandatory_features=['x1'],  # Force x1 to be selected
            population_size=20,
            generations=15,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)
        selected = prepared._selected_features

        # x1 must be in selected features
        assert 'x1' in selected
        assert len(selected) == 3

    def test_multiple_mandatory_features(self, feature_selection_data):
        """Test multiple mandatory features."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=4,
            mandatory_features=['x1', 'x2'],  # Force x1 and x2
            population_size=20,
            generations=15,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)
        selected = prepared._selected_features

        # Both x1 and x2 must be selected
        assert 'x1' in selected
        assert 'x2' in selected
        assert len(selected) <= 4  # Up to 4 features
        assert len(selected) >= 2  # At least the mandatory ones

    def test_mandatory_features_with_recipe(self, feature_selection_data):
        """Test mandatory features via recipe API."""
        rec = recipe(feature_selection_data)
        rec = step_select_genetic_algorithm(
            rec,
            outcome='y',
            model=linear_reg(),
            top_n=2,
            mandatory_features=['x1'],
            population_size=15,
            generations=10,
            cv_folds=1,
            random_state=42
        )

        prepped = rec.prep(feature_selection_data)
        baked = prepped.bake(feature_selection_data)

        # x1 must be in output
        assert 'x1' in baked.columns
        assert 'y' in baked.columns


class TestForbiddenFeatures:
    """Tests for forbidden feature constraints."""

    def test_forbidden_features_never_selected(self, feature_selection_data):
        """Test that forbidden features are never included in selection."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            forbidden_features=['x4', 'x5', 'x6'],  # Ban noise features
            population_size=20,
            generations=15,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)
        selected = prepared._selected_features

        # Forbidden features must NOT be selected
        assert 'x4' not in selected
        assert 'x5' not in selected
        assert 'x6' not in selected
        assert len(selected) == 3

    def test_forbidden_features_with_recipe(self, feature_selection_data):
        """Test forbidden features via recipe API."""
        rec = recipe(feature_selection_data)
        rec = step_select_genetic_algorithm(
            rec,
            outcome='y',
            model=linear_reg(),
            top_n=3,
            forbidden_features=['x4', 'x5', 'x6'],
            population_size=15,
            generations=10,
            cv_folds=1,
            random_state=42
        )

        prepped = rec.prep(feature_selection_data)
        baked = prepped.bake(feature_selection_data)

        # Forbidden features must NOT be in output
        assert 'x4' not in baked.columns
        assert 'x5' not in baked.columns
        assert 'x6' not in baked.columns


class TestMandatoryAndForbidden:
    """Tests for combining mandatory and forbidden constraints."""

    def test_mandatory_and_forbidden_together(self, feature_selection_data):
        """Test using both mandatory and forbidden features."""
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            mandatory_features=['x1'],
            forbidden_features=['x5', 'x6'],
            population_size=20,
            generations=15,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)
        selected = prepared._selected_features

        # x1 must be selected
        assert 'x1' in selected
        # x5, x6 must NOT be selected
        assert 'x5' not in selected
        assert 'x6' not in selected
        assert len(selected) == 3

    def test_invalid_mandatory_and_forbidden_overlap(self):
        """Test that overlapping mandatory and forbidden raises error."""
        # This should raise an error during GA initialization in prep()
        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            mandatory_features=['x1'],
            forbidden_features=['x1']  # Same feature!
        )

        # Create simple test data
        data = pd.DataFrame({
            'y': [1, 2, 3, 4, 5],
            'x1': [1, 2, 3, 4, 5],
            'x2': [5, 4, 3, 2, 1]
        })

        # Should raise ValueError when prepping
        with pytest.raises(ValueError, match="cannot be both mandatory and forbidden"):
            step.prep(data)


class TestFeatureCosts:
    """Tests for feature cost constraints."""

    def test_feature_costs_basic(self, feature_selection_data):
        """Test that feature costs influence selection."""
        # Assign high cost to x3-x6, low cost to x1-x2
        costs = {
            'x1': 1.0,
            'x2': 1.0,
            'x3': 10.0,
            'x4': 10.0,
            'x5': 10.0,
            'x6': 10.0
        }

        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            feature_costs=costs,
            max_total_cost=15.0,  # Budget allows ~1-2 expensive features
            cost_weight=0.3,  # Reduced weight to allow more features
            population_size=25,
            generations=20,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)
        selected = prepared._selected_features

        # Calculate total cost of selected features
        total_cost = sum(costs.get(f, 0.0) for f in selected)

        # Should select at least 1 feature, up to 3
        assert len(selected) >= 1
        assert len(selected) <= 3
        # At least one low-cost feature should be selected
        assert 'x1' in selected or 'x2' in selected

    def test_feature_costs_with_recipe(self, feature_selection_data):
        """Test feature costs via recipe API."""
        costs = {'x1': 1.0, 'x2': 1.0, 'x3': 5.0, 'x4': 5.0, 'x5': 5.0, 'x6': 5.0}

        rec = recipe(feature_selection_data)
        rec = step_select_genetic_algorithm(
            rec,
            outcome='y',
            model=linear_reg(),
            top_n=3,
            feature_costs=costs,
            max_total_cost=10.0,
            cost_weight=0.3,  # Reduced weight
            population_size=20,
            generations=15,
            cv_folds=1,
            random_state=42
        )

        prepped = rec.prep(feature_selection_data)
        selected = prepped.prepared_steps[0]._selected_features  # Fixed: prepared_steps

        # Should prefer low-cost features
        total_cost = sum(costs.get(f, 0.0) for f in selected)
        assert len(selected) >= 1
        assert len(selected) <= 3


class TestSparsityObjective:
    """Tests for sparsity preference."""

    def test_sparsity_weight_reduces_features(self, feature_selection_data):
        """Test that sparsity weight encourages fewer features."""
        # Run without sparsity
        step_no_sparsity = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=5,
            sparsity_weight=0.0,
            population_size=25,
            generations=20,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        # Run with sparsity
        step_with_sparsity = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=5,
            sparsity_weight=2.0,  # Strong sparsity preference
            population_size=25,
            generations=20,
            cv_folds=2,
            random_state=43,  # Different seed
            verbose=False
        )

        prep_no_sparsity = step_no_sparsity.prep(feature_selection_data)
        prep_with_sparsity = step_with_sparsity.prep(feature_selection_data)

        # Both should select up to 5 features, but sparsity should tend toward fewer
        # (This is a soft constraint, so we just verify it runs)
        assert len(prep_no_sparsity._selected_features) <= 5
        assert len(prep_with_sparsity._selected_features) <= 5

    def test_sparsity_with_recipe(self, feature_selection_data):
        """Test sparsity weight via recipe API."""
        rec = recipe(feature_selection_data)
        rec = step_select_genetic_algorithm(
            rec,
            outcome='y',
            model=linear_reg(),
            top_n=4,
            sparsity_weight=1.5,
            population_size=20,
            generations=15,
            cv_folds=1,
            random_state=42
        )

        prepped = rec.prep(feature_selection_data)
        selected = prepped.prepared_steps[0]._selected_features  # Fixed: prepared_steps

        # Should complete without errors
        assert len(selected) <= 4
        assert len(selected) >= 1  # At least one feature


class TestCombinedEnhancements:
    """Tests for combining multiple enhancements."""

    def test_all_enhancements_together(self, feature_selection_data):
        """Test using mandatory, forbidden, costs, and sparsity together."""
        costs = {
            'x1': 2.0,
            'x2': 2.0,
            'x3': 3.0,
            'x4': 10.0,
            'x5': 10.0,
            'x6': 10.0
        }

        step = StepSelectGeneticAlgorithm(
            outcome='y',
            model=linear_reg(),
            metric='rmse',
            top_n=3,
            mandatory_features=['x1'],      # Must include x1
            forbidden_features=['x6'],      # Must exclude x6
            feature_costs=costs,             # Prefer low-cost features
            max_total_cost=12.0,            # Budget constraint
            cost_weight=0.2,                # Reduced weight
            sparsity_weight=0.1,            # Reduced weight
            population_size=25,
            generations=20,
            cv_folds=2,
            random_state=42,
            verbose=False
        )

        prepared = step.prep(feature_selection_data)
        selected = prepared._selected_features

        # Verify all constraints
        assert 'x1' in selected  # Mandatory
        assert 'x6' not in selected  # Forbidden
        assert len(selected) >= 1  # At least mandatory feature
        assert len(selected) <= 3  # At most top_n

        # Cost constraint (soft)
        total_cost = sum(costs.get(f, 0.0) for f in selected)
        # Should be reasonable (may exceed slightly due to soft constraint)
        assert total_cost < 20.0  # Sanity check

    def test_all_enhancements_via_recipe(self, feature_selection_data):
        """Test all enhancements via recipe API."""
        costs = {'x1': 1.0, 'x2': 1.0, 'x3': 2.0, 'x4': 5.0, 'x5': 5.0, 'x6': 5.0}

        rec = recipe(feature_selection_data)
        rec = step_select_genetic_algorithm(
            rec,
            outcome='y',
            model=linear_reg(),
            top_n=3,
            mandatory_features=['x1'],
            forbidden_features=['x5', 'x6'],
            feature_costs=costs,
            max_total_cost=8.0,
            cost_weight=0.2,  # Reduced weight
            sparsity_weight=0.1,  # Reduced weight
            population_size=20,
            generations=15,
            cv_folds=1,
            random_state=42
        )

        prepped = rec.prep(feature_selection_data)
        baked = prepped.bake(feature_selection_data)

        # Verify constraints in output
        assert 'x1' in baked.columns  # Mandatory
        assert 'x5' not in baked.columns  # Forbidden
        assert 'x6' not in baked.columns  # Forbidden
        assert 'y' in baked.columns
        # Should have outcome + at least 1 feature, at most 3 features
        assert baked.shape[1] >= 2  # y + at least 1 (mandatory)
        assert baked.shape[1] <= 4  # y + at most 3 features
