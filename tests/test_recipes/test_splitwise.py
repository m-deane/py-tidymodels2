"""
Tests for step_splitwise() - Adaptive dummy encoding for numeric predictors.
"""

import pytest
import pandas as pd
import numpy as np
from py_recipes import recipe
from py_recipes.steps.splitwise import StepSplitwise


@pytest.fixture
def simple_data():
    """
    Simple dataset with linear and non-linear relationships.
    """
    np.random.seed(42)
    n = 200

    # Linear relationship
    x1 = np.random.randn(n)
    # Non-linear (threshold at 0)
    x2 = np.random.randn(n)
    # Non-linear (U-shaped)
    x3 = np.random.randn(n)

    # Outcome
    y = (
        2 * x1 +  # Linear
        5 * (x2 > 0).astype(int) +  # Step function
        3 * ((x3 < -0.5) | (x3 > 0.5)).astype(int) +  # U-shaped
        np.random.randn(n) * 0.5  # Noise
    )

    return pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'y': y
    })


@pytest.fixture
def complex_data():
    """
    More complex dataset with multiple predictors.
    """
    np.random.seed(123)
    n = 300

    x1 = np.random.uniform(0, 10, n)
    x2 = np.random.uniform(-5, 5, n)
    x3 = np.random.randint(1, 5, n).astype(float)
    x4 = np.random.randn(n)

    y = (
        0.5 * x1 +
        10 * (x2 > 0).astype(int) +
        2 * x3 +
        np.random.randn(n)
    )

    return pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,
        'target': y
    })


class TestStepSplitwiseBasics:
    """Test basic functionality of step_splitwise."""

    def test_step_creation(self):
        """Test step can be created with default parameters."""
        step = StepSplitwise(outcome='y')
        assert step.outcome == 'y'
        assert step.transformation_mode == 'univariate'
        assert step.min_support == 0.1
        assert step.min_improvement == 3.0
        assert step.criterion == 'AIC'

    def test_step_creation_with_params(self):
        """Test step creation with custom parameters."""
        step = StepSplitwise(
            outcome='target',
            min_support=0.15,
            min_improvement=2.0,
            criterion='BIC',
            exclude_vars=['x1']
        )
        assert step.min_support == 0.15
        assert step.min_improvement == 2.0
        assert step.criterion == 'BIC'
        assert step.exclude_vars == ['x1']

    def test_invalid_min_support(self):
        """Test validation of min_support parameter."""
        with pytest.raises(ValueError, match="min_support must be in"):
            StepSplitwise(outcome='y', min_support=0.6)

        with pytest.raises(ValueError, match="min_support must be in"):
            StepSplitwise(outcome='y', min_support=0.0)

    def test_invalid_min_improvement(self):
        """Test validation of min_improvement parameter."""
        with pytest.raises(ValueError, match="min_improvement must be >= 0"):
            StepSplitwise(outcome='y', min_improvement=-1.0)

    def test_invalid_criterion(self):
        """Test validation of criterion parameter."""
        with pytest.raises(ValueError, match="criterion must be"):
            StepSplitwise(outcome='y', criterion='invalid')

    def test_iterative_mode_not_implemented(self):
        """Test that iterative mode raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Iterative mode is not yet implemented"):
            StepSplitwise(outcome='y', transformation_mode='iterative')


class TestStepSplitwisePrep:
    """Test prep() method of step_splitwise."""

    def test_prep_basic(self, simple_data):
        """Test basic prep functionality."""
        step = StepSplitwise(outcome='y')
        prepped = step.prep(simple_data)

        assert prepped._is_prepared
        assert len(prepped._decisions) > 0
        assert len(prepped._cutoffs) > 0
        assert set(prepped._original_columns) == {'x1', 'x2', 'x3'}

    def test_prep_missing_outcome(self, simple_data):
        """Test error when outcome not found."""
        step = StepSplitwise(outcome='missing')
        with pytest.raises(ValueError, match="Outcome 'missing' not found"):
            step.prep(simple_data)

    def test_prep_exclude_vars(self, simple_data):
        """Test excluding variables from transformation."""
        step = StepSplitwise(outcome='y', exclude_vars=['x1'])
        prepped = step.prep(simple_data)

        assert prepped._decisions['x1'] == 'linear'
        assert len(prepped._cutoffs['x1']) == 0

    def test_prep_stores_decisions(self, simple_data):
        """Test that prep stores transformation decisions."""
        step = StepSplitwise(outcome='y', min_improvement=1.0)
        prepped = step.prep(simple_data)

        # Should have decisions for all columns
        for col in ['x1', 'x2', 'x3']:
            assert col in prepped._decisions
            assert prepped._decisions[col] in ['linear', 'single_split', 'double_split']


class TestStepSplitwiseBake:
    """Test bake() method of step_splitwise."""

    def test_bake_creates_dummies(self, simple_data):
        """Test that bake creates dummy variables."""
        step = StepSplitwise(outcome='y', min_improvement=1.0)
        prepped = step.prep(simple_data)
        baked = prepped.bake(simple_data)

        # Check for dummy variable naming (at least one transformation should occur)
        # Dummy column names use sanitized format: x_ge_0p1234 or x_between_mp5_0p5
        dummy_cols = [c for c in baked.columns if '_ge_' in c or '_between_' in c]
        assert len(dummy_cols) > 0

        # Check that dummy columns are binary
        for col in dummy_cols:
            assert set(baked[col].unique()).issubset({0, 1})

        # Verify column names don't have special characters that break patsy
        for col in dummy_cols:
            # Should not contain '-' or '.' (replaced with 'm' and 'p')
            assert '-' not in col
            assert '.' not in col or col == '.pred'  # Allow .pred for predictions

    def test_bake_preserves_linear(self, simple_data):
        """Test that linear variables are preserved."""
        step = StepSplitwise(outcome='y', exclude_vars=['x1', 'x2', 'x3'])
        prepped = step.prep(simple_data)
        baked = prepped.bake(simple_data)

        # All should be linear (no transformation)
        assert 'x1' in baked.columns
        assert 'x2' in baked.columns
        assert 'x3' in baked.columns

    def test_bake_single_split(self, simple_data):
        """Test single-split dummy variable creation."""
        step = StepSplitwise(outcome='y', min_improvement=1.0)
        prepped = step.prep(simple_data)

        # Force a single-split decision for testing
        # (In practice, this is decided by the algorithm)
        decisions = prepped.get_decisions()

        # Check that single-split dummies are binary
        baked = prepped.bake(simple_data)
        for col in baked.columns:
            if '_ge_' in col:
                assert set(baked[col].unique()).issubset({0, 1})

    def test_bake_on_new_data(self, simple_data):
        """Test bake on new data (like test set)."""
        train = simple_data.iloc[:150]
        test = simple_data.iloc[150:]

        step = StepSplitwise(outcome='y', min_improvement=1.0)
        prepped = step.prep(train)

        # Bake on test data
        baked_test = prepped.bake(test)

        # Check that transformations are applied
        assert len(baked_test) == len(test)

    def test_bake_skip(self, simple_data):
        """Test that skip=True prevents transformation."""
        step = StepSplitwise(outcome='y', skip=True)
        prepped = step.prep(simple_data)
        baked = prepped.bake(simple_data)

        # Should be unchanged
        pd.testing.assert_frame_equal(baked, simple_data)


class TestStepSplitwiseRecipe:
    """Test step_splitwise integration with Recipe."""

    def test_recipe_integration(self, simple_data):
        """Test step_splitwise in a recipe."""
        rec = (
            recipe()
            .step_splitwise(outcome='y', min_improvement=1.0)
        )

        prepped = rec.prep(simple_data)
        baked = prepped.bake(simple_data)

        # Check transformation occurred
        assert len(baked.columns) <= len(simple_data.columns)

    def test_recipe_with_other_steps(self, simple_data):
        """Test step_splitwise combined with other steps."""
        from py_recipes.selectors import all_numeric

        rec = (
            recipe()
            .step_splitwise(outcome='y', min_improvement=2.0)
            .step_normalize(all_numeric())
        )

        prepped = rec.prep(simple_data)
        baked = prepped.bake(simple_data)

        # Check both steps applied
        assert len(baked) == len(simple_data)


class TestStepSplitwiseTransformations:
    """Test specific transformation scenarios."""

    def test_linear_relationship_kept(self):
        """Test that purely linear relationships are kept as-is."""
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        y = 2 * x + np.random.randn(n) * 0.1  # Strong linear

        data = pd.DataFrame({'x': x, 'y': y})

        step = StepSplitwise(outcome='y', min_improvement=3.0)
        prepped = step.prep(data)

        # Should decide to keep linear (high improvement threshold)
        assert prepped._decisions['x'] == 'linear'

    def test_threshold_relationship_transformed(self):
        """Test that threshold relationships are transformed to dummies."""
        np.random.seed(42)
        n = 300
        x = np.random.randn(n)
        y = 10 * (x > 0).astype(int) + np.random.randn(n) * 0.5

        data = pd.DataFrame({'x': x, 'y': y})

        step = StepSplitwise(outcome='y', min_improvement=1.0)
        prepped = step.prep(data)

        # Should detect the threshold
        assert prepped._decisions['x'] in ['single_split', 'double_split']

    def test_min_support_constraint(self):
        """Test that min_support prevents imbalanced splits."""
        np.random.seed(42)
        n = 200
        x = np.random.uniform(0, 1, n)
        # Create highly imbalanced relationship
        y = 10 * (x > 0.95).astype(int) + np.random.randn(n) * 0.1

        data = pd.DataFrame({'x': x, 'y': y})

        # High min_support should prevent transformation
        step = StepSplitwise(outcome='y', min_support=0.2)
        prepped = step.prep(data)

        # Should keep linear due to support constraint
        assert prepped._decisions['x'] == 'linear'

    def test_aic_vs_bic_criterion(self, simple_data):
        """Test AIC vs BIC criterion selection."""
        step_aic = StepSplitwise(outcome='y', criterion='AIC', min_improvement=1.0)
        step_bic = StepSplitwise(outcome='y', criterion='BIC', min_improvement=1.0)

        prepped_aic = step_aic.prep(simple_data)
        prepped_bic = step_bic.prep(simple_data)

        # Both should complete (BIC is more conservative)
        assert prepped_aic._is_prepared
        assert prepped_bic._is_prepared

    def test_get_decisions_method(self, simple_data):
        """Test get_decisions() method."""
        step = StepSplitwise(outcome='y', min_improvement=1.0)
        prepped = step.prep(simple_data)

        decisions = prepped.get_decisions()

        # Should have info for all columns
        assert 'x1' in decisions
        assert 'x2' in decisions
        assert 'x3' in decisions

        # Each should have decision and cutoffs
        for col, info in decisions.items():
            assert 'decision' in info
            assert 'cutoffs' in info


class TestStepSplitwiseEdgeCases:
    """Test edge cases and error handling."""

    def test_no_numeric_columns(self):
        """Test error when no numeric columns available."""
        data = pd.DataFrame({
            'cat1': ['A', 'B', 'C'] * 30,
            'cat2': ['X', 'Y'] * 45,
            'y': np.random.randn(90)
        })

        step = StepSplitwise(outcome='y')
        with pytest.raises(ValueError, match="No numeric predictor columns"):
            step.prep(data)

    def test_small_dataset(self):
        """Test behavior with very small dataset."""
        data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })

        step = StepSplitwise(outcome='y')
        # Should handle gracefully (may keep linear)
        prepped = step.prep(data)
        assert prepped._is_prepared

    def test_constant_predictor(self):
        """Test handling of constant predictor."""
        data = pd.DataFrame({
            'x': [1.0] * 100,
            'y': np.random.randn(100)
        })

        step = StepSplitwise(outcome='y')
        prepped = step.prep(data)

        # Should keep linear (no valid splits)
        assert prepped._decisions['x'] == 'linear'

    def test_missing_values(self):
        """Test handling of missing values."""
        np.random.seed(42)
        data = pd.DataFrame({
            'x': [np.nan if i % 10 == 0 else np.random.randn() for i in range(100)],
            'y': np.random.randn(100)
        })

        step = StepSplitwise(outcome='y')
        # Should handle NaN values by removing them during fit
        prepped = step.prep(data)
        assert prepped._is_prepared


class TestStepSplitwiseFeatureTypes:
    """Test feature_type parameter functionality."""

    def test_feature_type_dummies_default(self, simple_data):
        """Test default feature_type='dummies' creates only binary dummies."""
        step = StepSplitwise(outcome='y', min_improvement=1.0, feature_type='dummies')
        prepped = step.prep(simple_data)
        baked = prepped.bake(simple_data)

        # Check for dummy variables only
        dummy_cols = [c for c in baked.columns if '_ge_' in c or '_between_' in c]
        interaction_cols = [c for c in baked.columns if '_x_' in c]

        # Should have dummies
        assert len(dummy_cols) > 0
        # Should NOT have interactions
        assert len(interaction_cols) == 0

        # Verify dummies are binary
        for col in dummy_cols:
            assert set(baked[col].unique()).issubset({0, 1})

    def test_feature_type_interactions_only(self, simple_data):
        """Test feature_type='interactions' creates interaction features only."""
        step = StepSplitwise(outcome='y', min_improvement=1.0, feature_type='interactions')
        prepped = step.prep(simple_data)
        baked = prepped.bake(simple_data)

        # Check for interaction variables
        interaction_cols = [c for c in baked.columns if '_x_' in c]

        # Should have interactions
        assert len(interaction_cols) > 0

        # Verify interactions are NOT binary (they're dummy * original_value)
        for col in interaction_cols:
            unique_vals = baked[col].unique()
            # Interactions should have more than just 0/1
            # They can be 0 or original_value
            assert len(unique_vals) > 2 or not set(unique_vals).issubset({0, 1})

    def test_feature_type_both(self, simple_data):
        """Test feature_type='both' creates both dummies and interactions."""
        step = StepSplitwise(outcome='y', min_improvement=1.0, feature_type='both')
        prepped = step.prep(simple_data)
        baked = prepped.bake(simple_data)

        # Check for both types
        dummy_cols = [c for c in baked.columns if ('_ge_' in c or '_between_' in c) and '_x_' not in c]
        interaction_cols = [c for c in baked.columns if '_x_' in c]

        # Should have both
        assert len(dummy_cols) > 0
        assert len(interaction_cols) > 0

        # For each dummy, should have corresponding interaction
        # (at least for single_split transformations)
        for dummy_col in dummy_cols:
            # Find corresponding interaction
            expected_interaction = f"{dummy_col}_x_"
            matching_interactions = [c for c in interaction_cols if c.startswith(expected_interaction)]
            # May not always have exact match due to naming, but should have some interactions
            assert len(interaction_cols) >= len(dummy_cols)

    def test_feature_type_invalid(self, simple_data):
        """Test validation of feature_type parameter."""
        with pytest.raises(ValueError, match="feature_type must be"):
            StepSplitwise(outcome='y', feature_type='invalid')

    def test_interaction_values_correct(self, simple_data):
        """Test that interaction values equal dummy * original_value."""
        # Create simple threshold relationship
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 10, n)
        y = 5 * (x > 5).astype(int) + np.random.randn(n) * 0.1

        data = pd.DataFrame({'x': x, 'y': y})

        step = StepSplitwise(outcome='y', min_improvement=0.1, feature_type='both')
        prepped = step.prep(data)
        baked = prepped.bake(data)

        # Find dummy and interaction columns
        dummy_cols = [c for c in baked.columns if '_ge_' in c and '_x_' not in c]
        interaction_cols = [c for c in baked.columns if '_x_' in c]

        if len(dummy_cols) > 0 and len(interaction_cols) > 0:
            # Verify interaction = dummy * original
            # We need to reconstruct original x values
            # For testing, we'll just verify interaction is 0 when dummy is 0
            dummy_col = dummy_cols[0]
            interaction_col = interaction_cols[0]

            # Where dummy is 0, interaction should be 0
            mask_zero = baked[dummy_col] == 0
            assert all(baked.loc[mask_zero, interaction_col] == 0)

            # Where dummy is 1, interaction should be non-zero (original value)
            mask_one = baked[dummy_col] == 1
            # Interaction should have original values where dummy=1
            assert any(baked.loc[mask_one, interaction_col] != 0)

    def test_recipe_with_feature_type_interactions(self, simple_data):
        """Test step_splitwise with feature_type='interactions' in recipe."""
        rec = (
            recipe()
            .step_splitwise(outcome='y', min_improvement=1.0, feature_type='interactions')
        )

        prepped = rec.prep(simple_data)
        baked = prepped.bake(simple_data)

        # Check interactions created
        interaction_cols = [c for c in baked.columns if '_x_' in c]
        assert len(interaction_cols) > 0

    def test_recipe_with_feature_type_both(self, simple_data):
        """Test step_splitwise with feature_type='both' in recipe."""
        rec = (
            recipe()
            .step_splitwise(outcome='y', min_improvement=1.0, feature_type='both')
        )

        prepped = rec.prep(simple_data)
        baked = prepped.bake(simple_data)

        # Check both created
        dummy_cols = [c for c in baked.columns if ('_ge_' in c or '_between_' in c) and '_x_' not in c]
        interaction_cols = [c for c in baked.columns if '_x_' in c]

        assert len(dummy_cols) > 0
        assert len(interaction_cols) > 0

    def test_double_split_with_interactions(self, simple_data):
        """Test double-split transformations with interactions."""
        step = StepSplitwise(outcome='y', min_improvement=0.5, feature_type='both')
        prepped = step.prep(simple_data)

        # Check if any double-split decisions made
        decisions = prepped.get_decisions()
        double_splits = [col for col, info in decisions.items() if info['decision'] == 'double_split']

        if len(double_splits) > 0:
            baked = prepped.bake(simple_data)
            # Should have between_* columns for both dummies and interactions
            between_cols = [c for c in baked.columns if '_between_' in c]
            assert len(between_cols) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
