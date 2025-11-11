"""
Tests for step_eix() - EIX (Explain Interactions in XGBoost) recipe step.
"""

import pytest
import pandas as pd
import numpy as np
from py_recipes import recipe
from py_recipes.steps.interaction_detection import StepEIX
from py_workflows import workflow
from py_parsnip import linear_reg

# Import tree models
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# ================================================================================
# Fixtures
# ================================================================================

@pytest.fixture
def regression_data():
    """Simple regression dataset for testing."""
    np.random.seed(42)
    n = 300

    x1 = np.random.uniform(0, 10, n)
    x2 = np.random.uniform(-5, 5, n)
    x3 = np.random.randn(n)

    # Create interactions
    # y depends on x1, x2, and interaction x1*x2
    y = 2 * x1 + 0.5 * x2 + 0.3 * x1 * x2 + 0.1 * x3 + np.random.randn(n) * 0.5

    return pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'y': y
    })


@pytest.fixture
def fitted_xgb_model(regression_data):
    """Pre-fitted XGBoost model for testing."""
    if not HAS_XGB:
        pytest.skip("XGBoost not installed")

    data = regression_data
    X = data.drop('y', axis=1)
    y = data['y']

    model = XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X, y)

    return model


@pytest.fixture
def fitted_lgb_model(regression_data):
    """Pre-fitted LightGBM model for testing."""
    if not HAS_LGB:
        pytest.skip("LightGBM not installed")

    data = regression_data
    X = data.drop('y', axis=1)
    y = data['y']

    model = LGBMRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    model.fit(X, y)

    return model


# ================================================================================
# Basic Tests
# ================================================================================

class TestStepEIXBasics:
    """Test basic StepEIX functionality."""

    def test_step_creation_default_params(self, fitted_xgb_model):
        """Test step creation with default parameters."""
        step = StepEIX(
            tree_model=fitted_xgb_model,
            outcome='y'
        )

        assert step.tree_model is fitted_xgb_model
        assert step.outcome == 'y'
        assert step.option == 'both'
        assert step.top_n is None
        assert step.min_gain == 0.0
        assert step.create_interactions is True
        assert step.keep_original_cols is False
        assert step.skip is False

    def test_step_creation_custom_params(self, fitted_xgb_model):
        """Test step creation with custom parameters."""
        step = StepEIX(
            tree_model=fitted_xgb_model,
            outcome='target',
            option='interactions',
            top_n=5,
            min_gain=0.1,
            create_interactions=False,
            keep_original_cols=True
        )

        assert step.outcome == 'target'
        assert step.option == 'interactions'
        assert step.top_n == 5
        assert step.min_gain == 0.1
        assert step.create_interactions is False
        assert step.keep_original_cols is True

    def test_missing_tree_model_error(self):
        """Test error when tree_model is None."""
        with pytest.raises(ValueError, match="tree_model is required"):
            StepEIX(tree_model=None, outcome='y')

    def test_unfitted_model_error(self):
        """Test error when tree model is not fitted."""
        if not HAS_XGB:
            pytest.skip("XGBoost not installed")

        unfitted_model = XGBRegressor()

        with pytest.raises(ValueError, match="tree_model must be fitted"):
            StepEIX(tree_model=unfitted_model, outcome='y')

    def test_missing_outcome_error(self, fitted_xgb_model):
        """Test error when outcome is None."""
        with pytest.raises(ValueError, match="outcome is required"):
            StepEIX(tree_model=fitted_xgb_model, outcome='')

    def test_invalid_option_error(self, fitted_xgb_model):
        """Test error with invalid option parameter."""
        with pytest.raises(ValueError, match="option must be"):
            StepEIX(tree_model=fitted_xgb_model, outcome='y', option='invalid')

    def test_invalid_top_n_error(self, fitted_xgb_model):
        """Test error with negative top_n."""
        with pytest.raises(ValueError, match="top_n must be positive"):
            StepEIX(tree_model=fitted_xgb_model, outcome='y', top_n=-1)

    def test_invalid_min_gain_error(self, fitted_xgb_model):
        """Test error with negative min_gain."""
        with pytest.raises(ValueError, match="min_gain must be non-negative"):
            StepEIX(tree_model=fitted_xgb_model, outcome='y', min_gain=-0.5)


# ================================================================================
# Prep Tests
# ================================================================================

class TestStepEIXPrep:
    """Test StepEIX.prep() functionality."""

    def test_prep_basic(self, regression_data, fitted_xgb_model):
        """Test basic prep functionality."""
        step = StepEIX(tree_model=fitted_xgb_model, outcome='y')
        prepped = step.prep(regression_data)

        assert prepped._is_prepped is True
        assert len(prepped._selected_features) > 0
        assert prepped._importance_table is not None

    def test_prep_missing_outcome_error(self, regression_data, fitted_xgb_model):
        """Test error when outcome column not in data."""
        step = StepEIX(tree_model=fitted_xgb_model, outcome='missing_col')

        with pytest.raises(ValueError, match="Outcome column 'missing_col' not found"):
            step.prep(regression_data)

    def test_prep_creates_importance_table(self, regression_data, fitted_xgb_model):
        """Test that prep creates importance table."""
        step = StepEIX(tree_model=fitted_xgb_model, outcome='y', option='both')
        prepped = step.prep(regression_data)

        importance = prepped.get_importance()

        assert isinstance(importance, pd.DataFrame)
        assert 'sumGain' in importance.columns
        assert 'frequency' in importance.columns
        assert 'type' in importance.columns
        assert len(importance) > 0

    def test_prep_option_variables(self, regression_data, fitted_xgb_model):
        """Test prep with option='variables'."""
        step = StepEIX(
            tree_model=fitted_xgb_model,
            outcome='y',
            option='variables'
        )
        prepped = step.prep(regression_data)

        importance = prepped.get_importance()
        assert all(importance['type'] == 'variable')

    def test_prep_option_interactions(self, regression_data, fitted_xgb_model):
        """Test prep with option='interactions'."""
        step = StepEIX(
            tree_model=fitted_xgb_model,
            outcome='y',
            option='interactions'
        )
        prepped = step.prep(regression_data)

        importance = prepped.get_importance()
        assert all(importance['type'] == 'interaction')

    def test_prep_option_both(self, regression_data, fitted_xgb_model):
        """Test prep with option='both'."""
        step = StepEIX(
            tree_model=fitted_xgb_model,
            outcome='y',
            option='both'
        )
        prepped = step.prep(regression_data)

        importance = prepped.get_importance()
        types = importance['type'].unique()
        # Should have at least one type (may not have both if tree is simple)
        assert len(types) > 0


# ================================================================================
# Bake Tests
# ================================================================================

class TestStepEIXBake:
    """Test StepEIX.bake() functionality."""

    def test_bake_without_prep_returns_original(self, regression_data, fitted_xgb_model):
        """Test that bake without prep returns original data."""
        step = StepEIX(tree_model=fitted_xgb_model, outcome='y')
        result = step.bake(regression_data)

        pd.testing.assert_frame_equal(result, regression_data)

    def test_bake_creates_features(self, regression_data, fitted_xgb_model):
        """Test that bake creates transformed features."""
        step = StepEIX(
            tree_model=fitted_xgb_model,
            outcome='y',
            option='both',
            top_n=10
        )
        prepped = step.prep(regression_data, training=True)
        transformed = prepped.bake(regression_data)

        # Should have some features (at least outcome)
        assert len(transformed.columns) > 0

        # Outcome should be preserved
        assert 'y' in transformed.columns

    def test_bake_creates_interaction_features(self, regression_data, fitted_xgb_model):
        """Test that bake creates interaction features (parent × child)."""
        step = StepEIX(
            tree_model=fitted_xgb_model,
            outcome='y',
            option='interactions',
            create_interactions=True,
            top_n=5
        )
        prepped = step.prep(regression_data, training=True)
        transformed = prepped.bake(regression_data)

        # Check if any interaction columns exist (format: parent_x_child)
        interaction_cols = [col for col in transformed.columns if '_x_' in col]

        # Should have some interactions if interactions were found
        interactions = prepped.get_interactions()
        if len(interactions) > 0:
            assert len(interaction_cols) > 0

    def test_bake_no_create_interactions(self, regression_data, fitted_xgb_model):
        """Test bake with create_interactions=False."""
        step = StepEIX(
            tree_model=fitted_xgb_model,
            outcome='y',
            option='interactions',
            create_interactions=False,
            top_n=5
        )
        prepped = step.prep(regression_data, training=True)
        transformed = prepped.bake(regression_data)

        # Should not have interaction columns
        interaction_cols = [col for col in transformed.columns if '_x_' in col]
        assert len(interaction_cols) == 0

    def test_bake_keep_original_cols(self, regression_data, fitted_xgb_model):
        """Test bake with keep_original_cols=True."""
        step = StepEIX(
            tree_model=fitted_xgb_model,
            outcome='y',
            option='variables',
            top_n=2,  # Select only 2 features
            keep_original_cols=True
        )
        prepped = step.prep(regression_data, training=True)
        transformed = prepped.bake(regression_data)

        # Should keep all original columns
        for col in ['x1', 'x2', 'x3']:
            assert col in transformed.columns

    def test_bake_top_n_selection(self, regression_data, fitted_xgb_model):
        """Test bake with top_n parameter."""
        step = StepEIX(
            tree_model=fitted_xgb_model,
            outcome='y',
            option='variables',
            top_n=2,
            create_interactions=False,
            keep_original_cols=False
        )
        prepped = step.prep(regression_data, training=True)
        transformed = prepped.bake(regression_data)

        # Should have at most top_n + outcome columns
        # (may be less if some features not in data)
        assert len(transformed.columns) <= 3  # 2 features + outcome


# ================================================================================
# Recipe Integration Tests
# ================================================================================

class TestStepEIXRecipeIntegration:
    """Test StepEIX integration with Recipe."""

    def test_recipe_with_step_eix(self, regression_data, fitted_xgb_model):
        """Test creating recipe with step_eix()."""
        rec = recipe().step_eix(
            tree_model=fitted_xgb_model,
            outcome='y',
            option='interactions',
            top_n=5
        )

        assert len(rec.steps) == 1
        assert isinstance(rec.steps[0], StepEIX)

    def test_recipe_prep_and_bake(self, regression_data, fitted_xgb_model):
        """Test recipe prep and bake with step_eix()."""
        rec = recipe().step_eix(
            tree_model=fitted_xgb_model,
            outcome='y',
            option='both',
            top_n=10
        )

        prepped = rec.prep(regression_data)
        transformed = prepped.bake(regression_data)

        assert isinstance(transformed, pd.DataFrame)
        assert 'y' in transformed.columns

    def test_recipe_multiple_steps(self, regression_data, fitted_xgb_model):
        """Test recipe with step_eix() and other steps."""
        from py_recipes.selectors import all_numeric_predictors

        rec = (
            recipe()
            .step_eix(
                tree_model=fitted_xgb_model,
                outcome='y',
                option='interactions',
                top_n=10
            )
            .step_normalize(all_numeric_predictors())
        )

        prepped = rec.prep(regression_data)
        transformed = prepped.bake(regression_data)

        assert isinstance(transformed, pd.DataFrame)


# ================================================================================
# LightGBM Tests
# ================================================================================

class TestStepEIXLightGBM:
    """Test StepEIX with LightGBM models."""

    def test_prep_with_lightgbm(self, regression_data, fitted_lgb_model):
        """Test prep with LightGBM model."""
        step = StepEIX(
            tree_model=fitted_lgb_model,
            outcome='y',
            option='both'
        )
        prepped = step.prep(regression_data)

        assert prepped._is_prepped is True
        assert len(prepped._selected_features) > 0

    def test_bake_with_lightgbm(self, regression_data, fitted_lgb_model):
        """Test bake with LightGBM model."""
        step = StepEIX(
            tree_model=fitted_lgb_model,
            outcome='y',
            option='interactions',
            top_n=5
        )
        prepped = step.prep(regression_data)
        transformed = prepped.bake(regression_data)

        assert 'y' in transformed.columns


# ================================================================================
# Edge Cases Tests
# ================================================================================

class TestStepEIXEdgeCases:
    """Test edge cases for StepEIX."""

    def test_skip_parameter(self, regression_data, fitted_xgb_model):
        """Test step with skip=True."""
        step = StepEIX(
            tree_model=fitted_xgb_model,
            outcome='y',
            skip=True
        )
        prepped = step.prep(regression_data)
        transformed = prepped.bake(regression_data)

        # Should return original data when skip=True
        pd.testing.assert_frame_equal(transformed, regression_data)

    def test_min_gain_filter(self, regression_data, fitted_xgb_model):
        """Test with high min_gain threshold."""
        step = StepEIX(
            tree_model=fitted_xgb_model,
            outcome='y',
            option='both',
            min_gain=1000.0  # Very high threshold
        )
        prepped = step.prep(regression_data)
        transformed = prepped.bake(regression_data)

        # Should have at least outcome column
        assert 'y' in transformed.columns

    def test_top_n_larger_than_available(self, regression_data, fitted_xgb_model):
        """Test with top_n larger than available features."""
        step = StepEIX(
            tree_model=fitted_xgb_model,
            outcome='y',
            option='variables',
            top_n=1000  # More than available
        )
        prepped = step.prep(regression_data)
        transformed = prepped.bake(regression_data)

        # Should work fine, just select all available
        assert 'y' in transformed.columns


# ================================================================================
# Inspection Methods Tests
# ================================================================================

class TestStepEIXInspection:
    """Test StepEIX inspection methods."""

    def test_get_importance(self, regression_data, fitted_xgb_model):
        """Test get_importance() method."""
        step = StepEIX(
            tree_model=fitted_xgb_model,
            outcome='y',
            option='both'
        )
        prepped = step.prep(regression_data)

        importance = prepped.get_importance()

        assert isinstance(importance, pd.DataFrame)
        assert 'sumGain' in importance.columns
        assert 'frequency' in importance.columns
        assert 'meanGain' in importance.columns
        assert len(importance) > 0

    def test_get_importance_before_prep_error(self, fitted_xgb_model):
        """Test get_importance() before prep raises error."""
        step = StepEIX(tree_model=fitted_xgb_model, outcome='y')

        with pytest.raises(RuntimeError, match="Must call prep"):
            step.get_importance()

    def test_get_interactions(self, regression_data, fitted_xgb_model):
        """Test get_interactions() method."""
        step = StepEIX(
            tree_model=fitted_xgb_model,
            outcome='y',
            option='interactions',
            create_interactions=True,
            top_n=5
        )
        prepped = step.prep(regression_data)

        interactions = prepped.get_interactions()

        assert isinstance(interactions, list)
        # May or may not have interactions depending on tree structure
        if len(interactions) > 0:
            assert 'parent' in interactions[0]
            assert 'child' in interactions[0]
            assert 'name' in interactions[0]
            assert 'sumGain' in interactions[0]

    def test_get_interactions_before_prep_error(self, fitted_xgb_model):
        """Test get_interactions() before prep raises error."""
        step = StepEIX(tree_model=fitted_xgb_model, outcome='y')

        with pytest.raises(RuntimeError, match="Must call prep"):
            step.get_interactions()


# ================================================================================
# Workflow Integration Tests
# ================================================================================

class TestStepEIXWorkflowIntegration:
    """Test StepEIX integration with workflows."""

    def test_workflow_with_step_eix(self, regression_data, fitted_xgb_model):
        """Test workflow with step_eix()."""
        rec = recipe().step_eix(
            tree_model=fitted_xgb_model,
            outcome='y',
            option='both',
            top_n=10
        )

        wf = workflow().add_recipe(rec).add_model(linear_reg())

        # Split data
        train = regression_data.iloc[:200]
        test = regression_data.iloc[200:]

        # Fit workflow
        fit = wf.fit(train)
        predictions = fit.predict(test)

        assert '.pred' in predictions.columns
        assert len(predictions) == len(test)

    def test_workflow_predictions(self, regression_data, fitted_xgb_model):
        """Test that workflow predictions work with EIX features."""
        rec = recipe().step_eix(
            tree_model=fitted_xgb_model,
            outcome='y',
            option='both',  # Use both to ensure we have features
            create_interactions=True,
            top_n=15  # Higher top_n to ensure we get features
        )

        wf = workflow().add_recipe(rec).add_model(linear_reg())

        train = regression_data.iloc[:200]
        test = regression_data.iloc[200:]

        fit = wf.fit(train)
        predictions = fit.predict(test)

        # Predictions should be numeric
        assert predictions['.pred'].dtype in [np.float64, np.float32]
        assert not predictions['.pred'].isna().any()
