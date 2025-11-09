"""
Tests for step_safe() - Surrogate Assisted Feature Extraction.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from py_recipes import recipe
from py_recipes.steps.feature_extraction import StepSafe


@pytest.fixture
def simple_regression_data():
    """
    Simple regression dataset with numeric predictors.
    """
    np.random.seed(42)
    n = 200

    x1 = np.random.uniform(0, 10, n)
    x2 = np.random.uniform(-5, 5, n)
    x3 = np.random.randn(n)

    # Non-linear relationship that surrogate can capture
    y = (
        2 * x1 +
        5 * np.where(x2 > 0, 1, 0) +  # Step function
        0.5 * x3**2 +  # Quadratic
        np.random.randn(n) * 0.5
    )

    return pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'y': y
    })


@pytest.fixture
def fitted_surrogate(simple_regression_data):
    """
    Pre-fitted surrogate model for testing.
    """
    data = simple_regression_data
    X = data.drop('y', axis=1)
    y = data['y']

    surrogate = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=3,
        random_state=42
    )
    surrogate.fit(X, y)

    return surrogate


@pytest.fixture
def mixed_data():
    """
    Dataset with both numeric and categorical predictors.
    """
    np.random.seed(123)
    n = 300

    x1 = np.random.uniform(0, 10, n)
    x2 = np.random.choice(['A', 'B', 'C'], n)
    x3 = np.random.randn(n)

    y = (
        0.5 * x1 +
        np.where(x2 == 'A', 5, np.where(x2 == 'B', 2, 0)) +
        2 * x3 +
        np.random.randn(n)
    )

    return pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'target': y
    })


@pytest.fixture
def fitted_surrogate_mixed(mixed_data):
    """
    Surrogate fitted on mixed data (with categoricals one-hot encoded).
    """
    data = mixed_data
    X = pd.get_dummies(data.drop('target', axis=1), drop_first=True)
    y = data['target']

    surrogate = RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        random_state=42
    )
    surrogate.fit(X, y)

    return surrogate


class TestStepSafeBasics:
    """Test basic functionality of step_safe."""

    def test_step_creation(self, fitted_surrogate):
        """Test step can be created with default parameters."""
        step = StepSafe(
            surrogate_model=fitted_surrogate,
            outcome='y'
        )
        assert step.outcome == 'y'
        assert step.penalty == 3.0
        assert step.pelt_model == 'l2'
        assert step.no_changepoint_strategy == 'median'
        assert step.keep_original_cols is False
        assert step.top_n is None

    def test_step_creation_with_params(self, fitted_surrogate):
        """Test step creation with custom parameters."""
        step = StepSafe(
            surrogate_model=fitted_surrogate,
            outcome='target',
            penalty=5.0,
            pelt_model='l1',
            no_changepoint_strategy='drop',
            keep_original_cols=True,
            top_n=10,
            grid_resolution=500
        )
        assert step.penalty == 5.0
        assert step.pelt_model == 'l1'
        assert step.no_changepoint_strategy == 'drop'
        assert step.keep_original_cols is True
        assert step.top_n == 10
        assert step.grid_resolution == 500

    def test_invalid_penalty(self, fitted_surrogate):
        """Test validation of penalty parameter."""
        with pytest.raises(ValueError, match="penalty must be > 0"):
            StepSafe(
                surrogate_model=fitted_surrogate,
                outcome='y',
                penalty=0
            )

        with pytest.raises(ValueError, match="penalty must be > 0"):
            StepSafe(
                surrogate_model=fitted_surrogate,
                outcome='y',
                penalty=-1.0
            )

    def test_invalid_pelt_model(self, fitted_surrogate):
        """Test validation of pelt_model parameter."""
        with pytest.raises(ValueError, match="pelt_model must be"):
            StepSafe(
                surrogate_model=fitted_surrogate,
                outcome='y',
                pelt_model='invalid'
            )

    def test_invalid_changepoint_strategy(self, fitted_surrogate):
        """Test validation of no_changepoint_strategy parameter."""
        with pytest.raises(ValueError, match="no_changepoint_strategy must be"):
            StepSafe(
                surrogate_model=fitted_surrogate,
                outcome='y',
                no_changepoint_strategy='invalid'
            )

    def test_invalid_grid_resolution(self, fitted_surrogate):
        """Test validation of grid_resolution parameter."""
        with pytest.raises(ValueError, match="grid_resolution must be >= 100"):
            StepSafe(
                surrogate_model=fitted_surrogate,
                outcome='y',
                grid_resolution=50
            )

    def test_invalid_top_n(self, fitted_surrogate):
        """Test validation of top_n parameter."""
        with pytest.raises(ValueError, match="top_n must be >= 1"):
            StepSafe(
                surrogate_model=fitted_surrogate,
                outcome='y',
                top_n=0
            )

    def test_unfitted_surrogate_raises_error(self):
        """Test that unfitted surrogate model raises error."""
        unfitted_model = GradientBoostingRegressor()

        with pytest.raises(ValueError, match="surrogate_model must be pre-fitted"):
            StepSafe(
                surrogate_model=unfitted_model,
                outcome='y'
            )


class TestStepSafePrep:
    """Test prep() functionality."""

    def test_prep_basic(self, simple_regression_data, fitted_surrogate):
        """Test basic prep functionality."""
        step = StepSafe(
            surrogate_model=fitted_surrogate,
            outcome='y',
            penalty=3.0
        )

        prepped = step.prep(simple_regression_data, training=True)

        assert prepped._is_prepared
        assert len(prepped._variables) > 0
        assert len(prepped._original_columns) == 3  # x1, x2, x3

    def test_prep_missing_outcome(self, simple_regression_data, fitted_surrogate):
        """Test prep with missing outcome raises error."""
        step = StepSafe(
            surrogate_model=fitted_surrogate,
            outcome='missing_outcome'
        )

        with pytest.raises(ValueError, match="Outcome 'missing_outcome' not found"):
            step.prep(simple_regression_data, training=True)

    def test_prep_creates_transformations(self, simple_regression_data, fitted_surrogate):
        """Test that prep creates transformation metadata."""
        step = StepSafe(
            surrogate_model=fitted_surrogate,
            outcome='y',
            penalty=3.0
        )

        prepped = step.prep(simple_regression_data, training=True)

        # Should have transformations for each variable
        transformations = prepped.get_transformations()
        assert len(transformations) == 3  # x1, x2, x3
        assert 'x1' in transformations
        assert 'x2' in transformations
        assert 'x3' in transformations

    def test_prep_computes_feature_importances(self, simple_regression_data, fitted_surrogate):
        """Test that prep computes feature importances."""
        step = StepSafe(
            surrogate_model=fitted_surrogate,
            outcome='y',
            penalty=3.0
        )

        prepped = step.prep(simple_regression_data, training=True)

        importances = prepped.get_feature_importances()
        assert isinstance(importances, pd.DataFrame)
        assert 'feature' in importances.columns
        assert 'importance' in importances.columns
        assert len(importances) > 0


class TestStepSafeBake:
    """Test bake() functionality."""

    def test_bake_basic(self, simple_regression_data, fitted_surrogate):
        """Test basic bake functionality."""
        step = StepSafe(
            surrogate_model=fitted_surrogate,
            outcome='y',
            penalty=3.0
        )

        prepped = step.prep(simple_regression_data, training=True)
        transformed = prepped.bake(simple_regression_data)

        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(simple_regression_data)

    def test_bake_without_prep_returns_original(self, simple_regression_data, fitted_surrogate):
        """Test that bake without prep returns copy of data."""
        step = StepSafe(
            surrogate_model=fitted_surrogate,
            outcome='y',
            skip=False
        )

        # Bake without prep
        result = step.bake(simple_regression_data)
        assert isinstance(result, pd.DataFrame)

    def test_bake_creates_safe_features(self, simple_regression_data, fitted_surrogate):
        """Test that bake creates SAFE transformed features."""
        step = StepSafe(
            surrogate_model=fitted_surrogate,
            outcome='y',
            penalty=3.0,
            keep_original_cols=False
        )

        prepped = step.prep(simple_regression_data, training=True)
        transformed = prepped.bake(simple_regression_data)

        # Should have transformed features (one-hot encoded intervals)
        assert len(transformed.columns) > 0

        # Columns should follow SAFE naming convention (excluding outcome)
        # Numeric: "varname_threshold1_to_threshold2"
        # Categorical: "varname_level1_level2"
        safe_features = [col for col in transformed.columns if col != 'y']
        assert len(safe_features) > 0

        for col in safe_features:
            # Should have '_to_' for numeric intervals or multiple '_' for any SAFE feature
            assert '_to_' in col or col.count('_') >= 2

    def test_bake_keep_original_cols(self, simple_regression_data, fitted_surrogate):
        """Test that keep_original_cols works."""
        step = StepSafe(
            surrogate_model=fitted_surrogate,
            outcome='y',
            penalty=3.0,
            keep_original_cols=True
        )

        prepped = step.prep(simple_regression_data, training=True)
        transformed = prepped.bake(simple_regression_data)

        # Should have both original and transformed columns
        assert 'x1' in transformed.columns
        assert 'x2' in transformed.columns
        assert 'x3' in transformed.columns

        # And should also have SAFE features (with '_to_' for intervals)
        safe_features = [col for col in transformed.columns if '_to_' in col or (col not in ['x1', 'x2', 'x3', 'y'])]
        assert len(safe_features) > 0

    def test_bake_top_n_selection(self, simple_regression_data, fitted_surrogate):
        """Test that top_n feature selection works."""
        step = StepSafe(
            surrogate_model=fitted_surrogate,
            outcome='y',
            penalty=2.0,  # Lower penalty = more features
            top_n=3,
            keep_original_cols=False
        )

        prepped = step.prep(simple_regression_data, training=True)
        transformed = prepped.bake(simple_regression_data)

        # Should have at most top_n features (plus outcome column)
        safe_features = [col for col in transformed.columns if col != 'y']
        assert len(safe_features) <= 3


class TestStepSafeRecipeIntegration:
    """Test integration with recipe."""

    def test_recipe_with_step_safe(self, simple_regression_data, fitted_surrogate):
        """Test step_safe works in recipe."""
        rec = recipe().step_safe(
            surrogate_model=fitted_surrogate,
            outcome='y',
            penalty=3.0
        )

        assert len(rec.steps) == 1
        assert isinstance(rec.steps[0], StepSafe)

    def test_recipe_prep_and_bake(self, simple_regression_data, fitted_surrogate):
        """Test recipe prep and bake with step_safe."""
        rec = recipe().step_safe(
            surrogate_model=fitted_surrogate,
            outcome='y',
            penalty=3.0
        )

        prepped = rec.prep(simple_regression_data)
        transformed = prepped.bake(simple_regression_data)

        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(simple_regression_data)

    def test_recipe_with_multiple_steps(self, simple_regression_data, fitted_surrogate):
        """Test step_safe with other recipe steps."""
        from py_recipes.selectors import all_numeric_predictors

        rec = (
            recipe()
            .step_safe(
                surrogate_model=fitted_surrogate,
                outcome='y',
                penalty=3.0,
                keep_original_cols=True
            )
            .step_normalize(all_numeric_predictors())
        )

        prepped = rec.prep(simple_regression_data)
        transformed = prepped.bake(simple_regression_data)

        assert isinstance(transformed, pd.DataFrame)
        # Should have both SAFE features and normalized originals


class TestStepSafeCategorical:
    """Test SAFE with categorical variables."""

    def test_categorical_transformation(self, mixed_data, fitted_surrogate_mixed):
        """Test SAFE handles categorical variables."""
        step = StepSafe(
            surrogate_model=fitted_surrogate_mixed,
            outcome='target',
            penalty=3.0
        )

        prepped = step.prep(mixed_data, training=True)
        transformed = prepped.bake(mixed_data)

        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(mixed_data)

        # Check transformations include categorical variable
        transformations = prepped.get_transformations()
        assert 'x2' in transformations  # x2 is categorical
        assert transformations['x2']['type'] == 'categorical'


class TestStepSafeEdgeCases:
    """Test edge cases."""

    def test_skip_parameter(self, simple_regression_data, fitted_surrogate):
        """Test skip parameter works."""
        step = StepSafe(
            surrogate_model=fitted_surrogate,
            outcome='y',
            skip=True
        )

        prepped = step.prep(simple_regression_data, training=True)
        transformed = prepped.bake(simple_regression_data)

        # Should return copy of original data
        assert len(transformed) == len(simple_regression_data)

    def test_different_penalties(self, simple_regression_data, fitted_surrogate):
        """Test different penalty values affect number of changepoints."""
        # Low penalty = more changepoints = more features
        step_low = StepSafe(
            surrogate_model=fitted_surrogate,
            outcome='y',
            penalty=1.0
        )
        prepped_low = step_low.prep(simple_regression_data, training=True)
        transformed_low = prepped_low.bake(simple_regression_data)

        # High penalty = fewer changepoints = fewer features
        step_high = StepSafe(
            surrogate_model=fitted_surrogate,
            outcome='y',
            penalty=10.0
        )
        prepped_high = step_high.prep(simple_regression_data, training=True)
        transformed_high = prepped_high.bake(simple_regression_data)

        # Low penalty should produce more or equal features
        assert len(transformed_low.columns) >= len(transformed_high.columns)

    def test_no_changepoint_drop_strategy(self, simple_regression_data, fitted_surrogate):
        """Test no_changepoint_strategy='drop' works."""
        step = StepSafe(
            surrogate_model=fitted_surrogate,
            outcome='y',
            penalty=100.0,  # Very high penalty = likely no changepoints
            no_changepoint_strategy='drop'
        )

        prepped = step.prep(simple_regression_data, training=True)
        transformed = prepped.bake(simple_regression_data)

        # Should still return DataFrame (possibly with fewer columns)
        assert isinstance(transformed, pd.DataFrame)


class TestStepSafeFeatureImportances:
    """Test feature importance extraction."""

    def test_get_feature_importances(self, simple_regression_data, fitted_surrogate):
        """Test get_feature_importances() method."""
        step = StepSafe(
            surrogate_model=fitted_surrogate,
            outcome='y',
            penalty=3.0
        )

        prepped = step.prep(simple_regression_data, training=True)
        importances = prepped.get_feature_importances()

        assert isinstance(importances, pd.DataFrame)
        assert 'feature' in importances.columns
        assert 'importance' in importances.columns

        # Importances should be sorted (highest first)
        assert importances['importance'].is_monotonic_decreasing

    def test_get_feature_importances_before_prep_raises(self, simple_regression_data, fitted_surrogate):
        """Test get_feature_importances before prep raises error."""
        step = StepSafe(
            surrogate_model=fitted_surrogate,
            outcome='y'
        )

        with pytest.raises(ValueError, match="Step must be prepared"):
            step.get_feature_importances()

    def test_get_transformations_before_prep_raises(self, simple_regression_data, fitted_surrogate):
        """Test get_transformations before prep raises error."""
        step = StepSafe(
            surrogate_model=fitted_surrogate,
            outcome='y'
        )

        with pytest.raises(ValueError, match="Step must be prepared"):
            step.get_transformations()


class TestStepSafeWorkflowIntegration:
    """Test integration with workflows."""

    def test_workflow_with_step_safe(self, simple_regression_data, fitted_surrogate):
        """Test step_safe works in workflow."""
        from py_workflows import workflow
        from py_parsnip import linear_reg

        rec = recipe().step_safe(
            surrogate_model=fitted_surrogate,
            outcome='y',
            penalty=3.0,
            keep_original_cols=False
        )

        wf = workflow().add_recipe(rec).add_model(linear_reg())

        # Fit workflow
        train = simple_regression_data.iloc[:150]
        test = simple_regression_data.iloc[150:]

        fit = wf.fit(train)
        fit = fit.evaluate(test)

        # Should have outputs with standard columns
        outputs, _, stats = fit.extract_outputs()
        assert len(outputs) > 0
        assert 'fitted' in outputs.columns
        assert 'actuals' in outputs.columns

    def test_workflow_predict_with_safe(self, simple_regression_data, fitted_surrogate):
        """Test predictions work with SAFE features."""
        from py_workflows import workflow
        from py_parsnip import linear_reg

        rec = recipe().step_safe(
            surrogate_model=fitted_surrogate,
            outcome='y',
            penalty=3.0
        )

        wf = workflow().add_recipe(rec).add_model(linear_reg())

        # Fit and predict
        train = simple_regression_data.iloc[:150]
        test = simple_regression_data.iloc[150:]

        fit = wf.fit(train)
        predictions = fit.predict(test)

        assert isinstance(predictions, pd.DataFrame)
        assert '.pred' in predictions.columns
        assert len(predictions) == len(test)
