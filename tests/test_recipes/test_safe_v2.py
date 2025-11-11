"""
Tests for StepSafeV2 - SAFE with unfitted surrogate model
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from py_recipes import recipe
from py_recipes.steps.feature_extraction import StepSafeV2


@pytest.fixture
def regression_data():
    """Create synthetic regression data."""
    np.random.seed(42)
    n = 200

    data = pd.DataFrame({
        'x1': np.random.uniform(0, 10, n),
        'x2': np.random.uniform(-5, 5, n),
        'x3': np.random.uniform(0, 100, n),
        'cat1': np.random.choice(['A', 'B', 'C'], n),
        'cat2': np.random.choice(['Low', 'Medium', 'High'], n),
    })

    # Create outcome with nonlinear relationships
    data['y'] = (
        2 * data['x1'] +
        3 * np.sin(data['x2']) +
        0.1 * data['x3'] ** 0.5 +
        (data['cat1'] == 'A') * 5 +
        (data['cat2'] == 'High') * 3 +
        np.random.normal(0, 1, n)
    )

    return data


@pytest.fixture
def classification_data():
    """Create synthetic classification data."""
    np.random.seed(42)
    n = 200

    data = pd.DataFrame({
        'x1': np.random.uniform(0, 10, n),
        'x2': np.random.uniform(-5, 5, n),
        'cat1': np.random.choice(['A', 'B'], n),
    })

    # Create binary outcome
    prob = 1 / (1 + np.exp(-(data['x1'] - 5 + data['x2'])))
    data['y'] = (np.random.uniform(0, 1, n) < prob).astype(int)

    return data


class TestStepSafeV2Basic:
    """Test basic functionality."""

    def test_unfitted_model_acceptance(self, regression_data):
        """Test that unfitted model is accepted."""
        # Create UNFITTED model
        surrogate = GradientBoostingRegressor(n_estimators=10, random_state=42)

        # Should not raise error
        rec = recipe().step_safe_v2(
            surrogate_model=surrogate,
            outcome='y',
            penalty=10.0,
            max_thresholds=3
        )

        # Prep should fit the model
        prepped = rec.prep(regression_data)

        # Model should now be fitted (stored in step)
        step = prepped.prepared_steps[0]
        assert step._fitted_model is not None
        assert hasattr(step._fitted_model, 'n_features_in_')

    def test_fitted_model_warning(self, regression_data):
        """Test warning when passing fitted model."""
        # Fit model first
        X = regression_data.drop('y', axis=1)
        y = regression_data['y']

        # One-hot encode for fitting
        X_encoded = pd.get_dummies(X, drop_first=True)

        surrogate = GradientBoostingRegressor(n_estimators=10, random_state=42)
        surrogate.fit(X_encoded, y)

        # Should warn
        with pytest.warns(UserWarning, match="already fitted"):
            StepSafeV2(
                surrogate_model=surrogate,
                outcome='y',
                penalty=10.0
            )

    def test_max_thresholds_parameter(self, regression_data):
        """Test max_thresholds controls number of thresholds."""
        surrogate = GradientBoostingRegressor(n_estimators=10, random_state=42)

        # With max_thresholds=2
        rec = recipe().step_safe_v2(
            surrogate_model=surrogate,
            outcome='y',
            penalty=1.0,  # Low penalty = many changepoints
            max_thresholds=2
        )

        prepped = rec.prep(regression_data)
        step = prepped.prepared_steps[0]

        # Check that numeric variables have at most 2 thresholds
        for var in step._variables:
            if var['type'] == 'numeric':
                assert len(var.get('thresholds', [])) <= 2

    def test_feature_name_sanitization(self, regression_data):
        """Test that feature names are sanitized for LightGBM."""
        # Add columns with special characters
        data = regression_data.copy()
        data['x.1'] = data['x1']
        data['x-2'] = data['x2']
        data['x (3)'] = data['x3']

        surrogate = GradientBoostingRegressor(n_estimators=10, random_state=42)

        rec = recipe().step_safe_v2(
            surrogate_model=surrogate,
            outcome='y',
            penalty=10.0,
            max_thresholds=2,
            keep_original_cols=False  # Don't keep originals with special chars
        )

        prepped = rec.prep(data)
        baked = prepped.bake(data)

        # Check that TRANSFORMED column names don't have special characters
        # (outcome column 'y' is preserved as-is)
        for col in baked.columns:
            if col != 'y':
                # Should only have alphanumeric and underscore
                assert all(c.isalnum() or c == '_' for c in col), f"Column '{col}' has invalid characters"

    def test_importance_calculation_on_transformed(self, regression_data):
        """Test that importances are calculated on transformed features."""
        surrogate = GradientBoostingRegressor(n_estimators=10, random_state=42)

        rec = recipe().step_safe_v2(
            surrogate_model=surrogate,
            outcome='y',
            penalty=10.0,
            max_thresholds=3
        )

        prepped = rec.prep(regression_data)
        step = prepped.prepared_steps[0]

        # Check that importances were calculated
        assert len(step._feature_importances) > 0

        # Get importance DataFrame
        importances = step.get_feature_importances()

        assert not importances.empty
        assert 'feature' in importances.columns
        assert 'importance' in importances.columns

        # Raw importances (no longer normalized per variable group)
        # Check that importance values are valid
        assert (importances['importance'] >= 0).all(), "All importances should be non-negative"

        # Check that not all importances are the same (not uniform fallback)
        unique_values = importances['importance'].nunique()
        assert unique_values > 1, "Importances should vary (not all uniform)"


class TestStepSafeV2Transformations:
    """Test transformation logic."""

    def test_numeric_threshold_features(self, regression_data):
        """Test creation of binary threshold features."""
        surrogate = GradientBoostingRegressor(n_estimators=10, random_state=42)

        rec = recipe().step_safe_v2(
            surrogate_model=surrogate,
            outcome='y',
            penalty=10.0,
            max_thresholds=2,
            feature_type='numeric'
        )

        prepped = rec.prep(regression_data)
        baked = prepped.bake(regression_data)

        # Should have binary threshold features
        threshold_cols = [c for c in baked.columns if '_gt_' in c]
        assert len(threshold_cols) > 0

        # All should be binary (0 or 1)
        for col in threshold_cols:
            unique_vals = baked[col].unique()
            assert set(unique_vals).issubset({0, 1})

    def test_categorical_clustering(self, regression_data):
        """Test categorical variable clustering."""
        surrogate = GradientBoostingRegressor(n_estimators=10, random_state=42)

        rec = recipe().step_safe_v2(
            surrogate_model=surrogate,
            outcome='y',
            penalty=10.0,
            feature_type='categorical'
        )

        prepped = rec.prep(regression_data)
        baked = prepped.bake(regression_data)

        # Should have some categorical features transformed
        cat_cols = [c for c in baked.columns
                   if c not in ['y'] and 'cat' in c.lower()]
        assert len(cat_cols) > 0

    def test_feature_type_both(self, regression_data):
        """Test feature_type='both' includes numeric and categorical."""
        surrogate = GradientBoostingRegressor(n_estimators=10, random_state=42)

        rec = recipe().step_safe_v2(
            surrogate_model=surrogate,
            outcome='y',
            penalty=10.0,
            max_thresholds=2,
            feature_type='both'
        )

        prepped = rec.prep(regression_data)
        baked = prepped.bake(regression_data)

        # Should have both numeric thresholds and categorical features
        threshold_cols = [c for c in baked.columns if '_gt_' in c]
        cat_cols = [c for c in baked.columns
                   if c not in ['y'] and 'cat' in c.lower()]

        assert len(threshold_cols) > 0
        assert len(cat_cols) > 0

    def test_keep_original_cols_true(self, regression_data):
        """Test keep_original_cols=True preserves original features."""
        surrogate = GradientBoostingRegressor(n_estimators=10, random_state=42)

        rec = recipe().step_safe_v2(
            surrogate_model=surrogate,
            outcome='y',
            penalty=10.0,
            max_thresholds=2,
            keep_original_cols=True
        )

        prepped = rec.prep(regression_data)
        baked = prepped.bake(regression_data)

        # Original columns should still be present
        original_cols = ['x1', 'x2', 'x3', 'cat1', 'cat2']
        for col in original_cols:
            assert col in baked.columns

    def test_keep_original_cols_false(self, regression_data):
        """Test keep_original_cols=False removes original features."""
        surrogate = GradientBoostingRegressor(n_estimators=10, random_state=42)

        rec = recipe().step_safe_v2(
            surrogate_model=surrogate,
            outcome='y',
            penalty=10.0,
            max_thresholds=2,
            keep_original_cols=False
        )

        prepped = rec.prep(regression_data)
        baked = prepped.bake(regression_data)

        # Original numeric columns should NOT be present
        assert 'x1' not in baked.columns
        assert 'x2' not in baked.columns
        assert 'x3' not in baked.columns


class TestStepSafeV2FeatureSelection:
    """Test feature selection with top_n."""

    def test_top_n_selection(self, regression_data):
        """Test top_n selects most important features."""
        surrogate = GradientBoostingRegressor(n_estimators=10, random_state=42)

        rec = recipe().step_safe_v2(
            surrogate_model=surrogate,
            outcome='y',
            penalty=5.0,
            max_thresholds=3,
            top_n=5,
            keep_original_cols=False
        )

        prepped = rec.prep(regression_data)
        baked = prepped.bake(regression_data)

        # Should have exactly top_n + outcome columns
        # (may be fewer if not enough features created)
        feature_cols = [c for c in baked.columns if c != 'y']
        assert len(feature_cols) <= 5

    def test_top_n_selects_most_important(self, regression_data):
        """Test that top_n selects highest importance features."""
        surrogate = GradientBoostingRegressor(n_estimators=10, random_state=42)

        rec = recipe().step_safe_v2(
            surrogate_model=surrogate,
            outcome='y',
            penalty=5.0,
            max_thresholds=3,
            top_n=3
        )

        prepped = rec.prep(regression_data)
        step = prepped.prepared_steps[0]

        # Get importances
        importances = step.get_feature_importances()

        # Top 3 should match selected features
        top_features = importances.head(3)['feature'].tolist()

        # Selected features should be subset of top features
        assert len(step._selected_features) <= 3
        for feat in step._selected_features:
            assert feat in top_features


class TestStepSafeV2EdgeCases:
    """Test edge cases and error handling."""

    def test_missing_outcome_error(self, regression_data):
        """Test error when outcome not in data."""
        surrogate = GradientBoostingRegressor(n_estimators=10, random_state=42)

        rec = recipe().step_safe_v2(
            surrogate_model=surrogate,
            outcome='nonexistent',
            penalty=10.0
        )

        with pytest.raises(ValueError, match="not found"):
            rec.prep(regression_data)

    def test_invalid_penalty(self):
        """Test error with invalid penalty."""
        surrogate = GradientBoostingRegressor(n_estimators=10, random_state=42)

        with pytest.raises(ValueError, match="penalty must be > 0"):
            StepSafeV2(
                surrogate_model=surrogate,
                outcome='y',
                penalty=0.0
            )

    def test_invalid_max_thresholds(self):
        """Test error with invalid max_thresholds."""
        surrogate = GradientBoostingRegressor(n_estimators=10, random_state=42)

        with pytest.raises(ValueError, match="max_thresholds must be >= 1"):
            StepSafeV2(
                surrogate_model=surrogate,
                outcome='y',
                max_thresholds=0
            )

    def test_invalid_feature_type(self):
        """Test error with invalid feature_type."""
        surrogate = GradientBoostingRegressor(n_estimators=10, random_state=42)

        with pytest.raises(ValueError, match="feature_type must be"):
            StepSafeV2(
                surrogate_model=surrogate,
                outcome='y',
                feature_type='invalid'
            )

    def test_no_columns_after_filtering(self, regression_data):
        """Test handling when no columns to transform."""
        # Create data with only datetime and outcome
        data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'y': np.random.randn(100)
        })

        surrogate = GradientBoostingRegressor(n_estimators=10, random_state=42)

        rec = recipe().step_safe_v2(
            surrogate_model=surrogate,
            outcome='y',
            penalty=10.0
        )

        # Should warn but not error
        with pytest.warns(UserWarning, match="No columns to transform"):
            prepped = rec.prep(data)

        # Should return data unchanged (except outcome)
        baked = prepped.bake(data)
        assert 'y' in baked.columns


class TestStepSafeV2Classification:
    """Test with classification tasks."""

    def test_classification_task(self, classification_data):
        """Test SAFE v2 with classification outcome."""
        from sklearn.ensemble import GradientBoostingClassifier

        surrogate = GradientBoostingClassifier(n_estimators=10, random_state=42)

        rec = recipe().step_safe_v2(
            surrogate_model=surrogate,
            outcome='y',
            penalty=10.0,
            max_thresholds=3
        )

        prepped = rec.prep(classification_data)
        baked = prepped.bake(classification_data)

        # Should have transformed features
        assert baked.shape[1] > 1  # More than just outcome


class TestStepSafeV2DifferentSurrogates:
    """Test with different surrogate model types."""

    def test_random_forest_surrogate(self, regression_data):
        """Test with RandomForest surrogate."""
        surrogate = RandomForestRegressor(n_estimators=10, random_state=42)

        rec = recipe().step_safe_v2(
            surrogate_model=surrogate,
            outcome='y',
            penalty=10.0,
            max_thresholds=3
        )

        prepped = rec.prep(regression_data)
        baked = prepped.bake(regression_data)

        assert baked.shape[0] == regression_data.shape[0]
        assert 'y' in baked.columns

    def test_linear_regression_surrogate(self, regression_data):
        """Test with LinearRegression surrogate."""
        surrogate = LinearRegression()

        rec = recipe().step_safe_v2(
            surrogate_model=surrogate,
            outcome='y',
            penalty=10.0,
            max_thresholds=3
        )

        prepped = rec.prep(regression_data)
        baked = prepped.bake(regression_data)

        assert baked.shape[0] == regression_data.shape[0]
        assert 'y' in baked.columns


class TestStepSafeV2ColumnSelection:
    """Test column selection parameter."""

    def test_columns_parameter_list(self, regression_data):
        """Test columns parameter with list."""
        surrogate = GradientBoostingRegressor(n_estimators=10, random_state=42)

        rec = recipe().step_safe_v2(
            surrogate_model=surrogate,
            outcome='y',
            penalty=10.0,
            max_thresholds=2,
            columns=['x1', 'x2']  # Only transform x1 and x2
        )

        prepped = rec.prep(regression_data)
        step = prepped.prepared_steps[0]

        # Should only have x1 and x2 in original columns
        assert set(step._original_columns) == {'x1', 'x2'}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
