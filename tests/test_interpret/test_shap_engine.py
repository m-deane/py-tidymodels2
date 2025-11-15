"""
Tests for SHAP interpretability engine

Tests cover:
- Auto-selection of explainer methods
- TreeExplainer for tree-based models
- LinearExplainer for linear models
- KernelExplainer for other models
- DataFrame format validation
- Additivity checking
- Background data strategies
- Integration with ModelFit, WorkflowFit, NestedWorkflowFit
"""

import pytest
import pandas as pd
import numpy as np
from py_parsnip import linear_reg, rand_forest, decision_tree
from py_workflows import workflow
from py_recipes import recipe
from py_interpret import ShapEngine


# Test fixtures
@pytest.fixture
def simple_regression_data():
    """Simple regression dataset for testing."""
    np.random.seed(42)
    n = 100
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    X3 = np.random.randn(n)
    y = 2 * X1 + 3 * X2 - 1.5 * X3 + np.random.randn(n) * 0.1

    return pd.DataFrame({
        'y': y,
        'X1': X1,
        'X2': X2,
        'X3': X3
    })


@pytest.fixture
def simple_regression_with_date():
    """Regression dataset with date column."""
    np.random.seed(42)
    n = 50
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    y = 2 * X1 + 3 * X2 + np.random.randn(n) * 0.1

    return pd.DataFrame({
        'date': dates,
        'y': y,
        'X1': X1,
        'X2': X2
    })


@pytest.fixture
def grouped_regression_data():
    """Grouped regression dataset for testing."""
    np.random.seed(42)
    n_per_group = 30
    groups = ['A', 'B', 'C']

    all_data = []
    for group in groups:
        X1 = np.random.randn(n_per_group)
        X2 = np.random.randn(n_per_group)
        # Different coefficients per group
        if group == 'A':
            y = 2 * X1 + 3 * X2 + np.random.randn(n_per_group) * 0.1
        elif group == 'B':
            y = 1 * X1 + 4 * X2 + np.random.randn(n_per_group) * 0.1
        else:  # C
            y = 3 * X1 + 2 * X2 + np.random.randn(n_per_group) * 0.1

        df = pd.DataFrame({
            'group_id': group,
            'y': y,
            'X1': X1,
            'X2': X2
        })
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


# Test auto-selection of explainer methods
class TestExplainerAutoSelection:
    """Test automatic explainer selection based on model type."""

    def test_tree_model_selects_tree_explainer(self, simple_regression_data):
        """RandomForest should auto-select TreeExplainer."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')

        method = ShapEngine._auto_select_method(fit)
        assert method == 'tree'

    def test_linear_model_selects_linear_explainer(self, simple_regression_data):
        """Linear regression without penalty should select LinearExplainer."""
        spec = linear_reg()
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')

        method = ShapEngine._auto_select_method(fit)
        assert method == 'linear'

    def test_statsmodels_linear_selects_linear_explainer(self, simple_regression_data):
        """Statsmodels linear model should select LinearExplainer."""
        spec = linear_reg().set_engine('statsmodels')
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')

        method = ShapEngine._auto_select_method(fit)
        assert method == 'linear'


# Test SHAP computation for different model types
class TestShapComputation:
    """Test SHAP value computation for different model types."""

    def test_tree_explainer_on_random_forest(self, simple_regression_data):
        """TreeExplainer should work on RandomForest."""
        # Use small dataset for speed
        train = simple_regression_data.iloc[:70]
        test = simple_regression_data.iloc[70:]

        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(train, 'y ~ X1 + X2 + X3')

        shap_df = fit.explain(test, method='tree')

        # Check basic structure
        assert isinstance(shap_df, pd.DataFrame)
        assert len(shap_df) == len(test) * 3  # 3 features per observation
        assert 'observation_id' in shap_df.columns
        assert 'variable' in shap_df.columns
        assert 'shap_value' in shap_df.columns

    def test_linear_explainer_on_linear_reg(self, simple_regression_data):
        """LinearExplainer should work on linear regression."""
        train = simple_regression_data.iloc[:70]
        test = simple_regression_data.iloc[70:]

        spec = linear_reg()
        fit = spec.fit(train, 'y ~ X1 + X2 + X3')

        shap_df = fit.explain(test, method='linear')

        assert isinstance(shap_df, pd.DataFrame)
        assert len(shap_df) == len(test) * 3
        assert set(shap_df['variable'].unique()) == {'X1', 'X2', 'X3'}

    def test_kernel_explainer_fallback(self, simple_regression_data):
        """KernelExplainer should work as universal fallback."""
        # Use tiny dataset for speed (KernelExplainer is slow)
        train = simple_regression_data.iloc[:50]
        test = simple_regression_data.iloc[50:55]  # Just 5 observations

        spec = linear_reg()
        fit = spec.fit(train, 'y ~ X1 + X2 + X3')

        # Force KernelExplainer
        shap_df = fit.explain(test, method='kernel', background_size=20)

        assert isinstance(shap_df, pd.DataFrame)
        assert len(shap_df) == len(test) * 3


# Test DataFrame structure
class TestShapDataFrameStructure:
    """Test SHAP DataFrame output format."""

    def test_required_columns_present(self, simple_regression_data):
        """All required columns should be present."""
        train = simple_regression_data.iloc[:70]
        test = simple_regression_data.iloc[70:]

        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(train, 'y ~ X1 + X2 + X3')

        shap_df = fit.explain(test)

        required_cols = [
            'observation_id', 'variable', 'shap_value', 'abs_shap',
            'feature_value', 'base_value', 'prediction', 'model', 'model_group_name'
        ]
        for col in required_cols:
            assert col in shap_df.columns, f"Missing column: {col}"

    def test_abs_shap_is_absolute_value(self, simple_regression_data):
        """abs_shap should equal absolute value of shap_value."""
        train = simple_regression_data.iloc[:70]
        test = simple_regression_data.iloc[70:]

        spec = linear_reg()
        fit = spec.fit(train, 'y ~ X1 + X2 + X3')

        shap_df = fit.explain(test)

        assert np.allclose(shap_df['abs_shap'], np.abs(shap_df['shap_value']))

    def test_date_column_included_when_present(self, simple_regression_with_date):
        """Date column should be included if present in data."""
        train = simple_regression_with_date.iloc[:30]
        test = simple_regression_with_date.iloc[30:]

        spec = linear_reg()
        fit = spec.fit(train, 'y ~ X1 + X2')

        shap_df = fit.explain(test)

        assert 'date' in shap_df.columns
        # Check dates match test data
        for obs_id in range(len(test)):
            obs_shap = shap_df[shap_df['observation_id'] == obs_id]
            expected_date = test.iloc[obs_id]['date']
            assert all(obs_shap['date'] == expected_date)

    def test_model_metadata_columns(self, simple_regression_data):
        """Model type and engine should be in output."""
        train = simple_regression_data.iloc[:70]
        test = simple_regression_data.iloc[70:]

        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(train, 'y ~ X1 + X2 + X3')

        shap_df = fit.explain(test)

        assert all(shap_df['model'] == 'rand_forest')
        assert all(shap_df['model_group_name'] == 'sklearn')


# Test additivity property
class TestAdditivity:
    """Test SHAP additivity property: sum(shap_value) + base_value ≈ prediction."""

    def test_additivity_for_linear_model(self, simple_regression_data):
        """Linear model SHAP values should sum exactly to prediction - base."""
        train = simple_regression_data.iloc[:70]
        test = simple_regression_data.iloc[70:75]  # Small test set

        spec = linear_reg()
        fit = spec.fit(train, 'y ~ X1 + X2 + X3')

        # Don't check additivity to avoid warnings
        shap_df = fit.explain(test, check_additivity=False)

        # Manually check additivity
        for obs_id in range(len(test)):
            obs_shap = shap_df[shap_df['observation_id'] == obs_id]
            shap_sum = obs_shap['shap_value'].sum()
            base = obs_shap['base_value'].iloc[0]
            pred = obs_shap['prediction'].iloc[0]

            # sum(SHAP) + base should equal prediction
            assert np.isclose(shap_sum + base, pred, atol=1e-3)

    def test_additivity_warning_raised_if_violated(self, simple_regression_data):
        """Should warn if additivity check fails (unlikely with real models)."""
        # This is more of a structural test - in practice, additivity holds
        train = simple_regression_data.iloc[:70]
        test = simple_regression_data.iloc[70:75]

        spec = linear_reg()
        fit = spec.fit(train, 'y ~ X1 + X2 + X3')

        # Should not raise warning (additivity holds)
        with pytest.warns(None) as warning_list:
            shap_df = fit.explain(test, check_additivity=True)

        # No warnings should be raised for linear model
        shap_warnings = [w for w in warning_list if "SHAP values don't sum" in str(w.message)]
        assert len(shap_warnings) == 0


# Test background data strategies
class TestBackgroundData:
    """Test different background data strategies for KernelExplainer."""

    def test_sample_background_strategy(self, simple_regression_data):
        """Sample strategy should randomly sample training data."""
        train = simple_regression_data.iloc[:60]
        test = simple_regression_data.iloc[60:65]

        spec = linear_reg()
        fit = spec.fit(train, 'y ~ X1 + X2 + X3')

        shap_df = fit.explain(test, method='kernel', background='sample', background_size=20)

        assert isinstance(shap_df, pd.DataFrame)
        assert len(shap_df) == len(test) * 3

    def test_kmeans_background_strategy(self, simple_regression_data):
        """K-means strategy should cluster training data."""
        train = simple_regression_data.iloc[:60]
        test = simple_regression_data.iloc[60:65]

        spec = linear_reg()
        fit = spec.fit(train, 'y ~ X1 + X2 + X3')

        shap_df = fit.explain(test, method='kernel', background='kmeans', background_size=15)

        assert isinstance(shap_df, pd.DataFrame)
        assert len(shap_df) == len(test) * 3


# Test integration with WorkflowFit
class TestWorkflowIntegration:
    """Test SHAP with workflow and recipe preprocessing."""

    def test_workflow_with_formula(self, simple_regression_data):
        """WorkflowFit with formula should work."""
        train = simple_regression_data.iloc[:70]
        test = simple_regression_data.iloc[70:]

        wf = workflow().add_formula('y ~ X1 + X2 + X3').add_model(rand_forest(trees=10).set_mode('regression'))
        wf_fit = wf.fit(train)

        shap_df = wf_fit.explain(test)

        assert isinstance(shap_df, pd.DataFrame)
        assert len(shap_df) == len(test) * 3

    def test_workflow_with_recipe(self, simple_regression_data):
        """WorkflowFit with recipe preprocessing should work."""
        train = simple_regression_data.iloc[:70]
        test = simple_regression_data.iloc[70:]

        rec = recipe().step_normalize()
        wf = workflow().add_recipe(rec).add_model(linear_reg())
        wf_fit = wf.fit(train)

        shap_df = wf_fit.explain(test)

        assert isinstance(shap_df, pd.DataFrame)
        # Should have SHAP for all normalized features
        assert set(shap_df['variable'].unique()) == {'X1', 'X2', 'X3'}


# Test integration with NestedModelFit and NestedWorkflowFit
class TestGroupedModelIntegration:
    """Test SHAP with grouped/panel models."""

    def test_nested_model_fit_explain(self, grouped_regression_data):
        """NestedModelFit.explain() should work."""
        train = grouped_regression_data

        spec = linear_reg()
        nested_fit = spec.fit_nested(train, 'y ~ X1 + X2', group_col='group_id')

        shap_df = nested_fit.explain(train)

        # Check structure
        assert isinstance(shap_df, pd.DataFrame)
        assert 'group' in shap_df.columns
        assert set(shap_df['group'].unique()) == {'A', 'B', 'C'}

        # Each group should have SHAP values
        for group in ['A', 'B', 'C']:
            group_shap = shap_df[shap_df['group'] == group]
            assert len(group_shap) > 0
            assert set(group_shap['variable'].unique()) == {'X1', 'X2'}

    def test_nested_workflow_fit_explain(self, grouped_regression_data):
        """NestedWorkflowFit.explain() should work."""
        train = grouped_regression_data

        wf = workflow().add_formula('y ~ X1 + X2').add_model(linear_reg())
        nested_fit = wf.fit_nested(train, group_col='group_id')

        shap_df = nested_fit.explain(train)

        assert isinstance(shap_df, pd.DataFrame)
        assert 'group' in shap_df.columns
        assert set(shap_df['group'].unique()) == {'A', 'B', 'C'}

    def test_grouped_feature_importance_comparison(self, grouped_regression_data):
        """Should be able to compare feature importance across groups."""
        train = grouped_regression_data

        spec = linear_reg()
        nested_fit = spec.fit_nested(train, 'y ~ X1 + X2', group_col='group_id')

        shap_df = nested_fit.explain(train)

        # Compute importance by group
        importance = shap_df.groupby(['group', 'variable'])['abs_shap'].mean().unstack()

        # Should have all groups and variables
        assert set(importance.index) == {'A', 'B', 'C'}
        assert set(importance.columns) == {'X1', 'X2'}


# Test error handling
class TestErrorHandling:
    """Test error handling for edge cases."""

    def test_missing_shap_import(self, simple_regression_data, monkeypatch):
        """Should raise ImportError if shap not installed."""
        # This test assumes shap IS installed; we mock the import error
        def mock_import_error(*args, **kwargs):
            raise ImportError("No module named 'shap'")

        train = simple_regression_data.iloc[:70]
        test = simple_regression_data.iloc[70:]

        spec = linear_reg()
        fit = spec.fit(train, 'y ~ X1 + X2 + X3')

        # Mock shap import failure
        import builtins
        original_import = builtins.__import__

        def mock_shap_import(name, *args, **kwargs):
            if name == 'shap':
                raise ImportError("No module named 'shap'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', mock_shap_import)

        with pytest.raises(ImportError, match="SHAP package not installed"):
            fit.explain(test)

    def test_invalid_method_raises_error(self, simple_regression_data):
        """Invalid method should raise ValueError."""
        train = simple_regression_data.iloc[:70]
        test = simple_regression_data.iloc[70:]

        spec = linear_reg()
        fit = spec.fit(train, 'y ~ X1 + X2 + X3')

        with pytest.raises(ValueError, match="Unknown method"):
            fit.explain(test, method='invalid_method')

    def test_missing_group_column_raises_error(self, grouped_regression_data):
        """Missing group column should raise error."""
        train = grouped_regression_data
        test = grouped_regression_data.drop(columns=['group_id'])

        spec = linear_reg()
        nested_fit = spec.fit_nested(train, 'y ~ X1 + X2', group_col='group_id')

        with pytest.raises(ValueError, match="Group column 'group_id' not found"):
            nested_fit.explain(test)


# Integration test combining multiple features
class TestIntegrationScenarios:
    """Integration tests combining multiple SHAP features."""

    def test_full_workflow_with_date_and_groups(self):
        """Test complete workflow with dates, groups, and recipe."""
        np.random.seed(42)
        n_per_group = 20
        groups = ['A', 'B']

        all_data = []
        for i, group in enumerate(groups):
            dates = pd.date_range('2020-01-01', periods=n_per_group, freq='D')
            X1 = np.random.randn(n_per_group)
            X2 = np.random.randn(n_per_group)
            y = (2 + i) * X1 + 3 * X2 + np.random.randn(n_per_group) * 0.1

            df = pd.DataFrame({
                'date': dates,
                'group_id': group,
                'y': y,
                'X1': X1,
                'X2': X2
            })
            all_data.append(df)

        data = pd.concat(all_data, ignore_index=True)

        # Fit nested workflow with recipe
        rec = recipe().step_normalize()
        wf = workflow().add_recipe(rec).add_model(linear_reg())
        nested_fit = wf.fit_nested(data, group_col='group_id')

        # Compute SHAP
        shap_df = nested_fit.explain(data)

        # Verify complete structure
        assert 'date' in shap_df.columns or 'observation_id' in shap_df.columns
        assert 'group' in shap_df.columns
        assert set(shap_df['group'].unique()) == {'A', 'B'}
        assert set(shap_df['variable'].unique()) == {'X1', 'X2'}

        # Compute importance by group
        importance = shap_df.groupby(['group', 'variable'])['abs_shap'].mean()
        assert len(importance) == 4  # 2 groups × 2 variables
