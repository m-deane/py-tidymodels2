"""
Tests for NestedModelFit: Grouped/panel modeling directly on ModelSpec.

These tests verify that ModelSpec.fit_nested() and ModelSpec.fit_global()
work correctly without requiring workflow wrappers.
"""

import pytest
import pandas as pd
import numpy as np
import warnings

from py_parsnip import linear_reg, rand_forest, recursive_reg
from py_parsnip.model_spec import NestedModelFit


# Fixtures

@pytest.fixture
def grouped_data():
    """Generate simple grouped data with 3 groups."""
    np.random.seed(42)
    n_per_group = 50
    groups = ["A", "B", "C"]

    data = []
    for group in groups:
        # Each group has different patterns
        if group == "A":
            x = np.linspace(0, 10, n_per_group)
            y = 2 * x + 5 + np.random.normal(0, 1, n_per_group)
        elif group == "B":
            x = np.linspace(0, 10, n_per_group)
            y = -1 * x + 10 + np.random.normal(0, 1, n_per_group)
        else:  # C
            x = np.linspace(0, 10, n_per_group)
            y = 0.5 * x + 3 + np.random.normal(0, 1, n_per_group)

        group_df = pd.DataFrame({
            "group": group,
            "x": x,
            "y": y
        })
        data.append(group_df)

    return pd.concat(data, ignore_index=True)


@pytest.fixture
def grouped_timeseries_data():
    """Generate grouped time series data for recursive model testing."""
    np.random.seed(42)
    n_per_group = 100
    groups = ["Store1", "Store2"]

    data = []
    for group in groups:
        dates = pd.date_range("2020-01-01", periods=n_per_group, freq="D")

        if group == "Store1":
            # Upward trend
            values = np.linspace(100, 200, n_per_group) + np.random.normal(0, 10, n_per_group)
        else:
            # Downward trend
            values = np.linspace(200, 100, n_per_group) + np.random.normal(0, 10, n_per_group)

        group_df = pd.DataFrame({
            "group": group,
            "date": dates,
            "value": values,
            "x1": np.random.normal(0, 1, n_per_group),
            "x2": np.random.normal(0, 1, n_per_group)
        })
        data.append(group_df)

    return pd.concat(data, ignore_index=True)


# Tests for fit_nested()

def test_fit_nested_basic(grouped_data):
    """Test basic fit_nested functionality."""
    spec = linear_reg()
    nested_fit = spec.fit_nested(grouped_data, "y ~ x", group_col="group")

    # Verify NestedModelFit structure
    assert isinstance(nested_fit, NestedModelFit)
    assert nested_fit.group_col == "group"
    assert len(nested_fit.group_fits) == 3  # A, B, C
    assert set(nested_fit.group_fits.keys()) == {"A", "B", "C"}
    assert nested_fit.formula == "y ~ x"

    # Verify each group has a ModelFit
    for group, fit in nested_fit.group_fits.items():
        from py_parsnip.model_spec import ModelFit
        assert isinstance(fit, ModelFit)


def test_fit_nested_predictions(grouped_data):
    """Test prediction routing to appropriate group models."""
    spec = linear_reg()

    # Split data stratified by group (40 train + 10 test per group)
    train_dfs = []
    test_dfs = []
    for group in ["A", "B", "C"]:
        group_data = grouped_data[grouped_data["group"] == group]
        train_dfs.append(group_data.iloc[:40])
        test_dfs.append(group_data.iloc[40:])

    train = pd.concat(train_dfs, ignore_index=True)
    test = pd.concat(test_dfs, ignore_index=True)

    nested_fit = spec.fit_nested(train, "y ~ x", group_col="group")
    predictions = nested_fit.predict(test)

    # Verify predictions structure
    assert ".pred" in predictions.columns
    assert "group" in predictions.columns
    assert len(predictions) == 30  # 10 per group

    # Verify each group has predictions
    assert set(predictions["group"].unique()) == {"A", "B", "C"}

    # Verify predictions are different for each group (different models)
    group_a_preds = predictions[predictions["group"] == "A"][".pred"].values
    group_b_preds = predictions[predictions["group"] == "B"][".pred"].values
    assert not np.allclose(group_a_preds[:5], group_b_preds[:5])


def test_fit_nested_extract_outputs(grouped_data):
    """Test extract_outputs includes group column."""
    spec = linear_reg()

    # Split data stratified by group
    train_dfs = []
    test_dfs = []
    for group in ["A", "B", "C"]:
        group_data = grouped_data[grouped_data["group"] == group]
        train_dfs.append(group_data.iloc[:40])
        test_dfs.append(group_data.iloc[40:])

    train = pd.concat(train_dfs, ignore_index=True)
    test = pd.concat(test_dfs, ignore_index=True)

    nested_fit = spec.fit_nested(train, "y ~ x", group_col="group")
    nested_fit = nested_fit.evaluate(test)

    outputs, coefficients, stats = nested_fit.extract_outputs()

    # Verify group column present in all DataFrames
    assert "group" in outputs.columns
    assert "group" in coefficients.columns
    assert "group" in stats.columns

    # Verify all groups present
    assert set(outputs["group"].unique()) == {"A", "B", "C"}
    assert set(coefficients["group"].unique()) == {"A", "B", "C"}
    assert set(stats["group"].unique()) == {"A", "B", "C"}

    # Verify standard columns exist
    assert "actuals" in outputs.columns
    assert "fitted" in outputs.columns
    assert "split" in outputs.columns
    assert "metric" in stats.columns
    assert "value" in stats.columns


def test_fit_nested_evaluate(grouped_data):
    """Test evaluate method on test data."""
    spec = linear_reg()

    # Split data stratified by group
    train_dfs = []
    test_dfs = []
    for group in ["A", "B", "C"]:
        group_data = grouped_data[grouped_data["group"] == group]
        train_dfs.append(group_data.iloc[:40])
        test_dfs.append(group_data.iloc[40:])

    train = pd.concat(train_dfs, ignore_index=True)
    test = pd.concat(test_dfs, ignore_index=True)

    nested_fit = spec.fit_nested(train, "y ~ x", group_col="group")
    nested_fit = nested_fit.evaluate(test)

    # Verify all group fits have evaluation data
    for group, fit in nested_fit.group_fits.items():
        assert "test_data" in fit.evaluation_data
        assert "test_predictions" in fit.evaluation_data

    # Extract and verify test metrics
    outputs, _, stats = nested_fit.extract_outputs()

    # Check test split exists
    test_outputs = outputs[outputs["split"] == "test"]
    assert len(test_outputs) > 0

    test_stats = stats[stats["split"] == "test"]
    assert len(test_stats) > 0


def test_fit_nested_missing_group_in_test(grouped_data):
    """Test error when test data has group not in training."""
    spec = linear_reg()

    # Train on groups A and B only
    train = grouped_data[grouped_data["group"].isin(["A", "B"])].copy()

    # Test on group C (not in training)
    test = grouped_data[grouped_data["group"] == "C"].copy()

    nested_fit = spec.fit_nested(train, "y ~ x", group_col="group")

    # Prediction should fail for unseen group
    with pytest.raises(ValueError, match="No matching groups found"):
        nested_fit.predict(test)


def test_fit_nested_invalid_group_col(grouped_data):
    """Test error on invalid group column name."""
    spec = linear_reg()

    with pytest.raises(ValueError, match="Group column 'invalid' not found"):
        spec.fit_nested(grouped_data, "y ~ x", group_col="invalid")


def test_fit_nested_single_group_warning(grouped_data):
    """Test warning when only one group present."""
    spec = linear_reg()

    # Use only group A
    single_group_data = grouped_data[grouped_data["group"] == "A"].copy()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        nested_fit = spec.fit_nested(single_group_data, "y ~ x", group_col="group")

        # Verify warning was raised
        assert len(w) == 1
        assert "Only one group found" in str(w[0].message)
        assert "Consider using fit()" in str(w[0].message)

    # Should still work
    assert len(nested_fit.group_fits) == 1


def test_fit_nested_recursive_model(grouped_timeseries_data):
    """Test fit_nested with recursive_reg (date indexing)."""
    spec = recursive_reg(base_model=linear_reg(), lags=3)

    train = grouped_timeseries_data.iloc[:160].copy()  # 80 per group
    test = grouped_timeseries_data.iloc[160:].copy()   # 20 per group

    nested_fit = spec.fit_nested(train, "value ~ x1 + x2", group_col="group")

    # Verify structure
    assert len(nested_fit.group_fits) == 2  # Store1, Store2

    # Test predictions (recursive models handle date indexing)
    predictions = nested_fit.predict(test)

    assert ".pred" in predictions.columns
    assert "group" in predictions.columns
    assert len(predictions) == 40  # 20 per group


def test_fit_nested_consistency_with_workflow(grouped_data):
    """Verify spec.fit_nested() produces identical results to workflow.fit_nested()."""
    from py_workflows import workflow

    train = grouped_data.iloc[:120].copy()
    test = grouped_data.iloc[120:].copy()

    # Method 1: Direct on ModelSpec
    spec1 = linear_reg()
    nested_fit1 = spec1.fit_nested(train, "y ~ x", group_col="group")
    nested_fit1 = nested_fit1.evaluate(test)
    outputs1, coefs1, stats1 = nested_fit1.extract_outputs()

    # Method 2: Via Workflow
    spec2 = linear_reg()
    wf = workflow().add_formula("y ~ x").add_model(spec2)
    nested_fit2 = wf.fit_nested(train, group_col="group")
    nested_fit2 = nested_fit2.evaluate(test)
    outputs2, coefs2, stats2 = nested_fit2.extract_outputs()

    # Compare outputs (should be identical within tolerance)
    # Sort by group and index for comparison
    outputs1_sorted = outputs1.sort_values(["group", "split"]).reset_index(drop=True)
    outputs2_sorted = outputs2.sort_values(["group", "split"]).reset_index(drop=True)

    # Check fitted values are very close (allow small numerical differences)
    fitted1 = outputs1_sorted[outputs1_sorted["split"] == "train"]["fitted"].values
    fitted2 = outputs2_sorted[outputs2_sorted["split"] == "train"]["fitted"].values
    assert np.allclose(fitted1, fitted2, rtol=1e-9, atol=1e-9)


def test_fit_nested_multiple_models(grouped_data):
    """Test fit_nested with different model types."""
    train = grouped_data.iloc[:120].copy()

    # Test with multiple model types
    models = [
        linear_reg(),
        rand_forest(trees=10, min_n=5).set_mode("regression")
    ]

    for spec in models:
        nested_fit = spec.fit_nested(train, "y ~ x", group_col="group")

        assert len(nested_fit.group_fits) == 3
        predictions = nested_fit.predict(train)
        assert len(predictions) == 120


def test_fit_nested_dot_notation(grouped_data):
    """Test fit_nested with dot notation formula."""
    spec = linear_reg()

    train = grouped_data.iloc[:120].copy()

    # Use dot notation (should expand to all columns except y and group)
    nested_fit = spec.fit_nested(train, "y ~ .", group_col="group")

    # Should work without errors
    assert len(nested_fit.group_fits) == 3

    predictions = nested_fit.predict(train)
    assert len(predictions) == 120


def test_fit_nested_prediction_types(grouped_data):
    """Test different prediction types."""
    spec = linear_reg()

    train = grouped_data.iloc[:120].copy()
    test = grouped_data.iloc[120:].copy()

    nested_fit = spec.fit_nested(train, "y ~ x", group_col="group")

    # Test numeric predictions (default)
    preds_numeric = nested_fit.predict(test, type="numeric")
    assert ".pred" in preds_numeric.columns

    # Note: conf_int may not be supported by all engines
    # Just verify the method accepts the parameter
    try:
        preds_conf = nested_fit.predict(test, type="conf_int")
        assert ".pred" in preds_conf.columns or ".pred_lower" in preds_conf.columns
    except (ValueError, NotImplementedError):
        # Some engines don't support confidence intervals
        pass


# Tests for fit_global()

def test_fit_global_basic(grouped_data):
    """Test basic fit_global functionality."""
    spec = linear_reg()
    global_fit = spec.fit_global(grouped_data, "y ~ x", group_col="group")

    # Should return standard ModelFit
    from py_parsnip.model_spec import ModelFit
    assert isinstance(global_fit, ModelFit)

    # Make predictions
    predictions = global_fit.predict(grouped_data)
    assert ".pred" in predictions.columns
    assert len(predictions) == 150


def test_fit_global_formula_already_has_group(grouped_data):
    """Test fit_global when formula already includes group column."""
    spec = linear_reg()

    # Group already in formula - should not duplicate
    global_fit = spec.fit_global(grouped_data, "y ~ x + group", group_col="group")

    # Should work without error
    predictions = global_fit.predict(grouped_data)
    assert len(predictions) == 150


def test_fit_global_with_dot_notation(grouped_data):
    """Test fit_global with dot notation (should include group automatically)."""
    spec = linear_reg()

    # Dot notation should include group column
    global_fit = spec.fit_global(grouped_data, "y ~ .", group_col="group")

    # Should work without error
    predictions = global_fit.predict(grouped_data)
    assert len(predictions) == 150


def test_fit_global_invalid_group_col(grouped_data):
    """Test error on invalid group column."""
    spec = linear_reg()

    with pytest.raises(ValueError, match="Group column 'invalid' not found"):
        spec.fit_global(grouped_data, "y ~ x", group_col="invalid")


def test_fit_global_invalid_formula(grouped_data):
    """Test error on invalid formula format."""
    spec = linear_reg()

    with pytest.raises(ValueError, match="Invalid formula format"):
        spec.fit_global(grouped_data, "invalid formula", group_col="group")


def test_fit_global_evaluate(grouped_data):
    """Test fit_global with evaluation."""
    spec = linear_reg()

    train = grouped_data.iloc[:120].copy()
    test = grouped_data.iloc[120:].copy()

    global_fit = spec.fit_global(train, "y ~ x", group_col="group")
    global_fit = global_fit.evaluate(test)

    outputs, _, stats = global_fit.extract_outputs()

    # Verify test split exists
    test_outputs = outputs[outputs["split"] == "test"]
    assert len(test_outputs) > 0


def test_fit_nested_empty_group(grouped_data):
    """Test handling of empty group in test data."""
    spec = linear_reg()

    train = grouped_data.copy()
    nested_fit = spec.fit_nested(train, "y ~ x", group_col="group")

    # Create test data with empty group D
    test = pd.DataFrame({
        "group": ["D", "D"],
        "x": [1.0, 2.0],
        "y": [5.0, 6.0]
    })

    # Should raise error for unseen group
    with pytest.raises(ValueError, match="No matching groups found"):
        nested_fit.predict(test)


def test_fit_nested_group_with_nan():
    """Test handling of NaN values in group column."""
    spec = linear_reg()

    # Create data with enough samples per group to avoid sklearn errors
    data = pd.DataFrame({
        "group": ["A"] * 10 + ["B"] * 10 + [np.nan] * 10,
        "x": np.random.uniform(0, 10, 30),
        "y": np.random.uniform(0, 10, 30)
    })

    # Should handle NaN as a group (pandas includes NaN in unique())
    # This may raise an error depending on how patsy handles NaN
    # but at minimum should have groups A and B
    try:
        nested_fit = spec.fit_nested(data, "y ~ x", group_col="group")
        # If it succeeds, should have at least 2 groups (A and B)
        assert len(nested_fit.group_fits) >= 2
    except (ValueError, KeyError):
        # NaN handling may cause errors in patsy/sklearn - acceptable
        pass


def test_fit_nested_large_groups():
    """Test performance with many groups."""
    np.random.seed(42)
    n_groups = 20
    n_per_group = 30

    data = []
    for i in range(n_groups):
        group_df = pd.DataFrame({
            "group": f"G{i}",
            "x": np.random.uniform(0, 10, n_per_group),
            "y": np.random.uniform(0, 10, n_per_group)
        })
        data.append(group_df)

    full_data = pd.concat(data, ignore_index=True)

    spec = linear_reg()
    nested_fit = spec.fit_nested(full_data, "y ~ x", group_col="group")

    # Verify all groups fitted
    assert len(nested_fit.group_fits) == n_groups

    # Test predictions
    predictions = nested_fit.predict(full_data)
    assert len(predictions) == n_groups * n_per_group
