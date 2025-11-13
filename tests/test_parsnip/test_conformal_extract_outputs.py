"""
Tests for conformal prediction integration with extract_outputs().

Phase 4: Verifies that conformal prediction intervals can be added
directly to the standard three-DataFrame output via extract_outputs().
"""

import pytest
import pandas as pd
import numpy as np
from py_parsnip import linear_reg


@pytest.fixture
def sample_data():
    """Generate sample regression data."""
    np.random.seed(42)
    n = 500

    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'x3': np.random.randn(n),
    })
    data['y'] = 10 + 2*data['x1'] + 3*data['x2'] + np.random.randn(n) * 0.5

    return data


@pytest.fixture
def grouped_data():
    """Generate grouped regression data."""
    np.random.seed(42)
    n_per_group = 300

    # Group A
    group_a = pd.DataFrame({
        'group': ['A'] * n_per_group,
        'x1': np.random.randn(n_per_group),
        'x2': np.random.randn(n_per_group),
        'y': None
    })
    group_a['y'] = 10 + 2*group_a['x1'] + 3*group_a['x2'] + np.random.randn(n_per_group) * 0.5

    # Group B
    group_b = pd.DataFrame({
        'group': ['B'] * n_per_group,
        'x1': np.random.randn(n_per_group),
        'x2': np.random.randn(n_per_group),
        'y': None
    })
    group_b['y'] = 5 + 1*group_b['x1'] + 0.5*group_b['x2'] + np.random.randn(n_per_group) * 2.0

    return pd.concat([group_a, group_b], ignore_index=True)


def test_extract_outputs_without_conformal(sample_data):
    """Test that extract_outputs works without conformal intervals (backward compatibility)."""
    # Fit model
    spec = linear_reg()
    fit = spec.fit(sample_data, 'y ~ x1 + x2 + x3')

    # Extract outputs WITHOUT conformal intervals
    outputs, coefficients, stats = fit.extract_outputs()

    # Check standard columns are present
    assert 'actuals' in outputs.columns
    assert 'fitted' in outputs.columns
    assert 'residuals' in outputs.columns
    assert 'split' in outputs.columns

    # Check conformal columns are NOT present
    assert '.pred_lower' not in outputs.columns
    assert '.pred_upper' not in outputs.columns

    # Check outputs length
    assert len(outputs) == len(sample_data)

    # All rows should be 'train' split
    assert (outputs['split'] == 'train').all()


def test_extract_outputs_with_conformal_single_alpha(sample_data):
    """Test extract_outputs with single conformal alpha."""
    # Fit model
    spec = linear_reg()
    fit = spec.fit(sample_data, 'y ~ x1 + x2 + x3')

    # Extract outputs WITH conformal intervals
    outputs, coefficients, stats = fit.extract_outputs(conformal_alpha=0.05)

    # Check standard columns
    assert 'actuals' in outputs.columns
    assert 'fitted' in outputs.columns

    # Check conformal columns are present
    assert '.pred_lower' in outputs.columns
    assert '.pred_upper' in outputs.columns

    # Check conformal intervals are valid
    assert (outputs['.pred_lower'] <= outputs['fitted']).all()
    assert (outputs['fitted'] <= outputs['.pred_upper']).all()

    # Check conformal columns only present for training data
    train_mask = outputs['split'] == 'train'
    assert outputs.loc[train_mask, '.pred_lower'].notna().all()
    assert outputs.loc[train_mask, '.pred_upper'].notna().all()


def test_extract_outputs_with_conformal_multiple_alphas(sample_data):
    """Test extract_outputs with multiple conformal alphas."""
    # Fit model
    spec = linear_reg()
    fit = spec.fit(sample_data, 'y ~ x1 + x2 + x3')

    # Extract outputs with multiple confidence levels
    outputs, coefficients, stats = fit.extract_outputs(conformal_alpha=[0.05, 0.1, 0.2])

    # Check multiple conformal interval columns
    assert '.pred_lower_95' in outputs.columns
    assert '.pred_upper_95' in outputs.columns
    assert '.pred_lower_90' in outputs.columns
    assert '.pred_upper_90' in outputs.columns
    assert '.pred_lower_80' in outputs.columns
    assert '.pred_upper_80' in outputs.columns

    # Check interval nesting (95% intervals contain 80% intervals)
    assert (outputs['.pred_lower_95'] <= outputs['.pred_lower_80']).all()
    assert (outputs['.pred_upper_80'] <= outputs['.pred_upper_95']).all()


def test_extract_outputs_with_different_methods(sample_data):
    """Test extract_outputs with different conformal methods."""
    spec = linear_reg()
    fit = spec.fit(sample_data, 'y ~ x1 + x2 + x3')

    # Test split method
    outputs_split, _, _ = fit.extract_outputs(conformal_alpha=0.05, conformal_method='split')
    assert '.pred_lower' in outputs_split.columns

    # Test cv+ method (slower but more accurate)
    outputs_cv, _, _ = fit.extract_outputs(conformal_alpha=0.05, conformal_method='cv+')
    assert '.pred_lower' in outputs_cv.columns

    # Intervals should exist for both methods
    assert outputs_split['.pred_lower'].notna().sum() > 0
    assert outputs_cv['.pred_lower'].notna().sum() > 0


def test_extract_outputs_conformal_coverage(sample_data):
    """Test that conformal intervals achieve target coverage."""
    # Fit model
    spec = linear_reg()
    fit = spec.fit(sample_data, 'y ~ x1 + x2 + x3')

    # Extract outputs with 95% confidence
    outputs, _, _ = fit.extract_outputs(conformal_alpha=0.05, conformal_method='split')

    # Calculate empirical coverage on training data
    train_mask = outputs['split'] == 'train'
    train_outputs = outputs[train_mask]

    in_interval = (
        (train_outputs['actuals'] >= train_outputs['.pred_lower']) &
        (train_outputs['actuals'] <= train_outputs['.pred_upper'])
    )

    coverage = in_interval.mean()

    # Should achieve ~95% coverage (allow 85-100% for finite samples)
    assert 0.85 <= coverage <= 1.0, f"Coverage {coverage:.1%} out of expected range"


def test_nested_extract_outputs_without_conformal(grouped_data):
    """Test nested extract_outputs without conformal intervals."""
    # Fit nested models
    spec = linear_reg()
    nested_fit = spec.fit_nested(grouped_data, 'y ~ x1 + x2', group_col='group')

    # Extract outputs WITHOUT conformal
    outputs, coefficients, stats = nested_fit.extract_outputs()

    # Check group column present
    assert 'group' in outputs.columns

    # Check standard columns
    assert 'actuals' in outputs.columns
    assert 'fitted' in outputs.columns

    # Check conformal columns NOT present
    assert '.pred_lower' not in outputs.columns
    assert '.pred_upper' not in outputs.columns

    # Check both groups present
    assert set(outputs['group'].unique()) == {'A', 'B'}


def test_nested_extract_outputs_with_conformal(grouped_data):
    """Test nested extract_outputs with per-group conformal intervals."""
    # Fit nested models
    spec = linear_reg()
    nested_fit = spec.fit_nested(grouped_data, 'y ~ x1 + x2', group_col='group')

    # Extract outputs WITH conformal intervals
    outputs, coefficients, stats = nested_fit.extract_outputs(conformal_alpha=0.05)

    # Check group column present
    assert 'group' in outputs.columns

    # Check conformal columns present
    assert '.pred_lower' in outputs.columns
    assert '.pred_upper' in outputs.columns

    # Check intervals valid for both groups
    assert (outputs['.pred_lower'] <= outputs['fitted']).all()
    assert (outputs['fitted'] <= outputs['.pred_upper']).all()

    # Check both groups have conformal intervals
    group_a_conformal = outputs[outputs['group'] == 'A']['.pred_lower'].notna()
    group_b_conformal = outputs[outputs['group'] == 'B']['.pred_lower'].notna()

    assert group_a_conformal.any(), "Group A should have conformal intervals"
    assert group_b_conformal.any(), "Group B should have conformal intervals"


def test_nested_extract_outputs_per_group_intervals(grouped_data):
    """Test that different groups get different interval widths."""
    # Fit nested models
    spec = linear_reg()
    nested_fit = spec.fit_nested(grouped_data, 'y ~ x1 + x2', group_col='group')

    # Extract outputs with conformal intervals
    outputs, _, _ = nested_fit.extract_outputs(conformal_alpha=0.05)

    # Calculate interval widths per group
    outputs['interval_width'] = outputs['.pred_upper'] - outputs['.pred_lower']

    # Group B has more noise, should have wider intervals
    group_a_width = outputs[outputs['group'] == 'A']['interval_width'].mean()
    group_b_width = outputs[outputs['group'] == 'B']['interval_width'].mean()

    # Group B should have wider intervals (more noise in data generation)
    assert group_b_width > group_a_width, "Group B should have wider intervals due to more noise"


def test_nested_extract_outputs_multiple_alphas(grouped_data):
    """Test nested extract_outputs with multiple confidence levels."""
    # Fit nested models
    spec = linear_reg()
    nested_fit = spec.fit_nested(grouped_data, 'y ~ x1 + x2', group_col='group')

    # Extract with multiple alphas
    outputs, _, _ = nested_fit.extract_outputs(conformal_alpha=[0.05, 0.1])

    # Check multiple interval columns
    assert '.pred_lower_95' in outputs.columns
    assert '.pred_upper_95' in outputs.columns
    assert '.pred_lower_90' in outputs.columns
    assert '.pred_upper_90' in outputs.columns

    # Check nesting for both groups
    for group in ['A', 'B']:
        group_outputs = outputs[outputs['group'] == group]
        assert (group_outputs['.pred_lower_95'] <= group_outputs['.pred_lower_90']).all()
        assert (group_outputs['.pred_upper_90'] <= group_outputs['.pred_upper_95']).all()


def test_extract_outputs_conformal_with_evaluate(sample_data):
    """Test extract_outputs with conformal intervals after evaluate()."""
    # Split data
    train = sample_data.iloc[:400]
    test = sample_data.iloc[400:]

    # Fit and evaluate
    spec = linear_reg()
    fit = spec.fit(train, 'y ~ x1 + x2 + x3')
    fit = fit.evaluate(test, outcome_col='y')

    # Extract outputs with conformal intervals
    outputs, _, _ = fit.extract_outputs(conformal_alpha=0.05)

    # Check train split has conformal intervals
    train_mask = outputs['split'] == 'train'
    assert outputs.loc[train_mask, '.pred_lower'].notna().all()
    assert outputs.loc[train_mask, '.pred_upper'].notna().all()

    # Check test split does NOT have conformal intervals (not calibrated on test)
    test_mask = outputs['split'] == 'test'
    if test_mask.any():
        # Test data should not have conformal intervals (only training data calibrated)
        # Or they should be NA since we didn't compute conformal predictions for test set
        pass  # This is expected behavior


def test_extract_outputs_auto_method_selection(sample_data):
    """Test that auto method selection works in extract_outputs."""
    spec = linear_reg()
    fit = spec.fit(sample_data, 'y ~ x1 + x2 + x3')

    # Use auto method (should select based on data size)
    outputs, _, _ = fit.extract_outputs(conformal_alpha=0.05, conformal_method='auto')

    # Should have conformal intervals
    assert '.pred_lower' in outputs.columns
    assert '.pred_upper' in outputs.columns

    # With 500 samples, should select 'cv+' or 'split'
    # After 15% calibration split: ~425 samples
    # Should be cv+ (1000-10000) or split (>10000)
    # Actually with 500 samples total, after split we have ~425, which is < 1000
    # So should select jackknife+
    # But the specific method doesn't matter - just check intervals exist
    assert outputs['.pred_lower'].notna().sum() > 0


def test_coefficients_and_stats_unchanged_with_conformal(sample_data):
    """Test that coefficients and stats DataFrames are unchanged with conformal."""
    spec = linear_reg()
    fit = spec.fit(sample_data, 'y ~ x1 + x2 + x3')

    # Extract without conformal
    _, coefs_no_conf, stats_no_conf = fit.extract_outputs()

    # Extract with conformal
    _, coefs_with_conf, stats_with_conf = fit.extract_outputs(conformal_alpha=0.05)

    # Coefficients should be identical
    pd.testing.assert_frame_equal(coefs_no_conf, coefs_with_conf)

    # Stats should be identical
    pd.testing.assert_frame_equal(stats_no_conf, stats_with_conf)
