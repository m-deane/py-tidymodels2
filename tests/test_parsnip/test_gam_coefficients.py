"""
Test gen_additive_mod() coefficients format standardization (Issue 4)
"""
import pytest
import pandas as pd
import numpy as np
from py_parsnip import gen_additive_mod


@pytest.fixture
def simple_data():
    """Simple regression dataset"""
    np.random.seed(42)
    n = 100
    x1 = np.linspace(0, 10, n)
    x2 = np.linspace(0, 5, n)
    y = 2 * x1 + np.sin(x2) + np.random.normal(0, 0.5, n)

    return pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'y': y
    })


def test_gam_coefficients_format(simple_data):
    """Test that GAM returns standard coefficients format"""
    spec = gen_additive_mod(adjust_deg_free=10)
    fit = spec.fit(simple_data, 'y ~ x1 + x2')

    outputs, coefficients, stats = fit.extract_outputs()

    # Verify coefficients DataFrame has standard columns
    expected_columns = ['variable', 'coefficient', 'std_error', 't_stat', 'p_value',
                       'ci_0.025', 'ci_0.975', 'vif', 'model', 'model_group_name', 'group']

    for col in expected_columns:
        assert col in coefficients.columns, f"Missing column: {col}"

    # Verify we have entries for features (may include Intercept)
    assert len(coefficients) >= 2
    assert 'x1' in coefficients['variable'].values
    assert 'x2' in coefficients['variable'].values

    # Verify coefficient values are numeric (effect_range)
    assert coefficients['coefficient'].dtype in [np.float64, np.float32]
    assert not coefficients['coefficient'].isna().all()

    # Verify statistical columns are NaN (not applicable for GAM)
    assert coefficients['std_error'].isna().all()
    assert coefficients['t_stat'].isna().all()
    assert coefficients['p_value'].isna().all()
    assert coefficients['ci_0.025'].isna().all()
    assert coefficients['ci_0.975'].isna().all()
    assert coefficients['vif'].isna().all()


def test_gam_no_longer_returns_partial_effects(simple_data):
    """Test that GAM no longer returns partial_effects format"""
    spec = gen_additive_mod(adjust_deg_free=10)
    fit = spec.fit(simple_data, 'y ~ x1 + x2')

    outputs, coefficients, stats = fit.extract_outputs()

    # Verify old partial_effects columns are NOT present
    old_columns = ['feature', 'feature_index', 'effect_range', 'data_range', 'data_min', 'data_max']

    for col in old_columns:
        assert col not in coefficients.columns, f"Old partial_effects column still present: {col}"


def test_gam_coefficients_match_other_engines_format(simple_data):
    """Test that GAM coefficients match the same format as sklearn_rand_forest"""
    spec = gen_additive_mod(adjust_deg_free=10)
    fit = spec.fit(simple_data, 'y ~ x1 + x2')

    outputs, coefficients, stats = fit.extract_outputs()

    # Verify exact column order matches standard format
    expected_order = ['variable', 'coefficient', 'std_error', 't_stat', 'p_value',
                     'ci_0.025', 'ci_0.975', 'vif', 'model', 'model_group_name', 'group']

    actual_columns = list(coefficients.columns)
    assert actual_columns == expected_order, f"Column order mismatch: {actual_columns}"


def test_gam_coefficient_values_are_effect_ranges(simple_data):
    """Test that coefficient values represent effect ranges (feature importance)"""
    spec = gen_additive_mod(adjust_deg_free=10)
    fit = spec.fit(simple_data, 'y ~ x1 + x2')

    outputs, coefficients, stats = fit.extract_outputs()

    # Effect ranges should be positive (range of partial dependence)
    assert (coefficients['coefficient'] >= 0).all(), "Effect ranges should be non-negative"

    # At least one feature should have non-zero effect
    assert (coefficients['coefficient'] > 0).any(), "At least one feature should have effect"
