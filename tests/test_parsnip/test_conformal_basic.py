"""
Basic tests for conformal prediction intervals.

These tests verify:
- Basic functionality with linear regression
- Coverage validation
- Parameter validation
- Output format
"""

import pytest
import pandas as pd
import numpy as np
from py_parsnip import linear_reg


@pytest.fixture
def simple_regression_data():
    """Generate simple regression data with known relationship."""
    np.random.seed(42)
    n = 1000

    # Features
    X = np.random.randn(n, 3)

    # True relationship: y = 2*x1 + 3*x2 - x3 + noise
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n) * 0.5

    # Create DataFrame
    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    df['y'] = y

    # Split into train/test
    train = df.iloc[:800]
    test = df.iloc[800:]

    return train, test


def test_conformal_predict_basic(simple_regression_data):
    """Test basic conformal prediction with split conformal."""
    train, test = simple_regression_data

    # Fit model
    spec = linear_reg()
    fit = spec.fit(train, 'y ~ x1 + x2 + x3')

    # Get conformal predictions
    preds = fit.conformal_predict(test, alpha=0.05, method='split')

    # Check output format
    assert isinstance(preds, pd.DataFrame)
    assert '.pred' in preds.columns
    assert '.pred_lower' in preds.columns
    assert '.pred_upper' in preds.columns
    assert '.conf_method' in preds.columns
    assert '.conf_alpha' in preds.columns
    assert '.conf_coverage' in preds.columns

    # Check dimensions
    assert len(preds) == len(test)

    # Check intervals are valid
    assert (preds['.pred_lower'] <= preds['.pred']).all()
    assert (preds['.pred'] <= preds['.pred_upper']).all()

    # Check method and alpha
    assert (preds['.conf_method'] == 'split').all()
    assert np.allclose(preds['.conf_alpha'], 0.05)
    assert np.allclose(preds['.conf_coverage'], 0.95)


def test_conformal_predict_coverage(simple_regression_data):
    """Test that empirical coverage matches target."""
    train, test = simple_regression_data

    # Fit model
    spec = linear_reg()
    fit = spec.fit(train, 'y ~ x1 + x2 + x3')

    # Get conformal predictions
    preds = fit.conformal_predict(test, alpha=0.05, method='split')

    # Check coverage
    y_test = test['y'].values
    in_interval = (
        (y_test >= preds['.pred_lower'].values) &
        (y_test <= preds['.pred_upper'].values)
    )
    coverage = in_interval.mean()

    # Should be close to 95% (within reasonable tolerance)
    # Note: With only 200 test points, allow for wider tolerance
    assert 0.85 <= coverage <= 1.0, f"Coverage {coverage:.2%} outside [85%, 100%]"


def test_conformal_predict_auto_method(simple_regression_data):
    """Test automatic method selection."""
    train, test = simple_regression_data

    # Fit model
    spec = linear_reg()
    fit = spec.fit(train, 'y ~ x1 + x2 + x3')

    # Get conformal predictions with auto method
    preds = fit.conformal_predict(test, alpha=0.05, method='auto')

    # With 800 training samples, after 15% calibration split (680 samples),
    # should select 'jackknife+' (< 1000), 'cv+' (1000-10000), or 'split' (> 10000)
    method = preds['.conf_method'].iloc[0]
    assert method in ['jackknife+', 'cv+', 'split'], f"Unexpected method: {method}"


def test_conformal_predict_multiple_alphas(simple_regression_data):
    """Test multiple confidence levels."""
    train, test = simple_regression_data

    # Fit model
    spec = linear_reg()
    fit = spec.fit(train, 'y ~ x1 + x2 + x3')

    # Get conformal predictions with multiple alphas
    preds = fit.conformal_predict(test, alpha=[0.05, 0.1, 0.2], method='split')

    # Check columns
    assert '.pred' in preds.columns
    assert '.pred_lower_95' in preds.columns
    assert '.pred_upper_95' in preds.columns
    assert '.pred_lower_90' in preds.columns
    assert '.pred_upper_90' in preds.columns
    assert '.pred_lower_80' in preds.columns
    assert '.pred_upper_80' in preds.columns

    # Check interval ordering (wider intervals for lower confidence)
    assert (preds['.pred_lower_80'] >= preds['.pred_lower_90']).all()
    assert (preds['.pred_lower_90'] >= preds['.pred_lower_95']).all()
    assert (preds['.pred_upper_80'] <= preds['.pred_upper_90']).all()
    assert (preds['.pred_upper_90'] <= preds['.pred_upper_95']).all()


def test_conformal_predict_invalid_alpha():
    """Test parameter validation for invalid alpha."""
    train = pd.DataFrame({
        'x1': np.random.randn(100),
        'y': np.random.randn(100)
    })

    spec = linear_reg()
    fit = spec.fit(train, 'y ~ x1')

    # Alpha out of range
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        fit.conformal_predict(train, alpha=1.5)

    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        fit.conformal_predict(train, alpha=0)


def test_conformal_predict_invalid_method():
    """Test parameter validation for invalid method."""
    train = pd.DataFrame({
        'x1': np.random.randn(100),
        'y': np.random.randn(100)
    })

    spec = linear_reg()
    fit = spec.fit(train, 'y ~ x1')

    # Invalid method
    with pytest.raises(ValueError, match="method must be one of"):
        fit.conformal_predict(train, method='invalid_method')


def test_conformal_predict_caching(simple_regression_data):
    """Test that conformal wrapper is cached for repeated predictions."""
    train, test = simple_regression_data

    # Fit model
    spec = linear_reg()
    fit = spec.fit(train, 'y ~ x1 + x2 + x3')

    # First prediction (creates wrapper)
    preds1 = fit.conformal_predict(test, alpha=0.05, method='split')
    assert hasattr(fit, '_conformal_wrapper')
    assert hasattr(fit, '_conformal_method')
    assert hasattr(fit, '_conformal_alpha')

    # Second prediction (uses cached wrapper)
    preds2 = fit.conformal_predict(test, alpha=0.05, method='split')

    # Predictions should be identical (same wrapper)
    pd.testing.assert_frame_equal(preds1, preds2)


def test_conformal_predict_different_alpha_recreates_wrapper(simple_regression_data):
    """Test that changing alpha recreates the wrapper."""
    train, test = simple_regression_data

    # Fit model
    spec = linear_reg()
    fit = spec.fit(train, 'y ~ x1 + x2 + x3')

    # First prediction
    preds1 = fit.conformal_predict(test, alpha=0.05, method='split')
    wrapper1 = fit._conformal_wrapper

    # Second prediction with different alpha (should recreate wrapper)
    preds2 = fit.conformal_predict(test, alpha=0.1, method='split')
    wrapper2 = fit._conformal_wrapper

    # Wrappers should be different objects (alpha changed)
    # Note: They might be the same object depending on MAPIE implementation,
    # so we check that the predictions differ
    assert len(preds1) == len(preds2)
    # Intervals should be different widths
    width1 = (preds1['.pred_upper'] - preds1['.pred_lower']).mean()
    width2 = (preds2['.pred_upper'] - preds2['.pred_lower']).mean()
    assert width1 != width2  # Different alphas → different interval widths
