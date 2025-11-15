"""
Tests for Prior Sensitivity Analysis

Tests the compare_priors() function that compares different prior specifications.
"""

import pytest
import pandas as pd
import numpy as np

# Try to import PyMC and ArviZ, skip tests if not available
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

from py_parsnip import linear_reg
from py_bayes.analysis import compare_priors
from py_yardstick import metric_set, rmse, mae


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def small_linear_data():
    """Small synthetic dataset for fast comparison."""
    np.random.seed(42)
    n = 40

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)

    # True model: y = 2 + 3*x1 - 1.5*x2 + noise
    y = 2.0 + 3.0*x1 - 1.5*x2 + np.random.randn(n) * 0.5

    return pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})


@pytest.fixture
def train_test_split(small_linear_data):
    """Train/test split for sensitivity analysis."""
    train = small_linear_data.iloc[:30]
    test = small_linear_data.iloc[30:]
    return train, test


# ============================================================================
# BASIC COMPARISON TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_compare_priors_basic(small_linear_data):
    """Test basic prior comparison functionality."""
    spec = linear_reg().set_engine("pymc")

    priors = {
        "weak": {"prior_coefs": "normal(0, 10)"},
        "medium": {"prior_coefs": "normal(0, 5)"},
        "strong": {"prior_coefs": "normal(0, 1)"}
    }

    results = compare_priors(
        model_spec=spec,
        data=small_linear_data,
        formula="y ~ x1 + x2",
        priors=priors,
        draws=200,  # Small for speed
        tune=200,
        chains=2,
        verbose=False
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 3  # One row per prior
    assert "prior_name" in results.columns
    assert set(results["prior_name"]) == {"weak", "medium", "strong"}


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_compare_priors_with_test_data(train_test_split):
    """Test prior comparison with test data evaluation."""
    train, test = train_test_split

    spec = linear_reg().set_engine("pymc")

    priors = {
        "weak": {"prior_coefs": "normal(0, 10)"},
        "strong": {"prior_coefs": "normal(0, 1)"}
    }

    results = compare_priors(
        model_spec=spec,
        data=train,
        formula="y ~ x1 + x2",
        priors=priors,
        test_data=test,
        metrics=metric_set(rmse, mae),
        draws=200,
        tune=200,
        chains=2,
        verbose=False
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 2

    # Should have test metrics
    assert "test_rmse" in results.columns
    assert "test_mae" in results.columns


# ============================================================================
# POSTERIOR SUMMARY TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_compare_priors_posterior_summaries(small_linear_data):
    """Test that posterior summaries are included in results."""
    spec = linear_reg().set_engine("pymc")

    priors = {
        "weak": {"prior_coefs": "normal(0, 10)"},
        "strong": {"prior_coefs": "normal(0, 1)"}
    }

    results = compare_priors(
        model_spec=spec,
        data=small_linear_data,
        formula="y ~ x1 + x2",
        priors=priors,
        draws=200,
        tune=200,
        chains=2,
        verbose=False
    )

    # Should have posterior means and SDs for each coefficient
    assert "posterior_mean_Intercept" in results.columns
    assert "posterior_sd_Intercept" in results.columns
    assert "posterior_mean_x1" in results.columns
    assert "posterior_sd_x1" in results.columns
    assert "posterior_mean_x2" in results.columns
    assert "posterior_sd_x2" in results.columns


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_compare_priors_different_posteriors(small_linear_data):
    """Test that different priors yield different posterior estimates."""
    spec = linear_reg().set_engine("pymc")

    priors = {
        "weak": {"prior_coefs": "normal(0, 10)"},
        "strong": {"prior_coefs": "normal(0, 0.1)"}  # Very strong prior at 0
    }

    results = compare_priors(
        model_spec=spec,
        data=small_linear_data,
        formula="y ~ x1 + x2",
        priors=priors,
        draws=200,
        tune=200,
        chains=2,
        verbose=False
    )

    # Strong prior should pull coefficients toward 0
    weak_x1 = results[results["prior_name"] == "weak"]["posterior_mean_x1"].iloc[0]
    strong_x1 = results[results["prior_name"] == "strong"]["posterior_mean_x1"].iloc[0]

    # Strong prior should be closer to 0
    assert abs(strong_x1) < abs(weak_x1)


# ============================================================================
# DIAGNOSTICS TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_compare_priors_diagnostics(small_linear_data):
    """Test that convergence diagnostics are included."""
    spec = linear_reg().set_engine("pymc")

    priors = {
        "weak": {"prior_coefs": "normal(0, 10)"},
        "strong": {"prior_coefs": "normal(0, 1)"}
    }

    results = compare_priors(
        model_spec=spec,
        data=small_linear_data,
        formula="y ~ x1 + x2",
        priors=priors,
        draws=300,
        tune=300,
        chains=2,
        verbose=False
    )

    # Should have diagnostics
    assert "max_rhat" in results.columns
    assert "min_ess_bulk" in results.columns
    assert "n_divergences" in results.columns

    # Check Rhat is reasonable
    for _, row in results.iterrows():
        assert 0.9 < row["max_rhat"] < 1.2


# ============================================================================
# TRAINING METRICS TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_compare_priors_training_metrics(small_linear_data):
    """Test that training metrics are included."""
    spec = linear_reg().set_engine("pymc")

    priors = {
        "weak": {"prior_coefs": "normal(0, 10)"},
        "strong": {"prior_coefs": "normal(0, 1)"}
    }

    results = compare_priors(
        model_spec=spec,
        data=small_linear_data,
        formula="y ~ x1 + x2",
        priors=priors,
        draws=200,
        tune=200,
        chains=2,
        verbose=False
    )

    # Should have training metrics from stats DataFrame
    assert "train_rmse" in results.columns
    assert "train_mae" in results.columns
    assert "train_r_squared" in results.columns


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_compare_priors_empty_priors():
    """Test error when no priors provided."""
    spec = linear_reg().set_engine("pymc")
    data = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3]})

    priors = {}  # Empty

    with pytest.raises(ValueError, match="failed to fit"):
        compare_priors(
            model_spec=spec,
            data=data,
            formula="y ~ x",
            priors=priors,
            draws=100,
            tune=100,
            chains=1,
            verbose=False
        )


# ============================================================================
# VERBOSE OUTPUT TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_compare_priors_verbose(small_linear_data, capsys):
    """Test verbose output."""
    spec = linear_reg().set_engine("pymc")

    priors = {
        "weak": {"prior_coefs": "normal(0, 10)"},
        "strong": {"prior_coefs": "normal(0, 1)"}
    }

    results = compare_priors(
        model_spec=spec,
        data=small_linear_data,
        formula="y ~ x1 + x2",
        priors=priors,
        draws=100,
        tune=100,
        chains=1,
        verbose=True  # Enable verbose
    )

    captured = capsys.readouterr()
    assert "Fitting with prior: weak" in captured.out
    assert "Fitting with prior: strong" in captured.out


# ============================================================================
# CUSTOM METRICS TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_compare_priors_custom_metrics(train_test_split):
    """Test prior comparison with custom metrics."""
    from py_yardstick import r_squared

    train, test = train_test_split

    spec = linear_reg().set_engine("pymc")

    priors = {
        "weak": {"prior_coefs": "normal(0, 10)"},
        "strong": {"prior_coefs": "normal(0, 1)"}
    }

    results = compare_priors(
        model_spec=spec,
        data=train,
        formula="y ~ x1 + x2",
        priors=priors,
        test_data=test,
        metrics=metric_set(rmse, mae, r_squared),
        draws=200,
        tune=200,
        chains=2,
        verbose=False
    )

    # Should have all three test metrics
    assert "test_rmse" in results.columns
    assert "test_mae" in results.columns
    assert "test_r_squared" in results.columns
