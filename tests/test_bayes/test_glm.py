"""
Tests for Bayesian GLM (Poisson and Logistic) Engines

Uses small synthetic datasets for fast MCMC sampling:
- N = 50 observations
- chains = 2
- draws = 500
- tune = 500
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

from py_parsnip import poisson_bayes, logistic_bayes


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def small_poisson_data():
    """Small synthetic count data."""
    np.random.seed(42)
    n = 50

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)

    # True model: log(λ) = 1 + 0.5*x1 - 0.3*x2
    log_lambda = 1.0 + 0.5*x1 - 0.3*x2
    y = np.random.poisson(np.exp(log_lambda))

    return pd.DataFrame({'x1': x1, 'x2': x2, 'count': y})


@pytest.fixture
def small_logistic_data():
    """Small synthetic binary classification data."""
    np.random.seed(42)
    n = 50

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)

    # True model: logit(p) = 0.5 + 1.0*x1 - 0.8*x2
    logit_p = 0.5 + 1.0*x1 - 0.8*x2
    p = 1 / (1 + np.exp(-logit_p))
    y = np.random.binomial(1, p)

    return pd.DataFrame({'x1': x1, 'x2': x2, 'outcome': y})


@pytest.fixture
def train_test_split_poisson(small_poisson_data):
    """Train/test split for Poisson data."""
    train = small_poisson_data.iloc[:40]
    test = small_poisson_data.iloc[40:]
    return train, test


@pytest.fixture
def train_test_split_logistic(small_logistic_data):
    """Train/test split for logistic data."""
    train = small_logistic_data.iloc[:40]
    test = small_logistic_data.iloc[40:]
    return train, test


# ============================================================================
# POISSON REGRESSION TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_poisson_basic_fit(small_poisson_data):
    """Test basic Poisson regression fitting."""
    spec = poisson_bayes(draws=500, tune=500, chains=2, progressbar=False, random_seed=42)
    fit = spec.fit(small_poisson_data, "count ~ x1 + x2")

    assert fit is not None
    assert hasattr(fit, "fit_data")
    assert "posterior_samples" in fit.fit_data


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_poisson_predict_numeric(train_test_split_poisson):
    """Test numeric predictions (expected counts)."""
    train, test = train_test_split_poisson

    spec = poisson_bayes(draws=500, tune=500, chains=2, progressbar=False, random_seed=42)
    fit = spec.fit(train, "count ~ x1 + x2")

    predictions = fit.predict(test, type="numeric")

    assert isinstance(predictions, pd.DataFrame)
    assert ".pred" in predictions.columns
    assert len(predictions) == len(test)
    assert all(predictions[".pred"] >= 0)  # Counts are non-negative


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_poisson_predict_conf_int(train_test_split_poisson):
    """Test credible interval predictions."""
    train, test = train_test_split_poisson

    spec = poisson_bayes(draws=500, tune=500, chains=2, progressbar=False, random_seed=42)
    fit = spec.fit(train, "count ~ x1 + x2")

    predictions = fit.predict(test, type="conf_int")

    assert isinstance(predictions, pd.DataFrame)
    assert ".pred" in predictions.columns
    assert ".pred_lower" in predictions.columns
    assert ".pred_upper" in predictions.columns
    assert all(predictions[".pred_lower"] <= predictions[".pred"])
    assert all(predictions[".pred"] <= predictions[".pred_upper"])


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_poisson_predict_predictive(train_test_split_poisson):
    """Test posterior predictive samples (with Poisson noise)."""
    train, test = train_test_split_poisson

    spec = poisson_bayes(draws=500, tune=500, chains=2, progressbar=False, random_seed=42)
    fit = spec.fit(train, "count ~ x1 + x2")

    predictions = fit.predict(test, type="predictive", n_samples=100)

    assert isinstance(predictions, pd.DataFrame)
    assert ".pred_sample_1" in predictions.columns
    assert ".pred_sample_100" in predictions.columns
    assert len(predictions) == len(test)

    # All samples should be non-negative integers
    for col in predictions.columns:
        assert all(predictions[col] >= 0)


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_poisson_extract_outputs(small_poisson_data):
    """Test extract_outputs for Poisson model."""
    spec = poisson_bayes(draws=500, tune=500, chains=2, progressbar=False, random_seed=42)
    fit = spec.fit(small_poisson_data, "count ~ x1 + x2")

    outputs, coefficients, stats = fit.extract_outputs()

    # Outputs
    assert isinstance(outputs, pd.DataFrame)
    assert "actuals" in outputs.columns
    assert "fitted" in outputs.columns
    assert all(outputs["fitted"] >= 0)  # Expected counts are non-negative

    # Coefficients
    assert isinstance(coefficients, pd.DataFrame)
    terms = set(coefficients["term"])
    assert "Intercept" in terms
    assert "x1" in terms
    assert "x2" in terms

    # Stats
    assert isinstance(stats, pd.DataFrame)
    assert "rmse" in stats.columns
    assert "mae" in stats.columns


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_poisson_custom_priors(small_poisson_data):
    """Test Poisson with custom priors."""
    spec = poisson_bayes(
        prior_intercept="normal(0, 5)",
        prior_coefs="student_t(nu=3, mu=0, sigma=2)",
        draws=500,
        tune=500,
        chains=2,
        progressbar=False,
        random_seed=42
    )
    fit = spec.fit(small_poisson_data, "count ~ x1 + x2")

    assert fit is not None


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_poisson_non_negative_validation():
    """Test that Poisson rejects negative counts."""
    spec = poisson_bayes(draws=100, tune=100, chains=1, progressbar=False)

    # Create data with negative count
    data = pd.DataFrame({'x': [1, 2, 3], 'y': [1, -1, 3]})

    with pytest.raises(ValueError, match="non-negative"):
        spec.fit(data, "y ~ x")


# ============================================================================
# LOGISTIC REGRESSION TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_logistic_basic_fit(small_logistic_data):
    """Test basic logistic regression fitting."""
    spec = logistic_bayes(draws=500, tune=500, chains=2, progressbar=False, random_seed=42)
    fit = spec.fit(small_logistic_data, "outcome ~ x1 + x2")

    assert fit is not None
    assert hasattr(fit, "fit_data")
    assert "posterior_samples" in fit.fit_data


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_logistic_predict_prob(train_test_split_logistic):
    """Test probability predictions."""
    train, test = train_test_split_logistic

    spec = logistic_bayes(draws=500, tune=500, chains=2, progressbar=False, random_seed=42)
    fit = spec.fit(train, "outcome ~ x1 + x2")

    predictions = fit.predict(test, type="prob")

    assert isinstance(predictions, pd.DataFrame)
    assert ".pred_prob" in predictions.columns
    assert len(predictions) == len(test)
    assert all(predictions[".pred_prob"] >= 0)
    assert all(predictions[".pred_prob"] <= 1)


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_logistic_predict_class(train_test_split_logistic):
    """Test class predictions (0 or 1)."""
    train, test = train_test_split_logistic

    spec = logistic_bayes(draws=500, tune=500, chains=2, progressbar=False, random_seed=42)
    fit = spec.fit(train, "outcome ~ x1 + x2")

    predictions = fit.predict(test, type="class")

    assert isinstance(predictions, pd.DataFrame)
    assert ".pred_class" in predictions.columns
    assert len(predictions) == len(test)
    assert set(predictions[".pred_class"].unique()).issubset({0, 1})


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_logistic_predict_conf_int(train_test_split_logistic):
    """Test credible intervals for probabilities."""
    train, test = train_test_split_logistic

    spec = logistic_bayes(draws=500, tune=500, chains=2, progressbar=False, random_seed=42)
    fit = spec.fit(train, "outcome ~ x1 + x2")

    predictions = fit.predict(test, type="conf_int")

    assert isinstance(predictions, pd.DataFrame)
    assert ".pred_prob" in predictions.columns
    assert ".pred_prob_lower" in predictions.columns
    assert ".pred_prob_upper" in predictions.columns
    assert all(predictions[".pred_prob_lower"] <= predictions[".pred_prob"])
    assert all(predictions[".pred_prob"] <= predictions[".pred_prob_upper"])
    assert all(predictions[".pred_prob_lower"] >= 0)
    assert all(predictions[".pred_prob_upper"] <= 1)


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_logistic_extract_outputs(small_logistic_data):
    """Test extract_outputs for logistic model."""
    spec = logistic_bayes(draws=500, tune=500, chains=2, progressbar=False, random_seed=42)
    fit = spec.fit(small_logistic_data, "outcome ~ x1 + x2")

    outputs, coefficients, stats = fit.extract_outputs()

    # Outputs
    assert isinstance(outputs, pd.DataFrame)
    assert "actuals" in outputs.columns
    assert "fitted" in outputs.columns
    assert "fitted_class" in outputs.columns
    assert all(outputs["fitted"] >= 0) and all(outputs["fitted"] <= 1)  # Probabilities
    assert set(outputs["fitted_class"].unique()).issubset({0, 1})

    # Coefficients
    assert isinstance(coefficients, pd.DataFrame)
    terms = set(coefficients["term"])
    assert "Intercept" in terms
    assert "x1" in terms
    assert "x2" in terms

    # Stats
    assert isinstance(stats, pd.DataFrame)
    assert "accuracy" in stats.columns


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_logistic_custom_priors(small_logistic_data):
    """Test logistic with custom priors."""
    spec = logistic_bayes(
        prior_intercept="normal(0, 5)",
        prior_coefs="student_t(nu=3, mu=0, sigma=2)",
        draws=500,
        tune=500,
        chains=2,
        progressbar=False,
        random_seed=42
    )
    fit = spec.fit(small_logistic_data, "outcome ~ x1 + x2")

    assert fit is not None


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_logistic_binary_validation():
    """Test that logistic rejects non-binary outcomes."""
    spec = logistic_bayes(draws=100, tune=100, chains=1, progressbar=False)

    # Create data with non-binary outcome
    data = pd.DataFrame({'x': [1, 2, 3], 'y': [0, 1, 2]})

    with pytest.raises(ValueError, match="binary outcome"):
        spec.fit(data, "y ~ x")


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_logistic_per_coefficient_priors(small_logistic_data):
    """Test logistic with different priors per coefficient."""
    spec = logistic_bayes(
        prior_coefs={
            "x1": "normal(0, 10)",
            "x2": "normal(0, 1)"
        },
        draws=500,
        tune=500,
        chains=2,
        progressbar=False,
        random_seed=42
    )
    fit = spec.fit(small_logistic_data, "outcome ~ x1 + x2")

    assert fit is not None
    assert fit.fit_data["prior_coefs_dict"] == True


# ============================================================================
# EVALUATION TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_poisson_evaluate(train_test_split_poisson):
    """Test evaluate() for Poisson model."""
    train, test = train_test_split_poisson

    spec = poisson_bayes(draws=500, tune=500, chains=2, progressbar=False, random_seed=42)
    fit = spec.fit(train, "count ~ x1 + x2")

    fit = fit.evaluate(test)

    outputs, _, _ = fit.extract_outputs()
    test_outputs = outputs[outputs["split"] == "test"]
    assert len(test_outputs) > 0


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_logistic_evaluate(train_test_split_logistic):
    """Test evaluate() for logistic model."""
    train, test = train_test_split_logistic

    spec = logistic_bayes(draws=500, tune=500, chains=2, progressbar=False, random_seed=42)
    fit = spec.fit(train, "outcome ~ x1 + x2")

    fit = fit.evaluate(test)

    outputs, _, _ = fit.extract_outputs()
    test_outputs = outputs[outputs["split"] == "test"]
    assert len(test_outputs) > 0
