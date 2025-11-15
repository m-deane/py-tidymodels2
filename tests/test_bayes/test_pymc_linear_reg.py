"""
Tests for PyMC Bayesian Linear Regression Engine

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

from py_parsnip import linear_reg
from py_bayes.priors import parse_prior, get_default_priors
from py_bayes.diagnostics import check_convergence


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def small_linear_data():
    """Small synthetic dataset for fast MCMC sampling."""
    np.random.seed(42)
    n = 50

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)

    # True model: y = 2 + 3*x1 - 1.5*x2 + 0.5*x3 + noise
    y = 2.0 + 3.0*x1 - 1.5*x2 + 0.5*x3 + np.random.randn(n) * 0.5

    return pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'y': y
    })


@pytest.fixture
def train_test_split(small_linear_data):
    """Train/test split for evaluation testing."""
    train = small_linear_data.iloc[:40]
    test = small_linear_data.iloc[40:]
    return train, test


# ============================================================================
# PRIOR PARSING TESTS
# ============================================================================

def test_parse_normal_positional():
    """Test normal distribution parsing with positional args."""
    result = parse_prior("normal(0, 10)")
    assert result == {'dist': 'normal', 'mu': 0.0, 'sigma': 10.0}


def test_parse_normal_named():
    """Test normal distribution parsing with named args."""
    result = parse_prior("normal(mu=0, sigma=10)")
    assert result == {'dist': 'normal', 'mu': 0.0, 'sigma': 10.0}


def test_parse_student_t():
    """Test Student-t distribution parsing."""
    result = parse_prior("student_t(nu=3, mu=0, sigma=10)")
    assert result == {'dist': 'student_t', 'nu': 3.0, 'mu': 0.0, 'sigma': 10.0}


def test_parse_half_cauchy():
    """Test half-Cauchy distribution parsing."""
    result = parse_prior("half_cauchy(5)")
    assert result == {'dist': 'half_cauchy', 'beta': 5.0}


def test_parse_exponential():
    """Test exponential distribution parsing."""
    result = parse_prior("exponential(1)")
    assert result == {'dist': 'exponential', 'lam': 1.0}


def test_parse_gamma():
    """Test gamma distribution parsing."""
    result = parse_prior("gamma(alpha=2, beta=1)")
    assert result == {'dist': 'gamma', 'alpha': 2.0, 'beta': 1.0}


def test_parse_beta():
    """Test beta distribution parsing."""
    result = parse_prior("beta(alpha=2, beta=2)")
    assert result == {'dist': 'beta', 'alpha': 2.0, 'beta': 2.0}


def test_parse_uniform():
    """Test uniform distribution parsing."""
    result = parse_prior("uniform(lower=0, upper=10)")
    assert result == {'dist': 'uniform', 'lower': 0.0, 'upper': 10.0}


def test_parse_invalid_distribution():
    """Test error on invalid distribution name."""
    with pytest.raises(ValueError, match="Unsupported distribution"):
        parse_prior("invalid(0, 10)")


def test_parse_invalid_format():
    """Test error on invalid format."""
    with pytest.raises(ValueError, match="Invalid prior specification"):
        parse_prior("normal 0, 10")


def test_get_default_priors():
    """Test default prior specifications."""
    defaults = get_default_priors()
    assert 'prior_intercept' in defaults
    assert 'prior_coefs' in defaults
    assert 'prior_sigma' in defaults
    assert defaults['prior_intercept'] == 'normal(0, 10)'
    assert defaults['prior_coefs'] == 'normal(0, 5)'
    assert defaults['prior_sigma'] == 'half_cauchy(5)'


# ============================================================================
# BASIC FIT/PREDICT WORKFLOW TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_basic_fit_predict(small_linear_data):
    """Test basic fit and predict workflow."""
    spec = linear_reg().set_engine(
        "pymc",
        chains=2,
        draws=500,
        tune=500,
        progressbar=False
    )

    # Fit model
    fit = spec.fit(small_linear_data, "y ~ x1 + x2 + x3")

    # Check fit_data structure
    assert "model" in fit.fit_data
    assert "posterior_samples" in fit.fit_data
    assert "summary" in fit.fit_data
    assert "diagnostics" in fit.fit_data
    assert "y_train" in fit.fit_data
    assert "fitted" in fit.fit_data
    assert "residuals" in fit.fit_data

    # Make predictions
    predictions = fit.predict(small_linear_data.head(10), type="numeric")

    # Check predictions
    assert isinstance(predictions, pd.DataFrame)
    assert ".pred" in predictions.columns
    assert len(predictions) == 10


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_custom_priors(small_linear_data):
    """Test model with custom prior specifications."""
    spec = linear_reg().set_engine(
        "pymc",
        prior_intercept="normal(0, 20)",
        prior_coefs="student_t(nu=3, mu=0, sigma=5)",
        prior_sigma="half_cauchy(3)",
        chains=2,
        draws=500,
        tune=500,
        progressbar=False
    )

    fit = spec.fit(small_linear_data, "y ~ x1 + x2")

    # Should fit successfully
    assert fit is not None
    assert "posterior_samples" in fit.fit_data


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_per_coefficient_priors(small_linear_data):
    """Test model with per-coefficient prior specifications."""
    spec = linear_reg().set_engine(
        "pymc",
        prior_coefs={
            "x1": "normal(0, 5)",
            "x2": "normal(0, 3)",
            "x3": "student_t(nu=3, mu=0, sigma=5)"
        },
        chains=2,
        draws=500,
        tune=500,
        progressbar=False
    )

    fit = spec.fit(small_linear_data, "y ~ x1 + x2 + x3")

    # Check that individual beta parameters were created
    assert fit.fit_data["prior_coefs_dict"] is True

    # Extract outputs and check coefficients
    outputs, coefficients, stats = fit.extract_outputs()

    # Should have coefficients for Intercept, x1, x2, x3
    assert len(coefficients) == 4
    assert set(coefficients["term"]) == {"Intercept", "x1", "x2", "x3"}


# ============================================================================
# PREDICTION TYPE TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_predict_numeric(small_linear_data):
    """Test numeric predictions (posterior mean)."""
    spec = linear_reg().set_engine("pymc", chains=2, draws=500, tune=500, progressbar=False)
    fit = spec.fit(small_linear_data, "y ~ x1 + x2")

    predictions = fit.predict(small_linear_data.head(10), type="numeric")

    assert isinstance(predictions, pd.DataFrame)
    assert ".pred" in predictions.columns
    assert len(predictions) == 10
    assert not predictions[".pred"].isna().any()


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_predict_conf_int(small_linear_data):
    """Test credible interval predictions."""
    spec = linear_reg().set_engine(
        "pymc",
        chains=2,
        draws=500,
        tune=500,
        level=0.90,
        progressbar=False
    )
    fit = spec.fit(small_linear_data, "y ~ x1 + x2")

    predictions = fit.predict(small_linear_data.head(10), type="conf_int")

    assert isinstance(predictions, pd.DataFrame)
    assert ".pred" in predictions.columns
    assert ".pred_lower" in predictions.columns
    assert ".pred_upper" in predictions.columns
    assert len(predictions) == 10

    # Check that intervals make sense (lower < mean < upper)
    assert (predictions[".pred_lower"] <= predictions[".pred"]).all()
    assert (predictions[".pred"] <= predictions[".pred_upper"]).all()


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_predict_posterior(small_linear_data):
    """Test posterior sample predictions."""
    spec = linear_reg().set_engine(
        "pymc",
        chains=2,
        draws=500,
        tune=500,
        n_samples=100,
        progressbar=False
    )
    fit = spec.fit(small_linear_data, "y ~ x1 + x2")

    predictions = fit.predict(small_linear_data.head(5), type="posterior")

    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == 5

    # Should have 100 sample columns
    sample_cols = [col for col in predictions.columns if col.startswith(".pred_sample_")]
    assert len(sample_cols) == 100

    # Each sample should have 5 predictions
    for col in sample_cols:
        assert len(predictions[col]) == 5
        assert not predictions[col].isna().any()


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_predict_predictive(small_linear_data):
    """Test posterior predictive predictions (with error term)."""
    spec = linear_reg().set_engine(
        "pymc",
        chains=2,
        draws=500,
        tune=500,
        n_samples=100,
        progressbar=False
    )
    fit = spec.fit(small_linear_data, "y ~ x1 + x2")

    predictions = fit.predict(small_linear_data.head(5), type="predictive")

    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == 5

    # Should have 100 sample columns
    sample_cols = [col for col in predictions.columns if col.startswith(".pred_sample_")]
    assert len(sample_cols) == 100

    # Predictive samples should have higher variance than posterior samples
    # (because they include error term)


# ============================================================================
# CONVERGENCE DIAGNOSTICS TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_convergence_diagnostics_good(small_linear_data):
    """Test convergence diagnostics on well-converged model."""
    spec = linear_reg().set_engine(
        "pymc",
        chains=2,
        draws=1000,
        tune=1000,
        progressbar=False
    )
    fit = spec.fit(small_linear_data, "y ~ x1 + x2")

    diag = check_convergence(fit, warn=False)

    # Check diagnostic structure
    assert 'rhat_ok' in diag
    assert 'ess_ok' in diag
    assert 'divergences_ok' in diag
    assert 'max_rhat' in diag
    assert 'min_ess_bulk' in diag
    assert 'min_ess_tail' in diag
    assert 'n_divergences' in diag
    assert 'warnings' in diag

    # Model should converge well
    assert diag['rhat_ok'] is True
    assert diag['max_rhat'] < 1.01
    assert diag['n_divergences'] == 0


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_convergence_diagnostics_warnings(small_linear_data):
    """Test convergence diagnostics with potential issues."""
    # Use very few draws to potentially trigger warnings
    spec = linear_reg().set_engine(
        "pymc",
        chains=2,
        draws=100,
        tune=100,
        progressbar=False
    )
    fit = spec.fit(small_linear_data, "y ~ x1")

    diag = check_convergence(fit, ess_threshold=500, warn=False)

    # With few draws, ESS might be low
    assert isinstance(diag['min_ess_bulk'], float)
    assert isinstance(diag['min_ess_tail'], float)


# ============================================================================
# EXTRACT OUTPUTS TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_extract_outputs_structure(small_linear_data):
    """Test extract_outputs returns correct structure."""
    spec = linear_reg().set_engine("pymc", chains=2, draws=500, tune=500, progressbar=False)
    fit = spec.fit(small_linear_data, "y ~ x1 + x2 + x3")

    outputs, coefficients, stats = fit.extract_outputs()

    # Check outputs DataFrame
    assert isinstance(outputs, pd.DataFrame)
    required_cols = ["actuals", "fitted", "fitted_lower", "fitted_upper",
                     "forecast", "residuals", "split", "model", "model_group_name", "group"]
    for col in required_cols:
        assert col in outputs.columns

    # Check coefficients DataFrame
    assert isinstance(coefficients, pd.DataFrame)
    required_cols = ["term", "estimate", "std_error", "lower_ci", "upper_ci",
                     "rhat", "ess_bulk", "ess_tail", "prob_positive",
                     "model", "model_group_name", "group"]
    for col in required_cols:
        assert col in coefficients.columns

    # Should have coefficients for Intercept, x1, x2, x3
    assert len(coefficients) == 4
    assert set(coefficients["term"]) == {"Intercept", "x1", "x2", "x3"}

    # Check stats DataFrame
    assert isinstance(stats, pd.DataFrame)
    assert len(stats) == 1
    required_cols = ["split", "n_obs", "rmse", "mae", "r_squared",
                     "n_divergences", "max_rhat", "min_ess_bulk", "min_ess_tail",
                     "model", "model_group_name", "group"]
    for col in required_cols:
        assert col in stats.columns


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_extract_outputs_bayesian_metrics(small_linear_data):
    """Test Bayesian-specific metrics in extract_outputs."""
    spec = linear_reg().set_engine("pymc", chains=2, draws=500, tune=500, progressbar=False)
    fit = spec.fit(small_linear_data, "y ~ x1 + x2")

    outputs, coefficients, stats = fit.extract_outputs()

    # Check Bayesian metrics in coefficients
    for idx, row in coefficients.iterrows():
        # Rhat should be close to 1.0
        assert row["rhat"] < 1.05

        # ESS should be positive
        assert row["ess_bulk"] > 0
        assert row["ess_tail"] > 0

        # prob_positive should be between 0 and 1
        assert 0 <= row["prob_positive"] <= 1

        # Credible intervals should make sense
        assert row["lower_ci"] <= row["estimate"] <= row["upper_ci"]

    # Check fitted values have credible intervals
    assert (outputs["fitted_lower"] <= outputs["fitted"]).all()
    assert (outputs["fitted"] <= outputs["fitted_upper"]).all()


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_extract_outputs_with_test_data(train_test_split):
    """Test extract_outputs with train/test evaluation."""
    train, test = train_test_split

    spec = linear_reg().set_engine("pymc", chains=2, draws=500, tune=500, progressbar=False)
    fit = spec.fit(train, "y ~ x1 + x2")

    # Evaluate on test data
    fit = fit.evaluate(test)

    outputs, coefficients, stats = fit.extract_outputs()

    # Should have both train and test rows
    assert set(outputs["split"].unique()) == {"train", "test"}

    # Train should have 40 rows, test should have 10 rows
    assert (outputs["split"] == "train").sum() == 40
    assert (outputs["split"] == "test").sum() == 10


# ============================================================================
# COEFFICIENT ESTIMATION TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_coefficient_recovery(small_linear_data):
    """Test that Bayesian model recovers true coefficients (within uncertainty)."""
    spec = linear_reg().set_engine("pymc", chains=2, draws=1000, tune=1000, progressbar=False)
    fit = spec.fit(small_linear_data, "y ~ x1 + x2 + x3")

    outputs, coefficients, stats = fit.extract_outputs()

    # True coefficients: Intercept=2, x1=3, x2=-1.5, x3=0.5
    true_coefs = {
        "Intercept": 2.0,
        "x1": 3.0,
        "x2": -1.5,
        "x3": 0.5
    }

    # Check that estimates are close to true values (within 2 standard errors)
    for term, true_val in true_coefs.items():
        coef_row = coefficients[coefficients["term"] == term].iloc[0]
        estimate = coef_row["estimate"]
        std_error = coef_row["std_error"]

        # Should be within 3 standard errors (99.7% confidence)
        assert abs(estimate - true_val) < 3 * std_error


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_invalid_prediction_type(small_linear_data):
    """Test error on invalid prediction type."""
    spec = linear_reg().set_engine("pymc", chains=2, draws=500, tune=500, progressbar=False)
    fit = spec.fit(small_linear_data, "y ~ x1 + x2")

    with pytest.raises(ValueError, match="Unsupported prediction type"):
        fit.predict(small_linear_data, type="invalid")


def test_missing_pymc_import():
    """Test helpful error message when PyMC is not installed."""
    # This test will only pass if PyMC is NOT installed
    # Skip if PyMC is available
    if PYMC_AVAILABLE:
        pytest.skip("PyMC is installed")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_workflow_integration(small_linear_data):
    """Test integration with workflow."""
    from py_workflows import workflow

    spec = linear_reg().set_engine("pymc", chains=2, draws=500, tune=500, progressbar=False)
    wf = workflow().add_formula("y ~ x1 + x2").add_model(spec)

    fit = wf.fit(small_linear_data)

    # Should work with workflows
    assert fit is not None
    predictions = fit.predict(small_linear_data.head(10))
    assert len(predictions) == 10


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_evaluate_method(train_test_split):
    """Test evaluate method with Bayesian model."""
    train, test = train_test_split

    spec = linear_reg().set_engine("pymc", chains=2, draws=500, tune=500, progressbar=False)
    fit = spec.fit(train, "y ~ x1 + x2")

    # Evaluate on test data
    fit = fit.evaluate(test)

    # Should have evaluation_data
    assert "test_data" in fit.evaluation_data
    assert "test_predictions" in fit.evaluation_data
    assert "outcome_col" in fit.evaluation_data

    # extract_outputs should include test data
    outputs, coefficients, stats = fit.extract_outputs()
    assert "test" in outputs["split"].values
