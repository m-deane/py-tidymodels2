"""
Tests for Hierarchical Bayesian Linear Regression Engine

Uses small synthetic datasets for fast MCMC sampling:
- N = 60 observations (20 per group)
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
from py_workflows import workflow


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def small_hierarchical_data():
    """Small synthetic dataset with 3 groups."""
    np.random.seed(42)
    n_per_group = 20
    groups = ["A", "B", "C"]

    data_list = []
    for i, group in enumerate(groups):
        x1 = np.random.randn(n_per_group)
        x2 = np.random.randn(n_per_group)

        # Group-specific intercepts and slopes
        alpha_group = [2.0, 0.0, -1.5][i]
        beta1_group = [3.0, 2.0, 1.0][i]  # Varying slope for x1
        beta2 = -1.0  # Fixed slope for x2

        y = alpha_group + beta1_group*x1 + beta2*x2 + np.random.randn(n_per_group) * 0.5

        df_group = pd.DataFrame({
            'group': group,
            'x1': x1,
            'x2': x2,
            'y': y
        })
        data_list.append(df_group)

    return pd.concat(data_list, ignore_index=True)


@pytest.fixture
def train_test_split_hierarchical(small_hierarchical_data):
    """Train/test split preserving groups."""
    # Take first 15 obs per group for training, last 5 for testing
    train_list = []
    test_list = []

    for group in ["A", "B", "C"]:
        group_data = small_hierarchical_data[small_hierarchical_data["group"] == group]
        train_list.append(group_data.iloc[:15])
        test_list.append(group_data.iloc[15:])

    train = pd.concat(train_list, ignore_index=True)
    test = pd.concat(test_list, ignore_index=True)

    return train, test


# ============================================================================
# BASIC FITTING TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_hierarchical_group_varying_intercept_only(small_hierarchical_data):
    """Test hierarchical model with group-varying intercepts only."""
    spec = linear_reg().set_engine(
        "pymc_hierarchical",
        group_varying_intercept=True,
        group_varying_slopes=[],  # No varying slopes
        draws=500,
        tune=500,
        chains=2,
        progressbar=False,
        random_seed=42
    )

    wf = workflow().add_formula("y ~ x1 + x2").add_model(spec)
    fit = wf.fit_global(small_hierarchical_data, group_col="group")

    assert fit is not None
    assert hasattr(fit, "fit")
    assert fit.fit.fit_data["n_groups"] == 3


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_hierarchical_group_varying_intercept_and_slopes(small_hierarchical_data):
    """Test hierarchical model with group-varying intercepts and slopes."""
    spec = linear_reg().set_engine(
        "pymc_hierarchical",
        group_varying_intercept=True,
        group_varying_slopes=["x1"],  # Varying slope for x1
        draws=500,
        tune=500,
        chains=2,
        progressbar=False,
        random_seed=42
    )

    wf = workflow().add_formula("y ~ x1 + x2").add_model(spec)
    fit = wf.fit_global(small_hierarchical_data, group_col="group")

    assert fit is not None
    assert fit.fit.fit_data["group_varying_slopes"] == ["x1"]


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_hierarchical_fixed_intercept_varying_slopes(small_hierarchical_data):
    """Test hierarchical model with fixed intercept but varying slopes."""
    spec = linear_reg().set_engine(
        "pymc_hierarchical",
        group_varying_intercept=False,
        group_varying_slopes=["x1", "x2"],  # Both slopes vary
        draws=500,
        tune=500,
        chains=2,
        progressbar=False,
        random_seed=42
    )

    wf = workflow().add_formula("y ~ x1 + x2").add_model(spec)
    fit = wf.fit_global(small_hierarchical_data, group_col="group")

    assert fit is not None
    assert fit.fit.fit_data["group_varying_intercept"] == False
    assert set(fit.fit.fit_data["group_varying_slopes"]) == {"x1", "x2"}


# ============================================================================
# PREDICTION TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_hierarchical_predict_numeric(train_test_split_hierarchical):
    """Test numeric predictions from hierarchical model."""
    train, test = train_test_split_hierarchical

    spec = linear_reg().set_engine(
        "pymc_hierarchical",
        group_varying_intercept=True,
        group_varying_slopes=["x1"],
        draws=500,
        tune=500,
        chains=2,
        progressbar=False,
        random_seed=42
    )

    wf = workflow().add_formula("y ~ x1 + x2").add_model(spec)
    fit = wf.fit_global(train, group_col="group")

    # Predict on test data
    predictions = fit.predict(test)

    assert isinstance(predictions, pd.DataFrame)
    assert ".pred" in predictions.columns
    assert len(predictions) == len(test)


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_hierarchical_predict_conf_int(train_test_split_hierarchical):
    """Test credible interval predictions from hierarchical model."""
    train, test = train_test_split_hierarchical

    spec = linear_reg().set_engine(
        "pymc_hierarchical",
        group_varying_intercept=True,
        draws=500,
        tune=500,
        chains=2,
        progressbar=False,
        random_seed=42
    )

    wf = workflow().add_formula("y ~ x1 + x2").add_model(spec)
    fit = wf.fit_global(train, group_col="group")

    # Predict with confidence intervals
    predictions = fit.predict(test, type="conf_int")

    assert isinstance(predictions, pd.DataFrame)
    assert ".pred" in predictions.columns
    assert ".pred_lower" in predictions.columns
    assert ".pred_upper" in predictions.columns
    assert len(predictions) == len(test)

    # Check intervals are ordered correctly
    assert all(predictions[".pred_lower"] <= predictions[".pred"])
    assert all(predictions[".pred"] <= predictions[".pred_upper"])


# ============================================================================
# EXTRACT OUTPUTS TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_hierarchical_extract_outputs_structure(small_hierarchical_data):
    """Test extract_outputs returns correct structure."""
    spec = linear_reg().set_engine(
        "pymc_hierarchical",
        group_varying_intercept=True,
        group_varying_slopes=["x1"],
        draws=500,
        tune=500,
        chains=2,
        progressbar=False,
        random_seed=42
    )

    wf = workflow().add_formula("y ~ x1 + x2").add_model(spec)
    fit = wf.fit_global(small_hierarchical_data, group_col="group")

    outputs, coefficients, stats = fit.extract_outputs()

    # Check outputs DataFrame
    assert isinstance(outputs, pd.DataFrame)
    assert "actuals" in outputs.columns
    assert "fitted" in outputs.columns
    assert "group" in outputs.columns
    assert set(outputs["group"]) == {"A", "B", "C"}

    # Check coefficients DataFrame
    assert isinstance(coefficients, pd.DataFrame)
    assert "term" in coefficients.columns
    assert "estimate" in coefficients.columns
    assert "std_error" in coefficients.columns

    # Should have hyperparameters and group-specific parameters
    terms = set(coefficients["term"])
    assert "mu_alpha" in terms  # Global intercept mean
    assert "sigma_alpha" in terms  # Global intercept SD
    assert "mu_beta_x1" in terms  # Global x1 slope mean
    assert "sigma_beta_x1" in terms  # Global x1 slope SD

    # Check stats DataFrame
    assert isinstance(stats, pd.DataFrame)
    assert "n_groups" in stats.columns
    assert stats.iloc[0]["n_groups"] == 3


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_hierarchical_group_specific_coefficients(small_hierarchical_data):
    """Test that group-specific coefficients are extracted."""
    spec = linear_reg().set_engine(
        "pymc_hierarchical",
        group_varying_intercept=True,
        group_varying_slopes=["x1"],
        draws=500,
        tune=500,
        chains=2,
        progressbar=False,
        random_seed=42
    )

    wf = workflow().add_formula("y ~ x1 + x2").add_model(spec)
    fit = wf.fit_global(small_hierarchical_data, group_col="group")

    _, coefficients, _ = fit.extract_outputs()

    # Check group-specific parameters
    group_A_intercepts = coefficients[
        (coefficients["term"] == "Intercept") & (coefficients["group"] == "A")
    ]
    assert len(group_A_intercepts) == 1

    group_A_x1_slopes = coefficients[
        (coefficients["term"] == "x1") & (coefficients["group"] == "A")
    ]
    assert len(group_A_x1_slopes) == 1

    # Check fixed coefficient (shared across groups)
    x2_coefs = coefficients[
        (coefficients["term"] == "x2") & (coefficients["group"].isna())
    ]
    assert len(x2_coefs) == 1  # Only one, shared across all groups


# ============================================================================
# SHRINKAGE TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_hierarchical_shrinkage_toward_global_mean(small_hierarchical_data):
    """Test that group estimates shrink toward global mean."""
    spec = linear_reg().set_engine(
        "pymc_hierarchical",
        group_varying_intercept=True,
        draws=1000,
        tune=500,
        chains=2,
        progressbar=False,
        random_seed=42
    )

    wf = workflow().add_formula("y ~ x1 + x2").add_model(spec)
    fit = wf.fit_global(small_hierarchical_data, group_col="group")

    _, coefficients, _ = fit.extract_outputs()

    # Get global mean
    global_mean_row = coefficients[coefficients["term"] == "mu_alpha"]
    assert len(global_mean_row) == 1
    global_mean = global_mean_row.iloc[0]["estimate"]

    # Get group-specific intercepts
    group_intercepts = coefficients[
        (coefficients["term"] == "Intercept") & (coefficients["group"].notna())
    ]

    # All group intercepts should be reasonably close to global mean
    # (This is the shrinkage effect)
    for _, row in group_intercepts.iterrows():
        group_estimate = row["estimate"]
        # Deviation from global mean should be moderate (not extreme)
        assert abs(group_estimate - global_mean) < 5.0


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_hierarchical_requires_group_col():
    """Test that hierarchical engine requires group_col argument."""
    spec = linear_reg().set_engine(
        "pymc_hierarchical",
        # Missing group_col argument
        group_varying_intercept=True,
        draws=100,
        tune=100,
        chains=1,
        progressbar=False
    )

    data = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [1, 2, 3]
    })

    with pytest.raises(ValueError, match="require.*group_col"):
        spec.fit(data, "y ~ x")


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_hierarchical_invalid_varying_slope_name(small_hierarchical_data):
    """Test error when varying slope name doesn't exist in predictors."""
    spec = linear_reg().set_engine(
        "pymc_hierarchical",
        group_varying_intercept=True,
        group_varying_slopes=["nonexistent_var"],  # Invalid name
        draws=100,
        tune=100,
        chains=1,
        progressbar=False,
        random_seed=42
    )

    wf = workflow().add_formula("y ~ x1 + x2").add_model(spec)

    with pytest.raises(ValueError, match="not in predictors"):
        wf.fit_global(small_hierarchical_data, group_col="group")


# ============================================================================
# EVALUATION TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_hierarchical_evaluate(train_test_split_hierarchical):
    """Test evaluate() method with hierarchical models."""
    train, test = train_test_split_hierarchical

    spec = linear_reg().set_engine(
        "pymc_hierarchical",
        group_varying_intercept=True,
        draws=500,
        tune=500,
        chains=2,
        progressbar=False,
        random_seed=42
    )

    wf = workflow().add_formula("y ~ x1 + x2").add_model(spec)
    fit = wf.fit_global(train, group_col="group")

    # Evaluate on test data
    fit = fit.evaluate(test)

    outputs, _, stats = fit.extract_outputs()

    # Check test data is included
    test_outputs = outputs[outputs["split"] == "test"]
    assert len(test_outputs) > 0
    assert set(test_outputs["group"]) == {"A", "B", "C"}


# ============================================================================
# CUSTOM PRIOR TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_hierarchical_custom_priors(small_hierarchical_data):
    """Test hierarchical model with custom prior specifications."""
    spec = linear_reg().set_engine(
        "pymc_hierarchical",
        group_varying_intercept=True,
        prior_group_intercept_mean="normal(0, 5)",
        prior_group_intercept_sd="half_cauchy(3)",
        prior_group_slope_mean="normal(0, 2)",
        prior_group_slope_sd="half_cauchy(2)",
        prior_sigma="half_cauchy(3)",
        draws=500,
        tune=500,
        chains=2,
        progressbar=False,
        random_seed=42
    )

    wf = workflow().add_formula("y ~ x1 + x2").add_model(spec)
    fit = wf.fit_global(small_hierarchical_data, group_col="group")

    assert fit is not None
    _, coefficients, _ = fit.extract_outputs()

    # Check that hyperparameters exist
    assert len(coefficients[coefficients["term"] == "mu_alpha"]) == 1
    assert len(coefficients[coefficients["term"] == "sigma_alpha"]) == 1


# ============================================================================
# DIAGNOSTICS TESTS
# ============================================================================

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
def test_hierarchical_convergence_diagnostics(small_hierarchical_data):
    """Test that convergence diagnostics are computed."""
    spec = linear_reg().set_engine(
        "pymc_hierarchical",
        group_varying_intercept=True,
        draws=500,
        tune=500,
        chains=2,
        progressbar=False,
        random_seed=42
    )

    wf = workflow().add_formula("y ~ x1 + x2").add_model(spec)
    fit = wf.fit_global(small_hierarchical_data, group_col="group")

    _, _, stats = fit.extract_outputs()

    assert "max_rhat" in stats.columns
    assert "min_ess_bulk" in stats.columns
    assert "min_ess_tail" in stats.columns

    # Rhat should be close to 1.0 (good convergence)
    max_rhat = stats.iloc[0]["max_rhat"]
    assert 0.9 < max_rhat < 1.1
