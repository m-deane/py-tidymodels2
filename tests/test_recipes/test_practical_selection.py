"""
Tests for Phase 4 practical selection steps.

This module tests the 8 practical selection methods:
1. StepSelectLasso - L1 regularization
2. StepSelectRidge - L2 regularization
3. StepSelectElasticNet - L1+L2 regularization
4. StepSelectUnivariate - Statistical tests
5. StepSelectVarianceThreshold - Low-variance removal
6. StepSelectStationary - ADF test
7. StepSelectCointegration - Engle-Granger test
8. StepSelectSeasonal - FFT-based detection
"""

import pytest
import pandas as pd
import numpy as np
from py_recipes import recipe


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def sample_data():
    """Create sample regression dataset."""
    np.random.seed(42)
    n = 100

    data = pd.DataFrame({
        'target': np.random.randn(n) * 10 + 50,
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'x3': np.random.randn(n),
        'x4': np.random.randn(n),
        'x5': np.random.randn(n),
    })

    # Add linear relationships
    data['target'] = 10 * data['x1'] + 5 * data['x2'] + np.random.randn(n) * 2

    return data


@pytest.fixture
def classification_data():
    """Create sample classification dataset."""
    np.random.seed(42)
    n = 100

    data = pd.DataFrame({
        'target': np.random.choice([0, 1], size=n),
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'x3': np.random.randn(n),
        'x4': np.random.randn(n),
    })

    return data


@pytest.fixture
def time_series_data():
    """Create time series data with trend and seasonality."""
    np.random.seed(42)
    n = 200
    t = np.arange(n)

    # Trend component
    trend = 0.5 * t

    # Seasonal component (period=12)
    seasonal = 10 * np.sin(2 * np.pi * t / 12)

    # Random walk (non-stationary)
    random_walk = np.cumsum(np.random.randn(n))

    # Stationary noise
    stationary = np.random.randn(n)

    data = pd.DataFrame({
        'target': trend + seasonal + np.random.randn(n) * 2,
        'seasonal': seasonal + np.random.randn(n) * 0.5,
        'random_walk': random_walk,
        'stationary': stationary,
        'trend': trend + np.random.randn(n) * 0.5,
    })

    return data


@pytest.fixture
def low_variance_data():
    """Create data with low and high variance features."""
    np.random.seed(42)
    n = 100

    data = pd.DataFrame({
        'target': np.random.randn(n),
        'high_var': np.random.randn(n) * 10,
        'low_var': np.random.randn(n) * 0.01,
        'constant': np.ones(n),
        'near_constant': np.concatenate([np.ones(99), [2.0]]),
    })

    return data


# ============================================================
# Test StepSelectLasso
# ============================================================

class TestStepSelectLasso:
    """Test Lasso (L1) feature selection."""

    def test_basic_selection(self, sample_data):
        """Test basic Lasso feature selection."""
        rec = recipe().step_select_lasso(
            outcome='target',
            alpha=0.1,
            columns=None
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        # Should select x1 and x2 (strong predictors)
        assert 'target' in result.columns
        assert 'x1' in result.columns or 'x2' in result.columns
        assert len(result.columns) <= len(sample_data.columns)
        assert len(result) == len(sample_data)

    def test_threshold_selection(self, sample_data):
        """Test Lasso with coefficient threshold."""
        rec = recipe().step_select_lasso(
            outcome='target',
            alpha=0.1,
            threshold=0.5
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        # Only features with |coef| > 0.5
        assert 'target' in result.columns
        assert len(result.columns) >= 1  # At least target

    def test_top_n_selection(self, sample_data):
        """Test Lasso with top_n selection."""
        rec = recipe().step_select_lasso(
            outcome='target',
            alpha=0.1,
            top_n=2
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        # Should have target + 2 features
        assert len(result.columns) == 3
        assert 'target' in result.columns

    def test_skip_functionality(self, sample_data):
        """Test skip parameter."""
        rec = recipe().step_select_lasso(
            outcome='target',
            alpha=0.1,
            skip=True
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        # With skip=True, bake should keep all columns
        assert len(result.columns) == len(sample_data.columns)


# ============================================================
# Test StepSelectRidge
# ============================================================

class TestStepSelectRidge:
    """Test Ridge (L2) feature selection."""

    def test_basic_selection(self, sample_data):
        """Test basic Ridge feature selection."""
        rec = recipe().step_select_ridge(
            outcome='target',
            alpha=1.0,
            threshold=1.0
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        assert 'target' in result.columns
        assert len(result.columns) <= len(sample_data.columns)

    def test_top_n_selection(self, sample_data):
        """Test Ridge with top_n."""
        rec = recipe().step_select_ridge(
            outcome='target',
            alpha=1.0,
            top_n=3
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        assert len(result.columns) == 4  # target + 3 features
        assert 'target' in result.columns

    def test_high_alpha(self, sample_data):
        """Test Ridge with high regularization."""
        rec = recipe().step_select_ridge(
            outcome='target',
            alpha=100.0,
            top_n=2
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        # Even with high alpha, should select top_n
        assert len(result.columns) == 3  # target + 2

    def test_column_specification(self, sample_data):
        """Test with specific columns."""
        rec = recipe().step_select_ridge(
            outcome='target',
            alpha=1.0,
            top_n=1,
            columns=['x1', 'x2', 'x3']
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        # Should select from specified columns only
        assert 'target' in result.columns
        assert len(result.columns) == 2  # target + 1 from x1/x2/x3


# ============================================================
# Test StepSelectElasticNet
# ============================================================

class TestStepSelectElasticNet:
    """Test Elastic Net (L1+L2) feature selection."""

    def test_basic_selection(self, sample_data):
        """Test basic Elastic Net selection."""
        rec = recipe().step_select_elastic_net(
            outcome='target',
            alpha=0.5,
            l1_ratio=0.5
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        assert 'target' in result.columns
        assert len(result.columns) <= len(sample_data.columns)

    def test_pure_lasso(self, sample_data):
        """Test Elastic Net with l1_ratio=1.0 (pure Lasso)."""
        rec = recipe().step_select_elastic_net(
            outcome='target',
            alpha=0.1,
            l1_ratio=1.0,
            top_n=2
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        assert len(result.columns) == 3  # target + 2

    def test_pure_ridge(self, sample_data):
        """Test Elastic Net with l1_ratio=0.0 (pure Ridge)."""
        rec = recipe().step_select_elastic_net(
            outcome='target',
            alpha=1.0,
            l1_ratio=0.0,
            top_n=2
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        assert len(result.columns) == 3  # target + 2

    def test_mixed_penalty(self, sample_data):
        """Test Elastic Net with balanced L1/L2."""
        rec = recipe().step_select_elastic_net(
            outcome='target',
            alpha=0.5,
            l1_ratio=0.5,
            threshold=0.5
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        assert 'target' in result.columns


# ============================================================
# Test StepSelectUnivariate
# ============================================================

class TestStepSelectUnivariate:
    """Test univariate statistical feature selection."""

    def test_f_regression(self, sample_data):
        """Test F-statistic for regression."""
        rec = recipe().step_select_univariate(
            outcome='target',
            score_func='f_regression',
            top_n=3
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        assert len(result.columns) == 4  # target + 3
        assert 'target' in result.columns

    def test_mutual_info(self, sample_data):
        """Test mutual information for regression."""
        rec = recipe().step_select_univariate(
            outcome='target',
            score_func='mutual_info_regression',
            top_n=2
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        assert len(result.columns) == 3  # target + 2

    def test_chi2_classification(self, classification_data):
        """Test chi-squared for classification."""
        rec = recipe().step_select_univariate(
            outcome='target',
            score_func='chi2',
            top_n=2
        )

        prepped = rec.prep(classification_data)
        result = prepped.bake(classification_data)

        assert len(result.columns) == 3  # target + 2

    def test_threshold_selection(self, sample_data):
        """Test p-value threshold selection."""
        rec = recipe().step_select_univariate(
            outcome='target',
            score_func='f_regression',
            threshold=0.05
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        assert 'target' in result.columns


# ============================================================
# Test StepSelectVarianceThreshold
# ============================================================

class TestStepSelectVarianceThreshold:
    """Test variance-based feature selection."""

    def test_basic_variance_filtering(self, low_variance_data):
        """Test removing low-variance features."""
        rec = recipe().step_select_variance_threshold(
            threshold=0.1
        )

        prepped = rec.prep(low_variance_data)
        result = prepped.bake(low_variance_data)

        # Should remove constant and near_constant
        assert 'constant' not in result.columns
        assert 'target' in result.columns
        assert 'high_var' in result.columns

    def test_zero_variance(self, low_variance_data):
        """Test removing zero-variance features."""
        rec = recipe().step_select_variance_threshold(
            threshold=0.0
        )

        prepped = rec.prep(low_variance_data)
        result = prepped.bake(low_variance_data)

        # Should remove only constant
        assert 'constant' not in result.columns
        assert 'target' in result.columns

    def test_column_specification(self, low_variance_data):
        """Test with specific columns."""
        rec = recipe().step_select_variance_threshold(
            threshold=0.1,
            columns=['low_var', 'high_var', 'constant']
        )

        prepped = rec.prep(low_variance_data)
        result = prepped.bake(low_variance_data)

        # Should only filter specified columns
        assert 'target' in result.columns
        assert 'near_constant' in result.columns  # Not in columns list

    def test_skip_functionality(self, low_variance_data):
        """Test skip parameter."""
        rec = recipe().step_select_variance_threshold(
            threshold=0.1,
            skip=True
        )

        prepped = rec.prep(low_variance_data)
        result = prepped.bake(low_variance_data)

        # With skip=True, keep all columns
        assert len(result.columns) == len(low_variance_data.columns)


# ============================================================
# Test StepSelectStationary
# ============================================================

class TestStepSelectStationary:
    """Test stationarity-based feature selection."""

    def test_basic_stationarity(self, time_series_data):
        """Test selecting stationary series."""
        rec = recipe().step_select_stationary(
            alpha=0.05
        )

        prepped = rec.prep(time_series_data)
        result = prepped.bake(time_series_data)

        # Stationary should pass, random_walk should fail
        assert 'stationary' in result.columns

    def test_strict_threshold(self, time_series_data):
        """Test with strict alpha threshold."""
        rec = recipe().step_select_stationary(
            alpha=0.01
        )

        prepped = rec.prep(time_series_data)
        result = prepped.bake(time_series_data)

        # Should have at least some columns
        assert len(result.columns) >= 1

    def test_lenient_threshold(self, time_series_data):
        """Test with lenient alpha threshold."""
        rec = recipe().step_select_stationary(
            alpha=0.10
        )

        prepped = rec.prep(time_series_data)
        result = prepped.bake(time_series_data)

        # More lenient, may include more series
        assert len(result.columns) >= 1

    def test_column_specification(self, time_series_data):
        """Test with specific columns."""
        rec = recipe().step_select_stationary(
            alpha=0.05,
            columns=['stationary', 'random_walk']
        )

        prepped = rec.prep(time_series_data)
        result = prepped.bake(time_series_data)

        # Should only test specified columns
        assert 'target' in result.columns


# ============================================================
# Test StepSelectCointegration
# ============================================================

class TestStepSelectCointegration:
    """Test cointegration-based feature selection."""

    def test_basic_cointegration(self, time_series_data):
        """Test selecting cointegrated series."""
        # Add cointegrated series
        ts_data = time_series_data.copy()
        ts_data['cointegrated'] = ts_data['trend'] + np.random.randn(len(ts_data)) * 0.5

        rec = recipe().step_select_cointegration(
            outcome='target',
            alpha=0.05
        )

        prepped = rec.prep(ts_data)
        result = prepped.bake(ts_data)

        assert 'target' in result.columns

    def test_strict_threshold(self, time_series_data):
        """Test with strict alpha."""
        rec = recipe().step_select_cointegration(
            outcome='target',
            alpha=0.01
        )

        prepped = rec.prep(time_series_data)
        result = prepped.bake(time_series_data)

        assert 'target' in result.columns

    def test_column_specification(self, time_series_data):
        """Test with specific columns."""
        rec = recipe().step_select_cointegration(
            outcome='target',
            alpha=0.05,
            columns=['trend', 'seasonal']
        )

        prepped = rec.prep(time_series_data)
        result = prepped.bake(time_series_data)

        assert 'target' in result.columns

    def test_skip_functionality(self, time_series_data):
        """Test skip parameter."""
        rec = recipe().step_select_cointegration(
            outcome='target',
            alpha=0.05,
            skip=True
        )

        prepped = rec.prep(time_series_data)
        result = prepped.bake(time_series_data)

        # With skip=True, keep all columns
        assert len(result.columns) == len(time_series_data.columns)


# ============================================================
# Test StepSelectSeasonal
# ============================================================

class TestStepSelectSeasonal:
    """Test seasonality-based feature selection."""

    def test_basic_seasonality(self, time_series_data):
        """Test detecting seasonal patterns."""
        rec = recipe().step_select_seasonal(
            period=12,
            threshold=0.1
        )

        prepped = rec.prep(time_series_data)
        result = prepped.bake(time_series_data)

        # Should detect seasonal patterns
        assert len(result.columns) >= 1

    def test_auto_period(self, time_series_data):
        """Test with automatic period detection."""
        rec = recipe().step_select_seasonal(
            period=None,
            threshold=0.1
        )

        prepped = rec.prep(time_series_data)
        result = prepped.bake(time_series_data)

        # Should auto-detect and keep some features
        assert len(result.columns) >= 1

    def test_threshold_selection(self, time_series_data):
        """Test threshold-based selection."""
        rec = recipe().step_select_seasonal(
            period=12,
            threshold=0.2
        )

        prepped = rec.prep(time_series_data)
        result = prepped.bake(time_series_data)

        # Higher threshold, may filter more
        assert len(result.columns) >= 1

    def test_column_specification(self, time_series_data):
        """Test with specific columns."""
        rec = recipe().step_select_seasonal(
            period=12,
            threshold=0.1,
            columns=['seasonal', 'stationary']
        )

        prepped = rec.prep(time_series_data)
        result = prepped.bake(time_series_data)

        assert 'target' in result.columns


# ============================================================
# Test Integration
# ============================================================

class TestIntegration:
    """Test combining multiple practical selection steps."""

    def test_regularization_pipeline(self, sample_data):
        """Test combining Lasso and Ridge."""
        rec = (recipe()
               .step_select_lasso(outcome='target', alpha=0.1, top_n=4)
               .step_select_ridge(outcome='target', alpha=1.0, top_n=2))

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        # Should have target + 2 features (most restrictive)
        assert len(result.columns) == 3
        assert 'target' in result.columns

    def test_univariate_variance_pipeline(self, sample_data):
        """Test univariate + variance filtering."""
        rec = (recipe()
               .step_select_univariate(outcome='target', score_func='f_regression', top_n=4)
               .step_select_variance_threshold(threshold=0.01))

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        assert 'target' in result.columns
        assert len(result.columns) <= 5  # target + up to 4 features

    def test_time_series_pipeline(self, time_series_data):
        """Test stationarity + seasonality."""
        rec = (recipe()
               .step_select_stationary(alpha=0.05)
               .step_select_seasonal(period=12, threshold=0.1))

        prepped = rec.prep(time_series_data)
        result = prepped.bake(time_series_data)

        # Should keep some features
        assert len(result.columns) >= 1

    def test_full_pipeline(self, sample_data):
        """Test comprehensive selection pipeline."""
        rec = (recipe()
               .step_select_variance_threshold(threshold=0.01)
               .step_select_univariate(outcome='target', score_func='f_regression', top_n=4)
               .step_select_elastic_net(outcome='target', alpha=0.5, l1_ratio=0.5, top_n=2))

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        # Final selection: target + 2 features
        assert len(result.columns) == 3
        assert 'target' in result.columns
