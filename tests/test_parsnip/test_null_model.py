"""
Tests for null_model with strategy parameter support

Verifies that null_model correctly supports:
- strategy="mean": Predict mean of training outcomes
- strategy="median": Predict median of training outcomes
- strategy="last": Predict last observed value (time series baseline)
"""

import pytest
import pandas as pd
import numpy as np
from py_parsnip.models.null_model import null_model


@pytest.fixture
def regression_data():
    """Create simple regression data for testing"""
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'x': np.arange(n),
        'y': np.random.randn(n) * 10 + 50  # Mean around 50
    })
    return data


class TestNullModelStrategies:
    """Test null_model with different strategies"""

    def test_strategy_mean_default(self, regression_data):
        """Test default strategy='mean' behavior"""
        train = regression_data.iloc[:80].copy()
        test = regression_data.iloc[80:].copy()

        # Default strategy should be 'mean'
        spec = null_model()
        assert spec.args['strategy'] == 'mean'

        # Fit and predict
        fit = spec.fit(train, 'y ~ x')
        predictions = fit.predict(test)

        # All predictions should equal training mean
        expected_mean = train['y'].mean()
        assert np.allclose(predictions['.pred'].values, expected_mean)

    def test_strategy_mean_explicit(self, regression_data):
        """Test explicit strategy='mean'"""
        train = regression_data.iloc[:80].copy()
        test = regression_data.iloc[80:].copy()

        spec = null_model(strategy='mean')
        fit = spec.fit(train, 'y ~ x')
        predictions = fit.predict(test)

        # All predictions should equal training mean
        expected_mean = train['y'].mean()
        assert np.allclose(predictions['.pred'].values, expected_mean)

        # Check fit_data contains correct method
        assert fit.fit_data['model']['method'] == 'mean'
        assert fit.fit_data['model']['strategy'] == 'mean'

    def test_strategy_median(self, regression_data):
        """Test strategy='median'"""
        train = regression_data.iloc[:80].copy()
        test = regression_data.iloc[80:].copy()

        spec = null_model(strategy='median')
        assert spec.args['strategy'] == 'median'

        fit = spec.fit(train, 'y ~ x')
        predictions = fit.predict(test)

        # All predictions should equal training median
        expected_median = train['y'].median()
        assert np.allclose(predictions['.pred'].values, expected_median)

        # Check fit_data contains correct method
        assert fit.fit_data['model']['method'] == 'median'
        assert fit.fit_data['model']['strategy'] == 'median'

    def test_strategy_last(self, regression_data):
        """Test strategy='last' (time series naive baseline)"""
        train = regression_data.iloc[:80].copy()
        test = regression_data.iloc[80:].copy()

        spec = null_model(strategy='last')
        assert spec.args['strategy'] == 'last'

        fit = spec.fit(train, 'y ~ x')
        predictions = fit.predict(test)

        # All predictions should equal last training value
        expected_last = train['y'].iloc[-1]
        assert np.allclose(predictions['.pred'].values, expected_last)

        # Check fit_data contains correct method
        assert fit.fit_data['model']['method'] == 'last'
        assert fit.fit_data['model']['strategy'] == 'last'

    def test_invalid_strategy_raises_error(self, regression_data):
        """Test that invalid strategy raises ValueError"""
        train = regression_data.iloc[:80].copy()

        spec = null_model(strategy='invalid')  # This creates spec, error happens on fit

        # Should raise ValueError when fitting
        with pytest.raises(ValueError, match="Unsupported strategy"):
            spec.fit(train, 'y ~ x')


class TestNullModelEvaluate:
    """Test null_model evaluation and extract_outputs"""

    def test_evaluate_with_mean_strategy(self, regression_data):
        """Test evaluate() and extract_outputs() with mean strategy"""
        train = regression_data.iloc[:80].copy()
        test = regression_data.iloc[80:].copy()

        spec = null_model(strategy='mean')
        fit = spec.fit(train, 'y ~ x')
        fit = fit.evaluate(test)

        # Extract outputs
        outputs, coefficients, stats = fit.extract_outputs()

        # Check outputs structure
        assert 'actuals' in outputs.columns
        assert 'fitted' in outputs.columns
        assert 'residuals' in outputs.columns
        assert 'split' in outputs.columns

        # Check train and test splits exist
        assert (outputs['split'] == 'train').sum() == 80
        assert (outputs['split'] == 'test').sum() == 20

        # Check all fitted values are constant (mean)
        train_fitted = outputs[outputs['split'] == 'train']['fitted'].values
        test_fitted = outputs[outputs['split'] == 'test']['fitted'].values
        assert np.std(train_fitted) < 1e-10  # All same
        assert np.std(test_fitted) < 1e-10   # All same
        assert np.allclose(train_fitted[0], test_fitted[0])  # Same value

        # Check coefficients
        assert 'variable' in coefficients.columns
        assert coefficients.loc[0, 'variable'] == '(Intercept)'
        assert np.isclose(coefficients.loc[0, 'coefficient'], train['y'].mean())

        # Check stats
        assert 'metric' in stats.columns
        assert 'value' in stats.columns
        assert 'split' in stats.columns

        # Should have both train and test metrics
        assert (stats['split'] == 'train').sum() > 0
        assert (stats['split'] == 'test').sum() > 0

    def test_evaluate_with_median_strategy(self, regression_data):
        """Test evaluate() with median strategy"""
        train = regression_data.iloc[:80].copy()
        test = regression_data.iloc[80:].copy()

        spec = null_model(strategy='median')
        fit = spec.fit(train, 'y ~ x')
        fit = fit.evaluate(test)

        outputs, coefficients, stats = fit.extract_outputs()

        # Check fitted values equal median
        train_fitted = outputs[outputs['split'] == 'train']['fitted'].values
        expected_median = train['y'].median()
        assert np.allclose(train_fitted[0], expected_median)

        # Check coefficients show median value
        assert np.isclose(coefficients.loc[0, 'coefficient'], expected_median)

    def test_evaluate_with_last_strategy(self, regression_data):
        """Test evaluate() with last strategy"""
        train = regression_data.iloc[:80].copy()
        test = regression_data.iloc[80:].copy()

        spec = null_model(strategy='last')
        fit = spec.fit(train, 'y ~ x')
        fit = fit.evaluate(test)

        outputs, coefficients, stats = fit.extract_outputs()

        # Check fitted values equal last training value
        train_fitted = outputs[outputs['split'] == 'train']['fitted'].values
        expected_last = train['y'].iloc[-1]
        assert np.allclose(train_fitted[0], expected_last)

        # Check coefficients show last value
        assert np.isclose(coefficients.loc[0, 'coefficient'], expected_last)


class TestNullModelStrategyComparison:
    """Compare different strategies"""

    def test_strategies_produce_different_predictions(self, regression_data):
        """Verify that mean, median, and last produce different predictions"""
        train = regression_data.iloc[:80].copy()

        # Fit with all strategies
        fit_mean = null_model(strategy='mean').fit(train, 'y ~ x')
        fit_median = null_model(strategy='median').fit(train, 'y ~ x')
        fit_last = null_model(strategy='last').fit(train, 'y ~ x')

        # Get baseline values
        mean_val = fit_mean.fit_data['model']['baseline_value']
        median_val = fit_median.fit_data['model']['baseline_value']
        last_val = fit_last.fit_data['model']['baseline_value']

        # Should be different (assuming data not perfectly symmetric)
        assert not np.isclose(mean_val, median_val, rtol=0.01) or \
               not np.isclose(mean_val, last_val, rtol=0.01)

        # Verify values are correct
        assert np.isclose(mean_val, train['y'].mean())
        assert np.isclose(median_val, train['y'].median())
        assert np.isclose(last_val, train['y'].iloc[-1])

    def test_last_strategy_useful_for_time_series(self):
        """Test that 'last' strategy provides reasonable time series baseline"""
        # Create time series with trend
        np.random.seed(123)
        n = 100
        trend = np.linspace(10, 50, n)
        noise = np.random.randn(n) * 2

        data = pd.DataFrame({
            'x': np.arange(n),
            'y': trend + noise
        })

        train = data.iloc[:80].copy()
        test = data.iloc[80:].copy()

        # Fit with 'last' strategy
        spec = null_model(strategy='last')
        fit = spec.fit(train, 'y ~ x')
        fit = fit.evaluate(test)

        outputs, _, stats = fit.extract_outputs()

        # For data with trend, 'last' should perform better than 'mean'
        # (last value is closer to test values)
        test_rmse_last = stats[(stats['split'] == 'test') & (stats['metric'] == 'rmse')]['value'].values[0]

        # Compare with mean strategy
        spec_mean = null_model(strategy='mean')
        fit_mean = spec_mean.fit(train, 'y ~ x')
        fit_mean = fit_mean.evaluate(test)
        _, _, stats_mean = fit_mean.extract_outputs()
        test_rmse_mean = stats_mean[(stats_mean['split'] == 'test') & (stats_mean['metric'] == 'rmse')]['value'].values[0]

        # For trending data, last should be better (lower RMSE)
        assert test_rmse_last < test_rmse_mean


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
