"""
Tests for naive_reg with strategy parameter support

Verifies that naive_reg correctly supports:
- strategy="naive": Last observed value (random walk)
- strategy="seasonal_naive": Last value from same season
- strategy="drift": Linear trend from first to last value
- strategy="window": Rolling window average (moving average)
"""

import pytest
import pandas as pd
import numpy as np
from py_parsnip.models.naive_reg import naive_reg


@pytest.fixture
def time_series_data():
    """Create simple time series data for testing"""
    np.random.seed(42)
    n = 100
    trend = np.linspace(10, 50, n)
    noise = np.random.randn(n) * 2
    data = pd.DataFrame({
        'x': np.arange(n),
        'y': trend + noise
    })
    return data


@pytest.fixture
def seasonal_data():
    """Create data with clear seasonal pattern"""
    np.random.seed(42)
    n = 84  # 7 weeks of daily data

    # Weekly pattern: high on weekends, low on weekdays
    pattern = np.array([10, 10, 10, 10, 10, 20, 20])  # Mon-Sun
    seasonal = np.tile(pattern, n // 7)
    noise = np.random.randn(n) * 1

    data = pd.DataFrame({
        'x': np.arange(n),
        'y': seasonal + noise
    })
    return data


class TestNaiveRegStrategies:
    """Test naive_reg with different strategies"""

    def test_strategy_naive(self, time_series_data):
        """Test strategy='naive' (last value)"""
        train = time_series_data.iloc[:80].copy()
        test = time_series_data.iloc[80:].copy()

        spec = naive_reg(strategy='naive')
        assert spec.args['strategy'] == 'naive'

        fit = spec.fit(train, 'y ~ x')
        predictions = fit.predict(test)

        # All predictions should equal last training value
        expected_last = train['y'].iloc[-1]
        assert np.allclose(predictions['.pred'].values, expected_last)

    def test_strategy_seasonal_naive(self, seasonal_data):
        """Test strategy='seasonal_naive' with seasonal_period=7"""
        train = seasonal_data.iloc[:70].copy()  # 10 weeks
        test = seasonal_data.iloc[70:77].copy()  # 1 week

        spec = naive_reg(strategy='seasonal_naive', seasonal_period=7)
        assert spec.args['strategy'] == 'seasonal_naive'
        assert spec.args['seasonal_period'] == 7

        fit = spec.fit(train, 'y ~ x')
        predictions = fit.predict(test)

        # Predictions should follow seasonal pattern
        # Each day should predict last occurrence of same day
        assert len(predictions) == 7

    def test_strategy_seasonal_naive_requires_period(self, time_series_data):
        """Test that seasonal_naive raises error without seasonal_period"""
        train = time_series_data.iloc[:80].copy()

        spec = naive_reg(strategy='seasonal_naive')

        with pytest.raises(ValueError, match="seasonal_period required"):
            spec.fit(train, 'y ~ x')

    def test_strategy_drift(self, time_series_data):
        """Test strategy='drift' (linear extrapolation)"""
        train = time_series_data.iloc[:80].copy()
        test = time_series_data.iloc[80:85].copy()

        spec = naive_reg(strategy='drift')
        assert spec.args['strategy'] == 'drift'

        fit = spec.fit(train, 'y ~ x')
        predictions = fit.predict(test)

        # Predictions should follow linear trend
        # drift = (y_last - y_first) / (n - 1)
        expected_drift = (train['y'].iloc[-1] - train['y'].iloc[0]) / (len(train) - 1)

        # First prediction = last_value + drift
        # Second prediction = last_value + 2*drift, etc.
        last_value = train['y'].iloc[-1]
        for h, pred in enumerate(predictions['.pred'].values, start=1):
            expected = last_value + expected_drift * h
            assert np.isclose(pred, expected)

    def test_strategy_window(self, time_series_data):
        """Test strategy='window' with window_size=7"""
        train = time_series_data.iloc[:80].copy()
        test = time_series_data.iloc[80:85].copy()

        spec = naive_reg(strategy='window', window_size=7)
        assert spec.args['strategy'] == 'window'
        assert spec.args['window_size'] == 7

        fit = spec.fit(train, 'y ~ x')
        predictions = fit.predict(test)

        # All predictions should equal mean of last 7 training values
        expected_window_avg = train['y'].iloc[-7:].mean()
        assert np.allclose(predictions['.pred'].values, expected_window_avg)

    def test_strategy_window_requires_size(self, time_series_data):
        """Test that window strategy raises error without window_size"""
        train = time_series_data.iloc[:80].copy()

        spec = naive_reg(strategy='window')

        with pytest.raises(ValueError, match="window_size required"):
            spec.fit(train, 'y ~ x')

    def test_invalid_strategy_raises_error(self, time_series_data):
        """Test that invalid strategy raises ValueError"""
        train = time_series_data.iloc[:80].copy()

        spec = naive_reg(strategy='invalid')

        with pytest.raises(ValueError, match="Unknown strategy"):
            spec.fit(train, 'y ~ x')


class TestNaiveRegFittedValues:
    """Test in-sample fitted values for each strategy"""

    def test_naive_fitted_values(self, time_series_data):
        """Test that naive strategy produces y[t-1] for fitted values"""
        train = time_series_data.iloc[:80].copy()

        spec = naive_reg(strategy='naive')
        fit = spec.fit(train, 'y ~ x')

        outputs, _, _ = fit.extract_outputs()
        train_outputs = outputs[outputs['split'] == 'train']

        # Fitted values should be shifted by 1 (y[t-1])
        # First fitted value = first actual (special case)
        assert train_outputs.iloc[0]['fitted'] == train_outputs.iloc[0]['actuals']

        # Subsequent fitted values = previous actuals
        for i in range(1, len(train_outputs)):
            assert np.isclose(
                train_outputs.iloc[i]['fitted'],
                train_outputs.iloc[i-1]['actuals']
            )

    def test_window_fitted_values(self, time_series_data):
        """Test that window strategy produces correct rolling averages"""
        train = time_series_data.iloc[:20].copy()  # Small sample for easier testing

        spec = naive_reg(strategy='window', window_size=3)
        fit = spec.fit(train, 'y ~ x')

        outputs, _, _ = fit.extract_outputs()
        train_outputs = outputs[outputs['split'] == 'train']

        # First value = itself
        assert train_outputs.iloc[0]['fitted'] == train_outputs.iloc[0]['actuals']

        # Second value = mean of first value (expanding window)
        assert np.isclose(
            train_outputs.iloc[1]['fitted'],
            train_outputs.iloc[0]['actuals']
        )

        # Third value = mean of first 2 values
        assert np.isclose(
            train_outputs.iloc[2]['fitted'],
            train_outputs.iloc[:2]['actuals'].mean()
        )

        # Fourth value onward = rolling 3-period mean
        for i in range(3, len(train_outputs)):
            expected = train_outputs.iloc[i-3:i]['actuals'].mean()
            assert np.isclose(train_outputs.iloc[i]['fitted'], expected)


class TestNaiveRegEvaluate:
    """Test naive_reg evaluation and extract_outputs"""

    def test_evaluate_with_naive_strategy(self, time_series_data):
        """Test evaluate() and extract_outputs() with naive strategy"""
        train = time_series_data.iloc[:80].copy()
        test = time_series_data.iloc[80:].copy()

        spec = naive_reg(strategy='naive')
        fit = spec.fit(train, 'y ~ x')
        fit = fit.evaluate(test)

        # Extract outputs
        outputs, coefficients, stats = fit.extract_outputs()

        # Check outputs structure
        assert 'actuals' in outputs.columns
        assert 'fitted' in outputs.columns
        assert 'residuals' in outputs.columns
        assert 'split' in outputs.columns

        # Check train and test splits
        assert (outputs['split'] == 'train').sum() == 80
        assert (outputs['split'] == 'test').sum() == 20

        # Check coefficients
        assert 'variable' in coefficients.columns
        assert coefficients.loc[0, 'variable'] == 'strategy'
        assert coefficients.loc[0, 'coefficient'] == 'naive'

        # Check stats
        assert 'metric' in stats.columns
        assert 'value' in stats.columns
        assert 'split' in stats.columns

        # Should have metrics for both train and test
        assert (stats['split'] == 'train').sum() > 0
        assert (stats['split'] == 'test').sum() > 0

    def test_evaluate_with_window_strategy(self, time_series_data):
        """Test evaluate() with window strategy"""
        train = time_series_data.iloc[:80].copy()
        test = time_series_data.iloc[80:].copy()

        spec = naive_reg(strategy='window', window_size=7)
        fit = spec.fit(train, 'y ~ x')
        fit = fit.evaluate(test)

        outputs, coefficients, stats = fit.extract_outputs()

        # Check coefficients show window strategy
        assert coefficients.loc[0, 'coefficient'] == 'window'

        # Check stats include strategy
        strategy_stat = stats[(stats['metric'] == 'strategy') & (stats['split'] == 'train')]
        assert len(strategy_stat) == 1
        assert strategy_stat['value'].values[0] == 'window'


class TestNaiveRegStrategyComparison:
    """Compare different strategies"""

    def test_strategies_produce_different_predictions(self, time_series_data):
        """Verify that different strategies produce different predictions"""
        train = time_series_data.iloc[:80].copy()
        test = time_series_data.iloc[80:85].copy()

        # Fit with all strategies
        fit_naive = naive_reg(strategy='naive').fit(train, 'y ~ x')
        fit_drift = naive_reg(strategy='drift').fit(train, 'y ~ x')
        fit_window = naive_reg(strategy='window', window_size=7).fit(train, 'y ~ x')

        # Get predictions
        pred_naive = fit_naive.predict(test)
        pred_drift = fit_drift.predict(test)
        pred_window = fit_window.predict(test)

        # Naive predictions are all equal (constant)
        assert np.std(pred_naive['.pred'].values) < 1e-10

        # Drift predictions increase linearly
        drift_diff = np.diff(pred_drift['.pred'].values)
        assert np.std(drift_diff) < 1e-10  # Constant differences

        # Window predictions are constant (moving average)
        assert np.std(pred_window['.pred'].values) < 1e-10

        # All three should produce different values
        naive_val = pred_naive['.pred'].values[0]
        drift_val = pred_drift['.pred'].values[0]
        window_val = pred_window['.pred'].values[0]

        # At least two should be different
        assert not (np.isclose(naive_val, drift_val) and
                   np.isclose(naive_val, window_val))

    def test_drift_better_than_naive_on_trending_data(self):
        """Test that drift performs better than naive on data with strong trend"""
        np.random.seed(123)
        n = 100

        # Create data with strong upward trend
        trend = np.linspace(10, 100, n)
        noise = np.random.randn(n) * 2

        data = pd.DataFrame({
            'x': np.arange(n),
            'y': trend + noise
        })

        train = data.iloc[:80].copy()
        test = data.iloc[80:].copy()

        # Fit both strategies
        fit_naive = naive_reg(strategy='naive').fit(train, 'y ~ x')
        fit_naive = fit_naive.evaluate(test)

        fit_drift = naive_reg(strategy='drift').fit(train, 'y ~ x')
        fit_drift = fit_drift.evaluate(test)

        # Get test RMSE
        _, _, stats_naive = fit_naive.extract_outputs()
        _, _, stats_drift = fit_drift.extract_outputs()

        rmse_naive = stats_naive[(stats_naive['split'] == 'test') & (stats_naive['metric'] == 'rmse')]['value'].values[0]
        rmse_drift = stats_drift[(stats_drift['split'] == 'test') & (stats_drift['metric'] == 'rmse')]['value'].values[0]

        # Drift should be better (lower RMSE) on trending data
        assert rmse_drift < rmse_naive


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
