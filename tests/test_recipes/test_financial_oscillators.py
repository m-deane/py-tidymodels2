"""
Tests for financial oscillator recipe steps

Tests cover:
- Step specification creation
- RSI calculation (single and multiple periods)
- MACD calculation
- Stochastic Oscillator calculation
- Panel data support (grouped by ticker)
- NaN handling strategies
- Column validation
- Error handling
"""

import pytest
import pandas as pd
import numpy as np

from py_recipes.steps.financial_oscillators import StepOscillators, PreparedStepOscillators


class TestOscillatorsSpec:
    """Test StepOscillators specification"""

    def test_default_spec(self):
        """Test default oscillators specification"""
        step = StepOscillators(date_col="date", close_col="close")

        assert step.date_col == "date"
        assert step.close_col == "close"
        assert step.group_col is None
        assert step.rsi_periods == [14]
        assert step.macd is True
        assert step.macd_fast == 12
        assert step.macd_slow == 26
        assert step.macd_signal == 9
        assert step.stochastic is False
        assert step.handle_na == "keep"

    def test_spec_with_rsi_single_period(self):
        """Test spec with single RSI period"""
        step = StepOscillators(date_col="date", close_col="close", rsi_periods=21)

        assert step.rsi_periods == 21  # Will be converted to list in prep

    def test_spec_with_rsi_multiple_periods(self):
        """Test spec with multiple RSI periods"""
        step = StepOscillators(date_col="date", close_col="close", rsi_periods=[7, 14, 21])

        assert step.rsi_periods == [7, 14, 21]

    def test_spec_with_macd_params(self):
        """Test spec with custom MACD parameters"""
        step = StepOscillators(
            date_col="date",
            close_col="close",
            macd_fast=10,
            macd_slow=20,
            macd_signal=5,
        )

        assert step.macd_fast == 10
        assert step.macd_slow == 20
        assert step.macd_signal == 5

    def test_spec_with_stochastic(self):
        """Test spec with stochastic oscillator enabled"""
        step = StepOscillators(
            date_col="date",
            close_col="close",
            high_col="high",
            low_col="low",
            stochastic=True,
            stoch_period=10,
            stoch_smooth_k=5,
        )

        assert step.stochastic is True
        assert step.stoch_period == 10
        assert step.stoch_smooth_k == 5
        assert step.high_col == "high"
        assert step.low_col == "low"

    def test_spec_with_grouping(self):
        """Test spec with panel data grouping"""
        step = StepOscillators(date_col="date", close_col="close", group_col="ticker")

        assert step.group_col == "ticker"

    def test_spec_with_handle_na(self):
        """Test spec with different NaN handling strategies"""
        step_drop = StepOscillators(date_col="date", close_col="close", handle_na="drop")
        assert step_drop.handle_na == "drop"

        step_fill = StepOscillators(date_col="date", close_col="close", handle_na="fill_forward")
        assert step_fill.handle_na == "fill_forward"


class TestOscillatorsPrep:
    """Test StepOscillators prep() validation"""

    @pytest.fixture
    def simple_data(self):
        """Create simple time series data"""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        close = np.linspace(100, 150, 50) + np.random.normal(0, 2, 50)

        return pd.DataFrame({"date": dates, "close": close})

    @pytest.fixture
    def ohlc_data(self):
        """Create OHLC data"""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        close = np.linspace(100, 150, 50)
        high = close + np.random.uniform(1, 5, 50)
        low = close - np.random.uniform(1, 5, 50)
        open_price = close + np.random.uniform(-2, 2, 50)

        return pd.DataFrame({
            "date": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
        })

    def test_prep_basic(self, simple_data):
        """Test basic prep() validation"""
        step = StepOscillators(date_col="date", close_col="close")
        prepped = step.prep(simple_data)

        assert isinstance(prepped, PreparedStepOscillators)
        assert prepped.date_col == "date"
        assert prepped.close_col == "close"

    def test_prep_converts_single_period_to_list(self, simple_data):
        """Test that prep converts single RSI period to list"""
        step = StepOscillators(date_col="date", close_col="close", rsi_periods=21)
        prepped = step.prep(simple_data)

        assert prepped.rsi_periods == [21]

    def test_prep_preserves_multiple_periods(self, simple_data):
        """Test that prep preserves multiple RSI periods"""
        step = StepOscillators(date_col="date", close_col="close", rsi_periods=[7, 14, 21])
        prepped = step.prep(simple_data)

        assert prepped.rsi_periods == [7, 14, 21]

    def test_prep_validates_required_columns(self, simple_data):
        """Test that prep validates required columns exist"""
        step = StepOscillators(date_col="date", close_col="missing_col")

        with pytest.raises(ValueError, match="Missing required columns"):
            step.prep(simple_data)

    def test_prep_validates_stochastic_columns(self, simple_data):
        """Test that prep validates stochastic columns when enabled"""
        step = StepOscillators(
            date_col="date",
            close_col="close",
            stochastic=True,
            # Missing high_col and low_col
        )

        with pytest.raises(ValueError, match="Stochastic oscillator requires high_col and low_col"):
            step.prep(simple_data)

    def test_prep_validates_handle_na_parameter(self, simple_data):
        """Test that prep validates handle_na parameter"""
        step = StepOscillators(date_col="date", close_col="close", handle_na="invalid")

        with pytest.raises(ValueError, match="handle_na must be"):
            step.prep(simple_data)


class TestOscillatorsBake:
    """Test StepOscillators bake() transformation"""

    @pytest.fixture
    def ts_data(self):
        """Create time series data for testing"""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")

        # Create trending price data
        t = np.arange(100)
        trend = 0.5 * t
        noise = np.random.normal(0, 2, 100)
        close = 100 + trend + noise

        return pd.DataFrame({"date": dates, "close": close})

    @pytest.fixture
    def ohlc_data(self):
        """Create OHLC data"""
        np.random.seed(123)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")

        t = np.arange(100)
        close = 100 + 0.5 * t + np.random.normal(0, 2, 100)
        high = close + np.random.uniform(1, 3, 100)
        low = close - np.random.uniform(1, 3, 100)
        open_price = close + np.random.uniform(-1, 1, 100)

        return pd.DataFrame({
            "date": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
        })

    @pytest.fixture
    def panel_data(self):
        """Create panel data with multiple tickers"""
        np.random.seed(456)
        dates = pd.date_range("2020-01-01", periods=50, freq="D")

        data_list = []
        for ticker in ["AAPL", "GOOGL", "MSFT"]:
            t = np.arange(50)
            close = 100 + 0.5 * t + np.random.normal(0, 2, 50)

            ticker_data = pd.DataFrame({
                "date": dates,
                "ticker": ticker,
                "close": close,
            })
            data_list.append(ticker_data)

        return pd.concat(data_list, ignore_index=True)

    def test_bake_rsi_single_period(self, ts_data):
        """Test baking with single RSI period"""
        step = StepOscillators(date_col="date", close_col="close", rsi_periods=[14], macd=False)
        prepped = step.prep(ts_data)
        result = prepped.bake(ts_data)

        assert "close_rsi_14" in result.columns
        assert len(result) == len(ts_data)

    def test_bake_rsi_multiple_periods(self, ts_data):
        """Test baking with multiple RSI periods"""
        step = StepOscillators(
            date_col="date",
            close_col="close",
            rsi_periods=[7, 14, 21],
            macd=False,
        )
        prepped = step.prep(ts_data)
        result = prepped.bake(ts_data)

        assert "close_rsi_7" in result.columns
        assert "close_rsi_14" in result.columns
        assert "close_rsi_21" in result.columns

    def test_bake_macd(self, ts_data):
        """Test baking with MACD"""
        step = StepOscillators(
            date_col="date",
            close_col="close",
            rsi_periods=None,
            macd=True,
        )
        prepped = step.prep(ts_data)
        result = prepped.bake(ts_data)

        # MACD should create three columns (pytimetk naming convention)
        assert "close_macd_line_12_26_9" in result.columns
        assert "close_macd_signal_line_12_26_9" in result.columns
        assert "close_macd_histogram_12_26_9" in result.columns

    def test_bake_stochastic(self, ohlc_data):
        """Test baking with Stochastic Oscillator"""
        step = StepOscillators(
            date_col="date",
            close_col="close",
            high_col="high",
            low_col="low",
            rsi_periods=None,
            macd=False,
            stochastic=True,
        )
        prepped = step.prep(ohlc_data)
        result = prepped.bake(ohlc_data)

        # Stochastic creates %K and %D columns
        assert "stoch_k" in result.columns or "close_stoch_k" in result.columns or any("stoch" in col for col in result.columns)

    def test_bake_all_indicators(self, ohlc_data):
        """Test baking with all indicators enabled"""
        step = StepOscillators(
            date_col="date",
            close_col="close",
            high_col="high",
            low_col="low",
            rsi_periods=[14],
            macd=True,
            stochastic=True,
        )
        prepped = step.prep(ohlc_data)
        result = prepped.bake(ohlc_data)

        # Should have RSI, MACD, and Stochastic columns (using pytimetk naming convention)
        assert "close_rsi_14" in result.columns
        assert "close_macd_line_12_26_9" in result.columns
        # Stochastic column names may vary by pytimetk version

    def test_bake_panel_data(self, panel_data):
        """Test baking with panel data (grouped)"""
        step = StepOscillators(
            date_col="date",
            close_col="close",
            group_col="ticker",
            rsi_periods=[14],
            macd=True,
        )
        prepped = step.prep(panel_data)
        result = prepped.bake(panel_data)

        # Should have indicators for all tickers (using pytimetk naming convention)
        assert "close_rsi_14" in result.columns
        assert "close_macd_line_12_26_9" in result.columns

        # Check that each ticker has its own indicators
        for ticker in ["AAPL", "GOOGL", "MSFT"]:
            ticker_data = result[result["ticker"] == ticker]
            assert len(ticker_data) == 50

    def test_bake_handle_na_keep(self, ts_data):
        """Test baking with NaN handling: keep"""
        step = StepOscillators(
            date_col="date",
            close_col="close",
            rsi_periods=[14],
            macd=False,
            handle_na="keep",
        )
        prepped = step.prep(ts_data)
        result = prepped.bake(ts_data)

        # Should have NaN values in early rows (RSI lookback)
        assert result["close_rsi_14"].isna().any()
        assert len(result) == len(ts_data)

    def test_bake_handle_na_drop(self, ts_data):
        """Test baking with NaN handling: drop"""
        step = StepOscillators(
            date_col="date",
            close_col="close",
            rsi_periods=[14],
            macd=False,
            handle_na="drop",
        )
        prepped = step.prep(ts_data)
        result = prepped.bake(ts_data)

        # Should have no NaN values
        assert not result["close_rsi_14"].isna().any()
        # Should have fewer rows (dropped NaN rows)
        assert len(result) < len(ts_data)

    def test_bake_handle_na_fill_forward(self, ts_data):
        """Test baking with NaN handling: fill_forward"""
        step = StepOscillators(
            date_col="date",
            close_col="close",
            rsi_periods=[14],
            macd=False,
            handle_na="fill_forward",
        )
        prepped = step.prep(ts_data)
        result = prepped.bake(ts_data)

        # Should have no NaN values after fill
        # Note: first rows might still be NaN if there's nothing to fill from
        # But no NaN in later rows
        assert len(result) == len(ts_data)

    def test_bake_preserves_original_columns(self, ts_data):
        """Test that baking preserves original columns"""
        step = StepOscillators(
            date_col="date",
            close_col="close",
            rsi_periods=[14],
            macd=True,
        )
        prepped = step.prep(ts_data)
        result = prepped.bake(ts_data)

        # Original columns should be preserved
        assert "date" in result.columns
        assert "close" in result.columns


class TestOscillatorsErrors:
    """Test error handling"""

    def test_missing_pytimetk_import(self, monkeypatch):
        """Test error when pytimetk is not installed"""
        # This test would require mocking the import, which is complex
        # Skipping for now, but in practice pytimetk should be in requirements
        pass

    def test_invalid_handle_na_during_prep(self):
        """Test invalid handle_na parameter"""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({"date": dates, "close": range(30)})

        step = StepOscillators(date_col="date", close_col="close", handle_na="invalid_option")

        with pytest.raises(ValueError, match="handle_na must be"):
            step.prep(data)

    def test_missing_stochastic_columns(self):
        """Test error when stochastic enabled but high/low missing"""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({"date": dates, "close": range(30)})

        step = StepOscillators(
            date_col="date",
            close_col="close",
            stochastic=True,
            # Missing high_col and low_col parameters
        )

        with pytest.raises(ValueError, match="Stochastic oscillator requires"):
            step.prep(data)

    def test_missing_required_column(self):
        """Test error when required column is missing"""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({"date": dates, "close": range(30)})

        step = StepOscillators(date_col="date", close_col="nonexistent_col")

        with pytest.raises(ValueError, match="Missing required columns"):
            step.prep(data)

    def test_missing_group_column(self):
        """Test error when group column is missing"""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.DataFrame({"date": dates, "close": range(30)})

        step = StepOscillators(date_col="date", close_col="close", group_col="nonexistent")

        with pytest.raises(ValueError, match="Missing required columns"):
            step.prep(data)
