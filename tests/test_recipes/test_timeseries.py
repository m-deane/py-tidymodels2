"""
Tests for time series recipe steps
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from py_recipes import recipe


class TestStepLag:
    """Test step_lag functionality"""

    @pytest.fixture
    def ts_data(self):
        """Create time series data"""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'date': dates,
            'sales': np.random.randn(100).cumsum() + 100,
            'price': np.random.randn(100) * 10 + 50,
            'temperature': np.random.randn(100) * 5 + 20
        })

    def test_lag_single_column_single_lag(self, ts_data):
        """Test creating single lag feature"""
        rec = recipe().step_lag(columns=["sales"], lags=[1])
        rec_fit = rec.prep(ts_data)
        transformed = rec_fit.bake(ts_data)

        assert "sales_lag_1" in transformed.columns
        assert pd.isna(transformed["sales_lag_1"].iloc[0])  # First value should be NA
        assert transformed["sales_lag_1"].iloc[1] == ts_data["sales"].iloc[0]

    def test_lag_multiple_lags(self, ts_data):
        """Test creating multiple lag features"""
        rec = recipe().step_lag(columns=["sales"], lags=[1, 7, 30])
        rec_fit = rec.prep(ts_data)
        transformed = rec_fit.bake(ts_data)

        assert "sales_lag_1" in transformed.columns
        assert "sales_lag_7" in transformed.columns
        assert "sales_lag_30" in transformed.columns

        # Check lag values
        assert transformed["sales_lag_7"].iloc[7] == ts_data["sales"].iloc[0]

    def test_lag_multiple_columns(self, ts_data):
        """Test creating lags for multiple columns"""
        rec = recipe().step_lag(columns=["sales", "price"], lags=[1, 3])
        rec_fit = rec.prep(ts_data)
        transformed = rec_fit.bake(ts_data)

        assert "sales_lag_1" in transformed.columns
        assert "sales_lag_3" in transformed.columns
        assert "price_lag_1" in transformed.columns
        assert "price_lag_3" in transformed.columns

    def test_lag_preserves_original_columns(self, ts_data):
        """Test that original columns are preserved"""
        rec = recipe().step_lag(columns=["sales"], lags=[1])
        rec_fit = rec.prep(ts_data)
        transformed = rec_fit.bake(ts_data)

        assert "sales" in transformed.columns
        assert "price" in transformed.columns

    def test_lag_new_data(self, ts_data):
        """Test applying lag to new data"""
        train = ts_data[:80]
        test = ts_data[80:]

        rec = recipe().step_lag(columns=["sales"], lags=[1, 7])
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        assert "sales_lag_1" in test_transformed.columns
        assert len(test_transformed) == len(test)


class TestStepDiff:
    """Test step_diff functionality"""

    @pytest.fixture
    def ts_data(self):
        """Create time series data"""
        return pd.DataFrame({
            'value': [100, 102, 105, 103, 108, 110, 107],
            'other': [1, 2, 3, 4, 5, 6, 7]
        })

    def test_diff_single_column(self, ts_data):
        """Test differencing single column"""
        rec = recipe().step_diff(columns=["value"])
        rec_fit = rec.prep(ts_data)
        transformed = rec_fit.bake(ts_data)

        assert "value_diff_lag_1" in transformed.columns
        assert pd.isna(transformed["value_diff_lag_1"].iloc[0])  # First diff is NA
        assert transformed["value_diff_lag_1"].iloc[1] == 2  # 102 - 100

    def test_diff_custom_lag(self, ts_data):
        """Test differencing with custom lag"""
        rec = recipe().step_diff(columns=["value"], lag=2)
        rec_fit = rec.prep(ts_data)
        transformed = rec_fit.bake(ts_data)

        assert "value_diff_lag_2" in transformed.columns
        assert transformed["value_diff_lag_2"].iloc[2] == 5  # 105 - 100

    def test_diff_multiple_differences(self, ts_data):
        """Test second-order differencing"""
        rec = recipe().step_diff(columns=["value"], lag=1, differences=2)
        rec_fit = rec.prep(ts_data)
        transformed = rec_fit.bake(ts_data)

        assert "value_diff_lag_1_order_2" in transformed.columns

    def test_diff_all_numeric_default(self, ts_data):
        """Test differencing all numeric columns by default"""
        rec = recipe().step_diff()
        rec_fit = rec.prep(ts_data)
        transformed = rec_fit.bake(ts_data)

        assert "value_diff_lag_1" in transformed.columns
        assert "other_diff_lag_1" in transformed.columns

    def test_diff_preserves_originals(self, ts_data):
        """Test that original columns are preserved"""
        rec = recipe().step_diff(columns=["value"])
        rec_fit = rec.prep(ts_data)
        transformed = rec_fit.bake(ts_data)

        assert "value" in transformed.columns


class TestStepPctChange:
    """Test step_pct_change functionality"""

    @pytest.fixture
    def ts_data(self):
        """Create time series data"""
        return pd.DataFrame({
            'price': [100, 105, 110, 108, 115],
            'volume': [1000, 1100, 1050, 1200, 1150]
        })

    def test_pct_change_single_column(self, ts_data):
        """Test percent change for single column"""
        rec = recipe().step_pct_change(columns=["price"])
        rec_fit = rec.prep(ts_data)
        transformed = rec_fit.bake(ts_data)

        assert "price_pct_change_1" in transformed.columns
        assert pd.isna(transformed["price_pct_change_1"].iloc[0])
        assert abs(transformed["price_pct_change_1"].iloc[1] - 0.05) < 0.001  # 5% increase

    def test_pct_change_custom_periods(self, ts_data):
        """Test percent change with custom periods"""
        rec = recipe().step_pct_change(columns=["price"], periods=2)
        rec_fit = rec.prep(ts_data)
        transformed = rec_fit.bake(ts_data)

        assert "price_pct_change_2" in transformed.columns
        assert abs(transformed["price_pct_change_2"].iloc[2] - 0.10) < 0.001  # 10% increase

    def test_pct_change_all_numeric_default(self, ts_data):
        """Test percent change on all numeric by default"""
        rec = recipe().step_pct_change()
        rec_fit = rec.prep(ts_data)
        transformed = rec_fit.bake(ts_data)

        assert "price_pct_change_1" in transformed.columns
        assert "volume_pct_change_1" in transformed.columns

    def test_pct_change_preserves_originals(self, ts_data):
        """Test that original columns are preserved"""
        rec = recipe().step_pct_change(columns=["price"])
        rec_fit = rec.prep(ts_data)
        transformed = rec_fit.bake(ts_data)

        assert "price" in transformed.columns


class TestStepRolling:
    """Test step_rolling functionality"""

    @pytest.fixture
    def ts_data(self):
        """Create time series data"""
        return pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'other': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })

    def test_rolling_mean(self, ts_data):
        """Test rolling mean calculation"""
        rec = recipe().step_rolling(columns=["value"], window=3, stats=["mean"])
        rec_fit = rec.prep(ts_data)
        transformed = rec_fit.bake(ts_data)

        assert "value_rolling_3_mean" in transformed.columns
        assert pd.isna(transformed["value_rolling_3_mean"].iloc[0])
        assert transformed["value_rolling_3_mean"].iloc[2] == 2.0  # mean(1,2,3)
        assert transformed["value_rolling_3_mean"].iloc[3] == 3.0  # mean(2,3,4)

    def test_rolling_multiple_stats(self, ts_data):
        """Test rolling with multiple statistics"""
        rec = recipe().step_rolling(
            columns=["value"],
            window=3,
            stats=["mean", "std", "min", "max", "sum"]
        )
        rec_fit = rec.prep(ts_data)
        transformed = rec_fit.bake(ts_data)

        assert "value_rolling_3_mean" in transformed.columns
        assert "value_rolling_3_std" in transformed.columns
        assert "value_rolling_3_min" in transformed.columns
        assert "value_rolling_3_max" in transformed.columns
        assert "value_rolling_3_sum" in transformed.columns

        # Check sum calculation
        assert transformed["value_rolling_3_sum"].iloc[2] == 6.0  # sum(1,2,3)

    def test_rolling_different_windows(self, ts_data):
        """Test rolling with different window sizes"""
        rec = recipe().step_rolling(columns=["value"], window=5, stats=["mean"])
        rec_fit = rec.prep(ts_data)
        transformed = rec_fit.bake(ts_data)

        assert "value_rolling_5_mean" in transformed.columns
        assert transformed["value_rolling_5_mean"].iloc[4] == 3.0  # mean(1,2,3,4,5)

    def test_rolling_multiple_columns(self, ts_data):
        """Test rolling on multiple columns"""
        rec = recipe().step_rolling(columns=["value", "other"], window=3, stats=["mean"])
        rec_fit = rec.prep(ts_data)
        transformed = rec_fit.bake(ts_data)

        assert "value_rolling_3_mean" in transformed.columns
        assert "other_rolling_3_mean" in transformed.columns

    def test_rolling_preserves_originals(self, ts_data):
        """Test that original columns are preserved"""
        rec = recipe().step_rolling(columns=["value"], window=3, stats=["mean"])
        rec_fit = rec.prep(ts_data)
        transformed = rec_fit.bake(ts_data)

        assert "value" in transformed.columns


class TestStepDate:
    """Test step_date functionality"""

    @pytest.fixture
    def datetime_data(self):
        """Create data with datetime column"""
        dates = pd.date_range(start='2020-01-15', periods=100, freq='D')
        return pd.DataFrame({
            'date': dates,
            'value': np.random.randn(100)
        })

    def test_date_basic_features(self, datetime_data):
        """Test extracting basic date features"""
        rec = recipe().step_date("date", features=["year", "month", "day"])
        rec_fit = rec.prep(datetime_data)
        transformed = rec_fit.bake(datetime_data)

        assert "date_year" in transformed.columns
        assert "date_month" in transformed.columns
        assert "date_day" in transformed.columns

        assert transformed["date_year"].iloc[0] == 2020
        assert transformed["date_month"].iloc[0] == 1
        assert transformed["date_day"].iloc[0] == 15

    def test_date_dayofweek(self, datetime_data):
        """Test extracting day of week"""
        rec = recipe().step_date("date", features=["dayofweek"])
        rec_fit = rec.prep(datetime_data)
        transformed = rec_fit.bake(datetime_data)

        assert "date_dayofweek" in transformed.columns
        assert transformed["date_dayofweek"].iloc[0] in range(7)

    def test_date_quarter(self, datetime_data):
        """Test extracting quarter"""
        rec = recipe().step_date("date", features=["quarter"])
        rec_fit = rec.prep(datetime_data)
        transformed = rec_fit.bake(datetime_data)

        assert "date_quarter" in transformed.columns
        assert transformed["date_quarter"].iloc[0] == 1

    def test_date_is_weekend(self, datetime_data):
        """Test weekend indicator"""
        rec = recipe().step_date("date", features=["is_weekend"])
        rec_fit = rec.prep(datetime_data)
        transformed = rec_fit.bake(datetime_data)

        assert "date_is_weekend" in transformed.columns
        assert transformed["date_is_weekend"].dtype == int

    def test_date_is_month_start_end(self, datetime_data):
        """Test month start/end indicators"""
        rec = recipe().step_date("date", features=["is_month_start", "is_month_end"])
        rec_fit = rec.prep(datetime_data)
        transformed = rec_fit.bake(datetime_data)

        assert "date_is_month_start" in transformed.columns
        assert "date_is_month_end" in transformed.columns

    def test_date_multiple_features(self, datetime_data):
        """Test extracting multiple date features"""
        rec = recipe().step_date(
            "date",
            features=["year", "month", "day", "dayofweek", "quarter", "is_weekend"]
        )
        rec_fit = rec.prep(datetime_data)
        transformed = rec_fit.bake(datetime_data)

        assert "date_year" in transformed.columns
        assert "date_month" in transformed.columns
        assert "date_day" in transformed.columns
        assert "date_dayofweek" in transformed.columns
        assert "date_quarter" in transformed.columns
        assert "date_is_weekend" in transformed.columns

    def test_date_preserves_original(self, datetime_data):
        """Test that original date column is preserved"""
        rec = recipe().step_date("date", features=["year", "month"])
        rec_fit = rec.prep(datetime_data)
        transformed = rec_fit.bake(datetime_data)

        assert "date" in transformed.columns

    def test_date_string_conversion(self):
        """Test automatic conversion of string dates"""
        data = pd.DataFrame({
            'date': ['2020-01-15', '2020-01-16', '2020-01-17'],
            'value': [1, 2, 3]
        })

        rec = recipe().step_date("date", features=["year", "month"])
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        assert "date_year" in transformed.columns
        assert transformed["date_year"].iloc[0] == 2020


class TestTimeSeriesPipeline:
    """Test combinations of time series steps"""

    @pytest.fixture
    def ts_data(self):
        """Create time series data"""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'date': dates,
            'sales': np.random.randn(100).cumsum() + 100,
            'price': np.random.randn(100) * 10 + 50
        })

    def test_combined_time_series_steps(self, ts_data):
        """Test combining multiple time series steps"""
        rec = (
            recipe()
            .step_date("date", features=["year", "month", "dayofweek"])
            .step_lag(columns=["sales", "price"], lags=[1, 7])
            .step_rolling(columns=["sales"], window=7, stats=["mean", "std"])
            .step_pct_change(columns=["price"], periods=1)
        )

        rec_fit = rec.prep(ts_data)
        transformed = rec_fit.bake(ts_data)

        # Check all features created
        assert "date_year" in transformed.columns
        assert "sales_lag_1" in transformed.columns
        assert "sales_lag_7" in transformed.columns
        assert "sales_rolling_7_mean" in transformed.columns
        assert "price_pct_change_1" in transformed.columns

    def test_diff_and_lag_combination(self, ts_data):
        """Test combining differencing and lags"""
        rec = (
            recipe()
            .step_diff(columns=["sales"], lag=1)
            .step_lag(columns=["sales"], lags=[1, 2])
        )

        rec_fit = rec.prep(ts_data)
        transformed = rec_fit.bake(ts_data)

        assert "sales_diff_1" in transformed.columns
        assert "sales_lag_1" in transformed.columns
        assert len(transformed) == len(ts_data)
