"""
Tests for initial_time_split function
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from py_rsample import initial_time_split, RSplit


class TestInitialTimeSplit:
    """Test initial_time_split() function"""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data"""
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        return pd.DataFrame({
            "date": dates,
            "value": np.arange(100),
        })

    def test_basic_split(self, sample_data):
        """Test basic 75/25 split"""
        split = initial_time_split(sample_data, prop=0.75)

        assert isinstance(split, RSplit)

        train = split.training()
        test = split.testing()

        # Should have 75 train, 25 test
        assert len(train) == 75
        assert len(test) == 25

        # Training should be first 75 rows
        assert train["value"].iloc[0] == 0
        assert train["value"].iloc[-1] == 74

        # Testing should be last 25 rows
        assert test["value"].iloc[0] == 75
        assert test["value"].iloc[-1] == 99

    def test_default_prop(self, sample_data):
        """Test default proportion (0.75)"""
        split = initial_time_split(sample_data)

        train = split.training()
        test = split.testing()

        assert len(train) == 75
        assert len(test) == 25

    def test_different_proportions(self, sample_data):
        """Test different train proportions"""
        # 80/20 split
        split = initial_time_split(sample_data, prop=0.8)
        train = split.training()
        test = split.testing()

        assert len(train) == 80
        assert len(test) == 20

        # 60/40 split
        split = initial_time_split(sample_data, prop=0.6)
        train = split.training()
        test = split.testing()

        assert len(train) == 60
        assert len(test) == 40

    def test_integer_lag(self, sample_data):
        """Test with integer lag (forecast horizon)"""
        split = initial_time_split(sample_data, prop=0.75, lag=5)

        train = split.training()
        test = split.testing()

        # Train: 0-74 (75 rows)
        # Lag: 75-79 (5 rows, excluded)
        # Test: 80-99 (20 rows)
        assert len(train) == 75
        assert len(test) == 20

        # Check gap
        assert train["value"].iloc[-1] == 74
        assert test["value"].iloc[0] == 80

    def test_period_lag(self, sample_data):
        """Test with period string lag"""
        split = initial_time_split(
            sample_data,
            prop=0.75,
            lag="7 days",
            date_column="date"
        )

        train = split.training()
        test = split.testing()

        # Should have 7-day gap
        assert len(train) == 75
        # 100 - 75 - 7 = 18
        assert len(test) == 18

    def test_invalid_prop(self, sample_data):
        """Test that invalid proportions raise errors"""
        with pytest.raises(ValueError, match="prop must be between 0 and 1"):
            initial_time_split(sample_data, prop=1.5)

        with pytest.raises(ValueError, match="prop must be between 0 and 1"):
            initial_time_split(sample_data, prop=0)

        with pytest.raises(ValueError, match="prop must be between 0 and 1"):
            initial_time_split(sample_data, prop=-0.5)

    def test_insufficient_data(self):
        """Test with insufficient data"""
        tiny_data = pd.DataFrame({"value": [1]})

        with pytest.raises(ValueError, match="at least 2 rows"):
            initial_time_split(tiny_data)

    def test_lag_too_large(self, sample_data):
        """Test when lag is too large for available data"""
        with pytest.raises(ValueError, match="Not enough data for test set"):
            initial_time_split(sample_data, prop=0.75, lag=30)

    def test_rsplit_methods(self, sample_data):
        """Test RSplit methods"""
        split = initial_time_split(sample_data, prop=0.75)

        # training() and analysis() should be identical
        assert split.training().equals(split.analysis())

        # testing() and assessment() should be identical
        assert split.testing().equals(split.assessment())

    def test_chronological_order(self, sample_data):
        """Test that split maintains chronological order"""
        split = initial_time_split(sample_data, prop=0.8)

        train = split.training()
        test = split.testing()

        # All training dates should be before all testing dates
        last_train_date = train["date"].iloc[-1]
        first_test_date = test["date"].iloc[0]

        assert last_train_date < first_test_date

    def test_no_data_leakage(self, sample_data):
        """Test that there's no data leakage between train and test"""
        split = initial_time_split(sample_data, prop=0.75)

        train = split.training()
        test = split.testing()

        # No overlap in indices
        train_values = set(train["value"])
        test_values = set(test["value"])

        assert len(train_values & test_values) == 0

    def test_period_lag_requires_date_column(self, sample_data):
        """Test that period lag requires date_column"""
        with pytest.raises(ValueError, match="date_column required"):
            initial_time_split(sample_data, prop=0.75, lag="1 month")

    def test_explicit_absolute_dates(self, sample_data):
        """Test explicit absolute date ranges"""
        split = initial_time_split(
            sample_data,
            date_column="date",
            train_start="2020-01-01",
            train_end="2020-02-29",
            test_start="2020-03-01",
            test_end="2020-03-31"
        )

        train = split.training()
        test = split.testing()

        # Training: Jan 1 - Feb 29 (60 days in 2020, leap year)
        assert train["date"].min() == pd.Timestamp("2020-01-01")
        assert train["date"].max() == pd.Timestamp("2020-02-29")
        assert len(train) == 60

        # Testing: Mar 1 - Mar 31 (31 days)
        assert test["date"].min() == pd.Timestamp("2020-03-01")
        assert test["date"].max() == pd.Timestamp("2020-03-31")
        assert len(test) == 31

    def test_explicit_relative_dates_from_start(self, sample_data):
        """Test relative dates from data start"""
        split = initial_time_split(
            sample_data,
            date_column="date",
            train_start="start",
            train_end="start + 30 days",
            test_start="start + 37 days",  # 7-day gap
            test_end="start + 60 days"
        )

        train = split.training()
        test = split.testing()

        # Check training set
        assert train["date"].min() == sample_data["date"].min()
        assert len(train) == 31  # Days 0-30 inclusive

        # Check 7-day gap
        last_train = train["date"].max()
        first_test = test["date"].min()
        gap_days = (first_test - last_train).days
        assert gap_days == 7

        # Check test set
        assert len(test) == 24  # Days 37-60 inclusive

    def test_explicit_relative_dates_from_end(self, sample_data):
        """Test relative dates from data end"""
        split = initial_time_split(
            sample_data,
            date_column="date",
            train_start="end - 50 days",
            train_end="end - 25 days",
            test_start="end - 20 days",
            test_end="end"
        )

        train = split.training()
        test = split.testing()

        # Should have 26 days of training (from -50 to -25 inclusive)
        assert len(train) == 26

        # Should have 21 days of testing (from -20 to 0 inclusive)
        assert len(test) == 21

        # Last test date should be last data date
        assert test["date"].max() == sample_data["date"].max()

    def test_explicit_mixed_absolute_and_relative(self, sample_data):
        """Test mix of absolute and relative dates"""
        split = initial_time_split(
            sample_data,
            date_column="date",
            train_start="2020-01-01",
            train_end="2020-02-01",
            test_start="end - 10 days",
            test_end="end"
        )

        train = split.training()
        test = split.testing()

        # Training: absolute dates
        assert train["date"].min() == pd.Timestamp("2020-01-01")
        assert train["date"].max() == pd.Timestamp("2020-02-01")

        # Testing: relative to end
        assert test["date"].max() == sample_data["date"].max()
        assert len(test) == 11  # Last 11 days (inclusive)

    def test_explicit_defaults(self, sample_data):
        """Test that defaults work for explicit mode"""
        # Only specify train_end - should default train_start to "start"
        # and test_end to "end"
        split = initial_time_split(
            sample_data,
            date_column="date",
            train_end="start + 50 days"
        )

        train = split.training()
        test = split.testing()

        # Training should start from data start
        assert train["date"].min() == sample_data["date"].min()
        assert len(train) == 51  # Days 0-50

        # Testing should start after training and go to end
        assert test["date"].min() > train["date"].max()
        assert test["date"].max() == sample_data["date"].max()

    def test_explicit_dates_require_date_column(self, sample_data):
        """Test that explicit dates require date_column"""
        with pytest.raises(ValueError, match="date_column required"):
            initial_time_split(
                sample_data,
                train_start="2020-01-01",
                train_end="2020-02-01"
            )

    def test_explicit_dates_invalid_order(self, sample_data):
        """Test that train_start must be before train_end"""
        with pytest.raises(ValueError, match="must be before"):
            initial_time_split(
                sample_data,
                date_column="date",
                train_start="2020-02-01",
                train_end="2020-01-01",
                test_start="2020-03-01",
                test_end="2020-03-31"
            )

    def test_explicit_dates_test_before_train(self, sample_data):
        """Test that test must come after train"""
        with pytest.raises(ValueError, match="must be after"):
            initial_time_split(
                sample_data,
                date_column="date",
                train_start="2020-02-01",
                train_end="2020-03-01",
                test_start="2020-02-15",  # Overlaps with training
                test_end="2020-03-31"
            )

    def test_explicit_period_strings(self, sample_data):
        """Test explicit dates with period strings"""
        split = initial_time_split(
            sample_data,
            date_column="date",
            train_start="start",
            train_end="start + 2 months",
            test_start="start + 67 days",  # 2 months + 1 week
            test_end="start + 90 days"     # ~3 months
        )

        train = split.training()
        test = split.testing()

        # Training should be roughly 2 months
        assert 58 <= len(train) <= 62  # ~60 days

        # Gap should be roughly 1 week
        last_train = train["date"].max()
        first_test = test["date"].min()
        gap_days = (first_test - last_train).days
        assert 6 <= gap_days <= 8  # ~7 days

        # Test should be roughly 24 days (from day 67 to day 90)
        assert 22 <= len(test) <= 26  # ~24 days
