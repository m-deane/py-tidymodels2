"""
Tests for time_series_cv function
"""

import pytest
import pandas as pd
import numpy as np

from py_rsample import time_series_cv, TimeSeriesCV


class TestTimeSeriesCV:
    """Test time_series_cv() function"""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data"""
        dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
        return pd.DataFrame({
            "date": dates,
            "value": np.arange(365),
        })

    def test_expanding_window(self, sample_data):
        """Test expanding window (cumulative=True)"""
        cv = time_series_cv(
            data=sample_data,
            date_column="date",
            initial=100,  # 100 rows initial
            assess=30,    # 30 rows assessment
            cumulative=True
        )

        assert isinstance(cv, TimeSeriesCV)
        assert len(cv) > 0

        # Check first split
        split1 = cv[0]
        train1 = split1.training()
        test1 = split1.testing()

        assert len(train1) == 100
        assert len(test1) == 30

        # Check second split (should have larger training set)
        if len(cv) > 1:
            split2 = cv[1]
            train2 = split2.training()
            test2 = split2.testing()

            # Training should grow by assess size
            assert len(train2) == 100 + 30
            assert len(test2) == 30

    def test_rolling_window(self, sample_data):
        """Test rolling window (cumulative=False)"""
        cv = time_series_cv(
            data=sample_data,
            date_column="date",
            initial=100,
            assess=30,
            cumulative=False  # Rolling window
        )

        # All training sets should be same size
        for split in cv:
            train = split.training()
            assert len(train) == 100

    def test_with_lag(self, sample_data):
        """Test with lag (forecast horizon)"""
        cv = time_series_cv(
            data=sample_data,
            date_column="date",
            initial=100,
            assess=30,
            lag=7,  # 7-day forecast horizon
            cumulative=True
        )

        # Check that there's a gap between train and test
        split1 = cv[0]
        train1 = split1.training()
        test1 = split1.testing()

        # Last train index should be 99
        # Gap: 100-106 (7 rows)
        # First test index should be 107
        assert train1["value"].iloc[-1] == 99
        assert test1["value"].iloc[0] == 107

    def test_with_skip(self, sample_data):
        """Test with skip between folds"""
        cv = time_series_cv(
            data=sample_data,
            date_column="date",
            initial=100,
            assess=30,
            skip=10,  # Skip 10 rows between folds
            cumulative=True
        )

        # Check spacing between folds
        if len(cv) > 1:
            split1 = cv[0]
            split2 = cv[1]

            train1 = split1.training()
            train2 = split2.training()

            # Second training should be (assess + skip) larger
            expected_size = len(train1) + 30 + 10
            assert len(train2) == expected_size

    def test_period_strings(self, sample_data):
        """Test with period strings"""
        cv = time_series_cv(
            data=sample_data,
            date_column="date",
            initial="3 months",  # ~90 days
            assess="1 month",     # ~30 days
            lag="7 days",
            cumulative=True
        )

        assert len(cv) > 0

        # First split should have roughly 90 days training
        split1 = cv[0]
        train1 = split1.training()
        test1 = split1.testing()

        # Check approximate sizes (within 10% due to month approximation)
        assert 80 <= len(train1) <= 100
        assert 25 <= len(test1) <= 35

    def test_iterable(self, sample_data):
        """Test that TimeSeriesCV is iterable"""
        cv = time_series_cv(
            data=sample_data,
            date_column="date",
            initial=100,
            assess=30,
            cumulative=True
        )

        # Should be iterable
        splits = list(cv)
        assert len(splits) > 0
        assert all(hasattr(s, "training") for s in splits)

    def test_indexing(self, sample_data):
        """Test indexing TimeSeriesCV"""
        cv = time_series_cv(
            data=sample_data,
            date_column="date",
            initial=100,
            assess=30,
            cumulative=True
        )

        # Should support indexing
        first_split = cv[0]
        assert hasattr(first_split, "training")

        # Should support len()
        assert len(cv) > 0

    def test_invalid_parameters(self, sample_data):
        """Test that invalid parameters raise errors"""
        # initial < 1
        with pytest.raises(ValueError, match="initial must be at least 1"):
            time_series_cv(
                data=sample_data,
                date_column="date",
                initial=0,
                assess=30
            )

        # assess < 1
        with pytest.raises(ValueError, match="assess must be at least 1"):
            time_series_cv(
                data=sample_data,
                date_column="date",
                initial=100,
                assess=0
            )

        # negative lag (caught by period parser)
        with pytest.raises(ValueError, match="Period must be positive"):
            time_series_cv(
                data=sample_data,
                date_column="date",
                initial=100,
                assess=30,
                lag=-5
            )

    def test_insufficient_data(self):
        """Test with insufficient data"""
        tiny_data = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10),
            "value": range(10)
        })

        with pytest.raises(ValueError, match="Not enough data"):
            time_series_cv(
                data=tiny_data,
                date_column="date",
                initial=50,  # More than available
                assess=10
            )

    def test_no_overlap_between_folds(self, sample_data):
        """Test that test sets don't overlap"""
        cv = time_series_cv(
            data=sample_data,
            date_column="date",
            initial=100,
            assess=30,
            skip=0,  # No skip means consecutive test sets
            cumulative=True
        )

        if len(cv) > 1:
            test_values_all = []
            for split in cv:
                test = split.testing()
                test_values = set(test["value"])

                # No overlap with previous test sets
                for prev_values in test_values_all:
                    assert len(test_values & prev_values) == 0

                test_values_all.append(test_values)

    def test_chronological_order(self, sample_data):
        """Test that splits maintain chronological order"""
        cv = time_series_cv(
            data=sample_data,
            date_column="date",
            initial=100,
            assess=30,
            cumulative=True
        )

        for split in cv:
            train = split.training()
            test = split.testing()

            # All training dates should be before all testing dates
            last_train_date = train["date"].iloc[-1]
            first_test_date = test["date"].iloc[0]

            assert last_train_date < first_test_date

    def test_repr(self, sample_data):
        """Test string representation"""
        cv = time_series_cv(
            data=sample_data,
            date_column="date",
            initial=100,
            assess=30,
            cumulative=True
        )

        repr_str = repr(cv)
        assert "expanding window" in repr_str
        assert "initial=100" in repr_str
        assert "assess=30" in repr_str

        # Test rolling window repr
        cv_rolling = time_series_cv(
            data=sample_data,
            date_column="date",
            initial=100,
            assess=30,
            cumulative=False
        )

        repr_str_rolling = repr(cv_rolling)
        assert "rolling window" in repr_str_rolling

    def test_missing_date_column(self, sample_data):
        """Test error when date column is missing"""
        with pytest.raises(ValueError, match="Date column.*not found"):
            time_series_cv(
                data=sample_data,
                date_column="missing_column",
                initial=100,
                assess=30
            )

    def test_non_datetime_column(self):
        """Test error when date column is not datetime"""
        data = pd.DataFrame({
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],  # Strings, not datetime
            "value": [1, 2, 3]
        })

        with pytest.raises(ValueError, match="must be datetime type"):
            time_series_cv(
                data=data,
                date_column="date",
                initial=1,
                assess=1
            )

    def test_slice_limit(self, sample_data):
        """Test slice_limit parameter limits number of folds"""
        # Without slice_limit - should create many folds
        cv_all = time_series_cv(
            data=sample_data,
            date_column="date",
            initial=50,
            assess=20,
            skip=10,
            cumulative=True
        )

        total_folds = len(cv_all)
        assert total_folds > 5  # Should have more than 5 folds with this data

        # With slice_limit - should only create 5 folds
        cv_limited = time_series_cv(
            data=sample_data,
            date_column="date",
            initial=50,
            assess=20,
            skip=10,
            cumulative=True,
            slice_limit=5
        )

        assert len(cv_limited) == 5
        assert len(cv_limited) < total_folds

        # Verify the limited folds are the first N folds (not random)
        for i in range(5):
            train_all = cv_all[i].training()
            train_limited = cv_limited[i].training()
            test_all = cv_all[i].testing()
            test_limited = cv_limited[i].testing()

            # Should be identical to first 5 folds from full CV
            assert len(train_all) == len(train_limited)
            assert len(test_all) == len(test_limited)
            assert train_all["value"].iloc[0] == train_limited["value"].iloc[0]

    def test_slice_limit_zero(self, sample_data):
        """Test slice_limit=0 creates no folds"""
        cv = time_series_cv(
            data=sample_data,
            date_column="date",
            initial=50,
            assess=20,
            cumulative=True,
            slice_limit=0
        )

        # slice_limit=0 should result in empty list after slicing
        assert len(cv) == 0

    def test_slice_limit_none(self, sample_data):
        """Test slice_limit=None creates all folds (default behavior)"""
        cv_none = time_series_cv(
            data=sample_data,
            date_column="date",
            initial=50,
            assess=20,
            skip=10,
            cumulative=True,
            slice_limit=None
        )

        cv_unspecified = time_series_cv(
            data=sample_data,
            date_column="date",
            initial=50,
            assess=20,
            skip=10,
            cumulative=True
        )

        # Both should have the same number of folds
        assert len(cv_none) == len(cv_unspecified)
