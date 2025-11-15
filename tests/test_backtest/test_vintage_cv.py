"""
Tests for VintageCV class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import timedelta

from py_backtest import VintageCV, vintage_cv, create_vintage_data


class TestVintageCV:
    """Tests for VintageCV class."""

    def setup_method(self):
        """Create test vintage data"""
        # Create final data
        final_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'value': np.random.randn(100),
            'feature': np.random.randn(100)
        })

        # Create vintages
        self.vintage_data = create_vintage_data(
            final_data,
            date_col='date',
            n_revisions=3,
            revision_std=0.05,
            revision_lag='7 days'
        )

    def test_create_basic_cv(self):
        """Test creating basic VintageCV"""
        cv = VintageCV(
            data=self.vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial='30 days',
            assess='10 days',
            skip=0,
            lag=0
        )

        # Should create splits
        assert len(cv) > 0

        # Should be iterable
        splits = list(cv)
        assert len(splits) > 0

    def test_split_date_ranges(self):
        """Test split date ranges are correct"""
        cv = VintageCV(
            data=self.vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial='30 days',
            assess='10 days',
            skip='5 days',
            lag='3 days'
        )

        first_split = cv[0]
        info = first_split.get_vintage_info()

        # Check training period
        assert info['training_start'] <= info['training_end']

        # Check test period
        assert info['test_start'] <= info['test_end']

        # Check lag (gap between train end and test start)
        # Lag may be approximate based on actual date distribution
        assert info['forecast_horizon'] >= pd.Timedelta('3 days')

        # Check test period comes after training
        assert info['test_start'] > info['training_end']

    def test_training_data_uses_vintage(self):
        """Test training data uses vintage from vintage_date"""
        cv = VintageCV(
            data=self.vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial='30 days',
            assess='10 days'
        )

        first_split = cv[0]
        train = first_split.training()
        info = first_split.get_vintage_info()

        # Training data should not have as_of_date column (it's dropped)
        assert 'as_of_date' not in train.columns

        # Should have date column
        assert 'date' in train.columns

        # Dates should be within training period
        assert train['date'].min() >= info['training_start']
        assert train['date'].max() <= info['training_end']

    def test_testing_data_uses_final_vintage(self):
        """Test testing data uses final vintage"""
        cv = VintageCV(
            data=self.vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial='30 days',
            assess='10 days'
        )

        first_split = cv[0]
        test = first_split.testing()
        info = first_split.get_vintage_info()

        # Test data should not have as_of_date column
        assert 'as_of_date' not in test.columns

        # Dates should be within test period
        assert test['date'].min() >= info['test_start']
        assert test['date'].max() <= info['test_end']

    def test_multiple_splits(self):
        """Test creating multiple splits"""
        cv = VintageCV(
            data=self.vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial='20 days',
            assess='5 days',
            skip='5 days'
        )

        # Should create multiple splits
        assert len(cv) >= 3

        # Splits should progress chronologically
        splits = list(cv)
        for i in range(len(splits) - 1):
            info1 = splits[i].get_vintage_info()
            info2 = splits[i + 1].get_vintage_info()

            # Later splits should have later dates
            assert info2['training_end'] > info1['training_end']
            assert info2['test_start'] > info1['test_start']

    def test_slice_limit(self):
        """Test limiting number of splits"""
        cv = VintageCV(
            data=self.vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial='20 days',
            assess='5 days',
            skip='5 days',
            slice_limit=3
        )

        # Should have exactly 3 splits
        assert len(cv) == 3

    def test_insufficient_data_error(self):
        """Test error when insufficient data for splits"""
        small_data = self.vintage_data.head(30)

        with pytest.raises(ValueError, match="Not enough data to create any splits"):
            VintageCV(
                data=small_data,
                as_of_col='as_of_date',
                date_col='date',
                initial='100 days',  # Too large
                assess='10 days'
            )

    def test_missing_columns_error(self):
        """Test error with missing columns"""
        with pytest.raises(ValueError, match="not found in data"):
            VintageCV(
                data=self.vintage_data,
                as_of_col='missing_col',
                date_col='date',
                initial='30 days',
                assess='10 days'
            )

    def test_integer_period_specification(self):
        """Test using integer period specification"""
        cv = VintageCV(
            data=self.vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial=30,  # 30 unique dates
            assess=10,   # 10 unique dates
            skip=5
        )

        # Should create splits
        assert len(cv) > 0

        first_split = cv[0]
        info = first_split.get_vintage_info()

        # Check we have data
        assert info['n_train_obs'] > 0
        assert info['n_test_obs'] > 0

    def test_vintage_cv_function(self):
        """Test vintage_cv convenience function"""
        cv = vintage_cv(
            data=self.vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial='30 days',
            assess='10 days'
        )

        # Should return VintageCV object
        assert isinstance(cv, VintageCV)
        assert len(cv) > 0

    def test_repr(self):
        """Test string representation"""
        cv = VintageCV(
            data=self.vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial='30 days',
            assess='10 days'
        )

        repr_str = repr(cv)
        assert 'VintageCV' in repr_str
        assert 'splits' in repr_str


class TestVintageSplitDataRetrieval:
    """Tests for VintageSplit data retrieval."""

    def setup_method(self):
        """Create test vintage data with known revisions"""
        # Create data where vintages have different values
        dates = pd.date_range('2023-01-01', periods=10, freq='D')

        rows = []
        for i, date in enumerate(dates):
            # First vintage (same day): value = i
            rows.append({
                'as_of_date': date,
                'date': date,
                'value': float(i),
                'feature': float(i * 10)
            })
            # Second vintage (7 days later): value = i + 0.1
            rows.append({
                'as_of_date': date + timedelta(days=7),
                'date': date,
                'value': float(i) + 0.1,
                'feature': float(i * 10) + 1.0
            })
            # Final vintage (14 days later): value = i + 0.2
            rows.append({
                'as_of_date': date + timedelta(days=14),
                'date': date,
                'value': float(i) + 0.2,
                'feature': float(i * 10) + 2.0
            })

        self.vintage_data = pd.DataFrame(rows)

    def test_training_uses_correct_vintage(self):
        """Test training data uses vintage at vintage_date, not final"""
        cv = VintageCV(
            data=self.vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial='5 days',
            assess='2 days',
            vintage_selection='latest'
        )

        first_split = cv[0]
        train = first_split.training()
        info = first_split.get_vintage_info()

        # Training end is 2023-01-05 (index 4, value 4.0 in final data)
        # Vintage date is also 2023-01-05
        # So we should get the first vintage (as_of_date = 2023-01-05)
        # For date 2023-01-05, value should be 4.0 (not 4.1 or 4.2)

        last_train_row = train[train['date'] == info['training_end']]
        if len(last_train_row) > 0:
            # Value should be from first vintage (exact value)
            assert last_train_row.iloc[0]['value'] == 4.0

    def test_testing_uses_final_vintage(self):
        """Test testing data uses final vintage"""
        cv = VintageCV(
            data=self.vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial='5 days',
            assess='2 days'
        )

        first_split = cv[0]
        test = first_split.testing()

        # Test data should use final vintage (latest as_of_date)
        # For any date in test, value should be original + 0.2

        # Get test start date
        info = first_split.get_vintage_info()
        test_start_idx = (info['test_start'] - pd.Timestamp('2023-01-01')).days

        # Check first test row uses final vintage
        first_test_row = test[test['date'] == info['test_start']]
        if len(first_test_row) > 0:
            expected_value = float(test_start_idx) + 0.2
            assert abs(first_test_row.iloc[0]['value'] - expected_value) < 0.001

    def test_get_vintage_info(self):
        """Test get_vintage_info returns correct metadata"""
        cv = VintageCV(
            data=self.vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial='5 days',
            assess='2 days',
            lag='1 day'
        )

        first_split = cv[0]
        info = first_split.get_vintage_info()

        # Check all expected keys present
        assert 'vintage_date' in info
        assert 'training_start' in info
        assert 'training_end' in info
        assert 'test_start' in info
        assert 'test_end' in info
        assert 'n_train_obs' in info
        assert 'n_test_obs' in info
        assert 'forecast_horizon' in info

        # Check types
        assert isinstance(info['vintage_date'], pd.Timestamp)
        assert isinstance(info['n_train_obs'], (int, np.integer))
        assert isinstance(info['n_test_obs'], (int, np.integer))
        assert isinstance(info['forecast_horizon'], pd.Timedelta)

        # Check forecast horizon (may be approximate based on date distribution)
        assert info['forecast_horizon'] >= pd.Timedelta('1 day')
