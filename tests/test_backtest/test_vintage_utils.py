"""
Tests for vintage utility functions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from py_backtest.vintage_utils import (
    validate_vintage_data,
    select_vintage,
    create_vintage_data
)


class TestValidateVintageData:
    """Tests for validate_vintage_data function."""

    def test_validate_valid_data(self):
        """Test validation passes with valid vintage data"""
        data = pd.DataFrame({
            'as_of_date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'value': np.random.randn(10)
        })

        # Should not raise
        validate_vintage_data(data, 'as_of_date', 'date')

    def test_missing_as_of_column(self):
        """Test error when as_of_col is missing"""
        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'value': np.random.randn(10)
        })

        with pytest.raises(ValueError, match="Vintage date column 'as_of_date' not found"):
            validate_vintage_data(data, 'as_of_date', 'date')

    def test_missing_date_column(self):
        """Test error when date_col is missing"""
        data = pd.DataFrame({
            'as_of_date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'value': np.random.randn(10)
        })

        with pytest.raises(ValueError, match="Observation date column 'date' not found"):
            validate_vintage_data(data, 'as_of_date', 'date')

    def test_non_datetime_as_of_col(self):
        """Test error when as_of_col is not datetime"""
        data = pd.DataFrame({
            'as_of_date': ['2023-01-01'] * 10,
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'value': np.random.randn(10)
        })

        with pytest.raises(ValueError, match="must be datetime type"):
            validate_vintage_data(data, 'as_of_date', 'date')

    def test_chronology_violation(self):
        """Test error when as_of_date < date"""
        data = pd.DataFrame({
            'as_of_date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'date': pd.date_range('2023-01-10', periods=10, freq='D'),  # Dates after as_of
            'value': np.random.randn(10)
        })

        with pytest.raises(ValueError, match="chronology violation"):
            validate_vintage_data(data, 'as_of_date', 'date')


class TestSelectVintage:
    """Tests for select_vintage function."""

    def setup_method(self):
        """Create test vintage data"""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')

        # Create multiple vintages per date
        rows = []
        for date in dates:
            # 3 vintages per date
            rows.append({'as_of_date': date, 'date': date, 'value': 1.0})
            rows.append({'as_of_date': date + timedelta(days=7), 'date': date, 'value': 1.1})
            rows.append({'as_of_date': date + timedelta(days=14), 'date': date, 'value': 1.2})

        self.vintage_data = pd.DataFrame(rows)

    def test_select_latest_vintage(self):
        """Test selecting latest vintage available"""
        vintage_date = pd.Timestamp('2023-01-08')

        result = select_vintage(
            self.vintage_data,
            'as_of_date',
            'date',
            vintage_date,
            vintage_selection='latest'
        )

        # Should get 5 unique dates
        assert len(result) == 5

        # For dates on or before 2023-01-01, should get as_of_date = 2023-01-08
        # (second vintage, which is 7 days after observation)
        first_row = result[result['date'] == pd.Timestamp('2023-01-01')].iloc[0]
        assert first_row['as_of_date'] == pd.Timestamp('2023-01-08')
        assert first_row['value'] == 1.1

    def test_select_exact_vintage(self):
        """Test selecting exact vintage date"""
        vintage_date = pd.Timestamp('2023-01-01')

        result = select_vintage(
            self.vintage_data,
            'as_of_date',
            'date',
            vintage_date,
            vintage_selection='exact'
        )

        # Should get only dates with exact as_of_date match
        # Only the first date has as_of_date = 2023-01-01
        assert len(result) >= 1
        assert all(result['as_of_date'] == vintage_date)

    def test_no_vintage_available(self):
        """Test error when no vintage is available"""
        vintage_date = pd.Timestamp('2022-12-01')  # Before any data

        with pytest.raises(ValueError, match="No vintage data available"):
            select_vintage(
                self.vintage_data,
                'as_of_date',
                'date',
                vintage_date,
                vintage_selection='latest'
            )

    def test_unknown_selection_strategy(self):
        """Test error with unknown vintage_selection"""
        vintage_date = pd.Timestamp('2023-01-01')

        with pytest.raises(ValueError, match="Unknown vintage_selection"):
            select_vintage(
                self.vintage_data,
                'as_of_date',
                'date',
                vintage_date,
                vintage_selection='invalid'
            )


class TestCreateVintageData:
    """Tests for create_vintage_data function."""

    def test_create_basic_vintage_data(self):
        """Test creating basic vintage data"""
        final_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'value': np.arange(10, dtype=float)
        })

        vintage_df = create_vintage_data(
            final_data,
            date_col='date',
            n_revisions=3,
            revision_std=0.05,
            revision_lag='1 day'
        )

        # Should have 3 vintages per observation
        assert len(vintage_df) == 10 * 3

        # Should have as_of_date column
        assert 'as_of_date' in vintage_df.columns

        # as_of_date should be >= date
        assert all(vintage_df['as_of_date'] >= vintage_df['date'])

    def test_vintage_data_structure(self):
        """Test vintage data has correct structure"""
        final_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'value': np.arange(5, dtype=float)
        })

        vintage_df = create_vintage_data(
            final_data,
            date_col='date',
            n_revisions=2,
            revision_std=0.1,
            revision_lag='7 days'
        )

        # Check column order
        assert vintage_df.columns[0] == 'as_of_date'
        assert vintage_df.columns[1] == 'date'

        # Check sorted
        assert vintage_df['date'].is_monotonic_increasing

    def test_final_vintage_no_noise(self):
        """Test final vintage matches original data (no noise)"""
        np.random.seed(42)

        final_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'value': np.arange(5, dtype=float)
        })

        vintage_df = create_vintage_data(
            final_data,
            date_col='date',
            n_revisions=3,
            revision_std=0.1,
            revision_lag='1 day'
        )

        # Get final vintages (latest as_of_date for each date)
        final_vintages = vintage_df.groupby('date').tail(1)

        # Final vintage should match original (within floating point tolerance)
        assert np.allclose(final_vintages['value'].values, final_data['value'].values)

    def test_invalid_n_revisions(self):
        """Test error with invalid n_revisions"""
        final_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'value': np.arange(5, dtype=float)
        })

        with pytest.raises(ValueError, match="n_revisions must be at least 1"):
            create_vintage_data(final_data, 'date', n_revisions=0)

    def test_missing_date_column(self):
        """Test error when date_col is missing"""
        final_data = pd.DataFrame({
            'value': np.arange(5, dtype=float)
        })

        with pytest.raises(ValueError, match="Date column 'date' not found"):
            create_vintage_data(final_data, 'date', n_revisions=2)
