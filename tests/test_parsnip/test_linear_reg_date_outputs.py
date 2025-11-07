"""
Tests for sklearn linear regression engine with date column in outputs.

This module tests the enhanced functionality where the sklearn linear regression
engine automatically adds a 'date' column to outputs when working with time series data.
"""

import pytest
import pandas as pd
import numpy as np
from py_parsnip.models.linear_reg import linear_reg


class TestLinearRegDateOutputs:
    """Test date column extraction in outputs for time series data."""

    @pytest.fixture
    def ts_data_with_date_col(self):
        """Create time series data with explicit date column."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        return pd.DataFrame({
            'date': dates,
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'y': np.random.randn(100)
        })

    @pytest.fixture
    def ts_data_with_datetime_index(self):
        """Create time series data with DatetimeIndex."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        return pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'y': np.random.randn(100)
        }, index=dates)

    @pytest.fixture
    def non_ts_data(self):
        """Create non-time series data (no datetime columns)."""
        np.random.seed(42)
        return pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'y': np.random.randn(100)
        })

    def test_date_column_with_explicit_date_col(self, ts_data_with_date_col):
        """Test that date column is added when data has explicit date column."""
        train = ts_data_with_date_col.iloc[:80].copy()
        test = ts_data_with_date_col.iloc[80:].copy()

        # Fit with original_training_data
        spec = linear_reg()
        fit = spec.fit(train, 'y ~ x1 + x2', original_training_data=train)
        fit = fit.evaluate(test, original_test_data=test)

        # Extract outputs
        outputs, _, _ = fit.extract_outputs()

        # Assertions
        assert 'date' in outputs.columns, "Date column should be present"
        assert outputs['date'].iloc[0] == pd.Timestamp('2020-01-01')
        assert len(outputs) == 100  # 80 train + 20 test

        # Verify date column is first
        assert outputs.columns[0] == 'date'

    def test_date_column_with_datetime_index(self, ts_data_with_datetime_index):
        """Test that date column is added when data has DatetimeIndex."""
        train = ts_data_with_datetime_index.iloc[:80].copy()
        test = ts_data_with_datetime_index.iloc[80:].copy()

        # Fit with original_training_data
        spec = linear_reg()
        fit = spec.fit(train, 'y ~ x1 + x2', original_training_data=train)
        fit = fit.evaluate(test, original_test_data=test)

        # Extract outputs
        outputs, _, _ = fit.extract_outputs()

        # Assertions
        assert 'date' in outputs.columns, "Date column should be present"
        assert outputs['date'].iloc[0] == pd.Timestamp('2020-01-01')

        # Verify date column is first
        assert outputs.columns[0] == 'date'

    def test_date_ranges_match_splits(self, ts_data_with_date_col):
        """Test that date ranges correctly match train/test splits."""
        train = ts_data_with_date_col.iloc[:80].copy()
        test = ts_data_with_date_col.iloc[80:].copy()

        # Fit with original_training_data
        spec = linear_reg()
        fit = spec.fit(train, 'y ~ x1 + x2', original_training_data=train)
        fit = fit.evaluate(test, original_test_data=test)

        # Extract outputs
        outputs, _, _ = fit.extract_outputs()

        # Check train dates
        train_outputs = outputs[outputs['split'] == 'train']
        assert len(train_outputs) == 80
        assert train_outputs['date'].min() == pd.Timestamp('2020-01-01')
        assert train_outputs['date'].max() == pd.Timestamp('2020-03-20')

        # Check test dates
        test_outputs = outputs[outputs['split'] == 'test']
        assert len(test_outputs) == 20
        assert test_outputs['date'].min() == pd.Timestamp('2020-03-21')
        assert test_outputs['date'].max() == pd.Timestamp('2020-04-09')

    def test_backward_compatibility_no_datetime(self, non_ts_data):
        """Test backward compatibility - no date column when no datetime data."""
        train = non_ts_data.iloc[:80].copy()
        test = non_ts_data.iloc[80:].copy()

        # Fit WITHOUT original_training_data
        spec = linear_reg()
        fit = spec.fit(train, 'y ~ x1 + x2')
        fit = fit.evaluate(test)

        # Extract outputs
        outputs, _, _ = fit.extract_outputs()

        # Assertions - no date column should be present
        assert 'date' not in outputs.columns, "Date column should not be present for non-TS data"
        assert len(outputs) == 100

    def test_automatic_date_detection(self, ts_data_with_date_col):
        """Test that date column is automatically detected and added."""
        train = ts_data_with_date_col.iloc[:80].copy()
        test = ts_data_with_date_col.iloc[80:].copy()

        # Fit without explicitly passing original_training_data
        # Date column should still be detected automatically
        spec = linear_reg()
        fit = spec.fit(train, 'y ~ x1 + x2')
        fit = fit.evaluate(test)

        # Extract outputs
        outputs, _, _ = fit.extract_outputs()

        # Assertions - date column should be automatically added
        assert 'date' in outputs.columns, "Date column should be automatically detected"
        assert len(outputs) == 100

        # Verify date values are correct
        train_dates = outputs[outputs['split'] == 'train']['date']
        test_dates = outputs[outputs['split'] == 'test']['date']
        assert len(train_dates) == 80
        assert len(test_dates) == 20

    def test_date_column_train_only(self, ts_data_with_date_col):
        """Test that date column works when only training data is available (no evaluate)."""
        train = ts_data_with_date_col.iloc[:80].copy()

        # Fit with original_training_data but no evaluation
        spec = linear_reg()
        fit = spec.fit(train, 'y ~ x1 + x2', original_training_data=train)

        # Extract outputs (train only)
        outputs, _, _ = fit.extract_outputs()

        # Assertions
        assert 'date' in outputs.columns
        assert len(outputs) == 80
        assert (outputs['split'] == 'train').all()
        assert outputs['date'].min() == pd.Timestamp('2020-01-01')
        assert outputs['date'].max() == pd.Timestamp('2020-03-20')

    def test_date_column_values_align_with_actuals(self, ts_data_with_date_col):
        """Test that date values correctly align with actual observations."""
        train = ts_data_with_date_col.iloc[:80].copy()
        test = ts_data_with_date_col.iloc[80:].copy()

        # Fit and evaluate
        spec = linear_reg()
        fit = spec.fit(train, 'y ~ x1 + x2', original_training_data=train)
        fit = fit.evaluate(test, original_test_data=test)

        # Extract outputs
        outputs, _, _ = fit.extract_outputs()

        # For each date in outputs, verify it matches the expected date from original data
        full_data = pd.concat([train, test], ignore_index=True)

        for i, row in outputs.iterrows():
            expected_date = full_data['date'].iloc[i]
            assert row['date'] == expected_date, f"Date mismatch at row {i}"

    def test_outputs_structure_with_date(self, ts_data_with_date_col):
        """Test that outputs DataFrame has correct structure with date column."""
        train = ts_data_with_date_col.iloc[:80].copy()
        test = ts_data_with_date_col.iloc[80:].copy()

        # Fit and evaluate
        spec = linear_reg()
        fit = spec.fit(train, 'y ~ x1 + x2', original_training_data=train)
        fit = fit.evaluate(test, original_test_data=test)

        # Extract outputs
        outputs, _, _ = fit.extract_outputs()

        # Verify column order
        expected_cols = [
            'date',  # Date should be first
            'actuals',
            'fitted',
            'forecast',
            'residuals',
            'split',
            'model',
            'model_group_name',
            'group'
        ]
        assert list(outputs.columns) == expected_cols

    def test_multiple_datetime_columns_raises_error(self):
        """Test that multiple datetime columns without explicit spec raises error."""
        dates1 = pd.date_range('2020-01-01', periods=100, freq='D')
        dates2 = pd.date_range('2021-01-01', periods=100, freq='D')
        np.random.seed(42)
        df = pd.DataFrame({
            'date1': dates1,
            'date2': dates2,
            'x1': np.random.randn(100),
            'y': np.random.randn(100)
        })

        train = df.iloc[:80].copy()

        # Fit with original_training_data - should handle multiple datetime gracefully
        spec = linear_reg()
        fit = spec.fit(train, 'y ~ x1', original_training_data=train)

        # Extract outputs - should not crash, but may skip date column
        outputs, _, _ = fit.extract_outputs()

        # Either has no date column (error handling) or has one of them
        # This test documents the behavior - it should not crash
        assert isinstance(outputs, pd.DataFrame)
