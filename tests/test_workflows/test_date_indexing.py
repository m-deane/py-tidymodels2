"""
Test that extract_outputs() returns outputs indexed by date when using workflows with recipes.
"""

import pytest
import pandas as pd
import numpy as np
from py_workflows import workflow
from py_parsnip import linear_reg
from py_recipes import recipe


class TestDateIndexing:
    """Test that outputs are indexed by date."""

    @pytest.fixture
    def time_series_data(self):
        """Create time series data with date column."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'target': np.random.randn(100)
        })
        return data

    def test_workflow_with_recipe_outputs_indexed_by_date(self, time_series_data):
        """Test that extract_outputs() returns DataFrame with date as first column."""
        # Split data
        train_data = time_series_data.iloc[:80]
        test_data = time_series_data.iloc[80:]

        # Create workflow with recipe
        rec = recipe().step_normalize()
        wf = (
            workflow()
            .add_recipe(rec)
            .add_model(linear_reg().set_engine("sklearn"))
        )

        # Fit and evaluate
        fit = wf.fit(train_data)
        fit = fit.evaluate(test_data)

        # Extract outputs
        outputs, _, _ = fit.extract_outputs()

        # Verify date is first column
        assert outputs.columns[0] == 'date', \
            f"Expected 'date' as first column, got {outputs.columns[0]}"

        # Verify date column has datetime type
        assert pd.api.types.is_datetime64_any_dtype(outputs['date']), \
            f"Expected datetime type for date column, got {outputs['date'].dtype}"

        # Verify correct number of rows
        assert len(outputs) == 100, f"Expected 100 rows, got {len(outputs)}"

        # Verify split column exists
        assert 'split' in outputs.columns

        # Verify train and test data have correct date ranges
        train_outputs = outputs[outputs['split'] == 'train']
        test_outputs = outputs[outputs['split'] == 'test']

        assert len(train_outputs) == 80
        assert len(test_outputs) == 20

        # Verify dates match original data
        expected_train_dates = train_data['date'].values
        expected_test_dates = test_data['date'].values

        # Extract actual dates from column
        actual_train_dates = train_outputs['date'].values
        actual_test_dates = test_outputs['date'].values

        assert np.array_equal(actual_train_dates, expected_train_dates), \
            "Training dates don't match"
        assert np.array_equal(actual_test_dates, expected_test_dates), \
            "Test dates don't match"

    def test_workflow_with_formula_outputs_indexed_by_date(self, time_series_data):
        """Test that extract_outputs() returns DataFrame with date as first column when using formula."""
        # Split data
        train_data = time_series_data.iloc[:80]
        test_data = time_series_data.iloc[80:]

        # Create workflow with formula (auto-excludes date)
        wf = (
            workflow()
            .add_formula("target ~ x1 + x2")
            .add_model(linear_reg().set_engine("sklearn"))
        )

        # Fit and evaluate
        fit = wf.fit(train_data)
        fit = fit.evaluate(test_data)

        # Extract outputs
        outputs, _, _ = fit.extract_outputs()

        # Verify date is first column
        assert outputs.columns[0] == 'date', \
            f"Expected 'date' as first column, got {outputs.columns[0]}"

        # Verify date column has datetime type
        assert pd.api.types.is_datetime64_any_dtype(outputs['date']), \
            f"Expected datetime type for date column, got {outputs['date'].dtype}"

        # Verify correct number of rows
        assert len(outputs) == 100

    def test_direct_fit_outputs_indexed_by_date(self):
        """Test that direct model.fit() also returns date as first column."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'target': np.random.randn(100)
        })

        # Fit directly (no workflow)
        spec = linear_reg().set_engine("sklearn")
        fit = spec.fit(data, "target ~ x1 + x2")

        # Extract outputs
        outputs, _, _ = fit.extract_outputs()

        # Verify date is first column
        assert outputs.columns[0] == 'date', \
            f"Expected 'date' as first column, got {outputs.columns[0]}"

        # Verify date column has datetime type
        assert pd.api.types.is_datetime64_any_dtype(outputs['date']), \
            f"Expected datetime type for date column, got {outputs['date'].dtype}"

        # Verify dates match
        assert len(outputs) == 100
        assert np.array_equal(outputs['date'].values, dates.values)

    def test_no_date_column_returns_rangeindex(self):
        """Test that outputs without date column have RangeIndex."""
        np.random.seed(42)
        data = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'target': np.random.randn(100)
        })

        # Fit model
        spec = linear_reg().set_engine("sklearn")
        fit = spec.fit(data, "target ~ x1 + x2")

        # Extract outputs
        outputs, _, _ = fit.extract_outputs()

        # Verify RangeIndex (no date column)
        assert isinstance(outputs.index, pd.RangeIndex), \
            f"Expected RangeIndex when no date column, got {type(outputs.index)}"

        assert len(outputs) == 100
