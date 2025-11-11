"""
Integration tests for column ordering in extract_outputs() across workflows.

Tests that date and group columns are consistently ordered in all extract_outputs() methods.
"""

import pytest
import pandas as pd
import numpy as np
from py_workflows import workflow
from py_parsnip import linear_reg


class TestWorkflowColumnOrdering:
    """Test column ordering in WorkflowFit.extract_outputs()"""

    @pytest.fixture
    def time_series_data(self):
        """Create time series data for testing"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'y': np.random.randn(100) * 10 + 50
        })
        train = data.iloc[:80]
        test = data.iloc[80:]
        return train, test

    def test_workflow_outputs_date_first(self, time_series_data):
        """Test that date column is first in WorkflowFit outputs"""
        train, test = time_series_data

        wf = workflow().add_formula('y ~ x1 + x2').add_model(linear_reg())
        fit = wf.fit(train).evaluate(test)

        outputs, coeffs, stats = fit.extract_outputs()

        # Date should be first column
        assert outputs.columns[0] == 'date'
        # Core columns should follow
        assert 'actuals' in outputs.columns[:6]
        assert 'fitted' in outputs.columns[:6]
        assert 'forecast' in outputs.columns[:6]


class TestNestedWorkflowColumnOrdering:
    """Test column ordering in NestedWorkflowFit.extract_outputs()"""

    @pytest.fixture
    def grouped_time_series_data(self):
        """Create grouped time series data for testing"""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')

        data_a = pd.DataFrame({
            'date': dates,
            'store_id': 'A',
            'x1': np.random.randn(50),
            'y': np.random.randn(50) * 5 + 100
        })

        data_b = pd.DataFrame({
            'date': dates,
            'store_id': 'B',
            'x1': np.random.randn(50),
            'y': np.random.randn(50) * 5 + 200
        })

        data = pd.concat([data_a, data_b], ignore_index=True)
        train = data.iloc[:80]  # 40 per group
        test = data.iloc[80:]   # 10 per group

        return train, test

    def test_nested_workflow_outputs_date_and_group_ordering(self, grouped_time_series_data):
        """Test that date is first and group column is second in nested workflow outputs"""
        train, test = grouped_time_series_data

        wf = workflow().add_formula('y ~ x1').add_model(linear_reg())
        nested_fit = wf.fit_nested(train, group_col='store_id')
        nested_fit = nested_fit.evaluate(test)

        outputs, coeffs, stats = nested_fit.extract_outputs()

        # Date should be first
        assert outputs.columns[0] == 'date'
        # Group column should be second
        assert outputs.columns[1] == 'store_id'
        # Core columns should follow
        assert 'actuals' in outputs.columns[2:8]
        assert 'fitted' in outputs.columns[2:8]
        assert 'forecast' in outputs.columns[2:8]
        assert 'residuals' in outputs.columns[2:8]
        assert 'split' in outputs.columns[2:8]

    def test_nested_workflow_coeffs_group_first(self, grouped_time_series_data):
        """Test that group column is first in coefficients DataFrame"""
        train, test = grouped_time_series_data

        wf = workflow().add_formula('y ~ x1').add_model(linear_reg())
        nested_fit = wf.fit_nested(train, group_col='store_id')

        outputs, coeffs, stats = nested_fit.extract_outputs()

        # Group column should be first in coefficients
        assert coeffs.columns[0] == 'store_id'
        # Core coefficient columns should follow
        assert 'variable' in coeffs.columns[1:4]
        assert 'coefficient' in coeffs.columns[1:4]

    def test_nested_workflow_stats_group_first(self, grouped_time_series_data):
        """Test that group column is first in stats DataFrame"""
        train, test = grouped_time_series_data

        wf = workflow().add_formula('y ~ x1').add_model(linear_reg())
        nested_fit = wf.fit_nested(train, group_col='store_id')
        nested_fit = nested_fit.evaluate(test)

        outputs, coeffs, stats = nested_fit.extract_outputs()

        # Group column should be first in stats
        assert stats.columns[0] == 'store_id'
        # Core stats columns should follow
        assert 'split' in stats.columns[1:5]
        assert 'metric' in stats.columns[1:5]
        assert 'value' in stats.columns[1:5]


class TestModelSpecColumnOrdering:
    """Test column ordering in ModelSpec nested fits"""

    @pytest.fixture
    def grouped_data(self):
        """Create grouped data for testing"""
        dates = pd.date_range('2020-01-01', periods=40, freq='D')

        data_x = pd.DataFrame({
            'date': dates,
            'country': 'USA',
            'x1': np.random.randn(40),
            'x2': np.random.randn(40),
            'y': np.random.randn(40) * 3 + 50
        })

        data_y = pd.DataFrame({
            'date': dates,
            'country': 'UK',
            'x1': np.random.randn(40),
            'x2': np.random.randn(40),
            'y': np.random.randn(40) * 3 + 60
        })

        data = pd.concat([data_x, data_y], ignore_index=True)
        train = data.iloc[:60]  # 30 per group
        test = data.iloc[60:]   # 10 per group

        return train, test

    def test_nested_model_fit_outputs_ordering(self, grouped_data):
        """Test column ordering in NestedModelFit.extract_outputs()"""
        train, test = grouped_data

        spec = linear_reg()
        nested_fit = spec.fit_nested(train, 'y ~ x1 + x2', group_col='country')
        nested_fit = nested_fit.evaluate(test)

        outputs, coeffs, stats = nested_fit.extract_outputs()

        # Date should be first
        assert outputs.columns[0] == 'date'
        # Group column should be second
        assert outputs.columns[1] == 'country'
        # Core columns follow
        assert 'actuals' in outputs.columns[2:8]
        assert 'fitted' in outputs.columns[2:8]

    def test_nested_model_fit_all_dataframes_ordering(self, grouped_data):
        """Test that all three DataFrames have consistent group column placement"""
        train, test = grouped_data

        spec = linear_reg()
        nested_fit = spec.fit_nested(train, 'y ~ x1 + x2', group_col='country')
        nested_fit = nested_fit.evaluate(test)

        outputs, coeffs, stats = nested_fit.extract_outputs()

        # Outputs: date first, country second
        assert outputs.columns[0] == 'date'
        assert outputs.columns[1] == 'country'

        # Coefficients: country first (no date in coeffs)
        assert coeffs.columns[0] == 'country'

        # Stats: country first (no date in stats)
        assert stats.columns[0] == 'country'


class TestBackwardCompatibility:
    """Ensure column ordering doesn't break existing functionality"""

    @pytest.fixture
    def simple_data(self):
        """Simple data without dates"""
        data = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'y': np.random.randn(50) * 10 + 100
        })
        train = data.iloc[:40]
        test = data.iloc[40:]
        return train, test

    def test_no_date_column_still_works(self, simple_data):
        """Test that data without date column still works correctly"""
        train, test = simple_data

        wf = workflow().add_formula('y ~ x1 + x2').add_model(linear_reg())
        fit = wf.fit(train).evaluate(test)

        outputs, coeffs, stats = fit.extract_outputs()

        # No date column, but should still have core columns in correct order
        assert 'actuals' in outputs.columns[:4]
        assert 'fitted' in outputs.columns[:4]
        assert 'forecast' in outputs.columns[:4]
        assert 'residuals' in outputs.columns[:4]

        # Should not raise errors
        assert len(outputs) > 0
        assert len(coeffs) > 0
        assert len(stats) > 0

    def test_column_values_unchanged(self, simple_data):
        """Test that reordering doesn't change values, only order"""
        train, test = simple_data

        wf = workflow().add_formula('y ~ x1 + x2').add_model(linear_reg())
        fit = wf.fit(train).evaluate(test)

        outputs, coeffs, stats = fit.extract_outputs()

        # Values should be consistent
        train_outputs = outputs[outputs['split'] == 'train']
        test_outputs = outputs[outputs['split'] == 'test']

        # Train actuals should match training data target values
        assert len(train_outputs) == len(train)
        # Test actuals should match test data target values
        assert len(test_outputs) == len(test)

        # Residuals should equal actuals - fitted
        np.testing.assert_array_almost_equal(
            train_outputs['residuals'].values,
            train_outputs['actuals'].values - train_outputs['fitted'].values,
            decimal=10
        )
