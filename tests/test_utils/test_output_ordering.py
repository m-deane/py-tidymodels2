"""
Tests for output DataFrame column ordering utilities.

Ensures consistent column ordering across all extract_outputs() implementations.
"""

import pytest
import pandas as pd
import numpy as np
from py_parsnip.utils.output_ordering import (
    reorder_outputs_columns,
    reorder_coefficients_columns,
    reorder_stats_columns
)


class TestReorderOutputsColumns:
    """Test reorder_outputs_columns() function"""

    def test_basic_ordering_no_group(self):
        """Test basic column ordering without group column"""
        df = pd.DataFrame({
            'model': ['linear_reg'] * 3,
            'split': ['train', 'train', 'train'],
            'residuals': [0.1, -0.2, 0.3],
            'actuals': [1.0, 2.0, 3.0],
            'fitted': [0.9, 2.2, 2.7],
            'forecast': [0.9, 2.2, 2.7],
        })

        result = reorder_outputs_columns(df, group_col=None)

        # Expected order: actuals, fitted, forecast, residuals, split, model
        expected_order = ['actuals', 'fitted', 'forecast', 'residuals', 'split', 'model']
        assert list(result.columns) == expected_order

    def test_date_column_first(self):
        """Test that date column is always first"""
        df = pd.DataFrame({
            'split': ['train', 'train', 'train'],
            'actuals': [1.0, 2.0, 3.0],
            'date': pd.date_range('2020-01-01', periods=3),
            'fitted': [0.9, 2.2, 2.7],
            'residuals': [0.1, -0.2, 0.3],
        })

        result = reorder_outputs_columns(df, group_col=None)

        # Date should be first
        assert result.columns[0] == 'date'
        # Core columns should follow
        assert 'actuals' in result.columns[:5]
        assert 'fitted' in result.columns[:5]

    def test_group_column_second(self):
        """Test that group column is second (after date)"""
        df = pd.DataFrame({
            'split': ['train', 'train', 'train'],
            'store_id': ['A', 'A', 'A'],
            'actuals': [1.0, 2.0, 3.0],
            'date': pd.date_range('2020-01-01', periods=3),
            'fitted': [0.9, 2.2, 2.7],
            'residuals': [0.1, -0.2, 0.3],
        })

        result = reorder_outputs_columns(df, group_col='store_id')

        # Date first, group second
        assert result.columns[0] == 'date'
        assert result.columns[1] == 'store_id'
        # Core columns should follow
        assert 'actuals' in result.columns[2:7]
        assert 'fitted' in result.columns[2:7]

    def test_group_column_without_date(self):
        """Test group column first when no date column"""
        df = pd.DataFrame({
            'split': ['train', 'train', 'train'],
            'store_id': ['A', 'A', 'A'],
            'actuals': [1.0, 2.0, 3.0],
            'fitted': [0.9, 2.2, 2.7],
            'residuals': [0.1, -0.2, 0.3],
        })

        result = reorder_outputs_columns(df, group_col='store_id')

        # Group column should be first (no date)
        # But wait - looking at the function, group comes AFTER date
        # So without date, group is still second position
        # Actually, let me check the implementation...
        # The function puts date first if present, then group
        # So if no date, other columns come first

        # Actually, re-reading the implementation:
        # 1. date (if present)
        # 2. group_col (if present)
        # 3. core columns

        # So without date, group_col should be first
        # Wait, let me trace through the logic:
        # - ordered_cols = []
        # - if 'date' in df.columns: ordered_cols.append('date')  # Skipped (no date)
        # - if group_col and group_col in df.columns: ordered_cols.append(group_col)  # Added
        # - for col in core_cols: ...  # Then core columns

        # So yes, group should be first when no date
        # But that's not what the docstring says! Let me re-read...

        # Docstring says:
        # 1. 'date' (always first if present)
        # 2. group_col (e.g., 'country', 'store_id') - second if present

        # This implies group is ONLY second when date is present
        # When date is absent, group would naturally be first based on the code logic

        # Let's test what actually happens
        assert result.columns[0] == 'store_id'  # Group first when no date
        assert 'actuals' in result.columns[1:6]
        assert 'fitted' in result.columns[1:6]

    def test_metadata_columns_last(self):
        """Test that metadata columns come after core columns"""
        df = pd.DataFrame({
            'model_group_name': ['group1'] * 3,
            'date': pd.date_range('2020-01-01', periods=3),
            'model': ['linear_reg'] * 3,
            'split': ['train', 'train', 'train'],
            'group': ['global'] * 3,
            'actuals': [1.0, 2.0, 3.0],
            'fitted': [0.9, 2.2, 2.7],
            'forecast': [0.9, 2.2, 2.7],
            'residuals': [0.1, -0.2, 0.3],
        })

        result = reorder_outputs_columns(df, group_col=None)

        # Date first
        assert result.columns[0] == 'date'
        # Core columns (actuals, fitted, forecast, residuals, split)
        core_end_idx = 6  # date + 5 core columns
        assert set(result.columns[1:core_end_idx]) == {'actuals', 'fitted', 'forecast', 'residuals', 'split'}
        # Metadata columns last
        assert set(result.columns[core_end_idx:]) == {'model', 'model_group_name', 'group'}

    def test_empty_dataframe(self):
        """Test that empty DataFrame is handled gracefully"""
        df = pd.DataFrame()
        result = reorder_outputs_columns(df, group_col='store_id')
        assert result.empty
        assert len(result.columns) == 0

    def test_extra_columns_preserved(self):
        """Test that extra columns are preserved at the end"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=3),
            'custom_col': ['x', 'y', 'z'],
            'actuals': [1.0, 2.0, 3.0],
            'fitted': [0.9, 2.2, 2.7],
            'another_custom': [10, 20, 30],
            'split': ['train', 'train', 'train'],
        })

        result = reorder_outputs_columns(df, group_col=None)

        # Date first
        assert result.columns[0] == 'date'
        # Core columns next
        assert 'actuals' in result.columns[:5]
        # Custom columns preserved at end
        assert 'custom_col' in result.columns
        assert 'another_custom' in result.columns

    def test_date_in_index(self):
        """Test that date in index is reset to first column"""
        df = pd.DataFrame({
            'actuals': [1.0, 2.0, 3.0],
            'fitted': [0.9, 2.2, 2.7],
            'split': ['train', 'train', 'train'],
        }, index=pd.date_range('2020-01-01', periods=3))

        result = reorder_outputs_columns(df, group_col=None)

        # Date should be reset to first column
        assert result.columns[0] == 'date'
        # Should be a column, not index
        assert 'date' in result.columns
        # Original columns should follow
        assert 'actuals' in result.columns[1:5]

    def test_named_date_index(self):
        """Test that named date index is handled correctly"""
        df = pd.DataFrame({
            'actuals': [1.0, 2.0, 3.0],
            'fitted': [0.9, 2.2, 2.7],
            'split': ['train', 'train', 'train'],
        })
        df.index = pd.date_range('2020-01-01', periods=3)
        df.index.name = 'date'

        result = reorder_outputs_columns(df, group_col=None)

        # Date should be first column
        assert result.columns[0] == 'date'
        # Should have all original columns plus date
        assert 'actuals' in result.columns
        assert 'fitted' in result.columns
        assert 'split' in result.columns


class TestReorderCoefficientsColumns:
    """Test reorder_coefficients_columns() function"""

    def test_basic_ordering_no_group(self):
        """Test basic coefficient column ordering"""
        df = pd.DataFrame({
            'vif': [1.5, 2.3],
            'p_value': [0.01, 0.05],
            'coefficient': [0.5, -0.3],
            'variable': ['x1', 'x2'],
            'std_error': [0.1, 0.15],
        })

        result = reorder_coefficients_columns(df, group_col=None)

        # Expected order: variable, coefficient, std_error, p_value, vif
        expected_start = ['variable', 'coefficient', 'std_error']
        assert list(result.columns[:3]) == expected_start
        assert 'p_value' in result.columns
        assert 'vif' in result.columns

    def test_group_column_first(self):
        """Test that group column is first in coefficients"""
        df = pd.DataFrame({
            'coefficient': [0.5, -0.3],
            'country': ['USA', 'USA'],
            'variable': ['x1', 'x2'],
            'p_value': [0.01, 0.05],
        })

        result = reorder_coefficients_columns(df, group_col='country')

        # Group should be first
        assert result.columns[0] == 'country'
        # Core columns follow
        assert result.columns[1] == 'variable'
        assert result.columns[2] == 'coefficient'

    def test_confidence_intervals_ordered(self):
        """Test confidence interval columns are properly ordered"""
        df = pd.DataFrame({
            'conf_high': [0.8, 0.2],
            'variable': ['x1', 'x2'],
            'conf_low': [0.2, -0.8],
            'coefficient': [0.5, -0.3],
        })

        result = reorder_coefficients_columns(df, group_col=None)

        # conf_low and conf_high should be after main stats
        conf_low_idx = list(result.columns).index('conf_low')
        conf_high_idx = list(result.columns).index('conf_high')
        coef_idx = list(result.columns).index('coefficient')

        assert conf_low_idx > coef_idx
        assert conf_high_idx > coef_idx


class TestReorderStatsColumns:
    """Test reorder_stats_columns() function"""

    def test_basic_ordering_no_group(self):
        """Test basic stats column ordering"""
        df = pd.DataFrame({
            'value': [0.5, 0.8, 1.2],
            'model': ['linear_reg'] * 3,
            'metric': ['rmse', 'mae', 'r_squared'],
            'split': ['train', 'train', 'train'],
        })

        result = reorder_stats_columns(df, group_col=None)

        # Expected order: split, metric, value, then metadata
        expected_start = ['split', 'metric', 'value']
        assert list(result.columns[:3]) == expected_start
        assert result.columns[3] == 'model'

    def test_group_column_first(self):
        """Test that group column is first in stats"""
        df = pd.DataFrame({
            'value': [0.5, 0.8],
            'store_id': ['A', 'B'],
            'metric': ['rmse', 'rmse'],
            'split': ['train', 'train'],
        })

        result = reorder_stats_columns(df, group_col='store_id')

        # Group should be first
        assert result.columns[0] == 'store_id'
        # Core columns follow
        assert result.columns[1] == 'split'
        assert result.columns[2] == 'metric'
        assert result.columns[3] == 'value'


class TestIntegrationWithRealData:
    """Test ordering with realistic nested workflow outputs"""

    def test_nested_workflow_outputs(self):
        """Test realistic nested workflow outputs structure"""
        # Simulate what NestedWorkflowFit.extract_outputs() produces
        outputs = pd.DataFrame({
            'residuals': [0.1, -0.2, 0.3, 0.15],
            'country': ['USA', 'USA', 'UK', 'UK'],
            'fitted': [0.9, 2.2, 1.8, 3.1],
            'date': pd.date_range('2020-01-01', periods=4),
            'model': ['linear_reg'] * 4,
            'split': ['train', 'test', 'train', 'test'],
            'actuals': [1.0, 2.0, 2.0, 3.0],
            'forecast': [0.9, 2.2, 1.8, 3.1],
            'group': ['global'] * 4,
        })

        result = reorder_outputs_columns(outputs, group_col='country')

        # Verify strict ordering
        assert result.columns[0] == 'date'
        assert result.columns[1] == 'country'
        assert result.columns[2] == 'actuals'
        assert result.columns[3] == 'fitted'
        assert result.columns[4] == 'forecast'
        assert result.columns[5] == 'residuals'
        assert result.columns[6] == 'split'
        # Metadata last
        assert 'model' in result.columns[7:]
        assert 'group' in result.columns[7:]

    def test_non_nested_workflow_outputs(self):
        """Test realistic non-nested workflow outputs structure"""
        # Simulate what WorkflowFit.extract_outputs() produces (no group_col)
        outputs = pd.DataFrame({
            'residuals': [0.1, -0.2, 0.3],
            'fitted': [0.9, 2.2, 2.7],
            'date': pd.date_range('2020-01-01', periods=3),
            'model': ['linear_reg'] * 3,
            'split': ['train', 'train', 'train'],
            'actuals': [1.0, 2.0, 3.0],
            'forecast': [0.9, 2.2, 2.7],
        })

        result = reorder_outputs_columns(outputs, group_col=None)

        # Verify ordering (no group column)
        assert result.columns[0] == 'date'
        assert result.columns[1] == 'actuals'
        assert result.columns[2] == 'fitted'
        assert result.columns[3] == 'forecast'
        assert result.columns[4] == 'residuals'
        assert result.columns[5] == 'split'
        assert result.columns[6] == 'model'
