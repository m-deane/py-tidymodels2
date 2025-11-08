"""
Tests for column name space validation in formulas
"""

import pytest
import pandas as pd
import numpy as np

from py_hardhat.mold import mold


class TestColumnSpaceValidation:
    """Test that column names with spaces trigger helpful error messages"""

    @pytest.fixture
    def data_with_spaces(self):
        """Create data with column names containing spaces"""
        return pd.DataFrame({
            'column with spaces': [1, 2, 3, 4, 5],
            'another bad name': [10, 20, 30, 40, 50],
            'good_name': [100, 200, 300, 400, 500],
            'y': [1, 2, 3, 4, 5]
        })

    def test_space_in_column_name_error(self, data_with_spaces):
        """Column names with spaces should raise clear error when used in formula"""
        with pytest.raises(ValueError) as exc_info:
            mold("y ~ .", data_with_spaces)

        error_msg = str(exc_info.value)
        assert "Column names used in formula cannot contain spaces" in error_msg
        assert "column with spaces" in error_msg or "Found 2 invalid column(s)" in error_msg
        assert "rename" in error_msg.lower()

    def test_no_error_with_valid_names(self):
        """Column names without spaces should work fine"""
        data = pd.DataFrame({
            'good_name1': [1, 2, 3, 4, 5],
            'good_name2': [10, 20, 30, 40, 50],
            'y': [1, 2, 3, 4, 5]
        })

        # Should not raise
        result = mold("y ~ .", data)
        assert result is not None

    def test_error_shows_fix_suggestion(self, data_with_spaces):
        """Error message should suggest how to fix the issue"""
        with pytest.raises(ValueError) as exc_info:
            mold("y ~ .", data_with_spaces)

        error_msg = str(exc_info.value)
        assert "str.replace(' ', '_')" in error_msg or "rename(columns=" in error_msg

    def test_single_space_column(self):
        """Test with single problematic column"""
        data = pd.DataFrame({
            'bad column': [1, 2, 3],
            'x': [10, 20, 30],
            'y': [100, 200, 300]
        })

        with pytest.raises(ValueError) as exc_info:
            mold("y ~ .", data)

        error_msg = str(exc_info.value)
        assert "bad column" in error_msg

    def test_multiple_spaces_in_name(self):
        """Test with column containing multiple spaces"""
        data = pd.DataFrame({
            'very  bad   column   name': [1, 2, 3],
            'y': [100, 200, 300]
        })

        with pytest.raises(ValueError) as exc_info:
            mold("y ~ .", data)

        error_msg = str(exc_info.value)
        assert "Column names used in formula cannot contain spaces" in error_msg

    def test_space_in_outcome_column(self):
        """Test when outcome column has space"""
        data = pd.DataFrame({
            'target variable': [1, 2, 3],
            'x': [10, 20, 30]
        })

        with pytest.raises(ValueError) as exc_info:
            mold("target variable ~ x", data)

        error_msg = str(exc_info.value)
        assert "Column names used in formula cannot contain spaces" in error_msg

    def test_unused_column_with_space_no_error(self):
        """Columns with spaces that are NOT used in formula should not cause error"""
        data = pd.DataFrame({
            'column with spaces': [1, 2, 3, 4, 5],
            'another bad name': [10, 20, 30, 40, 50],
            'x1': [100, 200, 300, 400, 500],
            'x2': [1000, 2000, 3000, 4000, 5000],
            'y': [1, 2, 3, 4, 5]
        })

        # Formula only uses x1, x2, y - should NOT error on unused columns with spaces
        result = mold("y ~ x1 + x2", data)
        assert result is not None
        assert 'x1' in result.predictors.columns
        assert 'x2' in result.predictors.columns

    def test_shows_first_five_columns(self):
        """Error should show first 5 invalid columns when many exist"""
        # Create data with 10 columns with spaces
        data = pd.DataFrame({
            f'col {i}': [1, 2, 3] for i in range(10)
        })
        data['y'] = [1, 2, 3]

        with pytest.raises(ValueError) as exc_info:
            mold("y ~ .", data)

        error_msg = str(exc_info.value)
        assert "Found 10 invalid column(s)" in error_msg
        # Should show first 5
        assert "col 0" in error_msg or "col 1" in error_msg
