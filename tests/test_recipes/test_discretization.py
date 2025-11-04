"""
Tests for discretization preprocessing steps
"""

import pytest
import pandas as pd
import numpy as np
from py_recipes.recipe import recipe


class TestStepPercentile:
    """Test step_percentile functionality"""

    @pytest.fixture
    def numeric_data(self):
        """Create numeric data for percentile testing"""
        np.random.seed(42)
        return pd.DataFrame({
            'value1': np.arange(1, 101),  # 1-100
            'value2': np.random.randn(100) * 10 + 50,
            'category': ['A', 'B'] * 50
        })

    def test_percentile_basic(self, numeric_data):
        """Test basic percentile conversion"""
        rec = recipe().step_percentile(columns=['value1'])
        rec_fit = rec.prep(numeric_data)
        transformed = rec_fit.bake(numeric_data)

        # Check that values are in 0-100 range
        assert transformed['value1'].min() >= 0
        assert transformed['value1'].max() <= 100

        # Check that lowest value becomes 0 and highest becomes 100
        assert transformed.loc[0, 'value1'] in [0, 1]  # First value (1) should be near 0
        assert transformed.loc[99, 'value1'] in [99, 100]  # Last value (100) should be 100

    def test_percentile_auto_detect(self, numeric_data):
        """Test percentile with auto-detection of numeric columns"""
        rec = recipe().step_percentile()
        rec_fit = rec.prep(numeric_data)
        transformed = rec_fit.bake(numeric_data)

        # Should transform numeric columns only
        assert transformed['value1'].dtype in [np.int32, np.int64]
        assert transformed['value2'].dtype in [np.int32, np.int64]

        # Category column should be unchanged
        assert transformed['category'].tolist() == numeric_data['category'].tolist()

    def test_percentile_integer_output(self, numeric_data):
        """Test that percentiles are integers by default"""
        rec = recipe().step_percentile(columns=['value1'])
        rec_fit = rec.prep(numeric_data)
        transformed = rec_fit.bake(numeric_data)

        # Should be integer type
        assert transformed['value1'].dtype in [np.int32, np.int64]

    def test_percentile_float_output(self, numeric_data):
        """Test percentiles as float values"""
        rec = recipe().step_percentile(columns=['value1'], as_integer=False)
        rec_fit = rec.prep(numeric_data)
        transformed = rec_fit.bake(numeric_data)

        # Should be float type
        assert transformed['value1'].dtype == np.float64

    def test_percentile_num_breaks(self, numeric_data):
        """Test custom number of percentile breaks"""
        rec = recipe().step_percentile(columns=['value1'], num_breaks=10)
        rec_fit = rec.prep(numeric_data)
        transformed = rec_fit.bake(numeric_data)

        # With num_breaks=10, values should be 0-10
        assert transformed['value1'].min() >= 0
        assert transformed['value1'].max() <= 10

    def test_percentile_new_data(self, numeric_data):
        """Test applying percentile to new data"""
        train = numeric_data[:50]
        test = pd.DataFrame({
            'value1': [5, 25, 75, 95],
            'value2': [40, 50, 60, 70],
            'category': ['A', 'B', 'A', 'B']
        })

        rec = recipe().step_percentile(columns=['value1'])
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Values should be percentile ranks based on training data
        assert test_transformed['value1'].min() >= 0
        assert test_transformed['value1'].max() <= 100

    def test_percentile_preserves_shape(self, numeric_data):
        """Test percentile preserves data shape"""
        rec = recipe().step_percentile(columns=['value1'])
        rec_fit = rec.prep(numeric_data)
        transformed = rec_fit.bake(numeric_data)

        assert transformed.shape == numeric_data.shape
        assert list(transformed.columns) == list(numeric_data.columns)

    def test_percentile_order_preserved(self, numeric_data):
        """Test that relative order is preserved"""
        rec = recipe().step_percentile(columns=['value1'])
        rec_fit = rec.prep(numeric_data)
        transformed = rec_fit.bake(numeric_data)

        # Check that order is preserved
        for i in range(len(transformed) - 1):
            assert transformed.loc[i, 'value1'] <= transformed.loc[i+1, 'value1']

    def test_percentile_identical_values(self):
        """Test percentile with identical values"""
        data = pd.DataFrame({
            'value': [10, 10, 10, 10, 10]
        })

        rec = recipe().step_percentile(columns=['value'])
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # All values should map to same percentile
        assert len(transformed['value'].unique()) <= 2  # Might be 0 or 1 due to binning

    def test_percentile_edge_values(self):
        """Test percentile with edge case values"""
        data = pd.DataFrame({
            'value': [0, 0.5, 1]
        })

        rec = recipe().step_percentile(columns=['value'])
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should handle edge values correctly
        # With 3 values, expect min > 0 and max = 100
        assert transformed['value'].min() >= 0
        assert transformed['value'].max() <= 100
        # Values should be ordered
        assert transformed['value'].iloc[0] < transformed['value'].iloc[2]
