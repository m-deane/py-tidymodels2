"""
Tests for scaling and centering recipe steps
"""

import pytest
import pandas as pd
import numpy as np

from py_recipes import recipe


class TestStepCenter:
    """Test step_center functionality"""

    @pytest.fixture
    def numeric_data(self):
        """Create numeric data"""
        return pd.DataFrame({
            'x1': [10, 20, 30, 40, 50],
            'x2': [100, 200, 300, 400, 500],
            'x3': [-5, -3, 0, 3, 5]
        })

    def test_center_basic(self, numeric_data):
        """Test basic centering"""
        rec = recipe().step_center()
        rec_fit = rec.prep(numeric_data)
        transformed = rec_fit.bake(numeric_data)

        # All columns should have mean close to 0
        assert abs(transformed['x1'].mean()) < 1e-10
        assert abs(transformed['x2'].mean()) < 1e-10
        assert abs(transformed['x3'].mean()) < 1e-10

    def test_center_specific_columns(self, numeric_data):
        """Test centering specific columns"""
        rec = recipe().step_center(columns=['x1', 'x2'])
        rec_fit = rec.prep(numeric_data)
        transformed = rec_fit.bake(numeric_data)

        # x1 and x2 should be centered
        assert abs(transformed['x1'].mean()) < 1e-10
        assert abs(transformed['x2'].mean()) < 1e-10

        # x3 should be unchanged
        np.testing.assert_array_equal(transformed['x3'].values, numeric_data['x3'].values)

    def test_center_preserves_shape(self, numeric_data):
        """Test centering preserves data shape"""
        rec = recipe().step_center()
        rec_fit = rec.prep(numeric_data)
        transformed = rec_fit.bake(numeric_data)

        assert transformed.shape == numeric_data.shape
        assert list(transformed.columns) == list(numeric_data.columns)

    def test_center_new_data(self, numeric_data):
        """Test applying centering to new data"""
        train = numeric_data[:3]
        test = numeric_data[3:]

        rec = recipe().step_center()
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Should use training means
        train_mean_x1 = train['x1'].mean()
        expected = test['x1'] - train_mean_x1
        np.testing.assert_array_almost_equal(test_transformed['x1'].values, expected.values)

    def test_center_with_categorical(self):
        """Test centering with mixed data types"""
        data = pd.DataFrame({
            'numeric': [10, 20, 30, 40, 50],
            'category': ['A', 'B', 'C', 'D', 'E']
        })

        rec = recipe().step_center()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Numeric should be centered
        assert abs(transformed['numeric'].mean()) < 1e-10

        # Categorical should be unchanged
        assert list(transformed['category']) == list(data['category'])


class TestStepScale:
    """Test step_scale functionality"""

    @pytest.fixture
    def numeric_data(self):
        """Create numeric data"""
        return pd.DataFrame({
            'x1': [10, 20, 30, 40, 50],
            'x2': [1, 2, 3, 4, 5],
            'x3': [100, 110, 120, 130, 140]
        })

    def test_scale_basic(self, numeric_data):
        """Test basic scaling"""
        rec = recipe().step_scale()
        rec_fit = rec.prep(numeric_data)
        transformed = rec_fit.bake(numeric_data)

        # All columns should have std close to 1
        assert abs(transformed['x1'].std() - 1.0) < 1e-10
        assert abs(transformed['x2'].std() - 1.0) < 1e-10
        assert abs(transformed['x3'].std() - 1.0) < 1e-10

    def test_scale_specific_columns(self, numeric_data):
        """Test scaling specific columns"""
        rec = recipe().step_scale(columns=['x1', 'x2'])
        rec_fit = rec.prep(numeric_data)
        transformed = rec_fit.bake(numeric_data)

        # x1 and x2 should be scaled
        assert abs(transformed['x1'].std() - 1.0) < 1e-10
        assert abs(transformed['x2'].std() - 1.0) < 1e-10

        # x3 should be unchanged
        np.testing.assert_array_equal(transformed['x3'].values, numeric_data['x3'].values)

    def test_scale_new_data(self, numeric_data):
        """Test applying scaling to new data"""
        train = numeric_data[:3]
        test = numeric_data[3:]

        rec = recipe().step_scale()
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Should use training std
        train_std_x1 = train['x1'].std()
        expected = test['x1'] / train_std_x1
        np.testing.assert_array_almost_equal(test_transformed['x1'].values, expected.values)

    def test_scale_zero_std(self):
        """Test scaling with zero standard deviation"""
        data = pd.DataFrame({
            'constant': [5, 5, 5, 5, 5],
            'variable': [1, 2, 3, 4, 5]
        })

        rec = recipe().step_scale()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Constant column should be unchanged (std=0 check)
        np.testing.assert_array_equal(transformed['constant'].values, data['constant'].values)

        # Variable column should be scaled
        assert abs(transformed['variable'].std() - 1.0) < 1e-10


class TestStepRange:
    """Test step_range functionality"""

    @pytest.fixture
    def numeric_data(self):
        """Create numeric data"""
        return pd.DataFrame({
            'x1': [0, 25, 50, 75, 100],
            'x2': [-10, -5, 0, 5, 10],
            'x3': [1, 2, 3, 4, 5]
        })

    def test_range_default(self, numeric_data):
        """Test default range [0, 1]"""
        rec = recipe().step_range()
        rec_fit = rec.prep(numeric_data)
        transformed = rec_fit.bake(numeric_data)

        # All columns should be in [0, 1]
        assert transformed['x1'].min() == 0.0
        assert transformed['x1'].max() == 1.0
        assert transformed['x2'].min() == 0.0
        assert transformed['x2'].max() == 1.0

    def test_range_custom(self, numeric_data):
        """Test custom range [-1, 1]"""
        rec = recipe().step_range(min_val=-1.0, max_val=1.0)
        rec_fit = rec.prep(numeric_data)
        transformed = rec_fit.bake(numeric_data)

        # All columns should be in [-1, 1]
        assert transformed['x1'].min() == -1.0
        assert transformed['x1'].max() == 1.0
        assert transformed['x2'].min() == -1.0
        assert transformed['x2'].max() == 1.0

    def test_range_specific_columns(self, numeric_data):
        """Test range on specific columns"""
        rec = recipe().step_range(columns=['x1', 'x2'])
        rec_fit = rec.prep(numeric_data)
        transformed = rec_fit.bake(numeric_data)

        # x1 and x2 should be scaled
        assert 0.0 <= transformed['x1'].min() <= 0.01
        assert 0.99 <= transformed['x1'].max() <= 1.0

        # x3 should be unchanged
        np.testing.assert_array_equal(transformed['x3'].values, numeric_data['x3'].values)

    def test_range_new_data(self, numeric_data):
        """Test applying range to new data"""
        train = numeric_data[:3]
        test = numeric_data[3:]

        rec = recipe().step_range()
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Should use training min/max
        # Test data might go outside [0,1] if outside training range
        assert len(test_transformed) == len(test)

    def test_range_constant_column(self):
        """Test range with constant column"""
        data = pd.DataFrame({
            'constant': [5, 5, 5, 5, 5],
            'variable': [1, 2, 3, 4, 5]
        })

        rec = recipe().step_range()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Constant column should be unchanged (min=max check)
        np.testing.assert_array_equal(transformed['constant'].values, data['constant'].values)

        # Variable column should be scaled
        assert transformed['variable'].min() == 0.0
        assert transformed['variable'].max() == 1.0


class TestScalingPipeline:
    """Test combinations of scaling steps"""

    @pytest.fixture
    def numeric_data(self):
        """Create numeric data"""
        np.random.seed(42)
        return pd.DataFrame({
            'x1': np.random.normal(100, 20, 100),
            'x2': np.random.normal(0, 5, 100),
            'x3': np.random.uniform(0, 100, 100)
        })

    def test_center_then_scale(self, numeric_data):
        """Test centering followed by scaling (standardization)"""
        rec = (
            recipe()
            .step_center()
            .step_scale()
        )

        rec_fit = rec.prep(numeric_data)
        transformed = rec_fit.bake(numeric_data)

        # Should be standardized (mean=0, std=1)
        assert abs(transformed['x1'].mean()) < 1e-10
        assert abs(transformed['x1'].std() - 1.0) < 1e-10

    def test_normalize_vs_center_scale(self, numeric_data):
        """Test that step_normalize matches center+scale"""
        # Using center + scale
        rec1 = (
            recipe()
            .step_center()
            .step_scale()
        )
        rec1_fit = rec1.prep(numeric_data)
        result1 = rec1_fit.bake(numeric_data)

        # Using normalize with zscore
        rec2 = recipe().step_normalize(method='zscore')
        rec2_fit = rec2.prep(numeric_data)
        result2 = rec2_fit.bake(numeric_data)

        # Results should be similar (allowing for small numerical differences due to ddof)
        np.testing.assert_array_almost_equal(result1['x1'].values, result2['x1'].values, decimal=2)

    def test_range_after_log(self, numeric_data):
        """Test range scaling after log transform"""
        # Make all positive
        data = numeric_data.abs() + 1

        rec = (
            recipe()
            .step_log()
            .step_range()
        )

        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should be in [0, 1] after both transforms
        assert 0.0 <= transformed['x1'].min() <= 0.01
        assert 0.99 <= transformed['x1'].max() <= 1.0


class TestScalingEdgeCases:
    """Test edge cases for scaling steps"""

    def test_center_single_value(self):
        """Test centering single row"""
        data = pd.DataFrame({'x': [5]})

        rec = recipe().step_center()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Single value should become 0
        assert transformed['x'].iloc[0] == 0.0

    def test_scale_two_identical_values(self):
        """Test scaling with identical values"""
        data = pd.DataFrame({'x': [5, 5]})

        rec = recipe().step_scale()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should handle zero std gracefully
        assert not transformed.isna().any().any()

    def test_range_negative_to_positive(self):
        """Test range scaling from negative to positive values"""
        data = pd.DataFrame({'x': [-10, -5, 0, 5, 10]})

        rec = recipe().step_range(min_val=0, max_val=100)
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        assert transformed['x'].min() == 0.0
        assert transformed['x'].max() == 100.0
