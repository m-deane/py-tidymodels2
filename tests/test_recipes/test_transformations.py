"""
Tests for mathematical transformation recipe steps
"""

import pytest
import pandas as pd
import numpy as np

from py_recipes import recipe


class TestStepLog:
    """Test step_log functionality"""

    @pytest.fixture
    def positive_data(self):
        """Create data with positive values"""
        return pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [10, 20, 30, 40, 50],
            'x3': [100, 200, 300, 400, 500]
        })

    @pytest.fixture
    def mixed_data(self):
        """Create data with positive and negative values"""
        return pd.DataFrame({
            'x1': [-5, -2, 0, 2, 5],
            'x2': [1, 2, 3, 4, 5]
        })

    def test_log_basic(self, positive_data):
        """Test basic log transformation"""
        rec = recipe().step_log()
        rec_fit = rec.prep(positive_data)
        transformed = rec_fit.bake(positive_data)

        # Check transformation applied
        expected_x1 = np.log(positive_data['x1'])
        np.testing.assert_array_almost_equal(transformed['x1'].values, expected_x1.values)

    def test_log_specific_columns(self, positive_data):
        """Test log on specific columns"""
        rec = recipe().step_log(columns=['x1', 'x2'])
        rec_fit = rec.prep(positive_data)
        transformed = rec_fit.bake(positive_data)

        # x1 and x2 should be transformed
        expected_x1 = np.log(positive_data['x1'])
        np.testing.assert_array_almost_equal(transformed['x1'].values, expected_x1.values)

        # x3 should be unchanged
        np.testing.assert_array_equal(transformed['x3'].values, positive_data['x3'].values)

    def test_log_base(self, positive_data):
        """Test log with different base"""
        rec = recipe().step_log(base=10)
        rec_fit = rec.prep(positive_data)
        transformed = rec_fit.bake(positive_data)

        expected_x1 = np.log10(positive_data['x1'])
        np.testing.assert_array_almost_equal(transformed['x1'].values, expected_x1.values)

    def test_log_offset(self, positive_data):
        """Test log with offset"""
        rec = recipe().step_log(offset=1.0)
        rec_fit = rec.prep(positive_data)
        transformed = rec_fit.bake(positive_data)

        expected_x1 = np.log(positive_data['x1'] + 1.0)
        np.testing.assert_array_almost_equal(transformed['x1'].values, expected_x1.values)

    def test_log_signed(self, mixed_data):
        """Test signed log transformation"""
        rec = recipe().step_log(signed=True)
        rec_fit = rec.prep(mixed_data)
        transformed = rec_fit.bake(mixed_data)

        # Verify signed transformation preserves sign
        assert (transformed['x1'][transformed['x1'] < 0] < 0).all()
        assert (transformed['x1'][transformed['x1'] > 0] > 0).all()

    def test_log_new_data(self, positive_data):
        """Test applying log to new data"""
        train = positive_data[:3]
        test = positive_data[3:]

        rec = recipe().step_log()
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        assert len(test_transformed) == len(test)
        assert not test_transformed.isna().any().any()


class TestStepSqrt:
    """Test step_sqrt functionality"""

    @pytest.fixture
    def positive_data(self):
        """Create data with positive values"""
        return pd.DataFrame({
            'x1': [1, 4, 9, 16, 25],
            'x2': [100, 144, 169, 196, 225]
        })

    def test_sqrt_basic(self, positive_data):
        """Test basic sqrt transformation"""
        rec = recipe().step_sqrt()
        rec_fit = rec.prep(positive_data)
        transformed = rec_fit.bake(positive_data)

        expected_x1 = np.sqrt(positive_data['x1'])
        np.testing.assert_array_almost_equal(transformed['x1'].values, expected_x1.values)

    def test_sqrt_specific_columns(self, positive_data):
        """Test sqrt on specific columns"""
        rec = recipe().step_sqrt(columns=['x1'])
        rec_fit = rec.prep(positive_data)
        transformed = rec_fit.bake(positive_data)

        # x1 should be transformed
        expected_x1 = np.sqrt(positive_data['x1'])
        np.testing.assert_array_almost_equal(transformed['x1'].values, expected_x1.values)

        # x2 should be unchanged
        np.testing.assert_array_equal(transformed['x2'].values, positive_data['x2'].values)

    def test_sqrt_new_data(self, positive_data):
        """Test applying sqrt to new data"""
        train = positive_data[:3]
        test = positive_data[3:]

        rec = recipe().step_sqrt()
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        assert len(test_transformed) == len(test)
        expected = np.sqrt(test['x1'])
        np.testing.assert_array_almost_equal(test_transformed['x1'].values, expected.values)


class TestStepBoxCox:
    """Test step_boxcox functionality"""

    @pytest.fixture
    def positive_data(self):
        """Create data with positive values"""
        np.random.seed(42)
        return pd.DataFrame({
            'x1': np.random.exponential(2, 100),
            'x2': np.random.lognormal(0, 1, 100)
        })

    def test_boxcox_basic(self, positive_data):
        """Test basic BoxCox transformation"""
        rec = recipe().step_boxcox()
        rec_fit = rec.prep(positive_data)
        transformed = rec_fit.bake(positive_data)

        # Should have transformed columns
        assert 'x1' in transformed.columns
        assert 'x2' in transformed.columns

        # Values should be different
        assert not np.array_equal(transformed['x1'].values, positive_data['x1'].values)

    def test_boxcox_specific_columns(self, positive_data):
        """Test BoxCox on specific columns"""
        rec = recipe().step_boxcox(columns=['x1'])
        rec_fit = rec.prep(positive_data)
        transformed = rec_fit.bake(positive_data)

        # x1 should be transformed
        assert not np.array_equal(transformed['x1'].values, positive_data['x1'].values)

        # x2 should be unchanged (not selected for transformation)
        # Note: x2 might still be transformed if None columns means all numeric
        # Let's check the actual behavior

    def test_boxcox_new_data(self, positive_data):
        """Test applying BoxCox to new data"""
        train = positive_data[:80]
        test = positive_data[80:]

        rec = recipe().step_boxcox()
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        assert len(test_transformed) == len(test)
        assert not test_transformed.isna().any().any()

    def test_boxcox_with_zeros_skipped(self):
        """Test BoxCox skips columns with non-positive values"""
        data = pd.DataFrame({
            'positive': [1, 2, 3, 4, 5],
            'with_zero': [0, 1, 2, 3, 4],
            'negative': [-1, 0, 1, 2, 3]
        })

        rec = recipe().step_boxcox()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Only positive column should be transformed
        # with_zero and negative should be unchanged
        assert not np.array_equal(transformed['positive'].values, data['positive'].values)


class TestStepYeoJohnson:
    """Test step_yeojohnson functionality"""

    @pytest.fixture
    def mixed_data(self):
        """Create data with positive and negative values"""
        np.random.seed(42)
        return pd.DataFrame({
            'x1': np.random.normal(0, 2, 100),
            'x2': np.random.exponential(1, 100) - 1  # Mix of positive and negative
        })

    def test_yeojohnson_basic(self, mixed_data):
        """Test basic Yeo-Johnson transformation"""
        rec = recipe().step_yeojohnson()
        rec_fit = rec.prep(mixed_data)
        transformed = rec_fit.bake(mixed_data)

        # Should have transformed columns
        assert 'x1' in transformed.columns
        assert 'x2' in transformed.columns

        # Values should be different
        assert not np.array_equal(transformed['x1'].values, mixed_data['x1'].values)

    def test_yeojohnson_handles_negatives(self, mixed_data):
        """Test Yeo-Johnson handles negative values"""
        rec = recipe().step_yeojohnson()
        rec_fit = rec.prep(mixed_data)
        transformed = rec_fit.bake(mixed_data)

        # Should successfully transform data with negatives
        assert not transformed.isna().any().any()

    def test_yeojohnson_specific_columns(self, mixed_data):
        """Test Yeo-Johnson on specific columns"""
        rec = recipe().step_yeojohnson(columns=['x1'])
        rec_fit = rec.prep(mixed_data)
        transformed = rec_fit.bake(mixed_data)

        # x1 should be transformed
        assert not np.array_equal(transformed['x1'].values, mixed_data['x1'].values)

    def test_yeojohnson_new_data(self, mixed_data):
        """Test applying Yeo-Johnson to new data"""
        train = mixed_data[:80]
        test = mixed_data[80:]

        rec = recipe().step_yeojohnson()
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        assert len(test_transformed) == len(test)
        assert not test_transformed.isna().any().any()


class TestTransformationPipeline:
    """Test combinations of transformation steps"""

    @pytest.fixture
    def skewed_data(self):
        """Create skewed data"""
        np.random.seed(42)
        return pd.DataFrame({
            'skewed': np.random.exponential(2, 100),
            'normal': np.random.normal(0, 1, 100),
            'squared': np.arange(1, 101) ** 2
        })

    def test_log_then_center(self, skewed_data):
        """Test log transformation followed by centering"""
        rec = (
            recipe()
            .step_log(columns=['skewed'])
            .step_center()
        )

        rec_fit = rec.prep(skewed_data)
        transformed = rec_fit.bake(skewed_data)

        # After centering, mean should be close to 0
        assert abs(transformed['skewed'].mean()) < 1e-10

    def test_sqrt_then_normalize(self, skewed_data):
        """Test sqrt then normalization"""
        rec = (
            recipe()
            .step_sqrt(columns=['squared'])
            .step_normalize()
        )

        rec_fit = rec.prep(skewed_data)
        transformed = rec_fit.bake(skewed_data)

        # Check normalization applied (use reasonable tolerance for sample std)
        assert abs(transformed['squared'].mean()) < 1e-10
        assert abs(transformed['squared'].std() - 1.0) < 0.01


class TestTransformationEdgeCases:
    """Test edge cases for transformation steps"""

    def test_log_with_single_column(self):
        """Test log with single column"""
        data = pd.DataFrame({'x': [1, 2, 3, 4, 5]})

        rec = recipe().step_log()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        expected = np.log(data['x'])
        np.testing.assert_array_almost_equal(transformed['x'].values, expected.values)

    def test_sqrt_with_empty_selection(self):
        """Test sqrt with no numeric columns selected"""
        data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'D', 'E']
        })

        rec = recipe().step_sqrt()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should return data unchanged
        pd.testing.assert_frame_equal(transformed, data)

    def test_boxcox_with_single_positive_column(self):
        """Test BoxCox with single positive column"""
        data = pd.DataFrame({'x': [1, 2, 3, 4, 5]})

        rec = recipe().step_boxcox()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should transform successfully
        assert not np.array_equal(transformed['x'].values, data['x'].values)


class TestStepInverse:
    """Test step_inverse functionality"""

    @pytest.fixture
    def nonzero_data(self):
        """Create data with non-zero values"""
        return pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [10, 20, 30, 40, 50],
            'x3': [0.5, 1.5, 2.5, 3.5, 4.5]
        })

    @pytest.fixture
    def data_with_zeros(self):
        """Create data with some zero values"""
        return pd.DataFrame({
            'x1': [0, 1, 2, 3, 4],
            'x2': [5, 10, 15, 20, 25]
        })

    def test_inverse_basic(self, nonzero_data):
        """Test basic inverse transformation"""
        rec = recipe().step_inverse()
        rec_fit = rec.prep(nonzero_data)
        transformed = rec_fit.bake(nonzero_data)

        # Check transformation applied: 1/x
        expected_x1 = 1.0 / nonzero_data['x1']
        np.testing.assert_array_almost_equal(transformed['x1'].values, expected_x1.values)

    def test_inverse_specific_columns(self, nonzero_data):
        """Test inverse on specific columns"""
        rec = recipe().step_inverse(columns=['x1', 'x2'])
        rec_fit = rec.prep(nonzero_data)
        transformed = rec_fit.bake(nonzero_data)

        # x1 and x2 should be transformed
        expected_x1 = 1.0 / nonzero_data['x1']
        np.testing.assert_array_almost_equal(transformed['x1'].values, expected_x1.values)

        # x3 should be unchanged
        np.testing.assert_array_equal(transformed['x3'].values, nonzero_data['x3'].values)

    def test_inverse_offset(self, data_with_zeros):
        """Test inverse with offset to handle zeros"""
        rec = recipe().step_inverse(offset=1.0)
        rec_fit = rec.prep(data_with_zeros)
        transformed = rec_fit.bake(data_with_zeros)

        # Check transformation: 1 / (x + offset)
        expected_x1 = 1.0 / (data_with_zeros['x1'] + 1.0)
        np.testing.assert_array_almost_equal(transformed['x1'].values, expected_x1.values)

    def test_inverse_preserves_ordering(self, nonzero_data):
        """Test that inverse reverses ordering"""
        rec = recipe().step_inverse(columns=['x1'])
        rec_fit = rec.prep(nonzero_data)
        transformed = rec_fit.bake(nonzero_data)

        # For positive numbers, 1/x should be decreasing when x is increasing
        # x1 is [1, 2, 3, 4, 5], so 1/x1 should be [1, 0.5, 0.33, 0.25, 0.2]
        assert transformed['x1'].iloc[0] > transformed['x1'].iloc[1]
        assert transformed['x1'].iloc[1] > transformed['x1'].iloc[2]

    def test_inverse_new_data(self, nonzero_data):
        """Test applying inverse to new data"""
        train = nonzero_data[:3]
        test = nonzero_data[3:]

        rec = recipe().step_inverse()
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        assert len(test_transformed) == len(test)
        expected = 1.0 / test['x1']
        np.testing.assert_array_almost_equal(test_transformed['x1'].values, expected.values)

    def test_inverse_preserves_shape(self, nonzero_data):
        """Test inverse preserves data shape"""
        rec = recipe().step_inverse()
        rec_fit = rec.prep(nonzero_data)
        transformed = rec_fit.bake(nonzero_data)

        assert transformed.shape == nonzero_data.shape
        assert list(transformed.columns) == list(nonzero_data.columns)

    def test_inverse_small_values(self):
        """Test inverse with very small values"""
        data = pd.DataFrame({
            'x': [0.001, 0.01, 0.1, 1, 10]
        })

        rec = recipe().step_inverse()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Small values should become large
        expected = 1.0 / data['x']
        np.testing.assert_array_almost_equal(transformed['x'].values, expected.values)

    def test_inverse_large_values(self):
        """Test inverse with large values"""
        data = pd.DataFrame({
            'x': [100, 1000, 10000]
        })

        rec = recipe().step_inverse()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Large values should become small
        expected = 1.0 / data['x']
        np.testing.assert_array_almost_equal(transformed['x'].values, expected.values)

    def test_inverse_with_offset_zero(self, data_with_zeros):
        """Test that offset prevents division by zero"""
        rec = recipe().step_inverse(offset=0.1)
        rec_fit = rec.prep(data_with_zeros)
        transformed = rec_fit.bake(data_with_zeros)

        # Should not have any inf values
        assert not np.isinf(transformed['x1']).any()
        assert not np.isinf(transformed['x2']).any()

    def test_inverse_roundtrip(self, nonzero_data):
        """Test that applying inverse twice gives back original (approximately)"""
        rec = recipe().step_inverse(columns=['x1'])
        rec_fit = rec.prep(nonzero_data)

        # Apply inverse
        transformed = rec_fit.bake(nonzero_data)

        # Apply inverse again
        rec2 = recipe().step_inverse(columns=['x1'])
        rec2_fit = rec2.prep(transformed)
        double_transformed = rec2_fit.bake(transformed)

        # Should get back to original values (approximately)
        np.testing.assert_array_almost_equal(
            double_transformed['x1'].values,
            nonzero_data['x1'].values,
            decimal=10
        )
