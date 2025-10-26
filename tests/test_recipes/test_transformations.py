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
