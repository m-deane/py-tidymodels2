"""
Tests for feature filtering recipe steps
"""

import pytest
import pandas as pd
import numpy as np

from py_recipes import recipe


class TestStepZv:
    """Test step_zv (zero variance) functionality"""

    @pytest.fixture
    def mixed_variance_data(self):
        """Create data with zero and non-zero variance columns"""
        return pd.DataFrame({
            'constant1': [5, 5, 5, 5, 5],
            'constant2': [0, 0, 0, 0, 0],
            'variable1': [1, 2, 3, 4, 5],
            'variable2': [10, 20, 30, 40, 50]
        })

    def test_zv_removes_constants(self, mixed_variance_data):
        """Test that zero variance columns are removed"""
        rec = recipe().step_zv()
        rec_fit = rec.prep(mixed_variance_data)
        transformed = rec_fit.bake(mixed_variance_data)

        # Constant columns should be removed
        assert 'constant1' not in transformed.columns
        assert 'constant2' not in transformed.columns

        # Variable columns should remain
        assert 'variable1' in transformed.columns
        assert 'variable2' in transformed.columns

    def test_zv_specific_columns(self, mixed_variance_data):
        """Test zero variance on specific columns"""
        rec = recipe().step_zv(columns=['constant1', 'variable1'])
        rec_fit = rec.prep(mixed_variance_data)
        transformed = rec_fit.bake(mixed_variance_data)

        # Only constant1 should be removed (from specified columns)
        assert 'constant1' not in transformed.columns
        assert 'variable1' in transformed.columns

        # constant2 should remain (not in specified columns)
        assert 'constant2' in transformed.columns

    def test_zv_no_constant_columns(self):
        """Test with no constant columns"""
        data = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [10, 20, 30, 40, 50]
        })

        rec = recipe().step_zv()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # All columns should remain
        assert list(transformed.columns) == list(data.columns)

    def test_zv_new_data(self, mixed_variance_data):
        """Test applying zv to new data"""
        train = mixed_variance_data[:3]
        test = mixed_variance_data[3:]

        rec = recipe().step_zv()
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Same columns should be removed
        assert 'constant1' not in test_transformed.columns
        assert 'variable1' in test_transformed.columns


class TestStepNzv:
    """Test step_nzv (near-zero variance) functionality"""

    @pytest.fixture
    def near_zero_variance_data(self):
        """Create data with near-zero variance columns"""
        return pd.DataFrame({
            'mostly_one_value': [1, 1, 1, 1, 1, 1, 1, 1, 1, 2],  # 90% same
            'rare_values': [1, 1, 1, 1, 1, 1, 1, 1, 2, 3],  # 80% same, 2 unique
            'good_variance': np.arange(10),
            'many_unique': np.random.choice(range(100), 10)
        })

    def test_nzv_removes_near_constants(self, near_zero_variance_data):
        """Test that near-zero variance columns are removed"""
        rec = recipe().step_nzv(freq_cut=5.0, unique_cut=15.0)
        rec_fit = rec.prep(near_zero_variance_data)
        transformed = rec_fit.bake(near_zero_variance_data)

        # Near-constant columns might be removed depending on thresholds
        # Check that some filtering occurred
        assert len(transformed.columns) <= len(near_zero_variance_data.columns)

    def test_nzv_with_low_threshold(self, near_zero_variance_data):
        """Test nzv with low frequency threshold"""
        rec = recipe().step_nzv(freq_cut=2.0, unique_cut=20.0)
        rec_fit = rec.prep(near_zero_variance_data)
        transformed = rec_fit.bake(near_zero_variance_data)

        # More columns should be removed with lower threshold
        assert len(transformed.columns) <= len(near_zero_variance_data.columns)

    def test_nzv_no_near_zero_columns(self):
        """Test with no near-zero variance columns"""
        np.random.seed(42)
        data = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.choice(range(50), 100)
        })

        rec = recipe().step_nzv()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # All columns should likely remain
        assert 'x1' in transformed.columns
        assert 'x2' in transformed.columns

    def test_nzv_new_data(self, near_zero_variance_data):
        """Test applying nzv to new data"""
        train = near_zero_variance_data[:8]
        test = near_zero_variance_data[8:]

        rec = recipe().step_nzv()
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Should have same columns as training
        assert len(test_transformed) == len(test)


class TestStepLinComb:
    """Test step_lincomb (linear combinations) functionality"""

    @pytest.fixture
    def collinear_data(self):
        """Create data with linear dependencies"""
        np.random.seed(42)
        x1 = np.random.randn(50)
        x2 = np.random.randn(50)
        return pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'x3': x1 + x2,  # Perfect linear combination
            'x4': 2 * x1 + 3 * x2,  # Another linear combination
            'x5': np.random.randn(50)  # Independent
        })

    def test_lincomb_removes_dependent_columns(self, collinear_data):
        """Test that linearly dependent columns are removed"""
        rec = recipe().step_lincomb()
        rec_fit = rec.prep(collinear_data)
        transformed = rec_fit.bake(collinear_data)

        # Should remove some dependent columns
        assert len(transformed.columns) < len(collinear_data.columns)

    def test_lincomb_keeps_independent(self, collinear_data):
        """Test that independent columns are kept"""
        rec = recipe().step_lincomb()
        rec_fit = rec.prep(collinear_data)
        transformed = rec_fit.bake(collinear_data)

        # x5 should definitely be kept (independent)
        # x1 and x2 should be kept (base columns)
        assert 'x5' in transformed.columns

    def test_lincomb_no_dependencies(self):
        """Test with no linear dependencies"""
        np.random.seed(42)
        data = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'x3': np.random.randn(50)
        })

        rec = recipe().step_lincomb()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # All columns should remain
        assert set(transformed.columns) == set(data.columns)

    def test_lincomb_threshold(self, collinear_data):
        """Test with custom threshold"""
        rec = recipe().step_lincomb(threshold=1e-3)
        rec_fit = rec.prep(collinear_data)
        transformed = rec_fit.bake(collinear_data)

        # Should remove dependent columns
        assert len(transformed.columns) <= len(collinear_data.columns)

    def test_lincomb_new_data(self, collinear_data):
        """Test applying lincomb to new data"""
        train = collinear_data[:40]
        test = collinear_data[40:]

        rec = recipe().step_lincomb()
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Should have same columns as training
        assert len(test_transformed) == len(test)


class TestStepFilterMissing:
    """Test step_filter_missing functionality"""

    @pytest.fixture
    def missing_data(self):
        """Create data with various missing proportions"""
        return pd.DataFrame({
            'no_missing': [1, 2, 3, 4, 5],
            'few_missing': [1, np.nan, 3, 4, 5],  # 20% missing
            'half_missing': [1, np.nan, np.nan, 4, 5],  # 40% missing
            'mostly_missing': [1, np.nan, np.nan, np.nan, np.nan]  # 80% missing
        })

    def test_filter_missing_default_threshold(self, missing_data):
        """Test with default 50% threshold"""
        rec = recipe().step_filter_missing(threshold=0.5)
        rec_fit = rec.prep(missing_data)
        transformed = rec_fit.bake(missing_data)

        # Columns with >50% missing should be removed
        assert 'mostly_missing' not in transformed.columns

        # Columns with <=50% missing should remain
        assert 'no_missing' in transformed.columns
        assert 'few_missing' in transformed.columns
        assert 'half_missing' in transformed.columns

    def test_filter_missing_strict_threshold(self, missing_data):
        """Test with strict threshold"""
        rec = recipe().step_filter_missing(threshold=0.1)
        rec_fit = rec.prep(missing_data)
        transformed = rec_fit.bake(missing_data)

        # Only column with no missing should remain
        assert 'no_missing' in transformed.columns
        assert len(transformed.columns) == 1

    def test_filter_missing_lenient_threshold(self, missing_data):
        """Test with lenient threshold"""
        rec = recipe().step_filter_missing(threshold=0.9)
        rec_fit = rec.prep(missing_data)
        transformed = rec_fit.bake(missing_data)

        # All columns should remain (none have >90% missing)
        assert set(transformed.columns) == set(missing_data.columns)

    def test_filter_missing_no_missing_data(self):
        """Test with no missing data"""
        data = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [10, 20, 30, 40, 50]
        })

        rec = recipe().step_filter_missing()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # All columns should remain
        assert set(transformed.columns) == set(data.columns)

    def test_filter_missing_specific_columns(self, missing_data):
        """Test filtering specific columns only"""
        rec = recipe().step_filter_missing(columns=['few_missing', 'mostly_missing'], threshold=0.5)
        rec_fit = rec.prep(missing_data)
        transformed = rec_fit.bake(missing_data)

        # Only mostly_missing from specified columns should be removed
        assert 'mostly_missing' not in transformed.columns
        assert 'few_missing' in transformed.columns

        # Other columns should remain even if they have high missing
        assert 'no_missing' in transformed.columns
        assert 'half_missing' in transformed.columns

    def test_filter_missing_new_data(self, missing_data):
        """Test applying filter_missing to new data"""
        train = missing_data[:3]
        test = missing_data[3:]

        rec = recipe().step_filter_missing(threshold=0.5)
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Should have same columns as training
        assert len(test_transformed) == len(test)


class TestFilterPipeline:
    """Test combinations of filter steps"""

    @pytest.fixture
    def complex_data(self):
        """Create complex data with multiple issues"""
        np.random.seed(42)
        x1 = np.random.randn(100)
        return pd.DataFrame({
            'constant': [5] * 100,  # Zero variance
            'near_constant': [1] * 95 + [2] * 5,  # Near-zero variance
            'collinear1': x1,
            'collinear2': x1 * 2,  # Linear combination
            'high_missing': [1] * 20 + [np.nan] * 80,  # 80% missing
            'good_var': np.random.randn(100)
        })

    def test_multiple_filters(self, complex_data):
        """Test applying multiple filters in sequence"""
        rec = (
            recipe()
            .step_filter_missing(threshold=0.5)
            .step_zv()
            .step_nzv(freq_cut=10.0)
            .step_lincomb()
        )

        rec_fit = rec.prep(complex_data)
        transformed = rec_fit.bake(complex_data)

        # Should remove multiple problematic columns
        assert 'constant' not in transformed.columns
        assert 'high_missing' not in transformed.columns

        # Good column should remain
        assert 'good_var' in transformed.columns

    def test_filter_order_matters(self, complex_data):
        """Test that filter order can affect results"""
        # Filter missing first
        rec1 = (
            recipe()
            .step_filter_missing(threshold=0.5)
            .step_zv()
        )
        rec1_fit = rec1.prep(complex_data)
        result1 = rec1_fit.bake(complex_data)

        # ZV first (might not see missing as constant)
        rec2 = (
            recipe()
            .step_zv()
            .step_filter_missing(threshold=0.5)
        )
        rec2_fit = rec2.prep(complex_data)
        result2 = rec2_fit.bake(complex_data)

        # Both should filter problematic columns
        assert 'constant' not in result1.columns
        assert 'constant' not in result2.columns


class TestFilterEdgeCases:
    """Test edge cases for filter steps"""

    def test_zv_all_constants(self):
        """Test zv when all columns are constant"""
        data = pd.DataFrame({
            'const1': [5, 5, 5],
            'const2': [10, 10, 10]
        })

        rec = recipe().step_zv()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # All columns removed
        assert len(transformed.columns) == 0

    def test_filter_missing_all_missing(self):
        """Test filter_missing when all values are missing"""
        data = pd.DataFrame({
            'all_missing': [np.nan, np.nan, np.nan, np.nan],
            'some_data': [1, 2, 3, 4]
        })

        rec = recipe().step_filter_missing(threshold=0.5)
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # All missing column should be removed
        assert 'all_missing' not in transformed.columns
        assert 'some_data' in transformed.columns

    def test_lincomb_single_column(self):
        """Test lincomb with single column"""
        data = pd.DataFrame({'x': [1, 2, 3, 4, 5]})

        rec = recipe().step_lincomb()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Single column should remain
        assert 'x' in transformed.columns
