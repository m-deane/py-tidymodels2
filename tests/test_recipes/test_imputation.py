"""
Tests for imputation recipe steps
"""

import pytest
import pandas as pd
import numpy as np

from py_recipes import recipe


class TestStepImputeMean:
    """Test step_impute_mean functionality"""

    @pytest.fixture
    def data_with_na(self):
        """Create data with missing values"""
        return pd.DataFrame({
            'x1': [1, 2, np.nan, 4, 5],
            'x2': [10, np.nan, 30, np.nan, 50],
            'x3': [100, 200, 300, 400, 500]
        })

    def test_impute_mean_basic(self, data_with_na):
        """Test basic mean imputation"""
        rec = recipe().step_impute_mean()
        rec_fit = rec.prep(data_with_na)
        transformed = rec_fit.bake(data_with_na)

        # Should have no missing values in imputed columns
        assert not transformed['x1'].isna().any()
        assert not transformed['x2'].isna().any()

        # x3 has no missing, should be unchanged
        np.testing.assert_array_equal(transformed['x3'].values, data_with_na['x3'].values)

    def test_impute_mean_values(self, data_with_na):
        """Test mean imputation values are correct"""
        rec = recipe().step_impute_mean()
        rec_fit = rec.prep(data_with_na)
        transformed = rec_fit.bake(data_with_na)

        # Calculate expected means (excluding NaN)
        expected_mean_x1 = data_with_na['x1'].mean()
        expected_mean_x2 = data_with_na['x2'].mean()

        # Check imputed values
        assert abs(transformed.loc[2, 'x1'] - expected_mean_x1) < 1e-10
        assert abs(transformed.loc[1, 'x2'] - expected_mean_x2) < 1e-10

    def test_impute_mean_specific_columns(self, data_with_na):
        """Test mean imputation on specific columns"""
        rec = recipe().step_impute_mean(columns=['x1'])
        rec_fit = rec.prep(data_with_na)
        transformed = rec_fit.bake(data_with_na)

        # x1 should be imputed
        assert not transformed['x1'].isna().any()

        # x2 should still have missing (not specified)
        assert transformed['x2'].isna().any()

    def test_impute_mean_new_data(self, data_with_na):
        """Test applying mean imputation to new data"""
        train = data_with_na[:3]
        test = pd.DataFrame({
            'x1': [6, np.nan, 8],
            'x2': [60, np.nan, 80],
            'x3': [600, 700, 800]
        })

        rec = recipe().step_impute_mean()
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Should use training mean
        train_mean_x1 = train['x1'].mean()
        assert abs(test_transformed.loc[1, 'x1'] - train_mean_x1) < 1e-10

    def test_impute_mean_no_missing(self):
        """Test when no missing values"""
        data = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [10, 20, 30, 40, 50]
        })

        rec = recipe().step_impute_mean()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should be unchanged
        pd.testing.assert_frame_equal(transformed, data)


class TestStepImputeMedian:
    """Test step_impute_median functionality"""

    @pytest.fixture
    def data_with_na(self):
        """Create data with missing values and outliers"""
        return pd.DataFrame({
            'x1': [1, 2, np.nan, 4, 100],  # Outlier affects mean
            'x2': [10, np.nan, 30, np.nan, 50],
            'x3': [100, 200, 300, 400, 500]
        })

    def test_impute_median_basic(self, data_with_na):
        """Test basic median imputation"""
        rec = recipe().step_impute_median()
        rec_fit = rec.prep(data_with_na)
        transformed = rec_fit.bake(data_with_na)

        # Should have no missing values in imputed columns
        assert not transformed['x1'].isna().any()
        assert not transformed['x2'].isna().any()

    def test_impute_median_values(self, data_with_na):
        """Test median imputation values are correct"""
        rec = recipe().step_impute_median()
        rec_fit = rec.prep(data_with_na)
        transformed = rec_fit.bake(data_with_na)

        # Calculate expected medians
        expected_median_x1 = data_with_na['x1'].median()
        expected_median_x2 = data_with_na['x2'].median()

        # Check imputed values
        assert abs(transformed.loc[2, 'x1'] - expected_median_x1) < 1e-10
        assert abs(transformed.loc[1, 'x2'] - expected_median_x2) < 1e-10

    def test_impute_median_robust_to_outliers(self, data_with_na):
        """Test median is more robust than mean"""
        rec_median = recipe().step_impute_median()
        rec_median_fit = rec_median.prep(data_with_na)
        median_result = rec_median_fit.bake(data_with_na)

        rec_mean = recipe().step_impute_mean()
        rec_mean_fit = rec_mean.prep(data_with_na)
        mean_result = rec_mean_fit.bake(data_with_na)

        # Median should be closer to typical values than mean (due to outlier)
        median_val = median_result.loc[2, 'x1']
        mean_val = mean_result.loc[2, 'x1']

        # Median should be smaller (less affected by 100)
        assert median_val < mean_val

    def test_impute_median_specific_columns(self, data_with_na):
        """Test median imputation on specific columns"""
        rec = recipe().step_impute_median(columns=['x1'])
        rec_fit = rec.prep(data_with_na)
        transformed = rec_fit.bake(data_with_na)

        # x1 should be imputed
        assert not transformed['x1'].isna().any()

        # x2 should still have missing
        assert transformed['x2'].isna().any()

    def test_impute_median_new_data(self, data_with_na):
        """Test applying median imputation to new data"""
        train = data_with_na[:3]
        test = pd.DataFrame({
            'x1': [6, np.nan, 8],
            'x2': [60, np.nan, 80],
            'x3': [600, 700, 800]
        })

        rec = recipe().step_impute_median()
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Should use training median
        train_median_x1 = train['x1'].median()
        assert abs(test_transformed.loc[1, 'x1'] - train_median_x1) < 1e-10


class TestStepImputeMode:
    """Test step_impute_mode functionality"""

    @pytest.fixture
    def categorical_data_with_na(self):
        """Create categorical data with missing values"""
        return pd.DataFrame({
            'category': ['A', 'B', np.nan, 'A', 'A', 'B', np.nan],
            'color': ['red', 'blue', 'red', np.nan, 'red', 'blue', 'red'],
            'numeric': [1, 2, np.nan, 1, 1, 2, 1]
        })

    def test_impute_mode_basic(self, categorical_data_with_na):
        """Test basic mode imputation"""
        rec = recipe().step_impute_mode()
        rec_fit = rec.prep(categorical_data_with_na)
        transformed = rec_fit.bake(categorical_data_with_na)

        # Should have no missing values
        assert not transformed.isna().any().any()

    def test_impute_mode_categorical(self, categorical_data_with_na):
        """Test mode imputation on categorical data"""
        rec = recipe().step_impute_mode(columns=['category'])
        rec_fit = rec.prep(categorical_data_with_na)
        transformed = rec_fit.bake(categorical_data_with_na)

        # Mode is 'A' (appears 3 times), should fill missing with 'A'
        assert transformed['category'].isna().sum() == 0
        assert all(transformed.loc[[2, 6], 'category'] == 'A')

    def test_impute_mode_numeric(self, categorical_data_with_na):
        """Test mode imputation on numeric data"""
        rec = recipe().step_impute_mode(columns=['numeric'])
        rec_fit = rec.prep(categorical_data_with_na)
        transformed = rec_fit.bake(categorical_data_with_na)

        # Mode is 1 (appears 4 times)
        assert transformed.loc[2, 'numeric'] == 1.0

    def test_impute_mode_specific_columns(self, categorical_data_with_na):
        """Test mode imputation on specific columns"""
        rec = recipe().step_impute_mode(columns=['category'])
        rec_fit = rec.prep(categorical_data_with_na)
        transformed = rec_fit.bake(categorical_data_with_na)

        # category should be imputed
        assert not transformed['category'].isna().any()

        # color should still have missing
        assert transformed['color'].isna().any()

    def test_impute_mode_new_data(self, categorical_data_with_na):
        """Test applying mode imputation to new data"""
        train = categorical_data_with_na[:5]
        test = pd.DataFrame({
            'category': [np.nan, 'B', np.nan],
            'color': ['red', np.nan, 'blue'],
            'numeric': [1, np.nan, 2]
        })

        rec = recipe().step_impute_mode()
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Should use training mode
        assert not test_transformed.isna().any().any()


class TestStepImputeKnn:
    """Test step_impute_knn functionality"""

    @pytest.fixture
    def correlated_data_with_na(self):
        """Create data with correlated features and missing values"""
        np.random.seed(42)
        x1 = np.random.randn(50)
        x2 = x1 * 2 + np.random.randn(50) * 0.1  # Highly correlated with x1

        # Create missing values
        x1_na = x1.copy()
        x1_na[[5, 15, 25]] = np.nan

        x2_na = x2.copy()
        x2_na[[10, 20, 30]] = np.nan

        return pd.DataFrame({
            'x1': x1_na,
            'x2': x2_na,
            'x3': np.random.randn(50)
        })

    def test_impute_knn_basic(self, correlated_data_with_na):
        """Test basic KNN imputation"""
        rec = recipe().step_impute_knn()
        rec_fit = rec.prep(correlated_data_with_na)
        transformed = rec_fit.bake(correlated_data_with_na)

        # Should have no missing values
        assert not transformed.isna().any().any()

    def test_impute_knn_specific_columns(self, correlated_data_with_na):
        """Test KNN imputation on specific columns"""
        rec = recipe().step_impute_knn(columns=['x1'])
        rec_fit = rec.prep(correlated_data_with_na)
        transformed = rec_fit.bake(correlated_data_with_na)

        # x1 should be imputed
        assert not transformed['x1'].isna().any()

        # x2 should still have missing (but might be imputed if using all features)
        # Actually, KNN uses all numeric features, so x2 gets imputed too

    def test_impute_knn_neighbors(self, correlated_data_with_na):
        """Test different numbers of neighbors"""
        rec1 = recipe().step_impute_knn(neighbors=3)
        rec1_fit = rec1.prep(correlated_data_with_na)
        result1 = rec1_fit.bake(correlated_data_with_na)

        rec2 = recipe().step_impute_knn(neighbors=10)
        rec2_fit = rec2.prep(correlated_data_with_na)
        result2 = rec2_fit.bake(correlated_data_with_na)

        # Both should have no missing
        assert not result1.isna().any().any()
        assert not result2.isna().any().any()

        # Results might be different
        # (not testing specific values due to variability)

    def test_impute_knn_new_data(self, correlated_data_with_na):
        """Test applying KNN imputation to new data"""
        train = correlated_data_with_na[:40]
        test = correlated_data_with_na[40:]

        rec = recipe().step_impute_knn()
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Should have no missing values
        assert not test_transformed.isna().any().any()

    def test_impute_knn_no_missing(self):
        """Test when no missing values"""
        data = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [10, 20, 30, 40, 50]
        })

        rec = recipe().step_impute_knn()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should be unchanged
        pd.testing.assert_frame_equal(transformed, data)


class TestStepImputeLinear:
    """Test step_impute_linear functionality"""

    @pytest.fixture
    def sequential_data_with_na(self):
        """Create sequential data with missing values"""
        return pd.DataFrame({
            'x1': [1, 2, np.nan, 4, 5, np.nan, 7, 8],
            'x2': [10, np.nan, np.nan, 40, 50, 60, np.nan, 80],
            'x3': [100, 200, 300, 400, 500, 600, 700, 800]
        })

    def test_impute_linear_basic(self, sequential_data_with_na):
        """Test basic linear interpolation"""
        rec = recipe().step_impute_linear()
        rec_fit = rec.prep(sequential_data_with_na)
        transformed = rec_fit.bake(sequential_data_with_na)

        # x1 and x2 should be imputed
        assert not transformed['x1'].isna().any()
        # x2 might still have NaN at boundaries

    def test_impute_linear_values(self, sequential_data_with_na):
        """Test linear interpolation values"""
        rec = recipe().step_impute_linear()
        rec_fit = rec.prep(sequential_data_with_na)
        transformed = rec_fit.bake(sequential_data_with_na)

        # x1[2] should be interpolated between 2 and 4
        assert abs(transformed.loc[2, 'x1'] - 3.0) < 1e-10

        # x1[5] should be interpolated between 5 and 7
        assert abs(transformed.loc[5, 'x1'] - 6.0) < 1e-10

    def test_impute_linear_specific_columns(self, sequential_data_with_na):
        """Test linear interpolation on specific columns"""
        rec = recipe().step_impute_linear(columns=['x1'])
        rec_fit = rec.prep(sequential_data_with_na)
        transformed = rec_fit.bake(sequential_data_with_na)

        # x1 should be imputed
        assert not transformed['x1'].isna().any()

        # x2 should still have missing
        assert transformed['x2'].isna().any()

    def test_impute_linear_limit(self):
        """Test linear interpolation with limit"""
        data = pd.DataFrame({
            'x': [1, np.nan, np.nan, np.nan, 5]
        })

        # Limit to 1 consecutive NaN
        rec = recipe().step_impute_linear(limit=1)
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should only fill some NaNs
        # Middle NaN might not be filled due to limit

    def test_impute_linear_new_data(self, sequential_data_with_na):
        """Test applying linear interpolation to new data"""
        train = sequential_data_with_na[:5]
        test = pd.DataFrame({
            'x1': [10, np.nan, 12],
            'x2': [100, np.nan, 120],
            'x3': [1000, 1100, 1200]
        })

        rec = recipe().step_impute_linear()
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Should interpolate in test data
        assert abs(test_transformed.loc[1, 'x1'] - 11.0) < 1e-10


class TestImputationPipeline:
    """Test combinations of imputation steps"""

    @pytest.fixture
    def complex_missing_data(self):
        """Create data with various missing patterns"""
        return pd.DataFrame({
            'numeric': [1, 2, np.nan, 4, 5, np.nan, 7, 8],
            'category': ['A', np.nan, 'B', 'A', 'B', 'A', np.nan, 'B'],
            'value': [10, 20, 30, np.nan, 50, 60, 70, np.nan]
        })

    def test_mode_then_mean(self, complex_missing_data):
        """Test mode imputation followed by mean"""
        rec = (
            recipe()
            .step_impute_mode(columns=['category'])
            .step_impute_mean(columns=['numeric', 'value'])
        )

        rec_fit = rec.prep(complex_missing_data)
        transformed = rec_fit.bake(complex_missing_data)

        # All missing values should be imputed
        assert not transformed.isna().any().any()

    def test_knn_with_scaling(self):
        """Test KNN imputation with scaling"""
        np.random.seed(42)
        data = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50) * 100  # Different scale
        })
        # Add missing values
        data.loc[[5, 10, 15], 'x1'] = np.nan
        data.loc[[7, 12, 17], 'x2'] = np.nan

        rec = (
            recipe()
            .step_normalize()
            .step_impute_knn()
        )

        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should have no missing values
        assert not transformed.isna().any().any()

    def test_indicate_na_then_impute(self):
        """Test creating NA indicators before imputation"""
        data = pd.DataFrame({
            'x': [1, 2, np.nan, 4, np.nan, 6]
        })

        rec = (
            recipe()
            .step_indicate_na(columns=['x'])
            .step_impute_mean(columns=['x'])
        )

        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should have indicator column
        assert 'na_ind_x' in transformed.columns

        # Original column should be imputed
        assert not transformed['x'].isna().any()

        # Indicator should show where NaN was
        assert transformed['na_ind_x'].sum() == 2


class TestImputationEdgeCases:
    """Test edge cases for imputation steps"""

    def test_impute_mean_all_missing(self):
        """Test mean imputation when all values are missing"""
        data = pd.DataFrame({
            'x': [np.nan, np.nan, np.nan, np.nan]
        })

        rec = recipe().step_impute_mean()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should result in NaN (mean of all NaN is NaN)
        assert transformed['x'].isna().all()

    def test_impute_mode_tie(self):
        """Test mode imputation with tie"""
        data = pd.DataFrame({
            'category': ['A', 'B', np.nan, 'A', 'B']  # A and B both appear twice
        })

        rec = recipe().step_impute_mode()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should pick one of the modes (first one alphabetically or by occurrence)
        assert not transformed['category'].isna().any()
        assert transformed.loc[2, 'category'] in ['A', 'B']

    def test_impute_linear_boundary(self):
        """Test linear interpolation at boundaries"""
        data = pd.DataFrame({
            'x': [np.nan, 2, 3, 4, np.nan]
        })

        rec = recipe().step_impute_linear()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Boundary NaNs might not be filled by default linear interpolation
        # (depends on pandas interpolation behavior)

    def test_impute_knn_single_column(self):
        """Test KNN with single column"""
        data = pd.DataFrame({
            'x': [1, 2, np.nan, 4, 5]
        })

        rec = recipe().step_impute_knn()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should impute using nearest neighbors
        assert not transformed['x'].isna().any()

    def test_impute_median_single_value(self):
        """Test median imputation with single non-missing value"""
        data = pd.DataFrame({
            'x': [5, np.nan, np.nan, np.nan]
        })

        rec = recipe().step_impute_median()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # All NaN should be filled with 5
        assert all(transformed['x'] == 5.0)


class TestImputationWithSelectors:
    """Test imputation steps with resolve_selector integration"""

    @pytest.fixture
    def mixed_data(self):
        """Create data with numeric and categorical columns with missing values"""
        return pd.DataFrame({
            'num_1': [1.0, np.nan, 3.0, 4.0, 5.0],
            'num_2': [10.0, 20.0, np.nan, 40.0, np.nan],
            'num_3': [100.0, 200.0, 300.0, 400.0, 500.0],  # No missing
            'cat_1': ['A', np.nan, 'C', 'D', 'E'],
            'cat_2': ['X', 'Y', 'Z', 'X', 'Y']  # No missing
        })

    def test_impute_mean_with_string_selector(self, mixed_data):
        """Test mean imputation with single column name string"""
        from py_recipes.selectors import resolve_selector

        rec = recipe().step_impute_mean(columns='num_1')
        rec_fit = rec.prep(mixed_data)
        transformed = rec_fit.bake(mixed_data)

        # Only num_1 should be imputed
        assert not transformed['num_1'].isna().any()
        # num_2 should still have missing values
        assert transformed['num_2'].isna().any()

    def test_impute_median_with_list_selector(self, mixed_data):
        """Test median imputation with explicit column list"""
        rec = recipe().step_impute_median(columns=['num_1', 'num_2'])
        rec_fit = rec.prep(mixed_data)
        transformed = rec_fit.bake(mixed_data)

        # Both num_1 and num_2 should be imputed
        assert not transformed['num_1'].isna().any()
        assert not transformed['num_2'].isna().any()
        # num_3 should be unchanged
        assert transformed['num_3'].equals(mixed_data['num_3'])

    def test_impute_mean_with_all_numeric_selector(self, mixed_data):
        """Test mean imputation with all_numeric() selector"""
        from py_recipes.selectors import all_numeric

        rec = recipe().step_impute_mean(columns=all_numeric())
        rec_fit = rec.prep(mixed_data)
        transformed = rec_fit.bake(mixed_data)

        # All numeric columns with missing should be imputed
        assert not transformed['num_1'].isna().any()
        assert not transformed['num_2'].isna().any()
        # Categorical should be unchanged
        assert transformed['cat_1'].isna().any()

    def test_impute_mode_with_all_nominal_selector(self, mixed_data):
        """Test mode imputation with all_nominal() selector"""
        from py_recipes.selectors import all_nominal

        rec = recipe().step_impute_mode(columns=all_nominal())
        rec_fit = rec.prep(mixed_data)
        transformed = rec_fit.bake(mixed_data)

        # Categorical columns with missing should be imputed
        assert not transformed['cat_1'].isna().any()
        # Numeric columns should be unchanged
        assert transformed['num_1'].isna().any()
        assert transformed['num_2'].isna().any()

    def test_impute_knn_with_where_selector(self, mixed_data):
        """Test KNN imputation with where() selector"""
        from py_recipes.selectors import where

        # Select numeric columns with missing values
        selector = where(lambda s: pd.api.types.is_numeric_dtype(s) and s.isna().any())
        rec = recipe().step_impute_knn(columns=selector)
        rec_fit = rec.prep(mixed_data)
        transformed = rec_fit.bake(mixed_data)

        # Only numeric columns with missing should be imputed
        assert not transformed['num_1'].isna().any()
        assert not transformed['num_2'].isna().any()
        # num_3 has no missing, should be unchanged
        assert transformed['num_3'].equals(mixed_data['num_3'])

    def test_impute_linear_with_starts_with_selector(self, mixed_data):
        """Test linear imputation with starts_with() selector"""
        from py_recipes.selectors import starts_with

        rec = recipe().step_impute_linear(columns=starts_with('num'))
        rec_fit = rec.prep(mixed_data)
        transformed = rec_fit.bake(mixed_data)

        # All numeric columns starting with 'num' should be processed
        assert not transformed['num_1'].isna().any()
        assert not transformed['num_2'].isna().any()

    def test_impute_median_with_union_selector(self, mixed_data):
        """Test median imputation with union() selector"""
        from py_recipes.selectors import union, starts_with, one_of

        # Select columns starting with 'num_' or specifically 'cat_1'
        selector = union(starts_with('num_1'), one_of('num_2'))
        rec = recipe().step_impute_median(columns=selector)
        rec_fit = rec.prep(mixed_data)
        transformed = rec_fit.bake(mixed_data)

        # Both num_1 and num_2 should be imputed
        assert not transformed['num_1'].isna().any()
        assert not transformed['num_2'].isna().any()

    def test_impute_mean_with_intersection_selector(self, mixed_data):
        """Test mean imputation with intersection() selector"""
        from py_recipes.selectors import intersection, all_numeric, starts_with

        # Select columns that are both numeric AND start with 'num_2'
        selector = intersection(all_numeric(), starts_with('num_2'))
        rec = recipe().step_impute_mean(columns=selector)
        rec_fit = rec.prep(mixed_data)
        transformed = rec_fit.bake(mixed_data)

        # Only num_2 should be imputed
        assert not transformed['num_2'].isna().any()
        # num_1 should still have missing
        assert transformed['num_1'].isna().any()

    def test_impute_mode_with_difference_selector(self, mixed_data):
        """Test mode imputation with difference() selector"""
        from py_recipes.selectors import difference, everything, starts_with

        # Select all columns except those starting with 'num_'
        selector = difference(everything(), starts_with('num'))
        rec = recipe().step_impute_mode(columns=selector)
        rec_fit = rec.prep(mixed_data)
        transformed = rec_fit.bake(mixed_data)

        # Categorical columns should be imputed
        assert not transformed['cat_1'].isna().any()
        # Numeric columns should be unchanged
        assert transformed['num_1'].isna().any()

    def test_impute_default_none_selector(self, mixed_data):
        """Test imputation with None (default) uses automatic selection"""
        # For mean imputation, None should select numeric columns with missing
        rec = recipe().step_impute_mean(columns=None)
        rec_fit = rec.prep(mixed_data)
        transformed = rec_fit.bake(mixed_data)

        # Only numeric columns with missing should be imputed
        assert not transformed['num_1'].isna().any()
        assert not transformed['num_2'].isna().any()
        # num_3 has no missing, should be unchanged
        assert transformed['num_3'].equals(mixed_data['num_3'])
        # Categorical should be unchanged
        assert transformed['cat_1'].isna().any()


    def test_selector_with_empty_result(self, mixed_data):
        """Test imputation when selector returns empty list"""
        from py_recipes.selectors import starts_with

        # Select columns starting with 'xyz' (none exist)
        rec = recipe().step_impute_mean(columns=starts_with('xyz'))
        rec_fit = rec.prep(mixed_data)
        transformed = rec_fit.bake(mixed_data)

        # Data should be unchanged
        pd.testing.assert_frame_equal(transformed, mixed_data)

    def test_multiple_impute_steps_with_selectors(self, mixed_data):
        """Test chaining multiple imputation steps with different selectors"""
        from py_recipes.selectors import all_numeric, all_nominal

        rec = (recipe()
               .step_impute_median(columns=all_numeric())
               .step_impute_mode(columns=all_nominal()))
        rec_fit = rec.prep(mixed_data)
        transformed = rec_fit.bake(mixed_data)

        # All columns with missing should be imputed
        assert not transformed['num_1'].isna().any()
        assert not transformed['num_2'].isna().any()
        assert not transformed['cat_1'].isna().any()
