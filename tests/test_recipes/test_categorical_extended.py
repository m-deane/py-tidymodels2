"""
Tests for extended categorical recipe steps
"""

import pytest
import pandas as pd
import numpy as np

from py_recipes import recipe


class TestStepOther:
    """Test step_other functionality"""

    @pytest.fixture
    def categorical_data(self):
        """Create categorical data with various frequencies"""
        return pd.DataFrame({
            'color': ['red'] * 50 + ['blue'] * 30 + ['green'] * 10 + ['yellow'] * 5 + ['purple'] * 3 + ['orange'] * 2,
            'size': ['small'] * 40 + ['medium'] * 30 + ['large'] * 20 + ['xlarge'] * 10
        })

    def test_other_basic(self, categorical_data):
        """Test basic other pooling"""
        rec = recipe().step_other(columns=['color'], threshold=0.1)
        rec_fit = rec.prep(categorical_data)
        transformed = rec_fit.bake(categorical_data)

        # Low frequency levels should be pooled to 'other'
        unique_colors = set(transformed['color'].unique())
        assert 'other' in unique_colors

        # High frequency levels should remain
        assert 'red' in unique_colors
        assert 'blue' in unique_colors

    def test_other_specific_columns(self, categorical_data):
        """Test other on specific columns"""
        rec = recipe().step_other(columns=['color'], threshold=0.05)
        rec_fit = rec.prep(categorical_data)
        transformed = rec_fit.bake(categorical_data)

        # color should have 'other' category
        assert 'other' in transformed['color'].values

        # size should be unchanged
        assert 'small' in transformed['size'].values
        assert 'other' not in transformed['size'].values

    def test_other_threshold(self, categorical_data):
        """Test different thresholds"""
        # Low threshold - pools more categories
        rec1 = recipe().step_other(columns=['color'], threshold=0.2)
        rec1_fit = rec1.prep(categorical_data)
        result1 = rec1_fit.bake(categorical_data)

        # High threshold - pools fewer categories
        rec2 = recipe().step_other(columns=['color'], threshold=0.02)
        rec2_fit = rec2.prep(categorical_data)
        result2 = rec2_fit.bake(categorical_data)

        # Lower threshold should result in fewer unique levels
        assert len(result1['color'].unique()) <= len(result2['color'].unique())

    def test_other_new_data(self, categorical_data):
        """Test applying other to new data"""
        train = categorical_data[:80]
        test = categorical_data[80:]

        rec = recipe().step_other(columns=['color'], threshold=0.1)
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Should apply same pooling learned from training
        assert len(test_transformed) == len(test)

    def test_other_preserves_shape(self, categorical_data):
        """Test other preserves data shape"""
        rec = recipe().step_other(threshold=0.1)
        rec_fit = rec.prep(categorical_data)
        transformed = rec_fit.bake(categorical_data)

        assert transformed.shape == categorical_data.shape
        assert list(transformed.columns) == list(categorical_data.columns)


class TestStepNovel:
    """Test step_novel functionality"""

    @pytest.fixture
    def training_data(self):
        """Create training data"""
        return pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B', 'C'] * 10,
            'group': ['X', 'Y', 'Z', 'X', 'Y', 'Z'] * 10
        })

    @pytest.fixture
    def test_data_with_novel(self):
        """Create test data with novel categories"""
        return pd.DataFrame({
            'category': ['A', 'B', 'D', 'E'],  # D and E are novel
            'group': ['X', 'Y', 'W', 'Z']  # W is novel
        })

    def test_novel_basic(self, training_data, test_data_with_novel):
        """Test basic novel handling"""
        rec = recipe().step_novel()
        rec_fit = rec.prep(training_data)
        transformed = rec_fit.bake(test_data_with_novel)

        # Novel categories should be replaced with 'new'
        assert 'new' in transformed['category'].values
        assert 'D' not in transformed['category'].values
        assert 'E' not in transformed['category'].values

    def test_novel_specific_columns(self, training_data, test_data_with_novel):
        """Test novel on specific columns"""
        rec = recipe().step_novel(columns=['category'])
        rec_fit = rec.prep(training_data)
        transformed = rec_fit.bake(test_data_with_novel)

        # category should have 'new' for novel values
        assert 'new' in transformed['category'].values

        # group should keep novel values (not specified)
        assert 'W' in transformed['group'].values

    def test_novel_no_novel_values(self, training_data):
        """Test when test data has no novel values"""
        test = training_data[:10]

        rec = recipe().step_novel()
        rec_fit = rec.prep(training_data)
        transformed = rec_fit.bake(test)

        # Should have no 'new' category
        assert 'new' not in transformed['category'].values

    def test_novel_all_novel(self, training_data):
        """Test when all test values are novel"""
        test = pd.DataFrame({
            'category': ['X', 'Y', 'Z'],
            'group': ['P', 'Q', 'R']
        })

        rec = recipe().step_novel()
        rec_fit = rec.prep(training_data)
        transformed = rec_fit.bake(test)

        # All values should be 'new'
        assert all(transformed['category'] == 'new')
        assert all(transformed['group'] == 'new')

    def test_novel_preserves_known(self, training_data, test_data_with_novel):
        """Test that known categories are preserved"""
        rec = recipe().step_novel()
        rec_fit = rec.prep(training_data)
        transformed = rec_fit.bake(test_data_with_novel)

        # Known categories should be preserved
        assert 'A' in transformed['category'].values
        assert 'B' in transformed['category'].values


class TestStepIndicateNa:
    """Test step_indicate_na functionality"""

    @pytest.fixture
    def data_with_na(self):
        """Create data with missing values"""
        return pd.DataFrame({
            'x1': [1, 2, np.nan, 4, 5],
            'x2': [10, np.nan, 30, np.nan, 50],
            'x3': [100, 200, 300, 400, 500]
        })

    def test_indicate_na_basic(self, data_with_na):
        """Test basic NA indicator creation"""
        rec = recipe().step_indicate_na()
        rec_fit = rec.prep(data_with_na)
        transformed = rec_fit.bake(data_with_na)

        # Should create indicator columns
        assert 'na_ind_x1' in transformed.columns
        assert 'na_ind_x2' in transformed.columns

        # x3 has no missing, so no indicator
        assert 'na_ind_x3' not in transformed.columns

    def test_indicate_na_values(self, data_with_na):
        """Test NA indicator values"""
        rec = recipe().step_indicate_na()
        rec_fit = rec.prep(data_with_na)
        transformed = rec_fit.bake(data_with_na)

        # Check indicator values match missingness
        expected_x1 = [0, 0, 1, 0, 0]
        expected_x2 = [0, 1, 0, 1, 0]

        np.testing.assert_array_equal(transformed['na_ind_x1'].values, expected_x1)
        np.testing.assert_array_equal(transformed['na_ind_x2'].values, expected_x2)

    def test_indicate_na_specific_columns(self, data_with_na):
        """Test NA indicator on specific columns"""
        rec = recipe().step_indicate_na(columns=['x1'])
        rec_fit = rec.prep(data_with_na)
        transformed = rec_fit.bake(data_with_na)

        # Only x1 should have indicator
        assert 'na_ind_x1' in transformed.columns
        assert 'na_ind_x2' not in transformed.columns

    def test_indicate_na_no_missing(self):
        """Test with no missing values"""
        data = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [10, 20, 30, 40, 50]
        })

        rec = recipe().step_indicate_na()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # No indicator columns should be created
        assert 'na_ind_x1' not in transformed.columns
        assert 'na_ind_x2' not in transformed.columns

    def test_indicate_na_new_data(self, data_with_na):
        """Test applying NA indicator to new data"""
        train = data_with_na[:3]
        test = pd.DataFrame({
            'x1': [6, np.nan, 8],
            'x2': [60, 70, np.nan],
            'x3': [600, 700, 800]
        })

        rec = recipe().step_indicate_na()
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Should create same indicator columns
        assert 'na_ind_x1' in test_transformed.columns
        assert 'na_ind_x2' in test_transformed.columns

        # Check indicator values
        expected_x1 = [0, 1, 0]
        expected_x2 = [0, 0, 1]
        np.testing.assert_array_equal(test_transformed['na_ind_x1'].values, expected_x1)
        np.testing.assert_array_equal(test_transformed['na_ind_x2'].values, expected_x2)

    def test_indicate_na_preserves_original(self, data_with_na):
        """Test that original columns are preserved"""
        rec = recipe().step_indicate_na()
        rec_fit = rec.prep(data_with_na)
        transformed = rec_fit.bake(data_with_na)

        # Original columns should still exist
        assert 'x1' in transformed.columns
        assert 'x2' in transformed.columns
        assert 'x3' in transformed.columns


class TestStepInteger:
    """Test step_integer functionality"""

    @pytest.fixture
    def categorical_data(self):
        """Create categorical data"""
        return pd.DataFrame({
            'color': ['red', 'blue', 'green', 'red', 'blue'],
            'size': ['small', 'medium', 'large', 'small', 'large'],
            'numeric': [1, 2, 3, 4, 5]
        })

    def test_integer_basic(self, categorical_data):
        """Test basic integer encoding"""
        rec = recipe().step_integer()
        rec_fit = rec.prep(categorical_data)
        transformed = rec_fit.bake(categorical_data)

        # Categorical columns should be integers
        assert transformed['color'].dtype in [np.int32, np.int64]
        assert transformed['size'].dtype in [np.int32, np.int64]

        # Numeric should be unchanged
        assert transformed['numeric'].dtype in [np.int32, np.int64, np.float64]

    def test_integer_specific_columns(self, categorical_data):
        """Test integer encoding on specific columns"""
        rec = recipe().step_integer(columns=['color'])
        rec_fit = rec.prep(categorical_data)
        transformed = rec_fit.bake(categorical_data)

        # color should be integer
        assert transformed['color'].dtype in [np.int32, np.int64]

        # size should be unchanged (object/string)
        assert transformed['size'].dtype == object

    def test_integer_consistent_encoding(self, categorical_data):
        """Test that encoding is consistent"""
        rec = recipe().step_integer(columns=['color'])
        rec_fit = rec.prep(categorical_data)
        transformed = rec_fit.bake(categorical_data)

        # Same categories should have same encoding
        red_indices = categorical_data[categorical_data['color'] == 'red'].index
        assert len(transformed.loc[red_indices, 'color'].unique()) == 1

    def test_integer_new_data(self, categorical_data):
        """Test applying integer encoding to new data"""
        train = categorical_data[:3]
        test = categorical_data[3:]

        rec = recipe().step_integer(columns=['color'])
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Should apply same encoding learned from training
        assert test_transformed['color'].dtype in [np.int32, np.int64]
        assert len(test_transformed) == len(test)

    def test_integer_preserves_shape(self, categorical_data):
        """Test integer encoding preserves shape"""
        rec = recipe().step_integer()
        rec_fit = rec.prep(categorical_data)
        transformed = rec_fit.bake(categorical_data)

        assert transformed.shape == categorical_data.shape
        assert list(transformed.columns) == list(categorical_data.columns)

    def test_integer_zero_based(self, categorical_data):
        """Test that encoding is zero-based"""
        rec = recipe().step_integer(columns=['color'])
        rec_fit = rec.prep(categorical_data)
        transformed = rec_fit.bake(categorical_data)

        # Encoded values should start from 0
        assert transformed['color'].min() >= 0


class TestStepUnknown:
    """Test step_unknown functionality"""

    @pytest.fixture
    def categorical_data_with_na(self):
        """Create categorical data with missing values"""
        return pd.DataFrame({
            'color': ['red', 'blue', None, 'red', None, 'green'],
            'size': ['small', 'large', 'medium', None, 'small', None],
            'category': ['A', 'B', 'C', 'A', 'B', 'C']  # No missing
        })

    def test_unknown_basic(self, categorical_data_with_na):
        """Test basic unknown level assignment"""
        rec = recipe().step_unknown()
        rec_fit = rec.prep(categorical_data_with_na)
        transformed = rec_fit.bake(categorical_data_with_na)

        # Missing values should be replaced with '_unknown_'
        assert '_unknown_' in transformed['color'].values
        assert '_unknown_' in transformed['size'].values

        # No NaN should remain in categorical columns
        assert not transformed['color'].isna().any()
        assert not transformed['size'].isna().any()

    def test_unknown_specific_columns(self, categorical_data_with_na):
        """Test unknown on specific columns"""
        rec = recipe().step_unknown(columns=['color'])
        rec_fit = rec.prep(categorical_data_with_na)
        transformed = rec_fit.bake(categorical_data_with_na)

        # color should have '_unknown_'
        assert '_unknown_' in transformed['color'].values
        assert not transformed['color'].isna().any()

        # size should still have NaN (not specified)
        assert transformed['size'].isna().any()
        assert '_unknown_' not in transformed['size'].values

    def test_unknown_custom_label(self, categorical_data_with_na):
        """Test unknown with custom label"""
        rec = recipe().step_unknown(unknown_label='MISSING')
        rec_fit = rec.prep(categorical_data_with_na)
        transformed = rec_fit.bake(categorical_data_with_na)

        # Should use custom label
        assert 'MISSING' in transformed['color'].values
        assert 'MISSING' in transformed['size'].values
        assert '_unknown_' not in transformed['color'].values

    def test_unknown_no_missing(self):
        """Test unknown when no missing values exist"""
        data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B'],
            'group': ['X', 'Y', 'Z', 'X', 'Y']
        })

        rec = recipe().step_unknown()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # No unknown category should be added
        assert '_unknown_' not in transformed['category'].values
        assert '_unknown_' not in transformed['group'].values

        # Data should be unchanged
        assert transformed.equals(data)

    def test_unknown_all_missing(self):
        """Test unknown when all values are missing"""
        data = pd.DataFrame({
            'category': [None, None, None, None]
        })

        rec = recipe().step_unknown()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # All values should be '_unknown_'
        assert all(transformed['category'] == '_unknown_')
        assert not transformed['category'].isna().any()

    def test_unknown_new_data(self, categorical_data_with_na):
        """Test applying unknown to new data"""
        train = categorical_data_with_na[:3]
        test = pd.DataFrame({
            'color': ['red', None, 'yellow'],
            'size': [None, 'xlarge', 'small'],
            'category': ['A', 'B', 'C']
        })

        rec = recipe().step_unknown()
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Missing values in test should be replaced
        assert '_unknown_' in test_transformed['color'].values
        assert '_unknown_' in test_transformed['size'].values
        assert not test_transformed['color'].isna().any()
        assert not test_transformed['size'].isna().any()

    def test_unknown_preserves_known_values(self, categorical_data_with_na):
        """Test that known values are preserved"""
        rec = recipe().step_unknown()
        rec_fit = rec.prep(categorical_data_with_na)
        transformed = rec_fit.bake(categorical_data_with_na)

        # Known values should be unchanged
        assert 'red' in transformed['color'].values
        assert 'blue' in transformed['color'].values
        assert 'green' in transformed['color'].values
        assert 'small' in transformed['size'].values
        assert 'medium' in transformed['size'].values
        assert 'large' in transformed['size'].values

    def test_unknown_preserves_shape(self, categorical_data_with_na):
        """Test unknown preserves data shape"""
        rec = recipe().step_unknown()
        rec_fit = rec.prep(categorical_data_with_na)
        transformed = rec_fit.bake(categorical_data_with_na)

        assert transformed.shape == categorical_data_with_na.shape
        assert list(transformed.columns) == list(categorical_data_with_na.columns)

    def test_unknown_with_numeric(self):
        """Test unknown step ignores numeric columns"""
        data = pd.DataFrame({
            'category': ['A', None, 'B'],
            'numeric': [1, 2, 3],
            'float_col': [1.5, np.nan, 3.5]
        })

        rec = recipe().step_unknown()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Categorical should have '_unknown_'
        assert '_unknown_' in transformed['category'].values

        # Numeric columns should be unchanged
        np.testing.assert_array_equal(transformed['numeric'].values, data['numeric'].values)
        # Float NaN should remain (not categorical)
        assert pd.isna(transformed['float_col'].iloc[1])


class TestCategoricalPipeline:
    """Test combinations of categorical steps"""

    @pytest.fixture
    def complex_categorical_data(self):
        """Create complex categorical data"""
        return pd.DataFrame({
            'category': ['A'] * 50 + ['B'] * 30 + ['C'] * 15 + ['D'] * 3 + ['E'] * 2,
            'group': ['X', 'Y', 'Z'] * 33 + ['X'],
            'value': np.random.randn(100)
        })

    def test_other_then_integer(self, complex_categorical_data):
        """Test other pooling followed by integer encoding"""
        rec = (
            recipe()
            .step_other(columns=['category'], threshold=0.1)
            .step_integer(columns=['category'])
        )

        rec_fit = rec.prep(complex_categorical_data)
        transformed = rec_fit.bake(complex_categorical_data)

        # Should be integer encoded with 'other' category
        assert transformed['category'].dtype in [np.int32, np.int64]

    def test_novel_then_integer(self, complex_categorical_data):
        """Test novel handling followed by integer encoding"""
        train = complex_categorical_data[:80]
        test = pd.DataFrame({
            'category': ['A', 'B', 'F', 'G'],  # F and G are novel
            'group': ['X', 'Y', 'W', 'Z'],  # W is novel
            'value': [1, 2, 3, 4]
        })

        rec = (
            recipe()
            .step_novel(columns=['category'])
            .step_integer(columns=['category'])
        )

        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Known values should be encoded, novel values may be NaN or encoded
        # depending on implementation
        assert len(test_transformed) == len(test)
        # At minimum, known categories should be encoded correctly
        assert not pd.isna(test_transformed.loc[0, 'category'])  # 'A' was in training
        assert not pd.isna(test_transformed.loc[1, 'category'])  # 'B' was in training

    def test_unknown_then_integer(self):
        """Test unknown handling followed by integer encoding"""
        data = pd.DataFrame({
            'category': ['A', None, 'B', 'C', None, 'A']
        })

        rec = (
            recipe()
            .step_unknown(columns=['category'])
            .step_integer(columns=['category'])
        )

        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should be integer encoded
        assert transformed['category'].dtype in [np.int32, np.int64]

        # No NaN should remain
        assert not transformed['category'].isna().any()

        # All values should be encoded (including the '_unknown_' category)
        assert len(transformed) == len(data)

    def test_indicate_na_with_imputation(self):
        """Test NA indicator with subsequent imputation"""
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

        # Original column should be imputed (no NaN)
        assert not transformed['x'].isna().any()

        # Indicator should still show where NaN was
        assert transformed['na_ind_x'].sum() == 2


class TestCategoricalEdgeCases:
    """Test edge cases for categorical steps"""

    def test_other_all_frequent(self):
        """Test other when all categories are frequent"""
        data = pd.DataFrame({
            'category': ['A'] * 40 + ['B'] * 30 + ['C'] * 30
        })

        rec = recipe().step_other(threshold=0.1)
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # No 'other' category should be created
        assert 'other' not in transformed['category'].values

    def test_integer_single_category(self):
        """Test integer encoding with single category"""
        data = pd.DataFrame({
            'category': ['A', 'A', 'A', 'A', 'A']
        })

        rec = recipe().step_integer()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should encode to single value
        assert len(transformed['category'].unique()) == 1
        assert transformed['category'].dtype in [np.int32, np.int64]

    def test_indicate_na_all_missing(self):
        """Test NA indicator when all values are missing"""
        data = pd.DataFrame({
            'x': [np.nan, np.nan, np.nan, np.nan]
        })

        rec = recipe().step_indicate_na()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should create indicator with all 1s
        assert 'na_ind_x' in transformed.columns
        assert all(transformed['na_ind_x'] == 1)

    def test_novel_with_numeric(self):
        """Test novel step ignores numeric columns"""
        train = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'numeric': [1, 2, 3]
        })
        test = pd.DataFrame({
            'category': ['A', 'D'],
            'numeric': [4, 5]
        })

        rec = recipe().step_novel()
        rec_fit = rec.prep(train)
        transformed = rec_fit.bake(test)

        # Categorical should have 'new'
        assert 'new' in transformed['category'].values

        # Numeric should be unchanged
        np.testing.assert_array_equal(transformed['numeric'].values, test['numeric'].values)
