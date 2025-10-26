"""
Tests for core Recipe and PreparedRecipe classes
"""

import pytest
import pandas as pd
import numpy as np

from py_recipes import recipe, Recipe, PreparedRecipe


class TestRecipeCreation:
    """Test recipe creation and basic operations"""

    def test_create_empty_recipe(self):
        """Test creating an empty recipe"""
        rec = recipe()
        assert isinstance(rec, Recipe)
        assert len(rec.steps) == 0

    def test_create_recipe_with_template(self):
        """Test creating recipe with template data"""
        data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        rec = recipe(data)
        assert rec.template is not None
        assert rec.template.equals(data)

    def test_recipe_method_chaining(self):
        """Test method chaining for adding steps"""
        rec = (
            recipe()
            .step_normalize()
            .step_dummy(["category"])
            .step_impute_mean()
        )
        assert len(rec.steps) == 3


class TestRecipeNormalize:
    """Test step_normalize functionality"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with numeric columns"""
        np.random.seed(42)
        return pd.DataFrame({
            "feature1": np.random.randn(100) * 10 + 50,
            "feature2": np.random.randn(100) * 5 + 20,
            "feature3": np.random.randn(100) * 2 + 100,
            "id": range(100)
        })

    def test_normalize_all_numeric(self, sample_data):
        """Test normalizing all numeric columns"""
        rec = recipe().step_normalize()
        rec_fit = rec.prep(sample_data)

        assert isinstance(rec_fit, PreparedRecipe)
        assert len(rec_fit.prepared_steps) == 1

        # Bake the data
        transformed = rec_fit.bake(sample_data)
        assert len(transformed) == len(sample_data)

        # Check that numeric columns are standardized (mean ~0, std ~1)
        for col in ["feature1", "feature2", "feature3"]:
            assert abs(transformed[col].mean()) < 0.1
            assert abs(transformed[col].std() - 1.0) < 0.1

    def test_normalize_specific_columns(self, sample_data):
        """Test normalizing specific columns"""
        rec = recipe().step_normalize(columns=["feature1", "feature2"])
        rec_fit = rec.prep(sample_data)
        transformed = rec_fit.bake(sample_data)

        # feature1 and feature2 should be normalized
        assert abs(transformed["feature1"].mean()) < 0.1
        assert abs(transformed["feature2"].mean()) < 0.1

        # feature3 should be unchanged
        assert abs(transformed["feature3"].mean() - sample_data["feature3"].mean()) < 0.1

    def test_normalize_zscore_method(self, sample_data):
        """Test zscore normalization method"""
        rec = recipe().step_normalize(method="zscore")
        rec_fit = rec.prep(sample_data)
        transformed = rec_fit.bake(sample_data)

        # Check standardization
        assert abs(transformed["feature1"].mean()) < 0.1
        assert abs(transformed["feature1"].std() - 1.0) < 0.1

    def test_normalize_minmax_method(self, sample_data):
        """Test minmax normalization method"""
        rec = recipe().step_normalize(method="minmax")
        rec_fit = rec.prep(sample_data)
        transformed = rec_fit.bake(sample_data)

        # Check range [0, 1]
        for col in ["feature1", "feature2", "feature3"]:
            assert transformed[col].min() >= 0
            assert transformed[col].max() <= 1

    def test_normalize_new_data(self, sample_data):
        """Test applying normalization to new data"""
        train = sample_data[:80]
        test = sample_data[80:]

        rec = recipe().step_normalize()
        rec_fit = rec.prep(train)

        # Apply to test data
        test_transformed = rec_fit.bake(test)
        assert len(test_transformed) == len(test)


class TestRecipeDummy:
    """Test step_dummy functionality"""

    @pytest.fixture
    def categorical_data(self):
        """Create sample data with categorical columns"""
        return pd.DataFrame({
            "category": ["A", "B", "C", "A", "B", "C", "A", "B"],
            "group": ["X", "X", "Y", "Y", "X", "Y", "X", "Y"],
            "value": [1, 2, 3, 4, 5, 6, 7, 8]
        })

    def test_dummy_encoding(self, categorical_data):
        """Test one-hot encoding of categorical columns"""
        rec = recipe().step_dummy(["category"])
        rec_fit = rec.prep(categorical_data)
        transformed = rec_fit.bake(categorical_data)

        # Original column should be removed
        assert "category" not in transformed.columns

        # Dummy columns should be created
        assert "category_A" in transformed.columns
        assert "category_B" in transformed.columns
        assert "category_C" in transformed.columns

        # Check values
        assert transformed["category_A"].sum() == 3  # Three A's

    def test_dummy_multiple_columns(self, categorical_data):
        """Test encoding multiple categorical columns"""
        rec = recipe().step_dummy(["category", "group"])
        rec_fit = rec.prep(categorical_data)
        transformed = rec_fit.bake(categorical_data)

        # Check both columns encoded
        assert "category_A" in transformed.columns
        assert "group_X" in transformed.columns
        assert "group_Y" in transformed.columns

    def test_dummy_preserves_other_columns(self, categorical_data):
        """Test that non-encoded columns are preserved"""
        rec = recipe().step_dummy(["category"])
        rec_fit = rec.prep(categorical_data)
        transformed = rec_fit.bake(categorical_data)

        # value column should be preserved
        assert "value" in transformed.columns
        assert transformed["value"].tolist() == categorical_data["value"].tolist()

    def test_dummy_new_data(self, categorical_data):
        """Test applying dummy encoding to new data"""
        train = categorical_data[:6]
        test = categorical_data[6:]

        rec = recipe().step_dummy(["category"])
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        assert len(test_transformed) == len(test)


class TestRecipeImpute:
    """Test imputation steps"""

    @pytest.fixture
    def data_with_na(self):
        """Create data with missing values"""
        return pd.DataFrame({
            "feature1": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0],
            "feature2": [10.0, np.nan, 30.0, 40.0, np.nan, 60.0, 70.0],
            "feature3": [100, 200, 300, 400, 500, 600, 700]  # No NA
        })

    def test_impute_mean(self, data_with_na):
        """Test mean imputation"""
        rec = recipe().step_impute_mean()
        rec_fit = rec.prep(data_with_na)
        transformed = rec_fit.bake(data_with_na)

        # Check no NA values remain
        assert not transformed["feature1"].isna().any()
        assert not transformed["feature2"].isna().any()

        # Check mean is used
        original_mean_f1 = data_with_na["feature1"].mean()
        assert abs(transformed["feature1"].mean() - original_mean_f1) < 0.1

    def test_impute_median(self, data_with_na):
        """Test median imputation"""
        rec = recipe().step_impute_median()
        rec_fit = rec.prep(data_with_na)
        transformed = rec_fit.bake(data_with_na)

        # Check no NA values remain
        assert not transformed["feature1"].isna().any()
        assert not transformed["feature2"].isna().any()

    def test_impute_specific_columns(self, data_with_na):
        """Test imputing specific columns"""
        rec = recipe().step_impute_mean(columns=["feature1"])
        rec_fit = rec.prep(data_with_na)
        transformed = rec_fit.bake(data_with_na)

        # feature1 should be imputed
        assert not transformed["feature1"].isna().any()

        # feature2 should still have NA
        assert transformed["feature2"].isna().any()

    def test_impute_new_data(self, data_with_na):
        """Test applying imputation to new data"""
        train = data_with_na[:5]
        test = data_with_na[5:]

        rec = recipe().step_impute_mean()
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Test data should have no NA
        assert not test_transformed["feature1"].isna().any()


class TestRecipeMutate:
    """Test step_mutate functionality"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        return pd.DataFrame({
            "x1": [1, 2, 3, 4, 5],
            "x2": [10, 20, 30, 40, 50],
            "x3": [2, 4, 6, 8, 10]
        })

    def test_mutate_create_new_column(self, sample_data):
        """Test creating new column with mutate"""
        rec = recipe().step_mutate({
            "x1_squared": lambda df: df["x1"] ** 2
        })
        rec_fit = rec.prep(sample_data)
        transformed = rec_fit.bake(sample_data)

        assert "x1_squared" in transformed.columns
        assert transformed["x1_squared"].tolist() == [1, 4, 9, 16, 25]

    def test_mutate_multiple_columns(self, sample_data):
        """Test creating multiple columns"""
        rec = recipe().step_mutate({
            "sum": lambda df: df["x1"] + df["x2"],
            "product": lambda df: df["x1"] * df["x2"]
        })
        rec_fit = rec.prep(sample_data)
        transformed = rec_fit.bake(sample_data)

        assert "sum" in transformed.columns
        assert "product" in transformed.columns
        assert transformed["sum"].tolist() == [11, 22, 33, 44, 55]

    def test_mutate_log_transform(self, sample_data):
        """Test log transformation with mutate"""
        rec = recipe().step_mutate({
            "log_x2": lambda df: np.log(df["x2"])
        })
        rec_fit = rec.prep(sample_data)
        transformed = rec_fit.bake(sample_data)

        assert "log_x2" in transformed.columns
        assert transformed["log_x2"].iloc[0] == np.log(10)


class TestRecipePipeline:
    """Test multi-step recipe pipelines"""

    @pytest.fixture
    def complex_data(self):
        """Create complex data with various issues"""
        np.random.seed(42)
        return pd.DataFrame({
            "numeric1": np.random.randn(50) * 10 + 50,
            "numeric2": [np.nan if i % 5 == 0 else np.random.randn() * 5 + 20 for i in range(50)],
            "category": np.random.choice(["A", "B", "C"], 50),
            "target": np.random.randn(50)
        })

    def test_multi_step_pipeline(self, complex_data):
        """Test pipeline with multiple steps"""
        rec = (
            recipe()
            .step_impute_mean()
            .step_normalize(columns=["numeric1", "numeric2"])
            .step_dummy(["category"])
        )

        rec_fit = rec.prep(complex_data)
        transformed = rec_fit.bake(complex_data)

        # Check all transformations applied
        assert not transformed["numeric1"].isna().any()  # Imputed
        assert abs(transformed["numeric1"].mean()) < 0.1  # Normalized
        assert "category_A" in transformed.columns  # Dummy encoded

    def test_pipeline_order_matters(self, complex_data):
        """Test that step order affects results"""
        # Normalize then impute (different from impute then normalize)
        rec1 = recipe().step_normalize().step_impute_mean()
        rec2 = recipe().step_impute_mean().step_normalize()

        rec1_fit = rec1.prep(complex_data)
        rec2_fit = rec2.prep(complex_data)

        # Both should work but may give different results
        trans1 = rec1_fit.bake(complex_data)
        trans2 = rec2_fit.bake(complex_data)

        assert len(trans1) == len(trans2)

    def test_prep_and_bake_separately(self, complex_data):
        """Test prep on train and bake on test"""
        train = complex_data[:40]
        test = complex_data[40:]

        rec = (
            recipe()
            .step_impute_mean()
            .step_normalize()
            .step_dummy(["category"])
        )

        # Prep on train
        rec_fit = rec.prep(train)

        # Bake on test
        test_transformed = rec_fit.bake(test)

        assert len(test_transformed) == len(test)
        assert "category_A" in test_transformed.columns

    def test_juice_method(self, complex_data):
        """Test extracting transformed training data"""
        rec = recipe().step_normalize()
        rec_fit = rec.prep(complex_data)

        # juice() should return transformed training data
        juiced = rec_fit.juice()

        assert len(juiced) == len(complex_data)
        assert abs(juiced["numeric1"].mean()) < 0.1


class TestRecipeEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_recipe_prep(self):
        """Test prepping empty recipe"""
        data = pd.DataFrame({"x": [1, 2, 3]})
        rec = recipe()
        rec_fit = rec.prep(data)

        assert isinstance(rec_fit, PreparedRecipe)
        assert len(rec_fit.prepared_steps) == 0

    def test_empty_recipe_bake(self):
        """Test baking with empty recipe"""
        data = pd.DataFrame({"x": [1, 2, 3]})
        rec = recipe()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should return unchanged data
        assert transformed.equals(data)

    def test_normalize_no_numeric_columns(self):
        """Test normalizing data with no numeric columns"""
        data = pd.DataFrame({"cat": ["A", "B", "C"]})
        rec = recipe().step_normalize()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should return unchanged
        assert len(transformed) == len(data)

    def test_impute_no_missing_values(self):
        """Test imputation when there are no missing values"""
        data = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        rec = recipe().step_impute_mean()
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should return unchanged
        assert transformed["x"].tolist() == data["x"].tolist()

    def test_bake_with_new_columns(self):
        """Test baking data with additional columns"""
        train = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        test = pd.DataFrame({"x": [7, 8], "y": [9, 10], "z": [11, 12]})

        rec = recipe().step_normalize()
        rec_fit = rec.prep(train)
        transformed = rec_fit.bake(test)

        # Extra column should be preserved
        assert "z" in transformed.columns

    def test_bake_with_missing_columns(self):
        """Test baking data with missing columns"""
        train = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        test = pd.DataFrame({"x": [7, 8]})  # Missing 'y'

        rec = recipe().step_normalize(columns=["x"])
        rec_fit = rec.prep(train)
        transformed = rec_fit.bake(test)

        # Should still work on available columns
        assert len(transformed) == len(test)
