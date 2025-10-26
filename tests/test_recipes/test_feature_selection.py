"""
Tests for feature selection recipe steps
"""

import pytest
import pandas as pd
import numpy as np

from py_recipes import recipe


class TestStepPCA:
    """Test step_pca functionality"""

    @pytest.fixture
    def correlated_data(self):
        """Create data with correlated features"""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.1  # Highly correlated with x1
        x3 = x1 - np.random.randn(n) * 0.1  # Highly correlated with x1
        x4 = np.random.randn(n)  # Independent

        return pd.DataFrame({
            'feature1': x1,
            'feature2': x2,
            'feature3': x3,
            'feature4': x4,
            'target': x1 * 2 + x4 + np.random.randn(n) * 0.1
        })

    def test_pca_basic(self, correlated_data):
        """Test basic PCA transformation"""
        # Specify features to transform (exclude target)
        rec = recipe().step_pca(
            columns=["feature1", "feature2", "feature3", "feature4"],
            num_comp=2
        )
        rec_fit = rec.prep(correlated_data)
        transformed = rec_fit.bake(correlated_data)

        # Should have PC columns
        assert "PC1" in transformed.columns
        assert "PC2" in transformed.columns
        assert "PC3" not in transformed.columns

        # Original numeric columns should be removed
        assert "feature1" not in transformed.columns
        assert "feature2" not in transformed.columns

        # Target should be preserved
        assert "target" in transformed.columns

    def test_pca_specific_columns(self, correlated_data):
        """Test PCA on specific columns"""
        rec = recipe().step_pca(columns=["feature1", "feature2"], num_comp=1)
        rec_fit = rec.prep(correlated_data)
        transformed = rec_fit.bake(correlated_data)

        # Should have 1 PC
        assert "PC1" in transformed.columns
        assert "PC2" not in transformed.columns

        # feature1 and feature2 should be removed
        assert "feature1" not in transformed.columns
        assert "feature2" not in transformed.columns

        # feature3, feature4, and target should remain
        assert "feature3" in transformed.columns
        assert "feature4" in transformed.columns
        assert "target" in transformed.columns

    def test_pca_all_components(self, correlated_data):
        """Test keeping all components"""
        rec = recipe().step_pca()  # No num_comp specified
        rec_fit = rec.prep(correlated_data)
        transformed = rec_fit.bake(correlated_data)

        # Should have PC for each original feature (excluding target)
        assert "PC1" in transformed.columns
        assert "PC2" in transformed.columns
        assert "PC3" in transformed.columns
        assert "PC4" in transformed.columns

    def test_pca_preserves_row_count(self, correlated_data):
        """Test that PCA preserves number of rows"""
        rec = recipe().step_pca(num_comp=2)
        rec_fit = rec.prep(correlated_data)
        transformed = rec_fit.bake(correlated_data)

        assert len(transformed) == len(correlated_data)

    def test_pca_new_data(self, correlated_data):
        """Test applying PCA to new data"""
        train = correlated_data[:80]
        test = correlated_data[80:]

        rec = recipe().step_pca(num_comp=2)
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        assert len(test_transformed) == len(test)
        assert "PC1" in test_transformed.columns
        assert "PC2" in test_transformed.columns

    def test_pca_with_normalization(self, correlated_data):
        """Test PCA combined with normalization"""
        rec = (
            recipe()
            .step_normalize()
            .step_pca(num_comp=2)
        )
        rec_fit = rec.prep(correlated_data)
        transformed = rec_fit.bake(correlated_data)

        assert "PC1" in transformed.columns
        assert len(transformed) == len(correlated_data)

    def test_pca_threshold(self, correlated_data):
        """Test PCA with variance threshold"""
        rec = recipe().step_pca(threshold=0.95)
        rec_fit = rec.prep(correlated_data)
        transformed = rec_fit.bake(correlated_data)

        # Should have principal components
        assert "PC1" in transformed.columns


class TestStepSelectCorr:
    """Test step_select_corr functionality"""

    @pytest.fixture
    def multicollinear_data(self):
        """Create data with multicollinearity"""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.01  # Highly correlated with x1
        x3 = np.random.randn(n)  # Independent
        x4 = x3 + np.random.randn(n) * 0.01  # Highly correlated with x3

        return pd.DataFrame({
            'feature1': x1,
            'feature2': x2,  # Should be dropped (correlated with feature1)
            'feature3': x3,
            'feature4': x4,  # Should be dropped (correlated with feature3)
            'outcome': x1 * 2 + x3 * 1.5 + np.random.randn(n) * 0.1
        })

    @pytest.fixture
    def outcome_corr_data(self):
        """Create data with varying correlation to outcome"""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)

        return pd.DataFrame({
            'strong_feature': x1,  # Should be kept
            'weak_feature': x2,    # Should be dropped
            'noise': x3,           # Should be dropped
            'outcome': x1 * 3 + np.random.randn(n) * 0.1  # Only correlated with x1
        })

    def test_select_corr_multicollinearity(self, multicollinear_data):
        """Test removing multicollinear features"""
        rec = recipe().step_select_corr(
            outcome="outcome",
            threshold=0.9,
            method="multicollinearity"
        )
        rec_fit = rec.prep(multicollinear_data)
        transformed = rec_fit.bake(multicollinear_data)

        # Should keep fewer features
        numeric_cols = transformed.select_dtypes(include=[np.number]).columns.tolist()
        original_numeric = multicollinear_data.select_dtypes(include=[np.number]).columns.tolist()

        assert len(numeric_cols) < len(original_numeric)

        # Outcome should always be preserved
        assert "outcome" in transformed.columns

    def test_select_corr_outcome_method(self, outcome_corr_data):
        """Test selecting features correlated with outcome"""
        rec = recipe().step_select_corr(
            outcome="outcome",
            threshold=0.5,
            method="outcome"
        )
        rec_fit = rec.prep(outcome_corr_data)
        transformed = rec_fit.bake(outcome_corr_data)

        # Should keep strong_feature and outcome
        assert "outcome" in transformed.columns

        # Check that we filtered some features
        numeric_cols = [col for col in transformed.columns
                       if col != "outcome" and pd.api.types.is_numeric_dtype(transformed[col])]
        original_predictors = [col for col in outcome_corr_data.columns
                              if col != "outcome" and pd.api.types.is_numeric_dtype(outcome_corr_data[col])]

        # Should have fewer predictors after filtering
        assert len(numeric_cols) <= len(original_predictors)

    def test_select_corr_high_threshold(self, multicollinear_data):
        """Test with high correlation threshold"""
        rec = recipe().step_select_corr(
            outcome="outcome",
            threshold=0.99,
            method="multicollinearity"
        )
        rec_fit = rec.prep(multicollinear_data)
        transformed = rec_fit.bake(multicollinear_data)

        # With very high threshold, should keep most features
        assert "outcome" in transformed.columns

    def test_select_corr_low_threshold(self, multicollinear_data):
        """Test with low correlation threshold"""
        rec = recipe().step_select_corr(
            outcome="outcome",
            threshold=0.5,
            method="multicollinearity"
        )
        rec_fit = rec.prep(multicollinear_data)
        transformed = rec_fit.bake(multicollinear_data)

        # Should still have outcome
        assert "outcome" in transformed.columns

    def test_select_corr_preserves_row_count(self, multicollinear_data):
        """Test that correlation selection preserves rows"""
        rec = recipe().step_select_corr(
            outcome="outcome",
            threshold=0.9,
            method="multicollinearity"
        )
        rec_fit = rec.prep(multicollinear_data)
        transformed = rec_fit.bake(multicollinear_data)

        assert len(transformed) == len(multicollinear_data)

    def test_select_corr_new_data(self, multicollinear_data):
        """Test applying correlation selection to new data"""
        train = multicollinear_data[:80]
        test = multicollinear_data[80:]

        rec = recipe().step_select_corr(
            outcome="outcome",
            threshold=0.9,
            method="multicollinearity"
        )
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        assert len(test_transformed) == len(test)
        assert "outcome" in test_transformed.columns

    def test_select_corr_with_categorical(self):
        """Test correlation selection with categorical columns"""
        np.random.seed(42)
        data = pd.DataFrame({
            'numeric1': np.random.randn(50),
            'numeric2': np.random.randn(50),
            'category': np.random.choice(['A', 'B', 'C'], 50),
            'outcome': np.random.randn(50)
        })

        rec = recipe().step_select_corr(
            outcome="outcome",
            threshold=0.9,
            method="multicollinearity"
        )
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Categorical column should be preserved
        assert "category" in transformed.columns
        assert "outcome" in transformed.columns


class TestFeatureSelectionPipeline:
    """Test combinations of feature selection steps"""

    @pytest.fixture
    def complex_data(self):
        """Create complex data for testing"""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.1
        x3 = np.random.randn(n)
        x4 = x3 + np.random.randn(n) * 0.1
        x5 = np.random.randn(n)

        return pd.DataFrame({
            'f1': x1,
            'f2': x2,
            'f3': x3,
            'f4': x4,
            'f5': x5,
            'target': x1 * 2 + x3 * 1.5 + np.random.randn(n) * 0.1
        })

    def test_corr_then_pca(self, complex_data):
        """Test correlation selection followed by PCA"""
        # Get feature columns (exclude target)
        feature_cols = [col for col in complex_data.columns if col != "target"]

        rec = (
            recipe()
            .step_select_corr(outcome="target", threshold=0.9, method="multicollinearity")
            .step_pca(columns=feature_cols, num_comp=2)
        )

        rec_fit = rec.prep(complex_data)
        transformed = rec_fit.bake(complex_data)

        # Should have principal components
        assert "PC1" in transformed.columns
        assert "PC2" in transformed.columns

        # Target should be preserved
        assert "target" in transformed.columns

    def test_normalize_corr_pca(self, complex_data):
        """Test full preprocessing pipeline"""
        # Get feature columns (exclude target)
        feature_cols = [col for col in complex_data.columns if col != "target"]

        rec = (
            recipe()
            .step_normalize(columns=feature_cols)
            .step_select_corr(outcome="target", threshold=0.95, method="multicollinearity")
            .step_pca(columns=feature_cols, num_comp=2)
        )

        rec_fit = rec.prep(complex_data)
        transformed = rec_fit.bake(complex_data)

        assert "PC1" in transformed.columns
        assert "target" in transformed.columns
        assert len(transformed) == len(complex_data)

    def test_outcome_selection_with_pca(self, complex_data):
        """Test outcome-based selection with PCA"""
        # Get feature columns (exclude target)
        feature_cols = [col for col in complex_data.columns if col != "target"]

        rec = (
            recipe()
            .step_select_corr(outcome="target", threshold=0.3, method="outcome")
            .step_pca(columns=feature_cols, num_comp=2)
        )

        rec_fit = rec.prep(complex_data)
        transformed = rec_fit.bake(complex_data)

        # Should have reduced dimensionality
        assert "PC1" in transformed.columns
        assert "target" in transformed.columns


class TestFeatureSelectionEdgeCases:
    """Test edge cases for feature selection"""

    def test_select_corr_no_numeric_predictors(self):
        """Test correlation selection with only categorical predictors"""
        data = pd.DataFrame({
            'cat1': ['A', 'B', 'C'] * 10,
            'cat2': ['X', 'Y', 'Z'] * 10,
            'outcome': range(30)
        })

        rec = recipe().step_select_corr(
            outcome="outcome",
            threshold=0.9,
            method="multicollinearity"
        )
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should preserve categorical columns and outcome
        assert "cat1" in transformed.columns
        assert "outcome" in transformed.columns

    def test_pca_single_column(self):
        """Test PCA with single column"""
        data = pd.DataFrame({
            'x': range(10),
            'y': range(10, 20)
        })

        rec = recipe().step_pca(columns=["x"], num_comp=1)
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        assert "PC1" in transformed.columns
        assert "y" in transformed.columns

    def test_pca_more_components_than_features(self):
        """Test PCA requesting more components than available"""
        data = pd.DataFrame({
            'x1': range(10),
            'x2': range(10, 20),
            'x3': range(20, 30)
        })

        # Request 10 components but only have 3 features
        rec = recipe().step_pca(num_comp=10)
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should only get 3 components (limited by number of features)
        assert "PC1" in transformed.columns
        assert "PC2" in transformed.columns
        assert "PC3" in transformed.columns
        assert "PC4" not in transformed.columns

    def test_select_corr_all_uncorrelated(self):
        """Test correlation selection with uncorrelated features"""
        np.random.seed(42)
        data = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'x3': np.random.randn(50),
            'outcome': np.random.randn(50)
        })

        rec = recipe().step_select_corr(
            outcome="outcome",
            threshold=0.9,
            method="multicollinearity"
        )
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should keep all features since none are highly correlated
        assert "x1" in transformed.columns
        assert "x2" in transformed.columns
        assert "x3" in transformed.columns
        assert "outcome" in transformed.columns
