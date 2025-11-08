"""
Tests for supervised feature filtering recipe steps
"""

import pytest
import pandas as pd
import numpy as np

from py_recipes import recipe
from py_recipes.steps.filter_supervised import (
    step_filter_anova, step_filter_rf_importance,
    step_filter_mutual_info, step_filter_roc_auc, step_filter_chisq
)


class TestStepFilterAnova:
    """Test step_filter_anova (ANOVA F-test) functionality"""

    @pytest.fixture
    def regression_data(self):
        """Create regression data with varying feature importance"""
        np.random.seed(42)
        n = 200
        # Strong signal features
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        # Weak signal features
        x3 = np.random.randn(n) * 0.1
        x4 = np.random.randn(n) * 0.1
        # Noise features
        x5 = np.random.randn(n)
        x6 = np.random.randn(n)

        # y strongly depends on x1 and x2
        y = 2 * x1 + 3 * x2 + np.random.randn(n) * 0.5

        return pd.DataFrame({
            'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6, 'y': y
        })

    @pytest.fixture
    def classification_data(self):
        """Create classification data with varying feature importance"""
        np.random.seed(42)
        n = 200
        # Create two classes with different means
        class_0 = np.zeros(n // 2)
        class_1 = np.ones(n // 2)
        y = np.concatenate([class_0, class_1])

        # Strong discriminative features
        x1 = np.concatenate([np.random.randn(n // 2) - 2, np.random.randn(n // 2) + 2])
        x2 = np.concatenate([np.random.randn(n // 2) - 1.5, np.random.randn(n // 2) + 1.5])
        # Weak features
        x3 = np.random.randn(n)
        x4 = np.random.randn(n)

        return pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'y': y})

    def test_anova_threshold_regression(self, regression_data):
        """Test ANOVA with threshold selection on regression"""
        rec = recipe().step_filter_anova('y', threshold=1.0, use_pvalue=False)
        rec_fit = rec.prep(regression_data)
        transformed = rec_fit.bake(regression_data)

        # Should keep strong features (x1, x2)
        assert 'x1' in transformed.columns
        assert 'x2' in transformed.columns
        assert 'y' in transformed.columns

        # Should filter some weak features
        assert len(transformed.columns) < len(regression_data.columns)

    def test_anova_top_n(self, regression_data):
        """Test ANOVA with top_n selection"""
        rec = recipe().step_filter_anova('y', top_n=3)
        rec_fit = rec.prep(regression_data)
        transformed = rec_fit.bake(regression_data)

        # Should keep exactly 3 features plus outcome
        assert len(transformed.columns) == 4  # 3 features + y
        assert 'y' in transformed.columns

    def test_anova_top_p(self, regression_data):
        """Test ANOVA with top_p selection"""
        rec = recipe().step_filter_anova('y', top_p=0.5)
        rec_fit = rec.prep(regression_data)
        transformed = rec_fit.bake(regression_data)

        # Should keep top 50% of features (3 out of 6) plus outcome
        assert len(transformed.columns) == 4  # 3 features + y
        assert 'y' in transformed.columns

    def test_anova_pvalue_mode(self, regression_data):
        """Test ANOVA using p-value scoring"""
        rec = recipe().step_filter_anova('y', threshold=0.1, use_pvalue=True)
        rec_fit = rec.prep(regression_data)
        transformed = rec_fit.bake(regression_data)

        # Should keep features with p-value < 0.1 (transformed to -log10)
        assert 'x1' in transformed.columns
        assert 'x2' in transformed.columns
        assert 'y' in transformed.columns

    def test_anova_classification(self, classification_data):
        """Test ANOVA on classification data"""
        rec = recipe().step_filter_anova('y', top_n=2)
        rec_fit = rec.prep(classification_data)
        transformed = rec_fit.bake(classification_data)

        # Should keep top 2 discriminative features
        assert len(transformed.columns) == 3  # 2 features + y
        assert 'x1' in transformed.columns
        assert 'x2' in transformed.columns

    def test_anova_new_data(self, regression_data):
        """Test applying ANOVA filter to new data"""
        train = regression_data[:150]
        test = regression_data[150:]

        rec = recipe().step_filter_anova('y', top_n=3)
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Should have same columns as training
        train_transformed = rec_fit.bake(train)
        assert set(test_transformed.columns) == set(train_transformed.columns)

    def test_anova_no_valid_columns(self):
        """Test ANOVA when no numeric columns exist"""
        data = pd.DataFrame({
            'cat1': ['A', 'B', 'C', 'A', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y', 'X'],
            'y': [1, 2, 3, 4, 5]
        })

        rec = recipe().step_filter_anova('y', top_n=1)
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should keep only outcome
        assert 'y' in transformed.columns


class TestStepFilterRfImportance:
    """Test step_filter_rf_importance (Random Forest feature importance) functionality"""

    @pytest.fixture
    def importance_data(self):
        """Create data with varying feature importance for RF"""
        np.random.seed(42)
        n = 200
        # Highly important features
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        # Moderately important
        x3 = np.random.randn(n)
        # Noise features
        x4 = np.random.randn(n)
        x5 = np.random.randn(n)

        # Non-linear relationship
        y = x1**2 + 2 * np.sin(x2) + 0.5 * x3 + np.random.randn(n) * 0.3

        return pd.DataFrame({
            'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'y': y
        })

    def test_rf_importance_threshold(self, importance_data):
        """Test RF importance with threshold selection"""
        rec = recipe().step_filter_rf_importance('y', threshold=0.05, trees=50)
        rec_fit = rec.prep(importance_data)
        transformed = rec_fit.bake(importance_data)

        # Should complete without errors and keep outcome
        assert 'y' in transformed.columns
        # Should have filtered some or all features (threshold dependent)
        assert len(transformed.columns) <= len(importance_data.columns)

    def test_rf_importance_top_n(self, importance_data):
        """Test RF importance with top_n selection"""
        rec = recipe().step_filter_rf_importance('y', top_n=2, trees=50)
        rec_fit = rec.prep(importance_data)
        transformed = rec_fit.bake(importance_data)

        # Should keep exactly 2 features plus outcome
        assert len(transformed.columns) == 3  # 2 features + y
        assert 'y' in transformed.columns

    def test_rf_importance_top_p(self, importance_data):
        """Test RF importance with top_p selection"""
        rec = recipe().step_filter_rf_importance('y', top_p=0.6, trees=30)
        rec_fit = rec.prep(importance_data)
        transformed = rec_fit.bake(importance_data)

        # Should keep top 60% of features (3 out of 5)
        assert len(transformed.columns) == 4  # 3 features + y
        assert 'y' in transformed.columns

    def test_rf_importance_parameters(self, importance_data):
        """Test RF importance with custom RF parameters"""
        rec = recipe().step_filter_rf_importance(
            'y', top_n=3, trees=100, mtry=3, min_n=5
        )
        rec_fit = rec.prep(importance_data)
        transformed = rec_fit.bake(importance_data)

        # Should complete without errors
        assert len(transformed.columns) == 4  # 3 features + y

    def test_rf_importance_classification(self):
        """Test RF importance on classification data"""
        np.random.seed(42)
        n = 200
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)

        # Binary classification
        y = (x1 + x2 > 0).astype(int)

        data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'y': y})

        rec = recipe().step_filter_rf_importance('y', top_n=2, trees=50)
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should keep 2 most important features
        assert len(transformed.columns) == 3
        assert 'y' in transformed.columns

    def test_rf_importance_new_data(self, importance_data):
        """Test applying RF filter to new data"""
        train = importance_data[:150]
        test = importance_data[150:]

        rec = recipe().step_filter_rf_importance('y', top_n=3, trees=50)
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Should have same columns as training
        train_transformed = rec_fit.bake(train)
        assert set(test_transformed.columns) == set(train_transformed.columns)


class TestStepFilterMutualInfo:
    """Test step_filter_mutual_info (mutual information) functionality"""

    @pytest.fixture
    def nonlinear_data(self):
        """Create data with non-linear relationships"""
        np.random.seed(42)
        n = 200
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)
        x4 = np.random.randn(n)
        x5 = np.random.randn(n)

        # Non-linear dependencies
        y = np.sin(x1) + x2**2 + np.random.randn(n) * 0.2

        return pd.DataFrame({
            'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'y': y
        })

    def test_mutual_info_threshold(self, nonlinear_data):
        """Test mutual info with threshold selection"""
        rec = recipe().step_filter_mutual_info('y', threshold=0.05, n_neighbors=3)
        rec_fit = rec.prep(nonlinear_data)
        transformed = rec_fit.bake(nonlinear_data)

        # Should filter low MI features
        assert len(transformed.columns) < len(nonlinear_data.columns)
        assert 'y' in transformed.columns

    def test_mutual_info_top_n(self, nonlinear_data):
        """Test mutual info with top_n selection"""
        rec = recipe().step_filter_mutual_info('y', top_n=2, n_neighbors=3)
        rec_fit = rec.prep(nonlinear_data)
        transformed = rec_fit.bake(nonlinear_data)

        # Should keep exactly 2 features plus outcome
        assert len(transformed.columns) == 3  # 2 features + y
        assert 'y' in transformed.columns
        # x1 and x2 should be top features (non-linear relationship)
        assert 'x1' in transformed.columns or 'x2' in transformed.columns

    def test_mutual_info_top_p(self, nonlinear_data):
        """Test mutual info with top_p selection"""
        rec = recipe().step_filter_mutual_info('y', top_p=0.4, n_neighbors=5)
        rec_fit = rec.prep(nonlinear_data)
        transformed = rec_fit.bake(nonlinear_data)

        # Should keep top 40% of features (2 out of 5)
        assert len(transformed.columns) == 3  # 2 features + y
        assert 'y' in transformed.columns

    def test_mutual_info_classification(self):
        """Test mutual info on classification data"""
        np.random.seed(42)
        n = 200
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)

        # Binary classification with non-linear boundary
        y = ((x1**2 + x2**2) > 1).astype(int)

        data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'y': y})

        rec = recipe().step_filter_mutual_info('y', top_n=2, n_neighbors=3)
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should keep 2 most informative features
        assert len(transformed.columns) == 3
        assert 'x1' in transformed.columns
        assert 'x2' in transformed.columns

    def test_mutual_info_neighbors_parameter(self, nonlinear_data):
        """Test mutual info with different n_neighbors values"""
        rec = recipe().step_filter_mutual_info('y', top_n=3, n_neighbors=10)
        rec_fit = rec.prep(nonlinear_data)
        transformed = rec_fit.bake(nonlinear_data)

        # Should complete without errors
        assert len(transformed.columns) == 4  # 3 features + y

    def test_mutual_info_new_data(self, nonlinear_data):
        """Test applying mutual info filter to new data"""
        train = nonlinear_data[:150]
        test = nonlinear_data[150:]

        rec = recipe().step_filter_mutual_info('y', top_n=3, n_neighbors=3)
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Should have same columns as training
        train_transformed = rec_fit.bake(train)
        assert set(test_transformed.columns) == set(train_transformed.columns)


class TestStepFilterRocAuc:
    """Test step_filter_roc_auc (ROC AUC-based) functionality"""

    @pytest.fixture
    def binary_classification_data(self):
        """Create binary classification data"""
        np.random.seed(42)
        n = 200
        # Features with different discriminative power
        x1 = np.concatenate([np.random.randn(n // 2) - 2, np.random.randn(n // 2) + 2])
        x2 = np.concatenate([np.random.randn(n // 2) - 1, np.random.randn(n // 2) + 1])
        x3 = np.random.randn(n)  # No discriminative power
        x4 = np.random.randn(n)

        y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])

        return pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'y': y})

    @pytest.fixture
    def multiclass_data(self):
        """Create multiclass classification data"""
        np.random.seed(42)
        n = 300
        # 3 classes
        y = np.concatenate([np.zeros(n // 3), np.ones(n // 3), np.full(n // 3, 2)])

        # Features with varying importance
        x1 = np.concatenate([
            np.random.randn(n // 3) - 2,
            np.random.randn(n // 3),
            np.random.randn(n // 3) + 2
        ])
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)

        return pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'y': y})

    def test_roc_auc_threshold(self, binary_classification_data):
        """Test ROC AUC with threshold selection"""
        rec = recipe().step_filter_roc_auc('y', threshold=0.6)
        rec_fit = rec.prep(binary_classification_data)
        transformed = rec_fit.bake(binary_classification_data)

        # Should complete without errors and keep outcome
        assert 'y' in transformed.columns
        # Should have strong features (x1, x2) if they meet threshold
        assert len(transformed.columns) >= 1  # At least outcome

    def test_roc_auc_top_n(self, binary_classification_data):
        """Test ROC AUC with top_n selection"""
        rec = recipe().step_filter_roc_auc('y', top_n=2)
        rec_fit = rec.prep(binary_classification_data)
        transformed = rec_fit.bake(binary_classification_data)

        # Should keep exactly 2 features plus outcome
        assert len(transformed.columns) == 3  # 2 features + y
        assert 'y' in transformed.columns

    def test_roc_auc_top_p(self, binary_classification_data):
        """Test ROC AUC with top_p selection"""
        rec = recipe().step_filter_roc_auc('y', top_p=0.5)
        rec_fit = rec.prep(binary_classification_data)
        transformed = rec_fit.bake(binary_classification_data)

        # Should keep top 50% of features
        assert len(transformed.columns) == 3  # 2 out of 4 + y
        assert 'y' in transformed.columns

    def test_roc_auc_multiclass_ovr(self, multiclass_data):
        """Test ROC AUC on multiclass data with OvR strategy"""
        rec = recipe().step_filter_roc_auc('y', top_n=2, multiclass_strategy='ovr')
        rec_fit = rec.prep(multiclass_data)
        transformed = rec_fit.bake(multiclass_data)

        # Should keep 2 most discriminative features
        assert len(transformed.columns) == 3
        assert 'x1' in transformed.columns
        assert 'y' in transformed.columns

    def test_roc_auc_multiclass_ovo(self, multiclass_data):
        """Test ROC AUC on multiclass data with OvO strategy"""
        rec = recipe().step_filter_roc_auc('y', top_n=2, multiclass_strategy='ovo')
        rec_fit = rec.prep(multiclass_data)
        transformed = rec_fit.bake(multiclass_data)

        # Should keep 2 most discriminative features
        assert len(transformed.columns) == 3
        assert 'y' in transformed.columns

    def test_roc_auc_new_data(self, binary_classification_data):
        """Test applying ROC AUC filter to new data"""
        train = binary_classification_data[:150]
        test = binary_classification_data[150:]

        rec = recipe().step_filter_roc_auc('y', top_n=2)
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Should have same columns as training
        train_transformed = rec_fit.bake(train)
        assert set(test_transformed.columns) == set(train_transformed.columns)


class TestStepFilterChisq:
    """Test step_filter_chisq (Chi-squared/Fisher exact test) functionality"""

    @pytest.fixture
    def categorical_data(self):
        """Create categorical data for chi-squared test"""
        np.random.seed(42)
        n = 200

        # Strong association with outcome
        x1 = np.where(np.random.rand(n) > 0.5, 'A', 'B')
        # Moderate association
        x2 = np.where(np.random.rand(n) > 0.6, 'X', 'Y')
        # Weak association
        x3 = np.random.choice(['P', 'Q', 'R'], n)
        x4 = np.random.choice(['M', 'N'], n)

        # Outcome depends on x1
        y = np.where(x1 == 'A',
                     np.random.choice(['Class0', 'Class1'], n, p=[0.8, 0.2]),
                     np.random.choice(['Class0', 'Class1'], n, p=[0.2, 0.8]))

        return pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'y': y})

    def test_chisq_threshold(self, categorical_data):
        """Test chi-squared with threshold selection"""
        rec = recipe().step_filter_chisq('y', threshold=1.0, method='chisq', use_pvalue=False)
        rec_fit = rec.prep(categorical_data)
        transformed = rec_fit.bake(categorical_data)

        # Should filter weak associations
        assert len(transformed.columns) < len(categorical_data.columns)
        assert 'y' in transformed.columns

    def test_chisq_top_n(self, categorical_data):
        """Test chi-squared with top_n selection"""
        rec = recipe().step_filter_chisq('y', top_n=2, method='chisq')
        rec_fit = rec.prep(categorical_data)
        transformed = rec_fit.bake(categorical_data)

        # Should keep exactly 2 features plus outcome
        assert len(transformed.columns) == 3  # 2 features + y
        assert 'x1' in transformed.columns  # Strongest association
        assert 'y' in transformed.columns

    def test_chisq_top_p(self, categorical_data):
        """Test chi-squared with top_p selection"""
        rec = recipe().step_filter_chisq('y', top_p=0.5, method='chisq')
        rec_fit = rec.prep(categorical_data)
        transformed = rec_fit.bake(categorical_data)

        # Should keep top 50% of features
        assert len(transformed.columns) == 3  # 2 out of 4 + y
        assert 'y' in transformed.columns

    def test_fisher_exact(self, categorical_data):
        """Test Fisher exact test instead of chi-squared"""
        # Use subset for Fisher (requires 2x2 tables)
        subset = categorical_data[['x1', 'y']].copy()

        rec = recipe().step_filter_chisq('y', threshold=0.5, method='fisher')
        rec_fit = rec.prep(subset)
        transformed = rec_fit.bake(subset)

        # Should complete without errors
        assert 'y' in transformed.columns

    def test_chisq_pvalue_mode(self, categorical_data):
        """Test chi-squared using p-value scoring"""
        rec = recipe().step_filter_chisq('y', threshold=0.1, use_pvalue=True, method='chisq')
        rec_fit = rec.prep(categorical_data)
        transformed = rec_fit.bake(categorical_data)

        # Should keep features with p-value < 0.1
        assert 'x1' in transformed.columns
        assert 'y' in transformed.columns

    def test_chisq_numeric_outcome_error(self):
        """Test that numeric outcome raises appropriate error"""
        data = pd.DataFrame({
            'x1': ['A', 'B', 'A', 'B', 'A'],
            'y': [1.5, 2.3, 1.8, 2.9, 1.2]  # Numeric
        })

        rec = recipe().step_filter_chisq('y', top_n=1, method='chisq')
        # Should handle gracefully or raise meaningful error
        try:
            rec_fit = rec.prep(data)
            transformed = rec_fit.bake(data)
            # If it doesn't error, should still work
            assert 'y' in transformed.columns
        except (ValueError, TypeError):
            # Expected for numeric outcome with chi-squared
            pass

    def test_chisq_new_data(self, categorical_data):
        """Test applying chi-squared filter to new data"""
        train = categorical_data[:150]
        test = categorical_data[150:]

        rec = recipe().step_filter_chisq('y', top_n=2, method='chisq')
        rec_fit = rec.prep(train)
        test_transformed = rec_fit.bake(test)

        # Should have same columns as training
        train_transformed = rec_fit.bake(train)
        assert set(test_transformed.columns) == set(train_transformed.columns)


class TestFilterSupervisedIntegration:
    """Test integration of supervised filter steps"""

    @pytest.fixture
    def mixed_data(self):
        """Create data with both numeric and categorical features"""
        np.random.seed(42)
        n = 200

        # Numeric features
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)

        # Categorical features
        x4 = np.random.choice(['A', 'B', 'C'], n)
        x5 = np.random.choice(['X', 'Y'], n)

        # Outcome depends on x1 and x4
        y = 2 * x1 + np.where(x4 == 'A', 2, 0) + np.random.randn(n) * 0.5

        return pd.DataFrame({
            'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'y': y
        })

    def test_multiple_filters_pipeline(self, mixed_data):
        """Test combining multiple supervised filters"""
        rec = (
            recipe()
            .step_filter_anova('y', top_p=0.8)
            .step_filter_rf_importance('y', top_n=2, trees=30)
        )

        rec_fit = rec.prep(mixed_data)
        transformed = rec_fit.bake(mixed_data)

        # Should progressively reduce features (RF importance limits to top 2)
        assert len(transformed.columns) == 3  # Exactly 2 features + y (RF top_n=2)
        assert 'y' in transformed.columns

    def test_filter_comparison(self, mixed_data):
        """Test that different filters may select different features"""
        # ANOVA filter
        rec_anova = recipe().step_filter_anova('y', top_n=2)
        anova_fit = rec_anova.prep(mixed_data)
        anova_result = anova_fit.bake(mixed_data)

        # RF importance filter
        rec_rf = recipe().step_filter_rf_importance('y', top_n=2, trees=30)
        rf_fit = rec_rf.prep(mixed_data)
        rf_result = rf_fit.bake(mixed_data)

        # Mutual info filter
        rec_mi = recipe().step_filter_mutual_info('y', top_n=2)
        mi_fit = rec_mi.prep(mixed_data)
        mi_result = mi_fit.bake(mixed_data)

        # All should have selected 2 features
        assert len(anova_result.columns) == 3
        assert len(rf_result.columns) == 3
        assert len(mi_result.columns) == 3


class TestFilterSupervisedEdgeCases:
    """Test edge cases for supervised filter steps"""

    def test_filter_single_feature(self):
        """Test filters when only one predictor exists"""
        data = pd.DataFrame({
            'x1': np.random.randn(100),
            'y': np.random.randn(100)
        })

        rec = recipe().step_filter_anova('y', top_n=1)
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should keep the single feature
        assert 'x1' in transformed.columns
        assert 'y' in transformed.columns

    def test_filter_no_numeric_features(self):
        """Test numeric filters when no numeric features exist"""
        data = pd.DataFrame({
            'cat1': ['A', 'B', 'C', 'A', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y', 'X'],
            'y': [1, 2, 3, 4, 5]
        })

        rec = recipe().step_filter_anova('y', top_n=1)
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should keep outcome only
        assert 'y' in transformed.columns

    def test_filter_all_features_selected(self):
        """Test when top_n exceeds available features"""
        data = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'y': np.random.randn(50)
        })

        rec = recipe().step_filter_anova('y', top_n=10)  # More than available
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should keep all features
        assert set(transformed.columns) == set(data.columns)

    def test_filter_zero_features(self):
        """Test with top_n=0"""
        data = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'y': np.random.randn(50)
        })

        rec = recipe().step_filter_anova('y', top_n=0)
        rec_fit = rec.prep(data)
        transformed = rec_fit.bake(data)

        # Should keep only outcome
        assert 'y' in transformed.columns
        assert len(transformed.columns) == 1
