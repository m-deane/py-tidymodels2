"""Tests for advanced feature selection steps"""

import pytest
import pandas as pd
import numpy as np
from py_recipes.recipe import recipe


@pytest.fixture
def regression_data():
    """Create synthetic regression data with relevant and irrelevant features."""
    np.random.seed(42)
    n = 200

    # Create relevant features
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)

    # Create irrelevant features (noise)
    x4 = np.random.randn(n)
    x5 = np.random.randn(n)
    x6 = np.random.randn(n)

    # Target is a function of only x1, x2, x3
    y = 2 * x1 + 3 * x2 - 1.5 * x3 + np.random.randn(n) * 0.5

    return pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,  # noise
        'x5': x5,  # noise
        'x6': x6,  # noise
        'y': y
    })


@pytest.fixture
def classification_data():
    """Create synthetic classification data."""
    np.random.seed(42)
    n = 200

    # Create relevant features
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)

    # Create irrelevant features
    x4 = np.random.randn(n)
    x5 = np.random.randn(n)

    # Target is a function of only x1, x2, x3
    y_prob = 1 / (1 + np.exp(-(x1 + x2 - x3)))
    y = (y_prob > 0.5).astype(int)

    return pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,  # noise
        'x5': x5,  # noise
        'y': y
    })


@pytest.fixture
def correlated_features_data():
    """Create data with correlated features."""
    np.random.seed(42)
    n = 200

    x1 = np.random.randn(n)
    x2 = x1 + np.random.randn(n) * 0.1  # highly correlated with x1
    x3 = np.random.randn(n)
    x4 = x3 + np.random.randn(n) * 0.1  # highly correlated with x3

    y = 2 * x1 + 3 * x3 + np.random.randn(n) * 0.5

    return pd.DataFrame({
        'x1': x1,
        'x2': x2,  # redundant
        'x3': x3,
        'x4': x4,  # redundant
        'y': y
    })


class TestStepVip:
    """Tests for Variable Importance in Projection."""

    def test_vip_basic(self, regression_data):
        """Test basic VIP functionality."""
        rec = recipe().step_vip(outcome='y', threshold=1.0, num_comp=2)
        rec_fit = rec.prep(regression_data)
        transformed = rec_fit.bake(regression_data)

        # Should have selected some features
        assert len(transformed.columns) >= 2  # At least outcome + 1 feature
        assert 'y' in transformed.columns

        # Should have removed at least some irrelevant features
        assert len(transformed.columns) < len(regression_data.columns)

    def test_vip_threshold(self, regression_data):
        """Test that threshold affects number of selected features."""
        # Low threshold = more features
        rec_low = recipe().step_vip(outcome='y', threshold=0.5, num_comp=2)
        rec_low_fit = rec_low.prep(regression_data)
        transformed_low = rec_low_fit.bake(regression_data)

        # Medium threshold = fewer features
        rec_med = recipe().step_vip(outcome='y', threshold=1.0, num_comp=2)
        rec_med_fit = rec_med.prep(regression_data)
        transformed_med = rec_med_fit.bake(regression_data)

        # Low threshold should select more or equal features
        assert len(transformed_low.columns) >= len(transformed_med.columns)

    def test_vip_num_comp(self, regression_data):
        """Test effect of number of components."""
        rec1 = recipe().step_vip(outcome='y', threshold=1.0, num_comp=1)
        rec1_fit = rec1.prep(regression_data)
        transformed1 = rec1_fit.bake(regression_data)

        rec2 = recipe().step_vip(outcome='y', threshold=1.0, num_comp=3)
        rec2_fit = rec2.prep(regression_data)
        transformed2 = rec2_fit.bake(regression_data)

        # Both should work
        assert 'y' in transformed1.columns
        assert 'y' in transformed2.columns

    def test_vip_new_data(self, regression_data):
        """Test VIP on new data."""
        train = regression_data.iloc[:150]
        test = regression_data.iloc[150:]

        rec = recipe().step_vip(outcome='y', threshold=1.0, num_comp=2)
        rec_fit = rec.prep(train)

        # Apply to test data
        test_transformed = rec_fit.bake(test)

        # Should have same columns as training
        train_transformed = rec_fit.bake(train)
        assert set(test_transformed.columns) == set(train_transformed.columns)

    def test_vip_no_outcome_error(self, regression_data):
        """Test that VIP requires outcome."""
        with pytest.raises(ValueError, match="outcome must be specified"):
            rec = recipe().step_vip(outcome=None)
            rec.prep(regression_data)

    def test_vip_stores_scores(self, regression_data):
        """Test that VIP scores are stored."""
        rec = recipe().step_vip(outcome='y', threshold=1.0, num_comp=2)
        rec_fit = rec.prep(regression_data)

        # Access the prepared step
        prepared_step = rec_fit.prepared_steps[0]
        assert hasattr(prepared_step, 'vip_scores')
        assert len(prepared_step.vip_scores) > 0


class TestStepBoruta:
    """Tests for Boruta feature selection."""

    def test_boruta_basic(self, regression_data):
        """Test basic Boruta functionality."""
        rec = recipe().step_boruta(outcome='y', max_iter=50, random_state=42)
        rec_fit = rec.prep(regression_data)
        transformed = rec_fit.bake(regression_data)

        # Should have selected some features
        assert len(transformed.columns) >= 2  # At least outcome + 1 feature
        assert 'y' in transformed.columns

        # Should have removed at least some features
        assert len(transformed.columns) <= len(regression_data.columns)

    def test_boruta_classification(self, classification_data):
        """Test Boruta with classification data."""
        rec = recipe().step_boruta(outcome='y', max_iter=50, random_state=42)
        rec_fit = rec.prep(classification_data)
        transformed = rec_fit.bake(classification_data)

        # Should work with classification
        assert 'y' in transformed.columns
        assert len(transformed.columns) >= 2

    def test_boruta_max_iter(self, regression_data):
        """Test effect of max_iter parameter."""
        # Should complete without error even with few iterations
        rec = recipe().step_boruta(outcome='y', max_iter=10, random_state=42)
        rec_fit = rec.prep(regression_data)
        transformed = rec_fit.bake(regression_data)

        assert 'y' in transformed.columns

    def test_boruta_random_state(self, regression_data):
        """Test that random_state ensures reproducibility."""
        rec1 = recipe().step_boruta(outcome='y', max_iter=50, random_state=42)
        rec1_fit = rec1.prep(regression_data)
        transformed1 = rec1_fit.bake(regression_data)

        rec2 = recipe().step_boruta(outcome='y', max_iter=50, random_state=42)
        rec2_fit = rec2.prep(regression_data)
        transformed2 = rec2_fit.bake(regression_data)

        # Same random state should give same features
        assert set(transformed1.columns) == set(transformed2.columns)

    def test_boruta_new_data(self, regression_data):
        """Test Boruta on new data."""
        train = regression_data.iloc[:150]
        test = regression_data.iloc[150:]

        rec = recipe().step_boruta(outcome='y', max_iter=50, random_state=42)
        rec_fit = rec.prep(train)

        # Apply to test data
        test_transformed = rec_fit.bake(test)

        # Should have same columns as training
        train_transformed = rec_fit.bake(train)
        assert set(test_transformed.columns) == set(train_transformed.columns)

    def test_boruta_stores_ranks(self, regression_data):
        """Test that Boruta stores feature rankings."""
        rec = recipe().step_boruta(outcome='y', max_iter=50, random_state=42)
        rec_fit = rec.prep(regression_data)

        # Access the prepared step
        prepared_step = rec_fit.prepared_steps[0]
        assert hasattr(prepared_step, 'feature_ranks')
        assert len(prepared_step.feature_ranks) > 0


class TestStepRfe:
    """Tests for Recursive Feature Elimination."""

    def test_rfe_basic(self, regression_data):
        """Test basic RFE functionality."""
        rec = recipe().step_rfe(outcome='y', n_features=3)
        rec_fit = rec.prep(regression_data)
        transformed = rec_fit.bake(regression_data)

        # Should have exactly 3 features + outcome
        assert len(transformed.columns) == 4
        assert 'y' in transformed.columns

    def test_rfe_classification(self, classification_data):
        """Test RFE with classification data."""
        rec = recipe().step_rfe(outcome='y', n_features=3)
        rec_fit = rec.prep(classification_data)
        transformed = rec_fit.bake(classification_data)

        # Should work with classification
        assert len(transformed.columns) == 4
        assert 'y' in transformed.columns

    def test_rfe_auto_n_features(self, regression_data):
        """Test RFE with automatic n_features selection."""
        rec = recipe().step_rfe(outcome='y', n_features=None)
        rec_fit = rec.prep(regression_data)
        transformed = rec_fit.bake(regression_data)

        # Should select approximately half of features
        n_predictors = len(regression_data.columns) - 1
        expected_features = max(1, n_predictors // 2)

        # Should have selected features + outcome
        assert len(transformed.columns) == expected_features + 1
        assert 'y' in transformed.columns

    def test_rfe_step_parameter(self, regression_data):
        """Test RFE with different step sizes."""
        rec1 = recipe().step_rfe(outcome='y', n_features=3, step=1)
        rec1_fit = rec1.prep(regression_data)
        transformed1 = rec1_fit.bake(regression_data)

        rec2 = recipe().step_rfe(outcome='y', n_features=3, step=2)
        rec2_fit = rec2.prep(regression_data)
        transformed2 = rec2_fit.bake(regression_data)

        # Both should select 3 features
        assert len(transformed1.columns) == 4
        assert len(transformed2.columns) == 4

    def test_rfe_new_data(self, regression_data):
        """Test RFE on new data."""
        train = regression_data.iloc[:150]
        test = regression_data.iloc[150:]

        rec = recipe().step_rfe(outcome='y', n_features=3)
        rec_fit = rec.prep(train)

        # Apply to test data
        test_transformed = rec_fit.bake(test)

        # Should have same columns as training
        train_transformed = rec_fit.bake(train)
        assert set(test_transformed.columns) == set(train_transformed.columns)

    def test_rfe_stores_ranks(self, regression_data):
        """Test that RFE stores feature rankings."""
        rec = recipe().step_rfe(outcome='y', n_features=3)
        rec_fit = rec.prep(regression_data)

        # Access the prepared step
        prepared_step = rec_fit.prepared_steps[0]
        assert hasattr(prepared_step, 'feature_ranks')
        assert len(prepared_step.feature_ranks) > 0

    def test_rfe_custom_estimator(self, regression_data):
        """Test RFE with custom estimator."""
        from sklearn.ensemble import RandomForestRegressor

        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        rec = recipe().step_rfe(outcome='y', n_features=3, estimator=estimator)
        rec_fit = rec.prep(regression_data)
        transformed = rec_fit.bake(regression_data)

        # Should work with custom estimator
        assert len(transformed.columns) == 4
        assert 'y' in transformed.columns


class TestFeatureSelectionPipelines:
    """Tests for combining feature selection with other steps."""

    def test_vip_then_normalize(self, regression_data):
        """Test VIP followed by normalization."""
        rec = (
            recipe()
            .step_vip(outcome='y', threshold=1.0, num_comp=2)
            .step_normalize()
        )
        rec_fit = rec.prep(regression_data)
        transformed = rec_fit.bake(regression_data)

        # Should have selected and normalized features
        assert 'y' in transformed.columns
        assert len(transformed.columns) < len(regression_data.columns)

    def test_rfe_then_scale(self, regression_data):
        """Test RFE followed by scaling."""
        rec = (
            recipe()
            .step_rfe(outcome='y', n_features=3)
            .step_scale()
        )
        rec_fit = rec.prep(regression_data)
        transformed = rec_fit.bake(regression_data)

        # Should have 3 selected features + outcome
        assert len(transformed.columns) == 4

        # Selected features (except outcome) should be scaled
        for col in transformed.columns:
            if col != 'y':
                assert abs(transformed[col].std() - 1.0) < 0.1

    def test_impute_then_boruta(self, regression_data):
        """Test imputation followed by Boruta."""
        # Add some missing values
        data_with_na = regression_data.copy()
        data_with_na.loc[0:5, 'x1'] = np.nan
        data_with_na.loc[10:15, 'x2'] = np.nan

        rec = (
            recipe()
            .step_impute_mean()
            .step_boruta(outcome='y', max_iter=50, random_state=42)
        )
        rec_fit = rec.prep(data_with_na)
        transformed = rec_fit.bake(data_with_na)

        # Should have imputed and selected features
        assert not transformed.isna().any().any()
        assert 'y' in transformed.columns

    def test_multiple_selection_methods(self, regression_data):
        """Test combining different selection methods."""
        # This is not typical usage, but should work
        rec = (
            recipe()
            .step_rfe(outcome='y', n_features=5)
            .step_vip(outcome='y', threshold=0.8, num_comp=2)
        )
        rec_fit = rec.prep(regression_data)
        transformed = rec_fit.bake(regression_data)

        # Should apply both selections sequentially
        assert 'y' in transformed.columns
        assert len(transformed.columns) <= 6  # At most 5 features + outcome

    def test_feature_selection_then_pca(self, regression_data):
        """Test feature selection followed by PCA."""
        rec = (
            recipe()
            .step_rfe(outcome='y', n_features=4)
            .step_pca(num_comp=2)
        )
        rec_fit = rec.prep(regression_data)
        transformed = rec_fit.bake(regression_data)

        # Should have 2 PC components
        # Note: PCA operates on all numeric columns, outcome may or may not be included
        pc_cols = [col for col in transformed.columns if col.startswith('PC')]
        assert len(pc_cols) == 2


class TestEdgeCases:
    """Test edge cases for feature selection steps."""

    def test_vip_too_high_threshold(self, regression_data):
        """Test VIP with threshold that's too high."""
        rec = recipe().step_vip(outcome='y', threshold=100.0, num_comp=2)

        with pytest.raises(ValueError, match="No features have VIP"):
            rec.prep(regression_data)

    def test_rfe_more_features_than_available(self, regression_data):
        """Test RFE requesting more features than available."""
        n_predictors = len(regression_data.columns) - 1

        rec = recipe().step_rfe(outcome='y', n_features=100)
        rec_fit = rec.prep(regression_data)
        transformed = rec_fit.bake(regression_data)

        # Should select all available features
        assert len(transformed.columns) == n_predictors + 1

    def test_feature_selection_with_few_samples(self):
        """Test feature selection with very few samples."""
        np.random.seed(42)
        small_data = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [2, 4, 6, 8, 10],
            'y': [1, 2, 3, 4, 5]
        })

        # Should still work with few samples
        rec = recipe().step_rfe(outcome='y', n_features=1)
        rec_fit = rec.prep(small_data)
        transformed = rec_fit.bake(small_data)

        assert len(transformed.columns) == 2  # 1 feature + outcome
