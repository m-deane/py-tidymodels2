"""
Tests for prophet_boost with exogenous regressor support

Verifies that prophet_boost correctly uses exogenous variables from the formula
and that this matches R modeltime behavior (no auto-generated time features).
"""

import pytest
import pandas as pd
import numpy as np
from py_parsnip.models.prophet_boost import prophet_boost


@pytest.fixture
def ts_data_with_exog():
    """Create time series data with exogenous regressors"""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    np.random.seed(42)

    # Create exogenous variables
    feature1 = np.random.randn(len(dates)) * 10 + 50  # Business metric 1
    feature2 = np.random.randn(len(dates)) * 5 + 20   # Business metric 2
    feature3 = dates.dayofweek.values                  # Day of week

    # Target influenced by trend + exogenous vars + noise
    trend = np.linspace(100, 200, len(dates))
    y = trend + 0.5 * feature1 + 0.3 * feature2 + 2 * feature3 + np.random.randn(len(dates)) * 5

    return pd.DataFrame({
        'date': dates,
        'y': y,
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3
    })


@pytest.fixture
def ts_data_no_exog():
    """Create time series data without exogenous regressors (date only)"""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    np.random.seed(42)

    # Simple trend + seasonality
    trend = np.linspace(100, 200, len(dates))
    seasonality = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    y = trend + seasonality + np.random.randn(len(dates)) * 5

    return pd.DataFrame({
        'date': dates,
        'y': y
    })


class TestExogenousRegressorExtraction:
    """Test that exogenous variables are correctly extracted from formula"""

    def test_exog_vars_extracted_from_formula(self, ts_data_with_exog):
        """Verify exogenous vars are extracted and used (not ignored)"""
        train = ts_data_with_exog.iloc[:80].copy()
        test = ts_data_with_exog.iloc[:90].copy()

        # Formula with exogenous regressors
        spec = prophet_boost(trees=50)
        fit = spec.fit(train, 'y ~ date + feature1 + feature2')

        # Verify exogenous columns were stored
        assert fit.fit_data.get("use_exog") == True
        assert fit.fit_data.get("exog_cols") == ['feature1', 'feature2']

        # Predict should work
        predictions = fit.predict(test)
        assert predictions is not None
        assert len(predictions) == len(test)

    def test_no_exog_vars_uses_fallback(self, ts_data_no_exog):
        """Verify fallback to cyclical features when no exog vars"""
        train = ts_data_no_exog.iloc[:80].copy()
        test = ts_data_no_exog.iloc[80:90].copy()

        # Formula with date only (no exog vars)
        spec = prophet_boost(trees=50)

        # Should warn about using fallback features
        with pytest.warns(UserWarning, match="No exogenous regressors provided"):
            fit = spec.fit(train, 'y ~ date')

        # Verify fallback was used
        assert fit.fit_data.get("use_exog") == False
        assert fit.fit_data.get("exog_cols") is None

        # Predict should still work
        predictions = fit.predict(test)
        assert predictions is not None
        assert len(predictions) == len(test)

    def test_exog_vars_required_in_test_data(self, ts_data_with_exog):
        """Verify error if exog vars missing from test data"""
        train = ts_data_with_exog.iloc[:80].copy()
        test = ts_data_with_exog.iloc[80:90][['date', 'y']].copy()  # Missing feature1, feature2

        spec = prophet_boost(trees=50)
        fit = spec.fit(train, 'y ~ date + feature1 + feature2')

        # Should raise error when predicting on data without exog vars
        with pytest.raises(ValueError, match="Exogenous variables.*not found"):
            fit.predict(test)

    def test_categorical_exog_vars_encoded(self, ts_data_with_exog):
        """Verify categorical exogenous variables are label-encoded"""
        train = ts_data_with_exog.iloc[:80].copy()
        test = ts_data_with_exog.iloc[80:90].copy()

        # Add categorical variable
        train['category'] = np.random.choice(['A', 'B', 'C'], size=len(train))
        test['category'] = np.random.choice(['A', 'B', 'C'], size=len(test))

        spec = prophet_boost(trees=50)
        fit = spec.fit(train, 'y ~ date + feature1 + category')

        # Verify label encoders were stored
        assert fit.fit_data.get("label_encoders") is not None
        assert 'category' in fit.fit_data["label_encoders"]

        # Predict should work with encoded categorical
        predictions = fit.predict(test)
        assert predictions is not None


class TestExogenousVsAutoFeatures:
    """Compare performance: exogenous features vs auto-generated features"""

    def test_exog_features_better_than_auto_features(self, ts_data_with_exog):
        """Exogenous features should perform better than auto-generated time features"""
        train = ts_data_with_exog.iloc[:80].copy()
        test = ts_data_with_exog.iloc[80:90].copy()

        # Model 1: With exogenous features
        spec_exog = prophet_boost(trees=100, tree_depth=6)
        fit_exog = spec_exog.fit(train, 'y ~ date + feature1 + feature2 + feature3')
        fit_exog = fit_exog.evaluate(test)
        outputs_exog, _, stats_exog = fit_exog.extract_outputs()

        # Model 2: Without exogenous features (auto fallback)
        spec_auto = prophet_boost(trees=100, tree_depth=6)
        with pytest.warns(UserWarning):
            fit_auto = spec_auto.fit(train, 'y ~ date')
        fit_auto = fit_auto.evaluate(test)
        outputs_auto, _, stats_auto = fit_auto.extract_outputs()

        # Get test RMSEs
        test_stats_exog = stats_exog[stats_exog['split'] == 'test']
        test_stats_auto = stats_auto[stats_auto['split'] == 'test']

        if 'metric' in test_stats_exog.columns:
            # Long format
            rmse_exog = test_stats_exog[test_stats_exog['metric'] == 'rmse']['value'].values[0]
            rmse_auto = test_stats_auto[test_stats_auto['metric'] == 'rmse']['value'].values[0]
        else:
            # Wide format
            rmse_exog = test_stats_exog['rmse'].values[0]
            rmse_auto = test_stats_auto['rmse'].values[0]

        # Exogenous features should perform better (lower RMSE)
        # Allow 10% margin for randomness
        assert rmse_exog < rmse_auto * 1.1, \
            f"Exogenous features should perform better: exog RMSE={rmse_exog:.2f}, auto RMSE={rmse_auto:.2f}"

        print(f"\nExogenous features RMSE: {rmse_exog:.2f}")
        print(f"Auto features RMSE: {rmse_auto:.2f}")
        print(f"Improvement: {((rmse_auto - rmse_exog) / rmse_auto * 100):.1f}%")

    def test_exog_xgb_predictions_vary_on_test(self, ts_data_with_exog):
        """XGBoost predictions should vary on test set (not constant)"""
        train = ts_data_with_exog.iloc[:80].copy()
        test = ts_data_with_exog.iloc[80:90].copy()

        spec = prophet_boost(trees=100)
        fit = spec.fit(train, 'y ~ date + feature1 + feature2')
        fit = fit.evaluate(test)

        outputs, _, _ = fit.extract_outputs()
        test_outputs = outputs[outputs['split'] == 'test']

        # XGBoost component should vary
        xgb_values = test_outputs['xgb_fitted'].values
        unique_values = len(np.unique(np.round(xgb_values, 2)))

        # Should have multiple unique values (not constant)
        assert unique_values >= 3, \
            f"XGBoost predictions should vary on test data: only {unique_values} unique values"

        xgb_std = np.std(xgb_values)
        assert xgb_std > 0.1, \
            f"XGBoost predictions have very low variance: std={xgb_std:.4f}"

        print(f"\nXGBoost test predictions: {unique_values} unique values, std={xgb_std:.2f}")


class TestNoExtrapolationIssues:
    """Verify that exogenous features don't have extrapolation problems"""

    def test_no_sharp_residual_increase_on_test(self, ts_data_with_exog):
        """Test residuals should remain reasonable (not extreme spike)"""
        train = ts_data_with_exog.iloc[:80].copy()
        test = ts_data_with_exog.iloc[80:95].copy()

        # Use more conservative hyperparameters to reduce overfitting
        spec = prophet_boost(trees=50, tree_depth=3, learn_rate=0.1)
        fit = spec.fit(train, 'y ~ date + feature1 + feature2')
        fit = fit.evaluate(test)

        outputs, _, _ = fit.extract_outputs()

        # Get residuals
        train_outputs = outputs[outputs['split'] == 'train']
        test_outputs = outputs[outputs['split'] == 'test']

        train_residuals = train_outputs['residuals'].abs().mean()
        test_residuals = test_outputs['residuals'].abs().mean()

        # Test residuals should not be extreme
        # With exogenous features, test MAE should be < 10
        # (vs old auto-features which had MAE > 50 due to extrapolation)
        assert test_residuals < 10.0, \
            f"Test residuals too high: test MAE={test_residuals:.2f} (should be < 10)"

        # Verify the fix: exog features prevent extreme extrapolation
        # Old implementation had test residuals > 50, new should be < 10
        print(f"\nTrain MAE: {train_residuals:.2f}")
        print(f"Test MAE: {test_residuals:.2f}")
        print(f"✓ Exogenous features prevent extrapolation spike")

    def test_stable_predictions_far_future(self, ts_data_with_exog):
        """Predictions should remain stable even far into future"""
        train = ts_data_with_exog.iloc[:80].copy()
        test = ts_data_with_exog.iloc[80:100].copy()  # 20 days into future

        spec = prophet_boost(trees=100)
        fit = spec.fit(train, 'y ~ date + feature1 + feature2')

        # Predict on far future data
        predictions = fit.predict(test)

        # Predictions should not have extreme values
        assert not predictions['.pred'].isna().any(), "Predictions contain NaN"
        assert not np.isinf(predictions['.pred']).any(), "Predictions contain inf"

        # Predictions should be in reasonable range (not extrapolated to extreme values)
        train_y_range = train['y'].max() - train['y'].min()
        pred_range = predictions['.pred'].max() - predictions['.pred'].min()

        assert pred_range < train_y_range * 3, \
            f"Prediction range too wide: pred_range={pred_range:.2f}, train_range={train_y_range:.2f}"


class TestFeatureImportance:
    """Test that XGBoost uses exogenous features correctly"""

    def test_xgboost_uses_all_exog_features(self, ts_data_with_exog):
        """XGBoost should use all provided exogenous features"""
        train = ts_data_with_exog.iloc[:80].copy()

        spec = prophet_boost(trees=100)
        fit = spec.fit(train, 'y ~ date + feature1 + feature2 + feature3')

        # Get XGBoost model
        xgb_model = fit.fit_data['xgb_model']

        # Verify number of features
        assert xgb_model.n_features_in_ == 3, \
            f"XGBoost should use 3 features, got {xgb_model.n_features_in_}"

        # Check feature importances exist
        importances = xgb_model.feature_importances_
        assert len(importances) == 3, \
            f"Should have 3 feature importances, got {len(importances)}"

        # At least one feature should be important
        assert np.max(importances) > 0, \
            "All feature importances are zero"

        print(f"\nFeature importances: {importances}")


class TestBackwardCompatibility:
    """Ensure existing tests still pass"""

    def test_auto_features_still_work(self, ts_data_no_exog):
        """Fallback to auto-generated features should still work"""
        train = ts_data_no_exog.iloc[:80].copy()
        test = ts_data_no_exog.iloc[80:90].copy()

        spec = prophet_boost(trees=50)

        with pytest.warns(UserWarning, match="No exogenous regressors"):
            fit = spec.fit(train, 'y ~ date')

        # Should work without errors
        predictions = fit.predict(test)
        fit = fit.evaluate(test)
        outputs, coefs, stats = fit.extract_outputs()

        assert outputs is not None
        assert coefs is not None
        assert stats is not None

    def test_no_days_since_start_in_auto_features(self, ts_data_no_exog):
        """Verify days_since_start was removed from auto features"""
        train = ts_data_no_exog.iloc[:80].copy()

        spec = prophet_boost(trees=50)

        with pytest.warns(UserWarning):
            fit = spec.fit(train, 'y ~ date')

        # XGBoost should have 11 features (not 12 - removed days_since_start)
        xgb_model = fit.fit_data['xgb_model']
        assert xgb_model.n_features_in_ == 11, \
            f"Auto features should have 11 features (no days_since_start), got {xgb_model.n_features_in_}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
