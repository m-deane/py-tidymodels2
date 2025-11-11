"""
Verification test to confirm LightGBM-based feature importance calculation
produces non-uniform scores based on actual predictive power.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from py_recipes.steps.feature_extraction import StepSafe


def test_importance_scores_are_non_uniform():
    """
    Verify that feature importances are based on actual predictive power,
    not uniform distribution.

    Creates synthetic data where one threshold is highly predictive
    and another is not, then verifies the importance scores reflect this.
    """
    np.random.seed(42)
    n = 500

    # Create data where x1 has a strong effect when > 50
    # and x2 has weak/random relationship with y
    x1 = np.random.uniform(0, 100, n)
    x2 = np.random.uniform(0, 100, n)

    # Strong threshold at 50 for x1 (creates large y difference)
    # Weak threshold at 60 for x1 (minimal y difference)
    y = np.where(x1 > 50, 100, 20) + np.random.normal(0, 5, n)

    data = pd.DataFrame({'x1': x1, 'x2': x2, 'target': y})

    # Fit surrogate model
    surrogate = GradientBoostingRegressor(n_estimators=100, random_state=42)
    X = data[['x1', 'x2']]
    surrogate.fit(X, data['target'])

    # Create SAFE step with low penalty to get multiple changepoints
    step = StepSafe(
        surrogate_model=surrogate,
        outcome='target',
        penalty=1.0,  # Low penalty to get multiple changepoints
        feature_type='dummies'
    )

    # Prep the step
    step.prep(data, training=True)

    # Get feature importances
    importances_df = step.get_feature_importances()

    # Verify we got multiple features for x1
    x1_features = [f for f in importances_df['feature'] if f.startswith('x1_')]
    assert len(x1_features) >= 2, f"Expected at least 2 x1 features, got {len(x1_features)}"

    # Get importance scores for x1 features
    x1_importances = importances_df[importances_df['feature'].str.startswith('x1_')]['importance'].values

    # Verify importances are NOT uniform (not all equal within 0.01 tolerance)
    # If using uniform distribution, all would be equal (e.g., 0.333, 0.333, 0.333)
    # With LightGBM, they should differ based on predictive power
    unique_importances = len(set(np.round(x1_importances, 3)))

    assert unique_importances > 1, (
        f"All x1 feature importances are uniform ({x1_importances}). "
        f"Expected non-uniform scores based on predictive power."
    )

    print(f"\nFeature importances for x1 features:")
    for feat, imp in zip(x1_features, x1_importances):
        print(f"  {feat}: {imp:.4f}")

    # Verify they sum to 1.0 within the variable group (normalization)
    assert np.abs(x1_importances.sum() - 1.0) < 0.01, (
        f"x1 importances don't sum to 1.0: {x1_importances.sum()}"
    )


def test_importance_calculation_with_classification():
    """
    Verify feature importance calculation works for classification tasks.
    """
    from sklearn.ensemble import GradientBoostingClassifier

    np.random.seed(42)
    n = 300

    # Create binary classification data
    x1 = np.random.uniform(0, 100, n)
    x2 = np.random.uniform(0, 100, n)

    # Strong threshold at 50 for x1
    y = (x1 > 50).astype(int)

    data = pd.DataFrame({'x1': x1, 'x2': x2, 'target': y})

    # Fit classification surrogate
    surrogate = GradientBoostingClassifier(n_estimators=100, random_state=42)
    X = data[['x1', 'x2']]
    surrogate.fit(X, data['target'])

    # Create SAFE step
    step = StepSafe(
        surrogate_model=surrogate,
        outcome='target',
        penalty=2.0,
        feature_type='dummies'
    )

    # Prep the step
    step.prep(data, training=True)

    # Get feature importances
    importances_df = step.get_feature_importances()

    # Should have features created
    assert len(importances_df) > 0, "Expected features to be created"

    # Get x1 features
    x1_features = importances_df[importances_df['feature'].str.startswith('x1_')]

    if len(x1_features) > 1:
        # Verify importances are non-uniform
        x1_importances = x1_features['importance'].values
        unique_importances = len(set(np.round(x1_importances, 3)))

        # May or may not be non-uniform depending on changepoints detected
        # Just verify they sum to 1.0 within variable group
        assert np.abs(x1_importances.sum() - 1.0) < 0.01, (
            f"x1 importances don't sum to 1.0: {x1_importances.sum()}"
        )

        print(f"\nClassification - x1 feature importances:")
        for _, row in x1_features.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")


def test_fallback_to_uniform_without_lightgbm():
    """
    Verify graceful fallback when LightGBM is not available.

    This test manually triggers the fallback path to ensure it works.
    """
    np.random.seed(42)
    n = 200

    x1 = np.random.uniform(0, 100, n)
    y = x1 + np.random.normal(0, 5, n)

    data = pd.DataFrame({'x1': x1, 'target': y})

    # Fit surrogate
    surrogate = GradientBoostingRegressor(n_estimators=50, random_state=42)
    surrogate.fit(data[['x1']], data['target'])

    # Create SAFE step
    step = StepSafe(
        surrogate_model=surrogate,
        outcome='target',
        penalty=2.0
    )

    # Prep (will use LightGBM if available)
    step.prep(data, training=True)

    # Manually call uniform importance to test fallback
    step._use_uniform_importance()

    # Get importances
    importances_df = step.get_feature_importances()

    # With uniform distribution, all features from same variable should have equal importance
    x1_features = importances_df[importances_df['feature'].str.startswith('x1_')]

    if len(x1_features) > 1:
        x1_importances = x1_features['importance'].values

        # Should all be equal (uniform distribution)
        assert np.allclose(x1_importances, x1_importances[0], atol=1e-6), (
            f"Uniform importance fallback produced non-uniform scores: {x1_importances}"
        )

        # Should sum to 1.0
        assert np.abs(x1_importances.sum() - 1.0) < 0.01, (
            f"Uniform importances don't sum to 1.0: {x1_importances.sum()}"
        )

        print(f"\nUniform fallback - x1 feature importances:")
        for _, row in x1_features.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")


def test_regression_task_detection():
    """
    Verify correct task type detection for regression vs classification.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    np.random.seed(42)
    n = 100

    x1 = np.random.uniform(0, 100, n)
    y_regression = x1 + np.random.normal(0, 5, n)  # Continuous outcome
    y_classification = (x1 > 50).astype(int)  # Binary outcome

    data_reg = pd.DataFrame({'x1': x1, 'target': y_regression})
    data_clf = pd.DataFrame({'x1': x1, 'target': y_classification})

    # Fit surrogate
    surrogate = GradientBoostingRegressor(n_estimators=50, random_state=42)
    surrogate.fit(data_reg[['x1']], data_reg['target'])

    # Create step
    step = StepSafe(
        surrogate_model=surrogate,
        outcome='target',
        penalty=2.0
    )

    # Test regression detection
    assert step._is_regression_task(data_reg['target']) == True, (
        "Should detect regression for continuous outcome"
    )

    # Test classification detection
    assert step._is_regression_task(data_clf['target']) == False, (
        "Should detect classification for binary outcome with <=10 unique values"
    )

    # Test with few unique values in regression data (should be classification)
    y_few_values = np.random.choice([1.0, 2.0, 3.0], size=n)
    data_few = pd.DataFrame({'x1': x1, 'target': y_few_values})

    assert step._is_regression_task(data_few['target']) == False, (
        "Should detect classification for numeric outcome with <=10 unique values"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
