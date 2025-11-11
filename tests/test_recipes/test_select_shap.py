"""
Tests for SHAP-based feature selection step.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from py_recipes.steps.filter_supervised import StepSelectShap


@pytest.fixture
def regression_data():
    """Create synthetic regression dataset with known feature importance."""
    np.random.seed(42)
    n = 200

    # Create features with varying importance
    x1 = np.random.randn(n)  # High importance
    x2 = np.random.randn(n)  # Medium importance
    x3 = np.random.randn(n)  # Low importance (noise)
    x4 = np.random.randn(n)  # No importance (pure noise)
    x5 = np.random.randn(n)  # Medium importance

    # Target: strong relationship with x1, moderate with x2 and x5, weak with x3
    y = 3.0 * x1 + 1.5 * x2 + 1.2 * x5 + 0.2 * x3 + np.random.randn(n) * 0.5

    return pd.DataFrame({
        'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5,
        'y': y
    })


@pytest.fixture
def classification_data():
    """Create synthetic classification dataset."""
    np.random.seed(42)
    n = 200

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)
    x4 = np.random.randn(n)

    # Binary classification based on x1 and x2
    prob = 1 / (1 + np.exp(-(2*x1 + x2)))
    y = (np.random.rand(n) < prob).astype(int)

    return pd.DataFrame({
        'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4,
        'y': y
    })


def test_shap_basic_functionality_with_tree_model(regression_data):
    """Test basic SHAP selection with tree-based model (TreeExplainer)."""
    # Train model
    X = regression_data[['x1', 'x2', 'x3', 'x4', 'x5']]
    y = regression_data['y']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Create step
    step = StepSelectShap(
        outcome='y',
        model=model,
        top_n=3
    )

    # Prep and bake
    prepped = step.prep(regression_data, training=True)
    result = prepped.bake(regression_data)

    # Check that we kept top 3 features + outcome
    assert len(result.columns) == 4  # 3 features + outcome
    assert 'y' in result.columns
    assert len(prepped._selected_features) == 3

    # Check that high-importance features are selected
    # x1 should be selected (highest importance)
    assert 'x1' in prepped._selected_features

    # Check scores are computed
    assert len(prepped._scores) == 5  # All original features scored
    assert all(score >= 0 for score in prepped._scores.values())


def test_shap_with_linear_model(regression_data):
    """Test SHAP selection with linear model (KernelExplainer fallback)."""
    # Train model
    X = regression_data[['x1', 'x2', 'x3', 'x4', 'x5']]
    y = regression_data['y']
    model = LinearRegression()
    model.fit(X, y)

    # Create step with sampling to speed up kernel explainer
    step = StepSelectShap(
        outcome='y',
        model=model,
        top_n=2,
        shap_samples=100  # Sample for faster computation
    )

    # Prep and bake
    prepped = step.prep(regression_data, training=True)
    result = prepped.bake(regression_data)

    # Check that we kept top 2 features + outcome
    assert len(result.columns) == 3
    assert 'y' in result.columns
    assert len(prepped._selected_features) == 2


def test_shap_classification_task(classification_data):
    """Test SHAP selection with classification model."""
    # Train model
    X = classification_data[['x1', 'x2', 'x3', 'x4']]
    y = classification_data['y']
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Create step
    step = StepSelectShap(
        outcome='y',
        model=model,
        top_n=2
    )

    # Prep and bake
    prepped = step.prep(classification_data, training=True)
    result = prepped.bake(classification_data)

    # Check selection
    assert len(result.columns) == 3  # 2 features + outcome
    assert 'y' in result.columns

    # x1 and x2 should have higher importance
    assert 'x1' in prepped._selected_features or 'x2' in prepped._selected_features


def test_shap_threshold_selection(regression_data):
    """Test SHAP selection using threshold instead of top_n."""
    # Train model
    X = regression_data[['x1', 'x2', 'x3', 'x4', 'x5']]
    y = regression_data['y']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Create step with threshold
    step = StepSelectShap(
        outcome='y',
        model=model,
        threshold=0.05  # Keep features with |SHAP| > 0.05
    )

    # Prep and bake
    prepped = step.prep(regression_data, training=True)
    result = prepped.bake(regression_data)

    # At least one feature should be selected
    assert len(prepped._selected_features) >= 1
    assert 'y' in result.columns

    # All selected features should have importance >= threshold
    for feat in prepped._selected_features:
        assert prepped._scores[feat] >= 0.05


def test_shap_top_p_selection(regression_data):
    """Test SHAP selection using top_p (proportion)."""
    # Train model
    X = regression_data[['x1', 'x2', 'x3', 'x4', 'x5']]
    y = regression_data['y']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Create step with top_p
    step = StepSelectShap(
        outcome='y',
        model=model,
        top_p=0.4  # Keep top 40% of features
    )

    # Prep and bake
    prepped = step.prep(regression_data, training=True)
    result = prepped.bake(regression_data)

    # Should keep 2 features (40% of 5 = 2)
    assert len(prepped._selected_features) == 2
    assert len(result.columns) == 3  # 2 features + outcome


def test_shap_with_categorical_features():
    """Test SHAP selection with categorical features (one-hot encoding)."""
    np.random.seed(42)
    n = 200

    # Create data with categorical variable
    categories = np.random.choice(['A', 'B', 'C'], size=n)
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)

    # Target depends on category and x1
    y = (categories == 'A') * 2.0 + (categories == 'B') * 1.0 + x1 * 1.5 + np.random.randn(n) * 0.5

    data = pd.DataFrame({
        'category': categories,
        'x1': x1,
        'x2': x2,
        'y': y
    })

    # Train model
    X = data[['category', 'x1', 'x2']].copy()
    X_encoded = pd.get_dummies(X, columns=['category'], drop_first=True)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_encoded, data['y'])

    # Create step
    step = StepSelectShap(
        outcome='y',
        model=model,
        top_n=2
    )

    # Prep and bake
    prepped = step.prep(data, training=True)
    result = prepped.bake(data)

    # Check that categorical features are handled
    assert len(prepped._selected_features) == 2
    assert 'y' in result.columns

    # Category should be selected (it has high importance)
    assert 'category' in prepped._selected_features or 'x1' in prepped._selected_features


def test_shap_validation_errors():
    """Test that validation errors are raised correctly."""
    # No selection mode specified
    with pytest.raises(ValueError, match="Must specify one of"):
        StepSelectShap(outcome='y', model=None)

    # Multiple selection modes
    with pytest.raises(ValueError, match="Can only specify one of"):
        StepSelectShap(outcome='y', model=None, top_n=5, top_p=0.2)

    # Invalid top_p (test this separately without other parameters)
    with pytest.raises(ValueError, match="top_p must be in"):
        StepSelectShap(outcome='y', model=None, top_p=1.5)

    # Model is None
    with pytest.raises(ValueError, match="model parameter is required"):
        StepSelectShap(outcome='y', model=None, top_n=5)


def test_shap_skip_parameter(regression_data):
    """Test that skip=True prevents step execution."""
    # Train model
    X = regression_data[['x1', 'x2', 'x3', 'x4', 'x5']]
    y = regression_data['y']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Create step with skip=True
    step = StepSelectShap(
        outcome='y',
        model=model,
        top_n=3,
        skip=True
    )

    # Prep and bake
    prepped = step.prep(regression_data, training=True)
    result = prepped.bake(regression_data)

    # All columns should remain
    assert len(result.columns) == len(regression_data.columns)
    assert not prepped._is_prepared


def test_shap_missing_outcome(regression_data):
    """Test error when outcome column is missing."""
    # Train model
    X = regression_data[['x1', 'x2', 'x3', 'x4', 'x5']]
    y = regression_data['y']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Create step with non-existent outcome
    step = StepSelectShap(
        outcome='nonexistent',
        model=model,
        top_n=3
    )

    # Should raise error during prep
    with pytest.raises(ValueError, match="Outcome.*not found"):
        step.prep(regression_data, training=True)


def test_shap_bake_without_prep(regression_data):
    """Test error when bake is called before prep."""
    # Train model
    X = regression_data[['x1', 'x2', 'x3', 'x4', 'x5']]
    y = regression_data['y']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Create step but don't prep
    step = StepSelectShap(
        outcome='y',
        model=model,
        top_n=3
    )

    # Bake should fail
    with pytest.raises(ValueError, match="must be prepped"):
        step.bake(regression_data)


def test_shap_with_sampling(regression_data):
    """Test SHAP calculation with sampling to speed up computation."""
    # Train model
    X = regression_data[['x1', 'x2', 'x3', 'x4', 'x5']]
    y = regression_data['y']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Create step with sampling
    step = StepSelectShap(
        outcome='y',
        model=model,
        top_n=3,
        shap_samples=100,  # Use only 100 samples
        random_state=42
    )

    # Prep and bake
    prepped = step.prep(regression_data, training=True)
    result = prepped.bake(regression_data)

    # Should still work correctly
    assert len(result.columns) == 4  # 3 features + outcome
    assert len(prepped._selected_features) == 3
    assert 'x1' in prepped._selected_features  # High importance feature selected
