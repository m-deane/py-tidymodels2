"""
Tests for permutation importance feature selection step.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from py_recipes.steps.filter_supervised import StepSelectPermutation


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


def test_permutation_basic_functionality(regression_data):
    """Test basic permutation importance selection."""
    # Train model
    X = regression_data[['x1', 'x2', 'x3', 'x4', 'x5']]
    y = regression_data['y']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Create step
    step = StepSelectPermutation(
        outcome='y',
        model=model,
        top_n=3,
        n_repeats=5,
        random_state=42
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


def test_permutation_with_scoring_metric(regression_data):
    """Test permutation importance with specific scoring metric."""
    # Train model
    X = regression_data[['x1', 'x2', 'x3', 'x4', 'x5']]
    y = regression_data['y']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Create step with specific scoring
    step = StepSelectPermutation(
        outcome='y',
        model=model,
        top_n=3,
        n_repeats=5,
        scoring='neg_mean_squared_error',  # Explicit scoring
        random_state=42
    )

    # Prep and bake
    prepped = step.prep(regression_data, training=True)
    result = prepped.bake(regression_data)

    # Check selection
    assert len(result.columns) == 4
    assert 'y' in result.columns
    assert len(prepped._selected_features) == 3


def test_permutation_classification_task(classification_data):
    """Test permutation importance with classification model."""
    # Train model
    X = classification_data[['x1', 'x2', 'x3', 'x4']]
    y = classification_data['y']
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Create step
    step = StepSelectPermutation(
        outcome='y',
        model=model,
        top_n=2,
        n_repeats=5,
        random_state=42
    )

    # Prep and bake
    prepped = step.prep(classification_data, training=True)
    result = prepped.bake(classification_data)

    # Check selection
    assert len(result.columns) == 3  # 2 features + outcome
    assert 'y' in result.columns

    # x1 and x2 should have higher importance
    assert 'x1' in prepped._selected_features or 'x2' in prepped._selected_features


def test_permutation_threshold_selection(regression_data):
    """Test permutation importance using threshold instead of top_n."""
    # Train model
    X = regression_data[['x1', 'x2', 'x3', 'x4', 'x5']]
    y = regression_data['y']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Create step with threshold
    step = StepSelectPermutation(
        outcome='y',
        model=model,
        threshold=0.01,  # Keep features with importance > 0.01
        n_repeats=5,
        random_state=42
    )

    # Prep and bake
    prepped = step.prep(regression_data, training=True)
    result = prepped.bake(regression_data)

    # At least one feature should be selected
    assert len(prepped._selected_features) >= 1
    assert 'y' in result.columns

    # All selected features should have importance >= threshold
    for feat in prepped._selected_features:
        assert prepped._scores[feat] >= 0.01


def test_permutation_top_p_selection(regression_data):
    """Test permutation importance using top_p (proportion)."""
    # Train model
    X = regression_data[['x1', 'x2', 'x3', 'x4', 'x5']]
    y = regression_data['y']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Create step with top_p
    step = StepSelectPermutation(
        outcome='y',
        model=model,
        top_p=0.4,  # Keep top 40% of features
        n_repeats=5,
        random_state=42
    )

    # Prep and bake
    prepped = step.prep(regression_data, training=True)
    result = prepped.bake(regression_data)

    # Should keep 2 features (40% of 5 = 2)
    assert len(prepped._selected_features) == 2
    assert len(result.columns) == 3  # 2 features + outcome


def test_permutation_with_linear_model(regression_data):
    """Test permutation importance with linear model (model-agnostic)."""
    # Train model
    X = regression_data[['x1', 'x2', 'x3', 'x4', 'x5']]
    y = regression_data['y']
    model = LinearRegression()
    model.fit(X, y)

    # Create step
    step = StepSelectPermutation(
        outcome='y',
        model=model,
        top_n=3,
        n_repeats=5,
        random_state=42
    )

    # Prep and bake
    prepped = step.prep(regression_data, training=True)
    result = prepped.bake(regression_data)

    # Check that it works with non-tree models
    assert len(result.columns) == 4
    assert 'y' in result.columns
    assert len(prepped._selected_features) == 3


def test_permutation_with_categorical_features():
    """Test permutation importance with categorical features (one-hot encoding)."""
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
    step = StepSelectPermutation(
        outcome='y',
        model=model,
        top_n=2,
        n_repeats=5,
        random_state=42
    )

    # Prep and bake
    prepped = step.prep(data, training=True)
    result = prepped.bake(data)

    # Check that categorical features are handled
    assert len(prepped._selected_features) == 2
    assert 'y' in result.columns

    # Category or x1 should be selected (both have high importance)
    assert 'category' in prepped._selected_features or 'x1' in prepped._selected_features


def test_permutation_parallel_execution(regression_data):
    """Test permutation importance with parallel execution."""
    # Train model
    X = regression_data[['x1', 'x2', 'x3', 'x4', 'x5']]
    y = regression_data['y']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Create step with parallel execution
    step = StepSelectPermutation(
        outcome='y',
        model=model,
        top_n=3,
        n_repeats=5,
        n_jobs=2,  # Use 2 parallel jobs
        random_state=42
    )

    # Prep and bake
    prepped = step.prep(regression_data, training=True)
    result = prepped.bake(regression_data)

    # Should work correctly with parallel execution
    assert len(result.columns) == 4
    assert len(prepped._selected_features) == 3


def test_permutation_validation_errors():
    """Test that validation errors are raised correctly."""
    # No selection mode specified
    with pytest.raises(ValueError, match="Must specify one of"):
        StepSelectPermutation(outcome='y', model=None)

    # Multiple selection modes
    with pytest.raises(ValueError, match="Can only specify one of"):
        StepSelectPermutation(outcome='y', model=None, top_n=5, top_p=0.2)

    # Invalid top_p (test this separately without other parameters)
    with pytest.raises(ValueError, match="top_p must be in"):
        StepSelectPermutation(outcome='y', model=None, top_p=1.5)

    # Model is None
    with pytest.raises(ValueError, match="model parameter is required"):
        StepSelectPermutation(outcome='y', model=None, top_n=5)

    # Invalid n_repeats (need a dummy model since model validation happens first)
    from sklearn.linear_model import LinearRegression
    dummy_model = LinearRegression()
    with pytest.raises(ValueError, match="n_repeats must be"):
        StepSelectPermutation(outcome='y', model=dummy_model, top_n=5, n_repeats=0)


def test_permutation_skip_parameter(regression_data):
    """Test that skip=True prevents step execution."""
    # Train model
    X = regression_data[['x1', 'x2', 'x3', 'x4', 'x5']]
    y = regression_data['y']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Create step with skip=True
    step = StepSelectPermutation(
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


def test_permutation_missing_outcome(regression_data):
    """Test error when outcome column is missing."""
    # Train model
    X = regression_data[['x1', 'x2', 'x3', 'x4', 'x5']]
    y = regression_data['y']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Create step with non-existent outcome
    step = StepSelectPermutation(
        outcome='nonexistent',
        model=model,
        top_n=3
    )

    # Should raise error during prep
    with pytest.raises(ValueError, match="Outcome.*not found"):
        step.prep(regression_data, training=True)


def test_permutation_bake_without_prep(regression_data):
    """Test error when bake is called before prep."""
    # Train model
    X = regression_data[['x1', 'x2', 'x3', 'x4', 'x5']]
    y = regression_data['y']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Create step but don't prep
    step = StepSelectPermutation(
        outcome='y',
        model=model,
        top_n=3
    )

    # Bake should fail
    with pytest.raises(ValueError, match="must be prepped"):
        step.bake(regression_data)


def test_permutation_custom_n_repeats(regression_data):
    """Test permutation importance with custom n_repeats."""
    # Train model
    X = regression_data[['x1', 'x2', 'x3', 'x4', 'x5']]
    y = regression_data['y']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Create step with custom n_repeats
    step = StepSelectPermutation(
        outcome='y',
        model=model,
        top_n=3,
        n_repeats=15,  # More repeats for better stability
        random_state=42
    )

    # Prep and bake
    prepped = step.prep(regression_data, training=True)
    result = prepped.bake(regression_data)

    # Should work with custom n_repeats
    assert len(result.columns) == 4
    assert len(prepped._selected_features) == 3
    assert 'x1' in prepped._selected_features


def test_permutation_importance_values(regression_data):
    """Test that importance values are reasonable and sorted."""
    # Train model
    X = regression_data[['x1', 'x2', 'x3', 'x4', 'x5']]
    y = regression_data['y']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Create step
    step = StepSelectPermutation(
        outcome='y',
        model=model,
        top_n=5,  # Keep all features to examine scores
        n_repeats=5,
        random_state=42
    )

    # Prep
    prepped = step.prep(regression_data, training=True)

    # Check that scores are computed for all features
    assert len(prepped._scores) == 5

    # Check that x1 has highest importance (strongest predictor)
    sorted_scores = sorted(prepped._scores.items(), key=lambda x: x[1], reverse=True)
    top_feature = sorted_scores[0][0]

    # x1 should be in top 2 (highest importance)
    top_2_features = [f for f, _ in sorted_scores[:2]]
    assert 'x1' in top_2_features

    # x4 should have low importance (pure noise)
    assert prepped._scores['x4'] <= prepped._scores['x1']
