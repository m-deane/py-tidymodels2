"""
Test compare_train_cv() helper method for WorkflowSetNestedResamples.
"""

import pandas as pd
import numpy as np
import pytest
from py_workflowsets import WorkflowSet
from py_parsnip import linear_reg
from py_recipes import recipe
from py_rsample import time_series_cv
from py_yardstick import metric_set, rmse, mae, r_squared


def create_test_data():
    """Create simple test data with 3 groups."""
    np.random.seed(42)
    n_per_group = 80

    data_list = []
    for group in ['A', 'B', 'C']:
        df = pd.DataFrame({
            'group': [group] * n_per_group,
            'date': pd.date_range('2020-01-01', periods=n_per_group, freq='D'),
            'x1': np.random.randn(n_per_group) * 10 + 50,
            'x2': np.random.randn(n_per_group) * 5 + 20,
            'target': np.random.randn(n_per_group) * 100 + 500
        })
        data_list.append(df)

    return pd.concat(data_list, ignore_index=True)


def test_compare_train_cv_basic():
    """Test basic compare_train_cv functionality."""
    data = create_test_data()

    # Create simple workflow set
    rec1 = recipe()
    rec2 = recipe().step_normalize()

    wf_set = WorkflowSet.from_cross(
        preproc={'raw': rec1, 'norm': rec2},
        models=[linear_reg()]
    )

    # Create CV splits per group
    cv_by_group = {}
    for group in ['A', 'B', 'C']:
        group_data = data[data['group'] == group]
        cv_by_group[group] = time_series_cv(
            group_data,
            date_column='date',
            initial='40 days',
            assess='20 days'
        )

    # First fit on full training data to get training stats
    train_results = wf_set.fit_nested(data, group_col='group')
    outputs, coeffs, train_stats = train_results.extract_outputs()

    # Run CV evaluation
    cv_results = wf_set.fit_nested_resamples(
        resamples=cv_by_group,
        group_col='group',
        metrics=metric_set(rmse, mae, r_squared)
    )

    # Compare train vs CV
    comparison = cv_results.compare_train_cv(train_stats)

    # Assertions
    assert not comparison.empty, "Comparison should not be empty"
    assert 'wflow_id' in comparison.columns
    assert 'group' in comparison.columns
    assert 'rmse_train' in comparison.columns
    assert 'rmse_cv' in comparison.columns
    assert 'rmse_overfit_ratio' in comparison.columns
    assert 'fit_quality' in comparison.columns

    # Check all workflows and groups present
    assert len(comparison) == 2 * 3, "Should have 2 workflows × 3 groups = 6 rows"
    assert set(comparison['group']) == {'A', 'B', 'C'}
    assert len(comparison['wflow_id'].unique()) == 2

    print("✅ Basic compare_train_cv test passed")


def test_compare_train_cv_custom_metrics():
    """Test compare_train_cv with custom metrics."""
    data = create_test_data()

    wf_set = WorkflowSet.from_cross(
        preproc={'raw': recipe()},
        models=[linear_reg()]
    )

    cv_by_group = {}
    for group in ['A', 'B', 'C']:
        group_data = data[data['group'] == group]
        cv_by_group[group] = time_series_cv(
            group_data,
            date_column='date',
            initial='40 days',
            assess='20 days'
        )

    # Fit on full data first
    train_results = wf_set.fit_nested(data, group_col='group')
    outputs, coeffs, train_stats = train_results.extract_outputs()

    cv_results = wf_set.fit_nested_resamples(
        resamples=cv_by_group,
        group_col='group',
        metrics=metric_set(rmse, mae)  # Only 2 metrics
    )

    # Compare with custom metrics
    comparison = cv_results.compare_train_cv(train_stats, metrics=['rmse', 'mae'])

    # Assertions
    assert 'rmse_train' in comparison.columns
    assert 'rmse_cv' in comparison.columns
    assert 'mae_train' in comparison.columns
    assert 'mae_cv' in comparison.columns
    assert 'rmse_overfit_ratio' in comparison.columns
    assert 'mae_overfit_ratio' in comparison.columns

    # Should not have r_squared columns
    assert 'r_squared_train' not in comparison.columns
    assert 'r_squared_cv' not in comparison.columns

    print("✅ Custom metrics test passed")


def test_compare_train_cv_overfitting_indicators():
    """Test overfitting indicator calculations."""
    data = create_test_data()

    wf_set = WorkflowSet.from_cross(
        preproc={'raw': recipe()},
        models=[linear_reg()]
    )

    cv_by_group = {}
    for group in ['A', 'B', 'C']:
        group_data = data[data['group'] == group]
        cv_by_group[group] = time_series_cv(
            group_data,
            date_column='date',
            initial='40 days',
            assess='20 days'
        )

    # Fit on full data first
    train_results = wf_set.fit_nested(data, group_col='group')
    outputs, coeffs, train_stats = train_results.extract_outputs()

    cv_results = wf_set.fit_nested_resamples(
        resamples=cv_by_group,
        group_col='group',
        metrics=metric_set(rmse, mae, r_squared)
    )

    comparison = cv_results.compare_train_cv(train_stats)

    # Check overfit ratio calculation
    for idx, row in comparison.iterrows():
        if pd.notna(row['rmse_train']) and pd.notna(row['rmse_cv']):
            expected_ratio = row['rmse_cv'] / row['rmse_train']
            assert abs(row['rmse_overfit_ratio'] - expected_ratio) < 0.001

    # Check generalization drop for r_squared
    assert 'r_squared_generalization_drop' in comparison.columns
    for idx, row in comparison.iterrows():
        if pd.notna(row['r_squared_train']) and pd.notna(row['r_squared_cv']):
            expected_drop = row['r_squared_train'] - row['r_squared_cv']
            assert abs(row['r_squared_generalization_drop'] - expected_drop) < 0.001

    # Check fit_quality status
    assert comparison['fit_quality'].notna().all()
    assert set(comparison['fit_quality'].unique()).issubset({
        '🔴 Severe Overfit',
        '🟡 Moderate Overfit',
        '🟢 Good Generalization',
        '⚪ Normal',
        '❓ Unknown'
    })

    print("✅ Overfitting indicators test passed")


def test_compare_train_cv_sorting():
    """Test that results are sorted by CV performance."""
    data = create_test_data()

    # Create workflows that should perform differently
    rec1 = recipe()  # Raw
    rec2 = recipe().step_normalize()  # Normalized

    wf_set = WorkflowSet.from_cross(
        preproc={'raw': rec1, 'norm': rec2},
        models=[linear_reg()]
    )

    cv_by_group = {}
    for group in ['A', 'B', 'C']:
        group_data = data[data['group'] == group]
        cv_by_group[group] = time_series_cv(
            group_data,
            date_column='date',
            initial='40 days',
            assess='20 days'
        )

    # Fit on full data first
    train_results = wf_set.fit_nested(data, group_col='group')
    outputs, coeffs, train_stats = train_results.extract_outputs()

    cv_results = wf_set.fit_nested_resamples(
        resamples=cv_by_group,
        group_col='group',
        metrics=metric_set(rmse, mae)
    )

    comparison = cv_results.compare_train_cv(train_stats)

    # Check sorting (RMSE should be ascending)
    rmse_cv_values = comparison['rmse_cv'].dropna().values
    assert all(rmse_cv_values[i] <= rmse_cv_values[i+1] for i in range(len(rmse_cv_values)-1)), \
        "Results should be sorted by CV RMSE in ascending order"

    print("✅ Sorting test passed")


def test_compare_train_cv_missing_columns():
    """Test error handling for missing columns in train_stats."""
    data = create_test_data()

    wf_set = WorkflowSet.from_cross(
        preproc={'raw': recipe()},
        models=[linear_reg()]
    )

    cv_by_group = {}
    for group in ['A', 'B', 'C']:
        group_data = data[data['group'] == group]
        cv_by_group[group] = time_series_cv(
            group_data,
            date_column='date',
            initial='40 days',
            assess='20 days'
        )

    cv_results = wf_set.fit_nested_resamples(
        resamples=cv_by_group,
        group_col='group',
        metrics=metric_set(rmse, mae)
    )

    # Create invalid train_stats (missing required columns)
    invalid_stats = pd.DataFrame({
        'wflow_id': ['test'],
        'group': ['A']
        # Missing metric columns!
    })

    # Should raise ValueError
    with pytest.raises(ValueError, match="missing required columns"):
        cv_results.compare_train_cv(invalid_stats)

    print("✅ Missing columns error handling test passed")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels')

    print("\n" + "="*70)
    print("Testing compare_train_cv() Helper Method")
    print("="*70 + "\n")

    test_compare_train_cv_basic()
    test_compare_train_cv_custom_metrics()
    test_compare_train_cv_overfitting_indicators()
    test_compare_train_cv_sorting()
    test_compare_train_cv_missing_columns()

    print("\n" + "="*70)
    print("✅ ALL compare_train_cv() TESTS PASSED (5 tests)")
    print("="*70)
