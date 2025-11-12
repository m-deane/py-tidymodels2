"""
Test that supervised feature selection steps work with fit_nested_resamples().

This tests the fix for the issue where supervised selection steps (step_select_perm,
step_select_shap, step_select_safe) were failing because the group column was being
passed to internal models.
"""

import pandas as pd
import numpy as np
import pytest
from py_workflowsets import WorkflowSet
from py_parsnip import linear_reg
from py_recipes import recipe
from py_rsample import time_series_cv
from py_yardstick import metric_set, rmse, mae


def create_panel_data():
    """Create panel data with multiple groups."""
    np.random.seed(42)
    n_per_group = 100

    data_frames = []
    for country in ['Algeria', 'Denmark', 'Germany']:
        df = pd.DataFrame({
            'country': [country] * n_per_group,
            'date': pd.date_range('2020-01-01', periods=n_per_group, freq='D'),
            'x1': np.random.randn(n_per_group) * 10 + 50,
            'x2': np.random.randn(n_per_group) * 5 + 20,
            'x3': np.random.randn(n_per_group) * 3 + 15,
            'y': np.random.randn(n_per_group) * 100 + 500
        })
        data_frames.append(df)

    return pd.concat(data_frames, ignore_index=True)


def test_supervised_selection_with_nested_resamples():
    """Test that step_select_permutation works with fit_nested_resamples (group column excluded)."""
    from sklearn.linear_model import LinearRegression

    data = create_panel_data()

    # Create recipe with supervised permutation feature selection
    rec_perm = recipe().step_select_permutation(
        outcome='y',
        model=LinearRegression(),
        top_n=2,
        n_repeats=3
    )

    # Create workflow set
    wf_set = WorkflowSet.from_cross(
        preproc={'perm': rec_perm},
        models=[linear_reg()]
    )

    # Create CV splits per group
    cv_by_group = {}
    for country in ['Algeria', 'Denmark', 'Germany']:
        country_data = data[data['country'] == country]
        cv_by_group[country] = time_series_cv(
            country_data,
            date_column='date',
            initial='60 days',
            assess='20 days'
        )

    # This should work now (group column excluded before evaluation)
    results = wf_set.fit_nested_resamples(
        resamples=cv_by_group,
        group_col='country',
        metrics=metric_set(rmse, mae)
    )

    # Verify we got results
    assert len(results.results) > 0

    # Verify metrics exist
    metrics = results.collect_metrics(by_group=True, summarize=True)
    assert not metrics.empty
    assert 'perm' in metrics['wflow_id'].iloc[0]

    print("✅ Supervised selection with nested resamples test passed")


def test_supervised_selection_shap_with_nested_resamples():
    """Test that step_select_shap works with fit_nested_resamples."""
    from sklearn.linear_model import LinearRegression

    data = create_panel_data()

    # Create recipe with SHAP feature selection
    rec_shap = recipe().step_select_shap(
        outcome='y',
        model=LinearRegression(),
        top_n=2,
        shap_samples=50
    )

    # Create workflow set
    wf_set = WorkflowSet.from_cross(
        preproc={'shap': rec_shap},
        models=[linear_reg()]
    )

    # Create CV splits per group
    cv_by_group = {}
    for country in ['Algeria', 'Denmark', 'Germany']:
        country_data = data[data['country'] == country]
        cv_by_group[country] = time_series_cv(
            country_data,
            date_column='date',
            initial='60 days',
            assess='20 days'
        )

    # This should work now
    results = wf_set.fit_nested_resamples(
        resamples=cv_by_group,
        group_col='country',
        metrics=metric_set(rmse, mae)
    )

    # Verify results
    assert len(results.results) > 0
    metrics = results.collect_metrics(by_group=True, summarize=True)
    assert not metrics.empty

    print("✅ Supervised selection SHAP with nested resamples test passed")


def test_multiple_supervised_steps_with_nested_resamples():
    """Test multiple supervised selection steps together."""
    from sklearn.linear_model import LinearRegression

    data = create_panel_data()

    # Create recipes with different supervised selection methods
    rec_perm = recipe().step_select_permutation(
        outcome='y',
        model=LinearRegression(),
        top_n=2,
        n_repeats=3
    )
    rec_shap = recipe().step_select_shap(
        outcome='y',
        model=LinearRegression(),
        top_n=2,
        shap_samples=50
    )

    # Create workflow set with both
    wf_set = WorkflowSet.from_cross(
        preproc={'perm': rec_perm, 'shap': rec_shap},
        models=[linear_reg()]
    )

    # Create CV splits per group
    cv_by_group = {}
    for country in ['Algeria', 'Denmark', 'Germany']:
        country_data = data[data['country'] == country]
        cv_by_group[country] = time_series_cv(
            country_data,
            date_column='date',
            initial='60 days',
            assess='20 days'
        )

    # Both should work
    results = wf_set.fit_nested_resamples(
        resamples=cv_by_group,
        group_col='country',
        metrics=metric_set(rmse, mae)
    )

    # Verify we got results for all workflows
    assert len(results.results) == 2 * 3  # 2 workflows × 3 groups

    # Verify all workflow types present
    metrics = results.collect_metrics(by_group=False, summarize=True)
    workflow_ids = metrics['wflow_id'].unique()
    assert len(workflow_ids) == 2
    assert any('perm' in wf_id for wf_id in workflow_ids)
    assert any('shap' in wf_id for wf_id in workflow_ids)

    print("✅ Multiple supervised steps with nested resamples test passed")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels')

    print("\n" + "="*70)
    print("Testing Supervised Selection with fit_nested_resamples()")
    print("="*70 + "\n")

    test_supervised_selection_with_nested_resamples()
    test_supervised_selection_shap_with_nested_resamples()
    test_multiple_supervised_steps_with_nested_resamples()

    print("\n" + "="*70)
    print("✅ ALL SUPERVISED SELECTION TESTS PASSED (3 tests)")
    print("="*70)
