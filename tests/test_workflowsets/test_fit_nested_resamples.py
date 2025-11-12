"""
Test WorkflowSet.fit_nested_resamples() and fit_global_resamples().

Tests group-aware cross-validation methods for robust model evaluation.
"""

import pandas as pd
import numpy as np
import pytest
from py_workflowsets import WorkflowSet
from py_parsnip import linear_reg, rand_forest
from py_recipes import recipe
from py_yardstick import metric_set, rmse, mae


def create_panel_data():
    """Create panel data with multiple groups."""
    np.random.seed(42)
    n_per_group = 100

    data_frames = []
    for country in ['USA', 'Germany', 'Japan']:
        df = pd.DataFrame({
            'country': [country] * n_per_group,
            'date': pd.date_range('2020-01-01', periods=n_per_group, freq='D'),
            'x1': np.random.randn(n_per_group) * 10 + 50,
            'x2': np.random.randn(n_per_group) * 5 + 20,
            'y': np.random.randn(n_per_group) * 100 + 500
        })
        data_frames.append(df)

    return pd.concat(data_frames, ignore_index=True)


def test_fit_nested_resamples_basic():
    """Test basic fit_nested_resamples with time series CV."""
    from py_rsample import time_series_cv

    data = create_panel_data()

    # Create simple workflow set
    wf_set = WorkflowSet.from_cross(
        preproc=["y ~ x1", "y ~ x1 + x2"],
        models=[linear_reg()]
    )

    # Create CV splits per group
    cv_by_group = {}
    for country in ['USA', 'Germany', 'Japan']:
        country_data = data[data['country'] == country]
        cv_by_group[country] = time_series_cv(
            country_data,
            date_column='date',
            initial='60 days',
            assess='20 days'
        )

    # Fit with nested CV per group
    results = wf_set.fit_nested_resamples(
        resamples=cv_by_group,
        group_col='country',
        metrics=metric_set(rmse, mae)
    )

    # Verify structure
    assert hasattr(results, 'collect_metrics')
    assert hasattr(results, 'rank_results')
    assert hasattr(results, 'extract_best_workflow')

    # Verify results exist for all workflows and groups
    assert len(results.results) == 2 * 3  # 2 workflows × 3 groups

    print("✅ fit_nested_resamples basic test passed")


def test_fit_nested_resamples_collect_metrics():
    """Test collect_metrics with by_group parameter."""
    from py_rsample import time_series_cv

    data = create_panel_data()

    wf_set = WorkflowSet.from_cross(
        preproc=["y ~ x1"],
        models=[linear_reg()]
    )

    # Create CV splits per group
    cv_by_group = {}
    for country in ['USA', 'Germany', 'Japan']:
        country_data = data[data['country'] == country]
        cv_by_group[country] = time_series_cv(
            country_data,
            date_column='date',
            initial='60 days',
            assess='20 days'
        )

    results = wf_set.fit_nested_resamples(
        resamples=cv_by_group,
        group_col='country'
    )

    # Collect metrics per group
    metrics_by_group = results.collect_metrics(by_group=True, summarize=True)

    assert isinstance(metrics_by_group, pd.DataFrame)
    assert 'group' in metrics_by_group.columns
    assert 'mean' in metrics_by_group.columns
    assert 'std' in metrics_by_group.columns
    assert set(metrics_by_group['group'].unique()) == {'USA', 'Germany', 'Japan'}

    # Collect metrics averaged across groups
    metrics_overall = results.collect_metrics(by_group=False, summarize=True)

    assert 'group' in metrics_overall.columns
    assert metrics_overall['group'].iloc[0] == 'global'

    print("✅ fit_nested_resamples collect_metrics test passed")


def test_fit_nested_resamples_rank_results():
    """Test rank_results method."""
    from py_rsample import time_series_cv

    data = create_panel_data()

    wf_set = WorkflowSet.from_cross(
        preproc=["y ~ x1", "y ~ x1 + x2"],
        models=[linear_reg()]
    )

    # Create CV splits per group
    cv_by_group = {}
    for country in ['USA', 'Germany', 'Japan']:
        country_data = data[data['country'] == country]
        cv_by_group[country] = time_series_cv(
            country_data,
            date_column='date',
            initial='60 days',
            assess='20 days'
        )

    results = wf_set.fit_nested_resamples(
        resamples=cv_by_group,
        group_col='country'
    )

    # Rank overall
    ranked_overall = results.rank_results('rmse', by_group=False, n=2)

    assert isinstance(ranked_overall, pd.DataFrame)
    assert 'rank' in ranked_overall.columns
    assert len(ranked_overall) == 2
    assert ranked_overall['rank'].iloc[0] == 1

    # Rank per group
    ranked_by_group = results.rank_results('rmse', by_group=True, n=2)

    assert 'group' in ranked_by_group.columns
    assert len(ranked_by_group['group'].unique()) == 3

    print("✅ fit_nested_resamples rank_results test passed")


def test_fit_nested_resamples_extract_best():
    """Test extract_best_workflow method."""
    from py_rsample import time_series_cv

    data = create_panel_data()

    wf_set = WorkflowSet.from_cross(
        preproc=["y ~ x1", "y ~ x1 + x2"],
        models=[linear_reg()]
    )

    # Create CV splits per group
    cv_by_group = {}
    for country in ['USA', 'Germany', 'Japan']:
        country_data = data[data['country'] == country]
        cv_by_group[country] = time_series_cv(
            country_data,
            date_column='date',
            initial='60 days',
            assess='20 days'
        )

    results = wf_set.fit_nested_resamples(
        resamples=cv_by_group,
        group_col='country'
    )

    # Extract best overall
    best_overall = results.extract_best_workflow('rmse', by_group=False)

    assert isinstance(best_overall, str)
    assert best_overall in wf_set.workflows.keys()

    # Extract best per group
    best_by_group = results.extract_best_workflow('rmse', by_group=True)

    assert isinstance(best_by_group, pd.DataFrame)
    assert 'group' in best_by_group.columns
    assert len(best_by_group) == 3  # 3 groups

    print("✅ fit_nested_resamples extract_best test passed")


def test_fit_global_resamples_basic():
    """Test basic fit_global_resamples."""
    from py_rsample import time_series_cv

    data = create_panel_data()

    wf_set = WorkflowSet.from_cross(
        preproc=["y ~ x1"],
        models=[linear_reg()]
    )

    # Create CV splits per group
    cv_by_group = {}
    for country in ['USA', 'Germany', 'Japan']:
        country_data = data[data['country'] == country]
        cv_by_group[country] = time_series_cv(
            country_data,
            date_column='date',
            initial='60 days',
            assess='20 days'
        )

    # Fit global model with per-group CV
    results = wf_set.fit_global_resamples(
        data=data,
        resamples=cv_by_group,
        group_col='country',
        metrics=metric_set(rmse, mae)
    )

    # Verify structure (same as nested_resamples)
    assert hasattr(results, 'collect_metrics')
    assert hasattr(results, 'rank_results')

    # Should have results for all groups
    metrics = results.collect_metrics(by_group=True)
    assert set(metrics['group'].unique()) == {'USA', 'Germany', 'Japan'}

    print("✅ fit_global_resamples basic test passed")


def test_fit_nested_resamples_with_recipe():
    """Test fit_nested_resamples with recipe preprocessing."""
    from py_rsample import time_series_cv

    data = create_panel_data()

    rec = recipe().step_normalize(['x1', 'x2'])

    wf_set = WorkflowSet.from_cross(
        preproc={'rec_norm': rec},
        models=[linear_reg()]
    )

    # Create CV splits per group
    cv_by_group = {}
    for country in ['USA', 'Germany', 'Japan']:
        country_data = data[data['country'] == country]
        cv_by_group[country] = time_series_cv(
            country_data,
            date_column='date',
            initial='60 days',
            assess='20 days'
        )

    results = wf_set.fit_nested_resamples(
        resamples=cv_by_group,
        group_col='country'
    )

    metrics = results.collect_metrics(by_group=True)

    # Verify it worked
    assert not metrics.empty
    assert 'rec_norm' in metrics['wflow_id'].iloc[0]

    print("✅ fit_nested_resamples with recipe test passed")


def test_fit_nested_resamples_autoplot():
    """Test autoplot visualization."""
    from py_rsample import time_series_cv

    data = create_panel_data()

    wf_set = WorkflowSet.from_cross(
        preproc=["y ~ x1", "y ~ x1 + x2"],
        models=[linear_reg()]
    )

    # Create CV splits per group
    cv_by_group = {}
    for country in ['USA', 'Germany', 'Japan']:
        country_data = data[data['country'] == country]
        cv_by_group[country] = time_series_cv(
            country_data,
            date_column='date',
            initial='60 days',
            assess='20 days'
        )

    results = wf_set.fit_nested_resamples(
        resamples=cv_by_group,
        group_col='country'
    )

    # Test overall plot
    fig1 = results.autoplot('rmse', by_group=False, top_n=2)
    assert fig1 is not None

    # Test per-group plot
    fig2 = results.autoplot('rmse', by_group=True, top_n=2)
    assert fig2 is not None

    print("✅ fit_nested_resamples autoplot test passed")


def test_fit_nested_resamples_multiple_models():
    """Test with multiple models."""
    from py_rsample import time_series_cv

    data = create_panel_data()

    wf_set = WorkflowSet.from_cross(
        preproc=["y ~ x1"],
        models=[linear_reg(), rand_forest(trees=10).set_mode('regression')]
    )

    # Create CV splits per group
    cv_by_group = {}
    for country in ['USA', 'Germany', 'Japan']:
        country_data = data[data['country'] == country]
        cv_by_group[country] = time_series_cv(
            country_data,
            date_column='date',
            initial='60 days',
            assess='20 days'
        )

    results = wf_set.fit_nested_resamples(
        resamples=cv_by_group,
        group_col='country'
    )

    # Should have results for both models
    metrics = results.collect_metrics(by_group=False)
    assert len(metrics['wflow_id'].unique()) == 2

    # Rank should work
    ranked = results.rank_results('rmse', n=2)
    assert len(ranked) == 2

    print("✅ fit_nested_resamples multiple models test passed")


def test_fit_nested_resamples_fold_level_metrics():
    """Test getting fold-level metrics (unsummarized)."""
    from py_rsample import time_series_cv

    data = create_panel_data()

    wf_set = WorkflowSet.from_cross(
        preproc=["y ~ x1"],
        models=[linear_reg()]
    )

    # Create CV splits per group
    cv_by_group = {}
    for country in ['USA', 'Germany', 'Japan']:
        country_data = data[data['country'] == country]
        cv_by_group[country] = time_series_cv(
            country_data,
            date_column='date',
            initial='60 days',
            assess='20 days'
        )

    results = wf_set.fit_nested_resamples(
        resamples=cv_by_group,
        group_col='country'
    )

    # Get fold-level detail
    fold_metrics = results.collect_metrics(by_group=True, summarize=False)

    # Should have multiple rows per workflow/group (one per fold)
    assert len(fold_metrics) > len(results.results) * 2  # More than 2 metrics per (wf, group)
    assert 'fold' in fold_metrics.columns or 'value' in fold_metrics.columns

    print("✅ fit_nested_resamples fold-level metrics test passed")


def test_fit_nested_resamples_verbose():
    """Test verbose output parameter."""
    from py_rsample import time_series_cv
    import sys
    from io import StringIO

    data = create_panel_data()

    wf_set = WorkflowSet.from_cross(
        preproc=["y ~ x1", "y ~ x1 + x2"],
        models=[linear_reg()]
    )

    # Create CV splits per group
    cv_by_group = {}
    for country in ['USA', 'Germany', 'Japan']:
        country_data = data[data['country'] == country]
        cv_by_group[country] = time_series_cv(
            country_data,
            date_column='date',
            initial='60 days',
            assess='20 days'
        )

    # Capture stdout to verify verbose output
    captured_output = StringIO()
    sys.stdout = captured_output

    # Test with verbose=True
    results = wf_set.fit_nested_resamples(
        resamples=cv_by_group,
        group_col='country',
        verbose=True
    )

    # Restore stdout
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()

    # Verify verbose output contains expected elements
    assert "Total evaluations:" in output
    assert "Workflow:" in output
    assert "Group:" in output
    assert "folds" in output
    assert "✓" in output

    # Verify results are still correct
    assert len(results.results) == 2 * 3  # 2 workflows × 3 groups
    metrics = results.collect_metrics(by_group=True)
    assert not metrics.empty

    print("✅ fit_nested_resamples verbose test passed")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels')

    print("\n" + "="*60)
    print("Testing WorkflowSet Nested Resamples Methods")
    print("="*60 + "\n")

    test_fit_nested_resamples_basic()
    test_fit_nested_resamples_collect_metrics()
    test_fit_nested_resamples_rank_results()
    test_fit_nested_resamples_extract_best()
    test_fit_global_resamples_basic()
    test_fit_nested_resamples_with_recipe()
    test_fit_nested_resamples_autoplot()
    test_fit_nested_resamples_multiple_models()
    test_fit_nested_resamples_fold_level_metrics()
    test_fit_nested_resamples_verbose()

    print("\n" + "="*60)
    print("✅ ALL NESTED RESAMPLES TESTS PASSED (10 tests)")
    print("="*60)
