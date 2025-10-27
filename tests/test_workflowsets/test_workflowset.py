"""
Tests for py-workflowsets module.
"""

import pytest
import pandas as pd
import numpy as np
from py_workflowsets import WorkflowSet, WorkflowSetResults
from py_workflows import workflow
from py_parsnip import linear_reg
from py_rsample import vfold_cv
from py_yardstick import metric_set, rmse, mae, r_squared


# Fixtures
@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n) * 0.5

    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    df['y'] = y
    return df


@pytest.fixture
def sample_workflows():
    """Create sample workflows"""
    wf1 = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
    wf2 = workflow().add_formula("y ~ x1 + x2 + x3").add_model(linear_reg(penalty=0.1))
    return [wf1, wf2]


@pytest.fixture
def sample_formulas():
    """Create sample formulas for cross-product"""
    return ["y ~ x1", "y ~ x1 + x2", "y ~ x1 + x2 + x3"]


@pytest.fixture
def sample_models():
    """Create sample models for cross-product"""
    return [
        linear_reg(),
        linear_reg(penalty=0.1, mixture=0.5),
    ]


# WorkflowSet creation tests
def test_workflow_set_from_workflows(sample_workflows):
    """Test creating WorkflowSet from list of workflows"""
    wf_set = WorkflowSet.from_workflows(
        sample_workflows,
        ids=["linear_simple", "linear_ridge"]
    )

    assert len(wf_set) == 2
    assert "linear_simple" in wf_set.workflows
    assert "linear_ridge" in wf_set.workflows
    assert len(wf_set.info) == 2
    assert wf_set.info["wflow_id"].tolist() == ["linear_simple", "linear_ridge"]


def test_workflow_set_from_workflows_auto_ids(sample_workflows):
    """Test auto-generated IDs"""
    wf_set = WorkflowSet.from_workflows(sample_workflows)

    assert len(wf_set) == 2
    assert "workflow_1" in wf_set.workflows
    assert "workflow_2" in wf_set.workflows


def test_workflow_set_from_cross(sample_formulas, sample_models):
    """Test creating WorkflowSet from cross-product"""
    wf_set = WorkflowSet.from_cross(
        preproc=sample_formulas,
        models=sample_models
    )

    # Should have 3 formulas × 2 models = 6 workflows
    assert len(wf_set) == 6
    assert len(wf_set.info) == 6

    # Check that all combinations exist (with model index)
    assert "prep_1_linear_reg_1" in wf_set.workflows
    assert "prep_1_linear_reg_2" in wf_set.workflows
    assert "prep_2_linear_reg_1" in wf_set.workflows
    assert "prep_2_linear_reg_2" in wf_set.workflows
    assert "prep_3_linear_reg_1" in wf_set.workflows
    assert "prep_3_linear_reg_2" in wf_set.workflows


def test_workflow_set_from_cross_custom_ids(sample_formulas, sample_models):
    """Test cross-product with custom IDs"""
    wf_set = WorkflowSet.from_cross(
        preproc=sample_formulas,
        models=sample_models,
        ids=["simple", "medium", "full"]
    )

    assert "simple_linear_reg_1" in wf_set.workflows
    assert "medium_linear_reg_1" in wf_set.workflows
    assert "full_linear_reg_1" in wf_set.workflows


def test_workflow_set_len(sample_workflows):
    """Test __len__ method"""
    wf_set = WorkflowSet.from_workflows(sample_workflows)
    assert len(wf_set) == 2


def test_workflow_set_iter(sample_workflows):
    """Test iteration"""
    wf_set = WorkflowSet.from_workflows(sample_workflows, ids=["wf1", "wf2"])
    wf_ids = list(wf_set)
    assert wf_ids == ["wf1", "wf2"]


def test_workflow_set_getitem(sample_workflows):
    """Test __getitem__"""
    wf_set = WorkflowSet.from_workflows(sample_workflows, ids=["wf1", "wf2"])
    wf = wf_set["wf1"]
    assert wf is not None
    assert hasattr(wf, "fit")


# fit_resamples tests
def test_fit_resamples_basic(sample_data, sample_workflows):
    """Test basic fit_resamples functionality"""
    wf_set = WorkflowSet.from_workflows(sample_workflows, ids=["wf1", "wf2"])

    # Create CV folds
    folds = vfold_cv(sample_data, v=3, seed=42)

    # Create metrics
    metrics = metric_set(rmse, mae)

    # Fit resamples
    results = wf_set.fit_resamples(folds, metrics=metrics)

    assert isinstance(results, WorkflowSetResults)
    assert len(results.results) == 2  # Two workflows


def test_collect_metrics_summarized(sample_data, sample_workflows):
    """Test collect_metrics with summarization"""
    wf_set = WorkflowSet.from_workflows(sample_workflows, ids=["wf1", "wf2"])
    folds = vfold_cv(sample_data, v=3, seed=42)
    metrics = metric_set(rmse, mae)

    results = wf_set.fit_resamples(folds, metrics=metrics)
    metrics_df = results.collect_metrics(summarize=True)

    assert isinstance(metrics_df, pd.DataFrame)
    assert "wflow_id" in metrics_df.columns
    assert "metric" in metrics_df.columns
    assert "mean" in metrics_df.columns
    assert "std" in metrics_df.columns
    assert len(metrics_df) == 4  # 2 workflows × 2 metrics


def test_collect_metrics_unsummarized(sample_data, sample_workflows):
    """Test collect_metrics without summarization"""
    wf_set = WorkflowSet.from_workflows(sample_workflows, ids=["wf1", "wf2"])
    folds = vfold_cv(sample_data, v=3, seed=42)
    metrics = metric_set(rmse, mae)

    results = wf_set.fit_resamples(folds, metrics=metrics)
    metrics_df = results.collect_metrics(summarize=False)

    assert isinstance(metrics_df, pd.DataFrame)
    assert "wflow_id" in metrics_df.columns
    # Should have more rows (one per resample per workflow per metric)
    assert len(metrics_df) > 4


def test_rank_results_basic(sample_data, sample_workflows):
    """Test rank_results basic functionality"""
    wf_set = WorkflowSet.from_workflows(sample_workflows, ids=["wf1", "wf2"])
    folds = vfold_cv(sample_data, v=3, seed=42)
    metrics = metric_set(rmse, mae)

    results = wf_set.fit_resamples(folds, metrics=metrics)
    ranked = results.rank_results("rmse", n=10)

    assert isinstance(ranked, pd.DataFrame)
    assert "rank" in ranked.columns
    assert "wflow_id" in ranked.columns
    assert "mean" in ranked.columns
    assert len(ranked) == 2  # Both workflows


def test_rank_results_top_n(sample_data, sample_formulas, sample_models):
    """Test rank_results with top N"""
    wf_set = WorkflowSet.from_cross(sample_formulas, sample_models)
    folds = vfold_cv(sample_data, v=3, seed=42)
    metrics = metric_set(rmse)

    results = wf_set.fit_resamples(folds, metrics=metrics)
    ranked = results.rank_results("rmse", n=3)

    assert len(ranked) <= 3
    assert ranked["rank"].tolist() == [1, 2, 3]


def test_rank_results_select_best(sample_data, sample_formulas, sample_models):
    """Test rank_results with select_best"""
    wf_set = WorkflowSet.from_cross(sample_formulas, sample_models)
    folds = vfold_cv(sample_data, v=3, seed=42)
    metrics = metric_set(rmse)

    results = wf_set.fit_resamples(folds, metrics=metrics)
    ranked = results.rank_results("rmse", select_best=True)

    # Should have one row per model type
    assert isinstance(ranked, pd.DataFrame)
    assert len(ranked) <= len(sample_models)


def test_workflow_map_fit_resamples(sample_data, sample_workflows):
    """Test workflow_map with fit_resamples"""
    wf_set = WorkflowSet.from_workflows(sample_workflows, ids=["wf1", "wf2"])
    folds = vfold_cv(sample_data, v=3, seed=42)
    metrics = metric_set(rmse, mae)

    results = wf_set.workflow_map("fit_resamples", resamples=folds, metrics=metrics)

    assert isinstance(results, WorkflowSetResults)
    assert len(results.results) == 2


def test_workflow_map_invalid_fn(sample_workflows):
    """Test workflow_map with invalid function"""
    wf_set = WorkflowSet.from_workflows(sample_workflows)

    with pytest.raises(ValueError, match="Unknown function"):
        wf_set.workflow_map("invalid_function")


# Edge cases
def test_workflow_set_mismatched_ids(sample_workflows):
    """Test error when IDs length doesn't match workflows"""
    with pytest.raises(ValueError, match="Length of ids must match"):
        WorkflowSet.from_workflows(sample_workflows, ids=["only_one"])


def test_workflow_set_empty_workflows():
    """Test creating WorkflowSet with empty list"""
    wf_set = WorkflowSet.from_workflows([], ids=[])
    assert len(wf_set) == 0


def test_autoplot_basic(sample_data, sample_workflows):
    """Test autoplot basic functionality"""
    wf_set = WorkflowSet.from_workflows(sample_workflows, ids=["wf1", "wf2"])
    folds = vfold_cv(sample_data, v=3, seed=42)
    metrics = metric_set(rmse, mae)

    results = wf_set.fit_resamples(folds, metrics=metrics)

    # Test that autoplot runs without error
    try:
        fig = results.autoplot("rmse")
        assert fig is not None
    except Exception as e:
        pytest.fail(f"autoplot raised {e}")


def test_collect_predictions(sample_data, sample_workflows):
    """Test collect_predictions functionality"""
    wf_set = WorkflowSet.from_workflows(sample_workflows, ids=["wf1", "wf2"])
    folds = vfold_cv(sample_data, v=3, seed=42)
    metrics = metric_set(rmse)

    results = wf_set.fit_resamples(folds, metrics=metrics, control={"save_pred": True})
    preds_df = results.collect_predictions()

    assert isinstance(preds_df, pd.DataFrame)
    assert "wflow_id" in preds_df.columns


# Integration tests
def test_full_workflow_comparison(sample_data):
    """Test complete workflow comparison scenario"""
    # Create workflows with different complexity
    formulas = ["y ~ x1", "y ~ x1 + x2", "y ~ x1 + x2 + x3"]
    models = [linear_reg(), linear_reg(penalty=0.1)]

    # Create workflow set
    wf_set = WorkflowSet.from_cross(formulas, models)
    assert len(wf_set) == 6  # 3 formulas × 2 models

    # Create CV folds
    folds = vfold_cv(sample_data, v=5, seed=42)

    # Fit all workflows
    metrics = metric_set(rmse, mae, r_squared)
    results = wf_set.fit_resamples(folds, metrics=metrics)

    # Collect and rank
    metrics_df = results.collect_metrics()
    assert len(metrics_df) == 18  # 6 workflows × 3 metrics

    ranked = results.rank_results("rmse", n=3)
    assert len(ranked) == 3
    assert ranked.iloc[0]["rank"] == 1

    # Plot
    fig = results.autoplot("rmse", top_n=6)
    assert fig is not None
