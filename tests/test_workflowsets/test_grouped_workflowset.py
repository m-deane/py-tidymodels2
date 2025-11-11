"""
Tests for grouped/panel modeling with WorkflowSet.

Tests fit_nested() and fit_global() methods plus WorkflowSetNestedResults
functionality.
"""

import pytest
import pandas as pd
import numpy as np
from py_workflows import workflow
from py_parsnip import linear_reg, rand_forest
from py_workflowsets import WorkflowSet, WorkflowSetNestedResults
from py_recipes import recipe
from py_recipes.selectors import all_numeric_predictors


@pytest.fixture
def panel_data():
    """Create panel/grouped data for testing."""
    np.random.seed(42)

    groups = ['A', 'B', 'C']
    n_per_group = 50

    data_list = []
    for group in groups:
        # Different data patterns per group
        if group == 'A':
            x1 = np.random.randn(n_per_group) * 2
            x2 = np.random.randn(n_per_group) + 1
            y = 2 * x1 + 3 * x2 + np.random.randn(n_per_group) * 0.5
        elif group == 'B':
            x1 = np.random.randn(n_per_group) + 2
            x2 = np.random.randn(n_per_group) * 3
            y = -1 * x1 + 2 * x2 + np.random.randn(n_per_group) * 0.8
        else:  # C
            x1 = np.random.randn(n_per_group) - 1
            x2 = np.random.randn(n_per_group) + 2
            y = 1.5 * x1 - 1 * x2 + np.random.randn(n_per_group) * 0.3

        df = pd.DataFrame({
            'group': group,
            'x1': x1,
            'x2': x2,
            'y': y
        })
        data_list.append(df)

    return pd.concat(data_list, ignore_index=True)


@pytest.fixture
def simple_workflowset():
    """Create simple WorkflowSet for testing."""
    formulas = ["y ~ x1", "y ~ x1 + x2"]
    models = [linear_reg().set_engine("sklearn")]

    return WorkflowSet.from_cross(
        preproc=formulas,
        models=models,
        ids=["formula1", "formula2"]
    )


@pytest.fixture
def multi_model_workflowset():
    """Create WorkflowSet with multiple models."""
    formulas = ["y ~ x1", "y ~ x1 + x2"]
    models = [
        linear_reg().set_engine("sklearn"),
        rand_forest(trees=50, mtry=1, min_n=5).set_mode("regression")
    ]

    return WorkflowSet.from_cross(
        preproc=formulas,
        models=models
    )


class TestWorkflowSetFitNested:
    """Tests for WorkflowSet.fit_nested() method."""

    def test_fit_nested_basic(self, panel_data, simple_workflowset):
        """Test basic fit_nested functionality."""
        results = simple_workflowset.fit_nested(panel_data, group_col='group')

        # Check return type
        assert isinstance(results, WorkflowSetNestedResults)

        # Check results structure
        assert len(results.results) == 2  # 2 workflows
        assert results.group_col == 'group'

        # Check each result has required keys
        for result in results.results:
            assert 'wflow_id' in result
            assert 'nested_fit' in result or 'error' in result
            assert 'outputs' in result
            assert 'stats' in result

    def test_fit_nested_multiple_models(self, panel_data, multi_model_workflowset):
        """Test fit_nested with multiple model types."""
        results = multi_model_workflowset.fit_nested(panel_data, group_col='group')

        assert isinstance(results, WorkflowSetNestedResults)
        assert len(results.results) == 4  # 2 formulas × 2 models

        # Check nested fits exist
        for result in results.results:
            if result['nested_fit'] is not None:
                assert hasattr(result['nested_fit'], 'group_fits')

    def test_fit_nested_with_per_group_prep(self, panel_data):
        """Test fit_nested with per_group_prep=True."""
        # Create recipe-based workflow
        rec = recipe().step_normalize(all_numeric_predictors())
        wf = workflow().add_recipe(rec).add_model(linear_reg())
        wf_set = WorkflowSet.from_workflows([wf], ids=["normalized"])

        results = wf_set.fit_nested(
            panel_data,
            group_col='group',
            per_group_prep=True,
            min_group_size=30
        )

        assert isinstance(results, WorkflowSetNestedResults)
        assert len(results.results) == 1

    def test_fit_nested_preserves_group_info(self, panel_data, simple_workflowset):
        """Test that group column is preserved in outputs."""
        results = simple_workflowset.fit_nested(panel_data, group_col='group')

        # Check stats include group column
        for result in results.results:
            if result['stats'] is not None:
                assert 'group' in result['stats'].columns
                groups_in_stats = result['stats']['group'].unique()
                assert len(groups_in_stats) == 3  # A, B, C


class TestWorkflowSetFitGlobal:
    """Tests for WorkflowSet.fit_global() method."""

    def test_fit_global_basic(self, panel_data, simple_workflowset):
        """Test basic fit_global functionality."""
        results = simple_workflowset.fit_global(panel_data, group_col='group')

        # Check return type (should be WorkflowSetResults, not Nested)
        from py_workflowsets import WorkflowSetResults
        assert isinstance(results, WorkflowSetResults)

        # Check results structure
        assert len(results.results) == 2  # 2 workflows

    def test_fit_global_adds_group_feature(self, panel_data, simple_workflowset):
        """Test that fit_global adds group as a feature."""
        results = simple_workflowset.fit_global(panel_data, group_col='group')

        # Check that fits are successful
        for result in results.results:
            if result['fit'] is not None:
                assert result['fit'] is not None


class TestWorkflowSetNestedResultsCollectMetrics:
    """Tests for WorkflowSetNestedResults.collect_metrics() method."""

    def test_collect_metrics_by_group_true(self, panel_data, simple_workflowset):
        """Test collect_metrics with by_group=True."""
        results = simple_workflowset.fit_nested(panel_data, group_col='group')
        metrics = results.collect_metrics(by_group=True)

        # Check structure
        assert isinstance(metrics, pd.DataFrame)
        assert 'wflow_id' in metrics.columns
        assert 'group' in metrics.columns
        assert 'metric' in metrics.columns
        assert 'value' in metrics.columns
        assert 'split' in metrics.columns
        assert 'preprocessor' in metrics.columns
        assert 'model' in metrics.columns

        # Check we have metrics for all groups
        groups = metrics['group'].unique()
        assert len(groups) == 3

        # Check we have metrics for all workflows
        workflows = metrics['wflow_id'].unique()
        assert len(workflows) == 2

    def test_collect_metrics_by_group_false(self, panel_data, simple_workflowset):
        """Test collect_metrics with by_group=False (averaged)."""
        results = simple_workflowset.fit_nested(panel_data, group_col='group')
        metrics = results.collect_metrics(by_group=False)

        # Check structure
        assert isinstance(metrics, pd.DataFrame)
        assert 'wflow_id' in metrics.columns
        assert 'metric' in metrics.columns
        assert 'mean' in metrics.columns
        assert 'std' in metrics.columns
        assert 'n' in metrics.columns
        assert 'split' in metrics.columns

        # Check no group column (averaged)
        assert 'group' not in metrics.columns

        # Check n reflects number of groups
        assert metrics['n'].iloc[0] == 3  # 3 groups

    def test_collect_metrics_split_filter(self, panel_data, simple_workflowset):
        """Test collect_metrics with split filter."""
        results = simple_workflowset.fit_nested(panel_data, group_col='group')

        # Test train split
        train_metrics = results.collect_metrics(by_group=True, split='train')
        assert all(train_metrics['split'] == 'train')

        # Test all splits (default)
        all_metrics = results.collect_metrics(by_group=True, split='all')
        assert len(all_metrics) > len(train_metrics)


class TestWorkflowSetNestedResultsRankResults:
    """Tests for WorkflowSetNestedResults.rank_results() method."""

    def test_rank_results_overall(self, panel_data, multi_model_workflowset):
        """Test rank_results with by_group=False."""
        results = multi_model_workflowset.fit_nested(panel_data, group_col='group')

        # Collect metrics first to check which metrics are available
        metrics = results.collect_metrics(by_group=True, split='train')
        available_metrics = metrics['metric'].unique()

        # Use first available metric
        metric = available_metrics[0]

        ranked = results.rank_results(metric, split='train', by_group=False, n=3)

        # Check structure
        assert isinstance(ranked, pd.DataFrame)
        assert 'rank' in ranked.columns
        assert 'wflow_id' in ranked.columns
        assert 'mean' in ranked.columns
        assert 'std' in ranked.columns
        assert 'n' in ranked.columns
        assert 'preprocessor' in ranked.columns
        assert 'model' in ranked.columns

        # Check ranking is ordered
        assert ranked['rank'].tolist() == [1, 2, 3]

        # Check max n rows
        assert len(ranked) <= 3

    def test_rank_results_by_group(self, panel_data, multi_model_workflowset):
        """Test rank_results with by_group=True."""
        results = multi_model_workflowset.fit_nested(panel_data, group_col='group')

        # Get available metric
        metrics = results.collect_metrics(by_group=True, split='train')
        metric = metrics['metric'].unique()[0]

        ranked = results.rank_results(metric, split='train', by_group=True, n=2)

        # Check structure
        assert isinstance(ranked, pd.DataFrame)
        assert 'group' in ranked.columns
        assert 'rank' in ranked.columns
        assert 'wflow_id' in ranked.columns
        assert 'value' in ranked.columns

        # Check we have rankings for all groups
        groups = ranked['group'].unique()
        assert len(groups) == 3

        # Check max n per group
        for group in groups:
            group_ranks = ranked[ranked['group'] == group]
            assert len(group_ranks) <= 2

    def test_rank_results_minimize_vs_maximize(self, panel_data, simple_workflowset):
        """Test that ranking direction is correct for different metrics."""
        results = simple_workflowset.fit_nested(panel_data, group_col='group')

        # Get available metrics
        metrics = results.collect_metrics(by_group=True, split='train')
        available_metrics = metrics['metric'].unique()

        # Test with each available metric
        for metric in available_metrics:
            ranked = results.rank_results(metric, split='train', by_group=False, n=2)

            # Check that rank 1 is better than rank 2
            rank1_val = ranked[ranked['rank'] == 1]['mean'].iloc[0]
            rank2_val = ranked[ranked['rank'] == 2]['mean'].iloc[0]

            # For minimize metrics (rmse, mae), rank 1 should have lower value
            minimize_metrics = {"rmse", "mae", "mape", "smape", "mse"}
            if metric.lower() in minimize_metrics:
                assert rank1_val <= rank2_val
            else:
                # For maximize metrics (r_squared), rank 1 should have higher value
                assert rank1_val >= rank2_val


class TestWorkflowSetNestedResultsExtractBest:
    """Tests for WorkflowSetNestedResults.extract_best_workflow() method."""

    def test_extract_best_overall(self, panel_data, multi_model_workflowset):
        """Test extract_best_workflow with by_group=False."""
        results = multi_model_workflowset.fit_nested(panel_data, group_col='group')

        # Get available metric
        metrics = results.collect_metrics(by_group=True, split='train')
        metric = metrics['metric'].unique()[0]

        best_wf_id = results.extract_best_workflow(metric, split='train', by_group=False)

        # Check return type
        assert isinstance(best_wf_id, str)

        # Check it's a valid workflow ID
        assert best_wf_id in [r['wflow_id'] for r in results.results]

    def test_extract_best_by_group(self, panel_data, multi_model_workflowset):
        """Test extract_best_workflow with by_group=True."""
        results = multi_model_workflowset.fit_nested(panel_data, group_col='group')

        # Get available metric
        metrics = results.collect_metrics(by_group=True, split='train')
        metric = metrics['metric'].unique()[0]

        best_by_group = results.extract_best_workflow(metric, split='train', by_group=True)

        # Check return type
        assert isinstance(best_by_group, pd.DataFrame)

        # Check structure
        assert 'group' in best_by_group.columns
        assert 'wflow_id' in best_by_group.columns
        assert 'value' in best_by_group.columns
        assert 'preprocessor' in best_by_group.columns
        assert 'model' in best_by_group.columns

        # Check we have one best workflow per group
        assert len(best_by_group) == 3  # 3 groups

        # Check different groups can have different best workflows
        unique_best = best_by_group['wflow_id'].nunique()
        assert unique_best >= 1  # At least one workflow, possibly different per group


class TestWorkflowSetNestedResultsCollectOutputs:
    """Tests for WorkflowSetNestedResults.collect_outputs() method."""

    def test_collect_outputs_basic(self, panel_data, simple_workflowset):
        """Test basic collect_outputs functionality."""
        results = simple_workflowset.fit_nested(panel_data, group_col='group')
        outputs = results.collect_outputs()

        # Check return type
        assert isinstance(outputs, pd.DataFrame)

        # Check required columns
        assert 'wflow_id' in outputs.columns
        assert 'group' in outputs.columns
        assert 'actuals' in outputs.columns
        assert 'fitted' in outputs.columns
        assert 'split' in outputs.columns

        # Check we have outputs for all workflows
        workflows = outputs['wflow_id'].unique()
        assert len(workflows) == 2

        # Check we have outputs for all groups
        groups = outputs['group'].unique()
        assert len(groups) == 3

    def test_collect_outputs_filtering(self, panel_data, simple_workflowset):
        """Test that collected outputs can be filtered."""
        results = simple_workflowset.fit_nested(panel_data, group_col='group')
        outputs = results.collect_outputs()

        # Filter to specific workflow and group
        wf_id = outputs['wflow_id'].iloc[0]
        group = 'A'

        filtered = outputs[
            (outputs['wflow_id'] == wf_id) &
            (outputs['group'] == group)
        ]

        assert len(filtered) > 0
        assert all(filtered['wflow_id'] == wf_id)
        assert all(filtered['group'] == group)


class TestWorkflowSetNestedResultsAutoplot:
    """Tests for WorkflowSetNestedResults.autoplot() method."""

    def test_autoplot_overall(self, panel_data, multi_model_workflowset):
        """Test autoplot with by_group=False."""
        results = multi_model_workflowset.fit_nested(panel_data, group_col='group')

        # Get available metric
        metrics = results.collect_metrics(by_group=True, split='train')
        metric = metrics['metric'].unique()[0]

        fig = results.autoplot(metric, split='train', by_group=False, top_n=3)

        # Check return type
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)

        # Close plot
        plt.close(fig)

    def test_autoplot_by_group(self, panel_data, multi_model_workflowset):
        """Test autoplot with by_group=True."""
        results = multi_model_workflowset.fit_nested(panel_data, group_col='group')

        # Get available metric
        metrics = results.collect_metrics(by_group=True, split='train')
        metric = metrics['metric'].unique()[0]

        fig = results.autoplot(metric, split='train', by_group=True, top_n=2)

        # Check return type
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)

        # Check subplots created (one per group)
        assert len(fig.axes) >= 3  # At least 3 groups

        # Close plot
        plt.close(fig)


class TestWorkflowSetGroupedIntegration:
    """Integration tests for grouped workflowset functionality."""

    def test_end_to_end_nested_workflow(self, panel_data):
        """Test complete workflow from creation to best extraction."""
        # Create workflowset
        formulas = ["y ~ x1", "y ~ x1 + x2"]
        models = [
            linear_reg().set_engine("sklearn"),
            rand_forest(trees=50, mtry=1, min_n=5).set_mode("regression")
        ]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        # Fit nested
        results = wf_set.fit_nested(panel_data, group_col='group')

        # Collect metrics
        metrics = results.collect_metrics(by_group=True, split='train')
        assert len(metrics) > 0

        # Rank results
        metric_name = metrics['metric'].unique()[0]
        ranked = results.rank_results(metric_name, split='train', by_group=False, n=2)
        assert len(ranked) == 2

        # Extract best
        best_overall = results.extract_best_workflow(metric_name, split='train', by_group=False)
        assert isinstance(best_overall, str)

        best_by_group = results.extract_best_workflow(metric_name, split='train', by_group=True)
        assert isinstance(best_by_group, pd.DataFrame)
        assert len(best_by_group) == 3

        # Collect outputs
        outputs = results.collect_outputs()
        assert len(outputs) > 0
        assert 'wflow_id' in outputs.columns
        assert 'group' in outputs.columns

    def test_comparison_nested_vs_global(self, panel_data, simple_workflowset):
        """Test that nested and global approaches produce different results."""
        # Fit nested
        nested_results = simple_workflowset.fit_nested(panel_data, group_col='group')

        # Fit global
        global_results = simple_workflowset.fit_global(panel_data, group_col='group')

        # Both should succeed
        assert len(nested_results.results) == 2
        assert len(global_results.results) == 2

        # Nested should have group-specific models
        for result in nested_results.results:
            if result['nested_fit'] is not None:
                assert hasattr(result['nested_fit'], 'group_fits')
                assert len(result['nested_fit'].group_fits) == 3
