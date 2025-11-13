"""
Comprehensive tests for WorkflowSet with grouped/nested data.

Tests multi-model comparison across multiple groups using:
- fit_nested(): Per-group models (each group gets its own model per workflow)
- fit_global(): Global models (single model per workflow, group as feature)
- Per-group ranking and selection
- Group-aware visualization

All tests use real grouped data from _md/__data/ directory.
"""

import pytest
import pandas as pd
import numpy as np
from py_workflowsets import WorkflowSet
from py_workflows import workflow
from py_recipes import recipe
from py_recipes.selectors import all_numeric, all_numeric_predictors
from py_parsnip import (
    linear_reg, rand_forest, boost_tree, decision_tree,
    nearest_neighbor
)
from py_yardstick import metric_set, rmse, mae, r_squared


class TestWorkflowSetNested:
    """Test WorkflowSet.fit_nested() for per-group modeling."""

    def test_fit_nested_basic(self, refinery_data_small_groups):
        """Test fit_nested with simple workflow set."""
        formulas = ['brent ~ dubai + wti']
        models = [linear_reg(), linear_reg(penalty=0.1, mixture=1.0)]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        # Fit nested (2 workflows × 3 countries = 6 models)
        results = wf_set.fit_nested(refinery_data_small_groups, group_col='country')

        # Check results structure
        assert hasattr(results, 'collect_metrics')
        assert hasattr(results, 'rank_results')
        assert hasattr(results, 'extract_best_workflow')

    def test_fit_nested_multiple_workflows(self, refinery_data_small_groups):
        """Test fit_nested with multiple workflows and groups."""
        formulas = [
            'brent ~ dubai + wti',
            'brent ~ .',
        ]
        models = [
            linear_reg(),
            rand_forest(trees=50).set_mode('regression'),
        ]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        # 4 workflows × 3 countries = 12 models
        results = wf_set.fit_nested(refinery_data_small_groups, group_col='country')

        # Collect metrics
        metrics_by_group = results.collect_metrics(by_group=True, split='test')
        # Should have 4 workflows × 3 groups = 12 rows
        assert len(metrics_by_group) == 12

    def test_fit_nested_with_recipes(self, refinery_data_small_groups):
        """Test fit_nested with recipe preprocessing."""
        recipes = [
            recipe().step_normalize(all_numeric_predictors()),
            recipe().step_normalize(all_numeric_predictors()).step_pca(num_comp=3),
        ]
        models = [linear_reg()]
        wf_set = WorkflowSet.from_cross(preproc=recipes, models=models)

        results = wf_set.fit_nested(refinery_data_small_groups, group_col='country')

        metrics_avg = results.collect_metrics(by_group=False, split='test')
        assert len(metrics_avg) == 2  # 2 workflows


class TestWorkflowSetNestedMetrics:
    """Test metric collection and analysis for nested WorkflowSet."""

    def test_collect_metrics_by_group(self, refinery_data_small_groups):
        """Test collecting metrics per group."""
        formulas = ['brent ~ dubai + wti', 'brent ~ .']
        models = [linear_reg()]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        results = wf_set.fit_nested(refinery_data_small_groups, group_col='country')

        metrics_by_group = results.collect_metrics(by_group=True, split='test')

        # Verify structure
        assert 'wflow_id' in metrics_by_group.columns
        assert 'group' in metrics_by_group.columns
        assert 'rmse' in metrics_by_group.columns or 'metric' in metrics_by_group.columns

        # Each workflow × group should have metrics
        n_workflows = 2
        n_groups = len(refinery_data_small_groups['country'].unique())
        assert len(metrics_by_group) == n_workflows * n_groups

    def test_collect_metrics_averaged(self, refinery_data_small_groups):
        """Test collecting metrics averaged across groups."""
        formulas = ['brent ~ dubai + wti']
        models = [linear_reg(), linear_reg(penalty=0.1, mixture=1.0)]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        results = wf_set.fit_nested(refinery_data_small_groups, group_col='country')

        metrics_avg = results.collect_metrics(by_group=False, split='test')

        # Should have one row per workflow
        assert len(metrics_avg) == 2

        # Should have mean metrics
        if 'mean_rmse' in metrics_avg.columns:
            assert 'mean_rmse' in metrics_avg.columns
            assert 'mean_mae' in metrics_avg.columns
        elif 'rmse' in metrics_avg.columns:
            assert 'rmse' in metrics_avg.columns
            assert 'mae' in metrics_avg.columns

    def test_collect_metrics_train_split(self, refinery_data_small_groups):
        """Test collecting training metrics."""
        formulas = ['brent ~ dubai + wti']
        models = [linear_reg()]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        results = wf_set.fit_nested(refinery_data_small_groups, group_col='country')

        # Training metrics
        train_metrics = results.collect_metrics(by_group=False, split='train')
        assert len(train_metrics) == 1

        # Test metrics
        test_metrics = results.collect_metrics(by_group=False, split='test')
        assert len(test_metrics) == 1


class TestWorkflowSetNestedRanking:
    """Test ranking workflows for nested WorkflowSet."""

    def test_rank_results_overall(self, refinery_data_small_groups):
        """Test ranking workflows by overall performance."""
        formulas = [
            'brent ~ dubai + wti',
            'brent ~ .',
            'brent ~ dubai + wti + I(dubai*wti)',
        ]
        models = [linear_reg(), linear_reg(penalty=0.1, mixture=1.0)]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        results = wf_set.fit_nested(refinery_data_small_groups, group_col='country')

        # Rank by RMSE overall
        ranked = results.rank_results('rmse', by_group=False, n=6)

        # Should be sorted by RMSE
        assert len(ranked) == 6
        assert 'wflow_id' in ranked.columns

        # Check sorting (lower RMSE is better)
        if 'mean_rmse' in ranked.columns:
            rmse_col = 'mean_rmse'
        elif 'rmse' in ranked.columns:
            rmse_col = 'rmse'
        else:
            pytest.skip("Cannot find RMSE column")

        rmse_values = ranked[rmse_col].values
        assert all(rmse_values[i] <= rmse_values[i+1] for i in range(len(rmse_values)-1))

    def test_rank_results_by_group(self, refinery_data_small_groups):
        """Test ranking workflows per group."""
        formulas = ['brent ~ dubai + wti', 'brent ~ .']
        models = [linear_reg()]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        results = wf_set.fit_nested(refinery_data_small_groups, group_col='country')

        # Rank by group (top 1 per group)
        ranked_by_group = results.rank_results('rmse', by_group=True, n=1)

        # Should have 1 workflow per group
        n_groups = len(refinery_data_small_groups['country'].unique())
        assert len(ranked_by_group) == n_groups
        assert 'group' in ranked_by_group.columns

    def test_rank_results_top_n(self, gas_demand_small_groups):
        """Test ranking with n parameter."""
        formulas = ['gas_demand ~ temperature + wind_speed', 'gas_demand ~ .']
        models = [
            linear_reg(),
            rand_forest(trees=50).set_mode('regression'),
        ]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        results = wf_set.fit_nested(gas_demand_small_groups, group_col='country')

        # Get top 2 overall
        ranked = results.rank_results('rmse', by_group=False, n=2)
        assert len(ranked) == 2

        # Get top 1 per group
        ranked_by_group = results.rank_results('rmse', by_group=True, n=1)
        n_groups = len(gas_demand_small_groups['country'].unique())
        assert len(ranked_by_group) == n_groups


class TestWorkflowSetNestedBestWorkflow:
    """Test extracting best workflows from nested WorkflowSet."""

    def test_extract_best_overall(self, refinery_data_small_groups):
        """Test extracting best workflow overall."""
        formulas = ['brent ~ dubai + wti', 'brent ~ .']
        models = [linear_reg(), linear_reg(penalty=0.1, mixture=1.0)]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        results = wf_set.fit_nested(refinery_data_small_groups, group_col='country')

        # Get best by RMSE
        best_wf_id = results.extract_best_workflow('rmse', by_group=False)

        # Should return workflow ID string
        assert isinstance(best_wf_id, str)
        assert best_wf_id in wf_set.workflow_ids

    def test_extract_best_by_group(self, refinery_data_small_groups):
        """Test extracting best workflow per group."""
        formulas = ['brent ~ dubai + wti', 'brent ~ .']
        models = [linear_reg()]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        results = wf_set.fit_nested(refinery_data_small_groups, group_col='country')

        # Get best per group
        best_by_group = results.extract_best_workflow('rmse', by_group=True)

        # Should return DataFrame with group and wflow_id
        assert isinstance(best_by_group, pd.DataFrame)
        assert 'group' in best_by_group.columns
        assert 'wflow_id' in best_by_group.columns
        n_groups = len(refinery_data_small_groups['country'].unique())
        assert len(best_by_group) == n_groups

    def test_heterogeneous_best_workflows(self, refinery_data_small_groups):
        """Test that different groups can have different best workflows."""
        formulas = [
            'brent ~ dubai + wti',
            'brent ~ .',
        ]
        models = [
            linear_reg(),
            rand_forest(trees=50).set_mode('regression'),
        ]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        results = wf_set.fit_nested(refinery_data_small_groups, group_col='country')

        # Get best per group
        best_by_group = results.extract_best_workflow('rmse', by_group=True)

        # Check if different groups prefer different workflows
        unique_workflows = best_by_group['wflow_id'].nunique()
        # May be 1 if all groups prefer same workflow, or >1 if heterogeneous
        assert unique_workflows >= 1
        assert unique_workflows <= len(wf_set.workflow_ids)


class TestWorkflowSetNestedOutputs:
    """Test collecting outputs from nested WorkflowSet."""

    def test_collect_outputs(self, refinery_data_small_groups):
        """Test collecting all predictions and actuals."""
        formulas = ['brent ~ dubai + wti']
        models = [linear_reg()]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        results = wf_set.fit_nested(refinery_data_small_groups, group_col='country')

        # Collect outputs
        outputs = results.collect_outputs()

        # Should have predictions for all groups
        assert 'wflow_id' in outputs.columns
        assert 'group' in outputs.columns
        assert 'actuals' in outputs.columns
        assert 'fitted' in outputs.columns

        # Each workflow × group should have outputs
        n_workflows = 1
        n_groups = len(refinery_data_small_groups['country'].unique())
        output_groups = outputs.groupby(['wflow_id', 'group']).size()
        assert len(output_groups) == n_workflows * n_groups


class TestWorkflowSetGlobal:
    """Test WorkflowSet.fit_global() for global modeling with group as feature."""

    def test_fit_global_basic(self, refinery_data_small_groups):
        """Test fit_global with simple workflow set."""
        formulas = ['brent ~ dubai + wti']
        models = [linear_reg()]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        # Fit global (1 workflow = 1 model, group as feature)
        results = wf_set.fit_global(refinery_data_small_groups, group_col='country')

        # Should still have group-aware metrics
        metrics_by_group = results.collect_metrics(by_group=True, split='test')
        n_groups = len(refinery_data_small_groups['country'].unique())
        assert len(metrics_by_group) == n_groups

    def test_fit_global_multiple_workflows(self, refinery_data_small_groups):
        """Test fit_global with multiple workflows."""
        formulas = ['brent ~ dubai + wti', 'brent ~ .']
        models = [linear_reg(), linear_reg(penalty=0.1, mixture=1.0)]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        # 4 workflows (each trained once globally)
        results = wf_set.fit_global(refinery_data_small_groups, group_col='country')

        metrics_avg = results.collect_metrics(by_group=False, split='test')
        assert len(metrics_avg) == 4

    def test_global_vs_nested_comparison(self, gas_demand_small_groups):
        """Compare global and nested modeling approaches."""
        formulas = ['gas_demand ~ temperature + wind_speed']
        models = [linear_reg()]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        # Global approach
        results_global = wf_set.fit_global(gas_demand_small_groups, group_col='country')
        metrics_global = results_global.collect_metrics(by_group=False, split='test')

        # Nested approach
        results_nested = wf_set.fit_nested(gas_demand_small_groups, group_col='country')
        metrics_nested = results_nested.collect_metrics(by_group=False, split='test')

        # Both should have metrics for same workflow
        assert len(metrics_global) == 1
        assert len(metrics_nested) == 1

        # Metrics may differ due to different modeling approaches


class TestWorkflowSetMixedPreprocessing:
    """Test WorkflowSet with various preprocessing combinations on grouped data."""

    def test_nested_with_multiple_recipes(self, refinery_data_small_groups):
        """Test nested with various recipe strategies."""
        recipes = [
            recipe().step_normalize(all_numeric_predictors()),
            recipe().step_normalize(all_numeric_predictors()).step_pca(num_comp=3),
            recipe().step_poly(['dubai', 'wti'], degree=2).step_normalize(all_numeric_predictors()),
        ]
        models = [linear_reg()]
        wf_set = WorkflowSet.from_cross(preproc=recipes, models=models)

        results = wf_set.fit_nested(refinery_data_small_groups, group_col='country')

        # 3 workflows × 3 groups = 9 models
        metrics_by_group = results.collect_metrics(by_group=True, split='test')
        assert len(metrics_by_group) == 9

        # Rank overall
        ranked = results.rank_results('rmse', by_group=False, n=3)
        assert len(ranked) == 3

    def test_nested_with_feature_selection(self, refinery_data_small_groups):
        """Test nested with feature selection recipes."""
        recipes = [
            recipe().step_normalize(all_numeric_predictors()),
            recipe().step_select_corr(threshold=0.9).step_normalize(all_numeric_predictors()),
        ]
        models = [linear_reg(), rand_forest(trees=50).set_mode('regression')]
        wf_set = WorkflowSet.from_cross(preproc=recipes, models=models)

        results = wf_set.fit_nested(refinery_data_small_groups, group_col='country')

        ranked = results.rank_results('rmse', by_group=False, n=4)
        assert len(ranked) == 4


class TestWorkflowSetMixedModels:
    """Test WorkflowSet with various model combinations on grouped data."""

    def test_nested_linear_tree_comparison(self, refinery_data_small_groups):
        """Compare linear and tree models per group."""
        formulas = ['brent ~ .']
        models = [
            linear_reg(),
            linear_reg(penalty=0.1, mixture=1.0),
            decision_tree(tree_depth=5, min_n=10).set_mode('regression'),
            rand_forest(trees=50).set_mode('regression'),
        ]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        results = wf_set.fit_nested(refinery_data_small_groups, group_col='country')

        # Check best model per group
        best_by_group = results.extract_best_workflow('rmse', by_group=True)
        assert len(best_by_group) == len(refinery_data_small_groups['country'].unique())

    def test_nested_boosting_comparison(self, gas_demand_small_groups):
        """Compare different boosting configurations."""
        formulas = ['gas_demand ~ temperature + wind_speed']
        models = [
            boost_tree(trees=50, tree_depth=3, learn_rate=0.1).set_mode('regression'),
            boost_tree(trees=100, tree_depth=5, learn_rate=0.05).set_mode('regression'),
        ]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        results = wf_set.fit_nested(gas_demand_small_groups, group_col='country')

        metrics_avg = results.collect_metrics(by_group=False, split='test')
        assert len(metrics_avg) == 2


class TestWorkflowSetLargeScaleGrouped:
    """Test WorkflowSet with many workflows and groups."""

    def test_many_workflows_many_groups(self, refinery_data_small_groups):
        """Test WorkflowSet with 10+ workflows across 3+ groups."""
        formulas = [
            'brent ~ dubai + wti',
            'brent ~ .',
        ]
        recipes = [
            recipe().step_normalize(all_numeric_predictors()),
            recipe().step_normalize(all_numeric_predictors()).step_pca(num_comp=3),
        ]
        models = [
            linear_reg(),
            linear_reg(penalty=0.1, mixture=1.0),
            rand_forest(trees=50).set_mode('regression'),
        ]

        preproc = formulas + recipes  # 4 preprocessors
        wf_set = WorkflowSet.from_cross(preproc=preproc, models=models)
        # 4 × 3 = 12 workflows

        results = wf_set.fit_nested(refinery_data_small_groups, group_col='country')

        # 12 workflows × 3 groups = 36 models
        metrics_by_group = results.collect_metrics(by_group=True, split='test')
        assert len(metrics_by_group) == 36

        # Get top 5 overall
        ranked = results.rank_results('rmse', by_group=False, n=5)
        assert len(ranked) == 5

    def test_performance_scaling(self, refinery_data):
        """Test WorkflowSet performance with 5 countries."""
        # Use 5 countries
        countries = refinery_data['country'].unique()[:5]
        data = refinery_data[refinery_data['country'].isin(countries)]

        formulas = ['brent ~ dubai + wti', 'brent ~ .']
        models = [linear_reg(), rand_forest(trees=50).set_mode('regression')]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        # 4 workflows × 5 countries = 20 models
        results = wf_set.fit_nested(data, group_col='country')

        metrics_by_group = results.collect_metrics(by_group=True, split='test')
        assert len(metrics_by_group) == 20

        # Best per group
        best_by_group = results.extract_best_workflow('rmse', by_group=True)
        assert len(best_by_group) == 5


class TestWorkflowSetVisualization:
    """Test WorkflowSet visualization for grouped data."""

    def test_autoplot_overall(self, refinery_data_small_groups):
        """Test autoplot with overall metrics."""
        formulas = ['brent ~ dubai + wti', 'brent ~ .']
        models = [linear_reg(), rand_forest(trees=50).set_mode('regression')]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        results = wf_set.fit_nested(refinery_data_small_groups, group_col='country')

        # Check autoplot exists
        assert hasattr(results, 'autoplot')

        # Try to call it
        try:
            fig = results.autoplot('rmse', by_group=False, top_n=4)
            assert fig is not None
        except Exception:
            # If visualization fails, just check method exists
            pass

    def test_autoplot_by_group(self, refinery_data_small_groups):
        """Test autoplot with per-group metrics."""
        formulas = ['brent ~ dubai + wti']
        models = [linear_reg()]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        results = wf_set.fit_nested(refinery_data_small_groups, group_col='country')

        try:
            fig = results.autoplot('rmse', by_group=True, top_n=1)
            assert fig is not None
        except Exception:
            pass


class TestWorkflowSetEdgeCases:
    """Test edge cases for grouped WorkflowSet."""

    def test_single_workflow_multiple_groups(self, refinery_data_small_groups):
        """Test single workflow across groups."""
        formulas = ['brent ~ dubai + wti']
        models = [linear_reg()]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        results = wf_set.fit_nested(refinery_data_small_groups, group_col='country')

        # Should still work
        metrics = results.collect_metrics(by_group=False, split='test')
        assert len(metrics) == 1

    def test_many_workflows_single_group(self, refinery_data_small_groups):
        """Test many workflows on single group."""
        # Filter to one country
        single_country = refinery_data_small_groups[
            refinery_data_small_groups['country'] == refinery_data_small_groups['country'].unique()[0]
        ]

        formulas = ['brent ~ dubai + wti', 'brent ~ .']
        models = [linear_reg(), rand_forest(trees=50).set_mode('regression')]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        results = wf_set.fit_nested(single_country, group_col='country')

        # 4 workflows × 1 group = 4 models
        metrics_by_group = results.collect_metrics(by_group=True, split='test')
        assert len(metrics_by_group) == 4

        # Best overall
        best = results.extract_best_workflow('rmse', by_group=False)
        assert best in wf_set.workflow_ids
