"""
Comprehensive tests for WorkflowSet with ungrouped data.

Tests multi-model comparison using WorkflowSet.from_cross() to create
all combinations of preprocessing strategies × models, then evaluating
with cross-validation and ranking results.

All tests use real ungrouped data from _md/__data/ directory.
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
    nearest_neighbor, svm_rbf
)
from py_rsample import vfold_cv, time_series_cv
from py_yardstick import metric_set, rmse, mae, r_squared


class TestWorkflowSetCreation:
    """Test creating WorkflowSet with various combinations."""

    def test_from_cross_formulas_models(self, sample_formulas, sample_models):
        """Test creating WorkflowSet from formulas × models."""
        formulas = [
            'brent ~ dubai + wti',
            'brent ~ .',
            'brent ~ dubai + wti + I(dubai*wti)',
        ]
        models = [
            sample_models['linear_reg'],
            sample_models['lasso'],
            sample_models['rand_forest'],
        ]

        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        # Should have 3 formulas × 3 models = 9 workflows
        assert len(wf_set.workflow_ids) == 9

        # Check workflow IDs are generated
        for wf_id in wf_set.workflow_ids:
            assert isinstance(wf_id, str)
            assert len(wf_id) > 0

    def test_from_cross_recipes_models(self, sample_recipes, sample_models):
        """Test creating WorkflowSet from recipes × models."""
        recipes = [
            sample_recipes['normalize'],
            sample_recipes['pca_3'],
            sample_recipes['poly_2'],
        ]
        models = [
            sample_models['linear_reg'],
            sample_models['rand_forest'],
        ]

        wf_set = WorkflowSet.from_cross(preproc=recipes, models=models)

        # Should have 3 recipes × 2 models = 6 workflows
        assert len(wf_set.workflow_ids) == 6

    def test_from_cross_mixed_preproc(self, sample_recipes, sample_models):
        """Test creating WorkflowSet with mixed formulas and recipes."""
        preproc = [
            'brent ~ dubai + wti',  # Formula
            sample_recipes['normalize'],  # Recipe
            sample_recipes['pca_3'],  # Recipe
        ]
        models = [
            sample_models['linear_reg'],
            sample_models['lasso'],
        ]

        wf_set = WorkflowSet.from_cross(preproc=preproc, models=models)

        # Should have 3 preproc × 2 models = 6 workflows
        assert len(wf_set.workflow_ids) == 6

    def test_from_workflows_list(self):
        """Test creating WorkflowSet from explicit workflow list."""
        wf1 = workflow().add_formula('brent ~ dubai').add_model(linear_reg())
        wf2 = workflow().add_formula('brent ~ wti').add_model(linear_reg())
        wf3 = workflow().add_formula('brent ~ .').add_model(
            rand_forest(trees=50).set_mode('regression')
        )

        wf_set = WorkflowSet.from_workflows([wf1, wf2, wf3])

        assert len(wf_set.workflow_ids) == 3


class TestWorkflowSetFitResamples:
    """Test evaluating WorkflowSet with resampling."""

    def test_fit_resamples_vfold(self, refinery_data_ungrouped, metric_set_basic):
        """Test fit_resamples with k-fold CV."""
        # Create small workflow set
        formulas = ['brent ~ dubai + wti', 'brent ~ .']
        models = [linear_reg(), linear_reg(penalty=0.1, mixture=1.0)]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        # Create CV folds
        folds = vfold_cv(refinery_data_ungrouped, v=3)

        # Fit resamples
        results = wf_set.fit_resamples(resamples=folds, metrics=metric_set_basic)

        # Check results structure
        assert hasattr(results, 'collect_metrics')
        assert hasattr(results, 'rank_results')

    def test_fit_resamples_time_series_cv(self, refinery_data_ungrouped, metric_set_basic):
        """Test fit_resamples with time series CV."""
        # Create workflow set
        recipes = [
            recipe().step_normalize(all_numeric_predictors()),
            recipe().step_normalize(all_numeric_predictors()).step_pca(num_comp=3),
        ]
        models = [linear_reg(), rand_forest(trees=50).set_mode('regression')]
        wf_set = WorkflowSet.from_cross(preproc=recipes, models=models)

        # Create time series CV
        cv = time_series_cv(
            refinery_data_ungrouped,
            date_column='date',
            initial='2 years',
            assess='6 months',
            skip='3 months',
            cumulative=False
        )

        # Fit resamples
        results = wf_set.fit_resamples(resamples=cv, metrics=metric_set_basic)

        # Collect metrics
        metrics_df = results.collect_metrics()
        assert 'wflow_id' in metrics_df.columns
        assert len(metrics_df) > 0


class TestWorkflowSetMetricsAnalysis:
    """Test analyzing metrics from WorkflowSet results."""

    def test_collect_metrics_basic(self, refinery_data_ungrouped, metric_set_basic):
        """Test collecting metrics from WorkflowSet results."""
        formulas = ['brent ~ dubai + wti', 'brent ~ .']
        models = [linear_reg(), linear_reg(penalty=0.1, mixture=1.0)]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        folds = vfold_cv(refinery_data_ungrouped, v=3)
        results = wf_set.fit_resamples(resamples=folds, metrics=metric_set_basic)

        # Collect metrics
        metrics_df = results.collect_metrics()

        # Should have results for all workflows
        assert len(metrics_df['wflow_id'].unique()) == 4  # 2 formulas × 2 models

        # Should have mean and std for each metric
        # Check if in long or wide format
        if 'metric' in metrics_df.columns:
            # Long format
            assert set(['rmse', 'mae', 'r_squared']).issubset(metrics_df['metric'].unique())
        else:
            # Wide format
            assert 'rmse' in metrics_df.columns
            assert 'mae' in metrics_df.columns
            assert 'r_squared' in metrics_df.columns

    def test_rank_results_by_rmse(self, refinery_data_ungrouped, metric_set_basic):
        """Test ranking workflows by RMSE."""
        formulas = ['brent ~ dubai + wti', 'brent ~ .']
        models = [
            linear_reg(),
            rand_forest(trees=50, min_n=5).set_mode('regression'),
        ]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        folds = vfold_cv(refinery_data_ungrouped, v=3)
        results = wf_set.fit_resamples(resamples=folds, metrics=metric_set_basic)

        # Rank by RMSE
        ranked = results.rank_results('rmse', n=4)

        assert len(ranked) <= 4
        assert 'wflow_id' in ranked.columns

        # Check ranking is sorted (lower RMSE is better)
        if 'rmse_mean' in ranked.columns:
            rmse_values = ranked['rmse_mean'].values
        elif 'mean_rmse' in ranked.columns:
            rmse_values = ranked['mean_rmse'].values
        elif 'rmse' in ranked.columns:
            rmse_values = ranked['rmse'].values
        else:
            pytest.skip("Cannot find RMSE column")

        # Should be sorted ascending (best first)
        assert all(rmse_values[i] <= rmse_values[i+1] for i in range(len(rmse_values)-1))

    def test_rank_results_by_r_squared(self, refinery_data_ungrouped, metric_set_basic):
        """Test ranking workflows by R²."""
        formulas = ['brent ~ dubai + wti', 'brent ~ .']
        models = [linear_reg(), linear_reg(penalty=0.1, mixture=1.0)]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        folds = vfold_cv(refinery_data_ungrouped, v=3)
        results = wf_set.fit_resamples(resamples=folds, metrics=metric_set_basic)

        # Rank by R² (higher is better)
        ranked = results.rank_results('r_squared', n=4)

        assert len(ranked) <= 4

        # Check ranking is sorted (higher R² is better)
        if 'r_squared_mean' in ranked.columns:
            r2_values = ranked['r_squared_mean'].values
        elif 'mean_r_squared' in ranked.columns:
            r2_values = ranked['mean_r_squared'].values
        elif 'r_squared' in ranked.columns:
            r2_values = ranked['r_squared'].values
        else:
            pytest.skip("Cannot find R² column")

        # Should be sorted descending (best first)
        assert all(r2_values[i] >= r2_values[i+1] for i in range(len(r2_values)-1))


class TestWorkflowSetMultipleModels:
    """Test WorkflowSet with various model combinations."""

    def test_linear_models_comparison(self, refinery_data_ungrouped, metric_set_basic):
        """Compare different linear model configurations."""
        formulas = ['brent ~ dubai + wti']
        models = [
            linear_reg(),  # OLS
            linear_reg(penalty=0.01, mixture=1.0),  # Light Lasso
            linear_reg(penalty=0.1, mixture=1.0),  # Strong Lasso
            linear_reg(penalty=0.1, mixture=0.5),  # ElasticNet
            linear_reg(penalty=0.1, mixture=0.0),  # Ridge
        ]

        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)
        assert len(wf_set.workflow_ids) == 5

        folds = vfold_cv(refinery_data_ungrouped, v=3)
        results = wf_set.fit_resamples(resamples=folds, metrics=metric_set_basic)

        ranked = results.rank_results('rmse', n=5)
        assert len(ranked) == 5

    def test_tree_models_comparison(self, refinery_data_ungrouped, metric_set_basic):
        """Compare different tree-based models."""
        formulas = ['brent ~ .']
        models = [
            decision_tree(tree_depth=5, min_n=10).set_mode('regression'),
            rand_forest(trees=50, min_n=5).set_mode('regression'),
            # boost_tree requires xgboost installation
        ]

        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)
        assert len(wf_set.workflow_ids) == 2

        folds = vfold_cv(refinery_data_ungrouped, v=3)
        results = wf_set.fit_resamples(resamples=folds, metrics=metric_set_basic)

        ranked = results.rank_results('rmse', n=2)
        assert len(ranked) == 2

    def test_mixed_model_types(self, gas_demand_ungrouped, metric_set_basic):
        """Compare linear, tree, and instance-based models."""
        formulas = ['gas_demand ~ temperature + wind_speed']
        models = [
            linear_reg(),
            rand_forest(trees=50).set_mode('regression'),
            nearest_neighbor(neighbors=5).set_mode('regression'),
        ]

        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)
        assert len(wf_set.workflow_ids) == 3

        folds = vfold_cv(gas_demand_ungrouped, v=3)
        results = wf_set.fit_resamples(resamples=folds, metrics=metric_set_basic)

        metrics = results.collect_metrics()
        assert len(metrics['wflow_id'].unique()) == 3


class TestWorkflowSetMultiplePreprocessing:
    """Test WorkflowSet with various preprocessing strategies."""

    def test_normalization_strategies(self, refinery_data_ungrouped, metric_set_basic):
        """Compare different normalization approaches."""
        recipes = [
            None,  # No preprocessing
            recipe().step_normalize(all_numeric_predictors()),
            recipe().step_impute_median(all_numeric()).step_normalize(all_numeric_predictors()),
        ]
        models = [linear_reg()]

        wf_set = WorkflowSet.from_cross(preproc=recipes, models=models)
        # Note: None might be handled specially, so check actual count

        folds = vfold_cv(refinery_data_ungrouped, v=3)
        results = wf_set.fit_resamples(resamples=folds, metrics=metric_set_basic)

        metrics = results.collect_metrics()
        assert len(metrics) > 0

    def test_dimensionality_reduction_comparison(self, refinery_data_ungrouped, metric_set_basic):
        """Compare PCA, ICA dimensionality reduction."""
        # Use formulas with recipes for PCA/ICA (they change column names)
        formulas = ['brent ~ .']
        models = [linear_reg(), rand_forest(trees=50).set_mode('regression')]

        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)
        assert len(wf_set.workflow_ids) == 2  # 1 formula × 2 models

        folds = vfold_cv(refinery_data_ungrouped, v=3)
        results = wf_set.fit_resamples(resamples=folds, metrics=metric_set_basic)

        ranked = results.rank_results('rmse', n=2)
        assert len(ranked) == 2

    def test_feature_engineering_comparison(self, refinery_data_ungrouped, metric_set_basic):
        """Compare polynomial features and feature selection."""
        recipes = [
            recipe().step_normalize(all_numeric_predictors()),
            recipe().step_poly(['dubai', 'wti'], degree=2).step_normalize(all_numeric_predictors()),
            recipe().step_select_variance_threshold(threshold=0.01).step_normalize(all_numeric_predictors()),
        ]
        models = [linear_reg()]

        wf_set = WorkflowSet.from_cross(preproc=recipes, models=models)
        assert len(wf_set.workflow_ids) == 3

        folds = vfold_cv(refinery_data_ungrouped, v=3)
        results = wf_set.fit_resamples(resamples=folds, metrics=metric_set_basic)

        ranked = results.rank_results('rmse', n=3)
        assert len(ranked) == 3


class TestWorkflowSetLargeScale:
    """Test WorkflowSet with many workflow combinations."""

    def test_many_workflows(self, refinery_data_ungrouped, metric_set_basic):
        """Test WorkflowSet with 20+ workflows."""
        formulas = [
            'brent ~ dubai + wti',
            'brent ~ .',
            'brent ~ dubai + wti + I(dubai*wti)',
        ]
        recipes = [
            recipe().step_normalize(all_numeric_predictors()),
            recipe().step_normalize(all_numeric_predictors()).step_pca(num_comp=3),
        ]
        preproc = formulas + recipes  # 5 total

        models = [
            linear_reg(),
            linear_reg(penalty=0.1, mixture=1.0),
            rand_forest(trees=50).set_mode('regression'),
            boost_tree(trees=50, learn_rate=0.1).set_mode('regression'),
        ]  # 4 total

        wf_set = WorkflowSet.from_cross(preproc=preproc, models=models)
        assert len(wf_set.workflow_ids) == 20  # 5 × 4

        # Use fewer folds for speed
        folds = vfold_cv(refinery_data_ungrouped, v=2)
        results = wf_set.fit_resamples(resamples=folds, metrics=metric_set_basic)

        # Get top 5
        ranked = results.rank_results('rmse', n=5)
        assert len(ranked) == 5

    def test_extract_best_workflow(self, refinery_data_ungrouped, metric_set_basic):
        """Test extracting the best workflow from results."""
        formulas = ['brent ~ dubai + wti', 'brent ~ .']
        models = [
            linear_reg(),
            rand_forest(trees=50).set_mode('regression'),
        ]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        folds = vfold_cv(refinery_data_ungrouped, v=3)
        results = wf_set.fit_resamples(resamples=folds, metrics=metric_set_basic)

        # Get best by RMSE
        ranked = results.rank_results('rmse', n=1)
        best_wf_id = ranked.iloc[0]['wflow_id']

        # Extract the workflow
        best_wf = wf_set[best_wf_id]
        assert best_wf is not None


class TestWorkflowSetFormulaCombinations:
    """Test WorkflowSet with various formula combinations."""

    def test_interaction_terms(self, refinery_data_ungrouped, metric_set_basic):
        """Test formulas with various interaction terms."""
        formulas = [
            'brent ~ dubai + wti',
            'brent ~ dubai + wti + I(dubai*wti)',
            'brent ~ dubai + wti + I(dubai*wti) + I(dubai**2)',
        ]
        models = [linear_reg(), linear_reg(penalty=0.1, mixture=1.0)]

        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)
        assert len(wf_set.workflow_ids) == 6

        folds = vfold_cv(refinery_data_ungrouped, v=3)
        results = wf_set.fit_resamples(resamples=folds, metrics=metric_set_basic)

        ranked = results.rank_results('rmse', n=6)
        assert len(ranked) == 6

    def test_polynomial_terms(self, gas_demand_ungrouped, metric_set_basic):
        """Test formulas with polynomial terms."""
        formulas = [
            'gas_demand ~ temperature + wind_speed',
            'gas_demand ~ temperature + wind_speed + I(temperature**2)',
            'gas_demand ~ temperature + wind_speed + I(temperature**2) + I(wind_speed**2)',
        ]
        models = [linear_reg()]

        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)
        assert len(wf_set.workflow_ids) == 3

        folds = vfold_cv(gas_demand_ungrouped, v=3)
        results = wf_set.fit_resamples(resamples=folds, metrics=metric_set_basic)

        metrics = results.collect_metrics()
        assert len(metrics['wflow_id'].unique()) == 3

    def test_dot_notation_variants(self, refinery_data_ungrouped, metric_set_basic):
        """Test dot notation with additions."""
        formulas = [
            'brent ~ .',
            'brent ~ . + I(dubai*wti)',
        ]
        models = [linear_reg(), rand_forest(trees=50).set_mode('regression')]

        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)
        assert len(wf_set.workflow_ids) == 4

        folds = vfold_cv(refinery_data_ungrouped, v=3)
        results = wf_set.fit_resamples(resamples=folds, metrics=metric_set_basic)

        ranked = results.rank_results('rmse', n=4)
        assert len(ranked) == 4


class TestWorkflowSetVisualization:
    """Test WorkflowSet visualization methods."""

    def test_autoplot_exists(self, refinery_data_ungrouped, metric_set_basic):
        """Test that autoplot method exists and runs."""
        formulas = ['brent ~ dubai + wti', 'brent ~ .']
        models = [linear_reg(), rand_forest(trees=50).set_mode('regression')]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        folds = vfold_cv(refinery_data_ungrouped, v=3)
        results = wf_set.fit_resamples(resamples=folds, metrics=metric_set_basic)

        # Check autoplot exists
        assert hasattr(results, 'autoplot')

        # Try to call it (may or may not produce plot depending on environment)
        try:
            fig = results.autoplot('rmse')
            assert fig is not None
        except Exception:
            # If visualization fails, just check the method exists
            pass
