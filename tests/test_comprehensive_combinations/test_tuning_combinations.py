"""
Comprehensive tests for hyperparameter tuning combinations.

Tests various tuning scenarios with:
- tune() parameter marking
- grid_regular() and grid_random() grid creation
- tune_grid() and fit_resamples() execution
- TuneResults analysis (show_best, select_best, select_by_one_std_err)
- finalize_workflow() with best parameters

All tests use real data from _md/__data/ directory.
"""

import pytest
import pandas as pd
import numpy as np
from py_workflows import workflow
from py_recipes import recipe
from py_recipes.selectors import all_numeric_predictors
from py_parsnip import linear_reg, rand_forest, boost_tree, decision_tree, nearest_neighbor
from py_rsample import vfold_cv, time_series_cv
from py_tune import tune, grid_regular, grid_random, tune_grid, fit_resamples, finalize_workflow
from py_yardstick import metric_set, rmse, mae, r_squared


class TestGridCreation:
    """Test creating parameter grids."""

    def test_grid_regular_single_param(self):
        """Test creating regular grid for single parameter."""
        grid = grid_regular({
            'penalty': {'range': (0.001, 1.0), 'trans': 'log'}
        }, levels=5)

        assert isinstance(grid, pd.DataFrame)
        assert 'penalty' in grid.columns
        assert len(grid) == 5

        # Check log spacing
        penalties = grid['penalty'].values
        assert all(penalties[i] < penalties[i+1] for i in range(len(penalties)-1))

    def test_grid_regular_multiple_params(self):
        """Test creating regular grid for multiple parameters."""
        grid = grid_regular({
            'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
            'mixture': {'range': (0, 1)}
        }, levels=5)

        assert 'penalty' in grid.columns
        assert 'mixture' in grid.columns
        assert len(grid) == 25  # 5 × 5

    def test_grid_random_single_param(self):
        """Test creating random grid for single parameter."""
        grid = grid_random({
            'penalty': {'range': (0.001, 1.0), 'trans': 'log'}
        }, size=10)

        assert isinstance(grid, pd.DataFrame)
        assert 'penalty' in grid.columns
        assert len(grid) == 10

    def test_grid_random_multiple_params(self):
        """Test creating random grid for multiple parameters."""
        grid = grid_random({
            'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
            'mixture': {'range': (0, 1)}
        }, size=20)

        assert 'penalty' in grid.columns
        assert 'mixture' in grid.columns
        assert len(grid) == 20


class TestLinearRegTuning:
    """Test tuning linear regression models."""

    def test_tune_lasso_penalty(self, refinery_data_ungrouped, metric_set_basic):
        """Test tuning Lasso penalty parameter."""
        spec = linear_reg(penalty=tune(), mixture=1.0)

        grid = grid_regular({
            'penalty': {'range': (0.001, 1.0), 'trans': 'log'}
        }, levels=5)

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(spec)
        folds = vfold_cv(refinery_data_ungrouped, v=3)

        results = tune_grid(wf, resamples=folds, grid=grid, metrics=metric_set_basic)

        # Check results structure
        assert hasattr(results, 'show_best')
        assert hasattr(results, 'select_best')

        # Get best
        best = results.select_best('rmse', maximize=False)
        assert 'penalty' in best.columns or 'penalty' in best.index

    def test_tune_elasticnet_both_params(self, refinery_data_ungrouped, metric_set_basic):
        """Test tuning both penalty and mixture for ElasticNet."""
        spec = linear_reg(penalty=tune(), mixture=tune())

        grid = grid_regular({
            'penalty': {'range': (0.01, 0.5), 'trans': 'log'},
            'mixture': {'range': (0, 1)}
        }, levels=3)

        wf = workflow().add_formula('brent ~ .').add_model(spec)
        folds = vfold_cv(refinery_data_ungrouped, v=3)

        results = tune_grid(wf, resamples=folds, grid=grid, metrics=metric_set_basic)

        # 3 × 3 = 9 parameter combinations
        # Check we have results
        best = results.select_best('rmse', maximize=False)
        assert best is not None

    def test_finalize_workflow_with_best(self, refinery_data_ungrouped, train_test_split_80_20):
        """Test finalizing workflow with best parameters."""
        train, test = train_test_split_80_20(refinery_data_ungrouped)

        spec = linear_reg(penalty=tune(), mixture=1.0)
        grid = grid_regular({
            'penalty': {'range': (0.001, 1.0), 'trans': 'log'}
        }, levels=5)

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(spec)
        folds = vfold_cv(train, v=3)

        results = tune_grid(wf, resamples=folds, grid=grid, metrics=metric_set(rmse, mae))
        best = results.select_best('rmse', maximize=False)

        # Finalize and fit
        final_wf = finalize_workflow(wf, best)
        fit = final_wf.fit(train)

        # Predictions should work
        preds = fit.predict(test)
        assert len(preds) == len(test)


class TestTreeModelTuning:
    """Test tuning tree-based models."""

    def test_tune_decision_tree_depth(self, refinery_data_ungrouped, metric_set_basic):
        """Test tuning decision tree depth."""
        spec = decision_tree(tree_depth=tune(), min_n=10).set_mode('regression')

        grid = grid_regular({
            'tree_depth': {'range': (3, 10)}
        }, levels=4)

        wf = workflow().add_formula('brent ~ .').add_model(spec)
        folds = vfold_cv(refinery_data_ungrouped, v=3)

        results = tune_grid(wf, resamples=folds, grid=grid, metrics=metric_set_basic)

        best = results.select_best('rmse', maximize=False)
        assert best is not None

    def test_tune_random_forest_multiple_params(self, gas_demand_ungrouped, metric_set_basic):
        """Test tuning multiple random forest parameters."""
        spec = rand_forest(trees=50, mtry=tune(), min_n=tune()).set_mode('regression')

        grid = grid_regular({
            'mtry': {'range': (2, 5)},
            'min_n': {'range': (5, 20)}
        }, levels=3)

        wf = workflow().add_formula('gas_demand ~ .').add_model(spec)
        folds = vfold_cv(gas_demand_ungrouped, v=3)

        results = tune_grid(wf, resamples=folds, grid=grid, metrics=metric_set_basic)

        # 3 × 3 = 9 combinations
        best = results.select_best('rmse', maximize=False)
        assert best is not None

    def test_tune_boost_tree_learning_rate(self, refinery_data_ungrouped, metric_set_basic):
        """Test tuning boosting learning rate and tree depth."""
        spec = boost_tree(
            trees=50,
            tree_depth=tune(),
            learn_rate=tune()
        ).set_mode('regression')

        grid = grid_regular({
            'tree_depth': {'range': (3, 6)},
            'learn_rate': {'range': (0.01, 0.3), 'trans': 'log'}
        }, levels=3)

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(spec)
        folds = vfold_cv(refinery_data_ungrouped, v=3)

        results = tune_grid(wf, resamples=folds, grid=grid, metrics=metric_set_basic)

        best = results.select_best('rmse', maximize=False)
        assert best is not None


class TestKNNTuning:
    """Test tuning k-NN models."""

    def test_tune_knn_neighbors(self, gas_demand_ungrouped, metric_set_basic):
        """Test tuning number of neighbors."""
        spec = nearest_neighbor(neighbors=tune()).set_mode('regression')

        grid = grid_regular({
            'neighbors': {'range': (3, 15)}
        }, levels=5)

        rec = recipe().step_normalize(all_numeric_predictors())
        wf = workflow().add_recipe(rec).add_model(spec)
        folds = vfold_cv(gas_demand_ungrouped, v=3)

        results = tune_grid(wf, resamples=folds, grid=grid, metrics=metric_set_basic)

        best = results.select_best('rmse', maximize=False)
        assert best is not None


class TestTuneResultsAnalysis:
    """Test analyzing TuneResults."""

    def test_show_best_default(self, refinery_data_ungrouped, metric_set_basic):
        """Test show_best with default parameters."""
        spec = linear_reg(penalty=tune(), mixture=1.0)
        grid = grid_regular({
            'penalty': {'range': (0.001, 1.0), 'trans': 'log'}
        }, levels=5)

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(spec)
        folds = vfold_cv(refinery_data_ungrouped, v=3)

        results = tune_grid(wf, resamples=folds, grid=grid, metrics=metric_set_basic)

        # Show best 5
        best_5 = results.show_best('rmse', n=5, maximize=False)

        assert isinstance(best_5, pd.DataFrame)
        assert len(best_5) == 5
        assert 'penalty' in best_5.columns or 'penalty' in best_5.index

    def test_show_best_top_n(self, refinery_data_ungrouped, metric_set_basic):
        """Test show_best with n parameter."""
        spec = linear_reg(penalty=tune(), mixture=tune())
        grid = grid_regular({
            'penalty': {'range': (0.01, 0.5), 'trans': 'log'},
            'mixture': {'range': (0, 1)}
        }, levels=3)

        wf = workflow().add_formula('brent ~ .').add_model(spec)
        folds = vfold_cv(refinery_data_ungrouped, v=3)

        results = tune_grid(wf, resamples=folds, grid=grid, metrics=metric_set_basic)

        # Show top 3
        top_3 = results.show_best('rmse', n=3, maximize=False)
        assert len(top_3) == 3

    def test_select_best_rmse(self, gas_demand_ungrouped, metric_set_basic):
        """Test select_best for RMSE."""
        spec = nearest_neighbor(neighbors=tune()).set_mode('regression')
        grid = grid_regular({
            'neighbors': {'range': (3, 15)}
        }, levels=5)

        rec = recipe().step_normalize(all_numeric_predictors())
        wf = workflow().add_recipe(rec).add_model(spec)
        folds = vfold_cv(gas_demand_ungrouped, v=3)

        results = tune_grid(wf, resamples=folds, grid=grid, metrics=metric_set_basic)

        best = results.select_best('rmse', maximize=False)

        # Should be single row
        assert isinstance(best, (pd.DataFrame, pd.Series))
        if isinstance(best, pd.DataFrame):
            assert len(best) == 1

    def test_select_best_r_squared(self, refinery_data_ungrouped, metric_set_basic):
        """Test select_best for R² (maximize)."""
        spec = linear_reg(penalty=tune(), mixture=1.0)
        grid = grid_regular({
            'penalty': {'range': (0.001, 1.0), 'trans': 'log'}
        }, levels=5)

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(spec)
        folds = vfold_cv(refinery_data_ungrouped, v=3)

        results = tune_grid(wf, resamples=folds, grid=grid, metrics=metric_set_basic)

        best = results.select_best('r_squared', maximize=True)
        assert best is not None

    def test_select_by_one_std_err(self, refinery_data_ungrouped, metric_set_basic):
        """Test select_by_one_std_err for parsimony."""
        spec = linear_reg(penalty=tune(), mixture=1.0)
        grid = grid_regular({
            'penalty': {'range': (0.001, 1.0), 'trans': 'log'}
        }, levels=7)

        wf = workflow().add_formula('brent ~ .').add_model(spec)
        folds = vfold_cv(refinery_data_ungrouped, v=3)

        results = tune_grid(wf, resamples=folds, grid=grid, metrics=metric_set_basic)

        # Select by one std error rule
        best_1se = results.select_by_one_std_err('rmse', 'penalty', maximize=False, larger_is_simpler=True)

        assert best_1se is not None
        # Should select larger penalty (simpler model) within 1 std error


class TestTuningWithRecipes:
    """Test tuning with recipe preprocessing."""

    def test_tune_with_normalization(self, refinery_data_ungrouped, metric_set_basic):
        """Test tuning with normalization recipe."""
        spec = linear_reg(penalty=tune(), mixture=1.0)
        grid = grid_regular({
            'penalty': {'range': (0.001, 1.0), 'trans': 'log'}
        }, levels=5)

        rec = recipe().step_normalize(all_numeric_predictors())
        wf = workflow().add_recipe(rec).add_model(spec)
        folds = vfold_cv(refinery_data_ungrouped, v=3)

        results = tune_grid(wf, resamples=folds, grid=grid, metrics=metric_set_basic)

        best = results.select_best('rmse', maximize=False)
        assert best is not None

    def test_tune_with_pca(self, refinery_data_ungrouped, metric_set_basic):
        """Test tuning with PCA preprocessing."""
        spec = linear_reg(penalty=tune(), mixture=1.0)
        grid = grid_regular({
            'penalty': {'range': (0.001, 1.0), 'trans': 'log'}
        }, levels=5)

        rec = (recipe()
               .step_normalize(all_numeric_predictors())
               .step_pca(num_comp=5))
        wf = workflow().add_recipe(rec).add_model(spec)
        folds = vfold_cv(refinery_data_ungrouped, v=3)

        results = tune_grid(wf, resamples=folds, grid=grid, metrics=metric_set_basic)

        best = results.select_best('rmse', maximize=False)
        assert best is not None


class TestTuningWithTimeSeriesCV:
    """Test tuning with time series cross-validation."""

    def test_tune_with_time_series_cv(self, refinery_data_ungrouped, metric_set_basic):
        """Test tuning with time series CV splits."""
        spec = linear_reg(penalty=tune(), mixture=1.0)
        grid = grid_regular({
            'penalty': {'range': (0.001, 1.0), 'trans': 'log'}
        }, levels=5)

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(spec)

        # Time series CV
        cv = time_series_cv(
            refinery_data_ungrouped,
            date_column='date',
            initial='2 years',
            assess='6 months',
            skip='3 months',
            cumulative=False
        )

        results = tune_grid(wf, resamples=cv, grid=grid, metrics=metric_set_basic)

        best = results.select_best('rmse', maximize=False)
        assert best is not None

    def test_tune_boosting_with_ts_cv(self, gas_demand_ungrouped, metric_set_basic):
        """Test tuning boosting with time series CV."""
        spec = boost_tree(
            trees=50,
            tree_depth=tune(),
            learn_rate=tune()
        ).set_mode('regression')

        grid = grid_regular({
            'tree_depth': {'range': (3, 6)},
            'learn_rate': {'range': (0.01, 0.3), 'trans': 'log'}
        }, levels=3)

        wf = workflow().add_formula('gas_demand ~ .').add_model(spec)

        cv = time_series_cv(
            gas_demand_ungrouped,
            date_column='date',
            initial='1 year',
            assess='3 months',
            skip='1 month',
            cumulative=False
        )

        results = tune_grid(wf, resamples=cv, grid=grid, metrics=metric_set_basic)

        best = results.select_best('rmse', maximize=False)
        assert best is not None


class TestFitResamples:
    """Test fit_resamples (evaluate without tuning)."""

    def test_fit_resamples_basic(self, refinery_data_ungrouped, metric_set_basic):
        """Test fit_resamples with fixed parameters."""
        spec = linear_reg(penalty=0.1, mixture=1.0)
        wf = workflow().add_formula('brent ~ dubai + wti').add_model(spec)

        folds = vfold_cv(refinery_data_ungrouped, v=3)
        results = fit_resamples(wf, resamples=folds, metrics=metric_set_basic)

        # Should return results without tuning
        assert hasattr(results, 'collect_metrics') or hasattr(results, 'show_best')

    def test_fit_resamples_random_forest(self, gas_demand_ungrouped, metric_set_basic):
        """Test fit_resamples with random forest."""
        spec = rand_forest(trees=50, mtry=3, min_n=5).set_mode('regression')
        wf = workflow().add_formula('gas_demand ~ .').add_model(spec)

        folds = vfold_cv(gas_demand_ungrouped, v=3)
        results = fit_resamples(wf, resamples=folds, metrics=metric_set_basic)

        # Should have results
        assert results is not None


class TestRandomGridTuning:
    """Test tuning with random grids."""

    def test_random_grid_lasso(self, refinery_data_ungrouped, metric_set_basic):
        """Test tuning with random grid search."""
        spec = linear_reg(penalty=tune(), mixture=1.0)

        grid = grid_random({
            'penalty': {'range': (0.001, 1.0), 'trans': 'log'}
        }, size=10)

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(spec)
        folds = vfold_cv(refinery_data_ungrouped, v=3)

        results = tune_grid(wf, resamples=folds, grid=grid, metrics=metric_set_basic)

        # Should evaluate 10 random parameter values
        best = results.select_best('rmse', maximize=False)
        assert best is not None

    def test_random_grid_multiple_params(self, gas_demand_ungrouped, metric_set_basic):
        """Test random grid with multiple parameters."""
        spec = boost_tree(
            trees=50,
            tree_depth=tune(),
            learn_rate=tune(),
            min_n=tune()
        ).set_mode('regression')

        grid = grid_random({
            'tree_depth': {'range': (3, 10)},
            'learn_rate': {'range': (0.001, 0.3), 'trans': 'log'},
            'min_n': {'range': (5, 50)}
        }, size=15)

        wf = workflow().add_formula('gas_demand ~ .').add_model(spec)
        folds = vfold_cv(gas_demand_ungrouped, v=3)

        results = tune_grid(wf, resamples=folds, grid=grid, metrics=metric_set_basic)

        # 15 random combinations
        best = results.select_best('rmse', maximize=False)
        assert best is not None


class TestTuningWorkflow:
    """Test complete tuning workflow from start to finish."""

    def test_complete_tuning_workflow(self, refinery_data_ungrouped, train_test_split_80_20):
        """Test complete workflow: tune → select → finalize → fit → predict."""
        train, test = train_test_split_80_20(refinery_data_ungrouped)

        # Step 1: Define model with tune()
        spec = linear_reg(penalty=tune(), mixture=tune())

        # Step 2: Create grid
        grid = grid_regular({
            'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
            'mixture': {'range': (0, 1)}
        }, levels=3)

        # Step 3: Create workflow
        wf = workflow().add_formula('brent ~ .').add_model(spec)

        # Step 4: Tune with CV
        folds = vfold_cv(train, v=3)
        results = tune_grid(wf, resamples=folds, grid=grid, metrics=metric_set(rmse, mae))

        # Step 5: Select best
        best = results.select_best('rmse', maximize=False)

        # Step 6: Finalize workflow
        final_wf = finalize_workflow(wf, best)

        # Step 7: Fit on full training data
        fit = final_wf.fit(train)

        # Step 8: Predict on test
        preds = fit.predict(test)
        assert len(preds) == len(test)

        # Step 9: Evaluate
        eval_fit = fit.evaluate(test)
        outputs, coeffs, stats = eval_fit.extract_outputs()

        test_rmse = stats[(stats['split'] == 'test') & (stats['metric'] == 'rmse')]['value'].iloc[0]
        assert test_rmse > 0

    def test_tuning_with_complex_recipe(self, gas_demand_ungrouped, train_test_split_80_20):
        """Test tuning with complex preprocessing pipeline."""
        train, test = train_test_split_80_20(gas_demand_ungrouped)

        # Complex recipe
        rec = (recipe()
               .step_impute_median(all_numeric())
               .step_normalize(all_numeric_predictors())
               .step_pca(num_comp=2))

        # Tune model
        spec = rand_forest(trees=50, mtry=tune(), min_n=tune()).set_mode('regression')
        grid = grid_regular({
            'mtry': {'range': (1, 3)},
            'min_n': {'range': (5, 20)}
        }, levels=3)

        wf = workflow().add_recipe(rec).add_model(spec)
        folds = vfold_cv(train, v=3)

        results = tune_grid(wf, resamples=folds, grid=grid, metrics=metric_set(rmse, mae))

        best = results.select_best('rmse', maximize=False)
        final_wf = finalize_workflow(wf, best)
        fit = final_wf.fit(train)

        preds = fit.predict(test)
        assert len(preds) == len(test)
