"""
Tests for parallel execution in py_tune module.

Verifies that parallel execution with n_jobs parameter produces
identical results to sequential execution.
"""

import pytest
import pandas as pd
import numpy as np
from py_tune import fit_resamples, tune_grid, tune, grid_regular
from py_workflows import workflow
from py_parsnip import linear_reg
from py_rsample import vfold_cv
from py_yardstick import metric_set, rmse, mae, r_squared


@pytest.fixture
def sample_data():
    """Create sample regression data."""
    np.random.seed(42)
    n = 200
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'x3': np.random.randn(n)
    })
    data['y'] = 2 * data['x1'] + 3 * data['x2'] - 1.5 * data['x3'] + np.random.randn(n) * 0.5
    return data


@pytest.fixture
def simple_workflow():
    """Create simple workflow for testing."""
    spec = linear_reg()
    wf = workflow().add_formula("y ~ x1 + x2 + x3").add_model(spec)
    return wf


@pytest.fixture
def tunable_workflow():
    """Create workflow with tunable parameters."""
    spec = linear_reg(penalty=tune('penalty'), mixture=tune('mixture'))
    wf = workflow().add_formula("y ~ x1 + x2 + x3").add_model(spec)
    return wf


class TestFitResamplesParallel:
    """Tests for fit_resamples with parallel execution."""

    def test_fit_resamples_sequential_baseline(self, sample_data, simple_workflow):
        """Test baseline sequential execution."""
        folds = vfold_cv(sample_data, v=3, seed=123)
        metrics = metric_set(rmse, mae)

        results = fit_resamples(simple_workflow, folds, metrics=metrics, n_jobs=None)

        assert len(results.metrics) > 0
        assert '.resample' in results.metrics.columns
        assert '.config' in results.metrics.columns
        assert 'metric' in results.metrics.columns
        assert 'value' in results.metrics.columns

        # Should have 3 folds × 2 metrics = 6 rows
        assert len(results.metrics) == 6

    def test_fit_resamples_parallel_n_jobs_2(self, sample_data, simple_workflow):
        """Test parallel execution with n_jobs=2."""
        folds = vfold_cv(sample_data, v=3, seed=123)
        metrics = metric_set(rmse, mae)

        results_seq = fit_resamples(simple_workflow, folds, metrics=metrics, n_jobs=None)
        results_par = fit_resamples(simple_workflow, folds, metrics=metrics, n_jobs=2)

        # Results should be identical
        assert len(results_seq.metrics) == len(results_par.metrics)

        # Sort both by resample and metric for comparison
        seq_sorted = results_seq.metrics.sort_values(['.resample', 'metric']).reset_index(drop=True)
        par_sorted = results_par.metrics.sort_values(['.resample', 'metric']).reset_index(drop=True)

        # Compare values (allowing for small numerical differences)
        np.testing.assert_allclose(seq_sorted['value'].values, par_sorted['value'].values, rtol=1e-10)

    def test_fit_resamples_parallel_n_jobs_minus_1(self, sample_data, simple_workflow):
        """Test parallel execution with n_jobs=-1 (all cores)."""
        folds = vfold_cv(sample_data, v=3, seed=123)
        metrics = metric_set(rmse, mae)

        results_seq = fit_resamples(simple_workflow, folds, metrics=metrics, n_jobs=None)
        results_par = fit_resamples(simple_workflow, folds, metrics=metrics, n_jobs=-1)

        # Results should be identical
        assert len(results_seq.metrics) == len(results_par.metrics)

        seq_sorted = results_seq.metrics.sort_values(['.resample', 'metric']).reset_index(drop=True)
        par_sorted = results_par.metrics.sort_values(['.resample', 'metric']).reset_index(drop=True)

        np.testing.assert_allclose(seq_sorted['value'].values, par_sorted['value'].values, rtol=1e-10)

    def test_fit_resamples_parallel_verbose(self, sample_data, simple_workflow, capsys):
        """Test verbose output with parallel execution."""
        folds = vfold_cv(sample_data, v=3, seed=123)
        metrics = metric_set(rmse, mae)

        results = fit_resamples(simple_workflow, folds, metrics=metrics, n_jobs=2, verbose=True)

        captured = capsys.readouterr()
        assert "Fitting workflow" in captured.out
        assert "complete" in captured.out

    def test_fit_resamples_parallel_with_errors(self, sample_data):
        """Test error handling in parallel execution."""
        # Create workflow that will fail on some folds (intentionally malformed)
        spec = linear_reg()
        wf = workflow().add_formula("nonexistent ~ x1").add_model(spec)

        folds = vfold_cv(sample_data, v=3, seed=123)
        metrics = metric_set(rmse, mae)

        results = fit_resamples(wf, folds, metrics=metrics, n_jobs=2)

        # Should return empty metrics due to errors
        assert len(results.metrics) == 0

    def test_fit_resamples_parallel_save_predictions(self, sample_data, simple_workflow):
        """Test saving predictions with parallel execution."""
        folds = vfold_cv(sample_data, v=3, seed=123)
        metrics = metric_set(rmse, mae)
        control = {'save_pred': True}

        results = fit_resamples(simple_workflow, folds, metrics=metrics, n_jobs=2, control=control)

        assert len(results.predictions) > 0
        assert '.resample' in results.predictions.columns
        assert '.config' in results.predictions.columns
        assert '.pred' in results.predictions.columns


class TestTuneGridParallel:
    """Tests for tune_grid with parallel execution."""

    def test_tune_grid_sequential_baseline(self, sample_data, tunable_workflow):
        """Test baseline sequential execution."""
        folds = vfold_cv(sample_data, v=3, seed=123)
        metrics = metric_set(rmse, mae)

        param_info = {
            'penalty': {'range': (0.001, 0.1), 'trans': 'log'},
            'mixture': {'range': (0, 1)}
        }

        results = tune_grid(tunable_workflow, folds, grid=2, param_info=param_info,
                          metrics=metrics, n_jobs=None)

        assert len(results.metrics) > 0
        assert len(results.grid) == 4  # 2 levels × 2 params = 4 configs

        # Should have 4 configs × 3 folds × 2 metrics = 24 rows
        assert len(results.metrics) == 24

    def test_tune_grid_parallel_n_jobs_2(self, sample_data, tunable_workflow):
        """Test parallel execution with n_jobs=2."""
        folds = vfold_cv(sample_data, v=3, seed=123)
        metrics = metric_set(rmse, mae)

        param_info = {
            'penalty': {'range': (0.001, 0.1), 'trans': 'log'},
            'mixture': {'range': (0, 1)}
        }

        results_seq = tune_grid(tunable_workflow, folds, grid=2, param_info=param_info,
                               metrics=metrics, n_jobs=None)
        results_par = tune_grid(tunable_workflow, folds, grid=2, param_info=param_info,
                               metrics=metrics, n_jobs=2)

        # Results should be identical
        assert len(results_seq.metrics) == len(results_par.metrics)
        assert len(results_seq.grid) == len(results_par.grid)

        # Sort both for comparison
        seq_sorted = results_seq.metrics.sort_values(['.config', '.resample', 'metric']).reset_index(drop=True)
        par_sorted = results_par.metrics.sort_values(['.config', '.resample', 'metric']).reset_index(drop=True)

        # Compare values
        np.testing.assert_allclose(seq_sorted['value'].values, par_sorted['value'].values, rtol=1e-10)

    def test_tune_grid_parallel_n_jobs_minus_1(self, sample_data, tunable_workflow):
        """Test parallel execution with n_jobs=-1 (all cores)."""
        folds = vfold_cv(sample_data, v=3, seed=123)
        metrics = metric_set(rmse, mae)

        param_info = {
            'penalty': {'range': (0.001, 0.1), 'trans': 'log'},
            'mixture': {'range': (0, 1)}
        }

        results_seq = tune_grid(tunable_workflow, folds, grid=2, param_info=param_info,
                               metrics=metrics, n_jobs=None)
        results_par = tune_grid(tunable_workflow, folds, grid=2, param_info=param_info,
                               metrics=metrics, n_jobs=-1)

        # Results should be identical
        assert len(results_seq.metrics) == len(results_par.metrics)

        seq_sorted = results_seq.metrics.sort_values(['.config', '.resample', 'metric']).reset_index(drop=True)
        par_sorted = results_par.metrics.sort_values(['.config', '.resample', 'metric']).reset_index(drop=True)

        np.testing.assert_allclose(seq_sorted['value'].values, par_sorted['value'].values, rtol=1e-10)

    def test_tune_grid_parallel_verbose(self, sample_data, tunable_workflow, capsys):
        """Test verbose output with parallel execution."""
        folds = vfold_cv(sample_data, v=3, seed=123)
        metrics = metric_set(rmse, mae)

        param_info = {
            'penalty': {'range': (0.001, 0.1), 'trans': 'log'},
            'mixture': {'range': (0, 1)}
        }

        results = tune_grid(tunable_workflow, folds, grid=2, param_info=param_info,
                          metrics=metrics, n_jobs=2, verbose=True)

        captured = capsys.readouterr()
        assert "Tuning grid" in captured.out
        assert "complete" in captured.out

    def test_tune_grid_parallel_large_grid(self, sample_data, tunable_workflow):
        """Test parallel execution with larger grid."""
        folds = vfold_cv(sample_data, v=3, seed=123)
        metrics = metric_set(rmse, mae)

        param_info = {
            'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
            'mixture': {'range': (0, 1)}
        }

        # 3 levels × 2 params = 9 configs
        results_seq = tune_grid(tunable_workflow, folds, grid=3, param_info=param_info,
                               metrics=metrics, n_jobs=None)
        results_par = tune_grid(tunable_workflow, folds, grid=3, param_info=param_info,
                               metrics=metrics, n_jobs=2)

        # Should have 9 configs × 3 folds × 2 metrics = 54 rows
        assert len(results_seq.metrics) == 54
        assert len(results_par.metrics) == 54

        # Results should be identical
        seq_sorted = results_seq.metrics.sort_values(['.config', '.resample', 'metric']).reset_index(drop=True)
        par_sorted = results_par.metrics.sort_values(['.config', '.resample', 'metric']).reset_index(drop=True)

        np.testing.assert_allclose(seq_sorted['value'].values, par_sorted['value'].values, rtol=1e-10)

    def test_tune_grid_parallel_select_best(self, sample_data, tunable_workflow):
        """Test that select_best works identically with parallel execution."""
        folds = vfold_cv(sample_data, v=3, seed=123)
        metrics = metric_set(rmse, mae)

        param_info = {
            'penalty': {'range': (0.001, 0.1), 'trans': 'log'},
            'mixture': {'range': (0, 1)}
        }

        results_seq = tune_grid(tunable_workflow, folds, grid=2, param_info=param_info,
                               metrics=metrics, n_jobs=None)
        results_par = tune_grid(tunable_workflow, folds, grid=2, param_info=param_info,
                               metrics=metrics, n_jobs=2)

        # Select best should give same results
        best_seq = results_seq.select_best('rmse', maximize=False)
        best_par = results_par.select_best('rmse', maximize=False)

        # Compare parameter values
        for key in best_seq.keys():
            if isinstance(best_seq[key], float):
                np.testing.assert_allclose(best_seq[key], best_par[key], rtol=1e-10)
            else:
                assert best_seq[key] == best_par[key]

    def test_tune_grid_parallel_show_best(self, sample_data, tunable_workflow):
        """Test that show_best works identically with parallel execution."""
        folds = vfold_cv(sample_data, v=3, seed=123)
        metrics = metric_set(rmse, mae)

        param_info = {
            'penalty': {'range': (0.001, 0.1), 'trans': 'log'},
            'mixture': {'range': (0, 1)}
        }

        results_seq = tune_grid(tunable_workflow, folds, grid=2, param_info=param_info,
                               metrics=metrics, n_jobs=None)
        results_par = tune_grid(tunable_workflow, folds, grid=2, param_info=param_info,
                               metrics=metrics, n_jobs=2)

        # Show best should give same results
        top_seq = results_seq.show_best('rmse', n=2, maximize=False)
        top_par = results_par.show_best('rmse', n=2, maximize=False)

        assert len(top_seq) == len(top_par)

        # Compare mean values
        np.testing.assert_allclose(top_seq['mean'].values, top_par['mean'].values, rtol=1e-10)


class TestParallelEdgeCases:
    """Tests for edge cases and error conditions in parallel execution."""

    def test_parallel_with_single_fold(self, sample_data, simple_workflow):
        """Test parallel execution with single fold (edge case)."""
        folds = vfold_cv(sample_data, v=1, seed=123)
        metrics = metric_set(rmse, mae)

        results = fit_resamples(simple_workflow, folds, metrics=metrics, n_jobs=2)

        assert len(results.metrics) == 2  # 1 fold × 2 metrics

    def test_parallel_n_jobs_greater_than_folds(self, sample_data, simple_workflow):
        """Test n_jobs > number of folds (should work fine)."""
        folds = vfold_cv(sample_data, v=2, seed=123)
        metrics = metric_set(rmse, mae)

        results = fit_resamples(simple_workflow, folds, metrics=metrics, n_jobs=4)

        assert len(results.metrics) == 4  # 2 folds × 2 metrics

    def test_parallel_n_jobs_1_same_as_none(self, sample_data, simple_workflow):
        """Test that n_jobs=1 gives same results as n_jobs=None."""
        folds = vfold_cv(sample_data, v=3, seed=123)
        metrics = metric_set(rmse, mae)

        results_none = fit_resamples(simple_workflow, folds, metrics=metrics, n_jobs=None)
        results_one = fit_resamples(simple_workflow, folds, metrics=metrics, n_jobs=1)

        # Results should be identical
        assert len(results_none.metrics) == len(results_one.metrics)

        none_sorted = results_none.metrics.sort_values(['.resample', 'metric']).reset_index(drop=True)
        one_sorted = results_one.metrics.sort_values(['.resample', 'metric']).reset_index(drop=True)

        np.testing.assert_allclose(none_sorted['value'].values, one_sorted['value'].values, rtol=1e-10)
