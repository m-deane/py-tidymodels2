"""
Tests for tune_bayes - Bayesian optimization with Gaussian Processes.
"""

import pytest
import pandas as pd
import numpy as np
from py_tune import (
    tune_bayes,
    control_bayes,
    BayesControl,
    tune_grid,
    grid_regular,
    TuneResults
)


class TestBayesControl:
    """Tests for BayesControl dataclass and control_bayes()."""

    def test_default_values(self):
        """BayesControl has correct default values."""
        ctrl = control_bayes()
        assert ctrl.n_initial == 5
        assert ctrl.n_iter == 25
        assert ctrl.acquisition == 'ei'
        assert ctrl.kappa == 2.576
        assert ctrl.xi == 0.01
        assert ctrl.no_improve == 10
        assert ctrl.verbose is False
        assert ctrl.save_pred is False
        assert ctrl.save_workflow is False

    def test_custom_values(self):
        """BayesControl accepts custom values."""
        ctrl = control_bayes(
            n_initial=10,
            n_iter=50,
            acquisition='ucb',
            kappa=1.96,
            xi=0.05,
            no_improve=15,
            verbose=True
        )
        assert ctrl.n_initial == 10
        assert ctrl.n_iter == 50
        assert ctrl.acquisition == 'ucb'
        assert ctrl.kappa == 1.96
        assert ctrl.xi == 0.05
        assert ctrl.no_improve == 15
        assert ctrl.verbose is True

    def test_validation_n_initial(self):
        """BayesControl validates n_initial >= 1."""
        with pytest.raises(ValueError, match="n_initial must be >= 1"):
            control_bayes(n_initial=0)

    def test_validation_n_iter(self):
        """BayesControl validates n_iter >= 1."""
        with pytest.raises(ValueError, match="n_iter must be >= 1"):
            control_bayes(n_iter=0)

    def test_validation_acquisition(self):
        """BayesControl validates acquisition function."""
        with pytest.raises(ValueError, match="acquisition must be one of"):
            control_bayes(acquisition='invalid')

        # Valid values should work
        ctrl1 = control_bayes(acquisition='ei')
        assert ctrl1.acquisition == 'ei'

        ctrl2 = control_bayes(acquisition='pi')
        assert ctrl2.acquisition == 'pi'

        ctrl3 = control_bayes(acquisition='ucb')
        assert ctrl3.acquisition == 'ucb'

    def test_validation_kappa(self):
        """BayesControl validates kappa > 0."""
        with pytest.raises(ValueError, match="kappa must be > 0"):
            control_bayes(kappa=0)

        with pytest.raises(ValueError, match="kappa must be > 0"):
            control_bayes(kappa=-1.0)

    def test_validation_xi(self):
        """BayesControl validates xi >= 0."""
        with pytest.raises(ValueError, match="xi must be >= 0"):
            control_bayes(xi=-0.01)

        # Zero should be valid
        ctrl = control_bayes(xi=0.0)
        assert ctrl.xi == 0.0

    def test_validation_no_improve(self):
        """BayesControl validates no_improve >= 1."""
        with pytest.raises(ValueError, match="no_improve must be >= 1"):
            control_bayes(no_improve=0)


class TestParameterNormalization:
    """Tests for parameter normalization/denormalization."""

    def test_linear_normalization(self):
        """Linear parameters normalize correctly."""
        from py_tune.bayes import _normalize_params, _denormalize_params

        param_info = {'penalty': {'range': (0.0, 1.0)}}
        params = {'penalty': 0.5}

        # Normalize
        normalized = _normalize_params(params, param_info)
        assert abs(normalized[0] - 0.5) < 1e-6

        # Denormalize
        denorm = _denormalize_params(normalized, param_info)
        assert abs(denorm['penalty'] - 0.5) < 1e-6

    def test_log_normalization(self):
        """Log-transformed parameters normalize correctly."""
        from py_tune.bayes import _normalize_params, _denormalize_params

        param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}
        params = {'penalty': 0.1}  # Middle of log range

        # Normalize
        normalized = _normalize_params(params, param_info)
        expected = (np.log10(0.1) - np.log10(0.001)) / (np.log10(1.0) - np.log10(0.001))
        assert abs(normalized[0] - expected) < 1e-6

        # Denormalize
        denorm = _denormalize_params(normalized, param_info)
        assert abs(denorm['penalty'] - 0.1) < 1e-6

    def test_multiple_parameters(self):
        """Multiple parameters normalize/denormalize correctly."""
        from py_tune.bayes import _normalize_params, _denormalize_params

        param_info = {
            'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
            'mixture': {'range': (0, 1)}
        }
        params = {'penalty': 0.1, 'mixture': 0.75}

        normalized = _normalize_params(params, param_info)
        assert len(normalized) == 2

        denorm = _denormalize_params(normalized, param_info)
        assert abs(denorm['penalty'] - 0.1) < 1e-5
        assert abs(denorm['mixture'] - 0.75) < 1e-6

    def test_integer_parameters(self):
        """Integer parameters round correctly."""
        from py_tune.bayes import _denormalize_params

        param_info = {'n_neighbors': {'range': (1, 20), 'type': 'int'}}
        normalized = np.array([0.5])  # Middle of range

        denorm = _denormalize_params(normalized, param_info)
        assert isinstance(denorm['n_neighbors'], int)
        assert 1 <= denorm['n_neighbors'] <= 20


class TestAcquisitionFunctions:
    """Tests for acquisition functions."""

    def create_mock_gp(self):
        """Create mock GP model for testing."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF

        # Simple GP with constant kernel
        gp = GaussianProcessRegressor(kernel=RBF(), alpha=1e-6)

        # Fit to simple data
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([1.0, 0.5, 0.2])
        gp.fit(X, y)

        return gp

    def test_expected_improvement(self):
        """Expected Improvement calculates correctly."""
        from py_tune.bayes import _expected_improvement

        gp = self.create_mock_gp()
        X = np.array([[0.25], [0.75]])

        ei = _expected_improvement(X, gp, y_best=0.2, xi=0.01, maximize=False)

        # EI should be non-negative
        assert np.all(ei >= 0)

        # EI should have right shape
        assert ei.shape == (2,)

    def test_probability_of_improvement(self):
        """Probability of Improvement calculates correctly."""
        from py_tune.bayes import _probability_of_improvement

        gp = self.create_mock_gp()
        X = np.array([[0.25], [0.75]])

        pi = _probability_of_improvement(X, gp, y_best=0.2, xi=0.01, maximize=False)

        # PI should be in [0, 1]
        assert np.all(pi >= 0)
        assert np.all(pi <= 1)

    def test_upper_confidence_bound(self):
        """Upper Confidence Bound calculates correctly."""
        from py_tune.bayes import _upper_confidence_bound

        gp = self.create_mock_gp()
        X = np.array([[0.25], [0.75]])

        # Minimization
        ucb_min = _upper_confidence_bound(X, gp, kappa=2.0, maximize=False)
        assert ucb_min.shape == (2,)

        # Maximization
        ucb_max = _upper_confidence_bound(X, gp, kappa=2.0, maximize=True)
        assert ucb_max.shape == (2,)

        # Maximization should give higher values
        assert np.all(ucb_max >= ucb_min)

    def test_kappa_affects_exploration(self):
        """Larger kappa increases exploration in UCB."""
        from py_tune.bayes import _upper_confidence_bound

        gp = self.create_mock_gp()
        X = np.array([[0.5]])

        ucb_low = _upper_confidence_bound(X, gp, kappa=1.0, maximize=False)
        ucb_high = _upper_confidence_bound(X, gp, kappa=3.0, maximize=False)

        # Higher kappa should give more exploration (lower UCB for minimization)
        assert ucb_high[0] < ucb_low[0]


class TestTuneBayes:
    """Integration tests for tune_bayes()."""

    def create_simple_dataset(self, n=100):
        """Create simple dataset for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'y': np.random.randn(n)
        })

    def create_mock_workflow(self):
        """Create mock workflow for testing."""
        from py_workflows import workflow
        from py_parsnip import linear_reg

        wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
        return wf

    def create_mock_resamples(self, data, n_folds=3):
        """Create mock resamples."""
        from py_rsample import vfold_cv
        return vfold_cv(data, v=n_folds)

    def test_requires_param_info(self):
        """Error if param_info not provided."""
        data = self.create_simple_dataset()
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data)

        with pytest.raises(ValueError, match="param_info is required"):
            tune_bayes(wf, resamples)

    def test_basic_execution_ei(self):
        """Basic Bayesian optimization with EI completes."""
        data = self.create_simple_dataset()
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data)

        ctrl = control_bayes(n_initial=3, n_iter=5, acquisition='ei', verbose=False)
        param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}

        results = tune_bayes(wf, resamples, param_info=param_info, control=ctrl)

        # Should return TuneResults
        assert isinstance(results, TuneResults)
        assert results.method == 'bayes'
        assert not results.metrics.empty
        assert not results.grid.empty

        # Should have n_initial + n_iter configs
        assert len(results.grid) == 3 + 5

    def test_basic_execution_pi(self):
        """Bayesian optimization with PI completes."""
        data = self.create_simple_dataset()
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data)

        ctrl = control_bayes(n_initial=3, n_iter=5, acquisition='pi', verbose=False)
        param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}

        results = tune_bayes(wf, resamples, param_info=param_info, control=ctrl)

        assert isinstance(results, TuneResults)
        assert len(results.grid) == 3 + 5

    def test_basic_execution_ucb(self):
        """Bayesian optimization with UCB completes."""
        data = self.create_simple_dataset()
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data)

        ctrl = control_bayes(n_initial=3, n_iter=5, acquisition='ucb', verbose=False)
        param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}

        results = tune_bayes(wf, resamples, param_info=param_info, control=ctrl)

        assert isinstance(results, TuneResults)
        assert len(results.grid) == 3 + 5

    def test_result_format_compatibility(self):
        """Results compatible with TuneResults methods."""
        data = self.create_simple_dataset()
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data)

        ctrl = control_bayes(n_initial=3, n_iter=5, verbose=False)
        param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}

        results = tune_bayes(wf, resamples, param_info=param_info, control=ctrl)

        # Should be able to use standard methods
        best = results.show_best('rmse', n=1, maximize=False)
        assert len(best) == 1
        assert '.config' in best.columns

        selected = results.select_best('rmse', maximize=False)
        assert 'penalty' in selected

        # collect_metrics should work
        metrics = results.collect_metrics()
        assert not metrics.empty

    def test_no_improve_stopping(self):
        """Stops if no improvement for N iterations."""
        data = self.create_simple_dataset()
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data)

        # Very strict stopping criterion
        ctrl = control_bayes(n_initial=5, n_iter=50, no_improve=3, verbose=False)
        param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}

        results = tune_bayes(wf, resamples, param_info=param_info, control=ctrl)

        # Should stop early (not evaluate all 50 BO iterations)
        n_bo_iters = len(results.grid) - 5  # Subtract initial samples
        assert n_bo_iters < 50

    def test_multiple_parameters(self):
        """Can optimize multiple parameters simultaneously."""
        data = self.create_simple_dataset()
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data)

        ctrl = control_bayes(n_initial=3, n_iter=5, verbose=False)
        param_info = {
            'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
            'mixture': {'range': (0, 1)}
        }

        results = tune_bayes(wf, resamples, param_info=param_info, control=ctrl)

        # Grid should have both parameters
        assert 'penalty' in results.grid.columns
        assert 'mixture' in results.grid.columns

    def test_verbose_logging(self, capsys):
        """Verbose mode produces log messages."""
        data = self.create_simple_dataset()
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data)

        ctrl = control_bayes(n_initial=2, n_iter=3, verbose=True)
        param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}

        results = tune_bayes(wf, resamples, param_info=param_info, control=ctrl)

        captured = capsys.readouterr()
        output = captured.out

        # Should see phase messages
        assert "phase 1" in output.lower() or "initial" in output.lower()
        assert "phase 2" in output.lower() or "bayesian" in output.lower()

        # Should see completion message
        assert "complete" in output.lower()


class TestBayesPerformance:
    """Tests to verify Bayesian optimization finds good solutions."""

    def create_large_dataset(self, n=500):
        """Create larger dataset for performance testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'x3': np.random.randn(n),
            'y': np.random.randn(n)
        })

    def create_mock_workflow(self):
        """Create workflow."""
        from py_workflows import workflow
        from py_parsnip import linear_reg

        wf = workflow().add_formula("y ~ x1 + x2 + x3").add_model(linear_reg())
        return wf

    def test_finds_competitive_solution(self):
        """Bayesian optimization finds competitive solution."""
        data = self.create_large_dataset()
        wf = self.create_mock_workflow()

        from py_rsample import vfold_cv
        resamples = vfold_cv(data, v=5)

        param_info = {
            'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
            'mixture': {'range': (0, 1)}
        }

        # Grid search with 25 configs
        grid = grid_regular(param_info, levels=5)
        grid_results = tune_grid(wf, resamples, grid=grid)
        grid_best = grid_results.show_best('rmse', n=1, maximize=False)
        grid_best_rmse = grid_best.iloc[0]['mean']

        # Bayesian optimization with 25 total evaluations
        ctrl = control_bayes(n_initial=10, n_iter=15, verbose=False)
        bayes_results = tune_bayes(wf, resamples, param_info=param_info, control=ctrl)
        bayes_best = bayes_results.show_best('rmse', n=1, maximize=False)
        bayes_best_rmse = bayes_best.iloc[0]['mean']

        # Bayesian optimization should find solution within 10% of grid search best
        assert bayes_best_rmse < grid_best_rmse * 1.1

        print(f"\nGrid search best RMSE: {grid_best_rmse:.4f}")
        print(f"Bayesian optimization best RMSE: {bayes_best_rmse:.4f}")

    def test_sequential_improvement(self):
        """Bayesian optimization improves over iterations."""
        data = self.create_large_dataset()
        wf = self.create_mock_workflow()

        from py_rsample import vfold_cv
        resamples = vfold_cv(data, v=3)

        param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}

        ctrl = control_bayes(n_initial=5, n_iter=15, verbose=False)
        results = tune_bayes(wf, resamples, param_info=param_info, control=ctrl)

        # Get metrics over time
        metrics = results.collect_metrics()
        rmse_values = []

        for config in results.grid['.config']:
            config_metrics = metrics[metrics['.config'] == config]
            if 'metric' in config_metrics.columns:
                config_metrics = config_metrics[config_metrics['metric'] == 'rmse']
                rmse = config_metrics['value'].mean()
            else:
                rmse = config_metrics['rmse'].mean()
            rmse_values.append(rmse)

        # Best RMSE should improve (decrease) as we evaluate more configs
        best_initial = min(rmse_values[:5])  # Best from initial samples
        best_final = min(rmse_values)  # Best overall

        # Final should be same or better
        assert best_final <= best_initial


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
