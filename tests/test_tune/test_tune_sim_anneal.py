"""
Tests for tune_sim_anneal - Simulated annealing optimization.
"""

import pytest
import pandas as pd
import numpy as np
from py_tune import (
    tune_sim_anneal,
    control_sim_anneal,
    SimAnnealControl,
    tune_grid,
    grid_regular,
    TuneResults
)


class TestSimAnnealControl:
    """Tests for SimAnnealControl dataclass and control_sim_anneal()."""

    def test_default_values(self):
        """SimAnnealControl has correct default values."""
        ctrl = control_sim_anneal()
        assert ctrl.initial_temp == 1.0
        assert ctrl.cooling_schedule == 'exponential'
        assert ctrl.cooling_rate == 0.95
        assert ctrl.max_iter == 50
        assert ctrl.restart_after is None
        assert ctrl.no_improve == 20
        assert ctrl.verbose is False
        assert ctrl.save_pred is False
        assert ctrl.save_workflow is False

    def test_custom_values(self):
        """SimAnnealControl accepts custom values."""
        ctrl = control_sim_anneal(
            initial_temp=2.0,
            cooling_schedule='linear',
            cooling_rate=0.05,
            max_iter=100,
            restart_after=15,
            no_improve=10,
            verbose=True
        )
        assert ctrl.initial_temp == 2.0
        assert ctrl.cooling_schedule == 'linear'
        assert ctrl.cooling_rate == 0.05
        assert ctrl.max_iter == 100
        assert ctrl.restart_after == 15
        assert ctrl.no_improve == 10
        assert ctrl.verbose is True

    def test_validation_initial_temp(self):
        """SimAnnealControl validates initial_temp > 0."""
        with pytest.raises(ValueError, match="initial_temp must be > 0"):
            control_sim_anneal(initial_temp=0)

        with pytest.raises(ValueError, match="initial_temp must be > 0"):
            control_sim_anneal(initial_temp=-1.0)

    def test_validation_cooling_schedule(self):
        """SimAnnealControl validates cooling_schedule."""
        with pytest.raises(ValueError, match="cooling_schedule must be one of"):
            control_sim_anneal(cooling_schedule='invalid')

        # Valid values should work
        ctrl1 = control_sim_anneal(cooling_schedule='exponential')
        assert ctrl1.cooling_schedule == 'exponential'

        ctrl2 = control_sim_anneal(cooling_schedule='linear')
        assert ctrl2.cooling_schedule == 'linear'

        ctrl3 = control_sim_anneal(cooling_schedule='logarithmic')
        assert ctrl3.cooling_schedule == 'logarithmic'

    def test_validation_cooling_rate(self):
        """SimAnnealControl validates cooling_rate > 0."""
        with pytest.raises(ValueError, match="cooling_rate must be > 0"):
            control_sim_anneal(cooling_rate=0)

        with pytest.raises(ValueError, match="cooling_rate must be > 0"):
            control_sim_anneal(cooling_rate=-0.1)

    def test_validation_max_iter(self):
        """SimAnnealControl validates max_iter >= 1."""
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            control_sim_anneal(max_iter=0)

    def test_validation_no_improve(self):
        """SimAnnealControl validates no_improve >= 1."""
        with pytest.raises(ValueError, match="no_improve must be >= 1"):
            control_sim_anneal(no_improve=0)

    def test_validation_restart_after(self):
        """SimAnnealControl validates restart_after >= 1 or None."""
        with pytest.raises(ValueError, match="restart_after must be >= 1 or None"):
            control_sim_anneal(restart_after=0)

        # None should work
        ctrl = control_sim_anneal(restart_after=None)
        assert ctrl.restart_after is None


class TestCoolingSchedules:
    """Tests for cooling schedule functions."""

    def test_exponential_cooling(self):
        """Exponential cooling decreases exponentially."""
        from py_tune.sim_anneal import _cool_temperature

        initial_temp = 1.0
        rate = 0.95

        # After 10 iterations
        temp_10 = _cool_temperature(1.0, 10, 'exponential', rate, initial_temp)
        expected_10 = initial_temp * (rate ** 10)
        assert abs(temp_10 - expected_10) < 1e-6

        # Temperature should decrease
        temp_0 = _cool_temperature(1.0, 0, 'exponential', rate, initial_temp)
        temp_5 = _cool_temperature(1.0, 5, 'exponential', rate, initial_temp)
        assert temp_0 > temp_5 > temp_10

    def test_linear_cooling(self):
        """Linear cooling decreases linearly."""
        from py_tune.sim_anneal import _cool_temperature

        initial_temp = 1.0
        rate = 0.05

        # After 10 iterations
        temp_10 = _cool_temperature(1.0, 10, 'linear', rate, initial_temp)
        expected_10 = initial_temp - (rate * 10)
        assert abs(temp_10 - expected_10) < 1e-6

        # Temperature should not go negative
        temp_100 = _cool_temperature(1.0, 100, 'linear', rate, initial_temp)
        assert temp_100 > 0

    def test_logarithmic_cooling(self):
        """Logarithmic cooling decreases logarithmically."""
        from py_tune.sim_anneal import _cool_temperature

        initial_temp = 1.0
        rate = 0.1

        # After 10 iterations
        temp_10 = _cool_temperature(1.0, 10, 'logarithmic', rate, initial_temp)
        expected_10 = initial_temp / (1 + rate * np.log(1 + 10))
        assert abs(temp_10 - expected_10) < 1e-6

        # Temperature should decrease slower than linear
        temp_0 = _cool_temperature(1.0, 0, 'logarithmic', rate, initial_temp)
        temp_5 = _cool_temperature(1.0, 5, 'logarithmic', rate, initial_temp)
        assert temp_0 > temp_5 > temp_10


class TestAcceptanceProbability:
    """Tests for acceptance probability calculation."""

    def test_better_solution_accepted(self):
        """Better solutions always accepted (prob=1.0)."""
        from py_tune.sim_anneal import _acceptance_probability

        # Minimization: lower is better
        prob = _acceptance_probability(current_value=1.0, new_value=0.8, temperature=1.0, maximize=False)
        assert prob == 1.0

        # Maximization: higher is better
        prob = _acceptance_probability(current_value=0.8, new_value=1.0, temperature=1.0, maximize=True)
        assert prob == 1.0

    def test_worse_solution_probability(self):
        """Worse solutions accepted with probability < 1.0."""
        from py_tune.sim_anneal import _acceptance_probability

        # Minimization: higher is worse
        prob = _acceptance_probability(current_value=0.8, new_value=1.0, temperature=1.0, maximize=False)
        assert 0 < prob < 1.0

        # Higher temperature = higher acceptance probability
        prob_high_temp = _acceptance_probability(current_value=0.8, new_value=1.0, temperature=2.0, maximize=False)
        prob_low_temp = _acceptance_probability(current_value=0.8, new_value=1.0, temperature=0.5, maximize=False)
        assert prob_high_temp > prob_low_temp

    def test_zero_temperature(self):
        """Zero temperature rejects all worse solutions."""
        from py_tune.sim_anneal import _acceptance_probability

        prob = _acceptance_probability(current_value=0.8, new_value=1.0, temperature=0.0, maximize=False)
        assert prob == 0.0


class TestNeighborGeneration:
    """Tests for neighbor generation."""

    def test_neighbor_respects_bounds(self):
        """Neighbor stays within parameter bounds."""
        from py_tune.sim_anneal import _generate_neighbor

        current = {'penalty': 0.5}
        param_info = {'penalty': {'range': (0.001, 1.0)}}

        # Generate many neighbors
        for _ in range(100):
            neighbor = _generate_neighbor(current, param_info, temperature=1.0)
            assert 0.001 <= neighbor['penalty'] <= 1.0

    def test_neighbor_log_transformation(self):
        """Neighbor generation works with log transformation."""
        from py_tune.sim_anneal import _generate_neighbor

        current = {'penalty': 0.1}
        param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}

        neighbor = _generate_neighbor(current, param_info, temperature=1.0)
        assert 0.001 <= neighbor['penalty'] <= 1.0

    def test_neighbor_integer_type(self):
        """Neighbor generation rounds integers."""
        from py_tune.sim_anneal import _generate_neighbor

        current = {'n_neighbors': 5}
        param_info = {'n_neighbors': {'range': (1, 20), 'type': 'int'}}

        neighbor = _generate_neighbor(current, param_info, temperature=1.0)
        assert isinstance(neighbor['n_neighbors'], int)
        assert 1 <= neighbor['n_neighbors'] <= 20

    def test_temperature_affects_perturbation(self):
        """Higher temperature causes larger perturbations."""
        from py_tune.sim_anneal import _generate_neighbor

        np.random.seed(42)
        current = {'penalty': 0.5}
        param_info = {'penalty': {'range': (0.001, 1.0)}}

        # High temperature
        high_temp_neighbors = [
            _generate_neighbor(current, param_info, temperature=5.0)['penalty']
            for _ in range(100)
        ]

        # Low temperature
        np.random.seed(42)
        low_temp_neighbors = [
            _generate_neighbor(current, param_info, temperature=0.1)['penalty']
            for _ in range(100)
        ]

        # High temp should have larger variance
        high_temp_var = np.var(high_temp_neighbors)
        low_temp_var = np.var(low_temp_neighbors)
        assert high_temp_var > low_temp_var


class TestTuneSimAnneal:
    """Integration tests for tune_sim_anneal()."""

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

    def create_mock_resamples(self, data, n_folds=5):
        """Create mock resamples."""
        from py_rsample import vfold_cv
        return vfold_cv(data, v=n_folds)

    def test_requires_param_info(self):
        """Error if param_info not provided."""
        data = self.create_simple_dataset()
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data, n_folds=3)

        with pytest.raises(ValueError, match="param_info is required"):
            tune_sim_anneal(wf, resamples)

    def test_basic_execution(self):
        """Basic simulated annealing completes successfully."""
        data = self.create_simple_dataset()
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data, n_folds=3)

        ctrl = control_sim_anneal(max_iter=10, verbose=False)
        param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}

        results = tune_sim_anneal(wf, resamples, param_info=param_info, control=ctrl)

        # Should return TuneResults
        assert isinstance(results, TuneResults)
        assert results.method == 'sim_anneal'
        assert not results.metrics.empty
        assert not results.grid.empty

    def test_initial_configuration(self):
        """Can specify initial configuration."""
        data = self.create_simple_dataset()
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data, n_folds=3)

        initial = {'penalty': 0.1}
        ctrl = control_sim_anneal(max_iter=5, verbose=False)
        param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}

        results = tune_sim_anneal(
            wf, resamples,
            param_info=param_info,
            initial=initial,
            control=ctrl
        )

        # First config should be initial
        first_penalty = results.grid.iloc[0]['penalty']
        assert abs(first_penalty - 0.1) < 1e-6

    def test_result_format_compatibility(self):
        """Results compatible with TuneResults methods."""
        data = self.create_simple_dataset()
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data, n_folds=3)

        ctrl = control_sim_anneal(max_iter=10, verbose=False)
        param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}

        results = tune_sim_anneal(wf, resamples, param_info=param_info, control=ctrl)

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
        resamples = self.create_mock_resamples(data, n_folds=3)

        # Very strict stopping criterion
        ctrl = control_sim_anneal(max_iter=100, no_improve=3, verbose=False)
        param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}

        results = tune_sim_anneal(wf, resamples, param_info=param_info, control=ctrl)

        # Should stop early (not evaluate all 100 configs)
        n_evaluated = len(results.grid)
        assert n_evaluated < 100

    def test_multiple_parameters(self):
        """Can optimize multiple parameters simultaneously."""
        data = self.create_simple_dataset()
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data, n_folds=3)

        ctrl = control_sim_anneal(max_iter=10, verbose=False)
        param_info = {
            'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
            'mixture': {'range': (0, 1)}
        }

        results = tune_sim_anneal(wf, resamples, param_info=param_info, control=ctrl)

        # Grid should have both parameters
        assert 'penalty' in results.grid.columns
        assert 'mixture' in results.grid.columns

    def test_verbose_logging(self, capsys):
        """Verbose mode produces log messages."""
        data = self.create_simple_dataset()
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data, n_folds=3)

        ctrl = control_sim_anneal(max_iter=5, verbose=True)
        param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}

        results = tune_sim_anneal(wf, resamples, param_info=param_info, control=ctrl)

        captured = capsys.readouterr()
        output = captured.out

        # Should see initial message
        assert "initial" in output.lower() or "starting" in output.lower()

        # Should see completion message
        assert "complete" in output.lower()

    def test_restart_functionality(self):
        """Restart from best after N iterations without improvement."""
        data = self.create_simple_dataset(n=200)
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data, n_folds=3)

        ctrl = control_sim_anneal(
            max_iter=30,
            restart_after=5,
            no_improve=15,
            verbose=False
        )
        param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}

        results = tune_sim_anneal(wf, resamples, param_info=param_info, control=ctrl)

        # Should complete without error
        assert not results.metrics.empty


class TestSimAnnealPerformance:
    """Tests to verify simulated annealing finds good solutions."""

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
        """Simulated annealing finds competitive solution."""
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

        # Simulated annealing with 25 iterations
        ctrl = control_sim_anneal(max_iter=25, verbose=False)
        sa_results = tune_sim_anneal(wf, resamples, param_info=param_info, control=ctrl)
        sa_best = sa_results.show_best('rmse', n=1, maximize=False)
        sa_best_rmse = sa_best.iloc[0]['mean']

        # Simulated annealing should find solution within 10% of grid search best
        # (may be better or worse depending on random search)
        assert sa_best_rmse < grid_best_rmse * 1.1

        print(f"\nGrid search best RMSE: {grid_best_rmse:.4f}")
        print(f"Simulated annealing best RMSE: {sa_best_rmse:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
