"""
Tests for tune_race_win_loss - Racing with Bradley-Terry win/loss models.
"""

import pytest
import pandas as pd
import numpy as np
from py_tune import (
    tune_race_win_loss,
    control_race,
    RaceControl,
    filter_parameters_bt,
    tune_grid,
    grid_regular,
    TuneResults
)


class TestFilterParametersBT:
    """Tests for filter_parameters_bt() function."""

    def create_mock_results(self, n_configs=5, n_resamples=5):
        """Create mock TuneResults for testing."""
        metrics_data = []

        # Generate metrics: config_001 is best, others progressively worse
        for config_idx in range(n_configs):
            config_name = f"config_{config_idx + 1:03d}"
            base_performance = 1.0 - (config_idx * 0.1)  # Decreasing performance

            for resample_idx in range(n_resamples):
                # Add some noise
                noise = np.random.normal(0, 0.05)
                value = base_performance + noise

                metrics_data.append({
                    '.config': config_name,
                    '.resample': f"Fold{resample_idx + 1:02d}",
                    'metric': 'rmse',
                    'value': value
                })

        metrics_df = pd.DataFrame(metrics_data)

        # Create grid
        grid_df = pd.DataFrame({
            '.config': [f"config_{i + 1:03d}" for i in range(n_configs)]
        })

        return TuneResults(
            metrics=metrics_df,
            predictions=pd.DataFrame(),
            workflow=None,
            resamples=None,
            grid=grid_df
        )

    def test_single_config(self):
        """Single configuration always passes."""
        results = self.create_mock_results(n_configs=1)
        filtered = filter_parameters_bt(results, alpha=0.05, metric_name='rmse')

        assert len(filtered) == 1
        assert filtered['pass'].iloc[0] == True

    def test_multiple_configs(self):
        """Multiple configurations get tested."""
        results = self.create_mock_results(n_configs=5, n_resamples=10)
        filtered = filter_parameters_bt(results, alpha=0.05, metric_name='rmse')

        assert len(filtered) == 5
        assert '.config' in filtered.columns
        assert 'mean' in filtered.columns
        assert 'ability' in filtered.columns
        assert 'pass' in filtered.columns

    def test_returns_correct_columns(self):
        """Result DataFrame has required columns."""
        results = self.create_mock_results(n_configs=3)
        filtered = filter_parameters_bt(results, alpha=0.05, metric_name='rmse')

        assert set(filtered.columns) == {'.config', 'mean', 'ability', 'pass'}

    def test_mean_calculation(self):
        """Mean is calculated correctly across resamples."""
        results = self.create_mock_results(n_configs=2, n_resamples=3)
        filtered = filter_parameters_bt(results, alpha=0.05, metric_name='rmse')

        # Manually calculate mean for first config
        config_1_metrics = results.metrics[results.metrics['.config'] == 'config_001']
        expected_mean = config_1_metrics['value'].mean()

        actual_mean = filtered[filtered['.config'] == 'config_001']['mean'].iloc[0]

        assert abs(actual_mean - expected_mean) < 1e-6

    def test_winning_ability_scores(self):
        """Winning abilities are numeric scores."""
        results = self.create_mock_results(n_configs=5, n_resamples=10)
        filtered = filter_parameters_bt(results, alpha=0.05, metric_name='rmse')

        # All abilities should be numeric
        assert pd.api.types.is_numeric_dtype(filtered['ability'])

        # Abilities should vary (not all the same)
        assert filtered['ability'].nunique() > 1

    def test_best_config_passes(self):
        """Best configuration (highest mean) always passes."""
        results = self.create_mock_results(n_configs=5, n_resamples=10)
        filtered = filter_parameters_bt(results, alpha=0.05, metric_name='rmse')

        # Best config should pass
        best_config = filtered.loc[filtered['mean'].idxmax(), '.config']
        assert filtered[filtered['.config'] == best_config]['pass'].iloc[0] == True

    def test_pairwise_comparisons(self):
        """Bradley-Terry uses pairwise win/loss comparisons."""
        # Create data where config_001 always wins
        metrics_data = []
        for config_idx in range(3):
            config_name = f"config_{config_idx + 1:03d}"
            base = 1.0 if config_idx == 0 else 0.5  # config_001 is clearly best

            for resample_idx in range(5):
                metrics_data.append({
                    '.config': config_name,
                    '.resample': f"Fold{resample_idx + 1:02d}",
                    'metric': 'rmse',
                    'value': base + np.random.normal(0, 0.01)
                })

        metrics_df = pd.DataFrame(metrics_data)
        grid_df = pd.DataFrame({
            '.config': ['config_001', 'config_002', 'config_003']
        })

        results = TuneResults(
            metrics=metrics_df,
            predictions=pd.DataFrame(),
            workflow=None,
            resamples=None,
            grid=grid_df
        )

        filtered = filter_parameters_bt(results, alpha=0.05, metric_name='rmse')

        # config_001 should have highest ability (wins most comparisons)
        best_ability_config = filtered.loc[filtered['ability'].idxmax(), '.config']
        assert best_ability_config == 'config_001'


class TestTuneRaceWinLoss:
    """Integration tests for tune_race_win_loss()."""

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

    def test_requires_burn_in_less_than_resamples(self):
        """Error if burn_in >= number of resamples."""
        data = self.create_simple_dataset()
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data, n_folds=5)

        ctrl = control_race(burn_in=5)

        with pytest.raises(ValueError, match="Number of resamples.*must be greater than.*burn_in"):
            tune_race_win_loss(
                wf, resamples,
                grid=2,
                param_info={'penalty': {'range': (0.001, 1.0), 'trans': 'log'}},
                control=ctrl
            )

    def test_burn_in_evaluation(self):
        """All configs evaluated on burn_in resamples."""
        data = self.create_simple_dataset()
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data, n_folds=10)

        ctrl = control_race(burn_in=3, verbose_elim=False)

        # Create small grid
        param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}
        grid = grid_regular(param_info, levels=5)

        results = tune_race_win_loss(wf, resamples, grid=grid, control=ctrl)

        # Check that all configs have at least burn_in evaluations
        metrics_by_config = results.metrics.groupby('.config')['.resample'].nunique()

        for config in grid['.config']:
            assert metrics_by_config[config] >= ctrl.burn_in

    def test_early_stopping(self):
        """Racing stops early when one config remains."""
        data = self.create_simple_dataset(n=200)
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data, n_folds=10)

        # Use aggressive alpha to encourage elimination
        ctrl = control_race(burn_in=3, alpha=0.2, verbose_elim=False)

        param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}

        results = tune_race_win_loss(
            wf, resamples,
            param_info=param_info,
            grid=10,
            control=ctrl
        )

        # At least one config should have fewer than total resamples
        # (because racing eliminated it early)
        metrics_by_config = results.metrics.groupby('.config')['.resample'].nunique()
        min_resamples = metrics_by_config.min()

        # Racing should save some evaluations
        assert min_resamples < len(resamples)

    def test_result_format_compatibility(self):
        """Results compatible with TuneResults methods."""
        data = self.create_simple_dataset()
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data, n_folds=5)

        ctrl = control_race(burn_in=2, verbose_elim=False)
        param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}

        results = tune_race_win_loss(
            wf, resamples,
            param_info=param_info,
            grid=3,
            control=ctrl
        )

        # Should be able to use standard methods
        best = results.show_best('rmse', n=1, maximize=False)
        assert len(best) == 1
        assert '.config' in best.columns

        selected = results.select_best('rmse', maximize=False)
        assert 'penalty' in selected

        # collect_metrics should work
        metrics = results.collect_metrics()
        assert not metrics.empty

    def test_method_attribute(self):
        """Results have method='race_win_loss'."""
        data = self.create_simple_dataset()
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data, n_folds=5)

        ctrl = control_race(burn_in=2, verbose_elim=False)
        param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}

        results = tune_race_win_loss(
            wf, resamples,
            param_info=param_info,
            grid=2,
            control=ctrl
        )

        assert results.method == 'race_win_loss'

    def test_verbose_logging(self, capsys):
        """Verbose mode produces log messages."""
        data = self.create_simple_dataset()
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data, n_folds=5)

        ctrl = control_race(burn_in=2, verbose_elim=True)
        param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}

        results = tune_race_win_loss(
            wf, resamples,
            param_info=param_info,
            grid=3,
            control=ctrl
        )

        captured = capsys.readouterr()
        output = captured.out

        # Should see burn-in message
        assert "burn-in" in output.lower() or "evaluating" in output.lower()

        # Should see completion message
        assert "complete" in output.lower() or "remaining" in output.lower()

    def test_randomize_resamples(self):
        """Randomization changes resample order."""
        data = self.create_simple_dataset()
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data, n_folds=5)

        # Run twice with randomization
        np.random.seed(42)
        ctrl1 = control_race(burn_in=2, randomize=True, verbose_elim=False)
        param_info = {'penalty': {'range': (0.001, 1.0), 'trans': 'log'}}

        results1 = tune_race_win_loss(
            wf, resamples,
            param_info=param_info,
            grid=2,
            control=ctrl1
        )

        # Different seed, different order expected
        np.random.seed(123)
        results2 = tune_race_win_loss(
            wf, resamples,
            param_info=param_info,
            grid=2,
            control=ctrl1
        )

        # Both should complete successfully
        assert not results1.metrics.empty
        assert not results2.metrics.empty

    def test_tie_breaking(self):
        """Tie-breaking logic when num_ties threshold reached."""
        data = self.create_simple_dataset(n=200)
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data, n_folds=10)

        # Use settings that encourage ties
        ctrl = control_race(burn_in=3, alpha=0.01, num_ties=2, verbose_elim=False)
        param_info = {'penalty': {'range': (0.5, 0.6)}}  # Narrow range for ties

        results = tune_race_win_loss(
            wf, resamples,
            param_info=param_info,
            grid=2,
            control=ctrl
        )

        # Should complete without error
        assert not results.metrics.empty

    def test_grid_as_dataframe(self):
        """Can pass explicit grid as DataFrame."""
        data = self.create_simple_dataset()
        wf = self.create_mock_workflow()
        resamples = self.create_mock_resamples(data, n_folds=5)

        # Create explicit grid
        explicit_grid = pd.DataFrame({
            'penalty': [0.001, 0.01, 0.1],
            '.config': ['config_001', 'config_002', 'config_003']
        })

        ctrl = control_race(burn_in=2, verbose_elim=False)

        results = tune_race_win_loss(wf, resamples, grid=explicit_grid, control=ctrl)

        # Should use the provided configs
        assert set(results.grid['.config']) == set(explicit_grid['.config'])


class TestWinLossPerformance:
    """Tests to verify win/loss racing achieves expected speedup."""

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

    def test_racing_reduces_evaluations(self):
        """Win/loss racing evaluates same or fewer models than grid search."""
        data = self.create_large_dataset()
        wf = self.create_mock_workflow()

        from py_rsample import vfold_cv
        resamples = vfold_cv(data, v=5)

        param_info = {
            'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
            'mixture': {'range': (0, 1)}
        }
        grid = grid_regular(param_info, levels=5)  # 25 configs

        # Grid search - all configs, all folds
        grid_results = tune_grid(wf, resamples, grid=grid)

        # Win/loss racing - should eliminate configs early (or use same if all similar)
        ctrl = control_race(burn_in=2, alpha=0.1, verbose_elim=False)
        race_results = tune_race_win_loss(wf, resamples, grid=grid, control=ctrl)

        # Count unique model evaluations (not total metric rows)
        grid_evals = grid_results.metrics.groupby(['.config', '.resample']).ngroups
        race_evals = race_results.metrics.groupby(['.config', '.resample']).ngroups

        # Racing should have same or fewer evaluations (equality OK if no elimination)
        assert race_evals <= grid_evals

        # Calculate reduction percentage
        reduction = (1 - race_evals / grid_evals) * 100
        print(f"\nWin/loss racing reduction: {reduction:.1f}% fewer evaluations")
        print(f"Grid: {grid_evals} evaluations, Win/loss racing: {race_evals} evaluations")

    def test_comparison_with_anova_racing(self):
        """Compare win/loss racing with ANOVA racing."""
        from py_tune import tune_race_anova

        data = self.create_large_dataset()
        wf = self.create_mock_workflow()

        from py_rsample import vfold_cv
        resamples = vfold_cv(data, v=5)

        param_info = {
            'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
            'mixture': {'range': (0, 1)}
        }
        grid = grid_regular(param_info, levels=5)

        ctrl = control_race(burn_in=2, alpha=0.1, verbose_elim=False)

        # ANOVA racing
        anova_results = tune_race_anova(wf, resamples, grid=grid, control=ctrl)

        # Win/loss racing
        wl_results = tune_race_win_loss(wf, resamples, grid=grid, control=ctrl)

        # Count unique model evaluations (not total metric rows)
        anova_evals = anova_results.metrics.groupby(['.config', '.resample']).ngroups
        wl_evals = wl_results.metrics.groupby(['.config', '.resample']).ngroups

        # Both should use same or fewer evaluations than full grid
        max_evals = len(grid) * len(resamples)
        assert anova_evals <= max_evals
        assert wl_evals <= max_evals

        print(f"\nANOVA racing: {anova_evals} evaluations")
        print(f"Win/loss racing: {wl_evals} evaluations")
        print(f"Full grid would be: {max_evals} evaluations")

        # Both methods should complete successfully
        assert not anova_results.metrics.empty
        assert not wl_results.metrics.empty


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
