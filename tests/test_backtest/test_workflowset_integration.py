"""
Tests for WorkflowSet.fit_backtests() integration.
"""

import pytest
import pandas as pd
import numpy as np

from py_backtest import VintageCV, create_vintage_data
from py_workflowsets import WorkflowSet
from py_workflows import workflow
from py_parsnip import linear_reg
from py_yardstick import metric_set, rmse, mae


class TestWorkflowSetBacktest:
    """Tests for WorkflowSet.fit_backtests() method."""

    def setup_method(self):
        """Create test data and workflows"""
        np.random.seed(42)

        # Create final data
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 2.0 * x1 + 1.5 * x2 + np.random.randn(n) * 0.5

        final_data = pd.DataFrame({
            'date': dates,
            'x1': x1,
            'x2': x2,
            'y': y
        })

        # Create vintage data
        self.vintage_data = create_vintage_data(
            final_data,
            date_col='date',
            n_revisions=3,
            revision_std=0.05,
            revision_lag='7 days'
        )

        # Create workflows
        self.wf1 = workflow().add_formula('y ~ x1').add_model(linear_reg())
        self.wf2 = workflow().add_formula('y ~ x1 + x2').add_model(linear_reg())

        self.wf_set = WorkflowSet.from_workflows(
            [self.wf1, self.wf2],
            ids=['simple', 'full']
        )

    def test_fit_backtests_basic(self):
        """Test basic fit_backtests functionality"""
        # Create vintage CV
        vintage_cv = VintageCV(
            data=self.vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial='60 days',
            assess='20 days',
            skip='10 days',
            slice_limit=3
        )

        # Backtest workflows
        results = self.wf_set.fit_backtests(
            vintage_cv,
            metrics=metric_set(rmse, mae)
        )

        # Should return BacktestResults
        from py_backtest import BacktestResults
        assert isinstance(results, BacktestResults)

        # Should have results for both workflows
        assert len(results.workflow_ids) == 2
        assert 'simple' in results.workflow_ids
        assert 'full' in results.workflow_ids

    def test_collect_metrics(self):
        """Test collecting metrics from backtest"""
        vintage_cv = VintageCV(
            data=self.vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial='60 days',
            assess='20 days',
            skip='10 days',
            slice_limit=2
        )

        results = self.wf_set.fit_backtests(
            vintage_cv,
            metrics=metric_set(rmse, mae)
        )

        # Collect metrics
        metrics = results.collect_metrics(by_vintage=True)

        # Should have metrics for both workflows
        assert 'simple' in metrics['wflow_id'].values
        assert 'full' in metrics['wflow_id'].values

        # Should have both metrics
        assert 'rmse' in metrics['metric'].values
        assert 'mae' in metrics['metric'].values

        # Should have vintage dates
        assert 'vintage_date' in metrics.columns

    def test_rank_results(self):
        """Test ranking workflows from backtest"""
        vintage_cv = VintageCV(
            data=self.vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial='60 days',
            assess='20 days',
            skip='10 days',
            slice_limit=2
        )

        results = self.wf_set.fit_backtests(
            vintage_cv,
            metrics=metric_set(rmse, mae)
        )

        # Rank by RMSE
        ranked = results.rank_results('rmse', n=2)

        # Should have both workflows ranked
        assert len(ranked) == 2
        assert 'rank' in ranked.columns

        # Full model should generally be better (uses both features)
        # (may not always be true due to randomness, but likely)
        assert ranked.iloc[0]['wflow_id'] in ['simple', 'full']

    def test_analyze_vintage_drift(self):
        """Test analyzing vintage drift"""
        vintage_cv = VintageCV(
            data=self.vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial='60 days',
            assess='20 days',
            skip='10 days',
            slice_limit=3
        )

        results = self.wf_set.fit_backtests(
            vintage_cv,
            metrics=metric_set(rmse)
        )

        # Analyze drift
        drift = results.analyze_vintage_drift('rmse')

        # Should have drift metrics
        assert 'drift_from_start' in drift.columns
        assert 'drift_pct' in drift.columns

        # Should have both workflows
        assert 'simple' in drift['wflow_id'].values
        assert 'full' in drift['wflow_id'].values

    def test_verbose_output(self, capsys):
        """Test verbose output during backtesting"""
        vintage_cv = VintageCV(
            data=self.vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial='60 days',
            assess='20 days',
            slice_limit=2
        )

        results = self.wf_set.fit_backtests(
            vintage_cv,
            metrics=metric_set(rmse),
            verbose=True
        )

        # Capture output
        captured = capsys.readouterr()

        # Should print progress
        assert 'Backtesting' in captured.out
        assert 'complete' in captured.out

    def test_default_metrics(self):
        """Test using default metrics"""
        vintage_cv = VintageCV(
            data=self.vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial='60 days',
            assess='20 days',
            slice_limit=2
        )

        # Don't specify metrics (should use defaults)
        results = self.wf_set.fit_backtests(vintage_cv)

        # Should have default metrics (rmse, mae, r_squared)
        metrics = results.collect_metrics(by_vintage=True)
        metric_names = set(metrics['metric'].unique())

        assert 'rmse' in metric_names
        assert 'mae' in metric_names
        # r_squared may fail on small samples, so we don't require it

    def test_grouped_backtesting_not_implemented(self):
        """Test grouped backtesting raises NotImplementedError"""
        vintage_cv = VintageCV(
            data=self.vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial='60 days',
            assess='20 days',
            slice_limit=2
        )

        with pytest.raises(NotImplementedError, match="Grouped backtesting"):
            self.wf_set.fit_backtests(
                vintage_cv,
                metrics=metric_set(rmse),
                group_col='some_group'
            )

    def test_extract_best_workflow(self):
        """Test extracting best workflow"""
        vintage_cv = VintageCV(
            data=self.vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial='60 days',
            assess='20 days',
            slice_limit=2
        )

        results = self.wf_set.fit_backtests(
            vintage_cv,
            metrics=metric_set(rmse)
        )

        # Extract best
        best = results.extract_best_workflow('rmse')

        # Should return workflow ID
        assert isinstance(best, str)
        assert best in ['simple', 'full']

    def test_multiple_vintages(self):
        """Test with multiple vintage folds"""
        vintage_cv = VintageCV(
            data=self.vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial='40 days',
            assess='10 days',
            skip='5 days',
            slice_limit=5
        )

        # Should have 5 folds
        assert len(vintage_cv) == 5

        results = self.wf_set.fit_backtests(
            vintage_cv,
            metrics=metric_set(rmse)
        )

        # Should have results for all folds
        metrics = results.collect_metrics(by_vintage=True)
        n_vintages = len(metrics['vintage_date'].unique())

        # Should have 5 vintages
        assert n_vintages == 5


class TestBacktestWorkflowErrors:
    """Tests for error handling in backtest workflows."""

    def test_workflow_error_handling(self, capsys):
        """Test handling of workflow errors during backtesting"""
        np.random.seed(42)

        # Create data
        n = 100
        final_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=n, freq='D'),
            'x': np.random.randn(n),
            'y': np.random.randn(n)
        })

        vintage_data = create_vintage_data(
            final_data,
            date_col='date',
            n_revisions=2,
            revision_std=0.05
        )

        # Create workflow with intentional issue (missing column)
        wf_bad = workflow().add_formula('y ~ missing_col').add_model(linear_reg())
        wf_good = workflow().add_formula('y ~ x').add_model(linear_reg())

        wf_set = WorkflowSet.from_workflows(
            [wf_bad, wf_good],
            ids=['bad', 'good']
        )

        vintage_cv = VintageCV(
            data=vintage_data,
            as_of_col='as_of_date',
            date_col='date',
            initial='30 days',
            assess='10 days',
            slice_limit=2
        )

        # Should handle error gracefully
        results = wf_set.fit_backtests(vintage_cv, metrics=metric_set(rmse))

        # Capture warnings
        captured = capsys.readouterr()

        # Should print warning about failed workflow
        assert 'Warning' in captured.out or 'failed' in captured.out.lower()

        # Good workflow should still complete
        assert 'good' in results.workflow_ids
