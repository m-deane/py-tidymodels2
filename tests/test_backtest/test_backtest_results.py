"""
Tests for BacktestResults class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import timedelta

from py_backtest import BacktestResults


class TestBacktestResults:
    """Tests for BacktestResults class."""

    def setup_method(self):
        """Create mock backtest results"""
        # Create results for 2 workflows, 3 vintages each
        vintage_dates = [
            pd.Timestamp('2023-01-01'),
            pd.Timestamp('2023-02-01'),
            pd.Timestamp('2023-03-01')
        ]

        self.results = {}

        for wf_id in ['wf1', 'wf2']:
            folds = []

            for i, vintage_date in enumerate(vintage_dates):
                # Create metrics
                metrics = pd.DataFrame({
                    'metric': ['rmse', 'mae'],
                    'value': [5.0 + i * 0.5, 4.0 + i * 0.3] if wf_id == 'wf1' else [6.0 + i * 0.6, 5.0 + i * 0.4]
                })

                # Create predictions
                predictions = pd.DataFrame({
                    'date': pd.date_range(vintage_date, periods=10, freq='D'),
                    '.pred': np.random.randn(10),
                    'actual': np.random.randn(10)
                })

                # Create vintage info
                vintage_info = {
                    'vintage_date': vintage_date,
                    'training_start': vintage_date - timedelta(days=30),
                    'training_end': vintage_date,
                    'test_start': vintage_date + timedelta(days=1),
                    'test_end': vintage_date + timedelta(days=10),
                    'n_train_obs': 30,
                    'n_test_obs': 10,
                    'forecast_horizon': timedelta(days=1)
                }

                folds.append({
                    'fold_idx': i,
                    'vintage_info': vintage_info,
                    'predictions': predictions,
                    'metrics': metrics
                })

            self.results[wf_id] = {
                'wflow_id': wf_id,
                'folds': folds
            }

        self.backtest_results = BacktestResults(self.results)

    def test_initialization(self):
        """Test BacktestResults initialization"""
        assert len(self.backtest_results.workflow_ids) == 2
        assert 'wf1' in self.backtest_results.workflow_ids
        assert 'wf2' in self.backtest_results.workflow_ids

    def test_collect_metrics_by_vintage(self):
        """Test collecting metrics by vintage"""
        metrics = self.backtest_results.collect_metrics(by_vintage=True, summarize=False)

        # Should have metrics for both workflows and all vintages
        assert len(metrics) > 0
        assert 'wflow_id' in metrics.columns
        assert 'vintage_date' in metrics.columns
        assert 'metric' in metrics.columns
        assert 'value' in metrics.columns

        # Should have both workflows
        assert 'wf1' in metrics['wflow_id'].values
        assert 'wf2' in metrics['wflow_id'].values

        # Should have all vintages
        assert len(metrics['vintage_date'].unique()) == 3

    def test_collect_metrics_averaged(self):
        """Test collecting averaged metrics"""
        metrics = self.backtest_results.collect_metrics(by_vintage=False, summarize=False)

        # Should average across vintages
        assert len(metrics) > 0
        assert 'vintage_date' not in metrics.columns or len(metrics['vintage_date'].unique()) == 1

        # Should have both workflows
        assert 'wf1' in metrics['wflow_id'].values
        assert 'wf2' in metrics['wflow_id'].values

    def test_collect_metrics_summarized(self):
        """Test collecting summarized metrics"""
        metrics = self.backtest_results.collect_metrics(by_vintage=False, summarize=True)

        # Should have summary statistics
        assert 'mean' in metrics.columns
        assert 'std' in metrics.columns

        # Should have both workflows
        assert 'wf1' in metrics['wflow_id'].values
        assert 'wf2' in metrics['wflow_id'].values

    def test_rank_results(self):
        """Test ranking workflows"""
        ranked = self.backtest_results.rank_results('rmse', by_vintage=False, n=2)

        # Should have rank column
        assert 'rank' in ranked.columns

        # Should be sorted by rank
        assert ranked['rank'].tolist() == [1, 2]

        # First should be wf1 (lower RMSE)
        assert ranked.iloc[0]['wflow_id'] == 'wf1'

    def test_rank_results_by_vintage(self):
        """Test ranking workflows by vintage"""
        ranked = self.backtest_results.rank_results('rmse', by_vintage=True, n=2)

        # Should have vintage_date column
        assert 'vintage_date' in ranked.columns

        # Should have rankings for each vintage
        assert len(ranked['vintage_date'].unique()) == 3

    def test_extract_best_workflow(self):
        """Test extracting best workflow"""
        best = self.backtest_results.extract_best_workflow('rmse', by_vintage=False)

        # Should return workflow ID
        assert isinstance(best, str)
        assert best == 'wf1'  # wf1 has lower RMSE

    def test_extract_best_workflow_by_vintage(self):
        """Test extracting best workflow per vintage"""
        best = self.backtest_results.extract_best_workflow('rmse', by_vintage=True)

        # Should return DataFrame
        assert isinstance(best, pd.DataFrame)

        # Should have vintage_date column
        assert 'vintage_date' in best.columns

        # Should have one row per vintage
        assert len(best) == 3

    def test_analyze_vintage_drift(self):
        """Test analyzing vintage drift"""
        drift = self.backtest_results.analyze_vintage_drift('rmse')

        # Should have drift columns
        assert 'drift_from_start' in drift.columns
        assert 'drift_pct' in drift.columns
        assert 'metric_value' in drift.columns

        # Should have both workflows
        assert 'wf1' in drift['wflow_id'].values
        assert 'wf2' in drift['wflow_id'].values

        # First vintage should have zero drift
        first_vintage_rows = drift[drift['vintage_date'] == pd.Timestamp('2023-01-01')]
        assert all(first_vintage_rows['drift_from_start'] == 0.0)

        # Later vintages should have positive drift (RMSE increasing)
        last_vintage_rows = drift[drift['vintage_date'] == pd.Timestamp('2023-03-01')]
        assert all(last_vintage_rows['drift_from_start'] > 0.0)

    def test_analyze_forecast_horizon(self):
        """Test analyzing forecast horizon"""
        horizon = self.backtest_results.analyze_forecast_horizon('rmse')

        # Should have horizon column
        assert 'horizon' in horizon.columns
        assert 'n_folds' in horizon.columns

        # Should have both workflows
        assert 'wf1' in horizon['wflow_id'].values
        assert 'wf2' in horizon['wflow_id'].values

    def test_collect_predictions(self):
        """Test collecting predictions"""
        preds = self.backtest_results.collect_predictions()

        # Should have predictions from all workflows and vintages
        assert len(preds) > 0
        assert 'wflow_id' in preds.columns
        assert 'vintage_date' in preds.columns
        assert '.pred' in preds.columns

    def test_collect_predictions_filtered_by_vintage(self):
        """Test collecting predictions filtered by vintage"""
        vintage_date = pd.Timestamp('2023-01-01')
        preds = self.backtest_results.collect_predictions(vintage_date=vintage_date)

        # Should only have predictions for specified vintage
        assert all(preds['vintage_date'] == vintage_date)

    def test_repr(self):
        """Test string representation"""
        repr_str = repr(self.backtest_results)
        assert 'BacktestResults' in repr_str
        assert '2 workflows' in repr_str
        assert '3 vintages' in repr_str

    def test_metric_not_found_error(self):
        """Test error when metric not found"""
        with pytest.raises(ValueError, match="Metric 'invalid' not found"):
            self.backtest_results.rank_results('invalid')

    def test_compare_vintage_vs_final_warning(self):
        """Test compare_vintage_vs_final returns placeholder with warning"""
        with pytest.warns(UserWarning, match="requires final data metrics"):
            comparison = self.backtest_results.compare_vintage_vs_final('rmse')

        # Should have placeholder NaN values
        assert 'final_rmse' in comparison.columns
        assert comparison['final_rmse'].isna().all()
