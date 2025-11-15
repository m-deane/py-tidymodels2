"""
Tests for backtesting visualization functions.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

from py_backtest import BacktestResults
from py_backtest.visualizations import (
    plot_accuracy_over_time,
    plot_horizon_comparison,
    plot_vintage_drift,
    plot_revision_impact,
)


class TestPlotAccuracyOverTime:
    """Tests for plot_accuracy_over_time function."""

    def setup_method(self):
        """Create mock backtest results"""
        vintage_dates = [
            pd.Timestamp('2023-01-01'),
            pd.Timestamp('2023-02-01'),
            pd.Timestamp('2023-03-01')
        ]

        self.results = {}

        for wf_id in ['wf1', 'wf2']:
            folds = []

            for i, vintage_date in enumerate(vintage_dates):
                metrics = pd.DataFrame({
                    'metric': ['rmse', 'mae'],
                    'value': [
                        5.0 + i * 0.5,
                        4.0 + i * 0.3
                    ] if wf_id == 'wf1' else [
                        6.0 + i * 0.6,
                        5.0 + i * 0.4
                    ]
                })

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
                    'metrics': metrics
                })

            self.results[wf_id] = {
                'wflow_id': wf_id,
                'folds': folds
            }

        self.backtest_results = BacktestResults(self.results)

    def test_plot_by_workflow(self):
        """Test plotting separate lines per workflow"""
        fig = plot_accuracy_over_time(
            self.backtest_results,
            metric="rmse",
            by_workflow=True,
            show=False
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        ax = fig.axes[0]

        # Should have 2 lines (one per workflow)
        assert len(ax.lines) == 2

        # Check legend
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 2

        plt.close(fig)

    def test_plot_aggregated(self):
        """Test aggregated plotting (by_workflow=False)"""
        fig = plot_accuracy_over_time(
            self.backtest_results,
            metric="rmse",
            by_workflow=False,
            show=False
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        ax = fig.axes[0]

        # Should have 1 line (aggregated)
        assert len(ax.lines) == 1

        # Should have fill_between for confidence band
        assert len(ax.collections) > 0

        plt.close(fig)

    def test_plot_filtered_workflows(self):
        """Test filtering to specific workflows"""
        fig = plot_accuracy_over_time(
            self.backtest_results,
            metric="rmse",
            by_workflow=True,
            workflows=["wf1"],
            show=False
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Should only have 1 line
        assert len(ax.lines) == 1

        plt.close(fig)

    def test_plot_invalid_metric(self):
        """Test error on invalid metric"""
        with pytest.raises(ValueError, match="Metric 'invalid' not found"):
            plot_accuracy_over_time(
                self.backtest_results,
                metric="invalid",
                show=False
            )

    def test_plot_custom_kwargs(self):
        """Test custom plotting kwargs"""
        fig = plot_accuracy_over_time(
            self.backtest_results,
            metric="rmse",
            by_workflow=True,
            show=False,
            linewidth=4,
            marker="s",
            markersize=10
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Check custom linewidth
        for line in ax.lines:
            assert line.get_linewidth() == 4
            assert line.get_marker() == "s"

        plt.close(fig)


class TestPlotHorizonComparison:
    """Tests for plot_horizon_comparison function."""

    def setup_method(self):
        """Create mock backtest results with varying horizons"""
        vintage_dates = [
            pd.Timestamp('2023-01-01'),
            pd.Timestamp('2023-02-01'),
            pd.Timestamp('2023-03-01')
        ]

        horizons = [timedelta(days=1), timedelta(days=7), timedelta(days=14)]

        self.results = {}

        for wf_id in ['wf1', 'wf2']:
            folds = []

            for i, (vintage_date, horizon) in enumerate(zip(vintage_dates, horizons)):
                metrics = pd.DataFrame({
                    'metric': ['rmse', 'mae'],
                    'value': [
                        5.0 + i * 0.5,
                        4.0 + i * 0.3
                    ] if wf_id == 'wf1' else [
                        6.0 + i * 0.6,
                        5.0 + i * 0.4
                    ]
                })

                vintage_info = {
                    'vintage_date': vintage_date,
                    'training_start': vintage_date - timedelta(days=30),
                    'training_end': vintage_date,
                    'test_start': vintage_date + horizon,
                    'test_end': vintage_date + horizon + timedelta(days=10),
                    'n_train_obs': 30,
                    'n_test_obs': 10,
                    'forecast_horizon': horizon
                }

                folds.append({
                    'fold_idx': i,
                    'vintage_info': vintage_info,
                    'metrics': metrics
                })

            self.results[wf_id] = {
                'wflow_id': wf_id,
                'folds': folds
            }

        self.backtest_results = BacktestResults(self.results)

    def test_plot_single_workflow_bar(self):
        """Test bar chart for single workflow"""
        fig = plot_horizon_comparison(
            self.backtest_results,
            metric="rmse",
            workflows=["wf1"],
            show=False
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Should have bar chart
        assert len(ax.patches) > 0

        plt.close(fig)

    def test_plot_multiple_workflows_line(self):
        """Test line plot for multiple workflows"""
        fig = plot_horizon_comparison(
            self.backtest_results,
            metric="rmse",
            show=False
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Should have 2 lines (one per workflow)
        assert len(ax.lines) == 2

        # Check legend
        legend = ax.get_legend()
        assert legend is not None

        plt.close(fig)

    def test_plot_filtered_workflows(self):
        """Test filtering to specific workflows"""
        fig = plot_horizon_comparison(
            self.backtest_results,
            metric="rmse",
            workflows=["wf2"],
            show=False
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_invalid_metric(self):
        """Test error on invalid metric via analyze_forecast_horizon"""
        with pytest.raises(ValueError, match="Metric 'invalid' not found"):
            plot_horizon_comparison(
                self.backtest_results,
                metric="invalid",
                show=False
            )

    def test_plot_custom_kwargs(self):
        """Test custom plotting kwargs"""
        fig = plot_horizon_comparison(
            self.backtest_results,
            metric="rmse",
            show=False,
            linewidth=3,
            marker="^"
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Check custom kwargs applied
        for line in ax.lines:
            assert line.get_linewidth() == 3
            assert line.get_marker() == "^"

        plt.close(fig)


class TestPlotVintageDrift:
    """Tests for plot_vintage_drift function."""

    def setup_method(self):
        """Create mock backtest results"""
        vintage_dates = [
            pd.Timestamp('2023-01-01'),
            pd.Timestamp('2023-02-01'),
            pd.Timestamp('2023-03-01')
        ]

        self.results = {}

        for wf_id in ['wf1', 'wf2']:
            folds = []

            for i, vintage_date in enumerate(vintage_dates):
                # Simulate drift: metric increases over time
                metrics = pd.DataFrame({
                    'metric': ['rmse', 'mae'],
                    'value': [
                        5.0 + i * 0.5,
                        4.0 + i * 0.3
                    ] if wf_id == 'wf1' else [
                        6.0 + i * 0.6,
                        5.0 + i * 0.4
                    ]
                })

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
                    'metrics': metrics
                })

            self.results[wf_id] = {
                'wflow_id': wf_id,
                'folds': folds
            }

        self.backtest_results = BacktestResults(self.results)

    def test_plot_basic_drift(self):
        """Test basic drift plot"""
        fig = plot_vintage_drift(
            self.backtest_results,
            metric="rmse",
            show=False
        )

        assert isinstance(fig, plt.Figure)

        # Should have 2 subplots (absolute values + drift %)
        assert len(fig.axes) == 2

        # Each subplot should have 2 lines (one per workflow)
        for ax in fig.axes:
            assert len(ax.lines) >= 2  # May include axhline

        plt.close(fig)

    def test_plot_multiple_workflows(self):
        """Test drift plot with multiple workflows"""
        fig = plot_vintage_drift(
            self.backtest_results,
            metric="rmse",
            show=False
        )

        assert isinstance(fig, plt.Figure)

        # Check both workflows present in legends
        for ax in fig.axes:
            legend = ax.get_legend()
            if legend is not None:
                assert len(legend.get_texts()) == 2

        plt.close(fig)

    def test_plot_filtered_workflows(self):
        """Test filtering to specific workflows"""
        fig = plot_vintage_drift(
            self.backtest_results,
            metric="rmse",
            workflows=["wf1"],
            show=False
        )

        assert isinstance(fig, plt.Figure)

        # Should only have 1 workflow
        for ax in fig.axes:
            legend = ax.get_legend()
            if legend is not None:
                assert len(legend.get_texts()) == 1

        plt.close(fig)

    def test_plot_invalid_metric(self):
        """Test error on invalid metric"""
        with pytest.raises(ValueError, match="Metric 'invalid' not found"):
            plot_vintage_drift(
                self.backtest_results,
                metric="invalid",
                show=False
            )

    def test_plot_custom_kwargs(self):
        """Test custom plotting kwargs"""
        fig = plot_vintage_drift(
            self.backtest_results,
            metric="rmse",
            show=False,
            linewidth=5,
            marker="D"
        )

        assert isinstance(fig, plt.Figure)

        # Check custom kwargs applied to first subplot
        ax = fig.axes[0]
        for line in ax.lines:
            assert line.get_linewidth() == 5

        plt.close(fig)


class TestPlotRevisionImpact:
    """Tests for plot_revision_impact function."""

    def setup_method(self):
        """Create mock backtest results"""
        vintage_dates = [
            pd.Timestamp('2023-01-01'),
            pd.Timestamp('2023-02-01'),
            pd.Timestamp('2023-03-01')
        ]

        self.results = {}

        for wf_id in ['wf1', 'wf2']:
            folds = []

            for i, vintage_date in enumerate(vintage_dates):
                metrics = pd.DataFrame({
                    'metric': ['rmse', 'mae'],
                    'value': [5.0 + i * 0.5, 4.0 + i * 0.3]
                })

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
                    'metrics': metrics
                })

            self.results[wf_id] = {
                'wflow_id': wf_id,
                'folds': folds
            }

        self.backtest_results = BacktestResults(self.results)

    def test_plot_with_final_data(self):
        """Test plot with final data provided"""
        # Create mock vintage vs final comparison
        vintage_vs_final = pd.DataFrame({
            'wflow_id': ['wf1', 'wf2'],
            'vintage_rmse': [5.5, 6.6],
            'final_rmse': [5.2, 6.3]
        })

        fig = plot_revision_impact(
            self.backtest_results,
            metric="rmse",
            vintage_vs_final_data=vintage_vs_final,
            show=False
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Should have scatter points
        assert len(ax.collections) > 0

        # Should have diagonal line (y=x)
        assert len(ax.lines) > 0

        plt.close(fig)

    def test_plot_without_final_data(self):
        """Test plot without final data (placeholder)"""
        fig = plot_revision_impact(
            self.backtest_results,
            metric="rmse",
            show=False
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Should show warning message (no actual scatter plot)
        # Check that axis is turned off (placeholder mode)
        assert not ax.axison

        plt.close(fig)

    def test_plot_multiple_workflows(self):
        """Test plot with multiple workflows"""
        vintage_vs_final = pd.DataFrame({
            'wflow_id': ['wf1', 'wf2'],
            'vintage_rmse': [5.5, 6.6],
            'final_rmse': [5.2, 6.3]
        })

        fig = plot_revision_impact(
            self.backtest_results,
            metric="rmse",
            vintage_vs_final_data=vintage_vs_final,
            show=False
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Should have legend for multiple workflows
        legend = ax.get_legend()
        assert legend is not None

        plt.close(fig)

    def test_plot_single_workflow(self):
        """Test plot with single workflow"""
        vintage_vs_final = pd.DataFrame({
            'wflow_id': ['wf1'],
            'vintage_rmse': [5.5],
            'final_rmse': [5.2]
        })

        fig = plot_revision_impact(
            self.backtest_results,
            metric="rmse",
            workflows=["wf1"],
            vintage_vs_final_data=vintage_vs_final,
            show=False
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Should have scatter points
        assert len(ax.collections) > 0

        plt.close(fig)

    def test_plot_filtered_workflows(self):
        """Test filtering to specific workflows"""
        vintage_vs_final = pd.DataFrame({
            'wflow_id': ['wf1', 'wf2'],
            'vintage_rmse': [5.5, 6.6],
            'final_rmse': [5.2, 6.3]
        })

        fig = plot_revision_impact(
            self.backtest_results,
            metric="rmse",
            workflows=["wf1"],
            vintage_vs_final_data=vintage_vs_final,
            show=False
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Should only have 1 workflow
        assert len(ax.collections) > 0

        plt.close(fig)


class TestBacktestResultsPlotMethods:
    """Tests for plot methods on BacktestResults class."""

    def setup_method(self):
        """Create mock backtest results"""
        vintage_dates = [
            pd.Timestamp('2023-01-01'),
            pd.Timestamp('2023-02-01'),
        ]

        self.results = {}

        for wf_id in ['wf1']:
            folds = []

            for i, vintage_date in enumerate(vintage_dates):
                metrics = pd.DataFrame({
                    'metric': ['rmse', 'mae'],
                    'value': [5.0 + i * 0.5, 4.0 + i * 0.3]
                })

                vintage_info = {
                    'vintage_date': vintage_date,
                    'training_start': vintage_date - timedelta(days=30),
                    'training_end': vintage_date,
                    'test_start': vintage_date + timedelta(days=1),
                    'test_end': vintage_date + timedelta(days=10),
                    'n_train_obs': 30,
                    'n_test_obs': 10,
                    'forecast_horizon': timedelta(days=1 + i)
                }

                folds.append({
                    'fold_idx': i,
                    'vintage_info': vintage_info,
                    'metrics': metrics
                })

            self.results[wf_id] = {
                'wflow_id': wf_id,
                'folds': folds
            }

        self.backtest_results = BacktestResults(self.results)

    def test_plot_accuracy_over_time_method(self):
        """Test plot_accuracy_over_time method on BacktestResults"""
        fig = self.backtest_results.plot_accuracy_over_time(
            metric="rmse",
            show=False
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_horizon_comparison_method(self):
        """Test plot_horizon_comparison method on BacktestResults"""
        fig = self.backtest_results.plot_horizon_comparison(
            metric="rmse",
            show=False
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_vintage_drift_method(self):
        """Test plot_vintage_drift method on BacktestResults"""
        fig = self.backtest_results.plot_vintage_drift(
            metric="rmse",
            show=False
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_revision_impact_method(self):
        """Test plot_revision_impact method on BacktestResults"""
        fig = self.backtest_results.plot_revision_impact(
            metric="rmse",
            show=False
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)
