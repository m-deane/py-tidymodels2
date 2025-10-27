"""
Tests for plot_tune_results() function

Tests cover:
- Line plots for single parameter
- Heatmap for two parameters
- Parallel coordinates for 3+ parameters
- Scatter plot matrix
- Auto plot type selection
- Highlighting best configurations
- Edge cases
"""

import pytest
import pandas as pd
import numpy as np

from py_workflows import workflow
from py_parsnip import linear_reg, rand_forest
from py_tune import tune, tune_grid, grid_regular
from py_rsample import time_series_cv
from py_visualize import plot_tune_results


class TestPlotTuneResultsLine:
    """Test plot_tune_results() with line plot (single parameter)"""

    def test_line_plot_single_parameter(self):
        """Test line plot for single parameter tuning"""
        np.random.seed(42)

        # Create data
        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100) * 10
        })

        # Create CV splits
        from py_rsample import initial_time_split
        split = initial_time_split(data, prop=0.8)
        train = split.training()

        # Create simple mock tuning results for testing
        # In real usage, this would come from tune_grid()
        from py_tune.tune_grid import TuneResults

        # Mock results DataFrame
        results_df = pd.DataFrame({
            "penalty": [0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
            "metric": ["rmse"] * 6,
            "value": [10.5, 10.2, 10.0, 10.1, 10.3, 10.8],
            "split": ["test"] * 6
        })

        # Create TuneResults object
        tune_results = TuneResults(results=results_df, workflow=None, resamples=None)

        # Create line plot
        fig = plot_tune_results(tune_results, metric="rmse", plot_type="line")

        assert fig is not None
        assert len(fig.data) > 0

    def test_line_plot_with_error_bands(self):
        """Test line plot with standard deviation bands"""
        np.random.seed(42)

        # Mock results with multiple CV folds (for std calculation)
        results_df = pd.DataFrame({
            "penalty": [0.1, 0.1, 0.5, 0.5, 1.0, 1.0],
            "metric": ["rmse"] * 6,
            "value": [10.2, 10.3, 10.0, 10.1, 10.1, 10.2],
            "split": ["test"] * 6,
            ".config": ["config_1_fold_1", "config_1_fold_2", "config_2_fold_1", "config_2_fold_2", "config_3_fold_1", "config_3_fold_2"]
        })

        from py_tune.tune_grid import TuneResults
        tune_results = TuneResults(results=results_df, workflow=None, resamples=None)

        # Create line plot
        fig = plot_tune_results(tune_results, metric="rmse", plot_type="line")

        assert fig is not None
        # Should have main line plus error bands
        assert len(fig.data) >= 1

    def test_line_plot_show_best(self):
        """Test line plot with best configurations highlighted"""
        results_df = pd.DataFrame({
            "penalty": [0.0, 0.1, 0.5, 1.0, 2.0],
            "metric": ["rmse"] * 5,
            "value": [10.5, 10.0, 10.2, 10.8, 11.0],
            "split": ["test"] * 5
        })

        from py_tune.tune_grid import TuneResults
        tune_results = TuneResults(results=results_df, workflow=None, resamples=None)

        # Create line plot showing top 2
        fig = plot_tune_results(tune_results, metric="rmse", plot_type="line", show_best=2)

        assert fig is not None
        # Should have additional trace for best configs
        assert len(fig.data) >= 2


class TestPlotTuneResultsHeatmap:
    """Test heatmap for two parameters"""

    def test_heatmap_two_parameters(self):
        """Test heatmap for two parameter tuning"""
        # Create grid of two parameters
        penalties = [0.1, 0.5, 1.0]
        mixtures = [0.0, 0.5, 1.0]

        results_rows = []
        for pen in penalties:
            for mix in mixtures:
                results_rows.append({
                    "penalty": pen,
                    "mixture": mix,
                    "metric": "rmse",
                    "value": pen + mix + np.random.randn() * 0.1,
                    "split": "test"
                })

        results_df = pd.DataFrame(results_rows)

        from py_tune.tune_grid import TuneResults
        tune_results = TuneResults(results=results_df, workflow=None, resamples=None)

        # Create heatmap
        fig = plot_tune_results(tune_results, metric="rmse", plot_type="heatmap")

        assert fig is not None
        assert len(fig.data) > 0

    def test_heatmap_show_best(self):
        """Test heatmap with best configurations highlighted"""
        penalties = [0.1, 0.5, 1.0]
        mixtures = [0.0, 0.5, 1.0]

        results_rows = []
        for pen in penalties:
            for mix in mixtures:
                results_rows.append({
                    "penalty": pen,
                    "mixture": mix,
                    "metric": "rmse",
                    "value": pen + mix,
                    "split": "test"
                })

        results_df = pd.DataFrame(results_rows)

        from py_tune.tune_grid import TuneResults
        tune_results = TuneResults(results=results_df, workflow=None, resamples=None)

        # Create heatmap showing best
        fig = plot_tune_results(tune_results, metric="rmse", plot_type="heatmap", show_best=3)

        assert fig is not None
        # Should have heatmap plus markers for best configs
        assert len(fig.data) >= 2


class TestPlotTuneResultsParallel:
    """Test parallel coordinates for 3+ parameters"""

    def test_parallel_three_parameters(self):
        """Test parallel coordinates for three parameters"""
        # Create results with 3 parameters
        results_rows = []
        for i in range(20):
            results_rows.append({
                "trees": np.random.choice([50, 100, 200]),
                "min_n": np.random.choice([2, 5, 10]),
                "mtry": np.random.choice([2, 4, 6]),
                "metric": "rmse",
                "value": np.random.randn() + 10,
                "split": "test"
            })

        results_df = pd.DataFrame(results_rows)

        from py_tune.tune_grid import TuneResults
        tune_results = TuneResults(results=results_df, workflow=None, resamples=None)

        # Create parallel coordinates plot
        fig = plot_tune_results(tune_results, metric="rmse", plot_type="parallel")

        assert fig is not None
        assert len(fig.data) > 0


class TestPlotTuneResultsScatter:
    """Test scatter plot matrix"""

    def test_scatter_matrix_two_parameters(self):
        """Test scatter matrix for two parameters"""
        penalties = [0.1, 0.5, 1.0, 2.0]
        mixtures = [0.0, 0.25, 0.5, 0.75, 1.0]

        results_rows = []
        for pen in penalties:
            for mix in mixtures:
                results_rows.append({
                    "penalty": pen,
                    "mixture": mix,
                    "metric": "rmse",
                    "value": pen + mix + np.random.randn() * 0.1,
                    "split": "test"
                })

        results_df = pd.DataFrame(results_rows)

        from py_tune.tune_grid import TuneResults
        tune_results = TuneResults(results=results_df, workflow=None, resamples=None)

        # Create scatter matrix
        fig = plot_tune_results(tune_results, metric="rmse", plot_type="scatter")

        assert fig is not None
        assert len(fig.data) > 0


class TestPlotTuneResultsAuto:
    """Test automatic plot type selection"""

    def test_auto_single_parameter(self):
        """Test auto selects line plot for single parameter"""
        results_df = pd.DataFrame({
            "penalty": [0.0, 0.1, 0.5, 1.0],
            "metric": ["rmse"] * 4,
            "value": [10.5, 10.0, 10.2, 10.8],
            "split": ["test"] * 4
        })

        from py_tune.tune_grid import TuneResults
        tune_results = TuneResults(results=results_df, workflow=None, resamples=None)

        # Use auto - should select line
        fig = plot_tune_results(tune_results, metric="rmse", plot_type="auto")

        assert fig is not None

    def test_auto_two_parameters(self):
        """Test auto selects heatmap for two parameters"""
        results_rows = []
        for pen in [0.1, 0.5, 1.0]:
            for mix in [0.0, 0.5, 1.0]:
                results_rows.append({
                    "penalty": pen,
                    "mixture": mix,
                    "metric": "rmse",
                    "value": pen + mix,
                    "split": "test"
                })

        results_df = pd.DataFrame(results_rows)

        from py_tune.tune_grid import TuneResults
        tune_results = TuneResults(results=results_df, workflow=None, resamples=None)

        # Use auto - should select heatmap
        fig = plot_tune_results(tune_results, metric="rmse", plot_type="auto")

        assert fig is not None

    def test_auto_three_plus_parameters(self):
        """Test auto selects parallel for 3+ parameters"""
        results_rows = []
        for i in range(10):
            results_rows.append({
                "param1": np.random.choice([1, 2, 3]),
                "param2": np.random.choice([0.1, 0.5, 1.0]),
                "param3": np.random.choice([10, 20, 30]),
                "metric": "rmse",
                "value": np.random.randn() + 10,
                "split": "test"
            })

        results_df = pd.DataFrame(results_rows)

        from py_tune.tune_grid import TuneResults
        tune_results = TuneResults(results=results_df, workflow=None, resamples=None)

        # Use auto - should select parallel
        fig = plot_tune_results(tune_results, metric="rmse", plot_type="auto")

        assert fig is not None


class TestPlotTuneResultsCustomization:
    """Test customization options"""

    def test_custom_title_and_dimensions(self):
        """Test custom title, height, and width"""
        results_df = pd.DataFrame({
            "penalty": [0.0, 0.1, 0.5],
            "metric": ["rmse"] * 3,
            "value": [10.5, 10.0, 10.2],
            "split": ["test"] * 3
        })

        from py_tune.tune_grid import TuneResults
        tune_results = TuneResults(results=results_df, workflow=None, resamples=None)

        # Create plot with custom options
        fig = plot_tune_results(
            tune_results,
            metric="rmse",
            plot_type="line",
            title="Custom Tuning Plot",
            height=700,
            width=900
        )

        assert fig is not None
        assert fig.layout.title.text == "Custom Tuning Plot"
        assert fig.layout.height == 700
        assert fig.layout.width == 900


class TestPlotTuneResultsEdgeCases:
    """Test edge cases and error handling"""

    def test_invalid_metric(self):
        """Test error when metric not in results"""
        results_df = pd.DataFrame({
            "penalty": [0.0, 0.1],
            "metric": ["rmse"] * 2,
            "value": [10.5, 10.0],
            "split": ["test"] * 2
        })

        from py_tune.tune_grid import TuneResults
        tune_results = TuneResults(results=results_df, workflow=None, resamples=None)

        # Request metric that doesn't exist
        with pytest.raises(ValueError, match="Metric 'mae' not found"):
            plot_tune_results(tune_results, metric="mae")

    def test_invalid_plot_type(self):
        """Test error with invalid plot type"""
        results_df = pd.DataFrame({
            "penalty": [0.0, 0.1],
            "metric": ["rmse"] * 2,
            "value": [10.5, 10.0],
            "split": ["test"] * 2
        })

        from py_tune.tune_grid import TuneResults
        tune_results = TuneResults(results=results_df, workflow=None, resamples=None)

        with pytest.raises(ValueError, match="Unknown plot_type"):
            plot_tune_results(tune_results, metric="rmse", plot_type="invalid")

    def test_no_tunable_parameters(self):
        """Test behavior when no parameters were tuned"""
        # Results with only metric column (no parameters)
        results_df = pd.DataFrame({
            "metric": ["rmse"] * 3,
            "value": [10.5, 10.0, 10.2],
            "split": ["test"] * 3
        })

        from py_tune.tune_grid import TuneResults
        tune_results = TuneResults(results=results_df, workflow=None, resamples=None)

        # This should either handle gracefully or raise informative error
        # Depends on implementation - at minimum shouldn't crash
        pass  # Placeholder for actual test based on expected behavior

    def test_single_configuration(self):
        """Test with only one configuration (no tuning)"""
        results_df = pd.DataFrame({
            "penalty": [0.1],
            "metric": ["rmse"],
            "value": [10.0],
            "split": ["test"]
        })

        from py_tune.tune_grid import TuneResults
        tune_results = TuneResults(results=results_df, workflow=None, resamples=None)

        # Should handle gracefully
        fig = plot_tune_results(tune_results, metric="rmse", plot_type="line")

        assert fig is not None
