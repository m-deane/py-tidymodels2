"""
Tests for plot_model_comparison() function

Tests cover:
- Bar chart comparison
- Heatmap comparison
- Radar chart comparison
- Multiple models and metrics
- Train/test split handling
- Edge cases
"""

import pytest
import pandas as pd
import numpy as np

from py_workflows import workflow
from py_parsnip import linear_reg, rand_forest
from py_visualize import plot_model_comparison


class TestPlotModelComparisonBar:
    """Test plot_model_comparison() with bar chart mode"""

    def test_basic_bar_comparison(self):
        """Test basic bar chart comparing two models"""
        np.random.seed(42)

        # Create data
        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100) * 10
        })

        train = data.iloc[:80]
        test = data.iloc[80:]

        # Fit two models
        wf1 = workflow().add_formula("y ~ x").add_model(linear_reg())
        wf2 = workflow().add_formula("y ~ x").add_model(linear_reg(penalty=0.1))

        fit1 = wf1.fit(train).evaluate(test)
        fit2 = wf2.fit(train).evaluate(test)

        # Extract stats
        _, _, stats1 = fit1.extract_outputs()
        _, _, stats2 = fit2.extract_outputs()

        # Create comparison plot
        fig = plot_model_comparison(
            [stats1, stats2],
            model_names=["OLS", "Ridge"],
            metrics=["rmse", "mae", "r_squared"]
        )

        assert fig is not None
        assert len(fig.data) > 0

        # Should have one trace per model
        assert len(fig.data) == 2

    def test_bar_comparison_auto_metrics(self):
        """Test bar chart with automatic metric selection"""
        np.random.seed(42)

        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100) * 10
        })

        wf = workflow().add_formula("y ~ x").add_model(linear_reg())
        fit = wf.fit(data)

        _, _, stats = fit.extract_outputs()

        # Create comparison with auto metrics (None)
        fig = plot_model_comparison([stats], model_names=["Model"], metrics=None)

        assert fig is not None
        # Should automatically select common metrics

    def test_bar_comparison_multiple_models(self):
        """Test bar chart with 3+ models"""
        np.random.seed(42)

        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100) * 10
        })

        train = data.iloc[:80]
        test = data.iloc[80:]

        # Fit three models
        models = [
            linear_reg(),
            linear_reg(penalty=0.1),
            linear_reg(penalty=1.0)
        ]

        stats_list = []
        for model in models:
            wf = workflow().add_formula("y ~ x").add_model(model)
            fit = wf.fit(train).evaluate(test)
            _, _, stats = fit.extract_outputs()
            stats_list.append(stats)

        # Create comparison
        fig = plot_model_comparison(
            stats_list,
            model_names=["OLS", "Ridge (0.1)", "Ridge (1.0)"],
            metrics=["rmse", "mae"]
        )

        assert fig is not None
        assert len(fig.data) == 3  # One trace per model


class TestPlotModelComparisonHeatmap:
    """Test heatmap comparison mode"""

    def test_heatmap_comparison(self):
        """Test heatmap for many models and metrics"""
        np.random.seed(42)

        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100) * 10
        })

        train = data.iloc[:80]
        test = data.iloc[80:]

        # Fit multiple models
        penalties = [0.0, 0.1, 0.5, 1.0, 2.0]
        stats_list = []
        model_names = []

        for penalty in penalties:
            wf = workflow().add_formula("y ~ x").add_model(linear_reg(penalty=penalty))
            fit = wf.fit(train).evaluate(test)
            _, _, stats = fit.extract_outputs()
            stats_list.append(stats)
            model_names.append(f"Ridge ({penalty})")

        # Create heatmap
        fig = plot_model_comparison(
            stats_list,
            model_names=model_names,
            metrics=["rmse", "mae", "r_squared"],
            plot_type="heatmap"
        )

        assert fig is not None
        assert len(fig.data) > 0

        # Heatmap should have one trace
        assert len(fig.data) == 1


class TestPlotModelComparisonRadar:
    """Test radar chart comparison mode"""

    def test_radar_comparison(self):
        """Test radar chart with normalized metrics"""
        np.random.seed(42)

        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100) * 10
        })

        train = data.iloc[:80]
        test = data.iloc[80:]

        # Fit two models
        wf1 = workflow().add_formula("y ~ x").add_model(linear_reg())
        wf2 = workflow().add_formula("y ~ x").add_model(linear_reg(penalty=0.5))

        fit1 = wf1.fit(train).evaluate(test)
        fit2 = wf2.fit(train).evaluate(test)

        _, _, stats1 = fit1.extract_outputs()
        _, _, stats2 = fit2.extract_outputs()

        # Create radar chart
        fig = plot_model_comparison(
            [stats1, stats2],
            model_names=["OLS", "Ridge"],
            metrics=["rmse", "mae", "r_squared"],
            plot_type="radar"
        )

        assert fig is not None
        assert len(fig.data) > 0

        # Should have one trace per model
        assert len(fig.data) == 2

    def test_radar_with_single_model(self):
        """Test radar chart with single model"""
        np.random.seed(42)

        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100) * 10
        })

        wf = workflow().add_formula("y ~ x").add_model(linear_reg())
        fit = wf.fit(data)

        _, _, stats = fit.extract_outputs()

        # Create radar with single model
        fig = plot_model_comparison(
            [stats],
            model_names=["Model"],
            metrics=["rmse", "mae", "r_squared"],
            plot_type="radar"
        )

        assert fig is not None
        assert len(fig.data) == 1


class TestPlotModelComparisonSplits:
    """Test handling of train/test splits"""

    def test_comparison_test_split_only(self):
        """Test comparison using only test split"""
        np.random.seed(42)

        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100) * 10
        })

        train = data.iloc[:80]
        test = data.iloc[80:]

        # Fit two models
        wf1 = workflow().add_formula("y ~ x").add_model(linear_reg())
        wf2 = workflow().add_formula("y ~ x").add_model(linear_reg(penalty=0.1))

        fit1 = wf1.fit(train).evaluate(test)
        fit2 = wf2.fit(train).evaluate(test)

        _, _, stats1 = fit1.extract_outputs()
        _, _, stats2 = fit2.extract_outputs()

        # Compare using test split
        fig = plot_model_comparison(
            [stats1, stats2],
            model_names=["OLS", "Ridge"],
            metrics=["rmse", "mae"],
            split="test"
        )

        assert fig is not None

    def test_comparison_train_split_only(self):
        """Test comparison using only train split"""
        np.random.seed(42)

        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100) * 10
        })

        wf1 = workflow().add_formula("y ~ x").add_model(linear_reg())
        wf2 = workflow().add_formula("y ~ x").add_model(linear_reg(penalty=0.1))

        fit1 = wf1.fit(data)
        fit2 = wf2.fit(data)

        _, _, stats1 = fit1.extract_outputs()
        _, _, stats2 = fit2.extract_outputs()

        # Compare using train split
        fig = plot_model_comparison(
            [stats1, stats2],
            model_names=["OLS", "Ridge"],
            metrics=["rmse", "mae"],
            split="train"
        )

        assert fig is not None


class TestPlotModelComparisonCustomization:
    """Test customization options"""

    def test_custom_options(self):
        """Test custom title, height, width, and legend"""
        np.random.seed(42)

        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100) * 10
        })

        wf = workflow().add_formula("y ~ x").add_model(linear_reg())
        fit = wf.fit(data)

        _, _, stats = fit.extract_outputs()

        # Create plot with custom options
        fig = plot_model_comparison(
            [stats],
            model_names=["Model"],
            metrics=["rmse", "mae"],
            title="Custom Comparison",
            height=700,
            width=900,
            show_legend=False
        )

        assert fig is not None
        assert fig.layout.title.text == "Custom Comparison"
        assert fig.layout.height == 700
        assert fig.layout.width == 900
        assert fig.layout.showlegend == False


class TestPlotModelComparisonEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_stats_list(self):
        """Test error with empty stats list"""
        with pytest.raises(ValueError, match="stats_list cannot be empty"):
            plot_model_comparison([], model_names=[], metrics=["rmse"])

    def test_mismatched_names_length(self):
        """Test error when model_names length doesn't match stats_list"""
        np.random.seed(42)

        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100) * 10
        })

        wf = workflow().add_formula("y ~ x").add_model(linear_reg())
        fit = wf.fit(data)

        _, _, stats = fit.extract_outputs()

        # Mismatched lengths
        with pytest.raises(ValueError, match="Length of model_names"):
            plot_model_comparison(
                [stats, stats],
                model_names=["Model1"],  # Only 1 name for 2 models
                metrics=["rmse"]
            )

    def test_invalid_plot_type(self):
        """Test error with invalid plot type"""
        np.random.seed(42)

        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100) * 10
        })

        wf = workflow().add_formula("y ~ x").add_model(linear_reg())
        fit = wf.fit(data)

        _, _, stats = fit.extract_outputs()

        with pytest.raises(ValueError, match="Unknown plot_type"):
            plot_model_comparison(
                [stats],
                model_names=["Model"],
                metrics=["rmse"],
                plot_type="invalid"
            )

    def test_auto_model_names(self):
        """Test automatic model name generation"""
        np.random.seed(42)

        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100) * 10
        })

        wf1 = workflow().add_formula("y ~ x").add_model(linear_reg())
        wf2 = workflow().add_formula("y ~ x").add_model(linear_reg(penalty=0.1))

        fit1 = wf1.fit(data)
        fit2 = wf2.fit(data)

        _, _, stats1 = fit1.extract_outputs()
        _, _, stats2 = fit2.extract_outputs()

        # Don't provide model names - should auto-generate
        fig = plot_model_comparison(
            [stats1, stats2],
            model_names=None,  # Auto-generate
            metrics=["rmse", "mae"]
        )

        assert fig is not None
        # Should use "Model 1", "Model 2", etc.
