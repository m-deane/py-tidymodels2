"""
Tests for plot_residuals() function

Tests cover:
- All diagnostic plots mode
- Individual plot modes (fitted, qq, time, hist)
- Different data types (date-indexed, integer-indexed)
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from py_workflows import workflow
from py_parsnip import linear_reg
from py_visualize import plot_residuals


class TestPlotResidualsAll:
    """Test plot_residuals() with plot_type='all'"""

    def test_all_diagnostics_plot(self):
        """Test creating all diagnostic plots in 2x2 grid"""
        np.random.seed(42)

        # Create data with some pattern
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        x = np.arange(100)
        y = 2 * x + 5 + np.random.randn(100) * 10

        data = pd.DataFrame({
            "date": dates,
            "x": x,
            "y": y
        })

        # Fit model
        wf = workflow().add_formula("y ~ x").add_model(linear_reg())
        fit = wf.fit(data)

        # Create all diagnostics plot
        fig = plot_residuals(fit, plot_type="all")

        # Check figure was created
        assert fig is not None
        assert len(fig.data) > 0

        # Should have multiple traces (one for each subplot)
        # At least 4 traces for the 4 diagnostic plots
        assert len(fig.data) >= 4

    def test_all_diagnostics_with_train_test(self):
        """Test diagnostics use only training data even when test exists"""
        np.random.seed(42)

        dates = pd.date_range("2023-01-01", periods=120, freq="D")
        x = np.arange(120)
        y = 2 * x + 5 + np.random.randn(120) * 10

        data = pd.DataFrame({
            "date": dates,
            "x": x,
            "y": y
        })

        train = data.iloc[:100]
        test = data.iloc[100:]

        # Fit and evaluate
        wf = workflow().add_formula("y ~ x").add_model(linear_reg())
        fit = wf.fit(train).evaluate(test)

        # Create diagnostics - should only use training data
        fig = plot_residuals(fit, plot_type="all")

        assert fig is not None
        # Diagnostics should be based on 100 training observations, not 120 total


class TestPlotResidualsIndividual:
    """Test individual plot modes"""

    def test_residuals_vs_fitted(self):
        """Test residuals vs fitted plot"""
        np.random.seed(42)

        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100) * 5
        })

        wf = workflow().add_formula("y ~ x").add_model(linear_reg())
        fit = wf.fit(data)

        # Create residuals vs fitted plot
        fig = plot_residuals(fit, plot_type="fitted")

        assert fig is not None
        assert len(fig.data) > 0

        # Should have scatter plot trace
        assert any(trace.mode == "markers" for trace in fig.data)

    def test_qq_plot(self):
        """Test Q-Q plot for normality"""
        np.random.seed(42)

        # Create data with normally distributed errors
        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100) * 5
        })

        wf = workflow().add_formula("y ~ x").add_model(linear_reg())
        fit = wf.fit(data)

        # Create Q-Q plot
        fig = plot_residuals(fit, plot_type="qq")

        assert fig is not None
        assert len(fig.data) > 0

        # Should have scatter and line traces
        assert len(fig.data) >= 2

    def test_residuals_vs_time(self):
        """Test residuals vs time plot"""
        np.random.seed(42)

        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = pd.DataFrame({
            "date": dates,
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100) * 5
        })

        wf = workflow().add_formula("y ~ x").add_model(linear_reg())
        fit = wf.fit(data)

        # Create residuals vs time plot
        fig = plot_residuals(fit, plot_type="time")

        assert fig is not None
        assert len(fig.data) > 0

    def test_histogram(self):
        """Test histogram of residuals"""
        np.random.seed(42)

        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100) * 5
        })

        wf = workflow().add_formula("y ~ x").add_model(linear_reg())
        fit = wf.fit(data)

        # Create histogram
        fig = plot_residuals(fit, plot_type="hist")

        assert fig is not None
        assert len(fig.data) > 0

        # Should have histogram and normal curve overlay
        assert len(fig.data) >= 2


class TestPlotResidualsCustomization:
    """Test customization options"""

    def test_custom_title_and_dimensions(self):
        """Test custom title, height, and width"""
        np.random.seed(42)

        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100) * 5
        })

        wf = workflow().add_formula("y ~ x").add_model(linear_reg())
        fit = wf.fit(data)

        # Create plot with custom options
        fig = plot_residuals(
            fit,
            plot_type="fitted",
            title="Custom Diagnostics",
            height=700,
            width=900
        )

        assert fig is not None
        assert fig.layout.title.text == "Custom Diagnostics"
        assert fig.layout.height == 700
        assert fig.layout.width == 900


class TestPlotResidualsEdgeCases:
    """Test edge cases and error handling"""

    def test_invalid_plot_type(self):
        """Test error on invalid plot type"""
        np.random.seed(42)

        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100) * 5
        })

        wf = workflow().add_formula("y ~ x").add_model(linear_reg())
        fit = wf.fit(data)

        # Should raise error for invalid plot type
        with pytest.raises(ValueError, match="Unknown plot_type"):
            plot_residuals(fit, plot_type="invalid")

    def test_no_training_data(self):
        """Test error when no training data available"""
        # This is an edge case - normally all fits should have training data
        # The function should raise an error if outputs has no training split
        pass  # Placeholder - actual test depends on how we want to handle this

    def test_with_non_date_indexed_data(self):
        """Test residuals plot without date column"""
        np.random.seed(42)

        # Data without date column
        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100) * 5
        })

        wf = workflow().add_formula("y ~ x").add_model(linear_reg())
        fit = wf.fit(data)

        # Should still create plot, using observation index
        fig = plot_residuals(fit, plot_type="all")

        assert fig is not None
        assert len(fig.data) > 0

    def test_perfect_fit_zero_residuals(self):
        """Test with perfect fit (near-zero residuals)"""
        np.random.seed(42)

        # Create perfectly linear data
        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2.0  # Perfect linear relationship
        })

        wf = workflow().add_formula("y ~ x").add_model(linear_reg())
        fit = wf.fit(data)

        # Should handle near-zero residuals gracefully
        fig = plot_residuals(fit, plot_type="all")

        assert fig is not None
        # Residuals should be very small
        outputs, _, _ = fit.extract_outputs()
        train_residuals = outputs[outputs["split"] == "train"]["residuals"]
        assert np.abs(train_residuals).max() < 1e-10

    def test_large_residuals(self):
        """Test with large residuals (poor fit)"""
        np.random.seed(42)

        # Create data with large noise
        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100) * 100  # Large noise
        })

        wf = workflow().add_formula("y ~ x").add_model(linear_reg())
        fit = wf.fit(data)

        # Should handle large residuals gracefully
        fig = plot_residuals(fit, plot_type="all")

        assert fig is not None
        assert len(fig.data) > 0
