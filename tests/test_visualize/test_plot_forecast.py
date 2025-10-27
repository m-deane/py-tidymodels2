"""
Tests for plot_forecast() function

Tests cover:
- Single model forecast plots
- Nested model forecast plots
- Prediction intervals
- Date-indexed vs integer-indexed data
- Train/test split visualization
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from py_workflows import workflow
from py_parsnip import linear_reg, recursive_reg
from py_visualize import plot_forecast


class TestPlotForecastSingle:
    """Test plot_forecast() for single models"""

    def test_basic_forecast_plot(self):
        """Test basic forecast plot with train and test data"""
        np.random.seed(42)

        # Create train and test data
        dates = pd.date_range("2023-01-01", periods=120, freq="D")
        data = pd.DataFrame({
            "date": dates,
            "value": np.cumsum(np.random.randn(120)) + 100
        })

        train = data.iloc[:100].copy()
        test = data.iloc[100:].copy()

        # Fit model
        wf = workflow().add_formula("value ~ date").add_model(linear_reg())
        fit = wf.fit(train).evaluate(test)

        # Create forecast plot
        fig = plot_forecast(fit)

        # Check figure was created
        assert fig is not None
        assert len(fig.data) > 0

        # Should have at least training and fitted traces
        assert any("Training" in str(trace.name) for trace in fig.data)

    def test_forecast_plot_without_test(self):
        """Test forecast plot with only training data"""
        np.random.seed(42)

        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = pd.DataFrame({
            "date": dates,
            "value": np.cumsum(np.random.randn(100)) + 100
        })

        # Fit without evaluate()
        wf = workflow().add_formula("value ~ date").add_model(linear_reg())
        fit = wf.fit(data)

        # Create forecast plot
        fig = plot_forecast(fit)

        assert fig is not None
        # Should only have training data traces
        trace_names = [str(trace.name) for trace in fig.data]
        assert any("Training" in name for name in trace_names)
        assert not any("Test" in name for name in trace_names)

    def test_forecast_plot_with_prediction_intervals(self):
        """Test forecast plot with prediction intervals"""
        np.random.seed(42)

        dates = pd.date_range("2023-01-01", periods=120, freq="D")
        data = pd.DataFrame({
            "date": dates,
            "value": np.cumsum(np.random.randn(120)) + 100
        })
        data = data.set_index("date")

        train = data.iloc[:100]
        test_dates = pd.date_range("2023-04-11", periods=20, freq="D")
        test = pd.DataFrame(index=test_dates)

        # Fit recursive model (supports prediction intervals)
        from py_parsnip import rand_forest
        wf = workflow().add_formula("value ~ .").add_model(
            recursive_reg(base_model=rand_forest(trees=50).set_mode("regression"), lags=7)
        )
        fit = wf.fit(train.reset_index())

        # Predict with intervals
        preds = fit.predict(test.reset_index(), type="pred_int")

        # Since evaluate() needs actuals, create manual test data
        # For this test, just check the plot can be created
        fig = plot_forecast(fit, prediction_intervals=True)

        assert fig is not None

    def test_forecast_plot_customization(self):
        """Test forecast plot customization options"""
        np.random.seed(42)

        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = pd.DataFrame({
            "date": dates,
            "value": np.cumsum(np.random.randn(100)) + 100
        })

        wf = workflow().add_formula("value ~ date").add_model(linear_reg())
        fit = wf.fit(data)

        # Test with custom options
        fig = plot_forecast(
            fit,
            title="Custom Title",
            height=600,
            width=800,
            show_legend=False,
            prediction_intervals=False
        )

        assert fig is not None
        assert fig.layout.title.text == "Custom Title"
        assert fig.layout.height == 600
        assert fig.layout.width == 800


class TestPlotForecastNested:
    """Test plot_forecast() for nested/grouped models"""

    def test_nested_forecast_plot(self):
        """Test forecast plot with nested models"""
        np.random.seed(42)

        # Create grouped data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = []
        for store in ["A", "B"]:
            store_data = pd.DataFrame({
                "date": dates,
                "store_id": store,
                "sales": np.cumsum(np.random.randn(100)) + (100 if store == "A" else 200)
            })
            data.append(store_data)
        data = pd.concat(data, ignore_index=True)

        train = data[data["date"] < "2023-03-11"]
        test = data[data["date"] >= "2023-03-11"]

        # Fit nested model
        wf = workflow().add_formula("sales ~ date").add_model(linear_reg())
        nested_fit = wf.fit_nested(train, group_col="store_id").evaluate(test)

        # Create forecast plot
        fig = plot_forecast(nested_fit)

        assert fig is not None
        # Should have subplots for each group
        assert len(fig.data) > 0

    def test_nested_without_date_column(self):
        """Test nested forecast plot without date column"""
        np.random.seed(42)

        # Create data without explicit date column
        data = pd.DataFrame({
            "time": list(range(100)) * 2,
            "group": ["A"] * 100 + ["B"] * 100,
            "value": np.concatenate([
                np.linspace(100, 150, 100),
                np.linspace(200, 250, 100)
            ])
        })

        train = data[data["time"] < 80]
        test = data[data["time"] >= 80]

        wf = workflow().add_formula("value ~ time").add_model(linear_reg())
        nested_fit = wf.fit_nested(train, group_col="group").evaluate(test)

        # Create forecast plot
        fig = plot_forecast(nested_fit)

        assert fig is not None
        assert len(fig.data) > 0


class TestPlotForecastEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_fit(self):
        """Test with model that has no outputs"""
        # This is a edge case - normally all fits should have outputs
        # Just ensure it doesn't crash
        pass  # Placeholder - actual implementation depends on fit structure

    def test_missing_date_column(self):
        """Test with data that has no date column"""
        np.random.seed(42)

        # Data without date column
        data = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) * 2 + np.random.randn(100)
        })

        wf = workflow().add_formula("y ~ x").add_model(linear_reg())
        fit = wf.fit(data)

        # Should still create plot, using index
        fig = plot_forecast(fit)

        assert fig is not None
        assert len(fig.data) > 0
