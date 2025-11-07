"""
Tests for plot_forecast_multi() function

Verifies multi-model plotting functionality with:
- Combined outputs DataFrames
- List of fitted models
- Group filtering
- Residuals subplot
"""

import pytest
import pandas as pd
import numpy as np
from py_workflows import workflow
from py_parsnip import linear_reg, null_model
from py_visualize import plot_forecast_multi


@pytest.fixture
def sample_data():
    """Create simple time series data"""
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    trend = np.linspace(10, 50, n)
    noise = np.random.randn(n) * 2

    data = pd.DataFrame({
        'date': dates,
        'y': trend + noise,
        'x': np.arange(n)
    })
    return data


@pytest.fixture
def two_models_fit(sample_data):
    """Fit two simple models for testing"""
    train = sample_data.iloc[:80].copy()
    test = sample_data.iloc[80:].copy()

    # Model 1: Linear regression
    wf1 = workflow().add_formula('y ~ x').add_model(linear_reg().set_engine('sklearn'))
    fit1 = wf1.fit(train).evaluate(test)

    # Model 2: Null model (mean)
    wf2 = workflow().add_formula('y ~ x').add_model(null_model(strategy='mean'))
    fit2 = wf2.fit(train).evaluate(test)

    return fit1, fit2


class TestPlotForecastMultiBasic:
    """Test basic plot_forecast_multi functionality"""

    def test_accepts_list_of_fits(self, two_models_fit):
        """Test that function accepts list of fitted models"""
        fit1, fit2 = two_models_fit

        # Should work with list of fits
        fig = plot_forecast_multi([fit1, fit2])

        assert fig is not None
        assert len(fig.data) > 0  # Has traces

    def test_accepts_combined_dataframe(self, two_models_fit):
        """Test that function accepts combined outputs DataFrame"""
        fit1, fit2 = two_models_fit

        # Extract and combine outputs
        outputs1, _, _ = fit1.extract_outputs()
        outputs2, _, _ = fit2.extract_outputs()
        combined = pd.concat([outputs1, outputs2], ignore_index=True)

        # Should work with combined DataFrame
        fig = plot_forecast_multi(combined)

        assert fig is not None
        assert len(fig.data) > 0

    def test_requires_model_column(self, sample_data):
        """Test that error is raised if outputs missing 'model' column"""
        # Create DataFrame without 'model' column
        bad_df = pd.DataFrame({
            'date': sample_data['date'],
            'actuals': sample_data['y'],
            'fitted': sample_data['y'],
            'split': ['train'] * len(sample_data)
        })

        with pytest.raises(ValueError, match="must have 'model' column"):
            plot_forecast_multi(bad_df)

    def test_plots_multiple_models(self, two_models_fit):
        """Test that multiple models are plotted with distinct traces"""
        fit1, fit2 = two_models_fit

        fig = plot_forecast_multi([fit1, fit2])

        # Should have traces for:
        # - Training data (1)
        # - Model 1 train fitted (1)
        # - Model 1 test forecast (1)
        # - Model 2 train fitted (1)
        # - Model 2 test forecast (1)
        # - Test data (1)
        # = 6 traces minimum
        assert len(fig.data) >= 6


class TestPlotForecastMultiCustomization:
    """Test customization options"""

    def test_custom_model_names_list(self, two_models_fit):
        """Test custom model names as list"""
        fit1, fit2 = two_models_fit

        fig = plot_forecast_multi(
            [fit1, fit2],
            model_names=["Linear Regression", "Baseline Mean"]
        )

        # Check that custom names appear in legend
        trace_names = [trace.name for trace in fig.data]
        assert any("Linear Regression" in name for name in trace_names)
        assert any("Baseline Mean" in name for name in trace_names)

    def test_custom_model_names_dict(self, two_models_fit):
        """Test custom model names as dict"""
        fit1, fit2 = two_models_fit

        # Extract outputs to get model names
        outputs1, _, _ = fit1.extract_outputs()
        outputs2, _, _ = fit2.extract_outputs()
        combined = pd.concat([outputs1, outputs2], ignore_index=True)

        model1_name = outputs1['model'].iloc[0]
        model2_name = outputs2['model'].iloc[0]

        fig = plot_forecast_multi(
            combined,
            model_names={
                model1_name: "OLS",
                model2_name: "Naive"
            }
        )

        # Check custom names in legend
        trace_names = [trace.name for trace in fig.data]
        assert any("OLS" in name for name in trace_names)
        assert any("Naive" in name for name in trace_names)

    def test_include_residuals_subplot(self, two_models_fit):
        """Test that residuals subplot is added when requested"""
        fit1, fit2 = two_models_fit

        fig = plot_forecast_multi([fit1, fit2], include_residuals=True)

        # Should have more traces (residuals added)
        # and height should be doubled
        assert len(fig.data) > 6  # Additional residual traces

    def test_custom_title(self, two_models_fit):
        """Test custom plot title"""
        fit1, fit2 = two_models_fit

        custom_title = "My Custom Comparison"
        fig = plot_forecast_multi([fit1, fit2], title=custom_title)

        assert fig.layout.title.text == custom_title

    def test_custom_dimensions(self, two_models_fit):
        """Test custom height and width"""
        fit1, fit2 = two_models_fit

        fig = plot_forecast_multi([fit1, fit2], height=600, width=800)

        assert fig.layout.height == 600
        assert fig.layout.width == 800


class TestPlotForecastMultiDataHandling:
    """Test data handling edge cases"""

    def test_handles_missing_test_data(self, sample_data):
        """Test plotting when models have no test evaluation"""
        train = sample_data.iloc[:80].copy()

        # Fit models without evaluation (no test data)
        wf1 = workflow().add_formula('y ~ x').add_model(linear_reg().set_engine('sklearn'))
        fit1 = wf1.fit(train)

        wf2 = workflow().add_formula('y ~ x').add_model(null_model(strategy='mean'))
        fit2 = wf2.fit(train)

        # Should still work, just no test traces
        fig = plot_forecast_multi([fit1, fit2])

        assert fig is not None
        assert len(fig.data) > 0

    def test_filters_by_group(self, sample_data):
        """Test group filtering for grouped models"""
        # Create grouped data
        data_g1 = sample_data.copy()
        data_g1['group'] = 'group1'

        data_g2 = sample_data.copy()
        data_g2['group'] = 'group2'

        combined_data = pd.concat([data_g1, data_g2], ignore_index=True)

        train = combined_data.iloc[:160].copy()  # 80 per group
        test = combined_data.iloc[160:].copy()   # 20 per group

        # Fit models (would normally use fit_nested, but here we simulate)
        wf1 = workflow().add_formula('y ~ x').add_model(linear_reg().set_engine('sklearn'))
        fit1 = wf1.fit(train).evaluate(test)

        # Extract outputs (will have group column)
        outputs, _, _ = fit1.extract_outputs()

        # Should be able to filter by group
        if 'group' in outputs.columns:
            fig = plot_forecast_multi(outputs, group='group1')
            assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
