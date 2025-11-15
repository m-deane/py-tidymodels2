"""
Tests for SHAP visualization functions

Tests cover:
- Summary plots (beeswarm and bar)
- Waterfall plots for local explanations
- Force plots (matplotlib and HTML modes)
- Dependence plots with interaction detection
- Temporal plots for time series
- Integration with ModelFit.explain_plot()
- Integration with WorkflowFit.explain_plot()
- Integration with NestedWorkflowFit.explain_plot()
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from py_parsnip import linear_reg, rand_forest, decision_tree
from py_workflows import workflow
from py_recipes import recipe
from py_interpret.visualizations import (
    summary_plot,
    waterfall_plot,
    force_plot,
    dependence_plot,
    temporal_plot
)


# Test fixtures
@pytest.fixture
def simple_regression_data():
    """Simple regression dataset."""
    np.random.seed(42)
    n = 100
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    X3 = np.random.randn(n)
    y = 2 * X1 + 3 * X2 - 1.5 * X3 + np.random.randn(n) * 0.1

    return pd.DataFrame({
        'y': y,
        'X1': X1,
        'X2': X2,
        'X3': X3
    })


@pytest.fixture
def time_series_data():
    """Time series regression dataset."""
    np.random.seed(42)
    n = 50
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    y = 2 * X1 + 3 * X2 + np.random.randn(n) * 0.1

    return pd.DataFrame({
        'date': dates,
        'y': y,
        'X1': X1,
        'X2': X2
    })


@pytest.fixture
def grouped_regression_data():
    """Grouped regression dataset."""
    np.random.seed(42)
    n_per_group = 30
    groups = ['A', 'B']

    all_data = []
    for group in groups:
        X1 = np.random.randn(n_per_group)
        X2 = np.random.randn(n_per_group)
        y = 2 * X1 + 3 * X2 + np.random.randn(n_per_group) * 0.1

        df = pd.DataFrame({
            'group_id': group,
            'y': y,
            'X1': X1,
            'X2': X2
        })
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


# Test summary plots
class TestSummaryPlot:
    """Test summary plot functionality."""

    def test_summary_plot_beeswarm(self, simple_regression_data):
        """Test beeswarm summary plot."""
        # Fit model and get SHAP values
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')
        shap_df = fit.explain(simple_regression_data.head(20))

        # Create plot
        fig = summary_plot(shap_df, plot_type="beeswarm", show=False)

        # Validate
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_summary_plot_bar(self, simple_regression_data):
        """Test bar summary plot."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')
        shap_df = fit.explain(simple_regression_data.head(20))

        # Create plot
        fig = summary_plot(shap_df, plot_type="bar", show=False)

        # Validate
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_summary_plot_max_display(self, simple_regression_data):
        """Test max_display parameter."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')
        shap_df = fit.explain(simple_regression_data.head(20))

        # Create plot with max_display=2
        fig = summary_plot(shap_df, max_display=2, show=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_summary_plot_invalid_type(self, simple_regression_data):
        """Test error on invalid plot_type."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')
        shap_df = fit.explain(simple_regression_data.head(20))

        with pytest.raises(ValueError, match="Unknown plot_type"):
            summary_plot(shap_df, plot_type="invalid", show=False)


# Test waterfall plots
class TestWaterfallPlot:
    """Test waterfall plot functionality."""

    def test_waterfall_plot_basic(self, simple_regression_data):
        """Test basic waterfall plot."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')
        shap_df = fit.explain(simple_regression_data.head(20))

        # Create plot for first observation
        fig = waterfall_plot(shap_df, observation_id=0, show=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_waterfall_plot_different_observation(self, simple_regression_data):
        """Test waterfall plot for different observation."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')
        shap_df = fit.explain(simple_regression_data.head(20))

        # Create plot for observation 10
        fig = waterfall_plot(shap_df, observation_id=10, show=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_waterfall_plot_invalid_observation(self, simple_regression_data):
        """Test error on invalid observation_id."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')
        shap_df = fit.explain(simple_regression_data.head(20))

        with pytest.raises(ValueError, match="observation_id.*not found"):
            waterfall_plot(shap_df, observation_id=999, show=False)


# Test force plots
class TestForcePlot:
    """Test force plot functionality."""

    def test_force_plot_matplotlib(self, simple_regression_data):
        """Test matplotlib force plot."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')
        shap_df = fit.explain(simple_regression_data.head(20))

        # Create matplotlib plot
        fig = force_plot(shap_df, observation_id=0, matplotlib=True, show=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_force_plot_html(self, simple_regression_data):
        """Test HTML force plot."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')
        shap_df = fit.explain(simple_regression_data.head(20))

        # Create HTML plot
        html = force_plot(shap_df, observation_id=0, matplotlib=False, show=False)

        # SHAP force_plot returns a custom object, not a string
        assert html is not None

    def test_force_plot_invalid_observation(self, simple_regression_data):
        """Test error on invalid observation_id."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')
        shap_df = fit.explain(simple_regression_data.head(20))

        with pytest.raises(ValueError, match="observation_id.*not found"):
            force_plot(shap_df, observation_id=999, matplotlib=True, show=False)


# Test dependence plots
class TestDependencePlot:
    """Test dependence plot functionality."""

    def test_dependence_plot_auto_interaction(self, simple_regression_data):
        """Test dependence plot with auto-detected interaction."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')
        shap_df = fit.explain(simple_regression_data.head(20))

        # Create plot with auto interaction
        fig = dependence_plot(shap_df, feature="X1", interaction_feature="auto", show=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dependence_plot_specific_interaction(self, simple_regression_data):
        """Test dependence plot with specific interaction feature."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')
        shap_df = fit.explain(simple_regression_data.head(20))

        # Create plot with specific interaction
        fig = dependence_plot(shap_df, feature="X1", interaction_feature="X2", show=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dependence_plot_no_interaction(self, simple_regression_data):
        """Test dependence plot without interaction coloring."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')
        shap_df = fit.explain(simple_regression_data.head(20))

        # Create plot without interaction
        fig = dependence_plot(shap_df, feature="X1", interaction_feature=None, show=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dependence_plot_invalid_feature(self, simple_regression_data):
        """Test error on invalid feature name."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')
        shap_df = fit.explain(simple_regression_data.head(20))

        with pytest.raises(ValueError, match="Feature.*not found"):
            dependence_plot(shap_df, feature="InvalidFeature", show=False)


# Test temporal plots
class TestTemporalPlot:
    """Test temporal plot functionality."""

    def test_temporal_plot_all_features(self, time_series_data):
        """Test temporal plot with all features."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(time_series_data, 'y ~ X1 + X2')
        shap_df = fit.explain(time_series_data.head(30))

        # Create plot with all features
        fig = temporal_plot(shap_df, show=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_temporal_plot_single_feature(self, time_series_data):
        """Test temporal plot with single feature."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(time_series_data, 'y ~ X1 + X2')
        shap_df = fit.explain(time_series_data.head(30))

        # Create plot for single feature
        fig = temporal_plot(shap_df, features="X1", show=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_temporal_plot_multiple_features(self, time_series_data):
        """Test temporal plot with multiple features."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(time_series_data, 'y ~ X1 + X2')
        shap_df = fit.explain(time_series_data.head(30))

        # Create plot for multiple features
        fig = temporal_plot(shap_df, features=["X1", "X2"], show=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_temporal_plot_aggregations(self, time_series_data):
        """Test different aggregation methods."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(time_series_data, 'y ~ X1 + X2')
        shap_df = fit.explain(time_series_data.head(30))

        # Test mean aggregation
        fig1 = temporal_plot(shap_df, aggregation="mean", show=False)
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        # Test sum aggregation
        fig2 = temporal_plot(shap_df, aggregation="sum", show=False)
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)

        # Test abs_mean aggregation
        fig3 = temporal_plot(shap_df, aggregation="abs_mean", show=False)
        assert isinstance(fig3, plt.Figure)
        plt.close(fig3)

    def test_temporal_plot_plot_types(self, time_series_data):
        """Test different plot types."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(time_series_data, 'y ~ X1 + X2')
        shap_df = fit.explain(time_series_data.head(30))

        # Test line plot
        fig1 = temporal_plot(shap_df, plot_type="line", show=False)
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        # Test area plot
        fig2 = temporal_plot(shap_df, plot_type="area", show=False)
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)

    def test_temporal_plot_missing_date(self, simple_regression_data):
        """Test error when date column missing."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')
        shap_df = fit.explain(simple_regression_data.head(20))

        with pytest.raises(ValueError, match="date.*column"):
            temporal_plot(shap_df, show=False)


# Test integration with explain_plot
class TestExplainPlotIntegration:
    """Test integration with ModelFit.explain_plot() method."""

    def test_modelfit_explain_plot_summary(self, simple_regression_data):
        """Test ModelFit.explain_plot() with summary."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')

        # Use explain_plot method
        fig = fit.explain_plot(simple_regression_data.head(20), plot_type="summary", show=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_modelfit_explain_plot_waterfall(self, simple_regression_data):
        """Test ModelFit.explain_plot() with waterfall."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')

        fig = fit.explain_plot(
            simple_regression_data.head(20),
            plot_type="waterfall",
            observation_id=0,
            show=False
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_modelfit_explain_plot_dependence(self, simple_regression_data):
        """Test ModelFit.explain_plot() with dependence."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')

        fig = fit.explain_plot(
            simple_regression_data.head(20),
            plot_type="dependence",
            feature="X1",
            show=False
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_modelfit_explain_plot_missing_args(self, simple_regression_data):
        """Test error when required args missing."""
        spec = rand_forest(trees=10).set_mode('regression')
        fit = spec.fit(simple_regression_data, 'y ~ X1 + X2 + X3')

        # Waterfall without observation_id
        with pytest.raises(ValueError, match="observation_id required"):
            fit.explain_plot(simple_regression_data.head(20), plot_type="waterfall")

        # Dependence without feature
        with pytest.raises(ValueError, match="feature required"):
            fit.explain_plot(simple_regression_data.head(20), plot_type="dependence")

    def test_workflowfit_explain_plot(self, simple_regression_data):
        """Test WorkflowFit.explain_plot()."""
        rec = recipe().step_normalize()
        wf = workflow().add_recipe(rec).add_model(rand_forest(trees=10).set_mode('regression'))
        wf_fit = wf.fit(simple_regression_data)

        fig = wf_fit.explain_plot(simple_regression_data.head(20), plot_type="summary", show=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_nestedworkflowfit_explain_plot(self, grouped_regression_data):
        """Test NestedWorkflowFit.explain_plot()."""
        wf = workflow().add_formula("y ~ X1 + X2").add_model(linear_reg())
        nested_fit = wf.fit_nested(grouped_regression_data, group_col="group_id")

        # All groups
        fig1 = nested_fit.explain_plot(grouped_regression_data.head(20), plot_type="summary", show=False)
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        # Specific group
        fig2 = nested_fit.explain_plot(
            grouped_regression_data.head(20),
            plot_type="summary",
            group="A",
            show=False
        )
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)
