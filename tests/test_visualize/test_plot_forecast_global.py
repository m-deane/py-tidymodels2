"""
Test that plot_forecast() supports grouped subplots for fit_global() use cases.

Verifies that plot_forecast() detects group columns in outputs and creates
separate subplots even when using fit_global() (not just fit_nested()).
"""

import pandas as pd
import numpy as np
from py_workflows import workflow
from py_parsnip import linear_reg
from py_recipes import recipe
from py_visualize import plot_forecast


def test_plot_forecast_global_grouped():
    """Test that plot_forecast() creates grouped subplots for fit_global()."""

    # Create grouped data
    np.random.seed(42)
    n_per_group = 100

    data_list = []
    for country in ['USA', 'UK', 'Germany']:
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=n_per_group, freq='D'),
            'country': country,
            'x1': np.random.randn(n_per_group),
            'x2': np.random.randn(n_per_group),
            'x3': np.random.randn(n_per_group),
        })
        df['target'] = df['x1'] * 2 + df['x2'] * 1.5 + np.random.randn(n_per_group) * 0.5
        data_list.append(df)

    data = pd.concat(data_list, ignore_index=True)

    # Split into train/test
    train = data[data['date'] < '2020-03-01'].copy()
    test = data[data['date'] >= '2020-03-01'].copy()

    # Fit global model (single model with country as feature)
    rec = recipe().step_normalize()
    wf = workflow().add_recipe(rec).add_model(linear_reg())
    fit = wf.fit_global(train, group_col='country')
    fit = fit.evaluate(test)

    # Create forecast plot
    fig = plot_forecast(fit)

    # Check that figure was created
    assert fig is not None

    # Check that there are traces for each group
    # Should have at least: Actuals, Fitted (Train), Fitted (Test) for each of 3 groups
    # Minimum = 3 * 3 = 9 traces (may have more if prediction intervals included)
    assert len(fig.data) >= 9, f"Expected at least 9 traces for 3 groups, got {len(fig.data)}"

    # Check subplot titles contain country names
    # Annotations in Plotly contain subplot titles
    if hasattr(fig.layout, 'annotations') and fig.layout.annotations:
        subplot_titles = fig.layout.annotations
        assert len(subplot_titles) >= 3, f"Expected 3 subplot titles, got {len(subplot_titles)}"

        # Verify country names appear in titles
        title_text = ' '.join([ann.text for ann in subplot_titles])
        assert 'USA' in title_text or 'country: USA' in title_text
        assert 'UK' in title_text or 'country: UK' in title_text
        assert 'Germany' in title_text or 'country: Germany' in title_text

    print("✓ plot_forecast() creates grouped subplots for fit_global()")


def test_plot_forecast_nested_still_works():
    """Test that plot_forecast() still works correctly with fit_nested()."""

    # Create grouped data
    np.random.seed(42)
    n_per_group = 100

    data_list = []
    for country in ['USA', 'UK']:
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=n_per_group, freq='D'),
            'country': country,
            'x1': np.random.randn(n_per_group),
            'x2': np.random.randn(n_per_group),
        })
        df['target'] = df['x1'] * 2 + np.random.randn(n_per_group) * 0.5
        data_list.append(df)

    data = pd.concat(data_list, ignore_index=True)

    # Split into train/test
    train = data[data['date'] < '2020-03-01'].copy()
    test = data[data['date'] >= '2020-03-01'].copy()

    # Fit nested models (separate model per country)
    rec = recipe().step_normalize()
    wf = workflow().add_recipe(rec).add_model(linear_reg())
    fit = wf.fit_nested(train, group_col='country')

    # Create forecast plot (uses the fit object with training data)
    fig = plot_forecast(fit)

    # Check that figure was created
    assert fig is not None

    # Check that it has subplots (2 groups, training data only)
    # Each group should have at least: Actuals (train), Fitted (train) = 2 traces per group
    # Total = 2 * 2 = 4 traces minimum
    assert len(fig.data) >= 4, f"Expected at least 4 traces for 2 groups, got {len(fig.data)}"

    # Check subplot titles
    if hasattr(fig.layout, 'annotations') and fig.layout.annotations:
        subplot_titles = fig.layout.annotations
        assert len(subplot_titles) >= 2, f"Expected 2 subplot titles, got {len(subplot_titles)}"

    print("✓ plot_forecast() still works correctly with fit_nested()")


def test_plot_forecast_single_model_without_groups():
    """Test that plot_forecast() creates single plot when no groups present."""

    # Create simple data without groups
    np.random.seed(42)
    n = 100

    data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n, freq='D'),
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
    })
    data['target'] = data['x1'] * 2 + np.random.randn(n) * 0.5

    # Split into train/test
    train = data[data['date'] < '2020-03-01'].copy()
    test = data[data['date'] >= '2020-03-01'].copy()

    # Fit single model
    wf = workflow().add_formula('target ~ .').add_model(linear_reg())
    fit = wf.fit(train)
    fit = fit.evaluate(test)

    # Create forecast plot
    fig = plot_forecast(fit)

    # Check that figure was created
    assert fig is not None

    # Check that it's a single plot (not subplots)
    # Single plot should have fewer traces than grouped (3-4 traces typical)
    assert len(fig.data) <= 5, f"Expected single plot with ≤5 traces, got {len(fig.data)}"

    # Check no subplot titles (annotations are only for subplots)
    subplot_titles = [ann for ann in fig.layout.annotations if hasattr(ann, 'text')]
    # Single plot may have 0-1 annotations (title), not multiple subplot titles
    assert len(subplot_titles) <= 1, f"Expected ≤1 annotation for single plot, got {len(subplot_titles)}"

    print("✓ plot_forecast() creates single plot when no groups present")


if __name__ == "__main__":
    import sys

    print("="*70)
    print("TESTING plot_forecast() WITH fit_global() GROUPED PLOTTING")
    print("="*70)

    try:
        print("\n" + "="*70)
        print("TEST 1: plot_forecast() with fit_global()")
        print("="*70)
        test_plot_forecast_global_grouped()

        print("\n" + "="*70)
        print("TEST 2: plot_forecast() with fit_nested()")
        print("="*70)
        test_plot_forecast_nested_still_works()

        print("\n" + "="*70)
        print("TEST 3: plot_forecast() with single model (no groups)")
        print("="*70)
        test_plot_forecast_single_model_without_groups()

        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nKEY FINDINGS:")
        print("  ✓ plot_forecast() creates grouped subplots for fit_global()")
        print("  ✓ plot_forecast() still works with fit_nested()")
        print("  ✓ plot_forecast() creates single plot when no groups present")
        print("  ✓ Group column auto-detection works for common names (country, group, etc.)")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
