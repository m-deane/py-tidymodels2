"""
Test residual diagnostics (Issue 1) - Ljung-Box and Breusch-Pagan tests
"""
import pytest
import pandas as pd
import numpy as np
from py_parsnip import linear_reg


@pytest.fixture
def time_series_data():
    """Time series data for testing diagnostics"""
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    x = np.linspace(0, 10, n)
    y = 2 * x + np.random.normal(0, 1, n)

    return pd.DataFrame({
        'date': dates,
        'x': x,
        'y': y
    })


def test_ljung_box_not_nan_statsmodels(time_series_data):
    """Test that Ljung-Box statistic is not NaN for statsmodels engine"""
    spec = linear_reg().set_engine("statsmodels")
    fit = spec.fit(time_series_data, 'y ~ x')

    outputs, coefficients, stats = fit.extract_outputs()

    # Find Ljung-Box stats in stats DataFrame
    ljung_box_stat = stats[stats['metric'] == 'ljung_box_stat']
    ljung_box_p = stats[stats['metric'] == 'ljung_box_p']

    # Should have values, not NaN
    assert len(ljung_box_stat) > 0, "ljung_box_stat missing from stats"
    assert len(ljung_box_p) > 0, "ljung_box_p missing from stats"

    # Values should not be NaN (at least for this dataset)
    assert not pd.isna(ljung_box_stat['value'].iloc[0]), "ljung_box_stat is NaN"
    assert not pd.isna(ljung_box_p['value'].iloc[0]), "ljung_box_p is NaN"


def test_breusch_pagan_not_nan_statsmodels(time_series_data):
    """Test that Breusch-Pagan statistic is not NaN for statsmodels engine"""
    spec = linear_reg().set_engine("statsmodels")
    fit = spec.fit(time_series_data, 'y ~ x')

    outputs, coefficients, stats = fit.extract_outputs()

    # Find Breusch-Pagan stats in stats DataFrame
    bp_stat = stats[stats['metric'] == 'breusch_pagan_stat']
    bp_p = stats[stats['metric'] == 'breusch_pagan_p']

    # Should have values, not NaN
    assert len(bp_stat) > 0, "breusch_pagan_stat missing from stats"
    assert len(bp_p) > 0, "breusch_pagan_p missing from stats"

    # Values should not be NaN (at least for this dataset)
    assert not pd.isna(bp_stat['value'].iloc[0]), "breusch_pagan_stat is NaN"
    assert not pd.isna(bp_p['value'].iloc[0]), "breusch_pagan_p is NaN"


def test_ljung_box_sklearn_engine(time_series_data):
    """Test that sklearn engine also has Ljung-Box stats"""
    spec = linear_reg().set_engine("sklearn")
    fit = spec.fit(time_series_data, 'y ~ x')

    outputs, coefficients, stats = fit.extract_outputs()

    # Find Ljung-Box stats in stats DataFrame
    ljung_box_stat = stats[stats['metric'] == 'ljung_box_stat']

    # sklearn engine should also have diagnostics
    assert len(ljung_box_stat) > 0, "ljung_box_stat missing from sklearn stats"


def test_diagnostics_with_small_sample():
    """Test that diagnostics handle small sample sizes gracefully"""
    np.random.seed(42)
    small_data = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2, 4, 6, 8, 10]
    })

    spec = linear_reg().set_engine("statsmodels")
    fit = spec.fit(small_data, 'y ~ x')

    outputs, coefficients, stats = fit.extract_outputs()

    # Should not crash, but may have NaN for small samples
    ljung_box_stat = stats[stats['metric'] == 'ljung_box_stat']
    assert len(ljung_box_stat) > 0  # Should exist even if NaN
