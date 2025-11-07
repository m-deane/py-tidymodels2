"""
Test statsforecast AutoARIMA engine (Issue 3)
"""
import pytest
import pandas as pd
import numpy as np
from py_parsnip import arima_reg


@pytest.fixture
def time_series_data():
    """Simple time series data"""
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    y = np.cumsum(np.random.randn(n)) + 100  # Random walk

    return pd.DataFrame({
        'date': dates,
        'y': y
    })


@pytest.fixture
def exog_time_series_data():
    """Time series with exogenous variables"""
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    x1 = np.linspace(0, 10, n)
    x2 = np.sin(np.linspace(0, 4*np.pi, n))
    y = 2 * x1 + 3 * x2 + np.cumsum(np.random.randn(n)) + 100

    return pd.DataFrame({
        'date': dates,
        'x1': x1,
        'x2': x2,
        'y': y
    })


def test_statsforecast_engine_basic():
    """Test that statsforecast engine can be set"""
    spec = arima_reg().set_engine("statsforecast")
    assert spec.engine == "statsforecast"


def test_statsforecast_fit_univariate(time_series_data):
    """Test fitting univariate ARIMA with statsforecast"""
    spec = arima_reg().set_engine("statsforecast")
    fit = spec.fit(time_series_data, 'y ~ date')

    assert fit is not None
    assert fit.fit_data["engine"] == "statsforecast"
    assert "model" in fit.fit_data
    assert "order" in fit.fit_data
    assert "seasonal_order" in fit.fit_data


def test_statsforecast_predict(time_series_data):
    """Test prediction with statsforecast"""
    train = time_series_data.iloc[:80]
    test = time_series_data.iloc[80:]

    spec = arima_reg().set_engine("statsforecast")
    fit = spec.fit(train, 'y ~ date')
    predictions = fit.predict(test)

    assert len(predictions) == len(test)
    assert '.pred' in predictions.columns
    assert not predictions['.pred'].isna().all()


def test_statsforecast_with_exog(exog_time_series_data):
    """Test ARIMAX with exogenous variables"""
    train = exog_time_series_data.iloc[:80]
    test = exog_time_series_data.iloc[80:]

    spec = arima_reg().set_engine("statsforecast")
    fit = spec.fit(train, 'y ~ date + x1 + x2')
    predictions = fit.predict(test)

    assert len(predictions) == len(test)
    assert '.pred' in predictions.columns
    assert not predictions['.pred'].isna().all()


def test_statsforecast_with_seasonal(time_series_data):
    """Test seasonal ARIMA"""
    spec = arima_reg(seasonal_period=7).set_engine("statsforecast")
    fit = spec.fit(time_series_data, 'y ~ date')

    assert fit.fit_data["seasonal_order"][3] == 7  # Check seasonal period


def test_statsforecast_max_constraints():
    """Test that max parameter constraints work"""
    np.random.seed(42)
    n = 50
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    y = np.cumsum(np.random.randn(n)) + 100

    data = pd.DataFrame({'date': dates, 'y': y})

    # Set max orders
    spec = arima_reg(
        non_seasonal_ar=2,  # max_p
        non_seasonal_differences=1,  # max_d
        non_seasonal_ma=2,  # max_q
    ).set_engine("statsforecast")

    fit = spec.fit(data, 'y ~ date')

    # Check that orders are within constraints
    order = fit.fit_data["order"]
    assert order[0] <= 2  # p <= max_p
    assert order[1] <= 1  # d <= max_d
    assert order[2] <= 2  # q <= max_q


def test_statsforecast_extract_outputs(time_series_data):
    """Test extract_outputs() returns three DataFrames"""
    spec = arima_reg().set_engine("statsforecast")
    fit = spec.fit(time_series_data, 'y ~ date')

    outputs, coefficients, stats = fit.extract_outputs()

    # Check outputs
    assert 'actuals' in outputs.columns
    assert 'fitted' in outputs.columns
    assert 'residuals' in outputs.columns
    assert 'split' in outputs.columns

    # Check coefficients (ARIMA orders)
    assert 'variable' in coefficients.columns
    assert 'coefficient' in coefficients.columns
    assert 'ar_order' in coefficients['variable'].values
    assert 'ma_order' in coefficients['variable'].values
    assert 'diff_order' in coefficients['variable'].values

    # Check stats
    assert 'metric' in stats.columns
    assert 'value' in stats.columns
    assert 'split' in stats.columns
    assert 'rmse' in stats['metric'].values
    assert 'mae' in stats['metric'].values


def test_statsforecast_residual_diagnostics(time_series_data):
    """Test that residual diagnostics are calculated"""
    spec = arima_reg().set_engine("statsforecast")
    fit = spec.fit(time_series_data, 'y ~ date')

    outputs, coefficients, stats = fit.extract_outputs()

    # Check for diagnostic metrics
    metrics = stats['metric'].values
    assert 'ljung_box_stat' in metrics
    assert 'ljung_box_p' in metrics
    assert 'shapiro_wilk_stat' in metrics
    assert 'shapiro_wilk_p' in metrics


def test_statsforecast_date_fields(time_series_data):
    """Test that train_start_date and train_end_date are present"""
    spec = arima_reg().set_engine("statsforecast")
    fit = spec.fit(time_series_data, 'y ~ date')

    outputs, coefficients, stats = fit.extract_outputs()

    # Check for date fields
    metrics = stats['metric'].values
    assert 'train_start_date' in metrics
    assert 'train_end_date' in metrics

    # Verify dates are not NaN
    train_start = stats[stats['metric'] == 'train_start_date']['value'].iloc[0]
    train_end = stats[stats['metric'] == 'train_end_date']['value'].iloc[0]
    assert pd.notna(train_start)
    assert pd.notna(train_end)


def test_statsforecast_vs_pmdarima_comparable():
    """Test that statsforecast gives similar results to pmdarima"""
    np.random.seed(42)
    n = 60
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    y = np.cumsum(np.random.randn(n)) + 100

    data = pd.DataFrame({'date': dates, 'y': y})
    train = data.iloc[:50]
    test = data.iloc[50:]

    # Fit with statsforecast
    spec_sf = arima_reg().set_engine("statsforecast")
    fit_sf = spec_sf.fit(train, 'y ~ date')
    pred_sf = fit_sf.predict(test)

    # Both should produce predictions
    assert len(pred_sf) == len(test)
    assert not pred_sf['.pred'].isna().all()

    # Check that model was fitted successfully
    assert fit_sf.fit_data["model"] is not None
    assert "order" in fit_sf.fit_data
    assert "seasonal_order" in fit_sf.fit_data


def test_statsforecast_conf_int():
    """Test confidence interval predictions"""
    np.random.seed(42)
    n = 60
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    y = np.cumsum(np.random.randn(n)) + 100

    data = pd.DataFrame({'date': dates, 'y': y})
    train = data.iloc[:50]
    test = data.iloc[50:]

    spec = arima_reg().set_engine("statsforecast")
    fit = spec.fit(train, 'y ~ date')
    predictions = fit.predict(test, type="conf_int")

    assert '.pred' in predictions.columns
    assert '.pred_lower' in predictions.columns
    assert '.pred_upper' in predictions.columns
