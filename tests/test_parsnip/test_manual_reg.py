"""
Tests for manual_reg() model

Manual regression allows users to specify coefficients directly,
useful for comparing with external models or incorporating domain knowledge.
"""

import pytest
import pandas as pd
import numpy as np
from py_parsnip import manual_reg, linear_reg


@pytest.fixture
def simple_data():
    """Simple regression data"""
    np.random.seed(42)
    data = pd.DataFrame({
        'x1': np.linspace(0, 10, 100),
        'x2': np.linspace(0, 5, 100),
        'y': 2.0 * np.linspace(0, 10, 100) + 3.0 * np.linspace(0, 5, 100) + 10.0 + np.random.normal(0, 0.5, 100)
    })
    return data


@pytest.fixture
def time_series_data():
    """Time series data with date column"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'x': range(100),
        'y': np.linspace(0, 10, 100) + np.random.normal(0, 0.5, 100)
    })
    return data


class TestManualRegSpec:
    """Test manual_reg() specification function"""

    def test_create_basic_spec(self):
        """Test creating basic manual regression spec"""
        spec = manual_reg(
            coefficients={"x1": 2.0, "x2": 3.0},
            intercept=10.0
        )

        assert spec.model_type == 'manual_reg'
        assert spec.engine == 'parsnip'
        assert spec.mode == 'regression'
        assert spec.args['coefficients'] == {"x1": 2.0, "x2": 3.0}
        assert spec.args['intercept'] == 10.0

    def test_default_values(self):
        """Test default values when not specified"""
        spec = manual_reg()

        assert spec.args['coefficients'] == {}
        assert spec.args['intercept'] == 0.0

    def test_validation_coefficients_type(self):
        """Test validation of coefficients type"""
        with pytest.raises(TypeError, match="coefficients must be a dict"):
            manual_reg(coefficients=[1.0, 2.0, 3.0])

        with pytest.raises(TypeError, match="coefficients must be a dict"):
            manual_reg(coefficients="not a dict")

    def test_validation_coefficient_values(self):
        """Test validation of individual coefficient values"""
        with pytest.raises(TypeError, match="must be numeric"):
            manual_reg(coefficients={"x1": "not a number"})

        with pytest.raises(TypeError, match="must be numeric"):
            manual_reg(coefficients={"x1": 2.0, "x2": None})

    def test_validation_intercept_type(self):
        """Test validation of intercept type"""
        with pytest.raises(TypeError, match="intercept must be numeric"):
            manual_reg(intercept="not a number")

        # None defaults to 0.0, so it's valid
        spec = manual_reg(intercept=None)
        assert spec.args['intercept'] == 0.0

    def test_valid_numeric_types(self):
        """Test that both int and float are accepted"""
        # Integer coefficients
        spec1 = manual_reg(coefficients={"x": 2}, intercept=10)
        assert spec1.args['coefficients'] == {"x": 2}
        assert spec1.args['intercept'] == 10.0

        # Float coefficients
        spec2 = manual_reg(coefficients={"x": 2.5}, intercept=10.5)
        assert spec2.args['coefficients'] == {"x": 2.5}
        assert spec2.args['intercept'] == 10.5

        # Mixed
        spec3 = manual_reg(coefficients={"x1": 2, "x2": 3.5}, intercept=10)
        assert spec3.args['coefficients'] == {"x1": 2, "x2": 3.5}


class TestManualRegFit:
    """Test fitting manual regression models"""

    def test_fit_basic(self, simple_data):
        """Test basic fitting with manual coefficients"""
        spec = manual_reg(
            coefficients={"x1": 2.0, "x2": 3.0},
            intercept=10.0
        )

        fit = spec.fit(simple_data, 'y ~ x1 + x2')

        assert fit.fit_data['intercept'] == 10.0
        assert len(fit.fit_data['coefficients']) == 2
        assert 'fitted' in fit.fit_data
        assert len(fit.fit_data['fitted']) == len(simple_data)

    def test_fit_exact_coefficients_match(self, simple_data):
        """Test that exact coefficients produce expected fitted values"""
        # True model: y = 10.0 + 2.0*x1 + 3.0*x2 + noise
        spec = manual_reg(
            coefficients={"x1": 2.0, "x2": 3.0},
            intercept=10.0
        )

        fit = spec.fit(simple_data, 'y ~ x1 + x2')

        # Calculate expected fitted values manually
        expected_fitted = 10.0 + 2.0 * simple_data['x1'].values + 3.0 * simple_data['x2'].values

        np.testing.assert_allclose(fit.fit_data['fitted'], expected_fitted, rtol=1e-10)

    def test_fit_partial_coefficients(self, simple_data):
        """Test fitting with only some coefficients specified"""
        # Only specify x1, x2 defaults to 0.0
        spec = manual_reg(
            coefficients={"x1": 2.0},
            intercept=10.0
        )

        fit = spec.fit(simple_data, 'y ~ x1 + x2')

        # x2 should have coefficient of 0.0
        assert fit.fit_data['coefficients'][0] == 2.0  # x1
        assert fit.fit_data['coefficients'][1] == 0.0  # x2 (default)

    def test_fit_validation_extra_variables(self, simple_data):
        """Test that extra coefficient variables raise error"""
        spec = manual_reg(
            coefficients={"x1": 2.0, "x2": 3.0, "x3": 1.0},  # x3 not in formula
            intercept=10.0
        )

        with pytest.raises(ValueError, match="Coefficients specified for variables not in formula"):
            spec.fit(simple_data, 'y ~ x1 + x2')

    def test_fit_residuals_calculated(self, simple_data):
        """Test that residuals are calculated correctly"""
        spec = manual_reg(
            coefficients={"x1": 2.0, "x2": 3.0},
            intercept=10.0
        )

        fit = spec.fit(simple_data, 'y ~ x1 + x2')

        # Residuals should be actuals - fitted
        expected_residuals = simple_data['y'].values - fit.fit_data['fitted']
        np.testing.assert_allclose(fit.fit_data['residuals'], expected_residuals, rtol=1e-10)


class TestManualRegPredict:
    """Test prediction with manual regression"""

    def test_predict_basic(self, simple_data):
        """Test basic prediction"""
        spec = manual_reg(
            coefficients={"x1": 2.0, "x2": 3.0},
            intercept=10.0
        )

        fit = spec.fit(simple_data, 'y ~ x1 + x2')

        # Predict on test data
        test_data = pd.DataFrame({
            'x1': [15.0, 20.0],
            'x2': [7.5, 10.0]
        })

        predictions = fit.predict(test_data)

        assert '.pred' in predictions.columns
        assert len(predictions) == 2

    def test_predict_exact_values(self):
        """Test that predictions match exact calculation"""
        data = pd.DataFrame({
            'x1': [1.0, 2.0, 3.0],
            'x2': [4.0, 5.0, 6.0],
            'y': [0.0, 0.0, 0.0]  # Dummy values
        })

        spec = manual_reg(
            coefficients={"x1": 2.0, "x2": 3.0},
            intercept=10.0
        )

        fit = spec.fit(data, 'y ~ x1 + x2')

        test_data = pd.DataFrame({
            'x1': [10.0, 20.0],
            'x2': [5.0, 10.0]
        })

        predictions = fit.predict(test_data)

        # Expected: 10.0 + 2.0*x1 + 3.0*x2
        expected = [10.0 + 2.0*10.0 + 3.0*5.0,   # 10 + 20 + 15 = 45
                   10.0 + 2.0*20.0 + 3.0*10.0]  # 10 + 40 + 30 = 80

        np.testing.assert_allclose(predictions['.pred'].values, expected, rtol=1e-10)

    def test_predict_on_training_data(self, simple_data):
        """Test that predicting on training data gives same as fitted"""
        spec = manual_reg(
            coefficients={"x1": 2.0, "x2": 3.0},
            intercept=10.0
        )

        fit = spec.fit(simple_data, 'y ~ x1 + x2')

        # Predict on training data
        predictions = fit.predict(simple_data)

        # Should match fitted values
        np.testing.assert_allclose(
            predictions['.pred'].values,
            fit.fit_data['fitted'],
            rtol=1e-10
        )


class TestManualRegExtractOutputs:
    """Test extract_outputs() for manual regression"""

    def test_extract_outputs_structure(self, simple_data):
        """Test that extract_outputs returns correct structure"""
        spec = manual_reg(
            coefficients={"x1": 2.0, "x2": 3.0},
            intercept=10.0
        )

        fit = spec.fit(simple_data, 'y ~ x1 + x2')
        outputs, coefficients, stats = fit.extract_outputs()

        # Check outputs
        assert isinstance(outputs, pd.DataFrame)
        assert 'actuals' in outputs.columns
        assert 'fitted' in outputs.columns
        assert 'residuals' in outputs.columns
        assert 'split' in outputs.columns

        # Check coefficients
        assert isinstance(coefficients, pd.DataFrame)
        assert 'variable' in coefficients.columns
        assert 'coefficient' in coefficients.columns

        # Check stats
        assert isinstance(stats, pd.DataFrame)
        assert 'metric' in stats.columns
        assert 'value' in stats.columns

    def test_coefficients_include_intercept(self, simple_data):
        """Test that coefficients DataFrame includes intercept"""
        spec = manual_reg(
            coefficients={"x1": 2.5, "x2": -1.3},
            intercept=15.0
        )

        fit = spec.fit(simple_data, 'y ~ x1 + x2')
        _, coefficients, _ = fit.extract_outputs()

        # Check intercept
        intercept_row = coefficients[coefficients['variable'] == 'Intercept']
        assert len(intercept_row) == 1
        assert intercept_row['coefficient'].values[0] == 15.0

    def test_coefficients_values_match(self, simple_data):
        """Test that coefficient values match user input"""
        spec = manual_reg(
            coefficients={"x1": 2.5, "x2": -1.3},
            intercept=15.0
        )

        fit = spec.fit(simple_data, 'y ~ x1 + x2')
        _, coefficients, _ = fit.extract_outputs()

        # Check x1 coefficient
        x1_row = coefficients[coefficients['variable'] == 'x1']
        assert x1_row['coefficient'].values[0] == 2.5

        # Check x2 coefficient
        x2_row = coefficients[coefficients['variable'] == 'x2']
        assert x2_row['coefficient'].values[0] == -1.3

    def test_coefficients_statistical_columns_nan(self, simple_data):
        """Test that statistical inference columns are NaN (not applicable)"""
        spec = manual_reg(
            coefficients={"x1": 2.0},
            intercept=10.0
        )

        fit = spec.fit(simple_data, 'y ~ x1 + x2')
        _, coefficients, _ = fit.extract_outputs()

        # These should be NaN for manual coefficients
        assert coefficients['std_error'].isna().all()
        assert coefficients['t_stat'].isna().all()
        assert coefficients['p_value'].isna().all()
        assert coefficients['ci_0.025'].isna().all()
        assert coefficients['ci_0.975'].isna().all()

    def test_stats_include_metrics(self, simple_data):
        """Test that stats include standard metrics"""
        spec = manual_reg(
            coefficients={"x1": 2.0, "x2": 3.0},
            intercept=10.0
        )

        fit = spec.fit(simple_data, 'y ~ x1 + x2')
        _, _, stats = fit.extract_outputs()

        metric_names = stats['metric'].values
        assert 'rmse' in metric_names
        assert 'mae' in metric_names
        assert 'r_squared' in metric_names
        assert 'model_type' in metric_names

    def test_model_metadata_columns(self, simple_data):
        """Test that all DataFrames have model metadata"""
        spec = manual_reg(
            coefficients={"x1": 2.0},
            intercept=10.0
        )

        fit = spec.fit(simple_data, 'y ~ x1 + x2')
        outputs, coefficients, stats = fit.extract_outputs()

        # All should have metadata columns
        for df in [outputs, coefficients, stats]:
            assert 'model' in df.columns
            assert 'model_group_name' in df.columns
            assert 'group' in df.columns


class TestManualRegUseCases:
    """Test real-world use cases for manual_reg"""

    def test_compare_with_fitted_model(self, simple_data):
        """Test comparing manual vs fitted regression"""
        # Fit a standard linear regression
        fitted_spec = linear_reg()
        fitted_model = fitted_spec.fit(simple_data, 'y ~ x1 + x2')

        # Extract coefficients from fitted model
        _, fitted_coefs, _ = fitted_model.extract_outputs()

        intercept_val = fitted_coefs[fitted_coefs['variable'] == 'Intercept']['coefficient'].values[0]
        x1_coef = fitted_coefs[fitted_coefs['variable'] == 'x1']['coefficient'].values[0]
        x2_coef = fitted_coefs[fitted_coefs['variable'] == 'x2']['coefficient'].values[0]

        # Create manual model with same coefficients
        manual_spec = manual_reg(
            coefficients={"x1": x1_coef, "x2": x2_coef},
            intercept=intercept_val
        )
        manual_model = manual_spec.fit(simple_data, 'y ~ x1 + x2')

        # Predictions should be identical
        test_data = simple_data.iloc[:10]  # Use subset for testing
        fitted_preds = fitted_model.predict(test_data)
        manual_preds = manual_model.predict(test_data)

        np.testing.assert_allclose(
            fitted_preds['.pred'].values,
            manual_preds['.pred'].values,
            rtol=1e-10
        )

    def test_domain_knowledge_coefficients(self, time_series_data):
        """Test using domain knowledge to set coefficients"""
        # Domain expert says: sales increase by 0.5 per day
        spec = manual_reg(
            coefficients={"x": 0.5},
            intercept=5.0
        )

        fit = spec.fit(time_series_data, 'y ~ x')
        outputs, _, stats = fit.extract_outputs()

        # Should produce valid outputs
        assert len(outputs) == len(time_series_data)
        assert 'rmse' in stats['metric'].values

    def test_zero_coefficients_baseline(self, simple_data):
        """Test using all-zero coefficients as baseline"""
        # Baseline: predict intercept only (ignore all predictors)
        spec = manual_reg(
            coefficients={},  # No predictors used
            intercept=simple_data['y'].mean()  # Just use mean
        )

        fit = spec.fit(simple_data, 'y ~ x1 + x2')

        # All predictions should be the intercept (mean)
        predictions = fit.predict(simple_data)
        expected = np.full(len(simple_data), simple_data['y'].mean())

        np.testing.assert_allclose(predictions['.pred'].values, expected, rtol=1e-10)

    def test_external_model_comparison(self):
        """Test comparing with external model coefficients"""
        # Say an external tool gave you these coefficients
        external_coefficients = {
            "temperature": 1.5,
            "humidity": -0.3,
            "wind_speed": 0.2
        }
        external_intercept = 20.0

        # Create data
        data = pd.DataFrame({
            'temperature': [15, 20, 25, 30],
            'humidity': [60, 70, 80, 90],
            'wind_speed': [5, 10, 15, 20],
            'sales': [0, 0, 0, 0]  # Dummy
        })

        # Create manual model with external coefficients
        spec = manual_reg(
            coefficients=external_coefficients,
            intercept=external_intercept
        )

        fit = spec.fit(data, 'sales ~ temperature + humidity + wind_speed')

        # Can now use standard py-tidymodels tools
        outputs, coefficients, stats = fit.extract_outputs()

        assert len(outputs) == len(data)
        assert len(coefficients) == 4  # Intercept + 3 predictors
        assert 'rmse' in stats['metric'].values
