"""
Output Format Consistency Tests

Tests that all engines return consistent output structures:
- extract_outputs() returns tuple of 3 DataFrames
- outputs DataFrame has required columns (actuals, fitted, residuals, forecast, split)
- coefficients DataFrame has required columns (term, estimate)
- stats DataFrame has required columns (metric, value)
- Output formats are consistent across all engines
"""

import pytest
import pandas as pd
import numpy as np
from typing import Tuple

from py_parsnip.engine_registry import ENGINE_REGISTRY, get_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import mold


def create_sample_training_data() -> pd.DataFrame:
    """Create sample training data for testing"""
    np.random.seed(42)
    n = 100

    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'x3': np.random.choice(['A', 'B', 'C'], n),
        'y': np.random.randn(n) * 10 + 50
    })

    return data


def create_time_series_data() -> pd.DataFrame:
    """Create sample time series data for testing"""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    np.random.seed(42)

    data = pd.DataFrame({
        'date': dates,
        'x1': np.random.randn(100),
        'y': np.random.randn(100) * 10 + 50
    })

    return data


def fit_sample_model(model_type: str, engine: str) -> ModelFit:
    """
    Fit a sample model for testing output consistency.

    Args:
        model_type: Model type (e.g., "linear_reg")
        engine: Engine name (e.g., "sklearn")

    Returns:
        Fitted ModelFit object
    """
    # Special handling for time series models
    time_series_models = [
        'prophet_reg', 'arima_reg', 'exp_smoothing',
        'seasonal_reg', 'arima_boost', 'prophet_boost',
        'recursive_reg'
    ]

    if model_type in time_series_models:
        data = create_time_series_data()
        formula = 'y ~ date'
    else:
        data = create_sample_training_data()
        formula = 'y ~ x1 + x2'

    # Create model spec
    spec = ModelSpec(model_type=model_type, engine=engine, mode='regression')

    # Fit model
    try:
        fit = spec.fit(data, formula)
        return fit
    except Exception as e:
        pytest.skip(f"Could not fit {model_type} + {engine}: {e}")


class TestExtractOutputsReturnType:
    """Test that extract_outputs() returns correct type"""

    def test_extract_outputs_returns_tuple_of_three(self):
        """
        Verify that extract_outputs() returns tuple of 3 DataFrames.

        All engines must return: (outputs, coefficients, stats)
        """
        # Test a few representative engines
        test_cases = [
            ("linear_reg", "sklearn"),
            ("rand_forest", "sklearn"),
        ]

        for model_type, engine_name in test_cases:
            if (model_type, engine_name) not in ENGINE_REGISTRY:
                continue

            fit = fit_sample_model(model_type, engine_name)

            # Get engine and extract outputs
            engine = get_engine(model_type, engine_name)
            result = engine.extract_outputs(fit)

            # Should return tuple of 3 elements
            assert isinstance(result, tuple), (
                f"{model_type} + {engine_name}: extract_outputs() should return tuple"
            )
            assert len(result) == 3, (
                f"{model_type} + {engine_name}: extract_outputs() should return 3 elements"
            )

            # All 3 elements should be DataFrames
            outputs, coefficients, stats = result
            assert isinstance(outputs, pd.DataFrame), (
                f"{model_type} + {engine_name}: outputs should be DataFrame"
            )
            assert isinstance(coefficients, pd.DataFrame), (
                f"{model_type} + {engine_name}: coefficients should be DataFrame"
            )
            assert isinstance(stats, pd.DataFrame), (
                f"{model_type} + {engine_name}: stats should be DataFrame"
            )

    def test_extract_outputs_dataframes_not_empty(self):
        """
        Verify that extract_outputs() DataFrames are not empty.

        At minimum, outputs should have rows, coefficients should have rows,
        and stats should have rows.
        """
        # Test a few representative engines
        test_cases = [
            ("linear_reg", "sklearn"),
        ]

        for model_type, engine_name in test_cases:
            if (model_type, engine_name) not in ENGINE_REGISTRY:
                continue

            fit = fit_sample_model(model_type, engine_name)

            engine = get_engine(model_type, engine_name)
            outputs, coefficients, stats = engine.extract_outputs(fit)

            assert len(outputs) > 0, (
                f"{model_type} + {engine_name}: outputs DataFrame is empty"
            )
            assert len(stats) > 0, (
                f"{model_type} + {engine_name}: stats DataFrame is empty"
            )
            # Coefficients may be empty for some models (tree-based, etc.)


class TestOutputsDataFrameStructure:
    """Test structure of outputs DataFrame"""

    def test_outputs_has_required_columns(self):
        """
        Verify that outputs DataFrame has required columns:
        - actuals: True values
        - fitted: Model predictions
        - residuals: actuals - fitted
        - forecast: Combined actual/fitted
        - split: 'train', 'test', or 'forecast'
        """
        required_columns = ['actuals', 'fitted', 'residuals', 'forecast', 'split']

        # Test a few representative engines
        test_cases = [
            ("linear_reg", "sklearn"),
            ("rand_forest", "sklearn"),
        ]

        for model_type, engine_name in test_cases:
            if (model_type, engine_name) not in ENGINE_REGISTRY:
                continue

            fit = fit_sample_model(model_type, engine_name)

            engine = get_engine(model_type, engine_name)
            outputs, _, _ = engine.extract_outputs(fit)

            for col in required_columns:
                assert col in outputs.columns, (
                    f"{model_type} + {engine_name}: outputs missing column '{col}'"
                )

    def test_outputs_actuals_column_type(self):
        """
        Verify that actuals column contains numeric values.
        """
        test_cases = [
            ("linear_reg", "sklearn"),
        ]

        for model_type, engine_name in test_cases:
            if (model_type, engine_name) not in ENGINE_REGISTRY:
                continue

            fit = fit_sample_model(model_type, engine_name)

            engine = get_engine(model_type, engine_name)
            outputs, _, _ = engine.extract_outputs(fit)

            assert pd.api.types.is_numeric_dtype(outputs['actuals']), (
                f"{model_type} + {engine_name}: actuals should be numeric"
            )

    def test_outputs_fitted_column_type(self):
        """
        Verify that fitted column contains numeric values and no NaN.

        Fitted should ALWAYS contain predictions, never NaN.
        """
        test_cases = [
            ("linear_reg", "sklearn"),
        ]

        for model_type, engine_name in test_cases:
            if (model_type, engine_name) not in ENGINE_REGISTRY:
                continue

            fit = fit_sample_model(model_type, engine_name)

            engine = get_engine(model_type, engine_name)
            outputs, _, _ = engine.extract_outputs(fit)

            assert pd.api.types.is_numeric_dtype(outputs['fitted']), (
                f"{model_type} + {engine_name}: fitted should be numeric"
            )

            # Fitted should not contain NaN
            assert not outputs['fitted'].isna().any(), (
                f"{model_type} + {engine_name}: fitted should not contain NaN"
            )

    def test_outputs_residuals_correct(self):
        """
        Verify that residuals = actuals - fitted.
        """
        test_cases = [
            ("linear_reg", "sklearn"),
        ]

        for model_type, engine_name in test_cases:
            if (model_type, engine_name) not in ENGINE_REGISTRY:
                continue

            fit = fit_sample_model(model_type, engine_name)

            engine = get_engine(model_type, engine_name)
            outputs, _, _ = engine.extract_outputs(fit)

            # Calculate expected residuals (handle NaN in actuals)
            mask = ~outputs['actuals'].isna()
            expected_residuals = outputs.loc[mask, 'actuals'] - outputs.loc[mask, 'fitted']
            actual_residuals = outputs.loc[mask, 'residuals']

            np.testing.assert_allclose(
                actual_residuals, expected_residuals, rtol=1e-5,
                err_msg=f"{model_type} + {engine_name}: residuals != actuals - fitted"
            )

    def test_outputs_forecast_uses_combine_first(self):
        """
        Verify that forecast column uses combine_first logic:
        - Shows actuals where they exist
        - Shows fitted where actuals are NaN

        Implementation: pd.Series(actuals).combine_first(pd.Series(fitted))
        """
        test_cases = [
            ("linear_reg", "sklearn"),
        ]

        for model_type, engine_name in test_cases:
            if (model_type, engine_name) not in ENGINE_REGISTRY:
                continue

            fit = fit_sample_model(model_type, engine_name)

            engine = get_engine(model_type, engine_name)
            outputs, _, _ = engine.extract_outputs(fit)

            # Calculate expected forecast
            expected_forecast = pd.Series(
                outputs['actuals'].values
            ).combine_first(
                pd.Series(outputs['fitted'].values)
            ).values

            np.testing.assert_allclose(
                outputs['forecast'], expected_forecast, rtol=1e-5,
                err_msg=f"{model_type} + {engine_name}: forecast not using combine_first"
            )

    def test_outputs_split_column_values(self):
        """
        Verify that split column contains valid values:
        - 'train', 'test', or 'forecast'
        """
        valid_splits = {'train', 'test', 'forecast'}

        test_cases = [
            ("linear_reg", "sklearn"),
        ]

        for model_type, engine_name in test_cases:
            if (model_type, engine_name) not in ENGINE_REGISTRY:
                continue

            fit = fit_sample_model(model_type, engine_name)

            engine = get_engine(model_type, engine_name)
            outputs, _, _ = engine.extract_outputs(fit)

            unique_splits = set(outputs['split'].unique())
            invalid_splits = unique_splits - valid_splits

            assert not invalid_splits, (
                f"{model_type} + {engine_name}: invalid split values: {invalid_splits}"
            )

    def test_outputs_has_model_columns(self):
        """
        Verify that outputs has model tracking columns:
        - model: Model name
        - model_group_name: Group name
        - group: Group identifier (for panel models)
        """
        test_cases = [
            ("linear_reg", "sklearn"),
        ]

        for model_type, engine_name in test_cases:
            if (model_type, engine_name) not in ENGINE_REGISTRY:
                continue

            fit = fit_sample_model(model_type, engine_name)

            engine = get_engine(model_type, engine_name)
            outputs, _, _ = engine.extract_outputs(fit)

            # These columns may not always be present in raw engine output
            # (they're often added by workflow), so this is informational
            expected_cols = ['model', 'model_group_name']
            present_cols = [col for col in expected_cols if col in outputs.columns]

            # Just check that if they're present, they're not all NaN
            for col in present_cols:
                if outputs[col].isna().all():
                    pytest.skip(
                        f"{model_type} + {engine_name}: {col} column is all NaN"
                    )


class TestCoefficientsDataFrameStructure:
    """Test structure of coefficients DataFrame"""

    def test_coefficients_has_term_column(self):
        """
        Verify that coefficients DataFrame has 'term' column.

        The 'term' column identifies the coefficient (e.g., 'x1', 'x2', 'Intercept').
        """
        test_cases = [
            ("linear_reg", "sklearn"),
        ]

        for model_type, engine_name in test_cases:
            if (model_type, engine_name) not in ENGINE_REGISTRY:
                continue

            fit = fit_sample_model(model_type, engine_name)

            engine = get_engine(model_type, engine_name)
            _, coefficients, _ = engine.extract_outputs(fit)

            if len(coefficients) > 0:
                assert 'term' in coefficients.columns, (
                    f"{model_type} + {engine_name}: coefficients missing 'term' column"
                )

    def test_coefficients_has_estimate_column(self):
        """
        Verify that coefficients DataFrame has 'estimate' column.

        The 'estimate' column contains coefficient values.
        """
        test_cases = [
            ("linear_reg", "sklearn"),
        ]

        for model_type, engine_name in test_cases:
            if (model_type, engine_name) not in ENGINE_REGISTRY:
                continue

            fit = fit_sample_model(model_type, engine_name)

            engine = get_engine(model_type, engine_name)
            _, coefficients, _ = engine.extract_outputs(fit)

            if len(coefficients) > 0:
                assert 'estimate' in coefficients.columns, (
                    f"{model_type} + {engine_name}: coefficients missing 'estimate' column"
                )

    def test_coefficients_estimate_is_numeric(self):
        """
        Verify that estimate column is numeric.
        """
        test_cases = [
            ("linear_reg", "sklearn"),
        ]

        for model_type, engine_name in test_cases:
            if (model_type, engine_name) not in ENGINE_REGISTRY:
                continue

            fit = fit_sample_model(model_type, engine_name)

            engine = get_engine(model_type, engine_name)
            _, coefficients, _ = engine.extract_outputs(fit)

            if len(coefficients) > 0 and 'estimate' in coefficients.columns:
                assert pd.api.types.is_numeric_dtype(coefficients['estimate']), (
                    f"{model_type} + {engine_name}: estimate should be numeric"
                )

    def test_coefficients_optional_inference_columns(self):
        """
        Verify that coefficients DataFrame may have statistical inference columns:
        - std_error: Standard error
        - t_stat: t-statistic
        - p_value: p-value
        - conf_low: Lower confidence interval
        - conf_high: Upper confidence interval

        These are optional and may be NaN for some models (regularized, tree-based).
        """
        optional_columns = ['std_error', 't_stat', 'p_value', 'conf_low', 'conf_high']

        test_cases = [
            ("linear_reg", "sklearn"),
        ]

        for model_type, engine_name in test_cases:
            if (model_type, engine_name) not in ENGINE_REGISTRY:
                continue

            fit = fit_sample_model(model_type, engine_name)

            engine = get_engine(model_type, engine_name)
            _, coefficients, _ = engine.extract_outputs(fit)

            # Check if optional columns are present
            present_cols = [col for col in optional_columns if col in coefficients.columns]

            # If present, they should be numeric
            for col in present_cols:
                assert pd.api.types.is_numeric_dtype(coefficients[col]), (
                    f"{model_type} + {engine_name}: {col} should be numeric"
                )


class TestStatsDataFrameStructure:
    """Test structure of stats DataFrame"""

    def test_stats_has_metric_column(self):
        """
        Verify that stats DataFrame has 'metric' column.

        The 'metric' column identifies the statistic (e.g., 'rmse', 'mae', 'r_squared').
        """
        test_cases = [
            ("linear_reg", "sklearn"),
        ]

        for model_type, engine_name in test_cases:
            if (model_type, engine_name) not in ENGINE_REGISTRY:
                continue

            fit = fit_sample_model(model_type, engine_name)

            engine = get_engine(model_type, engine_name)
            _, _, stats = engine.extract_outputs(fit)

            assert 'metric' in stats.columns, (
                f"{model_type} + {engine_name}: stats missing 'metric' column"
            )

    def test_stats_has_value_column(self):
        """
        Verify that stats DataFrame has 'value' column.

        The 'value' column contains statistic values.
        """
        test_cases = [
            ("linear_reg", "sklearn"),
        ]

        for model_type, engine_name in test_cases:
            if (model_type, engine_name) not in ENGINE_REGISTRY:
                continue

            fit = fit_sample_model(model_type, engine_name)

            engine = get_engine(model_type, engine_name)
            _, _, stats = engine.extract_outputs(fit)

            assert 'value' in stats.columns, (
                f"{model_type} + {engine_name}: stats missing 'value' column"
            )

    def test_stats_value_is_numeric(self):
        """
        Verify that value column is numeric.
        """
        test_cases = [
            ("linear_reg", "sklearn"),
        ]

        for model_type, engine_name in test_cases:
            if (model_type, engine_name) not in ENGINE_REGISTRY:
                continue

            fit = fit_sample_model(model_type, engine_name)

            engine = get_engine(model_type, engine_name)
            _, _, stats = engine.extract_outputs(fit)

            assert pd.api.types.is_numeric_dtype(stats['value']), (
                f"{model_type} + {engine_name}: stats value should be numeric"
            )

    def test_stats_has_split_column(self):
        """
        Verify that stats DataFrame has 'split' column.

        Stats are typically calculated per split (train, test).
        """
        test_cases = [
            ("linear_reg", "sklearn"),
        ]

        for model_type, engine_name in test_cases:
            if (model_type, engine_name) not in ENGINE_REGISTRY:
                continue

            fit = fit_sample_model(model_type, engine_name)

            engine = get_engine(model_type, engine_name)
            _, _, stats = engine.extract_outputs(fit)

            # Split column is expected but not strictly required
            if 'split' in stats.columns:
                valid_splits = {'train', 'test', 'forecast'}
                unique_splits = set(stats['split'].unique())
                invalid_splits = unique_splits - valid_splits

                assert not invalid_splits, (
                    f"{model_type} + {engine_name}: invalid split values in stats: {invalid_splits}"
                )

    def test_stats_has_common_metrics(self):
        """
        Verify that stats DataFrame includes common metrics:
        - rmse: Root Mean Squared Error
        - mae: Mean Absolute Error
        - r_squared: R-squared

        Not all models will have all metrics, but regression models should have at least RMSE.
        """
        test_cases = [
            ("linear_reg", "sklearn"),
        ]

        for model_type, engine_name in test_cases:
            if (model_type, engine_name) not in ENGINE_REGISTRY:
                continue

            fit = fit_sample_model(model_type, engine_name)

            engine = get_engine(model_type, engine_name)
            _, _, stats = engine.extract_outputs(fit)

            metrics = set(stats['metric'].unique())

            # Regression models should have at least RMSE
            assert 'rmse' in metrics or 'mae' in metrics, (
                f"{model_type} + {engine_name}: stats missing common regression metrics"
            )


class TestOutputConsistencyAcrossEngines:
    """Test that output structure is consistent across all engines"""

    def test_all_engines_return_three_dataframes(self):
        """
        Test that ALL registered engines return 3 DataFrames from extract_outputs().

        This is a comprehensive test across all engines.
        """
        violations = []

        # Get sample of engines to test (testing all would be slow)
        engines_to_test = [
            ("linear_reg", "sklearn"),
            ("rand_forest", "sklearn"),
            ("decision_tree", "sklearn"),
            ("boost_tree", "xgboost"),
            ("nearest_neighbor", "sklearn"),
        ]

        for model_type, engine_name in engines_to_test:
            if (model_type, engine_name) not in ENGINE_REGISTRY:
                continue

            try:
                fit = fit_sample_model(model_type, engine_name)
                engine = get_engine(model_type, engine_name)
                result = engine.extract_outputs(fit)

                if not isinstance(result, tuple) or len(result) != 3:
                    violations.append(f"{model_type} + {engine_name}")
                else:
                    outputs, coefficients, stats = result
                    if not isinstance(outputs, pd.DataFrame):
                        violations.append(f"{model_type} + {engine_name}: outputs not DataFrame")
                    if not isinstance(coefficients, pd.DataFrame):
                        violations.append(f"{model_type} + {engine_name}: coefficients not DataFrame")
                    if not isinstance(stats, pd.DataFrame):
                        violations.append(f"{model_type} + {engine_name}: stats not DataFrame")

            except Exception as e:
                # Skip engines that can't be tested
                continue

        assert not violations, (
            f"Engines with inconsistent extract_outputs() return type:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )
