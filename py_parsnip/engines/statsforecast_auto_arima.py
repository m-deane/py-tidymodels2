"""
Statsforecast engine for auto ARIMA regression

Automatic ARIMA (AutoRegressive Integrated Moving Average) using statsforecast.
Automatically selects optimal (p,d,q)(P,D,Q)[m] parameters using information criteria.

This engine avoids the numpy 2.x compatibility issues present in pmdarima.
"""

from typing import Dict, Any, Literal
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData
from py_parsnip.utils import _infer_date_column, _parse_ts_formula


@register_engine("arima_reg", "statsforecast")
class StatsforecastAutoARIMAEngine(Engine):
    """
    Statsforecast engine for automatic ARIMA parameter selection.

    Auto ARIMA automatically searches for optimal (p,d,q)(P,D,Q)[m] values
    using information criteria (AIC, BIC, AICc) to balance fit quality and complexity.

    Parameters from model spec are used as search constraints:
    - seasonal_period: Seasonality period (required for seasonal models)
    - non_seasonal_ar: Max AR order to consider (default: 5)
    - non_seasonal_differences: Max differencing order (default: 2)
    - non_seasonal_ma: Max MA order to consider (default: 5)
    - seasonal_ar: Max seasonal AR order (default: 2)
    - seasonal_differences: Max seasonal differencing (default: 1)
    - seasonal_ma: Max seasonal MA order (default: 2)

    Note: These are MAX values - auto_arima will search for optimal within these limits.

    Advantages over pmdarima:
    - Compatible with numpy 2.x
    - Faster fitting for large datasets
    - Better handling of exogenous variables
    """

    param_map = {
        "non_seasonal_ar": "max_p",
        "non_seasonal_differences": "max_d",
        "non_seasonal_ma": "max_q",
        "seasonal_ar": "max_P",
        "seasonal_differences": "max_D",
        "seasonal_ma": "max_Q",
        "seasonal_period": "season_length",
    }

    def fit_raw(
        self, spec: ModelSpec, data: pd.DataFrame, formula: str, date_col: str = None
    ) -> tuple[Dict[str, Any], Any]:
        """
        Fit auto ARIMA model using statsforecast's AutoARIMA.

        Args:
            spec: ModelSpec with ARIMA configuration
            data: Training data DataFrame
            formula: Formula string (e.g., "sales ~ date" or "sales ~ 1")
            date_col: Name of date column, or '__index__' for DatetimeIndex

        Returns:
            Tuple of (fit_data dict, blueprint)

        Note:
            Auto ARIMA automatically selects optimal (p,d,q)(P,D,Q)[m] parameters.
            For univariate ARIMA, use formula like "y ~ 1"
            For ARIMAX with exogenous variables, use "y ~ x1 + x2"
        """
        try:
            from statsforecast.models import AutoARIMA
        except ImportError:
            raise ImportError(
                "statsforecast is not installed. "
                "Install it with: pip install statsforecast"
            )

        # Infer date column from data
        inferred_date_col = _infer_date_column(
            data,
            spec_date_col=spec.args.get("date_col") if spec.args else None,
            fit_date_col=date_col
        )

        # Parse formula to get outcome and exogenous variables
        outcome_name, exog_vars = _parse_ts_formula(formula, inferred_date_col)

        # Validate outcome exists
        if outcome_name not in data.columns:
            raise ValueError(f"Outcome '{outcome_name}' not found in data")

        # Handle __index__ case
        if inferred_date_col == '__index__':
            # Use DatetimeIndex
            y = data[outcome_name]

            # Get exogenous variables if present
            if exog_vars:
                exog = data[exog_vars].values if len(exog_vars) > 1 else data[exog_vars[0]].values.reshape(-1, 1)
            else:
                exog = None
        else:
            # Use date column as index
            y = data.set_index(inferred_date_col)[outcome_name]

            # Get exogenous variables if present (excluding date column)
            if exog_vars:
                exog_df = data.set_index(inferred_date_col)[exog_vars]
                exog = exog_df.values if len(exog_vars) > 1 else exog_df.values.reshape(-1, 1)
            else:
                exog = None

        # Get auto_arima search parameters from spec
        args = spec.args

        # Seasonal period
        seasonal_period = args.get("seasonal_period", 1)
        if seasonal_period is None:
            seasonal_period = 1

        # Determine if seasonal
        seasonal = seasonal_period > 1

        # Max orders (defaults similar to auto_arima defaults)
        max_p = args.get("non_seasonal_ar", 5)
        max_d = args.get("non_seasonal_differences", 2)
        max_q = args.get("non_seasonal_ma", 5)
        max_P = args.get("seasonal_ar", 2) if seasonal else 0
        max_D = args.get("seasonal_differences", 1) if seasonal else 0
        max_Q = args.get("seasonal_ma", 2) if seasonal else 0

        # Create AutoARIMA model with search constraints
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Create model with parameters
            model = AutoARIMA(
                season_length=seasonal_period if seasonal else 1,
                max_p=max_p,
                max_d=max_d,
                max_q=max_q,
                max_P=max_P if seasonal else 0,
                max_D=max_D if seasonal else 0,
                max_Q=max_Q if seasonal else 0,
                seasonal=seasonal,
                ic='aicc',  # Use AICc (default in statsforecast)
                stepwise=True,
                trace=False,
            )

            # Fit model - statsforecast uses different API
            # Note: statsforecast expects y as pandas Series or numpy array
            y_values = y.values if isinstance(y, pd.Series) else y

            # Fit using X for exogenous variables
            if exog is not None:
                fitted_model = model.fit(y=y_values, X=exog)
            else:
                fitted_model = model.fit(y=y_values)

        # Get fitted values (in-sample predictions)
        # statsforecast uses predict_in_sample or forward method
        if hasattr(fitted_model, 'model_'):
            # Get in-sample fitted values
            try:
                fitted_result = fitted_model.predict_in_sample()
                if isinstance(fitted_result, dict):
                    fitted_values = fitted_result['fitted']
                else:
                    fitted_values = fitted_result
            except:
                # Fallback: predict step-by-step (slower but works)
                fitted_values = np.full(len(y_values), np.nan)
                for i in range(1, len(y_values)):
                    try:
                        pred = fitted_model.forecast(h=1, X=exog[i:i+1] if exog is not None else None)
                        fitted_values[i] = pred['mean'][0] if isinstance(pred, dict) else pred[0]
                    except:
                        fitted_values[i] = y_values[i-1]  # Naive fallback
                fitted_values[0] = y_values[0]  # First value
        else:
            # Model not fitted properly, use naive
            fitted_values = np.roll(y_values, 1)
            fitted_values[0] = y_values[0]

        actuals = y_values
        residuals = actuals - fitted_values

        # Extract selected orders (if available)
        # Note: statsforecast doesn't expose order the same way as pmdarima
        # We'll try to extract from model attributes
        try:
            if hasattr(fitted_model, 'model_') and hasattr(fitted_model.model_, 'arma_order'):
                arma_order = fitted_model.model_.arma_order
                # arma_order is (p, q, d) or similar
                order = (arma_order[0], arma_order[2] if len(arma_order) > 2 else 0, arma_order[1])
            else:
                order = (0, 0, 0)  # Unknown

            if hasattr(fitted_model, 'model_') and hasattr(fitted_model.model_, 'seasonal_order'):
                seasonal_order = fitted_model.model_.seasonal_order
            else:
                seasonal_order = (0, 0, 0, seasonal_period if seasonal else 0)
        except:
            order = (0, 0, 0)
            seasonal_order = (0, 0, 0, seasonal_period if seasonal else 0)

        # Extract dates
        if inferred_date_col == '__index__':
            dates = data.index.values
        else:
            dates = data[inferred_date_col].values

        # Create blueprint
        blueprint = {
            "formula": formula,
            "outcome_name": outcome_name,
            "exog_vars": exog_vars,
            "date_col": inferred_date_col,
            "order": order,
            "seasonal_order": seasonal_order,
        }

        # Return fit data
        fit_data = {
            "model": fitted_model,
            "outcome_name": outcome_name,
            "exog_vars": exog_vars,
            "date_col": inferred_date_col,
            "order": order,
            "seasonal_order": seasonal_order,
            "n_obs": len(y),
            "y_train": actuals,
            "fitted": fitted_values,
            "residuals": residuals,
            "dates": dates,
            "engine": "statsforecast",
            "original_training_data": data,
        }

        return fit_data, blueprint

    def predict_raw(
        self,
        fit: ModelFit,
        new_data: pd.DataFrame,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make predictions using fitted AutoARIMA model.

        Args:
            fit: ModelFit with fitted model
            new_data: New data for predictions
            type: Prediction type ("numeric" or "conf_int")

        Returns:
            DataFrame with predictions
        """
        model = fit.fit_data["model"]
        outcome_name = fit.fit_data["outcome_name"]
        exog_vars = fit.fit_data["exog_vars"]
        date_col = fit.fit_data["date_col"]

        # Determine forecast horizon
        h = len(new_data)

        # Get exogenous variables from new_data if needed
        if exog_vars:
            if date_col == '__index__':
                exog = new_data[exog_vars].values if len(exog_vars) > 1 else new_data[exog_vars[0]].values.reshape(-1, 1)
            else:
                exog = new_data[exog_vars].values if len(exog_vars) > 1 else new_data[exog_vars[0]].values.reshape(-1, 1)
        else:
            exog = None

        # Make predictions
        if exog is not None:
            forecast_result = model.predict(h=h, X=exog)
        else:
            forecast_result = model.predict(h=h)

        # Extract forecast values
        if isinstance(forecast_result, dict):
            predictions = forecast_result['mean']
        else:
            predictions = forecast_result

        # Extract date index for predictions
        if date_col == '__index__':
            date_index = new_data.index
        else:
            date_index = new_data[date_col]

        if type == "numeric":
            return pd.DataFrame({".pred": predictions}, index=date_index)
        elif type == "conf_int":
            # Get prediction intervals if available
            if isinstance(forecast_result, dict) and 'lo' in forecast_result and 'hi' in forecast_result:
                return pd.DataFrame({
                    ".pred": predictions,
                    ".pred_lower": forecast_result['lo'],
                    ".pred_upper": forecast_result['hi'],
                }, index=date_index)
            else:
                # No intervals available, return point forecast with NaN intervals
                return pd.DataFrame({
                    ".pred": predictions,
                    ".pred_lower": np.nan,
                    ".pred_upper": np.nan,
                }, index=date_index)
        else:
            raise ValueError(f"type='{type}' not supported for ARIMA. Use 'numeric' or 'conf_int'")

    def fit(
        self, spec: ModelSpec, molded: MoldedData, original_training_data: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """Not used - ARIMA uses raw data path via fit_raw()"""
        raise NotImplementedError("ARIMA uses fit_raw() method, not fit()")

    def predict(
        self,
        fit: ModelFit,
        molded: MoldedData,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """Not used - ARIMA uses raw data path via predict_raw()"""
        raise NotImplementedError("ARIMA uses predict_raw() method, not predict()")

    def _calculate_residual_diagnostics(self, residuals: np.ndarray) -> Dict[str, float]:
        """Calculate residual diagnostics for model validation"""
        import statsmodels.stats.diagnostic as sm_diag
        from scipy import stats as scipy_stats

        results = {}
        n = len(residuals)

        # Ljung-Box test for autocorrelation
        try:
            n_lags = max(1, min(10, n // 5))
            lb_result = sm_diag.acorr_ljungbox(residuals, lags=n_lags)
            results["ljung_box_stat"] = lb_result['lb_stat'].iloc[-1]
            results["ljung_box_p"] = lb_result['lb_pvalue'].iloc[-1]
        except Exception as e:
            results["ljung_box_stat"] = np.nan
            results["ljung_box_p"] = np.nan

        # Shapiro-Wilk test for normality
        if n >= 3:
            shapiro_stat, shapiro_p = scipy_stats.shapiro(residuals)
            results["shapiro_wilk_stat"] = shapiro_stat
            results["shapiro_wilk_p"] = shapiro_p
        else:
            results["shapiro_wilk_stat"] = np.nan
            results["shapiro_wilk_p"] = np.nan

        # Breusch-Pagan not applicable for ARIMA (no exog matrix in OLS format)
        results["breusch_pagan_stat"] = np.nan
        results["breusch_pagan_p"] = np.nan

        return results

    def extract_outputs(
        self, fit: ModelFit
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract comprehensive three-DataFrame output.

        Returns:
            Tuple of (outputs, coefficients, stats)

            - Outputs: Observation-level results with actuals, fitted, residuals
            - Coefficients: ARIMA parameters (ar, ma, seasonal terms)
            - Stats: Comprehensive metrics by split + residual diagnostics
        """
        from py_yardstick import rmse, mae, mape, r_squared

        model = fit.fit_data["model"]

        # ====================
        # 1. OUTPUTS DataFrame
        # ====================
        outputs_list = []

        # Training data
        y_train = fit.fit_data.get("y_train")
        fitted = fit.fit_data.get("fitted")
        residuals = fit.fit_data.get("residuals")
        dates = fit.fit_data.get("dates")

        if y_train is not None and fitted is not None:
            # Create forecast: actuals where they exist, fitted where they don't
            forecast_train = pd.Series(y_train).combine_first(pd.Series(fitted)).values

            train_df = pd.DataFrame({
                "actuals": y_train,
                "fitted": fitted,
                "forecast": forecast_train,
                "residuals": residuals if residuals is not None else y_train - fitted,
                "split": "train",
            })

            # Add date column if available
            if dates is not None and len(dates) == len(train_df):
                train_df.insert(0, 'date', dates)

            # Add model metadata
            train_df["model"] = fit.model_name if fit.model_name else fit.spec.model_type
            train_df["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
            train_df["group"] = "global"

            outputs_list.append(train_df)

        # Test data (if evaluated)
        if "test_predictions" in fit.evaluation_data:
            test_data = fit.evaluation_data["test_data"]
            test_preds = fit.evaluation_data["test_predictions"]
            outcome_col = fit.evaluation_data["outcome_col"]

            test_actuals = test_data[outcome_col].values
            test_predictions = test_preds[".pred"].values
            test_residuals = test_actuals - test_predictions

            # Create forecast
            forecast_test = pd.Series(test_actuals).combine_first(pd.Series(test_predictions)).values

            test_df = pd.DataFrame({
                "actuals": test_actuals,
                "fitted": test_predictions,
                "forecast": forecast_test,
                "residuals": test_residuals,
                "split": "test",
            })

            # Add date column if available
            date_col = fit.fit_data.get("date_col")
            if date_col and date_col in test_data.columns:
                test_df.insert(0, 'date', test_data[date_col].values)
            elif date_col == '__index__':
                test_df.insert(0, 'date', test_data.index.values)

            # Add model metadata
            test_df["model"] = fit.model_name if fit.model_name else fit.spec.model_type
            test_df["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
            test_df["group"] = "global"

            outputs_list.append(test_df)

        outputs = pd.concat(outputs_list, ignore_index=True) if outputs_list else pd.DataFrame()

        # ====================
        # 2. COEFFICIENTS DataFrame
        # ====================
        # Extract ARIMA parameters as "coefficients"
        coefficients_list = []

        order = fit.fit_data.get("order", (0, 0, 0))
        seasonal_order = fit.fit_data.get("seasonal_order", (0, 0, 0, 0))

        # Add order information as hyperparameters
        coefficients_list.extend([
            {
                "variable": "ar_order",
                "coefficient": float(order[0]),
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
            },
            {
                "variable": "diff_order",
                "coefficient": float(order[1]),
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
            },
            {
                "variable": "ma_order",
                "coefficient": float(order[2]),
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
            },
            {
                "variable": "seasonal_ar_order",
                "coefficient": float(seasonal_order[0]),
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
            },
            {
                "variable": "seasonal_diff_order",
                "coefficient": float(seasonal_order[1]),
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
            },
            {
                "variable": "seasonal_ma_order",
                "coefficient": float(seasonal_order[2]),
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
            },
            {
                "variable": "seasonal_period",
                "coefficient": float(seasonal_order[3]) if len(seasonal_order) > 3 else np.nan,
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
            },
        ])

        coefficients = pd.DataFrame(coefficients_list)

        # Add model metadata
        coefficients["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        coefficients["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        coefficients["group"] = "global"

        # ====================
        # 3. STATS DataFrame
        # ====================
        stats_rows = []

        # Training metrics
        if y_train is not None and fitted is not None:
            # Calculate metrics using py_yardstick
            rmse_val = rmse(y_train, fitted)['value'].iloc[0]
            mae_val = mae(y_train, fitted)['value'].iloc[0]
            mape_val = mape(y_train, fitted)['value'].iloc[0]
            r2_val = r_squared(y_train, fitted)['value'].iloc[0]

            stats_rows.extend([
                {"metric": "rmse", "value": rmse_val, "split": "train"},
                {"metric": "mae", "value": mae_val, "split": "train"},
                {"metric": "mape", "value": mape_val, "split": "train"},
                {"metric": "r_squared", "value": r2_val, "split": "train"},
            ])

        # Test metrics (if evaluated)
        if "test_predictions" in fit.evaluation_data:
            test_data = fit.evaluation_data["test_data"]
            test_preds = fit.evaluation_data["test_predictions"]
            outcome_col = fit.evaluation_data["outcome_col"]

            test_actuals = test_data[outcome_col].values
            test_predictions = test_preds[".pred"].values

            test_rmse = rmse(test_actuals, test_predictions)['value'].iloc[0]
            test_mae = mae(test_actuals, test_predictions)['value'].iloc[0]
            test_mape = mape(test_actuals, test_predictions)['value'].iloc[0]
            test_r2 = r_squared(test_actuals, test_predictions)['value'].iloc[0]

            stats_rows.extend([
                {"metric": "rmse", "value": test_rmse, "split": "test"},
                {"metric": "mae", "value": test_mae, "split": "test"},
                {"metric": "mape", "value": test_mape, "split": "test"},
                {"metric": "r_squared", "value": test_r2, "split": "test"},
            ])

        # Model information
        n_obs = fit.fit_data.get("n_obs", 0)
        stats_rows.extend([
            {"metric": "n_obs_train", "value": n_obs, "split": "train"},
            {"metric": "ar_order", "value": float(order[0]), "split": ""},
            {"metric": "diff_order", "value": float(order[1]), "split": ""},
            {"metric": "ma_order", "value": float(order[2]), "split": ""},
            {"metric": "seasonal_period", "value": float(seasonal_order[3]) if len(seasonal_order) > 3 else 0, "split": ""},
        ])

        # Residual diagnostics
        if residuals is not None and len(residuals) > 0:
            diagnostics = self._calculate_residual_diagnostics(residuals)
            for key, value in diagnostics.items():
                stats_rows.append({"metric": key, "value": value, "split": "train"})

        # Add training date range
        train_dates = None
        try:
            from py_parsnip.utils import _infer_date_column

            if fit.fit_data.get("original_training_data") is not None:
                date_col = _infer_date_column(
                    fit.fit_data["original_training_data"],
                    spec_date_col=None,
                    fit_date_col=None
                )

                if date_col == '__index__':
                    train_dates = fit.fit_data["original_training_data"].index.values
                else:
                    train_dates = fit.fit_data["original_training_data"][date_col].values
        except (ValueError, ImportError, KeyError):
            pass

        if train_dates is not None and len(train_dates) > 0:
            stats_rows.extend([
                {"metric": "train_start_date", "value": str(train_dates[0]), "split": "train"},
                {"metric": "train_end_date", "value": str(train_dates[-1]), "split": "train"},
            ])

        stats = pd.DataFrame(stats_rows)

        # Add model metadata
        stats["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        stats["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        stats["group"] = "global"

        return outputs, coefficients, stats
