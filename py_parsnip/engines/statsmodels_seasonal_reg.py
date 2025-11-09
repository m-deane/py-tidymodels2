"""
Statsmodels engine for Seasonal Decomposition models

Implements STL (Seasonal-Trend decomposition using LOESS) followed by
forecasting on the seasonally adjusted series.

Uses statsmodels.tsa.seasonal.STL for decomposition, then fits ETS
to the seasonally adjusted data (trend + remainder).

Supports multiple seasonal periods through nested STL decomposition.
"""

from typing import Dict, Any, Literal, Optional
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData
from py_parsnip.utils.time_series_utils import _infer_date_column, _parse_ts_formula, _expand_dot_notation


@register_engine("seasonal_reg", "statsmodels")
class StatsmodelsSeasonalRegEngine(Engine):
    """
    Statsmodels engine for Seasonal Decomposition models.

    Uses STL decomposition followed by ETS forecasting.

    Process:
    1. Decompose series using STL: y = trend + seasonal + remainder
    2. Fit ETS to seasonally adjusted series (trend + remainder)
    3. Forecast = ETS forecast + seasonal component (last cycle repeated)

    For multiple seasonal periods, applies nested STL decomposition.
    """

    param_map = {
        "seasonal_period_1": "period",
        "seasonal_period_2": "period_2",
        "seasonal_period_3": "period_3",
    }

    def fit_raw(
        self, spec: ModelSpec, data: pd.DataFrame, formula: str, date_col: str
    ) -> tuple[Dict[str, Any], Any]:
        """
        Fit Seasonal Decomposition model using raw data (bypasses hardhat molding).

        Args:
            spec: ModelSpec with seasonal decomposition configuration
            data: Training data DataFrame
            formula: Formula string (e.g., "sales ~ date" or "sales ~ 1")
            date_col: Name of date column or '__index__' for DatetimeIndex

        Returns:
            Tuple of (fit_data dict, blueprint)

        Note:
            Seasonal decomposition is univariate - no exogenous variables.
            Formula should be "y ~ 1" or "y ~ date" (date used for index only).
        """
        from statsmodels.tsa.seasonal import STL
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        # Parse formula to extract outcome (exog_vars will be empty for STL)
        outcome_name, exog_vars = _parse_ts_formula(formula, date_col)

        # Expand "." notation to all columns except outcome and date
        exog_vars = _expand_dot_notation(exog_vars, data, outcome_name, date_col)

        # Validate outcome exists
        if outcome_name not in data.columns:
            raise ValueError(f"Outcome '{outcome_name}' not found in data")

        # Get outcome series
        y = data[outcome_name]

        # Handle time index
        if date_col == '__index__':
            # Data is already indexed by datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError(
                    f"date_col is '__index__' but data does not have DatetimeIndex. "
                    f"Got index type: {type(data.index).__name__}"
                )
            y = data[outcome_name]
        elif date_col is not None:
            # Set datetime column as index
            y = data.set_index(date_col)[outcome_name]

        # Get parameters
        args = spec.args
        seasonal_period_1 = args.get("seasonal_period_1")
        seasonal_period_2 = args.get("seasonal_period_2")
        seasonal_period_3 = args.get("seasonal_period_3")

        if seasonal_period_1 is None:
            raise ValueError("seasonal_period_1 must be specified")

        # Check minimum length requirement
        min_length = 2 * seasonal_period_1
        if len(y) < min_length:
            raise ValueError(
                f"Time series too short for STL decomposition. "
                f"Need at least {min_length} observations (2 full cycles), got {len(y)}"
            )

        # STL requires seasonal parameter to be odd and >= 3
        # If even, make it odd by adding 1
        stl_seasonal_1 = seasonal_period_1 if seasonal_period_1 % 2 == 1 else seasonal_period_1 + 1
        if stl_seasonal_1 < 3:
            stl_seasonal_1 = 3

        # Perform STL decomposition on primary seasonal period
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stl = STL(y, seasonal=stl_seasonal_1, trend=None)  # Auto-select trend window
            stl_result = stl.fit()

        # Extract components
        trend = stl_result.trend
        seasonal_1 = stl_result.seasonal
        remainder = stl_result.resid

        # Handle multiple seasonal periods through nested decomposition
        seasonal_2 = None
        seasonal_3 = None

        if seasonal_period_2 is not None:
            # Decompose the remainder for second seasonal period
            if len(y) >= 2 * seasonal_period_2:
                # Ensure odd and >= 3
                stl_seasonal_2 = seasonal_period_2 if seasonal_period_2 % 2 == 1 else seasonal_period_2 + 1
                if stl_seasonal_2 < 3:
                    stl_seasonal_2 = 3

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stl2 = STL(remainder, seasonal=stl_seasonal_2, trend=None)
                    stl_result2 = stl2.fit()
                    seasonal_2 = stl_result2.seasonal
                    remainder = stl_result2.resid  # Update remainder

        if seasonal_period_3 is not None and seasonal_2 is not None:
            # Decompose the remainder for third seasonal period
            if len(y) >= 2 * seasonal_period_3:
                # Ensure odd and >= 3
                stl_seasonal_3 = seasonal_period_3 if seasonal_period_3 % 2 == 1 else seasonal_period_3 + 1
                if stl_seasonal_3 < 3:
                    stl_seasonal_3 = 3

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stl3 = STL(remainder, seasonal=stl_seasonal_3, trend=None)
                    stl_result3 = stl3.fit()
                    seasonal_3 = stl_result3.seasonal
                    remainder = stl_result3.resid  # Update remainder

        # Create seasonally adjusted series (trend + remainder)
        seasonally_adjusted = trend + remainder

        # Fit ETS model to seasonally adjusted series
        # Use simple exponential smoothing with trend (no seasonality, already removed)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ets_model = ExponentialSmoothing(
                    seasonally_adjusted,
                    trend="add",  # Include trend
                    seasonal=None,  # No seasonality (already decomposed)
                )
                ets_fit = ets_model.fit(optimized=True, use_brute=False)
        except Exception:
            # Fallback: simple ES with no trend
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ets_model = ExponentialSmoothing(
                    seasonally_adjusted,
                    trend=None,
                    seasonal=None,
                )
                ets_fit = ets_model.fit()

        # Get fitted values for seasonally adjusted series
        fitted_sa = ets_fit.fittedvalues.values

        # Reconstruct fitted values by adding back seasonality
        fitted_values = fitted_sa + seasonal_1.values
        if seasonal_2 is not None:
            fitted_values = fitted_values + seasonal_2.values
        if seasonal_3 is not None:
            fitted_values = fitted_values + seasonal_3.values

        # Calculate residuals
        actuals = y.values if isinstance(y, pd.Series) else y
        residuals = actuals - fitted_values

        # Extract dates
        if date_col == '__index__':
            dates = data.index.values
        elif date_col is not None:
            dates = data[date_col].values
        else:
            # Fallback to integer index
            dates = np.arange(len(y))

        # Create blueprint
        blueprint = {
            "formula": formula,
            "outcome_name": outcome_name,
            "date_col": date_col,
            "seasonal_period_1": seasonal_period_1,
            "seasonal_period_2": seasonal_period_2,
            "seasonal_period_3": seasonal_period_3,
        }

        # Return fit data
        fit_data = {
            "model": ets_fit,  # For extract_fit_engine() compatibility
            "ets_model": ets_fit,  # Fitted ETS model on seasonally adjusted series
            "stl_result": stl_result,  # STL decomposition result
            "outcome_name": outcome_name,
            "seasonal_period_1": seasonal_period_1,
            "seasonal_period_2": seasonal_period_2,
            "seasonal_period_3": seasonal_period_3,
            "seasonal_1": seasonal_1.values,  # Store seasonal components
            "seasonal_2": seasonal_2.values if seasonal_2 is not None else None,
            "seasonal_3": seasonal_3.values if seasonal_3 is not None else None,
            "trend": trend.values,
            "remainder": remainder.values,
            "n_obs": len(y),
            "y_train": actuals,
            "fitted": fitted_values,
            "residuals": residuals,
            "dates": dates,
        }

        return fit_data, blueprint

    def fit(self, spec: ModelSpec, molded: MoldedData) -> Dict[str, Any]:
        """Not used for Seasonal Decomposition - use fit_raw() instead"""
        raise NotImplementedError("Seasonal Decomposition uses fit_raw() instead of fit()")

    def predict(
        self, fit: ModelFit, molded: MoldedData, type: str
    ) -> pd.DataFrame:
        """Not used for Seasonal Decomposition - use predict_raw() instead"""
        raise NotImplementedError("Seasonal Decomposition uses predict_raw() instead of predict()")

    def predict_raw(
        self,
        fit: ModelFit,
        new_data: pd.DataFrame,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make predictions using fitted Seasonal Decomposition model.

        Args:
            fit: ModelFit with fitted model
            new_data: DataFrame with forecast horizon (length determines forecast periods)
            type: Prediction type
                - "numeric": Point forecasts
                - "conf_int": Forecasts with prediction intervals

        Returns:
            DataFrame with predictions

        Note:
            Forecast = ETS forecast (on seasonally adjusted) + seasonal components
            Seasonal components are repeated from the last cycle
        """
        if type not in ("numeric", "conf_int"):
            raise ValueError(
                f"seasonal_reg supports type='numeric' or 'conf_int', got '{type}'"
            )

        ets_model = fit.fit_data["ets_model"]
        seasonal_1 = fit.fit_data["seasonal_1"]
        seasonal_2 = fit.fit_data["seasonal_2"]
        seasonal_3 = fit.fit_data["seasonal_3"]
        seasonal_period_1 = fit.fit_data["seasonal_period_1"]
        seasonal_period_2 = fit.fit_data["seasonal_period_2"]
        seasonal_period_3 = fit.fit_data["seasonal_period_3"]

        # Determine forecast horizon
        n_periods = len(new_data)

        # Get date column from blueprint (if present)
        date_col = fit.blueprint.get("date_col") if isinstance(fit.blueprint, dict) else None
        date_index = new_data[date_col] if date_col and date_col in new_data.columns else None

        # Forecast seasonally adjusted series using ETS
        forecast_sa = ets_model.forecast(steps=n_periods)

        # Add back seasonal components (repeat last cycle)
        def extend_seasonal(seasonal_values: np.ndarray, period: int, n_ahead: int) -> np.ndarray:
            """Extend seasonal component by repeating last cycle"""
            # Get last complete cycle
            last_cycle = seasonal_values[-period:]
            # Repeat for forecast horizon
            n_cycles = int(np.ceil(n_ahead / period))
            extended = np.tile(last_cycle, n_cycles)[:n_ahead]
            return extended

        # Extend primary seasonal component
        seasonal_1_forecast = extend_seasonal(seasonal_1, seasonal_period_1, n_periods)
        forecast = forecast_sa.values + seasonal_1_forecast

        # Add secondary seasonal component if present
        if seasonal_2 is not None and seasonal_period_2 is not None:
            seasonal_2_forecast = extend_seasonal(seasonal_2, seasonal_period_2, n_periods)
            forecast = forecast + seasonal_2_forecast

        # Add tertiary seasonal component if present
        if seasonal_3 is not None and seasonal_period_3 is not None:
            seasonal_3_forecast = extend_seasonal(seasonal_3, seasonal_period_3, n_periods)
            forecast = forecast + seasonal_3_forecast

        # Make predictions
        if type == "numeric":
            # Point forecasts
            result = pd.DataFrame({".pred": forecast})
            if date_index is not None:
                result.index = date_index
            return result
        else:  # conf_int
            # Prediction intervals
            # Use residual std as approximation (more sophisticated would use simulation)
            if "residuals" in fit.fit_data and len(fit.fit_data["residuals"]) > 0:
                resid_std = np.std(fit.fit_data["residuals"])
                # 95% intervals (1.96 * std), increasing with horizon
                margin = 1.96 * resid_std * np.sqrt(np.arange(1, n_periods + 1))
            else:
                # Fallback: use 10% of forecast value
                margin = 0.1 * np.abs(forecast)

            result = pd.DataFrame({
                ".pred": forecast,
                ".pred_lower": forecast - margin,
                ".pred_upper": forecast + margin,
            })
            if date_index is not None:
                result.index = date_index
            return result

    def _calculate_metrics(self, actuals: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics (same as ARIMA/Prophet)"""
        residuals = actuals - predictions
        n = len(actuals)

        # RMSE
        rmse = np.sqrt(np.mean(residuals ** 2))

        # MAE
        mae = np.mean(np.abs(residuals))

        # MAPE (avoid division by zero)
        mask = actuals != 0
        mape = np.mean(np.abs(residuals[mask] / actuals[mask]) * 100) if mask.any() else np.nan

        # SMAPE
        smape = np.mean(2 * np.abs(residuals) / (np.abs(actuals) + np.abs(predictions)) * 100)

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

        # MDA (Mean Directional Accuracy)
        if n > 1:
            actual_direction = np.diff(actuals) > 0
            pred_direction = np.diff(predictions) > 0
            mda = np.mean(actual_direction == pred_direction) * 100
        else:
            mda = np.nan

        return {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "smape": smape,
            "r_squared": r_squared,
            "mda": mda,
        }

    def _calculate_residual_diagnostics(self, residuals: np.ndarray) -> Dict[str, float]:
        """Calculate residual diagnostic statistics"""
        from scipy import stats as scipy_stats
        import statsmodels.stats.diagnostic as sm_diag

        results = {}
        n = len(residuals)

        # Durbin-Watson statistic
        if n > 1:
            diff_resid = np.diff(residuals)
            dw = np.sum(diff_resid ** 2) / np.sum(residuals ** 2)
            results["durbin_watson"] = dw
        else:
            results["durbin_watson"] = np.nan

        # Shapiro-Wilk test for normality
        if n >= 3:
            shapiro_stat, shapiro_p = scipy_stats.shapiro(residuals)
            results["shapiro_wilk_stat"] = shapiro_stat
            results["shapiro_wilk_p"] = shapiro_p
        else:
            results["shapiro_wilk_stat"] = np.nan
            results["shapiro_wilk_p"] = np.nan

        # Ljung-Box test for autocorrelation
        try:
            # Ensure we have enough lags (at least 1, max 10 or n//5)
            n_lags = max(1, min(10, n // 5))
            lb_result = sm_diag.acorr_ljungbox(residuals, lags=n_lags)
            # Returns DataFrame with columns 'lb_stat' and 'lb_pvalue'
            results["ljung_box_stat"] = lb_result['lb_stat'].iloc[-1]  # Last lag statistic
            results["ljung_box_p"] = lb_result['lb_pvalue'].iloc[-1]  # Last lag p-value
        except Exception as e:
            # Not enough data or other issue
            results["ljung_box_stat"] = np.nan
            results["ljung_box_p"] = np.nan

        # Breusch-Pagan test not applicable for STL decomposition (no exog matrix)
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

            - Outputs: Observation-level results with actuals, fitted, residuals, dates,
                      plus decomposed components (trend, seasonal, remainder)
            - Coefficients: ETS smoothing parameters from the model fit to adjusted series
            - Stats: Comprehensive metrics by split + residual diagnostics + model info
        """
        from scipy import stats as scipy_stats

        ets_model = fit.fit_data["ets_model"]
        stl_result = fit.fit_data["stl_result"]

        # ====================
        # 1. OUTPUTS DataFrame
        # ====================
        outputs_list = []

        # Training data
        y_train = fit.fit_data.get("y_train")
        fitted = fit.fit_data.get("fitted")
        residuals = fit.fit_data.get("residuals")
        dates = fit.fit_data.get("dates")

        # Decomposed components
        trend = fit.fit_data.get("trend")
        seasonal_1 = fit.fit_data.get("seasonal_1")
        seasonal_2 = fit.fit_data.get("seasonal_2")
        seasonal_3 = fit.fit_data.get("seasonal_3")
        remainder = fit.fit_data.get("remainder")

        if y_train is not None and fitted is not None:
            # Create forecast: actuals where they exist, fitted where they don't
            forecast_train = pd.Series(y_train).combine_first(pd.Series(fitted)).values

            train_df = pd.DataFrame({
                "date": dates if dates is not None else np.arange(len(y_train)),
                "actuals": y_train,
                "fitted": fitted,  # In-sample predictions
                "forecast": forecast_train,  # Actuals where available, fitted otherwise
                "residuals": residuals if residuals is not None else y_train - fitted,
                "trend": trend if trend is not None else np.nan,
                "seasonal": seasonal_1 if seasonal_1 is not None else np.nan,
                "remainder": remainder if remainder is not None else np.nan,
                "split": "train",
            })

            # Add secondary seasonal components if present
            if seasonal_2 is not None:
                train_df["seasonal_2"] = seasonal_2
            if seasonal_3 is not None:
                train_df["seasonal_3"] = seasonal_3

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

            # Try to get dates from test data
            test_dates = None
            if dates is not None:
                # Try to find date column in test data
                for col in test_data.columns:
                    if pd.api.types.is_datetime64_any_dtype(test_data[col]):
                        test_dates = test_data[col].values
                        break
            if test_dates is None:
                test_dates = np.arange(len(test_actuals))

            # Create forecast: actuals where they exist, fitted where they don't
            forecast_test = pd.Series(test_actuals).combine_first(pd.Series(test_predictions)).values

            test_df = pd.DataFrame({
                "date": test_dates,
                "actuals": test_actuals,
                "fitted": test_predictions,  # Out-of-sample predictions
                "forecast": forecast_test,  # Actuals where available, fitted otherwise
                "residuals": test_residuals,
                "split": "test",
            })

            # Add model metadata
            test_df["model"] = fit.model_name if fit.model_name else fit.spec.model_type
            test_df["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
            test_df["group"] = "global"

            outputs_list.append(test_df)

        outputs = pd.concat(outputs_list, ignore_index=True) if outputs_list else pd.DataFrame()

        # ====================
        # 2. COEFFICIENTS DataFrame
        # ====================
        coef_rows = []

        # Extract ETS smoothing parameters from fitted model
        # Alpha (level smoothing parameter)
        if hasattr(ets_model, 'params') and 'smoothing_level' in ets_model.params:
            alpha = ets_model.params['smoothing_level']
            coef_rows.append({
                "variable": "alpha (smoothing_level)",
                "coefficient": alpha,
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
            })

        # Beta (trend smoothing parameter)
        if hasattr(ets_model, 'params') and 'smoothing_trend' in ets_model.params:
            beta = ets_model.params['smoothing_trend']
            coef_rows.append({
                "variable": "beta (smoothing_trend)",
                "coefficient": beta,
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
            })

        # STL strength of seasonality and trend
        if hasattr(stl_result, 'seasonal') and hasattr(stl_result, 'trend'):
            # Calculate strength metrics
            var_remainder = np.var(stl_result.resid)
            var_seasonal_adj = np.var(stl_result.trend + stl_result.resid)

            if var_seasonal_adj > 0:
                strength_seasonality = 1 - (var_remainder / var_seasonal_adj)
                coef_rows.append({
                    "variable": "strength_seasonality",
                    "coefficient": max(0, strength_seasonality),  # Clamp to [0, 1]
                    "std_error": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "ci_0.025": np.nan,
                    "ci_0.975": np.nan,
                    "vif": np.nan,
                })

        coefficients = pd.DataFrame(coef_rows) if coef_rows else pd.DataFrame({
            "variable": [],
            "coefficient": [],
            "std_error": [],
            "t_stat": [],
            "p_value": [],
            "ci_0.025": [],
            "ci_0.975": [],
            "vif": [],
        })

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
            train_metrics = self._calculate_metrics(y_train, fitted)

            for metric_name, value in train_metrics.items():
                stats_rows.append({
                    "metric": metric_name,
                    "value": value,
                    "split": "train",
                })

        # Test metrics (if evaluated)
        if "test_predictions" in fit.evaluation_data:
            test_actuals = test_data[outcome_col].values
            test_forecast = test_preds[".pred"].values

            test_metrics = self._calculate_metrics(test_actuals, test_forecast)

            for metric_name, value in test_metrics.items():
                stats_rows.append({
                    "metric": metric_name,
                    "value": value,
                    "split": "test",
                })

        # Residual diagnostics (on training data)
        if residuals is not None and len(residuals) > 0:
            resid_diag = self._calculate_residual_diagnostics(residuals)
            for metric_name, value in resid_diag.items():
                stats_rows.append({
                    "metric": metric_name,
                    "value": value,
                    "split": "train",
                })

        # Model information
        blueprint = fit.blueprint
        seasonal_period_1 = fit.fit_data.get("seasonal_period_1")
        seasonal_period_2 = fit.fit_data.get("seasonal_period_2")
        seasonal_period_3 = fit.fit_data.get("seasonal_period_3")

        stats_rows.extend([
            {"metric": "formula", "value": blueprint.get("formula", "") if isinstance(blueprint, dict) else "", "split": ""},
            {"metric": "model_type", "value": fit.spec.model_type, "split": ""},
            {"metric": "decomposition", "value": "STL", "split": ""},
            {"metric": "forecasting_model", "value": "ETS", "split": ""},
            {"metric": "seasonal_period_1", "value": seasonal_period_1 if seasonal_period_1 is not None else 0, "split": ""},
            {"metric": "seasonal_period_2", "value": seasonal_period_2 if seasonal_period_2 is not None else 0, "split": ""},
            {"metric": "seasonal_period_3", "value": seasonal_period_3 if seasonal_period_3 is not None else 0, "split": ""},
            {"metric": "aic", "value": ets_model.aic if hasattr(ets_model, "aic") else np.nan, "split": ""},
            {"metric": "bic", "value": ets_model.bic if hasattr(ets_model, "bic") else np.nan, "split": ""},
            {"metric": "n_obs_train", "value": fit.fit_data.get("n_obs", 0), "split": "train"},
        ])

        # Add dates if available
        if dates is not None and len(dates) > 0:
            stats_rows.extend([
                {"metric": "train_start_date", "value": str(dates[0]), "split": "train"},
                {"metric": "train_end_date", "value": str(dates[-1]), "split": "train"},
            ])

        stats = pd.DataFrame(stats_rows)

        # Add model metadata
        stats["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        stats["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        stats["group"] = "global"

        return outputs, coefficients, stats
