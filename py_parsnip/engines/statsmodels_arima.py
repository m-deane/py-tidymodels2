"""
Statsmodels engine for ARIMA regression

SARIMAX (Seasonal ARIMA with eXogenous variables) implementation.
Handles both non-seasonal and seasonal ARIMA models.
"""

from typing import Dict, Any, Literal
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData
from py_parsnip.utils import _infer_date_column, _parse_ts_formula, _expand_dot_notation


@register_engine("arima_reg", "statsmodels")
class StatsmodelsARIMAEngine(Engine):
    """
    Statsmodels engine for ARIMA models.

    Parameter mapping: tidymodels → statsmodels
    - non_seasonal_ar → p (in order tuple)
    - non_seasonal_differences → d (in order tuple)
    - non_seasonal_ma → q (in order tuple)
    - seasonal_ar → P (in seasonal_order tuple)
    - seasonal_differences → D (in seasonal_order tuple)
    - seasonal_ma → Q (in seasonal_order tuple)
    - seasonal_period → m (in seasonal_order tuple)
    """

    param_map = {
        "non_seasonal_ar": "p",
        "non_seasonal_differences": "d",
        "non_seasonal_ma": "q",
        "seasonal_ar": "P",
        "seasonal_differences": "D",
        "seasonal_ma": "Q",
        "seasonal_period": "m",
    }

    def fit_raw(
        self, spec: ModelSpec, data: pd.DataFrame, formula: str, date_col: str = None
    ) -> tuple[Dict[str, Any], Any]:
        """
        Fit ARIMA model using raw data (bypasses hardhat molding).

        Args:
            spec: ModelSpec with ARIMA configuration
            data: Training data DataFrame
            formula: Formula string (e.g., "sales ~ date" or "sales ~ 1")
            date_col: Inferred date column name from ModelSpec (can be '__index__' for DatetimeIndex)

        Returns:
            Tuple of (fit_data dict, blueprint)

        Note:
            ARIMA requires proper time series data structure.
            For univariate ARIMA, use formula like "y ~ 1"
            For ARIMAX with exogenous variables, use "y ~ x1 + x2"
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        # Use provided date_col if given (already inferred by ModelSpec),
        # otherwise infer it here
        if date_col is None:
            inferred_date_col = _infer_date_column(
                data,
                spec_date_col=None,
                fit_date_col=None
            )
        else:
            # date_col was already inferred by ModelSpec, use it directly
            inferred_date_col = date_col

        # Parse formula to extract outcome and exogenous variables
        outcome_name, exog_vars = _parse_ts_formula(formula, inferred_date_col)

        # Expand "." notation to all columns except outcome and date
        exog_vars = _expand_dot_notation(exog_vars, data, outcome_name, inferred_date_col)

        # Validate outcome exists
        if outcome_name not in data.columns:
            raise ValueError(f"Outcome '{outcome_name}' not found in data")

        # Handle DatetimeIndex vs regular date column
        if inferred_date_col == '__index__':
            # Use existing DatetimeIndex
            y = data[outcome_name]
            if exog_vars:
                exog = data[exog_vars]
            else:
                exog = None
        else:
            # Set date column as index
            y = data.set_index(inferred_date_col)[outcome_name]
            if exog_vars:
                exog = data.set_index(inferred_date_col)[exog_vars]
            else:
                exog = None

        # Build order and seasonal_order tuples
        args = spec.args
        order = (
            args.get("non_seasonal_ar", 0),
            args.get("non_seasonal_differences", 0),
            args.get("non_seasonal_ma", 0),
        )

        seasonal_period = args.get("seasonal_period", 0)
        if seasonal_period is None:
            seasonal_period = 0

        seasonal_order = (
            args.get("seasonal_ar", 0),
            args.get("seasonal_differences", 0),
            args.get("seasonal_ma", 0),
            seasonal_period,
        )

        # Create and fit SARIMAX model
        model = SARIMAX(
            y, exog=exog, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False
        )

        # Fit with minimal output
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_model = model.fit(disp=False)

        # Get fitted values and residuals
        fitted_values = fitted_model.fittedvalues.values
        actuals = y.values if isinstance(y, pd.Series) else y
        residuals = fitted_model.resid.values

        # Extract dates for outputs DataFrame
        if inferred_date_col == '__index__':
            dates = data.index.values
        else:
            dates = data[inferred_date_col].values

        # Create blueprint
        blueprint = {
            "formula": formula,
            "outcome_name": outcome_name,
            "exog_vars": exog_vars,  # Store exogenous variables (excluding date)
            "date_col": inferred_date_col,  # Store '__index__' or column name
            "order": order,
            "seasonal_order": seasonal_order,
        }

        # Return fit data
        fit_data = {
            "model": fitted_model,
            "outcome_name": outcome_name,
            "exog_vars": exog_vars,  # Store for prediction validation
            "date_col": inferred_date_col,  # Store for prediction
            "order": order,
            "seasonal_order": seasonal_order,
            "n_obs": len(y),
            "y_train": actuals,
            "fitted": fitted_values,
            "residuals": residuals,
            "dates": dates,
        }

        return fit_data, blueprint

    def fit(self, spec: ModelSpec, molded: MoldedData) -> Dict[str, Any]:
        """Not used for ARIMA - use fit_raw() instead"""
        raise NotImplementedError("ARIMA uses fit_raw() instead of fit()")

    def predict(
        self, fit: ModelFit, molded: MoldedData, type: str
    ) -> pd.DataFrame:
        """Not used for ARIMA - use predict_raw() instead"""
        raise NotImplementedError("ARIMA uses predict_raw() instead of predict()")

    def predict_raw(
        self,
        fit: ModelFit,
        new_data: pd.DataFrame,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make predictions using fitted ARIMA model.

        Args:
            fit: ModelFit with fitted ARIMA model
            new_data: DataFrame with same structure as training
                      (Must have exogenous variables if used in training)
            type: Prediction type
                - "numeric": Point forecasts
                - "conf_int": Forecasts with prediction intervals

        Returns:
            DataFrame with predictions indexed by date

        Note:
            For pure ARIMA (no exog), new_data just needs length and date.
            For ARIMAX, new_data must contain exogenous variables.
        """
        if type not in ("numeric", "conf_int"):
            raise ValueError(
                f"arima_reg supports type='numeric' or 'conf_int', got '{type}'"
            )

        model = fit.fit_data["model"]
        exog_vars = fit.fit_data.get("exog_vars", [])
        fit_date_col = fit.fit_data.get("date_col")

        # Infer date column from new_data (prioritize fit_date_col for consistency)
        inferred_date_col = _infer_date_column(
            new_data,
            spec_date_col=None,
            fit_date_col=fit_date_col
        )

        # Determine forecast horizon
        n_periods = len(new_data)

        # Get exogenous variables if used during training
        if exog_vars:
            # Validate exog columns exist in new_data
            missing = [v for v in exog_vars if v not in new_data.columns]
            if missing:
                raise ValueError(
                    f"Exogenous variables {missing} not found in new_data. "
                    f"Required: {exog_vars}"
                )
            exog = new_data[exog_vars]
        else:
            exog = None

        # Extract date index for prediction DataFrame
        if inferred_date_col == '__index__':
            date_index = new_data.index
        else:
            date_index = new_data[inferred_date_col]

        # Make predictions
        if type == "numeric":
            # Point forecasts
            forecast = model.forecast(steps=n_periods, exog=exog)
            result = pd.DataFrame({".pred": forecast.values})
            if date_index is not None:
                result.index = date_index
            return result
        else:  # conf_int
            # Get forecast with prediction intervals
            forecast_obj = model.get_forecast(steps=n_periods, exog=exog)
            pred_mean = forecast_obj.predicted_mean
            pred_int = forecast_obj.conf_int(alpha=0.05)  # 95% intervals

            result = pd.DataFrame({
                ".pred": pred_mean.values,
                ".pred_lower": pred_int.iloc[:, 0].values,
                ".pred_upper": pred_int.iloc[:, 1].values,
            })
            if date_index is not None:
                result.index = date_index
            return result

    def _calculate_metrics(self, actuals: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics (same as sklearn/Prophet)"""
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

        # Breusch-Pagan test not applicable for ARIMA (no exog matrix in same format)
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

            - Outputs: Observation-level results with actuals, fitted, residuals, dates
            - Coefficients: ARIMA parameters with statistical inference
            - Stats: Comprehensive metrics by split + residual diagnostics + model info
        """
        from scipy import stats as scipy_stats

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
                "date": dates if dates is not None else np.arange(len(y_train)),
                "actuals": y_train,
                "fitted": fitted,  # In-sample predictions
                "forecast": forecast_train,  # Actuals where available, fitted otherwise
                "residuals": residuals if residuals is not None else y_train - fitted,
                "split": "train",
            })

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

            # Extract test dates using date_col from fit_data
            date_col = fit.fit_data.get("date_col")
            if date_col == '__index__':
                test_dates = test_data.index.values
            elif date_col and date_col in test_data.columns:
                test_dates = test_data[date_col].values
            else:
                # Fallback to sequence if no date column found
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
        if hasattr(model, "params") and model.params is not None:
            param_names = model.param_names if hasattr(model, "param_names") else list(model.params.index)

            # Get standard errors, t-stats, p-values
            std_errors = model.bse.values if hasattr(model, "bse") else [np.nan] * len(param_names)
            t_stats = model.tvalues.values if hasattr(model, "tvalues") else [np.nan] * len(param_names)
            p_values = model.pvalues.values if hasattr(model, "pvalues") else [np.nan] * len(param_names)

            # Calculate confidence intervals
            if hasattr(model, "conf_int"):
                conf_int = model.conf_int()
                ci_lower = conf_int.iloc[:, 0].values
                ci_upper = conf_int.iloc[:, 1].values
            else:
                ci_lower = [np.nan] * len(param_names)
                ci_upper = [np.nan] * len(param_names)

            coefficients = pd.DataFrame({
                "variable": param_names,
                "coefficient": model.params.values,
                "std_error": std_errors,
                "t_stat": t_stats,
                "p_value": p_values,
                "ci_0.025": ci_lower,
                "ci_0.975": ci_upper,
                "vif": [np.nan] * len(param_names),  # VIF not applicable for ARIMA
            })
        else:
            coefficients = pd.DataFrame({
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
        order = fit.fit_data.get("order", (0, 0, 0))
        seasonal_order = fit.fit_data.get("seasonal_order", (0, 0, 0, 0))
        blueprint = fit.blueprint

        stats_rows.extend([
            {"metric": "formula", "value": blueprint.get("formula", "") if isinstance(blueprint, dict) else "", "split": ""},
            {"metric": "model_type", "value": fit.spec.model_type, "split": ""},
            {"metric": "order", "value": str(order), "split": ""},
            {"metric": "seasonal_order", "value": str(seasonal_order), "split": ""},
            {"metric": "aic", "value": model.aic if hasattr(model, "aic") else np.nan, "split": ""},
            {"metric": "bic", "value": model.bic if hasattr(model, "bic") else np.nan, "split": ""},
            {"metric": "log_likelihood", "value": model.llf if hasattr(model, "llf") else np.nan, "split": ""},
            {"metric": "sigma2", "value": model.scale if hasattr(model, "scale") else np.nan, "split": ""},
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
