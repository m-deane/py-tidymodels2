"""
Pmdarima engine for auto ARIMA regression

Automatic ARIMA (AutoRegressive Integrated Moving Average) using pmdarima.
Automatically selects optimal (p,d,q)(P,D,Q)[m] parameters using information criteria.
"""

from typing import Dict, Any, Literal
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData
from py_parsnip.utils import _infer_date_column, _parse_ts_formula


@register_engine("arima_reg", "auto_arima")
class PmdarimaAutoARIMAEngine(Engine):
    """
    Pmdarima engine for automatic ARIMA parameter selection.

    Auto ARIMA automatically searches for optimal (p,d,q)(P,D,Q)[m] values
    using information criteria (AIC, BIC) to balance fit quality and complexity.

    Parameters from model spec are used as search constraints:
    - seasonal_period: Seasonality period (required for seasonal models)
    - non_seasonal_ar: Max AR order to consider (default: 5)
    - non_seasonal_differences: Max differencing order (default: 2)
    - non_seasonal_ma: Max MA order to consider (default: 5)
    - seasonal_ar: Max seasonal AR order (default: 2)
    - seasonal_differences: Max seasonal differencing (default: 1)
    - seasonal_ma: Max seasonal MA order (default: 2)

    Note: These are MAX values - auto_arima will search for optimal within these limits.
    """

    param_map = {
        "non_seasonal_ar": "max_p",
        "non_seasonal_differences": "max_d",
        "non_seasonal_ma": "max_q",
        "seasonal_ar": "max_P",
        "seasonal_differences": "max_D",
        "seasonal_ma": "max_Q",
        "seasonal_period": "m",
    }

    def fit_raw(
        self, spec: ModelSpec, data: pd.DataFrame, formula: str, date_col: str = None
    ) -> tuple[Dict[str, Any], Any]:
        """
        Fit auto ARIMA model using pmdarima's auto_arima.

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
            from pmdarima import auto_arima
        except ValueError as e:
            if "numpy.dtype size changed" in str(e):
                raise ImportError(
                    "pmdarima has a numpy compatibility issue with numpy 2.x. "
                    "Solutions:\n"
                    "1. Use the statsmodels ARIMA engine instead: "
                    "arima_reg().set_engine('statsmodels')\n"
                    "2. Downgrade numpy to 1.26.x: pip install 'numpy<2.0'\n"
                    "3. Wait for pmdarima to release numpy 2.x compatible wheels"
                ) from e
            raise

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
                exog = data[exog_vars] if len(exog_vars) > 1 else data[exog_vars[0]]
            else:
                exog = None
        else:
            # Use date column as index
            y = data.set_index(inferred_date_col)[outcome_name]

            # Get exogenous variables if present (excluding date column)
            if exog_vars:
                exog = data.set_index(inferred_date_col)[exog_vars]
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

        # Run auto_arima to find optimal parameters
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_model = auto_arima(
                y,
                exogenous=exog,
                seasonal=seasonal,
                m=seasonal_period if seasonal else 1,
                max_p=max_p,
                max_d=max_d,
                max_q=max_q,
                max_P=max_P,
                max_D=max_D,
                max_Q=max_Q,
                trace=False,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
                information_criterion="aic",
                n_fits=50,
            )

        # Get fitted values and residuals
        fitted_values = fitted_model.predict_in_sample(exogenous=exog)
        actuals = y.values if isinstance(y, pd.Series) else y
        residuals = actuals - fitted_values

        # Extract selected orders
        order = fitted_model.order
        seasonal_order = fitted_model.seasonal_order

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
            "engine": "auto_arima",
        }

        return fit_data, blueprint

    def fit(self, spec: ModelSpec, molded: MoldedData) -> Dict[str, Any]:
        """Not used for auto ARIMA - use fit_raw() instead"""
        raise NotImplementedError("Auto ARIMA uses fit_raw() instead of fit()")

    def predict(
        self, fit: ModelFit, molded: MoldedData, type: str
    ) -> pd.DataFrame:
        """Not used for auto ARIMA - use predict_raw() instead"""
        raise NotImplementedError("Auto ARIMA uses predict_raw() instead of predict()")

    def predict_raw(
        self,
        fit: ModelFit,
        new_data: pd.DataFrame,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make predictions using fitted auto ARIMA model.

        Args:
            fit: ModelFit with fitted auto ARIMA model
            new_data: DataFrame with same structure as training
                      (Must have exogenous variables if used in training)
            type: Prediction type
                - "numeric": Point forecasts
                - "conf_int": Forecasts with prediction intervals

        Returns:
            DataFrame with predictions

        Note:
            For pure ARIMA (no exog), new_data just needs length.
            For ARIMAX, new_data must contain exogenous variables.
        """
        if type not in ("numeric", "conf_int"):
            raise ValueError(
                f"arima_reg supports type='numeric' or 'conf_int', got '{type}'"
            )

        model = fit.fit_data["model"]
        exog_vars = fit.fit_data["exog_vars"]
        date_col = fit.fit_data["date_col"]

        # Determine forecast horizon
        n_periods = len(new_data)

        # Get exogenous variables if present
        if exog_vars:
            # Validate exog columns exist
            missing = [v for v in exog_vars if v not in new_data.columns]
            if missing:
                raise ValueError(
                    f"Exogenous variables {missing} not found in new_data. "
                    f"Required: {exog_vars}"
                )
            exog = new_data[exog_vars].values
        else:
            exog = None

        # Get date index from new_data
        if date_col == '__index__':
            date_index = new_data.index
        elif date_col in new_data.columns:
            date_index = new_data[date_col]
        else:
            date_index = None

        # Make predictions
        if type == "numeric":
            # Point forecasts
            forecast = model.predict(n_periods=n_periods, exogenous=exog)
            result = pd.DataFrame({".pred": forecast})
            if date_index is not None:
                result.index = date_index
            return result
        else:  # conf_int
            # Get forecast with prediction intervals
            forecast, conf_int = model.predict(
                n_periods=n_periods, exogenous=exog, return_conf_int=True, alpha=0.05
            )

            result = pd.DataFrame({
                ".pred": forecast,
                ".pred_lower": conf_int[:, 0],
                ".pred_upper": conf_int[:, 1],
            })
            if date_index is not None:
                result.index = date_index
            return result

    def _calculate_metrics(self, actuals: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics"""
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

        # Breusch-Pagan test not applicable for auto ARIMA (no exog matrix in same format)
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
            - Coefficients: ARIMA parameters from auto-selected model
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
        # Get model parameters from pmdarima model
        if hasattr(model, "params") and model.params is not None:
            try:
                # Get parameter names and values
                param_names = list(model.arima_res_.param_names) if hasattr(model, "arima_res_") else []

                if param_names:
                    # Get values
                    param_values = model.params()

                    # Get standard errors, t-stats, p-values from arima_res_
                    if hasattr(model, "arima_res_"):
                        arima_res = model.arima_res_
                        std_errors = arima_res.bse.values if hasattr(arima_res, "bse") else [np.nan] * len(param_names)
                        t_stats = arima_res.tvalues.values if hasattr(arima_res, "tvalues") else [np.nan] * len(param_names)
                        p_values = arima_res.pvalues.values if hasattr(arima_res, "pvalues") else [np.nan] * len(param_names)

                        # Calculate confidence intervals
                        if hasattr(arima_res, "conf_int"):
                            conf_int = arima_res.conf_int()
                            ci_lower = conf_int.iloc[:, 0].values
                            ci_upper = conf_int.iloc[:, 1].values
                        else:
                            ci_lower = [np.nan] * len(param_names)
                            ci_upper = [np.nan] * len(param_names)
                    else:
                        std_errors = [np.nan] * len(param_names)
                        t_stats = [np.nan] * len(param_names)
                        p_values = [np.nan] * len(param_names)
                        ci_lower = [np.nan] * len(param_names)
                        ci_upper = [np.nan] * len(param_names)

                    coefficients = pd.DataFrame({
                        "variable": param_names,
                        "coefficient": param_values,
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
            except Exception:
                # Fallback if param extraction fails
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

        # Get AIC, BIC from pmdarima model
        aic = model.aic() if hasattr(model, "aic") else np.nan
        bic = model.bic() if hasattr(model, "bic") else np.nan
        aicc = model.aicc() if hasattr(model, "aicc") else np.nan

        stats_rows.extend([
            {"metric": "formula", "value": blueprint.get("formula", "") if isinstance(blueprint, dict) else "", "split": ""},
            {"metric": "model_type", "value": fit.spec.model_type, "split": ""},
            {"metric": "engine", "value": "auto_arima", "split": ""},
            {"metric": "order", "value": str(order), "split": ""},
            {"metric": "seasonal_order", "value": str(seasonal_order), "split": ""},
            {"metric": "aic", "value": aic, "split": ""},
            {"metric": "bic", "value": bic, "split": ""},
            {"metric": "aicc", "value": aicc, "split": ""},
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
