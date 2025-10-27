"""
Statsmodels engine for Exponential Smoothing models

Implements ETS (Error, Trend, Seasonality) models using statsmodels:
- Simple Exponential Smoothing
- Holt's Linear Method
- Holt-Winters Seasonal Method

Uses statsmodels.tsa.holtwinters.ExponentialSmoothing for all variants.
"""

from typing import Dict, Any, Literal, Optional
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData


@register_engine("exp_smoothing", "statsmodels")
class StatsmodelsExpSmoothingEngine(Engine):
    """
    Statsmodels engine for Exponential Smoothing models.

    Parameter mapping: tidymodels → statsmodels
    - seasonal_period → seasonal_periods
    - error → error (same)
    - trend → trend (same, but None → None)
    - season → seasonal (same, but None → None)
    - damping → damped_trend

    Model selection:
    - Simple ES: trend=None, seasonal=None
    - Holt: trend set, seasonal=None
    - Holt-Winters: trend and seasonal both set
    """

    param_map = {
        "seasonal_period": "seasonal_periods",
        "error": "error",
        "trend": "trend",
        "season": "seasonal",
        "damping": "damped_trend",
    }

    def fit_raw(
        self, spec: ModelSpec, data: pd.DataFrame, formula: str
    ) -> tuple[Dict[str, Any], Any]:
        """
        Fit Exponential Smoothing model using raw data (bypasses hardhat molding).

        Args:
            spec: ModelSpec with Exponential Smoothing configuration
            data: Training data DataFrame
            formula: Formula string (e.g., "sales ~ date" or "sales ~ 1")

        Returns:
            Tuple of (fit_data dict, blueprint)

        Note:
            Exponential Smoothing is univariate - no exogenous variables.
            Formula should be "y ~ 1" or "y ~ date" (date used for index only).
        """
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        # Parse formula
        parts = formula.split("~")
        if len(parts) != 2:
            raise ValueError(f"Invalid formula: {formula}")

        outcome_name = parts[0].strip()
        predictor_part = parts[1].strip()

        # Validate outcome exists
        if outcome_name not in data.columns:
            raise ValueError(f"Outcome '{outcome_name}' not found in data")

        # Get outcome series
        y = data[outcome_name]

        # Handle time index
        date_col = None
        if predictor_part != "1":
            # Parse predictors (should just be date column)
            predictor_names = [p.strip() for p in predictor_part.split("+")]

            # Find datetime column to use as index
            for p in predictor_names:
                if p in data.columns and pd.api.types.is_datetime64_any_dtype(data[p]):
                    date_col = p
                    y = data.set_index(date_col)[outcome_name]
                    break

        # Get parameters
        args = spec.args
        seasonal_period = args.get("seasonal_period")
        error = args.get("error", "additive")
        trend = args.get("trend")  # Can be None
        season = args.get("season")  # Can be None
        damping = args.get("damping", False)

        # Map None to None for trend/season (statsmodels expects None, not string "none")
        trend_param = trend if trend is not None else None
        season_param = season if season is not None else None

        # Create ExponentialSmoothing model
        # Note: ExponentialSmoothing constructor doesn't take error parameter directly
        # It's specified in the fit() method
        model_kwargs = {
            "trend": trend_param,
            "seasonal": season_param,
        }

        # Add seasonal_periods only if seasonal component exists
        if season_param is not None:
            if seasonal_period is None:
                raise ValueError("seasonal_period must be specified when season is not None")
            model_kwargs["seasonal_periods"] = seasonal_period

        # Add damped_trend only if trend exists
        if trend_param is not None:
            model_kwargs["damped_trend"] = damping

        # Create model
        model = ExponentialSmoothing(y, **model_kwargs)

        # Fit with minimal output
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Fit with error type (not specified in constructor)
            # For newer statsmodels, error is called 'error', older versions don't support it
            try:
                fitted_model = model.fit(optimized=True, use_brute=False)
            except Exception:
                # Fallback for older statsmodels
                fitted_model = model.fit()

        # Get fitted values and residuals
        fitted_values = fitted_model.fittedvalues.values
        actuals = y.values if isinstance(y, pd.Series) else y
        residuals = fitted_model.resid.values if hasattr(fitted_model, 'resid') else actuals - fitted_values

        # Extract dates
        dates = None
        if date_col is not None:
            dates = data[date_col].values
        elif 'date' in data.columns:
            dates = data['date'].values
        else:
            # Try to find any datetime column
            for col in data.columns:
                if pd.api.types.is_datetime64_any_dtype(data[col]):
                    dates = data[col].values
                    date_col = col
                    break

        # If still no dates, use index
        if dates is None:
            dates = np.arange(len(y))

        # Create blueprint
        blueprint = {
            "formula": formula,
            "outcome_name": outcome_name,
            "date_col": date_col,
            "seasonal_period": seasonal_period,
            "error": error,
            "trend": trend,
            "season": season,
            "damping": damping,
        }

        # Determine method name
        if trend is None and season is None:
            method = "simple"
        elif trend is not None and season is None:
            method = "holt"
        else:
            method = "holt-winters"

        # Return fit data
        fit_data = {
            "model": fitted_model,
            "outcome_name": outcome_name,
            "seasonal_period": seasonal_period,
            "method": method,
            "n_obs": len(y),
            "y_train": actuals,
            "fitted": fitted_values,
            "residuals": residuals,
            "dates": dates,
        }

        return fit_data, blueprint

    def fit(self, spec: ModelSpec, molded: MoldedData) -> Dict[str, Any]:
        """Not used for Exponential Smoothing - use fit_raw() instead"""
        raise NotImplementedError("Exponential Smoothing uses fit_raw() instead of fit()")

    def predict(
        self, fit: ModelFit, molded: MoldedData, type: str
    ) -> pd.DataFrame:
        """Not used for Exponential Smoothing - use predict_raw() instead"""
        raise NotImplementedError("Exponential Smoothing uses predict_raw() instead of predict()")

    def predict_raw(
        self,
        fit: ModelFit,
        new_data: pd.DataFrame,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make predictions using fitted Exponential Smoothing model.

        Args:
            fit: ModelFit with fitted model
            new_data: DataFrame with forecast horizon (length determines forecast periods)
            type: Prediction type
                - "numeric": Point forecasts
                - "conf_int": Forecasts with prediction intervals

        Returns:
            DataFrame with predictions

        Note:
            Exponential Smoothing is univariate, so new_data just needs length.
            If new_data has a date column, it will be used for indexing.
        """
        if type not in ("numeric", "conf_int"):
            raise ValueError(
                f"exp_smoothing supports type='numeric' or 'conf_int', got '{type}'"
            )

        model = fit.fit_data["model"]

        # Determine forecast horizon
        n_periods = len(new_data)

        # Get date column from blueprint (if present)
        date_col = fit.blueprint.get("date_col") if isinstance(fit.blueprint, dict) else None
        date_index = new_data[date_col] if date_col and date_col in new_data.columns else None

        # Make predictions
        if type == "numeric":
            # Point forecasts
            forecast = model.forecast(steps=n_periods)
            result = pd.DataFrame({".pred": forecast.values})
            if date_index is not None:
                result.index = date_index
            return result
        else:  # conf_int
            # Get forecast with prediction intervals
            # ExponentialSmoothing doesn't have built-in prediction intervals like ARIMA
            # We'll use simulate to generate intervals
            forecast = model.forecast(steps=n_periods)

            # For now, use simple approximation based on residual std
            # A more sophisticated approach would use simulation
            if hasattr(model, 'resid') and len(model.resid) > 0:
                resid_std = np.std(model.resid)
                # 95% intervals (1.96 * std)
                margin = 1.96 * resid_std * np.sqrt(np.arange(1, n_periods + 1))
            else:
                # Fallback: use 10% of forecast value
                margin = 0.1 * np.abs(forecast.values)

            result = pd.DataFrame({
                ".pred": forecast.values,
                ".pred_lower": forecast.values - margin,
                ".pred_upper": forecast.values + margin,
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

        # Ljung-Box and Breusch-Pagan placeholders
        results["ljung_box_stat"] = np.nan
        results["ljung_box_p"] = np.nan
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
            - Coefficients: Smoothing parameters (alpha, beta, gamma)
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
        coef_rows = []

        # Extract smoothing parameters from fitted model
        # Alpha (level smoothing parameter)
        if hasattr(model, 'params') and 'smoothing_level' in model.params:
            alpha = model.params['smoothing_level']
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
        if hasattr(model, 'params') and 'smoothing_trend' in model.params:
            beta = model.params['smoothing_trend']
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

        # Gamma (seasonal smoothing parameter)
        if hasattr(model, 'params') and 'smoothing_seasonal' in model.params:
            gamma = model.params['smoothing_seasonal']
            coef_rows.append({
                "variable": "gamma (smoothing_seasonal)",
                "coefficient": gamma,
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
            })

        # Phi (damping parameter)
        if hasattr(model, 'params') and 'damping_trend' in model.params:
            phi = model.params['damping_trend']
            coef_rows.append({
                "variable": "phi (damping_trend)",
                "coefficient": phi,
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
            })

        # Initial level
        if hasattr(model, 'params') and 'initial_level' in model.params:
            l0 = model.params['initial_level']
            coef_rows.append({
                "variable": "l0 (initial_level)",
                "coefficient": l0,
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
            })

        # Initial trend
        if hasattr(model, 'params') and 'initial_trend' in model.params:
            b0 = model.params['initial_trend']
            coef_rows.append({
                "variable": "b0 (initial_trend)",
                "coefficient": b0,
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
        method = fit.fit_data.get("method", "unknown")
        seasonal_period = fit.fit_data.get("seasonal_period")

        stats_rows.extend([
            {"metric": "formula", "value": blueprint.get("formula", "") if isinstance(blueprint, dict) else "", "split": ""},
            {"metric": "model_type", "value": fit.spec.model_type, "split": ""},
            {"metric": "method", "value": method, "split": ""},
            {"metric": "seasonal_period", "value": seasonal_period if seasonal_period is not None else 0, "split": ""},
            {"metric": "error", "value": blueprint.get("error", "") if isinstance(blueprint, dict) else "", "split": ""},
            {"metric": "trend", "value": str(blueprint.get("trend", "")) if isinstance(blueprint, dict) else "", "split": ""},
            {"metric": "season", "value": str(blueprint.get("season", "")) if isinstance(blueprint, dict) else "", "split": ""},
            {"metric": "damping", "value": blueprint.get("damping", False) if isinstance(blueprint, dict) else False, "split": ""},
            {"metric": "aic", "value": model.aic if hasattr(model, "aic") else np.nan, "split": ""},
            {"metric": "bic", "value": model.bic if hasattr(model, "bic") else np.nan, "split": ""},
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
