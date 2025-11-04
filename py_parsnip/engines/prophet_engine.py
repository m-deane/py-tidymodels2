"""
Prophet engine for time series forecasting

Prophet requires specific column names:
- 'ds': datetime column
- 'y': value column

This engine handles the conversion from hardhat's molded format to Prophet's format.
"""

from typing import Dict, Any, Literal, Optional
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData
from py_parsnip.utils import _infer_date_column, _parse_ts_formula


@register_engine("prophet_reg", "prophet")
class ProphetEngine(Engine):
    """
    Prophet engine for time series regression.

    Parameter mapping: tidymodels → prophet
    (Prophet uses same parameter names, so no translation needed)
    """

    param_map = {}  # Prophet uses same names

    def fit(self, spec: ModelSpec, molded: MoldedData) -> Dict[str, Any]:
        """Not used for Prophet - use fit_raw() instead"""
        raise NotImplementedError("Prophet uses fit_raw() instead of fit()")

    def predict(
        self,
        fit: ModelFit,
        molded: MoldedData,
        type: str,
    ) -> pd.DataFrame:
        """Not used for Prophet - use predict_raw() instead"""
        raise NotImplementedError("Prophet uses predict_raw() instead of predict()")

    def fit_raw(self, spec: ModelSpec, data: pd.DataFrame, formula: str, date_col: str, original_training_data: Optional[pd.DataFrame] = None) -> tuple[Dict[str, Any], Any]:
        """
        Fit Prophet model using raw data (bypasses hardhat molding).

        Args:
            spec: ModelSpec with Prophet configuration
            data: Training data DataFrame (may be preprocessed)
            formula: Formula string (e.g., "sales ~ date" or "sales ~ lag1 + lag2 + date")
            date_col: Name of date column or '__index__' for DatetimeIndex
            original_training_data: Original unpreprocessed training data (for raw datetime values)

        Returns:
            Tuple of (fit_data dict, blueprint)

        Note:
            Prophet requires datetime column and doesn't work well with patsy's
            categorical treatment of dates. We bypass hardhat molding here.

            Exogenous Regressors:
            Prophet supports adding external regressors via model.add_regressor().
            Formula format: "outcome ~ exog1 + exog2 + date"
            The date column is automatically excluded from regressors.
        """
        from prophet import Prophet

        # Parse formula to extract outcome and exogenous variables
        outcome_name, exog_vars = _parse_ts_formula(formula, date_col)

        # Validate outcome exists
        if outcome_name not in data.columns:
            raise ValueError(f"Outcome '{outcome_name}' not found in data")

        # Get date values (handle __index__ case)
        if date_col == '__index__':
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("date_col is '__index__' but data does not have DatetimeIndex")
            date_values = data.index
        else:
            # Validate date column exists
            if date_col not in data.columns:
                raise ValueError(f"Date column '{date_col}' not found in data")

            # Get dates from original training data if available (to avoid normalized values)
            if original_training_data is not None and date_col in original_training_data.columns:
                date_values = original_training_data[date_col]
            else:
                date_values = data[date_col]

        # Create Prophet-format DataFrame with ds and y
        prophet_df = pd.DataFrame({
            "ds": date_values,
            "y": data[outcome_name],
        })

        # Add exogenous variables to prophet_df
        exog_vars_used = []
        if exog_vars:
            for var in exog_vars:
                if var not in data.columns:
                    raise ValueError(f"Exogenous variable '{var}' not found in data")
                prophet_df[var] = data[var].values
                exog_vars_used.append(var)

        # Create Prophet model with args
        args = spec.args.copy()
        model = Prophet(**args)

        # Add external regressors to model
        for var in exog_vars_used:
            model.add_regressor(var)

        # Fit model (Prophet captures stdout, we suppress warnings)
        import logging
        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        model.fit(prophet_df)

        # Get fitted values on training data
        fitted_forecast = model.predict(prophet_df[['ds'] + exog_vars_used])
        fitted = fitted_forecast['yhat'].values

        # Calculate residuals
        actuals = prophet_df['y'].values
        residuals = actuals - fitted

        # Extract dates (use raw datetime values, not normalized)
        dates = prophet_df['ds'].values

        # Create a simple blueprint for prediction
        # (Not a full hardhat Blueprint, just metadata)
        blueprint = {
            "formula": formula,
            "outcome_name": outcome_name,
            "date_col": date_col,
            "exog_vars": exog_vars_used,
        }

        # Return fitted model and metadata
        fit_data = {
            "model": model,
            "n_obs": len(prophet_df),
            "date_col": date_col,
            "outcome_name": outcome_name,
            "exog_vars": exog_vars_used,
            "prophet_df": prophet_df,
            "y_train": actuals,
            "fitted": fitted,
            "residuals": residuals,
            "dates": dates,
        }

        return fit_data, blueprint

    def predict_raw(
        self,
        fit: ModelFit,
        new_data: pd.DataFrame,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make predictions using fitted Prophet model (raw data, no forging).

        Args:
            fit: ModelFit with fitted Prophet model
            new_data: Raw DataFrame with date column and exogenous variables (if used)
            type: Prediction type
                - "numeric": Point predictions (yhat)
                - "conf_int": Prediction intervals (yhat, yhat_lower, yhat_upper)

        Returns:
            DataFrame with predictions

        Note:
            Prophet always returns prediction intervals, so we can provide
            both point predictions and confidence intervals.

            If exogenous regressors were used during fit, they must be present
            in new_data for prediction.
        """
        if type not in ("numeric", "conf_int"):
            raise ValueError(
                f"prophet_reg supports type='numeric' or 'conf_int', got '{type}'"
            )

        model = fit.fit_data["model"]
        date_col = fit.fit_data["date_col"]
        exog_vars = fit.fit_data.get("exog_vars", [])

        # Get date values (handle __index__ case)
        if date_col == '__index__':
            if not isinstance(new_data.index, pd.DatetimeIndex):
                raise ValueError("date_col is '__index__' but new_data does not have DatetimeIndex")
            date_values = new_data.index
        else:
            # Extract date column
            if date_col not in new_data.columns:
                raise ValueError(
                    f"Date column '{date_col}' not found in new data. "
                    f"Available columns: {list(new_data.columns)}"
                )
            date_values = new_data[date_col]

        # Create future DataFrame for Prophet
        future = pd.DataFrame({"ds": date_values})

        # Add exogenous variables if they were used during fit
        if exog_vars:
            for var in exog_vars:
                if var not in new_data.columns:
                    raise ValueError(
                        f"Exogenous variable '{var}' was used during fit but not found in new_data. "
                        f"Available columns: {list(new_data.columns)}"
                    )
                future[var] = new_data[var].values

        # Make predictions
        forecast = model.predict(future)

        # Return based on type
        if type == "numeric":
            # Return point predictions only, indexed by date
            result = pd.DataFrame({".pred": forecast["yhat"].values}, index=date_values)
            return result
        else:  # conf_int
            # Return point predictions + intervals, indexed by date
            result = pd.DataFrame({
                ".pred": forecast["yhat"].values,
                ".pred_lower": forecast["yhat_lower"].values,
                ".pred_upper": forecast["yhat_upper"].values,
            }, index=date_values)
            return result

    def _calculate_metrics(self, actuals: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics (same as sklearn)"""
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

        # Ljung-Box and Breusch-Pagan would require statsmodels
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
            - Coefficients: Prophet hyperparameters and components
            - Stats: Comprehensive metrics by split + residual diagnostics
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
            date_col = fit.fit_data["date_col"]

            test_actuals = test_data[outcome_col].values
            test_predictions = test_preds[".pred"].values
            test_residuals = test_actuals - test_predictions

            # Get original test data if available (for raw datetime values)
            # This prevents normalized/preprocessed date values from appearing in outputs
            original_test_data = fit.evaluation_data.get("original_test_data")

            # Handle __index__ case
            if date_col == '__index__':
                if original_test_data is not None and isinstance(original_test_data.index, pd.DatetimeIndex):
                    test_dates = original_test_data.index.values
                elif isinstance(test_data.index, pd.DatetimeIndex):
                    test_dates = test_data.index.values
                else:
                    test_dates = np.arange(len(test_actuals))
            else:
                if original_test_data is not None and date_col in original_test_data.columns:
                    test_dates = original_test_data[date_col].values
                else:
                    test_dates = test_data[date_col].values if date_col in test_data.columns else np.arange(len(test_actuals))

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
        # 2. COEFFICIENTS DataFrame (Hyperparameters for Prophet)
        # ====================
        # For Prophet, we'll report hyperparameters and components as "coefficients"
        coef_rows = []

        # Growth hyperparameter
        growth = fit.spec.args.get("growth", "linear")
        coef_rows.append({
            "variable": "growth",
            "coefficient": growth,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_0.025": np.nan,
            "ci_0.975": np.nan,
            "vif": np.nan,
        })

        # Changepoint prior scale
        cp_prior = fit.spec.args.get("changepoint_prior_scale", 0.05)
        coef_rows.append({
            "variable": "changepoint_prior_scale",
            "coefficient": cp_prior,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_0.025": np.nan,
            "ci_0.975": np.nan,
            "vif": np.nan,
        })

        # Seasonality prior scale
        seas_prior = fit.spec.args.get("seasonality_prior_scale", 10.0)
        coef_rows.append({
            "variable": "seasonality_prior_scale",
            "coefficient": seas_prior,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_0.025": np.nan,
            "ci_0.975": np.nan,
            "vif": np.nan,
        })

        # Seasonality mode
        seas_mode = fit.spec.args.get("seasonality_mode", "additive")
        coef_rows.append({
            "variable": "seasonality_mode",
            "coefficient": seas_mode,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_0.025": np.nan,
            "ci_0.975": np.nan,
            "vif": np.nan,
        })

        # Number of changepoints detected
        if hasattr(model, "changepoints"):
            n_changepoints = len(model.changepoints)
            coef_rows.append({
                "variable": "n_changepoints",
                "coefficient": n_changepoints,
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
            })

        coefficients = pd.DataFrame(coef_rows)

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
        stats_rows.extend([
            {"metric": "formula", "value": blueprint.get("formula", "") if isinstance(blueprint, dict) else "", "split": ""},
            {"metric": "model_type", "value": fit.spec.model_type, "split": ""},
            {"metric": "growth", "value": growth, "split": ""},
            {"metric": "seasonality_mode", "value": seas_mode, "split": ""},
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
