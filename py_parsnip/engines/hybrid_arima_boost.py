"""
Hybrid ARIMA + XGBoost engine

This engine implements a two-stage hybrid forecasting model:
1. Stage 1: Fit ARIMA to capture linear patterns and autocorrelation
2. Stage 2: Fit XGBoost on ARIMA residuals to capture non-linear patterns
3. Prediction: base_pred (ARIMA) + residual_pred (XGBoost)

The hybrid approach is effective when data has both linear temporal patterns
and complex non-linear relationships.
"""

from typing import Dict, Any, Literal
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData
from py_parsnip.utils import _infer_date_column, _parse_ts_formula


@register_engine("arima_boost", "hybrid_arima_xgboost")
class HybridARIMABoostEngine(Engine):
    """
    Hybrid ARIMA + XGBoost engine for time series forecasting.

    Parameter mapping:
    ARIMA parameters → statsmodels SARIMAX
    XGBoost parameters → xgboost.XGBRegressor
    - trees → n_estimators
    - tree_depth → max_depth
    - learn_rate → learning_rate
    - min_n → min_child_weight
    - loss_reduction → gamma
    - sample_size → subsample
    - mtry → colsample_bytree
    """

    param_map = {
        # ARIMA params (inherited from statsmodels_arima)
        "non_seasonal_ar": "p",
        "non_seasonal_differences": "d",
        "non_seasonal_ma": "q",
        "seasonal_ar": "P",
        "seasonal_differences": "D",
        "seasonal_ma": "Q",
        "seasonal_period": "m",
        # XGBoost params
        "trees": "n_estimators",
        "tree_depth": "max_depth",
        "learn_rate": "learning_rate",
        "min_n": "min_child_weight",
        "loss_reduction": "gamma",
        "sample_size": "subsample",
        "mtry": "colsample_bytree",
    }

    def fit_raw(
        self, spec: ModelSpec, data: pd.DataFrame, formula: str, date_col: str = None
    ) -> tuple[Dict[str, Any], Any]:
        """
        Fit hybrid ARIMA + XGBoost model using raw data.

        Strategy:
        1. Fit ARIMA model to capture linear patterns
        2. Calculate ARIMA residuals
        3. Fit XGBoost on residuals with same predictors (if any)
        4. Store both models

        Args:
            spec: ModelSpec with hybrid configuration
            data: Training data DataFrame
            formula: Formula string (e.g., "sales ~ date" or "sales ~ date + x1 + x2")
            date_col: Name of date column, or '__index__' for DatetimeIndex

        Returns:
            Tuple of (fit_data dict, blueprint)
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from xgboost import XGBRegressor

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

        # Handle "." notation (all columns except outcome and date)
        if exog_vars == ['.']:
            if inferred_date_col == '__index__':
                exog_vars = [col for col in data.columns if col != outcome_name]
            else:
                exog_vars = [col for col in data.columns if col != outcome_name and col != inferred_date_col]

        # Handle __index__ case
        if inferred_date_col == '__index__':
            # Use DatetimeIndex
            y = data[outcome_name]

            # Get exogenous variables if present
            if exog_vars:
                exog = data[exog_vars] if len(exog_vars) > 1 else data[[exog_vars[0]]]
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

        # ==================
        # STAGE 1: Fit ARIMA
        # ==================
        args = spec.args

        # Build ARIMA order tuples
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
        arima_model = SARIMAX(
            y,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        # Fit with minimal output
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_arima = arima_model.fit(disp=False)

        # Get ARIMA fitted values and residuals
        arima_fitted = fitted_arima.fittedvalues.values
        arima_residuals = fitted_arima.resid.values

        # =======================
        # STAGE 2: Fit XGBoost on residuals
        # =======================
        # For XGBoost, we need to create features from the time series
        # We'll use the exogenous variables (if any) plus lagged features

        # Prepare features for XGBoost
        if exog is not None:
            # Use exogenous variables as features
            X_boost = exog.values if isinstance(exog, pd.DataFrame) else exog.reshape(-1, 1)
        else:
            # No exogenous variables - use time index as feature
            X_boost = np.arange(len(y)).reshape(-1, 1)

        # Target for XGBoost is ARIMA residuals
        y_boost = arima_residuals

        # Extract XGBoost parameters
        xgb_params = {
            "n_estimators": args.get("trees", 100),
            "max_depth": args.get("tree_depth", 6),
            "learning_rate": args.get("learn_rate", 0.1),
            "min_child_weight": args.get("min_n", 1),
            "gamma": args.get("loss_reduction", 0.0),
            "subsample": args.get("sample_size", 1.0),
            "colsample_bytree": args.get("mtry", 1.0),
            "random_state": 42,
            "verbosity": 0,
        }

        # Fit XGBoost
        xgb_model = XGBRegressor(**xgb_params)
        xgb_model.fit(X_boost, y_boost)

        # Get XGBoost predictions on training data
        xgb_fitted = xgb_model.predict(X_boost)

        # ==================
        # Combine predictions
        # ==================
        # Final fitted values = ARIMA + XGBoost
        final_fitted = arima_fitted + xgb_fitted

        # Final residuals
        actuals = y.values if isinstance(y, pd.Series) else y
        final_residuals = actuals - final_fitted

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
            "xgb_params": xgb_params,
        }

        # Return fit data
        fit_data = {
            "arima_model": fitted_arima,
            "xgb_model": xgb_model,
            "outcome_name": outcome_name,
            "exog_vars": exog_vars,
            "date_col": inferred_date_col,
            "order": order,
            "seasonal_order": seasonal_order,
            "n_obs": len(y),
            "y_train": actuals,
            "arima_fitted": arima_fitted,
            "xgb_fitted": xgb_fitted,
            "fitted": final_fitted,
            "residuals": final_residuals,
            "dates": dates,
            "xgb_params": xgb_params,
        }

        return fit_data, blueprint

    def fit(self, spec: ModelSpec, molded: MoldedData) -> Dict[str, Any]:
        """Not used for hybrid ARIMA+XGBoost - use fit_raw() instead"""
        raise NotImplementedError("Hybrid ARIMA+XGBoost uses fit_raw() instead of fit()")

    def predict(
        self, fit: ModelFit, molded: MoldedData, type: str
    ) -> pd.DataFrame:
        """Not used for hybrid ARIMA+XGBoost - use predict_raw() instead"""
        raise NotImplementedError("Hybrid ARIMA+XGBoost uses predict_raw() instead of predict()")

    def predict_raw(
        self,
        fit: ModelFit,
        new_data: pd.DataFrame,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make predictions using fitted hybrid ARIMA+XGBoost model.

        Strategy:
        1. Get ARIMA predictions
        2. Get XGBoost predictions (on same features)
        3. Combine: arima_pred + xgb_pred

        Args:
            fit: ModelFit with fitted hybrid model
            new_data: DataFrame with same structure as training
            type: Prediction type
                - "numeric": Point forecasts
                - "conf_int": Not supported for hybrid models

        Returns:
            DataFrame with predictions
        """
        if type not in ("numeric",):
            raise ValueError(
                f"arima_boost currently supports type='numeric' only, got '{type}'"
            )

        arima_model = fit.fit_data["arima_model"]
        xgb_model = fit.fit_data["xgb_model"]
        exog_vars = fit.fit_data["exog_vars"]
        date_col = fit.fit_data["date_col"]

        # Determine forecast horizon
        n_periods = len(new_data)

        # ==================
        # ARIMA predictions
        # ==================
        # Get exogenous variables for ARIMA if present
        if exog_vars:
            missing = [v for v in exog_vars if v not in new_data.columns]
            if missing:
                raise ValueError(
                    f"Exogenous variables {missing} not found in new_data. "
                    f"Required: {exog_vars}"
                )
            exog = new_data[exog_vars]
        else:
            exog = None

        # Get ARIMA forecasts
        arima_forecast = arima_model.forecast(steps=n_periods, exog=exog)

        # ==================
        # XGBoost predictions
        # ==================
        # Prepare features for XGBoost (same as training)
        if exog is not None:
            X_boost = exog.values
        else:
            # Use time index
            last_train_idx = fit.fit_data["n_obs"]
            X_boost = np.arange(last_train_idx, last_train_idx + n_periods).reshape(-1, 1)

        # Get XGBoost predictions
        xgb_forecast = xgb_model.predict(X_boost)

        # ==================
        # Combine predictions
        # ==================
        final_forecast = arima_forecast.values + xgb_forecast

        # Get date index from new_data
        if date_col == '__index__':
            date_index = new_data.index
        elif date_col in new_data.columns:
            date_index = new_data[date_col]
        else:
            date_index = None

        # Return predictions
        result = pd.DataFrame({".pred": final_forecast})
        if date_index is not None:
            result.index = date_index
        return result

    def _calculate_metrics(
        self, actuals: np.ndarray, predictions: np.ndarray
    ) -> Dict[str, float]:
        """Calculate performance metrics"""
        residuals = actuals - predictions
        n = len(actuals)

        # RMSE
        rmse = np.sqrt(np.mean(residuals**2))

        # MAE
        mae = np.mean(np.abs(residuals))

        # MAPE (avoid division by zero)
        mask = actuals != 0
        mape = (
            np.mean(np.abs(residuals[mask] / actuals[mask]) * 100)
            if mask.any()
            else np.nan
        )

        # SMAPE
        smape = np.mean(
            2 * np.abs(residuals) / (np.abs(actuals) + np.abs(predictions)) * 100
        )

        # R-squared
        ss_res = np.sum(residuals**2)
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

    def extract_outputs(
        self, fit: ModelFit
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract comprehensive three-DataFrame output.

        Returns:
            Tuple of (outputs, coefficients, stats)

            - Outputs: Observation-level results with actuals, fitted, residuals
            - Coefficients: Combined ARIMA parameters and XGBoost parameters
            - Stats: Comprehensive metrics by split + model info
        """
        arima_model = fit.fit_data["arima_model"]
        xgb_model = fit.fit_data["xgb_model"]

        # ====================
        # 1. OUTPUTS DataFrame
        # ====================
        outputs_list = []

        # Training data
        y_train = fit.fit_data.get("y_train")
        arima_fitted = fit.fit_data.get("arima_fitted")
        xgb_fitted = fit.fit_data.get("xgb_fitted")
        fitted = fit.fit_data.get("fitted")
        residuals = fit.fit_data.get("residuals")
        dates = fit.fit_data.get("dates")

        if y_train is not None and fitted is not None:
            forecast_train = pd.Series(y_train).combine_first(pd.Series(fitted)).values

            train_df = pd.DataFrame(
                {
                    "date": dates if dates is not None else np.arange(len(y_train)),
                    "actuals": y_train,
                    "arima_fitted": arima_fitted,
                    "xgb_fitted": xgb_fitted,
                    "fitted": fitted,
                    "forecast": forecast_train,
                    "residuals": residuals if residuals is not None else y_train - fitted,
                    "split": "train",
                }
            )

            # Add model metadata
            train_df["model"] = (
                fit.model_name if fit.model_name else fit.spec.model_type
            )
            train_df["model_group_name"] = (
                fit.model_group_name if fit.model_group_name else ""
            )
            train_df["group"] = "global"

            outputs_list.append(train_df)

        # Test data (if evaluated)
        if "test_predictions" in fit.evaluation_data:
            test_data = fit.evaluation_data["test_data"]
            test_preds = fit.evaluation_data["test_predictions"]
            outcome_col = fit.evaluation_data["outcome_col"]
            date_col = fit.fit_data["date_col"]
            exog_vars = fit.fit_data.get("exog_vars", [])

            test_actuals = test_data[outcome_col].values
            test_predictions = test_preds[".pred"].values
            test_residuals = test_actuals - test_predictions

            # Get test dates
            test_dates = None
            if date_col == '__index__':
                test_dates = test_data.index.values
            elif date_col and date_col in test_data.columns:
                test_dates = test_data[date_col].values
            else:
                # Try to find any datetime column
                for col in test_data.columns:
                    if pd.api.types.is_datetime64_any_dtype(test_data[col]):
                        test_dates = test_data[col].values
                        break
            if test_dates is None:
                test_dates = np.arange(len(test_actuals))

            # Calculate component predictions for test data
            test_arima_fitted = None
            test_xgb_fitted = None

            try:
                n_periods = len(test_data)

                # Get ARIMA component for test data
                if exog_vars:
                    missing = [v for v in exog_vars if v not in test_data.columns]
                    if not missing:
                        exog_test = test_data[exog_vars]
                        arima_forecast = arima_model.forecast(steps=n_periods, exog=exog_test)
                        test_arima_fitted = arima_forecast.values
                else:
                    arima_forecast = arima_model.forecast(steps=n_periods)
                    test_arima_fitted = arima_forecast.values

                # Get XGBoost component for test data
                if exog_vars and not missing:
                    X_boost_test = exog_test.values
                else:
                    # Use time index
                    last_train_idx = fit.fit_data["n_obs"]
                    X_boost_test = np.arange(last_train_idx, last_train_idx + n_periods).reshape(-1, 1)

                test_xgb_fitted = xgb_model.predict(X_boost_test)

            except Exception:
                # If component calculation fails, leave as None
                pass

            forecast_test = pd.Series(test_actuals).combine_first(
                pd.Series(test_predictions)
            ).values

            test_df = pd.DataFrame(
                {
                    "date": test_dates,
                    "actuals": test_actuals,
                    "arima_fitted": test_arima_fitted if test_arima_fitted is not None else np.nan,
                    "xgb_fitted": test_xgb_fitted if test_xgb_fitted is not None else np.nan,
                    "fitted": test_predictions,
                    "forecast": forecast_test,
                    "residuals": test_residuals,
                    "split": "test",
                }
            )

            # Add model metadata
            test_df["model"] = (
                fit.model_name if fit.model_name else fit.spec.model_type
            )
            test_df["model_group_name"] = (
                fit.model_group_name if fit.model_group_name else ""
            )
            test_df["group"] = "global"

            outputs_list.append(test_df)

        outputs = (
            pd.concat(outputs_list, ignore_index=True)
            if outputs_list
            else pd.DataFrame()
        )

        # ====================
        # 2. COEFFICIENTS DataFrame
        # ====================
        coef_rows = []

        # ARIMA parameters
        if hasattr(arima_model, "params") and arima_model.params is not None:
            param_names = (
                arima_model.param_names
                if hasattr(arima_model, "param_names")
                else list(arima_model.params.index)
            )

            for i, param_name in enumerate(param_names):
                coef_rows.append(
                    {
                        "variable": f"arima_{param_name}",
                        "coefficient": arima_model.params.iloc[i],
                        "std_error": (
                            arima_model.bse.iloc[i] if hasattr(arima_model, "bse") else np.nan
                        ),
                        "t_stat": (
                            arima_model.tvalues.iloc[i]
                            if hasattr(arima_model, "tvalues")
                            else np.nan
                        ),
                        "p_value": (
                            arima_model.pvalues.iloc[i]
                            if hasattr(arima_model, "pvalues")
                            else np.nan
                        ),
                        "ci_0.025": np.nan,
                        "ci_0.975": np.nan,
                        "vif": np.nan,
                    }
                )

        # XGBoost parameters (hyperparameters)
        xgb_params = fit.fit_data.get("xgb_params", {})
        for param_name, param_value in xgb_params.items():
            if param_name != "random_state" and param_name != "verbosity":
                coef_rows.append(
                    {
                        "variable": f"xgb_{param_name}",
                        "coefficient": param_value,
                        "std_error": np.nan,
                        "t_stat": np.nan,
                        "p_value": np.nan,
                        "ci_0.025": np.nan,
                        "ci_0.975": np.nan,
                        "vif": np.nan,
                    }
                )

        coefficients = pd.DataFrame(coef_rows)

        # Add model metadata
        coefficients["model"] = (
            fit.model_name if fit.model_name else fit.spec.model_type
        )
        coefficients["model_group_name"] = (
            fit.model_group_name if fit.model_group_name else ""
        )
        coefficients["group"] = "global"

        # ====================
        # 3. STATS DataFrame
        # ====================
        stats_rows = []

        # Training metrics
        if y_train is not None and fitted is not None:
            train_metrics = self._calculate_metrics(y_train, fitted)

            for metric_name, value in train_metrics.items():
                stats_rows.append(
                    {
                        "metric": metric_name,
                        "value": value,
                        "split": "train",
                    }
                )

        # Test metrics (if evaluated)
        if "test_predictions" in fit.evaluation_data:
            test_actuals = test_data[outcome_col].values
            test_forecast = test_preds[".pred"].values

            test_metrics = self._calculate_metrics(test_actuals, test_forecast)

            for metric_name, value in test_metrics.items():
                stats_rows.append(
                    {
                        "metric": metric_name,
                        "value": value,
                        "split": "test",
                    }
                )

        # Model information
        order = fit.fit_data.get("order", (0, 0, 0))
        seasonal_order = fit.fit_data.get("seasonal_order", (0, 0, 0, 0))
        blueprint = fit.blueprint

        stats_rows.extend(
            [
                {
                    "metric": "formula",
                    "value": blueprint.get("formula", "")
                    if isinstance(blueprint, dict)
                    else "",
                    "split": "",
                },
                {"metric": "model_type", "value": "arima_boost", "split": ""},
                {"metric": "arima_order", "value": str(order), "split": ""},
                {
                    "metric": "arima_seasonal_order",
                    "value": str(seasonal_order),
                    "split": "",
                },
                {
                    "metric": "xgb_n_estimators",
                    "value": xgb_params.get("n_estimators", np.nan),
                    "split": "",
                },
                {
                    "metric": "xgb_max_depth",
                    "value": xgb_params.get("max_depth", np.nan),
                    "split": "",
                },
                {
                    "metric": "xgb_learning_rate",
                    "value": xgb_params.get("learning_rate", np.nan),
                    "split": "",
                },
                {
                    "metric": "arima_aic",
                    "value": arima_model.aic if hasattr(arima_model, "aic") else np.nan,
                    "split": "",
                },
                {
                    "metric": "arima_bic",
                    "value": arima_model.bic if hasattr(arima_model, "bic") else np.nan,
                    "split": "",
                },
                {
                    "metric": "n_obs_train",
                    "value": fit.fit_data.get("n_obs", 0),
                    "split": "train",
                },
            ]
        )

        # Add dates if available
        if dates is not None and len(dates) > 0:
            stats_rows.extend(
                [
                    {"metric": "train_start_date", "value": str(dates[0]), "split": "train"},
                    {"metric": "train_end_date", "value": str(dates[-1]), "split": "train"},
                ]
            )

        stats = pd.DataFrame(stats_rows)

        # Add model metadata
        stats["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        stats["model_group_name"] = (
            fit.model_group_name if fit.model_group_name else ""
        )
        stats["group"] = "global"

        return outputs, coefficients, stats
