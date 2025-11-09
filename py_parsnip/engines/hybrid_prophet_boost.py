"""
Hybrid Prophet + XGBoost engine

This engine implements a two-stage hybrid forecasting model:
1. Stage 1: Fit Prophet to capture trend, seasonality, and holiday effects
2. Stage 2: Fit XGBoost on Prophet residuals to capture non-linear patterns
3. Prediction: base_pred (Prophet) + residual_pred (XGBoost)

The hybrid approach is effective when data has strong seasonal patterns
plus complex non-linear relationships.
"""

from typing import Dict, Any, Literal
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData
from py_parsnip.utils import _infer_date_column, _parse_ts_formula, _expand_dot_notation


@register_engine("prophet_boost", "hybrid_prophet_xgboost")
class HybridProphetBoostEngine(Engine):
    """
    Hybrid Prophet + XGBoost engine for time series forecasting.

    Parameter mapping:
    Prophet parameters → prophet.Prophet (same names)
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
        # Prophet params (no translation needed)
        "growth": "growth",
        "changepoint_prior_scale": "changepoint_prior_scale",
        "seasonality_prior_scale": "seasonality_prior_scale",
        "seasonality_mode": "seasonality_mode",
        "n_changepoints": "n_changepoints",
        "changepoint_range": "changepoint_range",
        # XGBoost params
        "trees": "n_estimators",
        "tree_depth": "max_depth",
        "learn_rate": "learning_rate",
        "min_n": "min_child_weight",
        "loss_reduction": "gamma",
        "sample_size": "subsample",
        "mtry": "colsample_bytree",
    }

    @staticmethod
    def _create_xgb_features(dates: pd.Series) -> np.ndarray:
        """
        Create cyclical time features for XGBoost to capture seasonality.

        NOTE: This method is DEPRECATED and should only be used as fallback
        when no exogenous regressors are provided in the formula.
        The preferred approach is to use actual exogenous variables from the formula.

        Features include (NO days_since_start to avoid extrapolation):
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        - day_of_month: Day within month (1-31)
        - day_of_year: Day within year (1-365/366)
        - month: Month of year (1-12)
        - week_of_year: Week within year (1-52)
        - Cyclical encodings (sin/cos) for proper periodicity:
          - sin/cos day_of_week (weekly patterns)
          - sin/cos day_of_year (yearly patterns)
          - sin/cos month (monthly patterns)

        Args:
            dates: pandas Series of datetime values

        Returns:
            numpy array of shape (n_samples, 11) with cyclical time features
            (removed days_since_start to prevent extrapolation issues)
        """
        dates = pd.to_datetime(dates)

        # Categorical time features
        # Use .dt accessor for Series, direct access for DatetimeIndex
        if isinstance(dates, pd.Series):
            day_of_week = dates.dt.dayofweek.values          # 0-6
            day_of_month = dates.dt.day.values                # 1-31
            day_of_year = dates.dt.dayofyear.values           # 1-365
            month = dates.dt.month.values                     # 1-12
            week_of_year = dates.dt.isocalendar().week.values # 1-52
        else:
            day_of_week = dates.dayofweek.values          # 0-6
            day_of_month = dates.day.values                # 1-31
            day_of_year = dates.dayofyear.values           # 1-365
            month = dates.month.values                     # 1-12
            week_of_year = dates.isocalendar().week.values # 1-52

        # Cyclical encodings for smooth periodic patterns
        # Day of week (weekly seasonality)
        sin_day_of_week = np.sin(2 * np.pi * day_of_week / 7)
        cos_day_of_week = np.cos(2 * np.pi * day_of_week / 7)

        # Day of year (yearly seasonality)
        sin_day_of_year = np.sin(2 * np.pi * day_of_year / 365.25)
        cos_day_of_year = np.cos(2 * np.pi * day_of_year / 365.25)

        # Month (monthly seasonality)
        sin_month = np.sin(2 * np.pi * month / 12)
        cos_month = np.cos(2 * np.pi * month / 12)

        # Combine all features (REMOVED days_since_start)
        features = np.column_stack([
            day_of_week,
            day_of_month,
            day_of_year,
            month,
            week_of_year,
            sin_day_of_week,
            cos_day_of_week,
            sin_day_of_year,
            cos_day_of_year,
            sin_month,
            cos_month,
        ])

        return features

    def fit_raw(
        self, spec: ModelSpec, data: pd.DataFrame, formula: str, date_col: str = None
    ) -> tuple[Dict[str, Any], Any]:
        """
        Fit hybrid Prophet + XGBoost model using raw data.

        Strategy:
        1. Fit Prophet model to capture trend and seasonality
        2. Calculate Prophet residuals
        3. Fit XGBoost on residuals with time features
        4. Store both models

        Args:
            spec: ModelSpec with hybrid configuration
            data: Training data DataFrame
            formula: Formula string (e.g., "sales ~ date")
            date_col: Name of date column, or '__index__' for DatetimeIndex

        Returns:
            Tuple of (fit_data dict, blueprint)
        """
        from prophet import Prophet
        from xgboost import XGBRegressor

        # Infer date column from data
        inferred_date_col = _infer_date_column(
            data,
            spec_date_col=spec.args.get("date_col") if spec.args else None,
            fit_date_col=date_col
        )

        # Parse formula to extract outcome and exogenous regressors
        outcome_name, exog_vars = _parse_ts_formula(formula, inferred_date_col)

        # Expand "." notation to all columns except outcome and date
        exog_vars = _expand_dot_notation(exog_vars, data, outcome_name, inferred_date_col)

        # Validate outcome exists
        if outcome_name not in data.columns:
            raise ValueError(f"Outcome '{outcome_name}' not found in data")

        # Extract exogenous regressors if provided (matches R modeltime behavior)
        use_exog = False
        X_boost_cols = None

        if exog_vars:
            # Validate all exogenous variables exist
            missing_vars = [v for v in exog_vars if v not in data.columns]
            if missing_vars:
                raise ValueError(
                    f"Exogenous variables {missing_vars} not found in data. "
                    f"Available columns: {list(data.columns)}"
                )

            # Extract exogenous variables for XGBoost
            X_boost_data = data[exog_vars].copy()

            # Handle categorical variables
            from sklearn.preprocessing import LabelEncoder
            label_encoders = {}
            for col in X_boost_data.columns:
                if X_boost_data[col].dtype == 'object' or X_boost_data[col].dtype.name == 'category':
                    le = LabelEncoder()
                    X_boost_data[col] = le.fit_transform(X_boost_data[col].astype(str))
                    label_encoders[col] = le

            use_exog = True
            X_boost_cols = exog_vars

        # ==================
        # STAGE 1: Fit Prophet
        # ==================
        # Get date series
        if inferred_date_col == '__index__':
            date_series = data.index
        else:
            date_series = data[inferred_date_col]

        # Create Prophet-format DataFrame
        prophet_df = pd.DataFrame(
            {
                "ds": date_series,
                "y": data[outcome_name],
            }
        )

        # Extract Prophet parameters
        args = spec.args
        prophet_params = {
            "growth": args.get("growth", "linear"),
            "changepoint_prior_scale": args.get("changepoint_prior_scale", 0.05),
            "seasonality_prior_scale": args.get("seasonality_prior_scale", 10.0),
            "seasonality_mode": args.get("seasonality_mode", "additive"),
            "n_changepoints": args.get("n_changepoints", 25),
            "changepoint_range": args.get("changepoint_range", 0.8),
        }

        # Create and fit Prophet model
        prophet_model = Prophet(**prophet_params)

        # Suppress Prophet logging
        import logging

        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

        prophet_model.fit(prophet_df)

        # Get Prophet fitted values on training data
        prophet_forecast = prophet_model.predict(prophet_df[["ds"]])
        prophet_fitted = prophet_forecast["yhat"].values

        # Calculate Prophet residuals
        actuals = prophet_df["y"].values
        prophet_residuals = actuals - prophet_fitted

        # =======================
        # STAGE 2: Fit XGBoost on residuals
        # =======================
        # Determine XGBoost features: use exogenous regressors if available,
        # otherwise use cyclical time features (fallback)
        if use_exog:
            # Use actual exogenous regressors (matches R modeltime behavior)
            X_boost = X_boost_data.values
        else:
            # Fallback: use cyclical time features (deprecated, see docstring)
            import warnings
            warnings.warn(
                "No exogenous regressors provided in formula. "
                "Using auto-generated cyclical time features as fallback. "
                "For better performance, provide exogenous variables: "
                "e.g., 'y ~ date + feature1 + feature2'",
                UserWarning
            )
            dates = pd.to_datetime(prophet_df["ds"])
            X_boost = self._create_xgb_features(dates)

        # Target for XGBoost is Prophet residuals
        y_boost = prophet_residuals

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
        # Final fitted values = Prophet + XGBoost
        final_fitted = prophet_fitted + xgb_fitted

        # Final residuals
        final_residuals = actuals - final_fitted

        # Extract dates
        dates_values = prophet_df["ds"].values

        # Store date_min for potential fallback (though not used with exog regressors)
        dates_pd = pd.to_datetime(prophet_df["ds"])
        date_min_val = dates_pd.min()

        # Create blueprint
        blueprint = {
            "formula": formula,
            "outcome_name": outcome_name,
            "date_col": inferred_date_col,
            "prophet_params": prophet_params,
            "xgb_params": xgb_params,
            "use_exog": use_exog,
            "exog_cols": X_boost_cols,
        }

        # Return fit data
        fit_data = {
            "prophet_model": prophet_model,
            "xgb_model": xgb_model,
            "n_obs": len(prophet_df),
            "date_col": inferred_date_col,
            "outcome_name": outcome_name,
            "prophet_df": prophet_df,
            "y_train": actuals,
            "prophet_fitted": prophet_fitted,
            "xgb_fitted": xgb_fitted,
            "fitted": final_fitted,
            "residuals": final_residuals,
            "dates": dates_values,
            "prophet_params": prophet_params,
            "xgb_params": xgb_params,
            "date_min": date_min_val,  # Keep for backward compatibility
            "use_exog": use_exog,
            "exog_cols": X_boost_cols,
            "label_encoders": label_encoders if use_exog else None,
        }

        return fit_data, blueprint

    def fit(self, spec: ModelSpec, molded: MoldedData) -> Dict[str, Any]:
        """Not used for hybrid Prophet+XGBoost - use fit_raw() instead"""
        raise NotImplementedError("Hybrid Prophet+XGBoost uses fit_raw() instead of fit()")

    def predict(
        self, fit: ModelFit, molded: MoldedData, type: str
    ) -> pd.DataFrame:
        """Not used for hybrid Prophet+XGBoost - use predict_raw() instead"""
        raise NotImplementedError("Hybrid Prophet+XGBoost uses predict_raw() instead of predict()")

    def predict_raw(
        self,
        fit: ModelFit,
        new_data: pd.DataFrame,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make predictions using fitted hybrid Prophet+XGBoost model.

        Strategy:
        1. Get Prophet predictions
        2. Get XGBoost predictions (on time features)
        3. Combine: prophet_pred + xgb_pred

        Args:
            fit: ModelFit with fitted hybrid model
            new_data: DataFrame with predictor column
            type: Prediction type
                - "numeric": Point forecasts
                - "conf_int": Not supported for hybrid models

        Returns:
            DataFrame with predictions
        """
        if type not in ("numeric",):
            raise ValueError(
                f"prophet_boost currently supports type='numeric' only, got '{type}'"
            )

        prophet_model = fit.fit_data["prophet_model"]
        xgb_model = fit.fit_data["xgb_model"]
        date_col = fit.fit_data["date_col"]
        use_exog = fit.fit_data.get("use_exog", False)
        exog_cols = fit.fit_data.get("exog_cols")
        label_encoders = fit.fit_data.get("label_encoders")

        # Get date series from new_data
        if date_col == '__index__':
            date_series = new_data.index
        else:
            if date_col not in new_data.columns:
                raise ValueError(
                    f"Date column '{date_col}' not found in new data. "
                    f"Available columns: {list(new_data.columns)}"
                )
            date_series = new_data[date_col]

        # ==================
        # Prophet predictions
        # ==================
        # Create future DataFrame for Prophet
        future = pd.DataFrame({"ds": date_series})

        # Make Prophet predictions
        prophet_forecast = prophet_model.predict(future)
        prophet_pred = prophet_forecast["yhat"].values

        # ==================
        # XGBoost predictions
        # ==================
        # Use exogenous regressors if available, otherwise use cyclical features
        if use_exog and exog_cols:
            # Extract and encode exogenous variables from new_data
            missing_cols = [col for col in exog_cols if col not in new_data.columns]
            if missing_cols:
                raise ValueError(
                    f"Exogenous variables {missing_cols} not found in new data. "
                    f"Required: {exog_cols}, Available: {list(new_data.columns)}"
                )

            X_boost_pred = new_data[exog_cols].copy()

            # Apply same label encoding as training
            if label_encoders:
                for col, le in label_encoders.items():
                    if col in X_boost_pred.columns:
                        X_boost_pred[col] = le.transform(X_boost_pred[col].astype(str))

            X_boost = X_boost_pred.values
        else:
            # Fallback: use cyclical time features
            dates = pd.to_datetime(date_series)
            X_boost = self._create_xgb_features(dates)

        # Get XGBoost predictions
        xgb_pred = xgb_model.predict(X_boost)

        # ==================
        # Combine predictions
        # ==================
        final_forecast = prophet_pred + xgb_pred

        # Return predictions with date index
        result = pd.DataFrame({".pred": final_forecast}, index=date_series)
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
            - Coefficients: Combined Prophet and XGBoost parameters
            - Stats: Comprehensive metrics by split + model info
        """
        prophet_model = fit.fit_data["prophet_model"]
        xgb_model = fit.fit_data["xgb_model"]

        # ====================
        # 1. OUTPUTS DataFrame
        # ====================
        outputs_list = []

        # Training data
        y_train = fit.fit_data.get("y_train")
        prophet_fitted = fit.fit_data.get("prophet_fitted")
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
                    "prophet_fitted": prophet_fitted,
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
            date_min = fit.fit_data["date_min"]

            test_actuals = test_data[outcome_col].values
            test_predictions = test_preds[".pred"].values
            test_residuals = test_actuals - test_predictions

            # Get test dates
            if date_col == '__index__':
                test_dates = test_data.index.values
                date_series = test_data.index
            elif date_col in test_data.columns:
                test_dates = test_data[date_col].values
                date_series = test_data[date_col]
            else:
                test_dates = np.arange(len(test_actuals))
                date_series = None

            # Calculate component predictions for test data
            test_prophet_fitted = None
            test_xgb_fitted = None

            if date_series is not None:
                # Get Prophet component for test data
                future_test = pd.DataFrame({"ds": date_series})
                prophet_forecast = prophet_model.predict(future_test)
                test_prophet_fitted = prophet_forecast["yhat"].values

                # Get XGBoost component for test data
                use_exog = fit.fit_data.get("use_exog", False)
                exog_cols = fit.fit_data.get("exog_cols")
                label_encoders = fit.fit_data.get("label_encoders")

                if use_exog and exog_cols:
                    # Use exogenous regressors from test data
                    if all(col in test_data.columns for col in exog_cols):
                        X_boost_test_data = test_data[exog_cols].copy()

                        # Apply same label encoding as training
                        if label_encoders:
                            for col, le in label_encoders.items():
                                if col in X_boost_test_data.columns:
                                    X_boost_test_data[col] = le.transform(X_boost_test_data[col].astype(str))

                        X_boost_test = X_boost_test_data.values
                    else:
                        # Fallback if exogenous columns missing
                        dates_test = pd.to_datetime(date_series)
                        X_boost_test = self._create_xgb_features(dates_test)
                else:
                    # Use cyclical time features
                    dates_test = pd.to_datetime(date_series)
                    X_boost_test = self._create_xgb_features(dates_test)

                test_xgb_fitted = xgb_model.predict(X_boost_test)

            forecast_test = pd.Series(test_actuals).combine_first(
                pd.Series(test_predictions)
            ).values

            test_df = pd.DataFrame(
                {
                    "date": test_dates,
                    "actuals": test_actuals,
                    "prophet_fitted": test_prophet_fitted if test_prophet_fitted is not None else np.nan,
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

        # Prophet hyperparameters
        prophet_params = fit.fit_data.get("prophet_params", {})
        for param_name, param_value in prophet_params.items():
            coef_rows.append(
                {
                    "variable": f"prophet_{param_name}",
                    "coefficient": param_value,
                    "std_error": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "ci_0.025": np.nan,
                    "ci_0.975": np.nan,
                    "vif": np.nan,
                }
            )

        # Number of changepoints detected
        if hasattr(prophet_model, "changepoints"):
            n_changepoints = len(prophet_model.changepoints)
            coef_rows.append(
                {
                    "variable": "prophet_n_changepoints_detected",
                    "coefficient": n_changepoints,
                    "std_error": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
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
        blueprint = fit.blueprint
        growth = prophet_params.get("growth", "linear")
        seas_mode = prophet_params.get("seasonality_mode", "additive")

        stats_rows.extend(
            [
                {
                    "metric": "formula",
                    "value": blueprint.get("formula", "")
                    if isinstance(blueprint, dict)
                    else "",
                    "split": "",
                },
                {"metric": "model_type", "value": "prophet_boost", "split": ""},
                {"metric": "prophet_growth", "value": growth, "split": ""},
                {"metric": "prophet_seasonality_mode", "value": seas_mode, "split": ""},
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
