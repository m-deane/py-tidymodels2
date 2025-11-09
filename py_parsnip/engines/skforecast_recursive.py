"""
Skforecast engine for recursive forecasting

Implements recursive multi-step forecasting using skforecast's ForecasterAutoreg.
"""

from typing import Dict, Any, Tuple, Optional, Union, List
import pandas as pd
import numpy as np
from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_parsnip.utils import _infer_date_column, _parse_ts_formula, _expand_dot_notation


@register_engine("recursive_reg", "skforecast")
class SkforecastRecursiveEngine(Engine):
    """
    Skforecast recursive forecasting engine.

    Uses ForecasterAutoreg from skforecast to enable any sklearn-compatible
    model for multi-step time series forecasting via the recursive strategy.
    """

    param_map = {}  # No parameter translation needed

    def translate_params(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """No translation needed for recursive_reg parameters"""
        return args

    def fit_raw(
        self, spec: ModelSpec, data: pd.DataFrame, formula: str, date_col: str = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Fit recursive forecasting model using raw data.

        Uses raw data path because time series requires special handling
        and we need direct access to the time-indexed data.

        Args:
            spec: Model specification with base_model, lags, differentiation
            data: DataFrame with datetime index and target column
            formula: Formula like "y ~ date" or "y ~ ."
            date_col: Name of date column, or '__index__' for DatetimeIndex

        Returns:
            Tuple of (fit_data_dict, blueprint_dict)
        """
        from skforecast.recursive import ForecasterRecursive

        # Infer date column from data
        inferred_date_col = _infer_date_column(
            data,
            spec_date_col=spec.args.get("date_col") if spec.args else None,
            fit_date_col=date_col
        )

        # Parse formula to get outcome and exogenous variables
        outcome_name, exog_vars = _parse_ts_formula(formula, inferred_date_col)

        # Expand "." notation to all columns except outcome and date
        exog_vars = _expand_dot_notation(exog_vars, data, outcome_name, inferred_date_col)

        if outcome_name not in data.columns:
            raise ValueError(
                f"Outcome '{outcome_name}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )

        # Extract parameters
        args = dict(spec.args)
        base_model_spec = args["base_model"]
        lags = args["lags"]
        differentiation = args.get("differentiation")

        # Get the base sklearn model by fitting it
        # We'll use a simple formula to get the sklearn estimator
        if hasattr(base_model_spec, "fit"):
            # Ensure mode is set for models that need it (like rand_forest)
            if base_model_spec.mode == "unknown":
                base_model_spec = base_model_spec.set_mode("regression")

            # Fit base model to get sklearn estimator
            # Use a dummy fit to extract the underlying sklearn model
            dummy_data = pd.DataFrame({"y": [1, 2, 3], "x": [1, 2, 3]})
            try:
                base_fit = base_model_spec.fit(dummy_data, "y ~ x")
                if "model" in base_fit.fit_data:
                    regressor = base_fit.fit_data["model"]
                else:
                    raise ValueError("Base model fit_data missing 'model' key")
            except Exception as e:
                raise ValueError(
                    f"Failed to extract sklearn model from base_model: {e}"
                )
        else:
            raise TypeError(
                f"base_model must be a ModelSpec with fit() method, "
                f"got {type(base_model_spec)}"
            )

        # Create forecaster
        forecaster = ForecasterRecursive(
            regressor=regressor, lags=lags, differentiation=differentiation
        )

        # Handle __index__ case for DatetimeIndex
        if inferred_date_col == '__index__':
            # Data already has DatetimeIndex
            y = data[outcome_name]

            # Ensure y has a frequency if it's a DatetimeIndex
            if isinstance(y.index, pd.DatetimeIndex) and y.index.freq is None:
                freq = pd.infer_freq(y.index)
                if freq:
                    y.index = pd.DatetimeIndex(y.index, freq=freq)
                else:
                    # If freq cannot be inferred, use the most common diff
                    diffs = y.index[1:] - y.index[:-1]
                    most_common_diff = diffs.value_counts().idxmax()
                    y = y.asfreq(most_common_diff)

            # Get exogenous variables if present
            if exog_vars:
                exog = data[exog_vars]
                # Set frequency for exog
                if isinstance(exog.index, pd.DatetimeIndex) and exog.index.freq is None:
                    exog.index = y.index  # Use same index as y
            else:
                exog = None
        else:
            # Set date column as index
            data_indexed = data.set_index(inferred_date_col)
            y = data_indexed[outcome_name]

            # Ensure y has a frequency
            if isinstance(y.index, pd.DatetimeIndex) and y.index.freq is None:
                freq = pd.infer_freq(y.index)
                if freq:
                    y.index = pd.DatetimeIndex(y.index, freq=freq)
                else:
                    diffs = y.index[1:] - y.index[:-1]
                    most_common_diff = diffs.value_counts().idxmax()
                    y = y.asfreq(most_common_diff)

            # Get exogenous variables if present
            if exog_vars:
                exog = data_indexed[exog_vars]
                # Set frequency for exog
                if isinstance(exog.index, pd.DatetimeIndex) and exog.index.freq is None:
                    exog.index = y.index  # Use same index as y
            else:
                exog = None

        # Fit with store_in_sample_residuals=True for prediction intervals
        forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)

        # Get in-sample residuals (skforecast stores these internally)
        # We'll calculate fitted values from actuals - residuals
        if hasattr(forecaster, 'in_sample_residuals') and forecaster.in_sample_residuals is not None:
            # in_sample_residuals is a dict with key as step, we want step 1
            residuals = forecaster.in_sample_residuals.get(1, None)
            if residuals is not None:
                # Fitted = Actual - Residual (for the observations we have residuals for)
                # Note: first few observations won't have residuals (due to lags)
                y_train_pred = np.full(len(y), np.nan)
                n_residuals = len(residuals)
                y_train_pred[-n_residuals:] = y.values[-n_residuals:] - residuals
            else:
                # Fallback: use NaN for training predictions
                y_train_pred = np.full(len(y), np.nan)
        else:
            # Fallback: use NaN for training predictions
            y_train_pred = np.full(len(y), np.nan)

        # Prepare fit_data
        fit_data = {
            "forecaster": forecaster,
            "base_model_spec": base_model_spec,
            "outcome_name": outcome_name,
            "y_train": y.values,
            "y_train_pred": y_train_pred.values if isinstance(y_train_pred, pd.Series) else y_train_pred,
            "lags": lags,
            "differentiation": differentiation,
            "exog_vars": exog_vars,
            "date_col": inferred_date_col,
            "train_index": y.index,
        }

        # Create simple blueprint
        blueprint = {
            "formula": formula,
            "outcome_name": outcome_name,
            "exog_vars": exog_vars,
            "date_col": inferred_date_col,
        }

        return fit_data, blueprint

    def fit(self, spec: ModelSpec, molded: Any) -> Dict[str, Any]:
        """Not implemented - use fit_raw() for time series"""
        raise NotImplementedError(
            "recursive_reg uses raw data path. Use fit_raw() instead."
        )

    def predict_raw(
        self, fit: ModelFit, new_data: pd.DataFrame, type: str
    ) -> pd.DataFrame:
        """
        Make predictions on new data.

        Args:
            fit: Fitted model
            new_data: DataFrame with exogenous variables (if any)
            type: Prediction type ("numeric" for point forecasts)

        Returns:
            DataFrame with predictions indexed by forecast dates
        """
        forecaster = fit.fit_data["forecaster"]
        exog_vars = fit.fit_data["exog_vars"]
        date_col = fit.fit_data["date_col"]

        # Determine forecast horizon
        steps = len(new_data) if new_data is not None and len(new_data) > 0 else 1

        # Get exogenous variables if provided
        exog = None
        if exog_vars and new_data is not None and len(new_data) > 0:
            missing_cols = set(exog_vars) - set(new_data.columns)
            if missing_cols:
                raise ValueError(
                    f"Missing exogenous columns in new_data: {missing_cols}"
                )
            # Handle __index__ case
            if date_col == '__index__':
                exog = new_data[exog_vars]
            else:
                exog = new_data.set_index(date_col)[exog_vars] if date_col in new_data.columns else new_data[exog_vars]

        # Make predictions
        if type == "numeric":
            predictions = forecaster.predict(steps=steps, exog=exog)

            # Create result DataFrame with date index if available
            if new_data is not None and len(new_data) > 0:
                result_index = new_data.index[:steps]
            else:
                # Generate future index based on training data
                train_index = fit.fit_data["train_index"]
                if isinstance(train_index, pd.DatetimeIndex):
                    freq = pd.infer_freq(train_index)
                    result_index = pd.date_range(
                        start=train_index[-1], periods=steps + 1, freq=freq
                    )[1:]
                else:
                    result_index = range(len(train_index), len(train_index) + steps)

            return pd.DataFrame(
                {".pred": predictions.values if isinstance(predictions, pd.Series) else predictions},
                index=result_index
            )

        elif type == "pred_int":
            # Get prediction intervals
            predictions = forecaster.predict_interval(
                steps=steps, exog=exog, interval=[5, 95]
            )

            if new_data is not None and len(new_data) > 0:
                result_index = new_data.index[:steps]
            else:
                train_index = fit.fit_data["train_index"]
                if isinstance(train_index, pd.DatetimeIndex):
                    freq = pd.infer_freq(train_index)
                    result_index = pd.date_range(
                        start=train_index[-1], periods=steps + 1, freq=freq
                    )[1:]
                else:
                    result_index = range(len(train_index), len(train_index) + steps)

            return pd.DataFrame(
                {
                    ".pred": predictions["pred"].values,
                    ".pred_lower": predictions["lower_bound"].values,
                    ".pred_upper": predictions["upper_bound"].values,
                },
                index=result_index,
            )

        else:
            raise ValueError(f"Prediction type '{type}' not supported")

    def predict(self, fit: ModelFit, molded: Any, type: str) -> pd.DataFrame:
        """Not implemented - use predict_raw() for time series"""
        raise NotImplementedError(
            "recursive_reg uses raw data path. Use predict_raw() instead."
        )

    def extract_outputs(
        self, fit: ModelFit
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract three-DataFrame output structure.

        Returns:
            Tuple of (outputs, coefficients, stats) DataFrames
        """
        forecaster = fit.fit_data["forecaster"]
        outcome_name = fit.fit_data["outcome_name"]
        y_train = fit.fit_data["y_train"]
        y_train_pred = fit.fit_data["y_train_pred"]
        train_index = fit.fit_data["train_index"]
        lags = fit.fit_data["lags"]

        # ====================
        # 1. OUTPUTS DataFrame
        # ====================
        outputs_list = []

        # Training data (in-sample predictions)
        residuals_train = y_train - y_train_pred
        train_df = pd.DataFrame(
            {
                "date": train_index if isinstance(train_index, pd.DatetimeIndex) else range(len(y_train)),
                "actuals": y_train,
                "fitted": y_train_pred,
                "forecast": pd.Series(y_train).combine_first(pd.Series(y_train_pred)).values,
                "residuals": residuals_train,
                "split": "train",
                "model": fit.model_name or fit.spec.model_type,
                "model_group_name": fit.model_group_name or "",
                "group": "global",
            }
        )
        outputs_list.append(train_df)

        # Test data (if evaluated)
        if "test_predictions" in fit.evaluation_data:
            test_preds = fit.evaluation_data["test_predictions"]
            test_data = fit.evaluation_data.get("test_data")

            if test_data is not None:
                test_actuals = test_data[outcome_name].values
                test_forecast = test_preds[".pred"].values
                residuals_test = test_actuals - test_forecast

                test_df = pd.DataFrame(
                    {
                        "date": test_preds.index,
                        "actuals": test_actuals,
                        "fitted": test_forecast,
                        "forecast": pd.Series(test_actuals).combine_first(pd.Series(test_forecast)).values,
                        "residuals": residuals_test,
                        "split": "test",
                        "model": fit.model_name or fit.spec.model_type,
                        "model_group_name": fit.model_group_name or "",
                        "group": "global",
                    }
                )
                outputs_list.append(test_df)

        outputs = pd.concat(outputs_list, ignore_index=True)

        # ====================
        # 2. COEFFICIENTS DataFrame
        # ====================
        # For recursive models, report feature importances from base model
        base_regressor = forecaster.regressor
        coef_rows = []

        if hasattr(base_regressor, "feature_importances_"):
            # Tree-based models
            importances = base_regressor.feature_importances_
            feature_names = [f"lag_{i}" for i in (lags if isinstance(lags, list) else range(1, lags + 1))]

            for name, importance in zip(feature_names, importances):
                coef_rows.append(
                    {
                        "variable": name,
                        "coefficient": importance,
                        "std_error": np.nan,
                        "t_stat": np.nan,
                        "p_value": np.nan,
                        "ci_0.025": np.nan,
                        "ci_0.975": np.nan,
                        "vif": np.nan,
                        "model": fit.model_name or fit.spec.model_type,
                        "model_group_name": fit.model_group_name or "",
                        "group": "global",
                    }
                )
        elif hasattr(base_regressor, "coef_"):
            # Linear models
            coefficients = base_regressor.coef_
            feature_names = [f"lag_{i}" for i in (lags if isinstance(lags, list) else range(1, lags + 1))]

            for name, coef in zip(feature_names, coefficients):
                coef_rows.append(
                    {
                        "variable": name,
                        "coefficient": coef,
                        "std_error": np.nan,  # Not available for sklearn linear models
                        "t_stat": np.nan,
                        "p_value": np.nan,
                        "ci_0.025": np.nan,
                        "ci_0.975": np.nan,
                        "vif": np.nan,
                        "model": fit.model_name or fit.spec.model_type,
                        "model_group_name": fit.model_group_name or "",
                        "group": "global",
                    }
                )

            # Add intercept if present
            if hasattr(base_regressor, "intercept_"):
                coef_rows.append(
                    {
                        "variable": "Intercept",
                        "coefficient": base_regressor.intercept_,
                        "std_error": np.nan,
                        "t_stat": np.nan,
                        "p_value": np.nan,
                        "ci_0.025": np.nan,
                        "ci_0.975": np.nan,
                        "vif": np.nan,
                        "model": fit.model_name or fit.spec.model_type,
                        "model_group_name": fit.model_group_name or "",
                        "group": "global",
                    }
                )

        coefficients = pd.DataFrame(coef_rows) if coef_rows else pd.DataFrame()

        # ====================
        # 3. STATS DataFrame
        # ====================
        stats_rows = []

        # Training metrics
        rmse_train = np.sqrt(np.mean(residuals_train**2))
        mae_train = np.mean(np.abs(residuals_train))
        mape_train = np.mean(np.abs(residuals_train / (y_train + 1e-10))) * 100

        # R-squared
        ss_res = np.sum(residuals_train**2)
        ss_tot = np.sum((y_train - np.mean(y_train))**2)
        r_squared_train = 1 - (ss_res / (ss_tot + 1e-10))

        stats_rows.extend([
            {"metric": "rmse", "value": rmse_train, "split": "train"},
            {"metric": "mae", "value": mae_train, "split": "train"},
            {"metric": "mape", "value": mape_train, "split": "train"},
            {"metric": "r_squared", "value": r_squared_train, "split": "train"},
        ])

        # Test metrics (if evaluated)
        if "test_predictions" in fit.evaluation_data:
            test_data = fit.evaluation_data.get("test_data")
            if test_data is not None:
                test_actuals = test_data[outcome_name].values
                test_preds = fit.evaluation_data["test_predictions"]
                test_forecast = test_preds[".pred"].values
                residuals_test = test_actuals - test_forecast

                rmse_test = np.sqrt(np.mean(residuals_test**2))
                mae_test = np.mean(np.abs(residuals_test))
                mape_test = np.mean(np.abs(residuals_test / (test_actuals + 1e-10))) * 100

                ss_res_test = np.sum(residuals_test**2)
                ss_tot_test = np.sum((test_actuals - np.mean(test_actuals))**2)
                r_squared_test = 1 - (ss_res_test / (ss_tot_test + 1e-10))

                stats_rows.extend([
                    {"metric": "rmse", "value": rmse_test, "split": "test"},
                    {"metric": "mae", "value": mae_test, "split": "test"},
                    {"metric": "mape", "value": mape_test, "split": "test"},
                    {"metric": "r_squared", "value": r_squared_test, "split": "test"},
                ])

        # Model metadata
        # Handle both dict and Blueprint object for blueprint
        formula = fit.blueprint.get("formula") if isinstance(fit.blueprint, dict) else fit.blueprint.formula

        stats_rows.extend([
            {"metric": "formula", "value": formula, "split": ""},
            {"metric": "n_obs_train", "value": len(y_train), "split": "train"},
            {"metric": "lags", "value": str(lags), "split": ""},
            {"metric": "differentiation", "value": str(fit.fit_data.get("differentiation", "None")), "split": ""},
            {"metric": "base_model", "value": fit.fit_data["base_model_spec"].model_type, "split": ""},
        ])

        # Add training date range
        dates = fit.fit_data.get("dates")
        if dates is not None and len(dates) > 0:
            stats_rows.extend([
                {"metric": "train_start_date", "value": str(dates[0]), "split": "train"},
                {"metric": "train_end_date", "value": str(dates[-1]), "split": "train"},
            ])

        stats = pd.DataFrame(stats_rows)
        stats["model"] = fit.model_name or fit.spec.model_type
        stats["model_group_name"] = fit.model_group_name or ""
        stats["group"] = "global"

        return outputs, coefficients, stats
