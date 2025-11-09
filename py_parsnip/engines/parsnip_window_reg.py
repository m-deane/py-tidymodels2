"""
Parsnip engine for sliding window forecasting

Implements window-based forecasting using rolling aggregates:
- mean: Simple moving average
- median: Median of window
- weighted_mean: Weighted moving average with custom weights
"""

from typing import Dict, Any, Literal, Optional
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit


@register_engine("window_reg", "parsnip")
class ParsnipWindowEngine(Engine):
    """
    Parsnip engine for sliding window forecasting.

    Methods:
    - mean: Simple moving average
    - median: Median of rolling window
    - weighted_mean: Weighted moving average (emphasize recent observations)
    """

    def fit(self, spec: ModelSpec, molded: Any) -> Dict[str, Any]:
        """Not used for window_reg - use fit_raw() instead"""
        raise NotImplementedError("window_reg uses fit_raw() instead of fit()")

    def predict(
        self,
        fit: ModelFit,
        molded: Any,
        type: str,
    ) -> pd.DataFrame:
        """Not used for window_reg - use predict_raw() instead"""
        raise NotImplementedError("window_reg uses predict_raw() instead of predict()")

    def fit_raw(
        self,
        spec: ModelSpec,
        data: pd.DataFrame,
        formula: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Fit window model (stores training data and parameters).

        Args:
            spec: ModelSpec with model configuration
            data: Training data
            formula: Formula string (e.g., "y ~ x" or "y ~ .")

        Returns:
            Tuple of (fit_data dict, blueprint dict)
        """
        # Parse formula to extract outcome variable
        parts = formula.split("~")
        if len(parts) != 2:
            raise ValueError(f"Invalid formula: {formula}")

        outcome_col = parts[0].strip()

        # Validate outcome column exists
        if outcome_col not in data.columns:
            raise ValueError(f"Outcome column '{outcome_col}' not found in data")

        # Auto-detect date column (optional, not required)
        date_col = None
        for col in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                date_col = col
                break
        # If still None, check for DatetimeIndex
        if date_col is None and isinstance(data.index, pd.DatetimeIndex):
            date_col = "__index__"

        # Extract parameters
        window_size = spec.args.get("window_size", 7)
        method = spec.args.get("method", "mean")
        weights = spec.args.get("weights", None)
        min_periods = spec.args.get("min_periods", None)

        # Validate parameters
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

        if method not in ["mean", "median", "weighted_mean"]:
            raise ValueError(
                f"method must be 'mean', 'median', or 'weighted_mean', got '{method}'"
            )

        if method == "weighted_mean":
            if weights is None:
                raise ValueError("weights required for weighted_mean method")
            if len(weights) != window_size:
                raise ValueError(
                    f"weights length ({len(weights)}) must equal window_size ({window_size})"
                )
            # Normalize weights to sum to 1.0
            weights = np.array(weights)
            weights = weights / weights.sum()

        if min_periods is not None:
            if min_periods < 1:
                raise ValueError(f"min_periods must be >= 1, got {min_periods}")
            if min_periods > window_size:
                raise ValueError(
                    f"min_periods ({min_periods}) cannot exceed window_size ({window_size})"
                )
        else:
            min_periods = window_size  # Default: require full window

        # Extract outcome values
        y = data[outcome_col].values
        n = len(y)

        if window_size > n:
            raise ValueError(
                f"window_size ({window_size}) cannot exceed data length ({n})"
            )

        # Compute fitted values (in-sample rolling forecasts)
        fitted = np.full(n, np.nan)

        for i in range(n):
            # Determine window bounds
            start_idx = max(0, i - window_size)
            window_data = y[start_idx:i]

            # Check if we have enough observations
            if len(window_data) >= min_periods:
                if method == "mean":
                    fitted[i] = np.mean(window_data)
                elif method == "median":
                    fitted[i] = np.median(window_data)
                elif method == "weighted_mean":
                    # Use only the weights corresponding to available data
                    n_available = len(window_data)
                    if n_available < window_size:
                        # Use subset of weights (most recent ones)
                        subset_weights = weights[-n_available:]
                        subset_weights = subset_weights / subset_weights.sum()
                        fitted[i] = np.average(window_data, weights=subset_weights)
                    else:
                        fitted[i] = np.average(window_data, weights=weights)
            else:
                # Not enough data - use mean of available data as fallback
                if len(window_data) > 0:
                    fitted[i] = np.mean(window_data)
                else:
                    # First observation - no prior data
                    fitted[i] = y[i]

        # Calculate residuals
        residuals = y - fitted

        # Build fit_data dict
        fit_data = {
            "model": {
                "window_size": window_size,
                "method": method,
                "weights": weights.tolist() if weights is not None else None,
                "min_periods": min_periods,
                "train_values": y,
                "n_train": n,
            },
            "fitted": fitted,
            "residuals": residuals,
            "outcomes": data[[outcome_col]],
            "outcome_col": outcome_col,
            "date_col": date_col,
            "original_training_data": data,
        }

        # Build blueprint dict (minimal for raw path)
        blueprint = {
            "formula": formula,
            "outcome_name": outcome_col,  # Use outcome_name for evaluate() compatibility
            "outcome_col": outcome_col,
            "date_col": date_col,
        }

        return fit_data, blueprint

    def predict_raw(
        self,
        fit: ModelFit,
        new_data: pd.DataFrame,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make window-based predictions.

        For h-step ahead forecast:
        - Uses last window_size observations from training data
        - Applies aggregation method (mean/median/weighted_mean)
        - Returns constant forecast (all predictions equal)

        Args:
            fit: ModelFit with fitted model
            new_data: Data for prediction (used for determining horizon)
            type: Prediction type

        Returns:
            DataFrame with predictions
        """
        model = fit.fit_data["model"]
        window_size = model["window_size"]
        method = model["method"]
        weights = model["weights"]
        min_periods = model["min_periods"]
        train_values = model["train_values"]

        # Get number of predictions (horizon)
        n_pred = len(new_data)

        # Use last window_size observations from training data
        if window_size > len(train_values):
            window_data = train_values
        else:
            window_data = train_values[-window_size:]

        # Compute aggregate
        if method == "mean":
            forecast_value = np.mean(window_data)
        elif method == "median":
            forecast_value = np.median(window_data)
        elif method == "weighted_mean":
            weights_array = np.array(weights)
            # Use only the weights corresponding to available data
            n_available = len(window_data)
            if n_available < window_size:
                subset_weights = weights_array[-n_available:]
                subset_weights = subset_weights / subset_weights.sum()
                forecast_value = np.average(window_data, weights=subset_weights)
            else:
                forecast_value = np.average(window_data, weights=weights_array)
        else:
            raise ValueError(f"Unknown method: {method}")

        # All predictions are the same (constant forecast)
        predictions = np.full(n_pred, forecast_value)

        # Create result DataFrame with date index if available
        date_col = fit.fit_data.get("date_col")
        if date_col and date_col in new_data.columns:
            result = pd.DataFrame(
                {".pred": predictions},
                index=new_data[date_col].values
            )
        else:
            result = pd.DataFrame({".pred": predictions})

        return result

    def extract_outputs(
        self, fit: ModelFit
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract three-DataFrame output structure.

        Returns:
            - outputs: Observation-level DataFrame (actuals, fitted, forecast, residuals, split)
            - coefficients: Window parameters (window_size, method, weights)
            - stats: Model-level statistics (RMSE, MAE, R², etc.)
        """
        from py_yardstick import rmse, mae, mape, r_squared

        fit_output = fit.fit_data
        fitted = fit_output["fitted"]
        residuals = fit_output["residuals"]
        outcomes = fit_output["outcomes"]
        outcome_col = fit_output["outcome_col"]
        model = fit_output["model"]

        method = model["method"]
        window_size = model["window_size"]
        weights = model["weights"]
        min_periods = model["min_periods"]

        if isinstance(outcomes, pd.DataFrame):
            actuals = outcomes[outcome_col].values
        else:
            actuals = outcomes.values

        # ====================
        # 1. OUTPUTS DataFrame
        # ====================
        outputs_list = []

        # Training data
        # Use combine_first pattern for forecast column
        train_df = pd.DataFrame({
            "actuals": actuals,
            "fitted": fitted,
            "residuals": residuals,
            "split": "train",
        })
        # Forecast = actuals where available, fitted otherwise
        train_df["forecast"] = pd.Series(actuals).combine_first(
            pd.Series(fitted)
        ).values
        outputs_list.append(train_df)

        # Test data (if evaluated)
        if "test_predictions" in fit.evaluation_data:
            test_data = fit.evaluation_data["test_data"]
            test_preds = fit.evaluation_data["test_predictions"]
            test_outcome_col = fit.evaluation_data["outcome_col"]

            test_actuals = test_data[test_outcome_col].values
            test_predictions = test_preds[".pred"].values
            test_residuals = test_actuals - test_predictions

            test_df = pd.DataFrame({
                "actuals": test_actuals,
                "fitted": test_predictions,
                "residuals": test_residuals,
                "split": "test",
            })
            # Forecast = actuals where available, fitted otherwise
            test_df["forecast"] = pd.Series(test_actuals).combine_first(
                pd.Series(test_predictions)
            ).values
            outputs_list.append(test_df)

        outputs = pd.concat(outputs_list, ignore_index=True) if outputs_list else pd.DataFrame()

        # Add date column if available
        try:
            from py_parsnip.utils import _infer_date_column

            if fit.fit_data.get("original_training_data") is not None:
                date_col = _infer_date_column(
                    fit.fit_data["original_training_data"],
                    spec_date_col=fit.fit_data.get("date_col"),
                    fit_date_col=None,
                )

                # Extract date values for training data
                if date_col == "__index__":
                    train_dates = fit.fit_data["original_training_data"].index.values
                else:
                    train_dates = fit.fit_data["original_training_data"][date_col].values

                # Handle test data if present
                if fit.evaluation_data and "original_test_data" in fit.evaluation_data:
                    test_data_orig = fit.evaluation_data["original_test_data"]
                    if date_col == "__index__":
                        test_dates = test_data_orig.index.values
                    else:
                        test_dates = test_data_orig[date_col].values

                    all_dates = np.concatenate([train_dates, test_dates])
                else:
                    all_dates = train_dates

                # Insert date column at position 0
                outputs.insert(0, "date", all_dates)

        except (ValueError, ImportError, KeyError):
            # No datetime columns or error - skip date column
            pass

        # Add model metadata columns
        outputs["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        outputs["model_group_name"] = (
            fit.model_group_name if fit.model_group_name else ""
        )
        outputs["group"] = "global"

        # ====================
        # 2. COEFFICIENTS DataFrame
        # ====================
        coef_rows = [
            {
                "variable": "window_size",
                "coefficient": window_size,
                "std_error": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
            },
            {
                "variable": "method",
                "coefficient": method,
                "std_error": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
            },
            {
                "variable": "min_periods",
                "coefficient": min_periods,
                "std_error": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
            },
        ]

        # Add weights if weighted_mean
        if weights is not None:
            for i, w in enumerate(weights):
                coef_rows.append(
                    {
                        "variable": f"weight_{i}",
                        "coefficient": w,
                        "std_error": np.nan,
                        "p_value": np.nan,
                        "ci_0.025": np.nan,
                        "ci_0.975": np.nan,
                    }
                )

        coefficients = pd.DataFrame(coef_rows)

        # Add model metadata columns
        coefficients["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        coefficients["model_group_name"] = (
            fit.model_group_name if fit.model_group_name else ""
        )
        coefficients["group"] = "global"

        # ====================
        # 3. STATS DataFrame
        # ====================
        stats_rows = []

        # Training metrics
        # Remove NaN residuals for metrics calculation
        valid_mask = ~np.isnan(residuals)
        valid_actuals = actuals[valid_mask]
        valid_fitted = fitted[valid_mask]

        if len(valid_actuals) > 0:
            # Yardstick functions return DataFrames, extract scalar values
            rmse_val = rmse(valid_actuals, valid_fitted)["value"].iloc[0]
            mae_val = mae(valid_actuals, valid_fitted)["value"].iloc[0]
            mape_val = mape(valid_actuals, valid_fitted)["value"].iloc[0]
            r2_val = r_squared(valid_actuals, valid_fitted)["value"].iloc[0]
        else:
            rmse_val = mae_val = mape_val = r2_val = np.nan

        stats_rows.extend(
            [
                {"metric": "rmse", "value": rmse_val, "split": "train"},
                {"metric": "mae", "value": mae_val, "split": "train"},
                {"metric": "mape", "value": mape_val, "split": "train"},
                {"metric": "r_squared", "value": r2_val, "split": "train"},
                {"metric": "window_size", "value": window_size, "split": "train"},
                {"metric": "method", "value": method, "split": "train"},
                {"metric": "min_periods", "value": min_periods, "split": "train"},
            ]
        )

        # Test metrics (if evaluated)
        if "test_predictions" in fit.evaluation_data:
            test_data = fit.evaluation_data["test_data"]
            test_preds = fit.evaluation_data["test_predictions"]
            test_outcome_col = fit.evaluation_data["outcome_col"]

            test_actuals = test_data[test_outcome_col].values
            test_predictions = test_preds[".pred"].values

            # Calculate test metrics
            test_rmse = rmse(test_actuals, test_predictions)["value"].iloc[0]
            test_mae = mae(test_actuals, test_predictions)["value"].iloc[0]
            test_mape = mape(test_actuals, test_predictions)["value"].iloc[0]
            test_r2 = r_squared(test_actuals, test_predictions)["value"].iloc[0]

            stats_rows.extend(
                [
                    {"metric": "rmse", "value": test_rmse, "split": "test"},
                    {"metric": "mae", "value": test_mae, "split": "test"},
                    {"metric": "mape", "value": test_mape, "split": "test"},
                    {"metric": "r_squared", "value": test_r2, "split": "test"},
                ]
            )

        # Add training date range (if available)
        train_dates = None
        try:
            from py_parsnip.utils import _infer_date_column

            if fit.fit_data.get("original_training_data") is not None:
                date_col = _infer_date_column(
                    fit.fit_data["original_training_data"],
                    spec_date_col=fit.fit_data.get("date_col"),
                    fit_date_col=None,
                )

                if date_col == "__index__":
                    train_dates = fit.fit_data["original_training_data"].index.values
                else:
                    train_dates = fit.fit_data["original_training_data"][date_col].values
        except (ValueError, ImportError, KeyError):
            pass

        if train_dates is not None and len(train_dates) > 0:
            stats_rows.extend(
                [
                    {
                        "metric": "train_start_date",
                        "value": str(train_dates[0]),
                        "split": "train",
                    },
                    {
                        "metric": "train_end_date",
                        "value": str(train_dates[-1]),
                        "split": "train",
                    },
                ]
            )

        stats = pd.DataFrame(stats_rows)

        # Add model metadata columns
        stats["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        stats["model_group_name"] = (
            fit.model_group_name if fit.model_group_name else ""
        )
        stats["group"] = "global"

        return outputs, coefficients, stats
