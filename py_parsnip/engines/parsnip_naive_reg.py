"""
Parsnip engine for naive forecasting

Implements four naive forecasting strategies:
- naive: Last observed value (random walk)
- seasonal_naive: Last value from same season
- drift: Linear trend from first to last value
- window: Rolling window average (moving average)
"""

from typing import Dict, Any, Literal, Optional
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData


@register_engine("naive_reg", "parsnip")
class ParsnipNaiveEngine(Engine):
    """
    Parsnip engine for naive forecasting methods.

    Methods:
    - naive: y_t = y_{t-1}
    - seasonal_naive: y_t = y_{t-s}
    - drift: y_t = y_{t-1} + (y_T - y_1) / (T - 1)
    """

    def fit(
        self,
        spec: ModelSpec,
        molded: MoldedData,
        original_training_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Fit naive model (just stores training values).

        Args:
            spec: ModelSpec with model configuration
            molded: MoldedData with outcomes
            original_training_data: Optional original training data with date columns

        Returns:
            Dict containing training values and method
        """
        # Extract outcomes
        y = molded.outcomes

        # Flatten y if it's a DataFrame with single column
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y = y.iloc[:, 0]

        # Convert to numpy array
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y

        # Get strategy from args (with backward compatibility for "method")
        strategy = spec.args.get("strategy") or spec.args.get("method", "naive")
        seasonal_period = spec.args.get("seasonal_period")
        window_size = spec.args.get("window_size")

        # Compute fitted values based on strategy
        n = len(y_values)
        fitted = np.full(n, np.nan)

        if strategy == "naive":
            # Naive: y_t = y_{t-1}
            fitted[1:] = y_values[:-1]
            fitted[0] = y_values[0]  # First value = itself

        elif strategy in ["seasonal_naive", "snaive"]:
            # Seasonal naive: y_t = y_{t-s}
            if seasonal_period is None:
                raise ValueError("seasonal_period required for seasonal_naive")

            s = seasonal_period
            if s >= n:
                raise ValueError(f"seasonal_period ({s}) must be less than data length ({n})")

            # For first season, use naive
            fitted[:s] = y_values[:s]

            # For subsequent periods, use last seasonal value
            for i in range(s, n):
                fitted[i] = y_values[i - s]

        elif strategy == "drift":
            # Drift: y_t = y_{t-1} + (y_T - y_1) / (T - 1)
            drift = (y_values[-1] - y_values[0]) / (n - 1)

            fitted[0] = y_values[0]
            for i in range(1, n):
                fitted[i] = y_values[i-1] + drift

        elif strategy == "window":
            # Window: Rolling window average
            if window_size is None:
                raise ValueError("window_size required for window strategy")

            w = window_size
            if w < 1:
                raise ValueError(f"window_size must be >= 1, got {w}")
            if w > n:
                raise ValueError(f"window_size ({w}) must be <= data length ({n})")

            # For first w values, use expanding window average
            for i in range(n):
                if i == 0:
                    fitted[i] = y_values[i]  # First value = itself
                elif i < w:
                    # Expanding window (use all available past values)
                    fitted[i] = np.mean(y_values[:i])
                else:
                    # Rolling window (use last w values)
                    fitted[i] = np.mean(y_values[i-w:i])

        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'naive', 'seasonal_naive', 'drift', or 'window'.")

        # Calculate residuals
        residuals = y_values - fitted

        return {
            "model": {
                "strategy": strategy,
                "seasonal_period": seasonal_period,
                "window_size": window_size,
                "train_values": y_values,
                "n_train": n,
            },
            "fitted": fitted,
            "residuals": residuals,
            "outcomes": y,
            "original_training_data": original_training_data,
        }

    def predict(
        self,
        fit: ModelFit,
        molded: MoldedData,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make naive predictions.

        Args:
            fit: ModelFit with fitted model
            molded: MoldedData with predictors (not used for naive)
            type: Prediction type

        Returns:
            DataFrame with predictions
        """
        model = fit.fit_data["model"]
        strategy = model["strategy"]
        train_values = model["train_values"]
        seasonal_period = model["seasonal_period"]
        window_size = model["window_size"]

        # Get number of predictions (horizon)
        n_pred = len(molded.predictors)

        predictions = np.full(n_pred, np.nan)

        if strategy == "naive":
            # All predictions = last training value
            last_value = train_values[-1]
            predictions = np.full(n_pred, last_value)

        elif strategy in ["seasonal_naive", "snaive"]:
            # Use last full seasonal cycle
            s = seasonal_period
            n_train = len(train_values)

            for h in range(n_pred):
                # Which season does this forecast belong to?
                season_idx = (n_train + h) % s

                # Find the last occurrence of this season in training data
                # Go backwards from end of training data
                for i in range(n_train - 1, -1, -1):
                    if i % s == season_idx:
                        predictions[h] = train_values[i]
                        break

        elif strategy == "drift":
            # Extrapolate drift
            last_value = train_values[-1]
            drift = (train_values[-1] - train_values[0]) / (len(train_values) - 1)

            for h in range(n_pred):
                predictions[h] = last_value + drift * (h + 1)

        elif strategy == "window":
            # Window: Use last window_size values for average
            # For forecasting, predict the mean of the last window
            w = window_size
            n_train = len(train_values)

            # Calculate window average from last w training values
            if w > n_train:
                # Use all available values
                window_avg = np.mean(train_values)
            else:
                window_avg = np.mean(train_values[-w:])

            # All predictions = window average (constant forecast)
            predictions = np.full(n_pred, window_avg)

        return pd.DataFrame({".pred": predictions})

    def extract_outputs(
        self, fit: ModelFit
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract three-DataFrame output structure.

        Returns:
            - outputs: Observation-level DataFrame
            - coefficients: Empty (no coefficients for naive methods)
            - stats: Model-level statistics
        """
        from py_yardstick import rmse, mae, mape, r_squared

        fit_output = fit.fit_data
        fitted = fit_output["fitted"]
        residuals = fit_output["residuals"]
        outcomes = fit_output["outcomes"]
        strategy = fit_output["model"]["strategy"]

        if isinstance(outcomes, pd.Series):
            actuals = outcomes.values
        else:
            actuals = outcomes

        # ====================
        # 1. OUTPUTS DataFrame
        # ====================
        outputs_list = []

        # Training data
        train_df = pd.DataFrame({
            "actuals": actuals,
            "fitted": fitted,
            "forecast": fitted,
            "residuals": residuals,
            "split": "train",
        })
        outputs_list.append(train_df)

        # Test data (if evaluated)
        if "test_predictions" in fit.evaluation_data:
            test_data = fit.evaluation_data["test_data"]
            test_preds = fit.evaluation_data["test_predictions"]
            outcome_col = fit.evaluation_data["outcome_col"]

            test_actuals = test_data[outcome_col].values
            test_predictions = test_preds[".pred"].values
            test_residuals = test_actuals - test_predictions

            test_df = pd.DataFrame({
                "actuals": test_actuals,
                "fitted": test_predictions,
                "forecast": test_predictions,
                "residuals": test_residuals,
                "split": "test",
            })
            outputs_list.append(test_df)

        outputs = pd.concat(outputs_list, ignore_index=True) if outputs_list else pd.DataFrame()

        # Try to add date column if original data has datetime columns
        try:
            from py_parsnip.utils import _infer_date_column

            # Check if we have original data with datetime
            if fit.fit_data.get("original_training_data") is not None:
                date_col = _infer_date_column(
                    fit.fit_data["original_training_data"],
                    spec_date_col=None,
                    fit_date_col=None
                )

                # Extract date values for training data
                if date_col == '__index__':
                    train_dates = fit.fit_data["original_training_data"].index.values
                else:
                    train_dates = fit.fit_data["original_training_data"][date_col].values

                # Handle test data if present
                if fit.evaluation_data and 'original_test_data' in fit.evaluation_data:
                    test_data_orig = fit.evaluation_data['original_test_data']
                    if date_col == '__index__':
                        test_dates = test_data_orig.index.values
                    else:
                        test_dates = test_data_orig[date_col].values

                    # Combine train and test dates
                    all_dates = np.concatenate([train_dates, test_dates])
                else:
                    all_dates = train_dates

                # Insert date column at position 0
                outputs.insert(0, 'date', all_dates)

        except (ValueError, ImportError):
            # No datetime columns or error - skip date column (backward compat)
            pass

        # Add model metadata columns
        outputs["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        outputs["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        outputs["group"] = "global"  # Default group for non-grouped models

        # ====================
        # 2. COEFFICIENTS DataFrame
        # ====================
        coefficients = pd.DataFrame({
            "variable": ["strategy"],
            "coefficient": [strategy],
            "std_error": [np.nan],
            "p_value": [np.nan],
            "ci_0.025": [np.nan],
            "ci_0.975": [np.nan],
        })

        # Add model metadata columns
        coefficients["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        coefficients["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
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
            rmse_val = rmse(valid_actuals, valid_fitted)['value'].iloc[0]
            mae_val = mae(valid_actuals, valid_fitted)['value'].iloc[0]
            mape_val = mape(valid_actuals, valid_fitted)['value'].iloc[0]
            r2_val = r_squared(valid_actuals, valid_fitted)['value'].iloc[0]
        else:
            rmse_val = mae_val = mape_val = r2_val = np.nan

        stats_rows.extend([
            {"metric": "rmse", "value": rmse_val, "split": "train"},
            {"metric": "mae", "value": mae_val, "split": "train"},
            {"metric": "mape", "value": mape_val, "split": "train"},
            {"metric": "r_squared", "value": r2_val, "split": "train"},
            {"metric": "strategy", "value": strategy, "split": "train"},
        ])

        # Test metrics (if evaluated)
        if "test_predictions" in fit.evaluation_data:
            test_data = fit.evaluation_data["test_data"]
            test_preds = fit.evaluation_data["test_predictions"]
            outcome_col = fit.evaluation_data["outcome_col"]

            test_actuals = test_data[outcome_col].values
            test_predictions = test_preds[".pred"].values

            # Calculate test metrics
            # Yardstick functions return DataFrames, extract scalar values
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

        # Add training date range (if available from original data)
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

        # Add model metadata columns
        stats["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        stats["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        stats["group"] = "global"

        return outputs, coefficients, stats
