"""
Parsnip engine for naive forecasting

Implements three naive forecasting methods:
- naive: Last observed value (random walk)
- seasonal_naive: Last value from same season
- drift: Linear trend from first to last value
"""

from typing import Dict, Any, Literal
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

    def fit(self, spec: ModelSpec, molded: MoldedData) -> Dict[str, Any]:
        """
        Fit naive model (just stores training values).

        Args:
            spec: ModelSpec with model configuration
            molded: MoldedData with outcomes

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

        # Get method and seasonal_period from args
        method = spec.args.get("method", "naive")
        seasonal_period = spec.args.get("seasonal_period")

        # Compute fitted values based on method
        n = len(y_values)
        fitted = np.full(n, np.nan)

        if method == "naive":
            # Naive: y_t = y_{t-1}
            fitted[1:] = y_values[:-1]
            fitted[0] = y_values[0]  # First value = itself

        elif method in ["seasonal_naive", "snaive"]:
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

        elif method == "drift":
            # Drift: y_t = y_{t-1} + (y_T - y_1) / (T - 1)
            drift = (y_values[-1] - y_values[0]) / (n - 1)

            fitted[0] = y_values[0]
            for i in range(1, n):
                fitted[i] = y_values[i-1] + drift

        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate residuals
        residuals = y_values - fitted

        return {
            "model": {
                "method": method,
                "seasonal_period": seasonal_period,
                "train_values": y_values,
                "n_train": n,
            },
            "fitted": fitted,
            "residuals": residuals,
            "outcomes": y,
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
        model = fit.fit_output["model"]
        method = model["method"]
        train_values = model["train_values"]
        seasonal_period = model["seasonal_period"]

        # Get number of predictions (horizon)
        n_pred = len(molded.predictors)

        predictions = np.full(n_pred, np.nan)

        if method == "naive":
            # All predictions = last training value
            last_value = train_values[-1]
            predictions = np.full(n_pred, last_value)

        elif method in ["seasonal_naive", "snaive"]:
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

        elif method == "drift":
            # Extrapolate drift
            last_value = train_values[-1]
            drift = (train_values[-1] - train_values[0]) / (len(train_values) - 1)

            for h in range(n_pred):
                predictions[h] = last_value + drift * (h + 1)

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

        fit_output = fit.fit_output
        fitted = fit_output["fitted"]
        residuals = fit_output["residuals"]
        outcomes = fit_output["outcomes"]
        method = fit_output["model"]["method"]

        if isinstance(outcomes, pd.Series):
            actuals = outcomes.values
        else:
            actuals = outcomes

        # Create outputs DataFrame
        outputs = pd.DataFrame({
            "actuals": actuals,
            "fitted": fitted,
            "forecast": fitted,
            "residuals": residuals,
            "split": "train",
        })

        # Coefficients DataFrame (empty - no coefficients)
        coefficients = pd.DataFrame({
            "variable": ["method"],
            "coefficient": [method],
            "std_error": [np.nan],
            "p_value": [np.nan],
            "ci_0.025": [np.nan],
            "ci_0.975": [np.nan],
        })

        # Stats DataFrame
        # Remove NaN residuals for metrics calculation
        valid_mask = ~np.isnan(residuals)
        valid_actuals = actuals[valid_mask]
        valid_fitted = fitted[valid_mask]

        if len(valid_actuals) > 0:
            rmse_val = rmse(valid_actuals, valid_fitted)
            mae_val = mae(valid_actuals, valid_fitted)
            mape_val = mape(valid_actuals, valid_fitted)
            r2_val = r_squared(valid_actuals, valid_fitted)
        else:
            rmse_val = mae_val = mape_val = r2_val = np.nan

        stats = pd.DataFrame({
            "metric": ["rmse", "mae", "mape", "r_squared", "method"],
            "value": [rmse_val, mae_val, mape_val, r2_val, method],
            "split": ["train"] * 5,
        })

        return outputs, coefficients, stats
