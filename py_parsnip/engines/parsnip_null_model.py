"""
Parsnip engine for null model

A baseline model that predicts a constant value:
- Regression: mean or median of training outcomes
- Classification: mode (most frequent class)
"""

from typing import Dict, Any, Literal
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData


@register_engine("null_model", "parsnip")
class ParsnipNullEngine(Engine):
    """
    Parsnip engine for null model (baseline predictor).

    Regression: Predicts mean of training outcomes
    Classification: Predicts mode of training outcomes
    """

    def fit(self, spec: ModelSpec, molded: MoldedData) -> Dict[str, Any]:
        """
        Fit null model by computing baseline statistic.

        Args:
            spec: ModelSpec with model configuration
            molded: MoldedData with outcomes

        Returns:
            Dict containing baseline value
        """
        # Extract outcomes
        y = molded.outcomes

        # Flatten y if it's a DataFrame with single column
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y = y.iloc[:, 0]

        # Compute baseline value based on mode
        if spec.mode == "regression":
            # For regression, use mean
            baseline_value = float(np.mean(y))
            method = "mean"
        elif spec.mode == "classification":
            # For classification, use mode (most frequent)
            if isinstance(y, pd.Series):
                baseline_value = y.mode()[0]
            else:
                from scipy import stats
                baseline_value = stats.mode(y, keepdims=True)[0][0]
            method = "mode"
        else:
            raise ValueError(f"Unsupported mode: {spec.mode}")

        # Create "fitted" values (all equal to baseline)
        n = len(y)
        fitted = np.full(n, baseline_value)

        # Calculate residuals
        if isinstance(y, pd.Series):
            residuals = (y.values - fitted)
        else:
            residuals = (y - fitted)

        return {
            "model": {"baseline_value": baseline_value, "method": method},
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
        Make predictions (always the baseline value).

        Args:
            fit: ModelFit with fitted model
            molded: MoldedData with predictors
            type: Prediction type

        Returns:
            DataFrame with predictions
        """
        baseline_value = fit.fit_output["model"]["baseline_value"]

        # Get number of predictions
        n = len(molded.predictors)

        # Create predictions (all equal to baseline)
        predictions = np.full(n, baseline_value)

        return pd.DataFrame({".pred": predictions})

    def extract_outputs(
        self, fit: ModelFit
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract three-DataFrame output structure.

        Returns:
            - outputs: Observation-level DataFrame
            - coefficients: Empty (no coefficients for null model)
            - stats: Model-level statistics
        """
        from py_yardstick import rmse, mae, mape, r_squared

        fit_output = fit.fit_output
        fitted = fit_output["fitted"]
        residuals = fit_output["residuals"]
        outcomes = fit_output["outcomes"]
        baseline_value = fit_output["model"]["baseline_value"]

        if isinstance(outcomes, pd.Series):
            actuals = outcomes.values
        else:
            actuals = outcomes

        # Create outputs DataFrame
        outputs = pd.DataFrame({
            "actuals": actuals,
            "fitted": fitted,
            "forecast": fitted,  # Same as fitted for null model
            "residuals": residuals,
            "split": "train",
        })

        # Coefficients DataFrame (empty - no coefficients)
        coefficients = pd.DataFrame({
            "variable": ["(Intercept)"],
            "coefficient": [baseline_value],
            "std_error": [np.nan],
            "p_value": [np.nan],
            "ci_0.025": [np.nan],
            "ci_0.975": [np.nan],
        })

        # Stats DataFrame
        if fit.spec.mode == "regression":
            # Calculate regression metrics
            rmse_val = rmse(actuals, fitted)
            mae_val = mae(actuals, fitted)
            mape_val = mape(actuals, fitted)
            r2_val = r_squared(actuals, fitted)

            stats = pd.DataFrame({
                "metric": ["rmse", "mae", "mape", "r_squared", "baseline_value"],
                "value": [rmse_val, mae_val, mape_val, r2_val, baseline_value],
                "split": ["train"] * 5,
            })
        else:
            # Classification - compute accuracy
            accuracy = np.mean(actuals == fitted)
            stats = pd.DataFrame({
                "metric": ["accuracy", "baseline_class"],
                "value": [accuracy, baseline_value],
                "split": ["train", "train"],
            })

        return outputs, coefficients, stats
