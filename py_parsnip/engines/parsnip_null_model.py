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
        baseline_value = fit.fit_data["model"]["baseline_value"]

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

        fit_output = fit.fit_data
        fitted = fit_output["fitted"]
        residuals = fit_output["residuals"]
        outcomes = fit_output["outcomes"]
        baseline_value = fit_output["model"]["baseline_value"]

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
            "forecast": fitted,  # Same as fitted for null model
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

        # ====================
        # 2. COEFFICIENTS DataFrame
        # ====================
        coefficients = pd.DataFrame({
            "variable": ["(Intercept)"],
            "coefficient": [baseline_value],
            "std_error": [np.nan],
            "p_value": [np.nan],
            "ci_0.025": [np.nan],
            "ci_0.975": [np.nan],
        })

        # ====================
        # 3. STATS DataFrame
        # ====================
        stats_rows = []

        # Training metrics
        if fit.spec.mode == "regression":
            # Calculate regression metrics
            # Yardstick functions return DataFrames, extract scalar values
            rmse_val = rmse(actuals, fitted)['value'].iloc[0]
            mae_val = mae(actuals, fitted)['value'].iloc[0]
            mape_val = mape(actuals, fitted)['value'].iloc[0]
            r2_val = r_squared(actuals, fitted)['value'].iloc[0]

            stats_rows.extend([
                {"metric": "rmse", "value": rmse_val, "split": "train"},
                {"metric": "mae", "value": mae_val, "split": "train"},
                {"metric": "mape", "value": mape_val, "split": "train"},
                {"metric": "r_squared", "value": r2_val, "split": "train"},
                {"metric": "baseline_value", "value": baseline_value, "split": "train"},
            ])
        else:
            # Classification - compute accuracy
            accuracy = np.mean(actuals == fitted)
            stats_rows.extend([
                {"metric": "accuracy", "value": accuracy, "split": "train"},
                {"metric": "baseline_class", "value": baseline_value, "split": "train"},
            ])

        # Test metrics (if evaluated)
        if "test_predictions" in fit.evaluation_data:
            test_data = fit.evaluation_data["test_data"]
            test_preds = fit.evaluation_data["test_predictions"]
            outcome_col = fit.evaluation_data["outcome_col"]

            test_actuals = test_data[outcome_col].values
            test_predictions = test_preds[".pred"].values

            if fit.spec.mode == "regression":
                # Calculate test regression metrics
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
            else:
                # Classification - compute test accuracy
                test_accuracy = np.mean(test_actuals == test_predictions)
                stats_rows.append({"metric": "accuracy", "value": test_accuracy, "split": "test"})

        stats = pd.DataFrame(stats_rows)

        return outputs, coefficients, stats
