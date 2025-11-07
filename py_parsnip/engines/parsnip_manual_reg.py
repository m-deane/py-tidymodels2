"""
Manual regression engine - user-specified coefficients

This engine allows users to manually specify coefficients instead of fitting them.
Useful for comparing with external models or incorporating domain knowledge.
"""

from typing import Dict, Any, Literal, Optional
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData


@register_engine("manual_reg", "parsnip")
class ManualRegEngine(Engine):
    """
    Engine for manual regression with user-specified coefficients.

    Instead of fitting coefficients from data, this engine:
    1. Validates user-provided coefficients match formula variables
    2. Applies coefficients to training data to get fitted values
    3. Calculates residuals and statistics
    4. Uses same coefficients for prediction

    This enables:
    - Comparison with external/pre-existing models
    - Benchmarking against known coefficient values
    - Domain expert knowledge incorporation
    """

    def fit(
        self,
        spec: ModelSpec,
        molded: MoldedData,
        original_training_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        'Fit' manual regression model (validate and store coefficients).

        Args:
            spec: ModelSpec with manual coefficients
            molded: MoldedData with predictors and outcomes
            original_training_data: Original training DataFrame

        Returns:
            Dict containing coefficients and fitted values
        """
        # Extract user-specified coefficients and intercept
        user_coefficients = spec.args.get("coefficients", {})
        user_intercept = spec.args.get("intercept", 0.0)

        # Extract predictors and outcomes
        X = molded.predictors
        y = molded.outcomes

        # Flatten y if DataFrame with single column
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.iloc[:, 0]

        y_values = y.values if isinstance(y, pd.Series) else y

        # Get predictor column names
        predictor_names = list(X.columns)

        # Separate intercept column from other predictors
        # Patsy automatically adds "Intercept" column
        has_intercept = "Intercept" in predictor_names

        if has_intercept:
            # Remove "Intercept" from predictor names (handled separately)
            predictor_names_no_intercept = [col for col in predictor_names if col != "Intercept"]
        else:
            predictor_names_no_intercept = predictor_names

        # Validate that all user-specified coefficients match predictor names
        # (excluding intercept which is specified separately)
        missing_vars = set(user_coefficients.keys()) - set(predictor_names_no_intercept)
        if missing_vars:
            raise ValueError(
                f"Coefficients specified for variables not in formula: {missing_vars}. "
                f"Available predictors: {predictor_names_no_intercept}"
            )

        # Create coefficient vector for non-intercept predictors
        # Use user value if provided, otherwise default to 0.0
        coefficients = np.array([
            user_coefficients.get(var, 0.0) for var in predictor_names_no_intercept
        ])

        # Calculate fitted values
        if has_intercept:
            # X has intercept column (all 1s), so extract non-intercept columns
            X_no_intercept = X[[col for col in X.columns if col != "Intercept"]].values
            fitted = user_intercept + X_no_intercept @ coefficients
        else:
            # No intercept column in X
            X_values = X.values
            fitted = user_intercept + X_values @ coefficients

        # Calculate residuals
        residuals = y_values - fitted

        # Calculate basic statistics
        n = len(y_values)
        k = len(predictor_names_no_intercept)  # Number of predictors (excluding intercept)

        # Return fit data
        return {
            "coefficients": coefficients,
            "coefficient_names": predictor_names_no_intercept,
            "intercept": user_intercept,
            "user_coefficients": user_coefficients,
            "fitted": fitted,
            "residuals": residuals,
            "y_train": y_values,
            "n_obs": n,
            "n_features": k,
            "original_training_data": original_training_data,
            "has_intercept": has_intercept,
        }

    def predict(
        self,
        fit: ModelFit,
        molded: MoldedData,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make predictions using manual coefficients.

        Args:
            fit: ModelFit with stored coefficients
            molded: MoldedData with predictors
            type: Prediction type ("numeric" for regression)

        Returns:
            DataFrame with predictions
        """
        if type != "numeric":
            raise ValueError(f"manual_reg only supports type='numeric', got '{type}'")

        # Extract coefficients and intercept
        coefficients = fit.fit_data["coefficients"]
        intercept = fit.fit_data["intercept"]
        has_intercept = fit.fit_data.get("has_intercept", True)

        # Get predictors
        X = molded.predictors

        # Calculate predictions
        if has_intercept and "Intercept" in X.columns:
            # Remove intercept column before matrix multiplication
            X_no_intercept = X[[col for col in X.columns if col != "Intercept"]].values
            predictions = intercept + X_no_intercept @ coefficients
        else:
            # No intercept column
            X_values = X.values
            predictions = intercept + X_values @ coefficients

        return pd.DataFrame({".pred": predictions})

    def extract_outputs(
        self, fit: ModelFit
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract comprehensive three-DataFrame output.

        Returns:
            Tuple of (outputs, coefficients, stats)
        """
        from py_yardstick import rmse, mae, mape, r_squared

        # ====================
        # 1. OUTPUTS DataFrame
        # ====================
        outputs_list = []

        y_train = fit.fit_data.get("y_train")
        fitted = fit.fit_data.get("fitted")
        residuals = fit.fit_data.get("residuals")

        if y_train is not None and fitted is not None:
            forecast_train = pd.Series(y_train).combine_first(pd.Series(fitted)).values

            train_df = pd.DataFrame({
                "actuals": y_train,
                "fitted": fitted,
                "forecast": forecast_train,
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

            forecast_test = pd.Series(test_actuals).combine_first(pd.Series(test_predictions)).values

            test_df = pd.DataFrame({
                "actuals": test_actuals,
                "fitted": test_predictions,
                "forecast": forecast_test,
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
        coefficients_list = []

        # Add intercept
        coefficients_list.append({
            "variable": "Intercept",
            "coefficient": float(fit.fit_data.get("intercept", 0.0)),
            "std_error": np.nan,  # Not applicable for manual coefficients
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_0.025": np.nan,
            "ci_0.975": np.nan,
            "vif": np.nan,
        })

        # Add predictor coefficients
        coefficient_names = fit.fit_data.get("coefficient_names", [])
        coefficient_values = fit.fit_data.get("coefficients", [])

        for var, coef in zip(coefficient_names, coefficient_values):
            coefficients_list.append({
                "variable": var,
                "coefficient": float(coef),
                "std_error": np.nan,  # Not applicable for manual coefficients
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
            })

        coefficients = pd.DataFrame(coefficients_list)

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
            rmse_val = rmse(y_train, fitted)['value'].iloc[0]
            mae_val = mae(y_train, fitted)['value'].iloc[0]
            mape_val = mape(y_train, fitted)['value'].iloc[0]
            r2_val = r_squared(y_train, fitted)['value'].iloc[0]

            stats_rows.extend([
                {"metric": "rmse", "value": rmse_val, "split": "train"},
                {"metric": "mae", "value": mae_val, "split": "train"},
                {"metric": "mape", "value": mape_val, "split": "train"},
                {"metric": "r_squared", "value": r2_val, "split": "train"},
            ])

        # Test metrics (if evaluated)
        if "test_predictions" in fit.evaluation_data:
            test_data = fit.evaluation_data["test_data"]
            test_preds = fit.evaluation_data["test_predictions"]
            outcome_col = fit.evaluation_data["outcome_col"]

            test_actuals = test_data[outcome_col].values
            test_predictions = test_preds[".pred"].values

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

        # Model information
        n_obs = fit.fit_data.get("n_obs", 0)
        stats_rows.extend([
            {"metric": "formula", "value": fit.blueprint.formula if hasattr(fit.blueprint, "formula") else "", "split": ""},
            {"metric": "model_type", "value": "manual_reg", "split": ""},
            {"metric": "mode", "value": "manual", "split": ""},
            {"metric": "n_obs_train", "value": n_obs, "split": "train"},
        ])

        # Add training date range if available
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

        # Add model metadata
        stats["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        stats["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        stats["group"] = "global"

        return outputs, coefficients, stats
