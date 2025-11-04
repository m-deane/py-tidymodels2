"""
Sklearn engine for PLS regression

Maps pls to sklearn's PLSRegression for dimension reduction
and regression on latent components.
"""

from typing import Dict, Any, Literal
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData


@register_engine("pls", "sklearn")
class SklearnPLSEngine(Engine):
    """
    Sklearn engine for PLS regression.

    Parameter mapping:
    - num_comp → n_components (number of PLS components)
    """

    param_map = {
        "num_comp": "n_components",
    }

    def fit(self, spec: ModelSpec, molded: MoldedData) -> Dict[str, Any]:
        """
        Fit PLS regression model using sklearn.

        Args:
            spec: ModelSpec with model configuration
            molded: MoldedData with outcomes and predictors

        Returns:
            Dict containing fitted model and metadata
        """
        from sklearn.cross_decomposition import PLSRegression

        # Extract predictors and outcomes
        X = molded.predictors
        y = molded.outcomes

        # Flatten y if it's a DataFrame with single column
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y = y.iloc[:, 0]

        # Get number of components
        args = spec.args
        num_comp = args.get("num_comp", 2)  # Default to 2 components

        # Ensure num_comp doesn't exceed limits
        n_samples = X.shape[0]
        n_features = X.shape[1]
        max_comp = min(n_samples, n_features)

        if num_comp > max_comp:
            num_comp = max_comp

        # Create and fit PLS model
        model = PLSRegression(n_components=num_comp, scale=True)
        model.fit(X, y)

        # Calculate fitted values and residuals
        fitted = model.predict(X).ravel()
        y_array = y.values if isinstance(y, pd.Series) else y
        residuals = y_array - fitted

        # Check if there's a date column in outcomes
        date_col = None
        if hasattr(molded, "outcomes_original"):
            # Check if original outcomes DataFrame has date column
            orig = molded.outcomes_original
            if isinstance(orig, pd.DataFrame):
                for col in orig.columns:
                    if pd.api.types.is_datetime64_any_dtype(orig[col]):
                        date_col = col
                        break

        # Return fitted model and metadata
        return {
            "model": model,
            "n_features": X.shape[1],
            "n_obs": X.shape[0],
            "n_components": num_comp,
            "model_class": "PLSRegression",
            "X_train": X,
            "y_train": y,
            "fitted": fitted,
            "residuals": residuals,
            "date_col": date_col,
        }

    def predict(
        self,
        fit: ModelFit,
        molded: MoldedData,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make predictions using fitted PLS model.

        Args:
            fit: ModelFit with fitted model
            molded: MoldedData with predictors
            type: Prediction type ("numeric" for regression)

        Returns:
            DataFrame with predictions
        """
        if type != "numeric":
            raise ValueError(f"pls only supports type='numeric', got '{type}'")

        model = fit.fit_data["model"]
        X = molded.predictors

        # Make predictions
        predictions = model.predict(X).ravel()

        # Return as DataFrame with standard column name
        return pd.DataFrame({".pred": predictions})

    def _calculate_metrics(self, actuals: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics"""
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

        # MDA (Mean Directional Accuracy) - percentage of correct direction predictions
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

        # Ljung-Box test would require statsmodels
        results["ljung_box_stat"] = np.nan
        results["ljung_box_p"] = np.nan

        # Breusch-Pagan test would require statsmodels
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

            - Outputs: Observation-level results with actuals, fitted, residuals
            - Coefficients: PLS coefficients for original predictors
            - Stats: Comprehensive metrics by split + component info
        """
        model = fit.fit_data["model"]

        # ====================
        # 1. OUTPUTS DataFrame
        # ====================
        outputs_list = []

        # Training data
        y_train = fit.fit_data.get("y_train")
        fitted = fit.fit_data.get("fitted")
        residuals = fit.fit_data.get("residuals")

        if y_train is not None and fitted is not None:
            y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train
            # Create forecast: actuals where they exist, fitted where they don't
            forecast_train = pd.Series(y_train_array).combine_first(pd.Series(fitted)).values

            train_df = pd.DataFrame({
                "actuals": y_train_array,
                "fitted": fitted,  # In-sample predictions
                "forecast": forecast_train,  # Actuals where available, fitted otherwise
                "residuals": residuals if residuals is not None else y_train_array - fitted,
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

            # Create forecast: actuals where they exist, fitted where they don't
            forecast_test = pd.Series(test_actuals).combine_first(pd.Series(test_predictions)).values

            test_df = pd.DataFrame({
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
        # 2. COEFFICIENTS DataFrame
        # ====================
        # PLS coefficients: regression coefficients for original predictors
        coef_names = list(fit.blueprint.column_order)

        # PLS coefficient is the product of x_weights and y_loadings
        # model.coef_ gives the final regression coefficients
        coef_values = model.coef_.ravel() if hasattr(model, "coef_") else []

        # PLS doesn't provide standard errors like OLS, so we fill with NaN
        n_coef = len(coef_names)
        std_errors = [np.nan] * n_coef
        t_stats = [np.nan] * n_coef
        p_values = [np.nan] * n_coef
        ci_lower = [np.nan] * n_coef
        ci_upper = [np.nan] * n_coef
        vifs = [np.nan] * n_coef

        coefficients = pd.DataFrame({
            "variable": coef_names,
            "coefficient": coef_values,
            "std_error": std_errors,
            "t_stat": t_stats,
            "p_value": p_values,
            "ci_0.025": ci_lower,
            "ci_0.975": ci_upper,
            "vif": vifs,
        })

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
            train_metrics = self._calculate_metrics(
                y_train.values if isinstance(y_train, pd.Series) else y_train,
                fitted
            )

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

        # PLS-specific information
        n_comp = fit.fit_data.get("n_components", 0)
        stats_rows.extend([
            {"metric": "formula", "value": fit.blueprint.formula if hasattr(fit.blueprint, "formula") else "", "split": ""},
            {"metric": "model_type", "value": fit.spec.model_type, "split": ""},
            {"metric": "model_class", "value": fit.fit_data.get("model_class", ""), "split": ""},
            {"metric": "n_components", "value": n_comp, "split": ""},
            {"metric": "n_obs_train", "value": fit.fit_data.get("n_obs", 0), "split": "train"},
            {"metric": "n_features", "value": fit.fit_data.get("n_features", 0), "split": ""},
        ])

        stats = pd.DataFrame(stats_rows)

        # Add model metadata
        stats["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        stats["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        stats["group"] = "global"

        return outputs, coefficients, stats
