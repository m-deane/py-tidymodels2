"""
Statsmodels engine for Poisson regression

Maps poisson_reg to statsmodels' GLM with Poisson family:
- Uses GLM(family=sm.families.Poisson())
- Log link function (default for Poisson family)
- Maximum likelihood estimation
"""

from typing import Dict, Any, Literal, Optional
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData


@register_engine("poisson_reg", "statsmodels")
class StatsmodelsPoissonEngine(Engine):
    """
    Statsmodels engine for Poisson regression.

    Uses GLM with Poisson family and log link.
    Note: Regularization (penalty/mixture) is not directly supported
    in statsmodels GLM and will raise an error if specified.
    """

    def fit(self, spec: ModelSpec, molded: MoldedData, original_training_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Fit Poisson regression model using statsmodels GLM.

        Args:
            spec: ModelSpec with model configuration
            molded: MoldedData with outcomes and predictors
            original_training_data: Original training DataFrame (optional, for date column extraction)

        Returns:
            Dict containing fitted model and metadata
        """
        import statsmodels.api as sm

        # Check for unsupported regularization
        if spec.args.get("penalty") is not None:
            raise ValueError(
                "statsmodels Poisson GLM does not support regularization (penalty). "
                "Remove penalty parameter or use a different engine."
            )

        # Extract predictors and outcomes
        X = molded.predictors
        y = molded.outcomes

        # Flatten y if it's a DataFrame with single column
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y = y.iloc[:, 0]

        # Ensure y is integer counts
        if not np.all(y >= 0):
            raise ValueError("Poisson regression requires non-negative count outcomes")

        # Create and fit Poisson GLM
        poisson_family = sm.families.Poisson()
        model = sm.GLM(y, X, family=poisson_family)
        results = model.fit()

        # Calculate fitted values and residuals
        fitted = results.fittedvalues
        residuals = results.resid_response  # Raw residuals (y - mu)

        # Check if there's a date column in outcomes
        date_col = None
        if hasattr(molded, "outcomes_original"):
            orig = molded.outcomes_original
            if isinstance(orig, pd.DataFrame):
                for col in orig.columns:
                    if pd.api.types.is_datetime64_any_dtype(orig[col]):
                        date_col = col
                        break

        # Return fitted model and metadata
        return {
            "model": model,
            "results": results,
            "n_features": X.shape[1],
            "n_obs": X.shape[0],
            "X_train": X,
            "y_train": y,
            "fitted": fitted,
            "residuals": residuals,
            "date_col": date_col,
            "original_training_data": original_training_data,
            # GLM-specific statistics
            "aic": results.aic,
            "bic": results.bic,
            "deviance": results.deviance,
            "pearson_chi2": results.pearson_chi2,
            "log_likelihood": results.llf,
        }

    def predict(
        self,
        fit: ModelFit,
        molded: MoldedData,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make predictions using fitted Poisson model.

        Args:
            fit: ModelFit with fitted model
            molded: MoldedData with predictors
            type: Prediction type
                - "numeric": Expected counts (default)
                - "conf_int": Predictions with confidence intervals

        Returns:
            DataFrame with predictions
        """
        results = fit.fit_data["results"]
        X = molded.predictors

        if type == "numeric":
            # Predict expected counts
            predictions = results.predict(X)
            return pd.DataFrame({".pred": predictions})

        elif type == "conf_int":
            # Predict with confidence intervals
            predictions = results.get_prediction(X)
            pred_summary = predictions.summary_frame(alpha=0.05)

            return pd.DataFrame({
                ".pred": pred_summary["mean"],
                ".pred_lower": pred_summary["mean_ci_lower"],
                ".pred_upper": pred_summary["mean_ci_upper"],
            })

        else:
            raise ValueError(
                f"poisson_reg supports type='numeric' or 'conf_int', got '{type}'"
            )

    def _calculate_metrics(self, actuals: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics for count data"""
        residuals = actuals - predictions
        n = len(actuals)

        # RMSE
        rmse = np.sqrt(np.mean(residuals ** 2))

        # MAE
        mae = np.mean(np.abs(residuals))

        # MAPE (avoid division by zero)
        mask = actuals != 0
        mape = np.mean(np.abs(residuals[mask] / actuals[mask]) * 100) if mask.any() else np.nan

        # Mean Poisson deviance
        mask_pos = (actuals > 0) & (predictions > 0)
        if mask_pos.any():
            poisson_dev = 2 * np.mean(
                actuals[mask_pos] * np.log(actuals[mask_pos] / predictions[mask_pos]) -
                (actuals[mask_pos] - predictions[mask_pos])
            )
        else:
            poisson_dev = np.nan

        # Pseudo R-squared (McFadden)
        # Note: Requires null deviance which we calculate here
        null_mean = np.mean(actuals)
        null_predictions = np.full_like(actuals, null_mean, dtype=float)

        # Avoid log(0) warnings by only computing for positive actuals
        with np.errstate(divide='ignore', invalid='ignore'):
            null_dev = 2 * np.sum(
                np.where(actuals > 0, actuals * np.log(actuals / null_predictions), 0) -
                (actuals - null_predictions)
            )
            full_dev = 2 * np.sum(
                np.where(actuals > 0, actuals * np.log(actuals / predictions), 0) -
                (actuals - predictions)
            )
        pseudo_r_squared = 1 - (full_dev / null_dev) if null_dev != 0 else np.nan

        return {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "poisson_deviance": poisson_dev,
            "pseudo_r_squared": pseudo_r_squared,
        }

    def _calculate_residual_diagnostics(self, results) -> Dict[str, float]:
        """Calculate residual diagnostic statistics"""
        diagnostics = {}

        # Pearson residuals
        if hasattr(results, "resid_pearson"):
            pearson_resid = results.resid_pearson
            diagnostics["pearson_chi2"] = np.sum(pearson_resid ** 2)

        # Deviance residuals
        if hasattr(results, "resid_deviance"):
            dev_resid = results.resid_deviance
            diagnostics["deviance"] = np.sum(dev_resid ** 2)

        return diagnostics

    def extract_outputs(
        self, fit: ModelFit
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract comprehensive three-DataFrame output.

        Returns:
            Tuple of (outputs, coefficients, stats)

            - Outputs: Observation-level results with actuals, fitted, residuals
            - Coefficients: Enhanced coefficients with p-values, CI, z-stats
            - Stats: Comprehensive metrics by split + GLM diagnostics
        """
        results = fit.fit_data["results"]

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
            forecast_train = pd.Series(y_train_array).combine_first(pd.Series(fitted)).values

            train_df = pd.DataFrame({
                "actuals": y_train_array,
                "fitted": fitted,
                "forecast": forecast_train,
                "residuals": residuals if residuals is not None else y_train_array - fitted,
                "split": "train",
            })

            # Add Poisson-specific residuals
            if hasattr(results, "resid_pearson"):
                train_df["pearson_resid"] = results.resid_pearson
            if hasattr(results, "resid_deviance"):
                train_df["deviance_resid"] = results.resid_deviance

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
        coef_names = list(fit.blueprint.column_order)

        # Get coefficients and inference from statsmodels results
        params = results.params
        bse = results.bse  # Standard errors
        tvalues = results.tvalues  # z-statistics for GLM
        pvalues = results.pvalues
        conf_int = results.conf_int(alpha=0.05)

        coefficients = pd.DataFrame({
            "variable": coef_names,
            "coefficient": params.values,
            "std_error": bse.values,
            "z_stat": tvalues.values,  # z-statistic for GLM
            "p_value": pvalues.values,
            "ci_0.025": conf_int.iloc[:, 0].values,
            "ci_0.975": conf_int.iloc[:, 1].values,
        })

        # Add model metadata
        coefficients["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        coefficients["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        coefficients["group"] = "global"

        # ====================
        # 3. STATS DataFrame
        # ====================
        stats_rows = []

        n_obs = fit.fit_data.get("n_obs", 0)
        n_features = fit.fit_data.get("n_features", 0)

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

        # GLM-specific statistics (training only)
        glm_stats = {
            "aic": fit.fit_data.get("aic", np.nan),
            "bic": fit.fit_data.get("bic", np.nan),
            "deviance": fit.fit_data.get("deviance", np.nan),
            "pearson_chi2": fit.fit_data.get("pearson_chi2", np.nan),
            "log_likelihood": fit.fit_data.get("log_likelihood", np.nan),
        }

        for metric_name, value in glm_stats.items():
            stats_rows.append({
                "metric": metric_name,
                "value": value,
                "split": "train",
            })

        # Model information
        stats_rows.extend([
            {"metric": "formula", "value": fit.blueprint.formula if hasattr(fit.blueprint, "formula") else "", "split": ""},
            {"metric": "model_type", "value": fit.spec.model_type, "split": ""},
            {"metric": "family", "value": "Poisson", "split": ""},
            {"metric": "link", "value": "log", "split": ""},
            {"metric": "n_obs_train", "value": n_obs, "split": "train"},
            {"metric": "n_features", "value": n_features, "split": ""},
        ])

        # Add training date range
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
