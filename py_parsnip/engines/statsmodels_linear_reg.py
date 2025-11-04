"""
Statsmodels engine for linear regression (OLS)

Maps linear_reg to statsmodels OLS for classical linear regression
with full statistical inference (p-values, confidence intervals, diagnostics).
"""

from typing import Dict, Any, Literal, Optional
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData


@register_engine("linear_reg", "statsmodels")
class StatsmodelsLinearEngine(Engine):
    """
    Statsmodels engine for OLS linear regression.

    Provides full statistical inference including:
    - Coefficient p-values
    - Confidence intervals
    - Residual diagnostics
    - Model fit statistics (AIC, BIC, F-statistic)

    Note: Does not support regularization (penalty parameter).
    Use sklearn engine for Ridge/Lasso/ElasticNet.
    """

    param_map = {}  # Statsmodels OLS doesn't need parameter translation

    def fit(
        self,
        spec: ModelSpec,
        molded: MoldedData,
        original_training_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Fit OLS linear regression model using statsmodels.

        Args:
            spec: ModelSpec with model configuration
            molded: MoldedData with outcomes and predictors
            original_training_data: Optional original training data with date columns

        Returns:
            Dict containing fitted model and metadata
        """
        import statsmodels.api as sm

        # Check for unsupported parameters
        args = spec.args
        if "penalty" in args and args["penalty"] not in (None, 0):
            raise ValueError(
                "statsmodels engine does not support regularization (penalty parameter). "
                "Use sklearn engine for Ridge/Lasso/ElasticNet."
            )

        # Extract predictors and outcomes
        X = molded.predictors
        y = molded.outcomes

        # Flatten y if it's a DataFrame with single column
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y = y.iloc[:, 0]

        # Fit OLS model
        model = sm.OLS(y, X)
        fitted_model = model.fit()

        # Calculate fitted values and residuals
        fitted = fitted_model.fittedvalues.values
        residuals = fitted_model.resid.values

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
            "model": fitted_model,
            "n_features": X.shape[1],
            "n_obs": X.shape[0],
            "X_train": X,
            "y_train": y,
            "fitted": fitted,
            "residuals": residuals,
            "date_col": date_col,
            "original_training_data": original_training_data,
        }

    def predict(
        self,
        fit: ModelFit,
        molded: MoldedData,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make predictions using fitted statsmodels OLS model.

        Args:
            fit: ModelFit with fitted model
            molded: MoldedData with predictors
            type: Prediction type
                - "numeric": Point predictions
                - "conf_int": Predictions with confidence intervals

        Returns:
            DataFrame with predictions
        """
        if type not in ("numeric", "conf_int"):
            raise ValueError(
                f"linear_reg with statsmodels supports type='numeric' or 'conf_int', got '{type}'"
            )

        model = fit.fit_data["model"]
        X = molded.predictors

        # Make predictions
        predictions = model.predict(X)

        if type == "numeric":
            return pd.DataFrame({".pred": predictions.values})
        else:  # conf_int
            # Get prediction intervals
            pred_summary = model.get_prediction(X)
            pred_int = pred_summary.conf_int(alpha=0.05)  # 95% CI (returns numpy array)

            return pd.DataFrame({
                ".pred": predictions.values,
                ".pred_lower": pred_int[:, 0],
                ".pred_upper": pred_int[:, 1],
            })

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

    def _calculate_residual_diagnostics(self, residuals: np.ndarray, model) -> Dict[str, float]:
        """Calculate residual diagnostic statistics"""
        from scipy import stats as scipy_stats
        import statsmodels.stats.diagnostic as sm_diag

        results = {}
        n = len(residuals)

        # Durbin-Watson statistic (available from statsmodels)
        if hasattr(model, "durbin_watson"):
            results["durbin_watson"] = model.durbin_watson()
        elif n > 1:
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

        # Ljung-Box test for autocorrelation (using statsmodels)
        try:
            lb_result = sm_diag.acorr_ljungbox(residuals, lags=min(10, n // 5), return_df=False)
            results["ljung_box_stat"] = lb_result[0][-1]  # Last lag statistic
            results["ljung_box_p"] = lb_result[1][-1]  # Last lag p-value
        except:
            results["ljung_box_stat"] = np.nan
            results["ljung_box_p"] = np.nan

        # Breusch-Pagan test for heteroskedasticity
        try:
            bp_result = sm_diag.het_breuschpagan(residuals, model.model.exog)
            results["breusch_pagan_stat"] = bp_result[0]
            results["breusch_pagan_p"] = bp_result[1]
        except:
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
            - Coefficients: Full statistical inference from statsmodels
            - Stats: Comprehensive metrics by split + residual diagnostics + model info
        """
        from scipy import stats as scipy_stats

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

        # ====================
        # 2. COEFFICIENTS DataFrame (with full statsmodels inference)
        # ====================
        coef_names = list(fit.blueprint.column_order)
        coef_values = model.params.values

        # Get standard errors, t-stats, p-values from statsmodels
        std_errors = model.bse.values
        t_stats = model.tvalues.values
        p_values = model.pvalues.values

        # Get confidence intervals
        conf_int = model.conf_int(alpha=0.05)  # 95% CI
        ci_lower = conf_int.iloc[:, 0].values
        ci_upper = conf_int.iloc[:, 1].values

        # Calculate VIF for each predictor (skip intercept if present)
        X_train = fit.fit_data.get("X_train")
        vifs = [np.nan] * len(coef_names)

        if X_train is not None and X_train.shape[1] > 1:
            for i, col_name in enumerate(coef_names):
                if col_name != "Intercept":
                    try:
                        # VIF calculation
                        X_i = X_train.iloc[:, i:i+1]
                        X_not_i = X_train.drop(X_train.columns[i], axis=1)

                        if X_not_i.shape[1] > 0:
                            from sklearn.linear_model import LinearRegression as LR
                            vif_model = LR()
                            vif_model.fit(X_not_i, X_i)
                            r_squared_i = vif_model.score(X_not_i, X_i)
                            if r_squared_i < 0.9999:  # Avoid division by near-zero
                                vifs[i] = 1 / (1 - r_squared_i)
                    except:
                        pass  # Keep NaN if calculation fails

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
        n_obs = fit.fit_data.get("n_obs", 0)
        n_features = fit.fit_data.get("n_features", 0)

        # Training metrics
        if y_train is not None and fitted is not None:
            train_metrics = self._calculate_metrics(
                y_train.values if isinstance(y_train, pd.Series) else y_train,
                fitted
            )

            # Add adjusted R-squared
            r_sq = train_metrics["r_squared"]
            if not np.isnan(r_sq) and n_obs > n_features:
                adj_r_sq = 1 - (1 - r_sq) * (n_obs - 1) / (n_obs - n_features - 1)
                train_metrics["adj_r_squared"] = adj_r_sq
            else:
                train_metrics["adj_r_squared"] = np.nan

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

            # Add adjusted R-squared for test
            r_sq_test = test_metrics["r_squared"]
            n_test = len(test_actuals)
            if not np.isnan(r_sq_test) and n_test > n_features:
                adj_r_sq_test = 1 - (1 - r_sq_test) * (n_test - 1) / (n_test - n_features - 1)
                test_metrics["adj_r_squared"] = adj_r_sq_test
            else:
                test_metrics["adj_r_squared"] = np.nan

            for metric_name, value in test_metrics.items():
                stats_rows.append({
                    "metric": metric_name,
                    "value": value,
                    "split": "test",
                })

        # Residual diagnostics (on training data)
        if residuals is not None and len(residuals) > 0:
            resid_diag = self._calculate_residual_diagnostics(residuals, model)
            for metric_name, value in resid_diag.items():
                stats_rows.append({
                    "metric": metric_name,
                    "value": value,
                    "split": "train",
                })

        # Model-level statistics from statsmodels
        stats_rows.extend([
            {"metric": "aic", "value": model.aic, "split": ""},
            {"metric": "bic", "value": model.bic, "split": ""},
            {"metric": "log_likelihood", "value": model.llf, "split": ""},
            {"metric": "f_statistic", "value": model.fvalue, "split": ""},
            {"metric": "f_pvalue", "value": model.f_pvalue, "split": ""},
            {"metric": "condition_number", "value": model.condition_number, "split": ""},
            {"metric": "n_obs_train", "value": n_obs, "split": "train"},
            {"metric": "n_features", "value": n_features, "split": ""},
        ])

        stats = pd.DataFrame(stats_rows)

        # Add model metadata
        stats["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        stats["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        stats["group"] = "global"

        return outputs, coefficients, stats
