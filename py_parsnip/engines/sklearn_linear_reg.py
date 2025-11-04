"""
Sklearn engine for linear regression

Maps linear_reg to sklearn's linear models:
- No penalty → LinearRegression
- penalty + mixture=0.0 → Ridge
- penalty + mixture=1.0 → Lasso
- penalty + mixture=(0,1) → ElasticNet
"""

from typing import Dict, Any, Literal, Optional
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData


@register_engine("linear_reg", "sklearn")
class SklearnLinearEngine(Engine):
    """
    Sklearn engine for linear regression.

    Parameter mapping:
    - penalty → alpha (regularization strength)
    - mixture → l1_ratio (ElasticNet mixing parameter)
    """

    param_map = {
        "penalty": "alpha",
        "mixture": "l1_ratio",
    }

    def fit(
        self,
        spec: ModelSpec,
        molded: MoldedData,
        original_training_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Fit linear regression model using sklearn.

        Args:
            spec: ModelSpec with model configuration
            molded: MoldedData with outcomes and predictors
            original_training_data: Original training DataFrame (optional, for date column extraction)

        Returns:
            Dict containing fitted model and metadata
        """
        from sklearn.linear_model import (
            LinearRegression,
            Ridge,
            Lasso,
            ElasticNet,
        )

        # Extract predictors and outcomes
        X = molded.predictors
        y = molded.outcomes

        # Flatten y if it's a DataFrame with single column
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y = y.iloc[:, 0]

        # Determine which sklearn model to use based on penalty and mixture
        args = spec.args
        penalty = args.get("penalty")
        mixture = args.get("mixture")

        if penalty is None or penalty == 0:
            # No regularization - use LinearRegression
            model_class = LinearRegression
            model_args = {}
        elif mixture is None or mixture == 0.0:
            # Pure L2 penalty - use Ridge
            model_class = Ridge
            model_args = {"alpha": penalty}
        elif mixture == 1.0:
            # Pure L1 penalty - use Lasso
            model_class = Lasso
            model_args = {"alpha": penalty}
        else:
            # Mix of L1 and L2 - use ElasticNet
            model_class = ElasticNet
            model_args = {"alpha": penalty, "l1_ratio": mixture}

        # Create and fit model
        model = model_class(**model_args)
        model.fit(X, y)

        # Calculate fitted values and residuals
        fitted = model.predict(X)
        residuals = y.values if isinstance(y, pd.Series) else y - fitted

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
            "model_class": model_class.__name__,
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
        Make predictions using fitted sklearn model.

        Args:
            fit: ModelFit with fitted model
            molded: MoldedData with predictors
            type: Prediction type ("numeric" for regression)

        Returns:
            DataFrame with predictions
        """
        if type != "numeric":
            raise ValueError(f"linear_reg only supports type='numeric', got '{type}'")

        model = fit.fit_data["model"]
        X = molded.predictors

        # Make predictions
        predictions = model.predict(X)

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
            - Coefficients: Enhanced coefficients with p-values, CI, VIF
            - Stats: Comprehensive metrics by split + residual diagnostics
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

        # Add date column if available in original data
        if not outputs.empty:
            try:
                from py_parsnip.utils.time_series_utils import _infer_date_column

                # For training data
                original_training_data = fit.fit_data.get("original_training_data")
                if original_training_data is not None:
                    try:
                        date_col = _infer_date_column(
                            original_training_data,
                            spec_date_col=None,
                            fit_date_col=None
                        )

                        # Extract training dates
                        if date_col == '__index__':
                            train_dates = original_training_data.index.values
                        else:
                            train_dates = original_training_data[date_col].values

                        # For test data (if evaluated)
                        test_dates = None
                        original_test_data = fit.evaluation_data.get("original_test_data")
                        if original_test_data is not None:
                            try:
                                # Use same date_col from training
                                if date_col == '__index__':
                                    test_dates = original_test_data.index.values
                                else:
                                    test_dates = original_test_data[date_col].values
                            except (KeyError, AttributeError):
                                pass  # Skip test dates if not available

                        # Combine dates based on split
                        combined_dates = []
                        train_count = (outputs['split'] == 'train').sum()
                        test_count = (outputs['split'] == 'test').sum()

                        # Add training dates
                        if train_count > 0:
                            combined_dates.extend(train_dates[:train_count])

                        # Add test dates if they exist
                        if test_count > 0 and test_dates is not None:
                            combined_dates.extend(test_dates[:test_count])

                        # Add date column as first column (before model/group columns)
                        if len(combined_dates) == len(outputs):
                            outputs.insert(0, 'date', combined_dates)

                    except ValueError:
                        # No datetime columns or invalid date column - skip date column
                        pass
            except ImportError:
                # time_series_utils not available - skip date column
                pass

        # ====================
        # 2. COEFFICIENTS DataFrame
        # ====================
        coef_names = list(fit.blueprint.column_order)
        coef_values = model.coef_ if hasattr(model, "coef_") else []

        # Calculate standard errors, t-stats, p-values, CI using OLS formula
        X_train = fit.fit_data.get("X_train")
        n_obs = fit.fit_data.get("n_obs", 0)
        n_features = fit.fit_data.get("n_features", 0)

        std_errors = [np.nan] * len(coef_names)
        t_stats = [np.nan] * len(coef_names)
        p_values = [np.nan] * len(coef_names)
        ci_lower = [np.nan] * len(coef_names)
        ci_upper = [np.nan] * len(coef_names)
        vifs = [np.nan] * len(coef_names)

        # Only calculate if we have OLS model (no regularization)
        if (fit.fit_data.get("model_class") == "LinearRegression" and
            X_train is not None and residuals is not None and n_obs > n_features):

            # Calculate MSE
            mse = np.sum(residuals ** 2) / (n_obs - n_features)

            # Calculate standard errors
            try:
                XtX_inv = np.linalg.inv(X_train.T @ X_train)
                var_coef = mse * np.diag(XtX_inv)
                std_errors = np.sqrt(var_coef)

                # Calculate t-statistics
                t_stats = coef_values / std_errors

                # Calculate p-values (two-tailed)
                df = n_obs - n_features
                p_values = 2 * (1 - scipy_stats.t.cdf(np.abs(t_stats), df))

                # Calculate 95% confidence intervals
                t_crit = scipy_stats.t.ppf(0.975, df)
                ci_lower = coef_values - t_crit * std_errors
                ci_upper = coef_values + t_crit * std_errors

                # Calculate VIF for each predictor (skip intercept if present)
                for i, col_name in enumerate(coef_names):
                    if col_name != "Intercept" and X_train.shape[1] > 1:
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
                pass  # Keep NaN values if calculation fails

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

        # Model information
        stats_rows.extend([
            {"metric": "formula", "value": fit.blueprint.formula if hasattr(fit.blueprint, "formula") else "", "split": ""},
            {"metric": "model_type", "value": fit.spec.model_type, "split": ""},
            {"metric": "model_class", "value": fit.fit_data.get("model_class", ""), "split": ""},
            {"metric": "n_obs_train", "value": n_obs, "split": "train"},
        ])

        stats = pd.DataFrame(stats_rows)

        # Add model metadata
        stats["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        stats["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        stats["group"] = "global"

        return outputs, coefficients, stats
