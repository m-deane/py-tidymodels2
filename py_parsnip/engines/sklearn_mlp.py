"""
Sklearn engine for Multi-Layer Perceptron (Neural Network)

Maps mlp to sklearn's MLPRegressor
"""

from typing import Dict, Any, Literal, Optional
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData


@register_engine("mlp", "sklearn")
class SklearnMLPEngine(Engine):
    """
    Sklearn engine for Multi-Layer Perceptron.

    Parameter mapping:
    - hidden_units → hidden_layer_sizes
    - penalty → alpha
    - epochs → max_iter
    - learn_rate → learning_rate_init
    - activation → activation
    """

    param_map = {
        "hidden_units": "hidden_layer_sizes",
        "penalty": "alpha",
        "epochs": "max_iter",
        "learn_rate": "learning_rate_init",
        "activation": "activation",
    }

    def fit(self, spec: ModelSpec, molded: MoldedData, original_training_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Fit Multi-Layer Perceptron using sklearn.

        Args:
            spec: ModelSpec with model configuration
            molded: MoldedData with outcomes and predictors
            original_training_data: Optional original training data for date extraction

        Returns:
            Dict containing fitted model and metadata
        """
        from sklearn.neural_network import MLPRegressor

        # Extract predictors and outcomes
        X = molded.predictors
        y = molded.outcomes

        # MLPs don't use intercepts - remove Intercept column if present
        if "Intercept" in X.columns:
            X = X.drop(columns=["Intercept"])

        # Flatten y if it's a DataFrame
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y = y.iloc[:, 0]

        # Translate parameters
        model_args = self.translate_params(spec.args)

        # Set defaults if not provided
        if "hidden_layer_sizes" not in model_args:
            model_args["hidden_layer_sizes"] = (100,)
        if "alpha" not in model_args:
            model_args["alpha"] = 0.0001
        if "max_iter" not in model_args:
            model_args["max_iter"] = 200
        if "learning_rate_init" not in model_args:
            model_args["learning_rate_init"] = 0.001
        if "activation" not in model_args:
            model_args["activation"] = "relu"

        # Add random_state for reproducibility
        model_args["random_state"] = 42

        # Create and fit model
        model = MLPRegressor(**model_args)
        model.fit(X, y)

        # Calculate fitted values (in-sample predictions)
        fitted = model.predict(X)

        # Calculate residuals
        residuals = y.values if isinstance(y, pd.Series) else y - fitted

        # Return fitted model and metadata
        return {
            "model": model,
            "n_features": X.shape[1],
            "n_obs": X.shape[0],
            "model_class": "MLPRegressor",
            "feature_names": list(X.columns),
            "X_train": X,
            "y_train": y,
            "fitted": fitted,
            "residuals": residuals,
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
            raise ValueError(
                f"mlp only supports type='numeric', got '{type}'"
            )

        model = fit.fit_data["model"]
        X = molded.predictors

        # MLPs don't use intercepts - remove Intercept column if present
        if "Intercept" in X.columns:
            X = X.drop(columns=["Intercept"])

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

    def _calculate_residual_diagnostics(self, residuals: np.ndarray) -> Dict[str, float]:
        """Calculate residual diagnostic statistics"""
        from scipy import stats as scipy_stats
        import statsmodels.stats.diagnostic as sm_diag

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

        # Ljung-Box test for autocorrelation
        try:
            # Ensure we have enough lags (at least 1, max 10 or n//5)
            n_lags = max(1, min(10, n // 5))
            lb_result = sm_diag.acorr_ljungbox(residuals, lags=n_lags)
            # Returns DataFrame with columns 'lb_stat' and 'lb_pvalue'
            results["ljung_box_stat"] = lb_result['lb_stat'].iloc[-1]  # Last lag statistic
            results["ljung_box_p"] = lb_result['lb_pvalue'].iloc[-1]  # Last lag p-value
        except Exception as e:
            results["ljung_box_stat"] = np.nan
            results["ljung_box_p"] = np.nan

        # Breusch-Pagan test - requires exogenous variables, set to NaN for now
        results["breusch_pagan_stat"] = np.nan
        results["breusch_pagan_p"] = np.nan

        return results

    def extract_outputs(
        self, fit: ModelFit
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract comprehensive three-DataFrame output for MLP.

        Returns:
            Tuple of (outputs, coefficients, stats)

            - Outputs: Observation-level (actuals, fitted, forecast, residuals, split)
            - Coefficients: Layer weights summary (not individual weights due to complexity)
            - Stats: Model-level metrics by split (RMSE, MAE, R-squared, etc.)
        """
        model = fit.fit_data["model"]

        # ====================
        # 1. OUTPUTS DataFrame (observation-level)
        # ====================
        outputs_list = []

        # Training data
        y_train = fit.fit_data.get("y_train")
        fitted = fit.fit_data.get("fitted")
        residuals = fit.fit_data.get("residuals")

        if y_train is not None and fitted is not None:
            # Convert to numpy arrays
            if isinstance(y_train, pd.Series):
                y_train_values = y_train.values
            else:
                y_train_values = y_train

            if isinstance(fitted, pd.Series):
                fitted_values = fitted.values
            else:
                fitted_values = fitted

            # Create forecast: actuals where they exist, fitted where they don't
            forecast_train = pd.Series(y_train_values).combine_first(pd.Series(fitted_values)).values

            train_df = pd.DataFrame({
                "actuals": y_train_values,
                "fitted": fitted_values,
                "forecast": forecast_train,
                "split": "train",
                "model": fit.model_name if fit.model_name else fit.spec.model_type,
                "model_group_name": fit.model_group_name if fit.model_group_name else "",
                "group": "global"
            })

            # Add residuals
            if residuals is not None:
                train_df["residuals"] = residuals
            else:
                train_df["residuals"] = y_train_values - fitted_values

            outputs_list.append(train_df)

        # Test data (if evaluated via fit.evaluate())
        if "test_predictions" in fit.evaluation_data:
            test_data = fit.evaluation_data["test_data"]
            test_preds = fit.evaluation_data["test_predictions"]
            outcome_col = fit.evaluation_data["outcome_col"]

            test_actuals = test_data[outcome_col].values
            test_predictions = test_preds[".pred"].values

            # Create forecast: actuals where they exist, fitted where they don't
            forecast_test = pd.Series(test_actuals).combine_first(pd.Series(test_predictions)).values

            test_df = pd.DataFrame({
                "actuals": test_actuals,
                "fitted": test_predictions,
                "forecast": forecast_test,
                "split": "test",
                "model": fit.model_name if fit.model_name else fit.spec.model_type,
                "model_group_name": fit.model_group_name if fit.model_group_name else "",
                "group": "global"
            })

            # Add residuals
            test_df["residuals"] = test_actuals - test_predictions

            outputs_list.append(test_df)

        outputs = pd.concat(outputs_list, ignore_index=True) if outputs_list else pd.DataFrame()

        # ====================
        # 2. COEFFICIENTS DataFrame
        # ====================
        # For MLP, individual weights are too numerous to report meaningfully
        # Instead, report layer-wise weight statistics
        coef_rows = []

        if hasattr(model, 'coefs_'):
            for i, coef_matrix in enumerate(model.coefs_):
                layer_name = f"layer_{i}_to_{i+1}"
                coef_rows.append({
                    "variable": layer_name,
                    "coefficient": np.mean(coef_matrix),  # Mean weight
                    "std_error": np.std(coef_matrix),     # Std of weights
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "ci_0.025": np.percentile(coef_matrix.flatten(), 2.5),
                    "ci_0.975": np.percentile(coef_matrix.flatten(), 97.5),
                    "vif": np.nan,
                })

        coefficients = pd.DataFrame(coef_rows) if coef_rows else pd.DataFrame({
            "variable": [],
            "coefficient": [],
            "std_error": [],
            "t_stat": [],
            "p_value": [],
            "ci_0.025": [],
            "ci_0.975": [],
            "vif": [],
        })

        # Add model metadata
        coefficients["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        coefficients["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        coefficients["group"] = "global"

        # ====================
        # 3. STATS DataFrame (model-level metrics by split)
        # ====================
        stats_rows = []

        # Training metrics
        if y_train is not None and fitted is not None:
            train_metrics = self._calculate_metrics(y_train_values, fitted_values)

            # Calculate adjusted R-squared
            n = len(y_train_values)
            k = fit.fit_data["n_features"]
            if train_metrics["r_squared"] is not np.nan and n > k + 1:
                adj_r_squared = 1 - ((1 - train_metrics["r_squared"]) * (n - 1) / (n - k - 1))
                train_metrics["adj_r_squared"] = adj_r_squared
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

            # Adjusted R-squared for test
            n = len(test_actuals)
            k = fit.fit_data["n_features"]
            if test_metrics["r_squared"] is not np.nan and n > k + 1:
                adj_r_squared = 1 - ((1 - test_metrics["r_squared"]) * (n - 1) / (n - k - 1))
                test_metrics["adj_r_squared"] = adj_r_squared
            else:
                test_metrics["adj_r_squared"] = np.nan

            for metric_name, value in test_metrics.items():
                stats_rows.append({
                    "metric": metric_name,
                    "value": value,
                    "split": "test",
                })

        # Residual diagnostics (training data)
        if residuals is not None and len(residuals) > 0:
            resid_diag = self._calculate_residual_diagnostics(residuals)
            for metric_name, value in resid_diag.items():
                stats_rows.append({
                    "metric": metric_name,
                    "value": value,
                    "split": "train",
                })

        # Model information
        blueprint = fit.blueprint
        stats_rows.extend([
            {"metric": "formula", "value": blueprint.formula if hasattr(blueprint, 'formula') else "", "split": ""},
            {"metric": "model_type", "value": fit.spec.model_type, "split": ""},
            {"metric": "hidden_layer_sizes", "value": str(model.hidden_layer_sizes), "split": ""},
            {"metric": "activation", "value": model.activation, "split": ""},
            {"metric": "alpha", "value": model.alpha, "split": ""},
            {"metric": "learning_rate_init", "value": model.learning_rate_init, "split": ""},
            {"metric": "n_iter", "value": model.n_iter_, "split": ""},
            {"metric": "n_layers", "value": model.n_layers_, "split": ""},
            {"metric": "loss", "value": model.loss_, "split": "train"},
            {"metric": "n_features", "value": fit.fit_data["n_features"], "split": ""},
            {"metric": "n_obs_train", "value": fit.fit_data.get("n_obs", 0), "split": "train"},
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
