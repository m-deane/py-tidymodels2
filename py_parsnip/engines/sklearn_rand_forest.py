"""
Sklearn engine for random forest

Maps rand_forest to sklearn's Random Forest models:
- mode="regression" → RandomForestRegressor
- mode="classification" → RandomForestClassifier
"""

from typing import Dict, Any, Literal
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData


@register_engine("rand_forest", "sklearn")
class SklearnRandForestEngine(Engine):
    """
    Sklearn engine for random forest.

    Parameter mapping:
    - mtry → max_features
    - trees → n_estimators
    - min_n → min_samples_split
    """

    param_map = {
        "mtry": "max_features",
        "trees": "n_estimators",
        "min_n": "min_samples_split",
    }

    def fit(self, spec: ModelSpec, molded: MoldedData) -> Dict[str, Any]:
        """
        Fit random forest model using sklearn.

        Args:
            spec: ModelSpec with model configuration
            molded: MoldedData with outcomes and predictors

        Returns:
            Dict containing fitted model and metadata
        """
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        # Extract predictors and outcomes
        X = molded.predictors
        y = molded.outcomes

        # Random forests don't use intercepts - remove Intercept column if present
        if "Intercept" in X.columns:
            X = X.drop(columns=["Intercept"])

        # Determine which sklearn model to use based on mode
        if spec.mode == "regression":
            model_class = RandomForestRegressor
            # For regression, flatten y if it's a DataFrame
            if isinstance(y, pd.DataFrame):
                if y.shape[1] == 1:
                    y = y.iloc[:, 0]
        elif spec.mode == "classification":
            model_class = RandomForestClassifier
            # For classification, handle one-hot encoded outcomes
            if isinstance(y, pd.DataFrame):
                if y.shape[1] == 1:
                    # Single column - just flatten
                    y = y.iloc[:, 0]
                else:
                    # Multiple columns - convert from one-hot encoding to class labels
                    # Column names are like "species[A]", "species[B]" - extract just the class labels
                    class_labels = []
                    for col in y.columns:
                        # Extract class from "varname[class]" format
                        if "[" in col and "]" in col:
                            class_label = col.split("[")[1].split("]")[0]
                        else:
                            class_label = col
                        class_labels.append(class_label)

                    # Convert one-hot to class labels using extracted labels
                    y = pd.Series([class_labels[i] for i in y.values.argmax(axis=1)], index=y.index)
        else:
            raise ValueError(
                f"rand_forest mode must be 'regression' or 'classification', got '{spec.mode}'. "
                "Use .set_mode() to set the mode."
            )

        # Translate parameters
        model_args = self.translate_params(spec.args)

        # Set defaults if not provided
        if "n_estimators" not in model_args:
            model_args["n_estimators"] = 500
        if "min_samples_split" not in model_args:
            model_args["min_samples_split"] = 2
        # max_features default handled by sklearn

        # Add random_state for reproducibility
        model_args["random_state"] = 42

        # Create and fit model
        model = model_class(**model_args)
        model.fit(X, y)

        # Calculate fitted values (in-sample predictions)
        fitted = model.predict(X)

        # Calculate residuals (for regression only)
        if spec.mode == "regression":
            residuals = y.values if isinstance(y, pd.Series) else y - fitted
        else:
            # For classification, residuals not meaningful
            residuals = None

        # Return fitted model and metadata
        return {
            "model": model,
            "n_features": X.shape[1],
            "n_obs": X.shape[0],
            "model_class": model_class.__name__,
            "feature_names": list(X.columns),
            "X_train": X,
            "y_train": y,
            "fitted": fitted,
            "residuals": residuals,
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
            type: Prediction type
                - "numeric": Numeric predictions (regression)
                - "class": Class predictions (classification)
                - "prob": Class probabilities (classification)

        Returns:
            DataFrame with predictions
        """
        model = fit.fit_data["model"]
        X = molded.predictors

        # Random forests don't use intercepts - remove Intercept column if present
        if "Intercept" in X.columns:
            X = X.drop(columns=["Intercept"])

        if fit.spec.mode == "regression":
            if type != "numeric":
                raise ValueError(
                    f"rand_forest in regression mode only supports type='numeric', got '{type}'"
                )
            predictions = model.predict(X)
            return pd.DataFrame({".pred": predictions})

        elif fit.spec.mode == "classification":
            if type == "class":
                predictions = model.predict(X)
                return pd.DataFrame({".pred_class": predictions})
            elif type == "prob":
                probs = model.predict_proba(X)
                # Create DataFrame with column for each class
                class_names = model.classes_
                df = pd.DataFrame(
                    probs,
                    columns=[f".pred_{cls}" for cls in class_names]
                )
                return df
            else:
                raise ValueError(
                    f"rand_forest in classification mode supports type='class' or 'prob', got '{type}'"
                )
        else:
            raise ValueError(f"Unknown mode: {fit.spec.mode}")

    def _calculate_metrics(self, actuals: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics (regression only)"""
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

        # Adjusted R-squared (need n_features from somewhere)
        adj_r_squared = np.nan  # Will calculate in extract_outputs if needed

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
        """Calculate residual diagnostic statistics (regression only)"""
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

        # Placeholder for other tests
        results["ljung_box_stat"] = np.nan
        results["ljung_box_p"] = np.nan
        results["breusch_pagan_stat"] = np.nan
        results["breusch_pagan_p"] = np.nan

        return results

    def extract_outputs(
        self, fit: ModelFit
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract comprehensive three-DataFrame output for random forest.

        Returns:
            Tuple of (outputs, coefficients, stats)

            - Outputs: Observation-level (actuals, fitted, forecast, residuals, split)
            - Coefficients: Feature importances (instead of coefficients)
            - Stats: Model-level metrics by split (RMSE, MAE, etc. for regression)
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
                "fitted": fitted_values,  # In-sample predictions
                "forecast": forecast_train,  # Actuals where available, fitted otherwise
                "split": "train",
                "model": fit.model_name if fit.model_name else fit.spec.model_type,
                "model_group_name": fit.model_group_name if fit.model_group_name else "",
                "group": "global"
            })

            # Add residuals (for regression only)
            if fit.spec.mode == "regression":
                if residuals is not None:
                    train_df["residuals"] = residuals
                else:
                    train_df["residuals"] = y_train_values - fitted_values
            else:
                train_df["residuals"] = np.nan

            outputs_list.append(train_df)

        # Test data (if evaluated via fit.evaluate())
        if "test_predictions" in fit.evaluation_data:
            test_data = fit.evaluation_data["test_data"]
            test_preds = fit.evaluation_data["test_predictions"]
            outcome_col = fit.evaluation_data["outcome_col"]

            test_actuals = test_data[outcome_col].values
            test_predictions = test_preds[".pred"].values if ".pred" in test_preds.columns else test_preds[".pred_class"].values

            # Create forecast: actuals where they exist, fitted where they don't
            forecast_test = pd.Series(test_actuals).combine_first(pd.Series(test_predictions)).values

            test_df = pd.DataFrame({
                "actuals": test_actuals,
                "fitted": test_predictions,  # Out-of-sample predictions
                "forecast": forecast_test,  # Actuals where available, fitted otherwise
                "split": "test",
                "model": fit.model_name if fit.model_name else fit.spec.model_type,
                "model_group_name": fit.model_group_name if fit.model_group_name else "",
                "group": "global"
            })

            # Add residuals (for regression only)
            if fit.spec.mode == "regression":
                test_df["residuals"] = test_actuals - test_predictions
            else:
                test_df["residuals"] = np.nan

            outputs_list.append(test_df)

        outputs = pd.concat(outputs_list, ignore_index=True) if outputs_list else pd.DataFrame()

        # ====================
        # 2. COEFFICIENTS DataFrame (Feature Importances for Random Forest)
        # ====================
        feature_names = fit.fit_data.get("feature_names", [])
        feature_importances = model.feature_importances_

        # For Random Forest, we report feature importances instead of coefficients
        coefficients = pd.DataFrame({
            "variable": feature_names,
            "coefficient": feature_importances,  # Importance acts as coefficient
            "std_error": np.nan,  # Not applicable for Random Forest
            "t_stat": np.nan,  # Not applicable
            "p_value": np.nan,  # Not applicable
            "ci_0.025": np.nan,  # Not applicable
            "ci_0.975": np.nan,  # Not applicable
            "vif": np.nan,  # Not applicable for tree-based models
            "model": fit.model_name if fit.model_name else fit.spec.model_type,
            "model_group_name": fit.model_group_name if fit.model_group_name else "",
            "group": "global"
        })

        # ====================
        # 3. STATS DataFrame (model-level metrics by split)
        # ====================
        stats_rows = []

        # Training metrics (regression only)
        if fit.spec.mode == "regression" and y_train is not None and fitted is not None:
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

        # Test metrics (regression only, if evaluated)
        if fit.spec.mode == "regression" and "test_predictions" in fit.evaluation_data:
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

        # Residual diagnostics (regression only, training data)
        if fit.spec.mode == "regression" and residuals is not None and len(residuals) > 0:
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
            {"metric": "mode", "value": fit.spec.mode, "split": ""},
            {"metric": "n_trees", "value": model.n_estimators, "split": ""},
            {"metric": "n_features", "value": fit.fit_data["n_features"], "split": ""},
            {"metric": "n_obs_train", "value": fit.fit_data.get("n_obs", 0), "split": "train"},
        ])

        # OOB score (out-of-bag) if available
        if hasattr(model, 'oob_score_') and model.oob_score_ is not None:
            stats_rows.append({
                "metric": "oob_score",
                "value": model.oob_score_,
                "split": "train"
            })

        stats = pd.DataFrame(stats_rows)

        # Add model metadata
        stats["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        stats["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        stats["group"] = "global"

        return outputs, coefficients, stats
