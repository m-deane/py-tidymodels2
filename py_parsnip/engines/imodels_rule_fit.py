"""
imodels engine for RuleFit

Maps rule_fit to imodels' RuleFitRegressor/RuleFitClassifier
"""

from typing import Dict, Any, Literal, Optional
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData


@register_engine("rule_fit", "imodels")
class ImodelsRuleFitEngine(Engine):
    """
    imodels engine for RuleFit.

    Parameter mapping:
    - max_rules → max_rules
    - tree_depth → tree_size
    - penalty → alpha
    - tree_generator → tree_generator
    """

    param_map = {
        "max_rules": "max_rules",
        "tree_depth": "tree_size",
        "penalty": "alpha",
        "tree_generator": "tree_generator",
    }

    def fit(self, spec: ModelSpec, molded: MoldedData, original_training_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Fit RuleFit model using imodels.

        Args:
            spec: ModelSpec with model configuration
            molded: MoldedData with outcomes and predictors
            original_training_data: Original training data (optional)

        Returns:
            Dict containing fitted model and metadata
        """
        from imodels import RuleFitRegressor, RuleFitClassifier

        # Extract predictors and outcomes
        X = molded.predictors
        y = molded.outcomes

        # Remove Intercept column if present
        if "Intercept" in X.columns:
            X = X.drop(columns=["Intercept"])

        # Flatten y if it's a DataFrame
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y = y.iloc[:, 0]

        # Translate parameters
        model_args = self.translate_params(spec.args)

        # Set defaults if not provided
        if "max_rules" not in model_args:
            model_args["max_rules"] = 10
        if "tree_size" not in model_args:
            model_args["tree_size"] = 3
        if "alpha" not in model_args:
            # For classification, alpha=0.0 causes division by zero in imodels
            # Use small value instead for no regularization
            if spec.mode == "classification":
                model_args["alpha"] = 1e-10
            else:
                model_args["alpha"] = 0.0
        else:
            # If alpha is explicitly set to 0.0 for classification, use small value
            if spec.mode == "classification" and model_args["alpha"] == 0.0:
                model_args["alpha"] = 1e-10

        # Add random_state for reproducibility
        model_args["random_state"] = 42

        # Create model based on mode
        if spec.mode == "regression":
            model = RuleFitRegressor(**model_args)
        elif spec.mode == "classification":
            model = RuleFitClassifier(**model_args)
        else:
            raise ValueError(f"Unsupported mode '{spec.mode}' for rule_fit. Use 'regression' or 'classification'.")

        # Fit model
        model.fit(X.values, y.values if isinstance(y, pd.Series) else y)

        # Calculate fitted values (in-sample predictions)
        if spec.mode == "regression":
            fitted = model.predict(X.values)
        else:  # classification
            fitted = model.predict(X.values)

        # Calculate residuals (for regression)
        if spec.mode == "regression":
            y_values = y.values if isinstance(y, pd.Series) else y
            residuals = y_values - fitted
        else:
            residuals = None  # Not applicable for classification

        # Return fitted model and metadata
        return {
            "model": model,
            "n_features": X.shape[1],
            "n_obs": X.shape[0],
            "model_class": "RuleFitRegressor" if spec.mode == "regression" else "RuleFitClassifier",
            "feature_names": list(X.columns),
            "X_train": X,
            "y_train": y,
            "fitted": fitted,
            "residuals": residuals,
            "original_training_data": original_training_data,
            "mode": spec.mode,
        }

    def predict(
        self,
        fit: ModelFit,
        molded: MoldedData,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make predictions using fitted imodels model.

        Args:
            fit: ModelFit with fitted model
            molded: MoldedData with predictors
            type: Prediction type ("numeric" for regression, "class" or "prob" for classification)

        Returns:
            DataFrame with predictions
        """
        model = fit.fit_data["model"]
        mode = fit.fit_data["mode"]
        X = molded.predictors

        # Remove Intercept column if present
        if "Intercept" in X.columns:
            X = X.drop(columns=["Intercept"])

        # Make predictions based on type
        if type == "numeric":
            if mode != "regression":
                raise ValueError(f"type='numeric' only valid for regression models, got mode='{mode}'")
            predictions = model.predict(X.values)
            return pd.DataFrame({".pred": predictions})

        elif type == "class":
            if mode != "classification":
                raise ValueError(f"type='class' only valid for classification models, got mode='{mode}'")
            predictions = model.predict(X.values)
            return pd.DataFrame({".pred_class": predictions})

        elif type == "prob":
            if mode != "classification":
                raise ValueError(f"type='prob' only valid for classification models, got mode='{mode}'")
            probs = model.predict_proba(X.values)
            # Create DataFrame with probability columns for each class
            classes = model.classes_
            # Convert classes to int if they're floats (e.g., 0.0 -> 0)
            classes_str = [str(int(cls)) if isinstance(cls, (float, np.floating)) and cls == int(cls) else str(cls) for cls in classes]
            prob_cols = {f".pred_{cls}": probs[:, i] for i, cls in enumerate(classes_str)}
            return pd.DataFrame(prob_cols)

        elif type == "conf_int":
            raise ValueError("rule_fit does not support confidence intervals")

        else:
            raise ValueError(f"Invalid prediction type: '{type}'")

    def _calculate_metrics(self, actuals: np.ndarray, predictions: np.ndarray, mode: str) -> Dict[str, float]:
        """Calculate performance metrics based on mode"""
        if mode == "regression":
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
        else:  # classification
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            accuracy = accuracy_score(actuals, predictions)
            # Use weighted average for multi-class
            precision = precision_score(actuals, predictions, average='weighted', zero_division=0)
            recall = recall_score(actuals, predictions, average='weighted', zero_division=0)
            f1 = f1_score(actuals, predictions, average='weighted', zero_division=0)

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }

    def _extract_rules(self, model) -> pd.DataFrame:
        """
        Extract rules from fitted RuleFit model.

        Returns:
            DataFrame with rules, their coefficients, and importance
        """
        try:
            # RuleFit stores rules in model.rules_
            if not hasattr(model, 'rules_') or model.rules_ is None:
                return pd.DataFrame()

            rules = []
            coefficients = []
            importances = []

            # Get feature importances for rules
            feature_importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else None

            # Extract rules and their coefficients
            for i, rule in enumerate(model.rules_):
                rule_str = str(rule) if hasattr(rule, '__str__') else f"Rule {i}"

                # Get coefficient for this rule from the linear model
                # RuleFit uses a linear model on top of rules
                if hasattr(model, 'coef_'):
                    coef = model.coef_[i] if i < len(model.coef_) else 0.0
                else:
                    coef = 0.0

                # Get importance
                importance = feature_importances[i] if feature_importances is not None and i < len(feature_importances) else 0.0

                rules.append(rule_str)
                coefficients.append(coef)
                importances.append(importance)

            return pd.DataFrame({
                "rule": rules,
                "coefficient": coefficients,
                "importance": importances,
            })

        except Exception as e:
            # If rule extraction fails, return empty DataFrame
            return pd.DataFrame()

    def extract_outputs(
        self, fit: ModelFit
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract comprehensive three-DataFrame output for RuleFit.

        Returns:
            Tuple of (outputs, coefficients, stats)

            - Outputs: Observation-level (actuals, fitted, forecast, residuals, split)
            - Coefficients: Extracted rules with coefficients and importance
            - Stats: Model-level metrics by split (RMSE, MAE, R-squared, rule count, etc.)
        """
        model = fit.fit_data["model"]
        mode = fit.fit_data["mode"]

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

            # Add residuals (regression only)
            if mode == "regression" and residuals is not None:
                train_df["residuals"] = residuals
            elif mode == "regression":
                train_df["residuals"] = y_train_values - fitted_values

            outputs_list.append(train_df)

        # Test data (if evaluated via fit.evaluate())
        if "test_predictions" in fit.evaluation_data:
            test_data = fit.evaluation_data["test_data"]
            test_preds = fit.evaluation_data["test_predictions"]
            outcome_col = fit.evaluation_data["outcome_col"]

            test_actuals = test_data[outcome_col].values

            # Get predictions based on mode
            if mode == "regression":
                test_predictions = test_preds[".pred"].values
            else:  # classification
                test_predictions = test_preds[".pred_class"].values

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

            # Add residuals (regression only)
            if mode == "regression":
                test_df["residuals"] = test_actuals - test_predictions

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
        # 2. COEFFICIENTS DataFrame (Rules and their coefficients)
        # ====================
        # Extract rules from the model
        rules_df = self._extract_rules(model)

        if not rules_df.empty:
            coefficients = rules_df.copy()
            # Add standard columns
            coefficients["std_error"] = np.nan
            coefficients["t_stat"] = np.nan
            coefficients["p_value"] = np.nan
            coefficients["ci_0.025"] = np.nan
            coefficients["ci_0.975"] = np.nan
            coefficients["vif"] = np.nan
            coefficients["model"] = fit.model_name if fit.model_name else fit.spec.model_type
            coefficients["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
            coefficients["group"] = "global"

            # Rename 'rule' to 'variable' for consistency
            coefficients = coefficients.rename(columns={"rule": "variable"})

            # Reorder columns to match standard format
            cols = ["variable", "coefficient", "importance", "std_error", "t_stat", "p_value",
                    "ci_0.025", "ci_0.975", "vif", "model", "model_group_name", "group"]
            coefficients = coefficients[cols]
        else:
            # Fallback: use feature importances
            feature_names = fit.fit_data.get("feature_names", [])
            feature_importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.zeros(len(feature_names))

            coefficients = pd.DataFrame({
                "variable": feature_names,
                "coefficient": feature_importances,
                "importance": feature_importances,
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
                "model": fit.model_name if fit.model_name else fit.spec.model_type,
                "model_group_name": fit.model_group_name if fit.model_group_name else "",
                "group": "global"
            })

        # ====================
        # 3. STATS DataFrame (model-level metrics by split)
        # ====================
        stats_rows = []

        # Training metrics
        if y_train is not None and fitted is not None:
            train_metrics = self._calculate_metrics(y_train_values, fitted_values, mode)

            # Calculate adjusted R-squared (regression only)
            if mode == "regression":
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

            if mode == "regression":
                test_forecast = test_preds[".pred"].values
            else:
                test_forecast = test_preds[".pred_class"].values

            test_metrics = self._calculate_metrics(test_actuals, test_forecast, mode)

            # Adjusted R-squared for test (regression only)
            if mode == "regression":
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

        # Model information
        blueprint = fit.blueprint
        n_rules = len(model.rules_) if hasattr(model, 'rules_') and model.rules_ is not None else 0

        stats_rows.extend([
            {"metric": "formula", "value": blueprint.formula if hasattr(blueprint, 'formula') else "", "split": ""},
            {"metric": "model_type", "value": fit.spec.model_type, "split": ""},
            {"metric": "mode", "value": mode, "split": ""},
            {"metric": "n_rules", "value": n_rules, "split": ""},
            {"metric": "max_rules", "value": model.max_rules if hasattr(model, 'max_rules') else np.nan, "split": ""},
            {"metric": "tree_size", "value": model.tree_size if hasattr(model, 'tree_size') else np.nan, "split": ""},
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
