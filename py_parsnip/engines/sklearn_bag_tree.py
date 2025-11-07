"""
Sklearn engine for bagged trees

Maps bag_tree to sklearn's BaggingRegressor/BaggingClassifier
with DecisionTreeRegressor/DecisionTreeClassifier as base estimators.
"""

from typing import Dict, Any, Literal, Optional
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData


@register_engine("bag_tree", "sklearn")
class SklearnBagTreeEngine(Engine):
    """
    Sklearn engine for bagged trees.

    Parameter mapping:
    - trees → n_estimators (number of bootstrap samples)
    - min_n → min_samples_split (in base estimator)
    - cost_complexity → ccp_alpha (in base estimator)
    - tree_depth → max_depth (in base estimator)
    """

    param_map = {
        "trees": "n_estimators",
        "min_n": "min_samples_split",
        "cost_complexity": "ccp_alpha",
        "tree_depth": "max_depth",
    }

    def fit(self, spec: ModelSpec, molded: MoldedData, original_training_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Fit bagged tree model using sklearn.

        Args:
            spec: ModelSpec with model configuration
            molded: MoldedData with outcomes and predictors

        Returns:
            Dict containing fitted model and metadata
        """
        from sklearn.ensemble import BaggingRegressor, BaggingClassifier
        from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

        # Extract predictors and outcomes
        X = molded.predictors
        y = molded.outcomes

        # Bagged trees don't use intercepts - remove Intercept column if present
        if "Intercept" in X.columns:
            X = X.drop(columns=["Intercept"])

        # Translate parameters
        model_args = self.translate_params(spec.args)

        # Extract bagging-level parameters
        n_estimators = model_args.pop("n_estimators", 25)  # Default 25 trees

        # Remaining parameters go to base estimator
        base_estimator_args = model_args.copy()

        # Set base estimator defaults if not provided
        if "min_samples_split" not in base_estimator_args:
            base_estimator_args["min_samples_split"] = 2
        if "ccp_alpha" not in base_estimator_args:
            base_estimator_args["ccp_alpha"] = 0.0
        # max_depth defaults to None (unlimited) which is sklearn's default

        # Convert integer parameters to int
        if "n_estimators" in locals():
            n_estimators = max(1, int(n_estimators))
        if "min_samples_split" in base_estimator_args:
            base_estimator_args["min_samples_split"] = max(2, int(base_estimator_args["min_samples_split"]))
        if "max_depth" in base_estimator_args and base_estimator_args["max_depth"] is not None:
            base_estimator_args["max_depth"] = max(1, int(base_estimator_args["max_depth"]))

        # Determine which sklearn model to use based on mode
        if spec.mode == "regression":
            base_estimator = DecisionTreeRegressor(**base_estimator_args, random_state=42)
            model_class = BaggingRegressor
            # For regression, flatten y if it's a DataFrame
            if isinstance(y, pd.DataFrame):
                if y.shape[1] == 1:
                    y = y.iloc[:, 0]
        elif spec.mode == "classification":
            base_estimator = DecisionTreeClassifier(**base_estimator_args, random_state=42)
            model_class = BaggingClassifier
            # For classification, handle one-hot encoded outcomes
            if isinstance(y, pd.DataFrame):
                if y.shape[1] == 1:
                    # Single column - just flatten
                    y = y.iloc[:, 0]
                else:
                    # Multiple columns - convert from one-hot encoding to class labels
                    class_labels = []
                    for col in y.columns:
                        # Extract class from "varname[class]" format
                        if "[" in col and "]" in col:
                            class_label = col.split("[")[1].split("]")[0]
                        else:
                            class_label = col
                        class_labels.append(class_label)

                    # Convert one-hot to class labels
                    y = pd.Series([class_labels[i] for i in y.values.argmax(axis=1)], index=y.index)
        else:
            raise ValueError(
                f"bag_tree mode must be 'regression' or 'classification', got '{spec.mode}'. "
                "Use .set_mode() to set the mode."
            )

        # Create bagging model
        model = model_class(
            estimator=base_estimator,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )

        # Fit model
        model.fit(X, y)

        # Calculate fitted values (in-sample predictions)
        fitted = model.predict(X)

        # Calculate residuals (for regression only)
        if spec.mode == "regression":
            y_array = y.values if isinstance(y, pd.Series) else y
            residuals = y_array - fitted
        else:
            # For classification, residuals not meaningful
            residuals = None

        # Return fitted model and metadata
        return {
            "model": model,
            "n_features": X.shape[1],
            "n_obs": X.shape[0],
            "n_estimators": n_estimators,
            "model_class": model_class.__name__,
            "base_estimator": base_estimator.__class__.__name__,
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
        Make predictions using fitted bagged tree model.

        Args:
            fit: ModelFit with fitted model
            molded: MoldedData with predictors
            type: Prediction type

        Returns:
            DataFrame with predictions
        """
        model = fit.fit_data["model"]
        X = molded.predictors

        # Remove Intercept if present
        if "Intercept" in X.columns:
            X = X.drop(columns=["Intercept"])

        if fit.spec.mode == "regression":
            if type != "numeric":
                raise ValueError(f"For regression, type must be 'numeric', got '{type}'")

            predictions = model.predict(X)
            return pd.DataFrame({".pred": predictions})

        elif fit.spec.mode == "classification":
            if type == "class":
                predictions = model.predict(X)
                return pd.DataFrame({".pred_class": predictions})

            elif type == "prob":
                # Get probability predictions
                probs = model.predict_proba(X)
                class_names = model.classes_

                # Create DataFrame with probability columns
                prob_df = pd.DataFrame(
                    probs,
                    columns=[f".pred_{cls}" for cls in class_names]
                )
                return prob_df

            else:
                raise ValueError(
                    f"For classification, type must be 'class' or 'prob', got '{type}'"
                )

    def extract_outputs(
        self, fit: ModelFit
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract comprehensive three-DataFrame output.

        Returns:
            Tuple of (outputs, coefficients, stats)

            - Outputs: Observation-level results with actuals, fitted, residuals
            - Coefficients: Feature importance values from bagged trees
            - Stats: Comprehensive metrics by split + model info
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
            if fit.spec.mode == "regression":
                y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train
                # Create forecast: actuals where they exist, fitted where they don't
                forecast_train = pd.Series(y_train_array).combine_first(pd.Series(fitted)).values

                train_df = pd.DataFrame({
                    "actuals": y_train_array,
                    "fitted": fitted,
                    "forecast": forecast_train,
                    "residuals": residuals if residuals is not None else y_train_array - fitted,
                    "split": "train",
                })
            else:
                # Classification - no residuals/forecast
                y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train
                train_df = pd.DataFrame({
                    "actuals": y_train_array,
                    "predicted": fitted,
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

            if fit.spec.mode == "regression":
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
            else:
                test_predictions = test_preds[".pred_class"].values if ".pred_class" in test_preds.columns else test_preds.iloc[:, 0].values
                test_df = pd.DataFrame({
                    "actuals": test_actuals,
                    "predicted": test_predictions,
                    "split": "test",
                })

            # Add model metadata
            test_df["model"] = fit.model_name if fit.model_name else fit.spec.model_type
            test_df["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
            test_df["group"] = "global"

            outputs_list.append(test_df)

        outputs = pd.concat(outputs_list, ignore_index=True) if outputs_list else pd.DataFrame()

        # ====================
        # 2. COEFFICIENTS DataFrame (Feature Importance)
        # ====================
        # Bagged trees provide feature importance instead of coefficients
        feature_names = fit.fit_data.get("feature_names", [])

        # Get feature importances if available
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            # Calculate average importance from base estimators
            importances = np.zeros(len(feature_names))
            for estimator in model.estimators_:
                if hasattr(estimator, "feature_importances_"):
                    importances += estimator.feature_importances_
            importances /= len(model.estimators_)

        coefficients = pd.DataFrame({
            "variable": feature_names,
            "coefficient": importances,  # Using "coefficient" column for consistency
            "std_error": [np.nan] * len(feature_names),
            "t_stat": [np.nan] * len(feature_names),
            "p_value": [np.nan] * len(feature_names),
            "ci_0.025": [np.nan] * len(feature_names),
            "ci_0.975": [np.nan] * len(feature_names),
            "vif": [np.nan] * len(feature_names),
        })

        # Add model metadata
        coefficients["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        coefficients["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        coefficients["group"] = "global"

        # ====================
        # 3. STATS DataFrame
        # ====================
        stats_rows = []

        # Training metrics (regression only)
        if fit.spec.mode == "regression" and y_train is not None and fitted is not None:
            y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train
            train_metrics = self._calculate_metrics(y_train_array, fitted)

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

            for metric_name, value in test_metrics.items():
                stats_rows.append({
                    "metric": metric_name,
                    "value": value,
                    "split": "test",
                })

        # Residual diagnostics (regression only, on training data)
        if fit.spec.mode == "regression" and residuals is not None and len(residuals) > 0:
            resid_diag = self._calculate_residual_diagnostics(residuals)
            for metric_name, value in resid_diag.items():
                stats_rows.append({
                    "metric": metric_name,
                    "value": value,
                    "split": "train",
                })

        # Model-specific information
        n_estimators = fit.fit_data.get("n_estimators", 0)
        stats_rows.extend([
            {"metric": "formula", "value": fit.blueprint.formula if hasattr(fit.blueprint, "formula") else "", "split": ""},
            {"metric": "model_type", "value": fit.spec.model_type, "split": ""},
            {"metric": "model_class", "value": fit.fit_data.get("model_class", ""), "split": ""},
            {"metric": "base_estimator", "value": fit.fit_data.get("base_estimator", ""), "split": ""},
            {"metric": "n_estimators", "value": n_estimators, "split": ""},
            {"metric": "n_obs_train", "value": fit.fit_data.get("n_obs", 0), "split": "train"},
            {"metric": "n_features", "value": fit.fit_data.get("n_features", 0), "split": ""},
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

    def _calculate_metrics(self, actuals: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics for regression"""
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
            from statsmodels.stats import diagnostic as sm_diag
            # Ensure we have enough lags (at least 1, max 10 or n//5)
            n_lags = max(1, min(10, n // 5))
            lb_result = sm_diag.acorr_ljungbox(residuals, lags=n_lags)
            # Returns DataFrame with columns 'lb_stat' and 'lb_pvalue'
            results["ljung_box_stat"] = lb_result['lb_stat'].iloc[-1]  # Last lag statistic
            results["ljung_box_p"] = lb_result['lb_pvalue'].iloc[-1]  # Last lag p-value
        except Exception as e:
            results["ljung_box_stat"] = np.nan
            results["ljung_box_p"] = np.nan

        # Breusch-Pagan test placeholder (requires fitted model for heteroskedasticity)
        results["breusch_pagan_stat"] = np.nan
        results["breusch_pagan_p"] = np.nan

        return results
