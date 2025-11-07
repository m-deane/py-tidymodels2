"""
XGBoost engine for boosted trees

Maps boost_tree to XGBoost's XGBRegressor:
- trees → n_estimators
- tree_depth → max_depth
- learn_rate → learning_rate
- mtry → colsample_bytree (as fraction)
- min_n → min_child_weight
- loss_reduction → gamma
- sample_size → subsample
- stop_iter → early_stopping_rounds
"""

from typing import Dict, Any, Literal, Optional
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData


@register_engine("boost_tree", "xgboost")
class XGBoostBoostTreeEngine(Engine):
    """
    XGBoost engine for boosted trees.

    Parameter mapping:
    - trees → n_estimators
    - tree_depth → max_depth
    - learn_rate → learning_rate
    - mtry → colsample_bytree (as fraction of total features)
    - min_n → min_child_weight
    - loss_reduction → gamma
    - sample_size → subsample
    - stop_iter → early_stopping_rounds
    """

    param_map = {
        "trees": "n_estimators",
        "tree_depth": "max_depth",
        "learn_rate": "learning_rate",
        "mtry": "colsample_bytree",
        "min_n": "min_child_weight",
        "loss_reduction": "gamma",
        "sample_size": "subsample",
        "stop_iter": "early_stopping_rounds",
    }

    def fit(
        self,
        spec: ModelSpec,
        molded: MoldedData,
        original_training_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Fit boosted tree model using XGBoost.

        Args:
            spec: ModelSpec with model configuration
            molded: MoldedData with outcomes and predictors
            original_training_data: Original training DataFrame (optional, for date column extraction)

        Returns:
            Dict containing fitted model and metadata
        """
        from xgboost import XGBRegressor

        # Extract predictors and outcomes
        X = molded.predictors
        y = molded.outcomes

        # XGBoost doesn't like feature names with brackets
        # Rename columns to remove special characters
        X = X.copy()
        X.columns = [col.replace('[', '_').replace(']', '_').replace('<', '_') for col in X.columns]

        # Flatten y if it's a DataFrame with single column
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y = y.iloc[:, 0]

        # Build model arguments
        model_args = {}
        args = spec.args

        # Map parameters (convert integers to ensure they're not floats)
        if "trees" in args:
            model_args["n_estimators"] = int(args["trees"])
        if "tree_depth" in args:
            model_args["max_depth"] = int(args["tree_depth"])
        if "learn_rate" in args:
            model_args["learning_rate"] = args["learn_rate"]
        if "mtry" in args:
            # Convert to fraction if needed
            mtry = args["mtry"]
            n_features = X.shape[1]
            if mtry > 1:
                # Assume it's an integer count, convert to fraction
                model_args["colsample_bytree"] = min(int(mtry) / n_features, 1.0)
            else:
                # Already a fraction
                model_args["colsample_bytree"] = mtry
        if "min_n" in args:
            model_args["min_child_weight"] = int(args["min_n"])
        if "loss_reduction" in args:
            model_args["gamma"] = args["loss_reduction"]
        if "sample_size" in args:
            model_args["subsample"] = args["sample_size"]

        # Handle early stopping
        eval_set = None
        fit_params = {}
        if "stop_iter" in args:
            model_args["early_stopping_rounds"] = int(args["stop_iter"])
            # For early stopping, we need a validation set
            # Use last 20% of training data as validation
            split_idx = int(0.8 * len(X))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            eval_set = [(X_val, y_val)]
            fit_params["eval_set"] = eval_set
            fit_params["verbose"] = False
        else:
            X_train, y_train = X, y

        # Set random state for reproducibility
        model_args["random_state"] = 42

        # Create and fit model
        model = XGBRegressor(**model_args)
        model.fit(X_train, y_train, **fit_params)

        # Calculate fitted values and residuals on full training data
        fitted = model.predict(X)
        residuals = y.values if isinstance(y, pd.Series) else y
        residuals = residuals - fitted

        # Return fitted model and metadata
        return {
            "model": model,
            "n_features": X.shape[1],
            "n_obs": X.shape[0],
            "model_class": "XGBRegressor",
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
        Make predictions using fitted XGBoost model.

        Args:
            fit: ModelFit with fitted model
            molded: MoldedData with predictors
            type: Prediction type ("numeric" for regression)

        Returns:
            DataFrame with predictions
        """
        if type != "numeric":
            raise ValueError(f"boost_tree only supports type='numeric', got '{type}'")

        model = fit.fit_data["model"]
        X = molded.predictors

        # XGBoost doesn't like feature names with brackets
        # Rename columns to match training
        X = X.copy()
        X.columns = [col.replace('[', '_').replace(']', '_').replace('<', '_') for col in X.columns]

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
        denominator = np.abs(actuals) + np.abs(predictions)
        mask = denominator != 0
        smape = np.mean(2 * np.abs(residuals[mask]) / denominator[mask] * 100) if mask.any() else np.nan

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

    def extract_outputs(
        self, fit: ModelFit
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract comprehensive three-DataFrame output.

        Returns:
            Tuple of (outputs, feature_importance, stats)

            - Outputs: Observation-level results with actuals, fitted, residuals
            - Feature Importance: Variable importance scores
            - Stats: Comprehensive metrics by split
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
                "fitted": fitted,
                "forecast": forecast_train,
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

            # Create forecast
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
        # 2. FEATURE IMPORTANCE DataFrame
        # ====================
        feature_names = list(fit.blueprint.column_order)

        # Get feature importance from XGBoost
        if hasattr(model, "feature_importances_"):
            importance_values = model.feature_importances_
        else:
            importance_values = [0.0] * len(feature_names)

        feature_importance = pd.DataFrame({
            "variable": feature_names,
            "importance": importance_values,
        })

        # Sort by importance
        feature_importance = feature_importance.sort_values("importance", ascending=False).reset_index(drop=True)

        # Add model metadata
        feature_importance["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        feature_importance["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        feature_importance["group"] = "global"

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
            test_data = fit.evaluation_data["test_data"]
            test_preds = fit.evaluation_data["test_predictions"]
            outcome_col = fit.evaluation_data["outcome_col"]

            test_actuals = test_data[outcome_col].values
            test_forecast = test_preds[".pred"].values

            test_metrics = self._calculate_metrics(test_actuals, test_forecast)

            for metric_name, value in test_metrics.items():
                stats_rows.append({
                    "metric": metric_name,
                    "value": value,
                    "split": "test",
                })

        # Model information
        n_obs = fit.fit_data.get("n_obs", 0)
        stats_rows.extend([
            {"metric": "formula", "value": fit.blueprint.formula if hasattr(fit.blueprint, "formula") else "", "split": ""},
            {"metric": "model_type", "value": fit.spec.model_type, "split": ""},
            {"metric": "model_class", "value": fit.fit_data.get("model_class", ""), "split": ""},
            {"metric": "n_obs_train", "value": n_obs, "split": "train"},
            {"metric": "n_trees", "value": len(model.get_booster().get_dump()) if hasattr(model, "get_booster") else 0, "split": ""},
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

        return outputs, feature_importance, stats
