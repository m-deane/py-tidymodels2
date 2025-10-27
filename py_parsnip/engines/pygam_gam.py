"""
pygam engine for Generalized Additive Models (GAM)

Maps gen_additive_mod to pygam's LinearGAM:
- select_features → controls lambda grid search for feature selection
- adjust_deg_free → n_splines parameter (controls smoothness)
"""

from typing import Dict, Any, Literal
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData


@register_engine("gen_additive_mod", "pygam")
class PyGAMEngine(Engine):
    """
    pygam engine for Generalized Additive Models.

    Parameter mapping:
    - select_features → enables lambda search for regularization
    - adjust_deg_free → n_splines (number of splines per term)
    """

    def fit(self, spec: ModelSpec, molded: MoldedData) -> Dict[str, Any]:
        """
        Fit GAM using pygam.

        Args:
            spec: ModelSpec with model configuration
            molded: MoldedData with outcomes and predictors

        Returns:
            Dict containing fitted model and metadata
        """
        from pygam import LinearGAM, s

        # Extract predictors and outcomes
        X = molded.predictors
        y = molded.outcomes

        # Flatten y if it's a DataFrame with single column
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y = y.iloc[:, 0]

        # Convert to numpy for pygam
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y

        # Build GAM kwargs from spec args
        args = spec.args
        gam_kwargs = {}

        # Handle adjust_deg_free (maps to n_splines)
        n_splines = args.get("adjust_deg_free", 10)
        if isinstance(n_splines, float):
            n_splines = int(n_splines)

        # Build terms (smooth splines for each feature)
        n_features = X_array.shape[1]

        # Create terms for each feature
        if n_features == 0:
            raise ValueError("No features in input data")
        elif n_features == 1:
            terms = s(0, n_splines=n_splines)
        else:
            # Start with first term, then add remaining
            terms = s(0, n_splines=n_splines)
            for i in range(1, n_features):
                terms = terms + s(i, n_splines=n_splines)

        # Handle select_features
        select_features = args.get("select_features", False)

        # Create GAM
        model = LinearGAM(terms, **gam_kwargs)

        # Fit model
        if select_features:
            # Use gridsearch to find optimal lambda (regularization)
            # This effectively does feature selection
            model.gridsearch(X_array, y_array)
        else:
            # Standard fit
            model.fit(X_array, y_array)

        # Calculate fitted values and residuals
        fitted = model.predict(X_array)
        residuals = y_array - fitted

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
            "n_features": X_array.shape[1],
            "n_obs": X_array.shape[0],
            "n_splines": n_splines,
            "X_train": X,
            "y_train": y,
            "fitted": fitted,
            "residuals": residuals,
            "date_col": date_col,
            # GAM-specific statistics
            "aic": model.statistics_["AIC"] if hasattr(model, "statistics_") else np.nan,
            "aicc": model.statistics_["AICc"] if hasattr(model, "statistics_") else np.nan,
            "gcv": model.statistics_["GCV"] if hasattr(model, "statistics_") else np.nan,
            "pseudo_r2": model.statistics_["pseudo_r2"]["explained_deviance"]
                        if hasattr(model, "statistics_") else np.nan,
        }

    def predict(
        self,
        fit: ModelFit,
        molded: MoldedData,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make predictions using fitted GAM.

        Args:
            fit: ModelFit with fitted model
            molded: MoldedData with predictors
            type: Prediction type
                - "numeric": Point predictions (default)
                - "conf_int": Predictions with confidence intervals

        Returns:
            DataFrame with predictions
        """
        model = fit.fit_data["model"]
        X = molded.predictors

        # Convert to numpy
        X_array = X.values if isinstance(X, pd.DataFrame) else X

        if type == "numeric":
            # Point predictions
            predictions = model.predict(X_array)
            return pd.DataFrame({".pred": predictions})

        elif type == "conf_int":
            # Predictions with confidence intervals
            predictions = model.predict(X_array)
            intervals = model.prediction_intervals(X_array, width=0.95)

            return pd.DataFrame({
                ".pred": predictions,
                ".pred_lower": intervals[:, 0],
                ".pred_upper": intervals[:, 1],
            })

        else:
            raise ValueError(
                f"gen_additive_mod supports type='numeric' or 'conf_int', got '{type}'"
            )

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

    def extract_outputs(
        self, fit: ModelFit
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract comprehensive three-DataFrame output.

        Returns:
            Tuple of (outputs, partial_effects, stats)

            - Outputs: Observation-level results with actuals, fitted, residuals
            - Partial_effects: Summary of smooth term effects (feature importance)
            - Stats: Comprehensive metrics by split + GAM statistics
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

        # ============================
        # 2. PARTIAL EFFECTS DataFrame
        # ============================
        # Extract information about smooth terms
        feature_names = list(fit.blueprint.column_order)
        n_features = fit.fit_data.get("n_features", 0)

        partial_effects_list = []

        # Get coefficients and statistics if available
        if hasattr(model, "coef_"):
            coef = model.coef_

            # Calculate approximate contribution of each feature
            X_train = fit.fit_data.get("X_train")
            if X_train is not None and isinstance(X_train, pd.DataFrame):
                X_array = X_train.values

                for i in range(min(n_features, len(feature_names))):
                    feature_name = feature_names[i]

                    # Get partial dependence for this feature
                    # pygam stores coefficients for each spline basis function
                    # We'll compute a simple measure of effect size

                    # Range of the feature
                    x_min = X_array[:, i].min()
                    x_max = X_array[:, i].max()
                    x_range = x_max - x_min

                    # Create grid for this feature (hold others at mean)
                    XX = np.tile(X_array.mean(axis=0), (100, 1))
                    XX[:, i] = np.linspace(x_min, x_max, 100)

                    # Get predictions across the range
                    try:
                        partial_pred = model.partial_dependence(i, XX)
                        effect_range = partial_pred.max() - partial_pred.min()
                    except:
                        effect_range = np.nan

                    partial_effects_list.append({
                        "feature": feature_name,
                        "feature_index": i,
                        "effect_range": effect_range,
                        "data_range": x_range,
                        "data_min": x_min,
                        "data_max": x_max,
                    })

        partial_effects = pd.DataFrame(partial_effects_list) if partial_effects_list else pd.DataFrame(
            columns=["feature", "feature_index", "effect_range", "data_range", "data_min", "data_max"]
        )

        # Add model metadata
        if not partial_effects.empty:
            partial_effects["model"] = fit.model_name if fit.model_name else fit.spec.model_type
            partial_effects["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
            partial_effects["group"] = "global"

        # ====================
        # 3. STATS DataFrame
        # ====================
        stats_rows = []

        n_obs = fit.fit_data.get("n_obs", 0)
        n_splines = fit.fit_data.get("n_splines", 0)

        # Training metrics
        if y_train is not None and fitted is not None:
            train_metrics = self._calculate_metrics(
                y_train.values if isinstance(y_train, pd.Series) else y_train,
                fitted
            )

            # Add adjusted R-squared
            r_sq = train_metrics["r_squared"]
            # Use effective degrees of freedom from GAM
            edf = model.statistics_["edof"] if hasattr(model, "statistics_") else n_features * n_splines
            if not np.isnan(r_sq) and n_obs > edf:
                adj_r_sq = 1 - (1 - r_sq) * (n_obs - 1) / (n_obs - edf - 1)
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

        # GAM-specific statistics (training only)
        gam_stats = {
            "aic": fit.fit_data.get("aic", np.nan),
            "aicc": fit.fit_data.get("aicc", np.nan),
            "gcv": fit.fit_data.get("gcv", np.nan),
            "pseudo_r2": fit.fit_data.get("pseudo_r2", np.nan),
        }

        for metric_name, value in gam_stats.items():
            stats_rows.append({
                "metric": metric_name,
                "value": value,
                "split": "train",
            })

        # Effective degrees of freedom
        if hasattr(model, "statistics_"):
            stats_rows.append({
                "metric": "edof",
                "value": model.statistics_["edof"],
                "split": "train",
            })

        # Model information
        stats_rows.extend([
            {"metric": "formula", "value": fit.blueprint.formula if hasattr(fit.blueprint, "formula") else "", "split": ""},
            {"metric": "model_type", "value": fit.spec.model_type, "split": ""},
            {"metric": "n_obs_train", "value": n_obs, "split": "train"},
            {"metric": "n_features", "value": n_features, "split": ""},
            {"metric": "n_splines", "value": n_splines, "split": ""},
        ])

        stats = pd.DataFrame(stats_rows)

        # Add model metadata
        stats["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        stats["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        stats["group"] = "global"

        return outputs, partial_effects, stats
