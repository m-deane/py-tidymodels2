"""
py-earth engine for MARS (Multivariate Adaptive Regression Splines)

Maps mars to py-earth's Earth model:
- num_terms → max_terms
- prod_degree → max_degree
- prune_method → ('none', 'forward', or default backward pruning)
"""

from typing import Dict, Any, Literal
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData


@register_engine("mars", "pyearth")
class PyEarthMarsEngine(Engine):
    """
    py-earth engine for MARS regression.

    Parameter mapping:
    - num_terms → max_terms (maximum number of terms)
    - prod_degree → max_degree (maximum interaction degree)
    - prune_method → controls enable_pruning and fast_K
    """

    param_map = {
        "num_terms": "max_terms",
        "prod_degree": "max_degree",
    }

    def fit(self, spec: ModelSpec, molded: MoldedData) -> Dict[str, Any]:
        """
        Fit MARS model using py-earth.

        Args:
            spec: ModelSpec with model configuration
            molded: MoldedData with outcomes and predictors

        Returns:
            Dict containing fitted model and metadata
        """
        from pyearth import Earth

        # Extract predictors and outcomes
        X = molded.predictors
        y = molded.outcomes

        # Flatten y if it's a DataFrame with single column
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y = y.iloc[:, 0]

        # Build Earth kwargs from spec args
        args = spec.args
        earth_kwargs = {}

        # Map num_terms to max_terms
        if "num_terms" in args:
            earth_kwargs["max_terms"] = args["num_terms"]

        # Map prod_degree to max_degree
        if "prod_degree" in args:
            earth_kwargs["max_degree"] = args["prod_degree"]

        # Handle prune_method
        prune_method = args.get("prune_method", "backward")
        if prune_method == "none":
            earth_kwargs["enable_pruning"] = False
        elif prune_method == "forward":
            earth_kwargs["enable_pruning"] = False
            earth_kwargs["fast_K"] = earth_kwargs.get("max_terms", 100)
        # Default is backward pruning (Earth default)

        # Create and fit model
        model = Earth(**earth_kwargs)
        model.fit(X, y)

        # Calculate fitted values and residuals
        fitted = model.predict(X)
        residuals = (y.values if isinstance(y, pd.Series) else y) - fitted

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
            "n_terms": len(model.basis_),
            "gcv": model.gcv_ if hasattr(model, "gcv_") else np.nan,
            "rsq": model.rsq_ if hasattr(model, "rsq_") else np.nan,
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
        Make predictions using fitted MARS model.

        Args:
            fit: ModelFit with fitted model
            molded: MoldedData with predictors
            type: Prediction type ("numeric" for regression)

        Returns:
            DataFrame with predictions
        """
        if type != "numeric":
            raise ValueError(f"mars only supports type='numeric', got '{type}'")

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
            Tuple of (outputs, basis_functions, stats)

            - Outputs: Observation-level results with actuals, fitted, residuals
            - Basis_functions: MARS basis functions with coefficients
            - Stats: Comprehensive metrics by split + model statistics
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

            # Create forecast: actuals where they exist, fitted where they don't
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
        # 2. BASIS FUNCTIONS DataFrame
        # ============================
        # Extract basis function information
        basis_list = []

        # Get basis functions
        if hasattr(model, "basis_"):
            for i, basis in enumerate(model.basis_):
                # Get coefficient if available
                coef = model.coef_[i] if hasattr(model, "coef_") and i < len(model.coef_) else np.nan

                # Get basis function description
                basis_desc = str(basis) if basis is not None else f"basis_{i}"

                basis_list.append({
                    "basis_id": i,
                    "description": basis_desc,
                    "coefficient": coef,
                })

        # If no basis functions, use simple coefficients
        if not basis_list and hasattr(model, "coef_"):
            feature_names = list(fit.blueprint.column_order)
            for i, coef in enumerate(model.coef_):
                basis_list.append({
                    "basis_id": i,
                    "description": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                    "coefficient": coef,
                })

        basis_functions = pd.DataFrame(basis_list) if basis_list else pd.DataFrame(
            columns=["basis_id", "description", "coefficient"]
        )

        # Add model metadata
        if not basis_functions.empty:
            basis_functions["model"] = fit.model_name if fit.model_name else fit.spec.model_type
            basis_functions["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
            basis_functions["group"] = "global"

        # ====================
        # 3. STATS DataFrame
        # ====================
        stats_rows = []

        n_obs = fit.fit_data.get("n_obs", 0)
        n_features = fit.fit_data.get("n_features", 0)
        n_terms = fit.fit_data.get("n_terms", 0)

        # Training metrics
        if y_train is not None and fitted is not None:
            train_metrics = self._calculate_metrics(
                y_train.values if isinstance(y_train, pd.Series) else y_train,
                fitted
            )

            # Add adjusted R-squared
            r_sq = train_metrics["r_squared"]
            if not np.isnan(r_sq) and n_obs > n_terms:
                adj_r_sq = 1 - (1 - r_sq) * (n_obs - 1) / (n_obs - n_terms - 1)
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

        # MARS-specific statistics
        gcv = fit.fit_data.get("gcv", np.nan)
        rsq = fit.fit_data.get("rsq", np.nan)

        stats_rows.extend([
            {"metric": "gcv", "value": gcv, "split": "train"},
            {"metric": "mars_rsq", "value": rsq, "split": "train"},
            {"metric": "n_terms", "value": n_terms, "split": ""},
            {"metric": "formula", "value": fit.blueprint.formula if hasattr(fit.blueprint, "formula") else "", "split": ""},
            {"metric": "model_type", "value": fit.spec.model_type, "split": ""},
            {"metric": "n_obs_train", "value": n_obs, "split": "train"},
            {"metric": "n_features", "value": n_features, "split": ""},
        ])

        stats = pd.DataFrame(stats_rows)

        # Add model metadata
        stats["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        stats["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        stats["group"] = "global"

        return outputs, basis_functions, stats
