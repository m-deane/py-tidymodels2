"""
Generic hybrid engine for combining arbitrary models

Supports four strategies:
1. residual: Model2 trained on residuals from Model1
2. sequential: Different models for different time periods
3. weighted: Weighted combination of predictions
4. custom_data: Models trained on different/overlapping datasets
"""

from typing import Dict, Any, Literal, Optional, Union
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData


@register_engine("hybrid_model", "generic_hybrid")
class GenericHybridEngine(Engine):
    """
    Generic hybrid engine for combining two arbitrary models.

    Strategies:
    - residual: Train model2 on residuals from model1
    - sequential: Different models for different periods
    - weighted: Weighted combination of predictions
    - custom_data: Train models on different/overlapping datasets
    """

    def fit(
        self,
        spec: ModelSpec,
        molded: Optional[MoldedData],
        original_training_data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None
    ) -> Dict[str, Any]:
        """
        Fit hybrid model using two sub-models.

        Args:
            spec: ModelSpec with hybrid configuration
            molded: MoldedData with predictors and outcomes (None for dict input)
            original_training_data: Original training DataFrame or dict of DataFrames

        Returns:
            Dict containing both fitted models and metadata
        """
        # Must have original data and formula for hybrid models
        if original_training_data is None:
            raise ValueError("hybrid_model requires original_training_data to be provided")

        # Extract formula from blueprint
        if molded is None:
            raise ValueError("molded data is required (even if minimal for dict input)")

        formula = molded.blueprint.formula if hasattr(molded.blueprint, 'formula') else None
        if formula is None:
            raise ValueError("hybrid_model requires a formula to be specified")

        # Extract strategy and models
        strategy = spec.args.get("strategy", "residual")
        model1_spec = spec.args.get("model1_spec")
        model2_spec = spec.args.get("model2_spec")
        split_point = spec.args.get("split_point")
        weight1 = spec.args.get("weight1", 0.5)
        weight2 = spec.args.get("weight2", 0.5)
        blend_predictions = spec.args.get("blend_predictions", "weighted")

        # Ensure models have mode set if needed
        if model1_spec.mode == "unknown":
            model1_spec = model1_spec.set_mode("regression")
        if model2_spec.mode == "unknown":
            model2_spec = model2_spec.set_mode("regression")

        # Extract outcome name from formula
        outcome_name = formula.split('~')[0].strip()

        # Check if original_training_data is a dict (custom_data strategy)
        if isinstance(original_training_data, dict):
            # Custom data strategy - separate datasets for each model
            if 'model1' not in original_training_data or 'model2' not in original_training_data:
                raise ValueError(
                    "When using dict data, must provide 'model1' and 'model2' keys. "
                    f"Got keys: {list(original_training_data.keys())}"
                )

            data1 = original_training_data['model1']
            data2 = original_training_data['model2']

            # Fit models on their respective datasets
            model1_fit = model1_spec.fit(data1, formula)
            model2_fit = model2_spec.fit(data2, formula)

            # Get fitted values from each model on their training data
            model1_outputs, _, _ = model1_fit.extract_outputs()
            model2_outputs, _, _ = model2_fit.extract_outputs()

            model1_fitted = model1_outputs[model1_outputs['split'] == 'train']['fitted'].values
            model2_fitted = model2_outputs[model2_outputs['split'] == 'train']['fitted'].values

            # Get y values from each dataset
            y_values_1 = data1[outcome_name].values
            y_values_2 = data2[outcome_name].values

            # For custom_data, we store both models and their separate data
            # We can't easily create combined fitted values since datasets may differ
            # So we store them separately
            return {
                "model1_fit": model1_fit,
                "model2_fit": model2_fit,
                "model1_spec": model1_spec,
                "model2_spec": model2_spec,
                "strategy": "custom_data",
                "blend_predictions": blend_predictions,
                "weight1": weight1,
                "weight2": weight2,
                "data1": data1,
                "data2": data2,
                "model1_fitted": model1_fitted,
                "model2_fitted": model2_fitted,
                "y_train_1": y_values_1,
                "y_train_2": y_values_2,
                "n_obs_1": len(y_values_1),
                "n_obs_2": len(y_values_2),
                "formula": formula,
                "outcome_name": outcome_name,
            }

        # Get y values (for non-dict strategies)
        y_values = original_training_data[outcome_name].values

        if strategy == "residual":
            # Strategy 1: Residual approach
            # Step 1: Fit model1 on original data
            model1_fit = model1_spec.fit(original_training_data, formula)

            # Step 2: Get model1 fitted values from extract_outputs
            model1_outputs, _, _ = model1_fit.extract_outputs()
            model1_fitted = model1_outputs[model1_outputs['split'] == 'train']['fitted'].values

            # Step 3: Calculate residuals
            residuals = y_values - model1_fitted

            # Step 4: Create modified data with residuals as outcome
            residual_data = original_training_data.copy()
            residual_data[outcome_name] = residuals

            # Step 5: Fit model2 on residuals (same formula)
            model2_fit = model2_spec.fit(residual_data, formula)

            # Step 6: Get model2 fitted values (predictions on residuals)
            model2_outputs, _, _ = model2_fit.extract_outputs()
            model2_fitted = model2_outputs[model2_outputs['split'] == 'train']['fitted'].values

            # Step 7: Combined fitted values = model1_pred + model2_pred
            fitted = model1_fitted + model2_fitted

        elif strategy == "sequential":
            # Strategy 2: Sequential approach (different periods)
            n = len(y_values)

            # Determine split index
            if isinstance(split_point, int):
                split_idx = split_point
            elif isinstance(split_point, float):
                split_idx = int(split_point * n)
            elif isinstance(split_point, str):
                # Assume split_point is a date string
                from py_parsnip.utils import _infer_date_column
                try:
                    date_col = _infer_date_column(original_training_data)
                    if date_col == '__index__':
                        dates = original_training_data.index
                    else:
                        dates = original_training_data[date_col]

                    # Find index where date >= split_point
                    split_idx = (dates >= split_point).idxmax() if isinstance(dates, pd.Series) else np.where(dates >= split_point)[0][0]
                except:
                    # Fallback to midpoint
                    split_idx = n // 2
            else:
                # Default to midpoint
                split_idx = n // 2

            # Split data into two periods
            period1_data = original_training_data.iloc[:split_idx]
            period2_data = original_training_data.iloc[split_idx:]

            # Fit both models on their respective periods
            model1_fit = model1_spec.fit(period1_data, formula)
            model2_fit = model2_spec.fit(period2_data, formula)

            # Get fitted values for each period
            model1_outputs, _, _ = model1_fit.extract_outputs()
            model1_fitted_period1 = model1_outputs[model1_outputs['split'] == 'train']['fitted'].values

            model2_outputs, _, _ = model2_fit.extract_outputs()
            model2_fitted_period2 = model2_outputs[model2_outputs['split'] == 'train']['fitted'].values

            # Combine fitted values
            fitted = np.concatenate([model1_fitted_period1, model2_fitted_period2])

        elif strategy == "weighted":
            # Strategy 3: Weighted combination
            # Fit both models on same data
            model1_fit = model1_spec.fit(original_training_data, formula)
            model2_fit = model2_spec.fit(original_training_data, formula)

            # Get fitted values
            model1_outputs, _, _ = model1_fit.extract_outputs()
            model1_fitted = model1_outputs[model1_outputs['split'] == 'train']['fitted'].values

            model2_outputs, _, _ = model2_fit.extract_outputs()
            model2_fitted = model2_outputs[model2_outputs['split'] == 'train']['fitted'].values

            # Weighted combination
            fitted = weight1 * model1_fitted + weight2 * model2_fitted

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Calculate residuals
        residuals = y_values - fitted

        # Get number of features
        X = molded.predictors
        n_features = X.shape[1] if hasattr(X, 'shape') else len(X.columns)

        # Return fitted models and metadata
        return {
            "model1_fit": model1_fit,
            "model2_fit": model2_fit,
            "model1_spec": model1_spec,
            "model2_spec": model2_spec,
            "strategy": strategy,
            "split_point": split_point,
            "weight1": weight1,
            "weight2": weight2,
            "n_obs": len(y_values),
            "n_features": n_features,
            "fitted": fitted,
            "residuals": residuals,
            "y_train": y_values,
            "original_training_data": original_training_data,
            "formula": formula,
        }

    def predict(
        self,
        fit: ModelFit,
        molded: MoldedData,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make predictions using fitted hybrid model.

        For predictions, we need new_data as DataFrame. Since we only have molded data,
        we'll use forge to reconstruct the data, or use internal prediction if needed.

        Args:
            fit: ModelFit with fitted hybrid models
            molded: MoldedData with predictors
            type: Prediction type ("numeric" for regression)

        Returns:
            DataFrame with predictions
        """
        if type != "numeric":
            raise ValueError(f"hybrid_model only supports type='numeric', got '{type}'")

        strategy = fit.fit_data["strategy"]
        model1_fit = fit.fit_data["model1_fit"]
        model2_fit = fit.fit_data["model2_fit"]
        weight1 = fit.fit_data.get("weight1", 0.5)
        weight2 = fit.fit_data.get("weight2", 0.5)

        # For prediction, we need to create a temporary DataFrame from molded data
        # This is needed because model.predict() expects DataFrame
        X = molded.predictors
        temp_data = pd.DataFrame(X)

        if strategy == "residual":
            # Get predictions from both models
            model1_preds = model1_fit.predict(temp_data)
            model2_preds = model2_fit.predict(temp_data)

            # Combined prediction = model1 + model2
            predictions = model1_preds[".pred"].values + model2_preds[".pred"].values

        elif strategy == "sequential":
            # For new data, use model2 (latest model)
            # In practice, you'd need to know which period the new data belongs to
            # For now, default to model2 as it represents the most recent regime
            model2_preds = model2_fit.predict(temp_data)
            predictions = model2_preds[".pred"].values

        elif strategy == "weighted":
            # Weighted combination
            model1_preds = model1_fit.predict(temp_data)
            model2_preds = model2_fit.predict(temp_data)

            predictions = weight1 * model1_preds[".pred"].values + weight2 * model2_preds[".pred"].values

        elif strategy == "custom_data":
            # Both models were trained on different/overlapping data
            # Get predictions from both
            model1_preds = model1_fit.predict(temp_data)
            model2_preds = model2_fit.predict(temp_data)

            # Blend predictions based on blend_predictions arg
            blend_type = fit.fit_data.get("blend_predictions", "weighted")

            if blend_type == "weighted":
                predictions = weight1 * model1_preds[".pred"].values + weight2 * model2_preds[".pred"].values
            elif blend_type == "avg":
                predictions = 0.5 * (model1_preds[".pred"].values + model2_preds[".pred"].values)
            elif blend_type == "sum":
                # Sum predictions from both models
                predictions = model1_preds[".pred"].values + model2_preds[".pred"].values
            elif blend_type == "model1":
                predictions = model1_preds[".pred"].values
            elif blend_type == "model2":
                predictions = model2_preds[".pred"].values
            else:
                raise ValueError(f"Unknown blend_predictions: {blend_type}")

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return pd.DataFrame({".pred": predictions})

    def extract_outputs(
        self, fit: ModelFit
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract comprehensive three-DataFrame output.

        Returns:
            Tuple of (outputs, coefficients, stats)
        """
        from py_yardstick import rmse, mae, mape, r_squared

        # ====================
        # 1. OUTPUTS DataFrame
        # ====================
        outputs_list = []

        y_train = fit.fit_data.get("y_train")
        fitted = fit.fit_data.get("fitted")
        residuals = fit.fit_data.get("residuals")

        if y_train is not None and fitted is not None:
            forecast_train = pd.Series(y_train).combine_first(pd.Series(fitted)).values

            train_df = pd.DataFrame({
                "actuals": y_train,
                "fitted": fitted,
                "forecast": forecast_train,
                "residuals": residuals if residuals is not None else y_train - fitted,
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
        # 2. COEFFICIENTS DataFrame
        # ====================
        # For hybrid models, report hyperparameters as "coefficients"
        strategy = fit.fit_data.get("strategy")
        model1_spec = fit.fit_data.get("model1_spec")
        model2_spec = fit.fit_data.get("model2_spec")

        coefficients_list = [
            {
                "variable": "strategy",
                "coefficient": strategy,
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
            },
            {
                "variable": "model1_type",
                "coefficient": model1_spec.model_type if model1_spec else "unknown",
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
            },
            {
                "variable": "model2_type",
                "coefficient": model2_spec.model_type if model2_spec else "unknown",
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
            },
        ]

        if strategy == "weighted":
            coefficients_list.extend([
                {
                    "variable": "weight1",
                    "coefficient": float(fit.fit_data.get("weight1", 0.5)),
                    "std_error": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "ci_0.025": np.nan,
                    "ci_0.975": np.nan,
                    "vif": np.nan,
                },
                {
                    "variable": "weight2",
                    "coefficient": float(fit.fit_data.get("weight2", 0.5)),
                    "std_error": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "ci_0.025": np.nan,
                    "ci_0.975": np.nan,
                    "vif": np.nan,
                },
            ])

        elif strategy == "custom_data":
            coefficients_list.extend([
                {
                    "variable": "weight1",
                    "coefficient": float(fit.fit_data.get("weight1", 0.5)),
                    "std_error": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "ci_0.025": np.nan,
                    "ci_0.975": np.nan,
                    "vif": np.nan,
                },
                {
                    "variable": "weight2",
                    "coefficient": float(fit.fit_data.get("weight2", 0.5)),
                    "std_error": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "ci_0.025": np.nan,
                    "ci_0.975": np.nan,
                    "vif": np.nan,
                },
                {
                    "variable": "blend_predictions",
                    "coefficient": fit.fit_data.get("blend_predictions", "weighted"),
                    "std_error": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "ci_0.025": np.nan,
                    "ci_0.975": np.nan,
                    "vif": np.nan,
                },
            ])

        coefficients = pd.DataFrame(coefficients_list)

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
            rmse_val = rmse(y_train, fitted)['value'].iloc[0]
            mae_val = mae(y_train, fitted)['value'].iloc[0]
            mape_val = mape(y_train, fitted)['value'].iloc[0]
            r2_val = r_squared(y_train, fitted)['value'].iloc[0]

            stats_rows.extend([
                {"metric": "rmse", "value": rmse_val, "split": "train"},
                {"metric": "mae", "value": mae_val, "split": "train"},
                {"metric": "mape", "value": mape_val, "split": "train"},
                {"metric": "r_squared", "value": r2_val, "split": "train"},
            ])

        # Test metrics (if evaluated)
        if "test_predictions" in fit.evaluation_data:
            test_data = fit.evaluation_data["test_data"]
            test_preds = fit.evaluation_data["test_predictions"]
            outcome_col = fit.evaluation_data["outcome_col"]

            test_actuals = test_data[outcome_col].values
            test_predictions = test_preds[".pred"].values

            test_rmse = rmse(test_actuals, test_predictions)['value'].iloc[0]
            test_mae = mae(test_actuals, test_predictions)['value'].iloc[0]
            test_mape = mape(test_actuals, test_predictions)['value'].iloc[0]
            test_r2 = r_squared(test_actuals, test_predictions)['value'].iloc[0]

            stats_rows.extend([
                {"metric": "rmse", "value": test_rmse, "split": "test"},
                {"metric": "mae", "value": test_mae, "split": "test"},
                {"metric": "mape", "value": test_mape, "split": "test"},
                {"metric": "r_squared", "value": test_r2, "split": "test"},
            ])

        # Model information
        n_obs = fit.fit_data.get("n_obs", 0)
        stats_rows.extend([
            {"metric": "formula", "value": fit.blueprint.formula if hasattr(fit.blueprint, "formula") else "", "split": ""},
            {"metric": "model_type", "value": fit.spec.model_type, "split": ""},
            {"metric": "strategy", "value": strategy, "split": ""},
            {"metric": "n_obs_train", "value": n_obs, "split": "train"},
        ])

        # For custom_data strategy, add per-model observation counts
        if strategy == "custom_data":
            n_obs_1 = fit.fit_data.get("n_obs_1", 0)
            n_obs_2 = fit.fit_data.get("n_obs_2", 0)
            stats_rows.extend([
                {"metric": "n_obs_model1", "value": n_obs_1, "split": "train"},
                {"metric": "n_obs_model2", "value": n_obs_2, "split": "train"},
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
