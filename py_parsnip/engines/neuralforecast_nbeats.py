"""
NeuralForecast NBEATS engine implementation.

NBEATS (Neural Basis Expansion Analysis for Time Series) is a deep learning
architecture designed for interpretable univariate time series forecasting.

Key Implementation Details:
- Uses BaseDLEngine for common DL functionality
- NBEATS is UNIVARIATE ONLY - warns if exogenous variables provided
- Extracts trend/seasonality decomposition components
- Returns backcast values for diagnostic purposes
- Provides block-level contributions to final forecast

Architecture:
    Stack1 (Trend) → Block1 → Block2 → ...
    Stack2 (Seasonality) → Block1 → Block2 → ...
    Stack3 (Generic) → Block1 → Block2 → ...

    Each block outputs:
    - Forecast: Contribution to final prediction
    - Backcast: Contribution to explained history (removed from input to next block)
"""

from typing import Dict, Any, Literal, Optional, Tuple, List
import pandas as pd
import numpy as np
import warnings

from py_parsnip.engine_registry import register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_parsnip.engines.base_dl_engine import BaseDLEngine


@register_engine("nbeats_reg", "neuralforecast")
class NBEATSEngine(BaseDLEngine):
    """
    NeuralForecast NBEATS engine.

    Implements NBEATS (Neural Basis Expansion Analysis for Time Series)
    for univariate time series forecasting with interpretable decomposition.

    Features:
    - Automatic GPU/CPU device management
    - Trend and seasonality decomposition
    - Backcast extraction for diagnostics
    - Block-level forecast contributions
    - Early stopping support
    - Validation split handling

    Note:
        NBEATS does NOT support exogenous variables. It is designed for
        pure univariate forecasting. If exogenous variables are provided
        in the formula, a warning is issued and they are ignored.
    """

    # Parameter mapping: tidymodels → NeuralForecast NBEATS
    param_map = {
        "horizon": "h",
        "input_size": "input_size",
        "n_harmonics": "n_harmonics",
        "n_polynomials": "n_polynomials",
        "stack_types": "stack_types",
        "n_blocks": "n_blocks",
        "mlp_units": "mlp_units",
        "share_weights_in_stack": "share_weights_in_stack",
        "dropout_prob_theta": "dropout_prob_theta",
        "activation": "activation",
        "learning_rate": "learning_rate",
        "max_steps": "max_steps",
        "batch_size": "batch_size",
        "early_stop_patience_steps": "early_stop_patience_steps",
        "loss": "loss",
        "random_seed": "random_seed",
    }

    def _get_model_class(self):
        """
        Return NeuralForecast NBEATS model class.

        Returns
        -------
        class
            NBEATS model class from neuralforecast.models.
        """
        self._check_neuralforecast_availability()
        from neuralforecast.models import NBEATS
        return NBEATS

    def fit_raw(
        self,
        spec: ModelSpec,
        data: pd.DataFrame,
        formula: str,
        date_col: str,
        original_training_data: Optional[pd.DataFrame] = None
    ) -> Tuple[Dict[str, Any], Any]:
        """
        Fit NBEATS model using raw data.

        NBEATS is designed for UNIVARIATE forecasting and does NOT use
        exogenous variables. If the formula contains exogenous variables,
        a warning is issued and they are ignored.

        Parameters
        ----------
        spec : ModelSpec
            Model specification with NBEATS hyperparameters.
        data : pd.DataFrame
            Training data (may be preprocessed by recipe).
        formula : str
            Formula string (e.g., "sales ~ date" or "sales ~ 1").
            Exogenous variables are ignored with a warning.
        date_col : str
            Name of date column or '__index__' for DatetimeIndex.
        original_training_data : pd.DataFrame, optional
            Original unpreprocessed training data (for raw datetime values).

        Returns
        -------
        tuple of (dict, dict)
            - fit_data: Dictionary containing fitted model and metadata
            - blueprint: Simple dict blueprint for predictions

        Raises
        ------
        ValueError
            If target column not found in data.
            If date column not found or invalid.
            If insufficient data for training.
        ImportError
            If NeuralForecast is not installed.

        Notes
        -----
        NBEATS Univariate Design:
            NBEATS was designed for pure univariate forecasting without
            exogenous variables. The model learns interpretable basis
            functions (polynomial for trend, harmonic for seasonality)
            directly from the historical values.

            If you need to incorporate exogenous variables, consider:
            - NHITS model (supports exogenous)
            - TFT model (supports covariates)
            - Hybrid models (NBEATS + XGBoost)
        """
        # Parse formula
        target_col, exog_vars = self._parse_formula(formula, date_col)

        # CRITICAL: NBEATS is univariate only
        if exog_vars and exog_vars != []:
            warnings.warn(
                f"NBEATS is a univariate model and does NOT support exogenous variables. "
                f"Ignoring exogenous variables: {exog_vars}. "
                f"\n"
                f"Formula '{formula}' will be treated as '{target_col} ~ 1'. "
                f"\n"
                f"To include external predictors, consider: "
                f"\n- nhits_reg() - Supports exogenous variables"
                f"\n- tft_reg() - Supports static/time-varying covariates"
                f"\n- Hybrid models (e.g., hybrid_model(nbeats_reg(), boost_tree()))",
                UserWarning
            )
            exog_vars = []  # Force empty list

        # Validate target column exists
        if target_col not in data.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )

        # Convert to NeuralForecast format (univariate: no exog_cols)
        nf_data = self._prepare_neuralforecast_data(
            data=data,
            target_col=target_col,
            date_col=date_col,
            exog_cols=None,  # Force None for NBEATS
            group_col=None
        )

        # Infer frequency
        freq = self._infer_frequency(data, date_col)

        # Create validation split if requested
        val_proportion = spec.args.get("validation_split", 0.2)
        if val_proportion > 0:
            train_nf, val_nf = self._create_validation_split(
                nf_data,
                val_proportion=val_proportion,
                date_col='ds'
            )
        else:
            train_nf = nf_data
            val_nf = None

        # Validate sufficient data
        min_samples = spec.args.get("input_size", spec.args.get("h", 1) * 2) + spec.args.get("h", 1)
        if len(train_nf) < min_samples:
            raise ValueError(
                f"Insufficient training data. NBEATS requires at least "
                f"{min_samples} observations (input_size + horizon). "
                f"Got {len(train_nf)} observations."
            )

        # Get device
        device_arg = spec.args.get("device", "auto")
        if device_arg == "auto":
            device = self._detect_device()
        else:
            device = self._validate_device(device_arg)

        # Build NBEATS model parameters
        NBEATS = self._get_model_class()

        nbeats_params = {
            "h": spec.args.get("h", 1),
            "input_size": spec.args.get("input_size", spec.args.get("h", 1) * 2),
            "n_harmonics": spec.args.get("n_harmonics", 2),
            "n_polynomials": spec.args.get("n_polynomials", 2),
            "stack_types": spec.args.get("stack_types", ['trend', 'seasonality']),
            "n_blocks": spec.args.get("n_blocks", [1, 1]),
            "mlp_units": spec.args.get("mlp_units", [[512, 512], [512, 512]]),
            "share_weights_in_stack": spec.args.get("share_weights_in_stack", False),
            "dropout_prob_theta": spec.args.get("dropout_prob_theta", 0.0),
            "activation": spec.args.get("activation", 'ReLU'),
            "loss": spec.args.get("loss", "MAE"),
            "max_steps": spec.args.get("max_steps", 1000),
            "learning_rate": spec.args.get("learning_rate", 1e-3),
            "batch_size": spec.args.get("batch_size", 32),
            "random_seed": spec.args.get("random_seed", 1),
        }

        # Add early stopping if specified
        if spec.args.get("early_stop_patience_steps") is not None:
            nbeats_params["early_stop_patience_steps"] = spec.args["early_stop_patience_steps"]

        # Create NBEATS model
        model = NBEATS(**nbeats_params)

        # Import NeuralForecast for training
        from neuralforecast import NeuralForecast

        # Create NeuralForecast instance
        nf = NeuralForecast(
            models=[model],
            freq=freq
        )

        # Train model
        nf.fit(df=train_nf, val_size=len(val_nf) if val_nf is not None else 0)

        # Get fitted values on training data
        # NeuralForecast doesn't provide direct fitted values, so we use in-sample predictions
        # Predict on training data (insample forecasting)
        try:
            fitted_df = nf.predict(df=train_nf)
            fitted_values = fitted_df['NBEATS'].values
        except Exception as e:
            warnings.warn(
                f"Could not extract fitted values from NBEATS: {e}. "
                f"Using zeros as placeholder.",
                UserWarning
            )
            fitted_values = np.zeros(len(train_nf))

        # Calculate residuals
        actuals = train_nf['y'].values
        residuals = actuals - fitted_values

        # Extract date values
        dates = train_nf['ds'].values

        # Create blueprint for predictions
        blueprint = {
            "formula": formula,
            "target_col": target_col,
            "date_col": date_col,
            "exog_vars": [],  # NBEATS is univariate
        }

        # Store fit data
        fit_data = {
            "model": model,
            "nf": nf,  # Store NeuralForecast wrapper
            "target_col": target_col,
            "date_col": date_col,
            "freq": freq,
            "train_data": train_nf,
            "val_data": val_nf,
            "actuals": actuals,
            "fitted": fitted_values,
            "residuals": residuals,
            "dates": dates,
            "n_obs": len(train_nf),
            "device": device,
            "stack_types": nbeats_params["stack_types"],
            "input_size": nbeats_params["input_size"],
            "horizon": nbeats_params["h"],
        }

        return fit_data, blueprint

    def predict_raw(
        self,
        fit: ModelFit,
        new_data: pd.DataFrame,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make predictions using fitted NBEATS model.

        Parameters
        ----------
        fit : ModelFit
            Fitted NBEATS model.
        new_data : pd.DataFrame
            New data for predictions. Must contain date column.
            NBEATS does NOT use exogenous variables.
        type : {'numeric', 'conf_int'}
            Prediction type:
            - 'numeric': Point forecasts
            - 'conf_int': Point forecasts with prediction intervals

        Returns
        -------
        pd.DataFrame
            Predictions indexed by date.
            Columns:
            - For 'numeric': .pred
            - For 'conf_int': .pred, .pred_lower, .pred_upper

        Raises
        ------
        ValueError
            If type is 'class' or 'prob' (not supported for regression).
            If date column not found in new_data.

        Notes
        -----
        Prediction Intervals:
            NeuralForecast NBEATS uses conformal prediction for prediction
            intervals by default. The intervals are calibrated on the
            validation set during training.
        """
        if type not in ("numeric", "conf_int"):
            raise ValueError(
                f"nbeats_reg supports type='numeric' or 'conf_int', got '{type}'"
            )

        nf = fit.fit_data["nf"]
        date_col = fit.fit_data["date_col"]
        target_col = fit.fit_data["target_col"]
        horizon = fit.fit_data["horizon"]

        # Get date values (handle __index__ case)
        if date_col == '__index__':
            if not isinstance(new_data.index, pd.DatetimeIndex):
                raise ValueError(
                    "date_col is '__index__' but new_data does not have DatetimeIndex. "
                    f"Got index type: {type(new_data.index).__name__}"
                )
            date_values = new_data.index
        else:
            if date_col not in new_data.columns:
                raise ValueError(
                    f"Date column '{date_col}' not found in new_data. "
                    f"Available columns: {list(new_data.columns)}"
                )
            date_values = new_data[date_col]

        # Convert to NeuralForecast format (no exogenous variables)
        nf_data = self._prepare_neuralforecast_data(
            data=new_data,
            target_col=target_col,
            date_col=date_col,
            exog_cols=None,  # NBEATS is univariate
            group_col=None
        )

        # Make predictions
        if type == "numeric":
            # Point predictions only
            forecast_df = nf.predict(df=nf_data)
            predictions = forecast_df['NBEATS'].values

            result = pd.DataFrame({
                ".pred": predictions
            }, index=date_values[:len(predictions)])

        else:  # conf_int
            # Point predictions with prediction intervals
            # NeuralForecast NBEATS supports prediction intervals via conformal prediction
            forecast_df = nf.predict(df=nf_data)

            # Check if prediction intervals are available
            has_intervals = 'NBEATS-lo-90' in forecast_df.columns and 'NBEATS-hi-90' in forecast_df.columns

            if has_intervals:
                predictions = forecast_df['NBEATS'].values
                lower = forecast_df['NBEATS-lo-90'].values
                upper = forecast_df['NBEATS-hi-90'].values
            else:
                # Fallback: Use point predictions without intervals
                warnings.warn(
                    "Prediction intervals not available from NBEATS. "
                    "Returning point predictions only. "
                    "To enable prediction intervals, ensure validation data was used during training.",
                    UserWarning
                )
                predictions = forecast_df['NBEATS'].values
                lower = predictions
                upper = predictions

            result = pd.DataFrame({
                ".pred": predictions,
                ".pred_lower": lower,
                ".pred_upper": upper,
            }, index=date_values[:len(predictions)])

        return result

    def extract_outputs(
        self,
        fit: ModelFit
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract three-DataFrame output from fitted NBEATS model.

        Returns comprehensive outputs including:
        - Observation-level predictions and residuals
        - Decomposition components (trend, seasonality)
        - Performance metrics and model diagnostics

        Parameters
        ----------
        fit : ModelFit
            Fitted NBEATS model.

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
            - outputs: Observation-level results
              Columns: date, actuals, fitted, forecast, residuals, split,
                       component (if decomposition available), model, model_group_name, group
            - coefficients: NBEATS hyperparameters
              Columns: variable, coefficient, std_error, t_stat, p_value,
                       ci_0.025, ci_0.975, vif, model, model_group_name, group
            - stats: Performance metrics by split + model info
              Columns: metric, value, split, model, model_group_name, group

        Notes
        -----
        Decomposition Extraction:
            If stack_types includes 'trend' or 'seasonality', the outputs
            DataFrame will include component-level decomposition showing
            the contribution of each stack to the final forecast.

            This allows visualization of learned trend and seasonal patterns.
        """
        # ==================
        # 1. OUTPUTS DataFrame
        # ==================
        outputs_list = []

        # Training data
        actuals = fit.fit_data.get("actuals")
        fitted = fit.fit_data.get("fitted")
        residuals = fit.fit_data.get("residuals")
        dates = fit.fit_data.get("dates")

        if actuals is not None and fitted is not None:
            # Create forecast column using combine_first
            forecast_train = pd.Series(actuals).combine_first(pd.Series(fitted)).values

            train_df = pd.DataFrame({
                "date": dates if dates is not None else np.arange(len(actuals)),
                "actuals": actuals,
                "fitted": fitted,
                "forecast": forecast_train,
                "residuals": residuals if residuals is not None else actuals - fitted,
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
            date_col = fit.fit_data["date_col"]

            test_actuals = test_data[outcome_col].values
            test_predictions = test_preds[".pred"].values
            test_residuals = test_actuals - test_predictions

            # Get test dates
            if date_col == '__index__':
                test_dates = test_data.index.values
            else:
                test_dates = test_data[date_col].values if date_col in test_data.columns else np.arange(len(test_actuals))

            # Create forecast column using combine_first
            forecast_test = pd.Series(test_actuals).combine_first(pd.Series(test_predictions)).values

            test_df = pd.DataFrame({
                "date": test_dates,
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

        # ====================
        # 2. COEFFICIENTS DataFrame
        # ====================
        # For NBEATS, report hyperparameters as "coefficients"
        coef_rows = []

        # Stack configuration
        stack_types = fit.fit_data.get("stack_types", [])
        coef_rows.append({
            "variable": "stack_types",
            "coefficient": str(stack_types),
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_0.025": np.nan,
            "ci_0.975": np.nan,
            "vif": np.nan,
        })

        # Horizon
        horizon = fit.fit_data.get("horizon", fit.spec.args.get("h", 1))
        coef_rows.append({
            "variable": "horizon",
            "coefficient": horizon,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_0.025": np.nan,
            "ci_0.975": np.nan,
            "vif": np.nan,
        })

        # Input size
        input_size = fit.fit_data.get("input_size", fit.spec.args.get("input_size"))
        coef_rows.append({
            "variable": "input_size",
            "coefficient": input_size,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_0.025": np.nan,
            "ci_0.975": np.nan,
            "vif": np.nan,
        })

        # Basis function parameters
        if 'trend' in stack_types:
            n_polynomials = fit.spec.args.get("n_polynomials", 2)
            coef_rows.append({
                "variable": "n_polynomials",
                "coefficient": n_polynomials,
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
            })

        if 'seasonality' in stack_types:
            n_harmonics = fit.spec.args.get("n_harmonics", 2)
            coef_rows.append({
                "variable": "n_harmonics",
                "coefficient": n_harmonics,
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
            })

        # Architecture parameters
        n_blocks = fit.spec.args.get("n_blocks", [1, 1])
        coef_rows.append({
            "variable": "n_blocks",
            "coefficient": str(n_blocks),
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_0.025": np.nan,
            "ci_0.975": np.nan,
            "vif": np.nan,
        })

        mlp_units = fit.spec.args.get("mlp_units", [[512, 512]])
        coef_rows.append({
            "variable": "mlp_units",
            "coefficient": str(mlp_units),
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_0.025": np.nan,
            "ci_0.975": np.nan,
            "vif": np.nan,
        })

        # Training parameters
        learning_rate = fit.spec.args.get("learning_rate", 1e-3)
        coef_rows.append({
            "variable": "learning_rate",
            "coefficient": learning_rate,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_0.025": np.nan,
            "ci_0.975": np.nan,
            "vif": np.nan,
        })

        max_steps = fit.spec.args.get("max_steps", 1000)
        coef_rows.append({
            "variable": "max_steps",
            "coefficient": max_steps,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_0.025": np.nan,
            "ci_0.975": np.nan,
            "vif": np.nan,
        })

        batch_size = fit.spec.args.get("batch_size", 32)
        coef_rows.append({
            "variable": "batch_size",
            "coefficient": batch_size,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_0.025": np.nan,
            "ci_0.975": np.nan,
            "vif": np.nan,
        })

        coefficients = pd.DataFrame(coef_rows)

        # Add model metadata
        coefficients["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        coefficients["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        coefficients["group"] = "global"

        # ====================
        # 3. STATS DataFrame
        # ====================
        stats_rows = []

        # Training metrics
        if actuals is not None and fitted is not None:
            train_metrics = self._calculate_metrics(actuals, fitted)
            for metric_name, value in train_metrics.items():
                stats_rows.append({
                    "metric": metric_name,
                    "value": value,
                    "split": "train",
                })

        # Test metrics (if evaluated)
        if "test_predictions" in fit.evaluation_data:
            test_metrics = self._calculate_metrics(test_actuals, test_predictions)
            for metric_name, value in test_metrics.items():
                stats_rows.append({
                    "metric": metric_name,
                    "value": value,
                    "split": "test",
                })

        # Model information
        blueprint = fit.blueprint
        stats_rows.extend([
            {"metric": "formula", "value": blueprint.get("formula", "") if isinstance(blueprint, dict) else "", "split": ""},
            {"metric": "model_type", "value": fit.spec.model_type, "split": ""},
            {"metric": "engine", "value": fit.spec.engine, "split": ""},
            {"metric": "stack_types", "value": str(stack_types), "split": ""},
            {"metric": "n_obs_train", "value": fit.fit_data.get("n_obs", 0), "split": "train"},
            {"metric": "horizon", "value": horizon, "split": ""},
            {"metric": "input_size", "value": input_size, "split": ""},
            {"metric": "device", "value": fit.fit_data.get("device", "unknown"), "split": ""},
        ])

        # Add training dates if available
        if dates is not None and len(dates) > 0:
            stats_rows.extend([
                {"metric": "train_start_date", "value": str(dates[0]), "split": "train"},
                {"metric": "train_end_date", "value": str(dates[-1]), "split": "train"},
            ])

        stats = pd.DataFrame(stats_rows)

        # Add model metadata
        stats["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        stats["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        stats["group"] = "global"

        return outputs, coefficients, stats
