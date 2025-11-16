"""
NeuralForecast NHITS engine for deep learning time series forecasting.

This engine implements NHITS (Neural Hierarchical Interpolation for Time Series)
using the NeuralForecast library. It inherits from BaseDLEngine for common
deep learning functionality.
"""

from typing import Dict, Any, Literal, Optional, Tuple
import pandas as pd
import numpy as np
import warnings
import time

from py_parsnip.engine_registry import register_engine
from py_parsnip.engines.base_dl_engine import BaseDLEngine
from py_parsnip.model_spec import ModelSpec, ModelFit


@register_engine("nhits_reg", "neuralforecast")
class NHITSEngine(BaseDLEngine):
    """
    NHITS engine for time series forecasting using NeuralForecast.

    Implements Neural Hierarchical Interpolation for Time Series (NHITS),
    a deep learning model that uses multi-rate input processing and
    hierarchical interpolation for efficient long-horizon forecasting.

    Parameter mapping: tidymodels → NeuralForecast
    - horizon → h
    - device → accelerator (with 'auto' → optimal device)
    - learning_rate → learning_rate
    - max_steps → max_steps
    - (all other parameters map directly)
    """

    param_map = {
        "horizon": "h",
        "device": "accelerator",  # Special handling in translate_params
    }

    def _get_model_class(self):
        """Return NHITS model class from NeuralForecast."""
        try:
            from neuralforecast.models import NHITS
            return NHITS
        except ImportError as e:
            if "neuralforecast" in str(e).lower():
                raise ImportError(
                    "NeuralForecast is required for NHITS but is not installed. "
                    "Install it with:\n"
                    "  pip install neuralforecast\n"
                    "\n"
                    "For GPU support:\n"
                    "  NVIDIA GPU: pip install torch --index-url https://download.pytorch.org/whl/cu118\n"
                    "  Apple Silicon: pip install torch  # MPS support built-in\n"
                    "\n"
                    "Note: Requires PyTorch >= 1.13.0"
                )
            raise

    def fit_raw(
        self,
        spec: ModelSpec,
        data: pd.DataFrame,
        formula: str,
        date_col: str,
        original_training_data: Optional[pd.DataFrame] = None
    ) -> Tuple[Dict[str, Any], Any]:
        """
        Fit NHITS model using raw data (bypasses hardhat molding).

        Parameters
        ----------
        spec : ModelSpec
            Model specification with NHITS hyperparameters.
        data : pd.DataFrame
            Training data (may be preprocessed).
        formula : str
            Formula string (e.g., "sales ~ price + date").
        date_col : str
            Name of date column or '__index__' for DatetimeIndex.
        original_training_data : pd.DataFrame, optional
            Original unpreprocessed training data (for raw datetime values).

        Returns
        -------
        tuple of (dict, Any)
            - fit_data: Dictionary containing fitted model and metadata
            - blueprint: Simple dict blueprint for predictions

        Raises
        ------
        ValueError
            If data has insufficient observations (< input_size + horizon).
            If formula is invalid or missing required columns.
            If device is unavailable.
        ImportError
            If NeuralForecast is not installed.

        Notes
        -----
        Uses NeuralForecast's data format:
        - unique_id: Time series identifier
        - ds: Datetime column
        - y: Target variable
        - [exog_cols]: Optional exogenous variables
        """
        # Check NeuralForecast availability
        self._check_neuralforecast_availability()

        # Parse formula to extract target and exogenous variables
        target_col, exog_vars = self._parse_formula(formula, date_col)

        # Expand "." notation to all columns except target and date
        exog_vars = self._expand_dot_notation(exog_vars, data, target_col, date_col)

        # Validate target column exists
        if target_col not in data.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )

        # Infer frequency from date column
        freq = self._infer_frequency(data, date_col)

        # Convert to NeuralForecast format
        nf_data = self._prepare_neuralforecast_data(
            data=data,
            target_col=target_col,
            date_col=date_col,
            exog_cols=exog_vars if exog_vars else None,
            group_col=None  # Single time series for now
        )

        # Extract hyperparameters from spec
        horizon = spec.args.get("h", 1)
        input_size = spec.args.get("input_size")
        validation_split = spec.args.get("validation_split", 0.2)
        device = spec.args.get("device", "auto")
        random_seed = spec.args.get("random_seed", 1)

        # Auto-calculate input_size if not provided
        if input_size is None:
            # Heuristic: max(2 * horizon, 7 * frequency_multiplier)
            freq_multiplier = {
                'H': 24,   # Hourly: 7 days = 168 hours
                'D': 7,    # Daily: 7 days
                'W': 4,    # Weekly: 4 weeks
                'M': 12,   # Monthly: 12 months
                'Q': 4,    # Quarterly: 4 quarters
                'Y': 2,    # Yearly: 2 years
            }.get(freq[0] if freq else 'D', 7)
            input_size = max(2 * horizon, 7 * freq_multiplier)

        # Validate data has sufficient observations
        n_obs = len(nf_data)
        min_obs = input_size + horizon
        if n_obs < min_obs:
            raise ValueError(
                f"Insufficient data for NHITS training. "
                f"Need at least {min_obs} observations (input_size={input_size} + horizon={horizon}), "
                f"but got {n_obs}. Consider reducing input_size or horizon."
            )

        # Validate and get device
        validated_device = self._validate_device(device)

        # Create validation split
        if validation_split > 0:
            train_data, val_data = self._create_validation_split(
                nf_data, val_proportion=validation_split, date_col='ds'
            )
        else:
            train_data = nf_data
            val_data = None

        # Build model parameters
        model_params = self._build_model_params(spec, input_size, freq)

        # Create NHITS model
        NHITS = self._get_model_class()
        model = NHITS(**model_params)

        # Import NeuralForecast wrapper
        from neuralforecast import NeuralForecast

        # Create NeuralForecast wrapper with single model
        nf = NeuralForecast(
            models=[model],
            freq=freq
        )

        # Fit model (with timing)
        start_time = time.time()

        # Suppress NeuralForecast logging
        import logging
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        logging.getLogger("neuralforecast").setLevel(logging.ERROR)

        # Fit with validation if available
        if val_data is not None:
            # NeuralForecast doesn't have explicit validation parameter
            # So we concatenate and let it use the full data
            # The model's internal validation split will be used
            nf.fit(df=train_data)
        else:
            nf.fit(df=train_data)

        train_time = time.time() - start_time

        # Get training history if available
        training_history = self._extract_training_history(model)

        # Get fitted values on training data (in-sample predictions)
        # Use the fitted model to predict on training data
        fitted_forecast = nf.predict()

        # Extract fitted values for the target series
        # NeuralForecast returns forecasts in a specific format
        fitted = fitted_forecast['NHITS'].values

        # Get actuals (last horizon values from training data)
        actuals = train_data['y'].values[-horizon:]

        # For full training data actuals, we need all values
        all_actuals = train_data['y'].values

        # Pad fitted to match training data length (fill with NaN for early observations)
        n_train = len(train_data)
        fitted_full = np.full(n_train, np.nan)
        fitted_full[-horizon:] = fitted  # Only last horizon values are predicted

        # Calculate residuals (only for predicted values)
        residuals_full = np.full(n_train, np.nan)
        residuals_full[-horizon:] = all_actuals[-horizon:] - fitted

        # Get dates
        dates = train_data['ds'].values

        # Create blueprint for predictions
        blueprint = {
            "formula": formula,
            "outcome_name": target_col,
            "date_col": date_col,
            "exog_vars": exog_vars,
            "freq": freq,
            "input_size": input_size,
            "horizon": horizon,
        }

        # Store fit data
        fit_data = {
            "model": nf,  # Store the NeuralForecast wrapper
            "nhits_model": model,  # Store the NHITS model itself
            "n_obs": n_train,
            "date_col": date_col,
            "outcome_name": target_col,
            "exog_vars": exog_vars,
            "freq": freq,
            "input_size": input_size,
            "horizon": horizon,
            "y_train": all_actuals,
            "fitted": fitted_full,
            "residuals": residuals_full,
            "dates": dates,
            "training_history": training_history,
            "train_time": train_time,
            "device": validated_device,
            "random_seed": random_seed,
        }

        return fit_data, blueprint

    def predict_raw(
        self,
        fit: ModelFit,
        new_data: pd.DataFrame,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make predictions using fitted NHITS model.

        Parameters
        ----------
        fit : ModelFit
            Fitted model with trained NHITS.
        new_data : pd.DataFrame
            New data for predictions. Must contain date column and
            exogenous variables (if used during training).
        type : str
            Prediction type:
            - 'numeric': Point predictions
            - 'conf_int': Prediction intervals (if model supports)

        Returns
        -------
        pd.DataFrame
            Predictions indexed by date.
            Columns depend on type:
            - 'numeric': .pred
            - 'conf_int': .pred, .pred_lower, .pred_upper

        Raises
        ------
        ValueError
            If required columns missing from new_data.
            If type is unsupported.

        Notes
        -----
        NHITS returns forecasts for the next 'horizon' steps.
        The predictions are indexed by the forecast dates.
        """
        if type not in ("numeric", "conf_int"):
            raise ValueError(
                f"nhits_reg supports type='numeric' or 'conf_int', got '{type}'"
            )

        # Extract model and metadata
        nf = fit.fit_data["model"]
        date_col = fit.fit_data["date_col"]
        exog_vars = fit.fit_data.get("exog_vars", [])
        horizon = fit.fit_data["horizon"]
        freq = fit.fit_data["freq"]

        # Get date values (handle __index__ case)
        if date_col == '__index__':
            if not isinstance(new_data.index, pd.DatetimeIndex):
                raise ValueError(
                    "date_col is '__index__' but new_data does not have DatetimeIndex"
                )
            date_values = new_data.index
        else:
            if date_col not in new_data.columns:
                raise ValueError(
                    f"Date column '{date_col}' not found in new_data. "
                    f"Available columns: {list(new_data.columns)}"
                )
            date_values = new_data[date_col]

        # Convert to NeuralForecast format
        target_col = fit.fit_data["outcome_name"]

        # Create dummy target if not present (for forecasting future)
        if target_col not in new_data.columns:
            new_data_with_target = new_data.copy()
            new_data_with_target[target_col] = 0.0  # Dummy values
        else:
            new_data_with_target = new_data

        nf_data = self._prepare_neuralforecast_data(
            data=new_data_with_target,
            target_col=target_col,
            date_col=date_col,
            exog_cols=exog_vars if exog_vars else None,
            group_col=None
        )

        # Make predictions
        # NeuralForecast.predict() returns forecasts for horizon steps ahead
        forecast = nf.predict(df=nf_data)

        # Extract predictions
        # forecast has columns: unique_id, ds, NHITS
        predictions = forecast['NHITS'].values

        # Create date index for forecasted periods
        # The forecast starts from the last date in new_data
        last_date = pd.Timestamp(date_values.iloc[-1])

        # Generate future dates
        if freq == 'D':
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon,
                freq='D'
            )
        elif freq == 'W':
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(weeks=1),
                periods=horizon,
                freq='W'
            )
        elif freq == 'M':
            forecast_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=horizon,
                freq='M'
            )
        elif freq == 'H':
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(hours=1),
                periods=horizon,
                freq='H'
            )
        else:
            # Generic handling
            forecast_dates = pd.date_range(
                start=last_date,
                periods=horizon + 1,
                freq=freq
            )[1:]  # Exclude start date

        # Return based on type
        if type == "numeric":
            result = pd.DataFrame(
                {".pred": predictions},
                index=forecast_dates
            )
            return result
        else:  # conf_int
            # NHITS doesn't natively support prediction intervals
            # We could implement Monte Carlo dropout or quantile regression
            # For now, return point predictions with approximate intervals
            # based on training residuals

            residuals = fit.fit_data["residuals"]
            residuals_clean = residuals[~np.isnan(residuals)]

            if len(residuals_clean) > 0:
                # Use residual standard deviation for approximate intervals
                residual_std = np.std(residuals_clean)
                # 95% confidence interval (± 1.96 std)
                lower = predictions - 1.96 * residual_std
                upper = predictions + 1.96 * residual_std
            else:
                # No residuals available, return wide intervals
                lower = predictions - np.abs(predictions) * 0.2
                upper = predictions + np.abs(predictions) * 0.2

            result = pd.DataFrame(
                {
                    ".pred": predictions,
                    ".pred_lower": lower,
                    ".pred_upper": upper,
                },
                index=forecast_dates
            )
            return result

    def extract_outputs(
        self,
        fit: ModelFit
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract three-DataFrame output from fitted NHITS model.

        Parameters
        ----------
        fit : ModelFit
            Fitted NHITS model.

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
            - outputs: Observation-level results (actuals, fitted, forecast, residuals, split)
            - coefficients: Model hyperparameters (since no traditional coefficients)
            - stats: Performance metrics by split + model info

        Notes
        -----
        **outputs DataFrame:**
            - actuals: True values from training data
            - fitted: Model predictions (in-sample)
            - forecast: Combined actual/fitted (seamless series)
            - residuals: actuals - fitted
            - dates: Datetime values
            - split: 'train' (DL models don't have separate test split in fit)
            - model: 'nhits_reg'
            - model_group_name: 'nhits_reg_1'

        **coefficients DataFrame:**
            Deep learning models don't have traditional coefficients.
            Instead, we return key hyperparameters:
            - term: Parameter name
            - estimate: Parameter value
            - model, model_group_name: Model identifiers

        **stats DataFrame:**
            Performance metrics computed on training data:
            - rmse, mae, mape, smape, r_squared, mda: Performance metrics
            - train_time: Training duration (seconds)
            - n_obs: Number of observations
            - input_size: Lookback window
            - horizon: Forecast horizon
            - device: Compute device used
            - freq: Time series frequency
            - split: 'train'
            - model, model_group_name: Model identifiers
        """
        # Extract data from fit
        dates = fit.fit_data["dates"]
        actuals = fit.fit_data["y_train"]
        fitted = fit.fit_data["fitted"]
        residuals = fit.fit_data["residuals"]

        # Create forecast column (combine actuals with fitted using combine_first)
        forecast = pd.Series(actuals).combine_first(pd.Series(fitted)).values

        # Create outputs DataFrame
        outputs = pd.DataFrame({
            "actuals": actuals,
            "fitted": fitted,
            "forecast": forecast,
            "residuals": residuals,
            "dates": dates,
            "split": "train",
            "model": fit.spec.model_type,
            "model_group_name": f"{fit.spec.model_type}_1",
        })

        # Create coefficients DataFrame (hyperparameters)
        hyperparams = [
            ("horizon", fit.fit_data["horizon"]),
            ("input_size", fit.fit_data["input_size"]),
            ("learning_rate", fit.spec.args.get("learning_rate", np.nan)),
            ("max_steps", fit.spec.args.get("max_steps", np.nan)),
            ("batch_size", fit.spec.args.get("batch_size", np.nan)),
            ("n_stacks", len(fit.spec.args.get("n_freq_downsample", []))),
            ("dropout_prob_theta", fit.spec.args.get("dropout_prob_theta", np.nan)),
        ]

        coefficients = pd.DataFrame([
            {
                "term": name,
                "estimate": value,
                "model": fit.spec.model_type,
                "model_group_name": f"{fit.spec.model_type}_1",
            }
            for name, value in hyperparams
        ])

        # Calculate metrics on non-NaN predictions
        mask = ~np.isnan(fitted)
        if mask.sum() > 0:
            metrics = self._calculate_metrics(
                actuals[mask],
                fitted[mask]
            )
        else:
            # No predictions available (shouldn't happen)
            metrics = {
                "rmse": np.nan,
                "mae": np.nan,
                "mape": np.nan,
                "smape": np.nan,
                "r_squared": np.nan,
                "mda": np.nan,
            }

        # Create stats DataFrame
        stats = pd.DataFrame([{
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "mape": metrics["mape"],
            "smape": metrics["smape"],
            "r_squared": metrics["r_squared"],
            "mda": metrics["mda"],
            "train_time": fit.fit_data["train_time"],
            "n_obs": fit.fit_data["n_obs"],
            "input_size": fit.fit_data["input_size"],
            "horizon": fit.fit_data["horizon"],
            "device": fit.fit_data["device"],
            "freq": fit.fit_data["freq"],
            "split": "train",
            "model": fit.spec.model_type,
            "model_group_name": f"{fit.spec.model_type}_1",
        }])

        return outputs, coefficients, stats

    def _build_model_params(
        self,
        spec: ModelSpec,
        input_size: int,
        freq: str
    ) -> Dict[str, Any]:
        """
        Build NHITS model parameters from spec.

        Translates tidymodels parameters to NeuralForecast NHITS parameters.

        Parameters
        ----------
        spec : ModelSpec
            Model specification with hyperparameters.
        input_size : int
            Calculated input size (lookback window).
        freq : str
            Time series frequency.

        Returns
        -------
        dict
            Parameters for NHITS constructor.
        """
        args = spec.args.copy()

        # Translate device to accelerator
        device = args.pop("device", "auto")
        validated_device = self._validate_device(device)

        # Map device to NeuralForecast accelerator
        if validated_device == "cuda":
            accelerator = "gpu"
        elif validated_device == "mps":
            # NeuralForecast may not support MPS directly, fall back to CPU
            warnings.warn(
                "MPS (Apple Silicon) may not be fully supported by NeuralForecast. "
                "Falling back to CPU. Use device='cpu' to suppress this warning.",
                UserWarning
            )
            accelerator = "cpu"
        else:
            accelerator = "cpu"

        # Build model parameters
        model_params = {
            "h": args["h"],  # Horizon
            "input_size": input_size,
            "futr_exog_list": args.get("exog_vars"),  # Exogenous variables
            "n_freq_downsample": args.get("n_freq_downsample", [8, 4, 1]),
            "n_blocks": args.get("n_blocks", [1, 1, 1]),
            "mlp_units": args.get("mlp_units", [[512, 512], [512, 512], [512, 512]]),
            "n_pool_kernel_size": args.get("n_pool_kernel_size", [8, 4, 1]),
            "n_theta_hidden": args.get("n_theta_hidden", [256, 256, 256]),
            "pooling_mode": args.get("pooling_mode", "MaxPool1d"),
            "interpolation_mode": args.get("interpolation_mode", "linear"),
            "dropout_prob_theta": args.get("dropout_prob_theta", 0.0),
            "activation": args.get("activation", "ReLU"),
            "learning_rate": args.get("learning_rate", 1e-3),
            "max_steps": args.get("max_steps", 1000),
            "batch_size": args.get("batch_size", 32),
            "random_seed": args.get("random_seed", 1),
            "loss": args.get("loss", "mae").upper(),  # NeuralForecast expects uppercase
        }

        # Add early stopping if specified
        early_stop = args.get("early_stop_patience_steps")
        if early_stop is not None:
            model_params["early_stop_patience_steps"] = early_stop

        # Remove validation_split (not a model parameter)
        model_params.pop("validation_split", None)

        return model_params

    def _extract_training_history(self, model) -> Optional[Dict[str, Any]]:
        """
        Extract training history from fitted model.

        Parameters
        ----------
        model : NHITS
            Fitted NHITS model.

        Returns
        -------
        dict or None
            Training history if available (loss curves, etc.).
        """
        # NeuralForecast models store training info in trainer
        if hasattr(model, 'trainer') and model.trainer is not None:
            history = {
                "current_epoch": model.trainer.current_epoch,
                "global_step": model.trainer.global_step,
            }

            # Try to get logged metrics
            if hasattr(model.trainer, 'logged_metrics'):
                history["metrics"] = model.trainer.logged_metrics

            return history

        return None
