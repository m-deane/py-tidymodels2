"""
Base deep learning engine for NeuralForecast models.

This module provides a base class for all NeuralForecast-based engines (NHITS, NBEATS, TFT, etc.).
Handles common functionality like device management, data formatting, and validation splits.
"""

from typing import Dict, Any, Literal, Optional, List, Tuple
from abc import abstractmethod
import pandas as pd
import numpy as np
import warnings

from py_parsnip.engine_registry import Engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData
from py_parsnip.utils.device_utils import (
    get_optimal_device,
    validate_device,
    check_gpu_memory,
    device_context
)
from py_parsnip.utils.neuralforecast_utils import (
    convert_to_neuralforecast_format,
    parse_formula_for_dl,
    infer_frequency,
    create_validation_split,
    expand_dot_notation_for_dl
)


class BaseDLEngine(Engine):
    """
    Base class for NeuralForecast deep learning engines.

    Provides common functionality for all DL models:
    - Automatic GPU/CPU device management
    - NeuralForecast data formatting
    - Validation split creation
    - Frequency inference
    - Common metric calculations

    Subclasses must implement:
    - fit_raw(): Model training
    - predict_raw(): Predictions
    - extract_outputs(): Three-DataFrame output extraction
    - _get_model_class(): Return the NeuralForecast model class

    Note:
        All NeuralForecast models use the RAW path (fit_raw/predict_raw)
        instead of the standard path (fit/predict) because they require
        specific date handling that bypasses hardhat molding.
    """

    # Parameter mapping: tidymodels → NeuralForecast
    # Override in subclasses as needed
    param_map: Dict[str, str] = {}

    def fit(self, spec: ModelSpec, molded: MoldedData) -> Dict[str, Any]:
        """Not used for NeuralForecast models - use fit_raw() instead"""
        raise NotImplementedError(
            "NeuralForecast models use fit_raw() instead of fit(). "
            "This is because they require specific datetime handling that bypasses hardhat molding."
        )

    def predict(
        self,
        fit: ModelFit,
        molded: MoldedData,
        type: str,
    ) -> pd.DataFrame:
        """Not used for NeuralForecast models - use predict_raw() instead"""
        raise NotImplementedError(
            "NeuralForecast models use predict_raw() instead of predict(). "
            "This is because they require specific datetime handling that bypasses hardhat molding."
        )

    @abstractmethod
    def _get_model_class(self):
        """
        Return the NeuralForecast model class.

        Subclasses must implement this to return the appropriate model class
        (e.g., NHITS, NBEATS, TFT).

        Returns
        -------
        class
            NeuralForecast model class (e.g., NHITS, NBEATS, TFT).

        Examples
        --------
        >>> # In NHITSEngine subclass:
        >>> def _get_model_class(self):
        ...     from neuralforecast.models import NHITS
        ...     return NHITS
        """
        pass

    def _detect_device(self, prefer_gpu: bool = True) -> str:
        """
        Detect optimal compute device for model training.

        Automatically selects best available device:
        - CUDA GPU (NVIDIA) - highest priority
        - MPS (Apple Silicon) - second priority
        - CPU - fallback

        Parameters
        ----------
        prefer_gpu : bool, default=True
            If True, prefer GPU over CPU when available.
            If False, always use CPU.

        Returns
        -------
        str
            Device name: 'cuda', 'mps', or 'cpu'.

        Examples
        --------
        >>> engine = NHITSEngine()
        >>> device = engine._detect_device()
        >>> print(device)
        'cuda'  # On NVIDIA GPU system
        """
        return get_optimal_device(prefer_gpu=prefer_gpu)

    def _validate_device(self, device: str) -> str:
        """
        Validate device string and check availability.

        Parameters
        ----------
        device : str
            Requested device: 'cuda', 'mps', 'cpu', or 'auto'.

        Returns
        -------
        str
            Validated device string. Falls back to CPU if requested device unavailable.

        Examples
        --------
        >>> engine = NHITSEngine()
        >>> device = engine._validate_device('auto')
        >>> print(device)
        'cuda'  # On GPU system
        """
        return validate_device(device)

    def _check_device_memory(self, device: str, min_gb: float = 1.0) -> bool:
        """
        Check if device has sufficient memory for training.

        Parameters
        ----------
        device : str
            Device to check: 'cuda', 'mps', or 'cpu'.
        min_gb : float, default=1.0
            Minimum required memory in gigabytes.

        Returns
        -------
        bool
            True if sufficient memory available, False otherwise.

        Examples
        --------
        >>> engine = NHITSEngine()
        >>> if engine._check_device_memory('cuda', min_gb=2.0):
        ...     device = 'cuda'
        ... else:
        ...     device = 'cpu'
        """
        return check_gpu_memory(min_gb=min_gb, device=device)

    def _prepare_neuralforecast_data(
        self,
        data: pd.DataFrame,
        target_col: str,
        date_col: str,
        exog_cols: Optional[List[str]] = None,
        group_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Convert data to NeuralForecast format.

        Transforms py-tidymodels data structure to NeuralForecast's expected format:
        - unique_id: Time series identifier
        - ds: Datetime
        - y: Target variable
        - [exog_cols]: Exogenous variables (optional)

        Parameters
        ----------
        data : pd.DataFrame
            Input data in py-tidymodels format.
        target_col : str
            Name of target variable.
        date_col : str
            Name of date column. Use '__index__' for DatetimeIndex.
        exog_cols : list of str, optional
            Names of exogenous variables. Default is None.
        group_col : str, optional
            Name of group column for panel data. Default is None.

        Returns
        -------
        pd.DataFrame
            Data in NeuralForecast format.

        Examples
        --------
        >>> engine = NHITSEngine()
        >>> nf_data = engine._prepare_neuralforecast_data(
        ...     data=train_df,
        ...     target_col='sales',
        ...     date_col='date',
        ...     exog_cols=['price', 'promo']
        ... )
        >>> print(nf_data.columns.tolist())
        ['unique_id', 'ds', 'y', 'price', 'promo']
        """
        return convert_to_neuralforecast_format(
            data=data,
            date_col=date_col,
            target_col=target_col,
            exog_cols=exog_cols,
            group_col=group_col
        )

    def _infer_frequency(self, data: pd.DataFrame, date_col: str) -> str:
        """
        Infer time series frequency from data.

        NeuralForecast models require explicit frequency specification.
        This method infers it from the date column.

        Parameters
        ----------
        data : pd.DataFrame
            Input data with datetime column or DatetimeIndex.
        date_col : str
            Name of date column. Use '__index__' for DatetimeIndex.

        Returns
        -------
        str
            Inferred frequency string (e.g., 'D', 'W', 'M', 'H').

        Raises
        ------
        ValueError
            If frequency cannot be inferred.

        Examples
        --------
        >>> engine = NHITSEngine()
        >>> freq = engine._infer_frequency(daily_data, 'date')
        >>> print(freq)
        'D'
        """
        if date_col == '__index__':
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError(
                    "date_col is '__index__' but data does not have DatetimeIndex"
                )
            datetime_index = data.index
        else:
            if date_col not in data.columns:
                raise ValueError(
                    f"Date column '{date_col}' not found in data. "
                    f"Available columns: {list(data.columns)}"
                )
            datetime_index = pd.DatetimeIndex(data[date_col])

        return infer_frequency(datetime_index)

    def _create_validation_split(
        self,
        data: pd.DataFrame,
        val_proportion: float = 0.2,
        date_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create chronological train/validation split.

        Splits data into training and validation sets using chronological ordering
        to prevent data leakage in time series forecasting.

        Parameters
        ----------
        data : pd.DataFrame
            Input data to split.
        val_proportion : float, default=0.2
            Proportion of data for validation (0 < val_proportion < 1).
        date_col : str, optional
            Name of date column for sorting. Use '__index__' for DatetimeIndex.
            If None, uses row order without sorting.

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame)
            - train_data: Training subset (first 80%)
            - val_data: Validation subset (last 20%)

        Examples
        --------
        >>> engine = NHITSEngine()
        >>> train, val = engine._create_validation_split(
        ...     data=df,
        ...     val_proportion=0.2,
        ...     date_col='date'
        ... )
        >>> len(train), len(val)
        (800, 200)  # For 1000 observations
        """
        return create_validation_split(
            data=data,
            val_proportion=val_proportion,
            method='time_based',
            date_col=date_col
        )

    def _parse_formula(self, formula: str, date_col: str) -> Tuple[str, List[str]]:
        """
        Parse formula to extract target and exogenous variables.

        Parameters
        ----------
        formula : str
            Formula string (e.g., "sales ~ price + promo + date").
        date_col : str
            Name of date column to exclude. Use '__index__' for DatetimeIndex.

        Returns
        -------
        tuple of (str, list of str)
            - target: Target variable name
            - exog_vars: List of exogenous variable names (excludes date)

        Examples
        --------
        >>> engine = NHITSEngine()
        >>> target, exog = engine._parse_formula("sales ~ price + date", "date")
        >>> print(target, exog)
        'sales' ['price']
        """
        return parse_formula_for_dl(formula, date_col)

    def _expand_dot_notation(
        self,
        exog_vars: List[str],
        data: pd.DataFrame,
        target_col: str,
        date_col: str
    ) -> List[str]:
        """
        Expand "." notation to all columns except target and date.

        Parameters
        ----------
        exog_vars : list of str
            Exogenous variable names. May contain ['.'].
        data : pd.DataFrame
            Data containing all columns.
        target_col : str
            Target variable name to exclude.
        date_col : str
            Date column name to exclude.

        Returns
        -------
        list of str
            Expanded column names or original list if no '.'.

        Examples
        --------
        >>> engine = NHITSEngine()
        >>> expanded = engine._expand_dot_notation(['.'], df, 'sales', 'date')
        >>> print(expanded)
        ['price', 'promo', 'inventory']
        """
        return expand_dot_notation_for_dl(exog_vars, data, target_col, date_col)

    def _calculate_metrics(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate standard time series performance metrics.

        Computes:
        - RMSE: Root Mean Squared Error
        - MAE: Mean Absolute Error
        - MAPE: Mean Absolute Percentage Error
        - SMAPE: Symmetric Mean Absolute Percentage Error
        - R²: Coefficient of determination
        - MDA: Mean Directional Accuracy

        Parameters
        ----------
        actuals : np.ndarray
            True values.
        predictions : np.ndarray
            Predicted values.

        Returns
        -------
        dict
            Dictionary of metric names and values.

        Examples
        --------
        >>> engine = NHITSEngine()
        >>> metrics = engine._calculate_metrics(y_true, y_pred)
        >>> print(metrics['rmse'])
        2.34
        """
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

    def _check_neuralforecast_availability(self):
        """
        Check if NeuralForecast is installed and provide helpful error message.

        Raises
        ------
        ImportError
            If NeuralForecast is not installed, with installation instructions.

        Examples
        --------
        >>> engine = NHITSEngine()
        >>> engine._check_neuralforecast_availability()  # Raises if not installed
        """
        try:
            import neuralforecast
        except ImportError:
            raise ImportError(
                "NeuralForecast is required for deep learning models but is not installed. "
                "Install it with:\n"
                "  pip install neuralforecast\n"
                "\n"
                "For GPU support (NVIDIA), also install PyTorch with CUDA:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu118\n"
                "\n"
                "For Apple Silicon (M1/M2), PyTorch with MPS support:\n"
                "  pip install torch\n"
                "\n"
                "Note: NeuralForecast requires PyTorch >= 1.13.0"
            )

    def _get_model_info(self, model, spec: ModelSpec) -> Dict[str, Any]:
        """
        Extract model metadata for stats DataFrame.

        Parameters
        ----------
        model
            Fitted NeuralForecast model instance.
        spec : ModelSpec
            Model specification.

        Returns
        -------
        dict
            Dictionary of model information (hyperparameters, architecture details).

        Examples
        --------
        >>> engine = NHITSEngine()
        >>> info = engine._get_model_info(fitted_model, spec)
        >>> print(info['model_type'])
        'nhits_reg'
        """
        info = {
            'model_type': spec.model_type,
            'engine': spec.engine,
        }

        # Add common hyperparameters from spec.args
        for param_name in ['h', 'input_size', 'learning_rate', 'max_steps', 'batch_size']:
            if param_name in spec.args:
                info[param_name] = spec.args[param_name]

        return info

    # Abstract methods that subclasses must implement
    @abstractmethod
    def fit_raw(
        self,
        spec: ModelSpec,
        data: pd.DataFrame,
        formula: str,
        date_col: str,
        original_training_data: Optional[pd.DataFrame] = None
    ) -> Tuple[Dict[str, Any], Any]:
        """
        Fit NeuralForecast model using raw data.

        Subclasses must implement this method to train their specific model.

        Parameters
        ----------
        spec : ModelSpec
            Model specification with hyperparameters.
        data : pd.DataFrame
            Training data (may be preprocessed).
        formula : str
            Formula string (e.g., "sales ~ price + date").
        date_col : str
            Name of date column or '__index__' for DatetimeIndex.
        original_training_data : pd.DataFrame, optional
            Original unpreprocessed training data.

        Returns
        -------
        tuple of (dict, Any)
            - fit_data: Dictionary containing fitted model and metadata
            - blueprint: Simple dict blueprint for predictions

        Examples
        --------
        >>> # Implemented by NHITSEngine, NBEATSEngine, etc.
        >>> fit_data, blueprint = engine.fit_raw(spec, train_data, formula, date_col)
        """
        pass

    @abstractmethod
    def predict_raw(
        self,
        fit: ModelFit,
        new_data: pd.DataFrame,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make predictions using fitted NeuralForecast model.

        Subclasses must implement this method to generate predictions.

        Parameters
        ----------
        fit : ModelFit
            Fitted model with trained NeuralForecast model.
        new_data : pd.DataFrame
            New data for predictions.
        type : str
            Prediction type ('numeric' or 'conf_int').

        Returns
        -------
        pd.DataFrame
            Predictions indexed by date.

        Examples
        --------
        >>> # Implemented by NHITSEngine, NBEATSEngine, etc.
        >>> predictions = engine.predict_raw(fit, test_data, type='numeric')
        """
        pass

    @abstractmethod
    def extract_outputs(
        self,
        fit: ModelFit
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract three-DataFrame output from fitted model.

        Subclasses must implement this method to return standardized outputs.

        Parameters
        ----------
        fit : ModelFit
            Fitted model.

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
            - outputs: Observation-level results (actuals, fitted, residuals, dates)
            - coefficients: Model hyperparameters
            - stats: Performance metrics by split + model info

        Examples
        --------
        >>> # Implemented by NHITSEngine, NBEATSEngine, etc.
        >>> outputs, coeffs, stats = engine.extract_outputs(fit)
        """
        pass
