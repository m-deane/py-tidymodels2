"""
Utilities for conformal prediction intervals.

This module provides helper functions for:
- Auto-selecting conformal prediction methods
- Splitting calibration data
- Configuring MAPIE wrappers
- Formatting conformal prediction outputs
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split


def is_time_series_model(model_type: str) -> bool:
    """
    Check if a model type is a time series model.

    Time series models require special handling (EnbPI, block bootstrap)
    to preserve temporal structure and handle non-exchangeable data.

    Parameters
    ----------
    model_type : str
        The model type identifier (e.g., 'prophet_reg', 'linear_reg')

    Returns
    -------
    bool
        True if model is time series, False otherwise
    """
    time_series_models = {
        'prophet_reg',
        'arima_reg',
        'seasonal_reg',
        'exp_smoothing',
        'varmax_reg',
        'arima_boost',
        'prophet_boost',
        'recursive_reg'
    }
    return model_type in time_series_models


def auto_select_method(
    model_type: str,
    n_samples: int,
    is_time_series: Optional[bool] = None
) -> str:
    """
    Automatically select the best conformal prediction method.

    Selection logic:
    1. Time series models → 'enbpi' (always)
    2. Large datasets (>10k) → 'split' (O(1) complexity)
    3. Medium datasets (1k-10k) → 'cv+' (O(K) complexity, better intervals)
    4. Small datasets (<1k) → 'jackknife+' (O(n) complexity, data-efficient)

    Parameters
    ----------
    model_type : str
        The model type identifier
    n_samples : int
        Number of training samples
    is_time_series : bool, optional
        Override automatic time series detection

    Returns
    -------
    str
        Recommended method: 'split', 'cv+', 'jackknife+', or 'enbpi'

    References
    ----------
    - Split conformal: Vovk et al. (2005)
    - CV+: Barber et al. (2021)
    - Jackknife+: Barber et al. (2021)
    - EnbPI: Xu & Xie (2021)
    """
    # Check if time series
    if is_time_series is None:
        is_time_series = is_time_series_model(model_type)

    if is_time_series:
        return 'enbpi'

    # Select based on dataset size
    if n_samples > 10000:
        return 'split'
    elif n_samples > 1000:
        return 'cv+'
    else:
        return 'jackknife+'


def split_calibration_data(
    data: pd.DataFrame,
    formula: str,
    calibration_size: float = 0.15,
    random_state: Optional[int] = None,
    stratify_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and calibration sets.

    Reserves a portion of training data for calibration (computing
    nonconformity scores). The calibration set should be:
    - Independent from training (no data leakage)
    - Representative of test distribution
    - Large enough (aim for 1000+ samples, minimum 100)

    Parameters
    ----------
    data : DataFrame
        Full training dataset
    formula : str
        Model formula (e.g., 'y ~ x1 + x2')
    calibration_size : float, default=0.15
        Proportion to reserve for calibration (typically 0.1-0.2)
    random_state : int, optional
        Random seed for reproducibility
    stratify_col : str, optional
        Column name for stratified splitting (classification tasks)

    Returns
    -------
    train_data : DataFrame
        Data for model training
    calibration_data : DataFrame
        Data for conformal calibration

    Notes
    -----
    For time series data, use temporal splitting instead (most recent
    observations for calibration).
    """
    if calibration_size <= 0 or calibration_size >= 1:
        raise ValueError(
            f"calibration_size must be between 0 and 1, got {calibration_size}"
        )

    # Determine stratification
    stratify = None
    if stratify_col and stratify_col in data.columns:
        stratify = data[stratify_col]

    # Split
    train_data, cal_data = train_test_split(
        data,
        test_size=calibration_size,
        random_state=random_state,
        stratify=stratify
    )

    return train_data, cal_data


def split_calibration_time_series(
    data: pd.DataFrame,
    date_col: str,
    calibration_size: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data preserving temporal ordering.

    Uses most recent observations for calibration (not random sampling).
    This preserves exchangeability within each set and avoids look-ahead bias.

    Parameters
    ----------
    data : DataFrame
        Time series dataset (should be sorted by date)
    date_col : str
        Name of date/time column
    calibration_size : float, default=0.15
        Proportion to reserve for calibration

    Returns
    -------
    train_data : DataFrame
        Earlier observations for training
    calibration_data : DataFrame
        Recent observations for calibration
    """
    if calibration_size <= 0 or calibration_size >= 1:
        raise ValueError(
            f"calibration_size must be between 0 and 1, got {calibration_size}"
        )

    # Sort by date
    data = data.sort_values(date_col).reset_index(drop=True)

    # Split point
    n = len(data)
    split_idx = int(n * (1 - calibration_size))

    train_data = data.iloc[:split_idx]
    cal_data = data.iloc[split_idx:]

    return train_data, cal_data


def validate_conformal_params(
    alpha: Union[float, List[float]],
    method: str,
    n_samples: int
) -> None:
    """
    Validate conformal prediction parameters.

    Parameters
    ----------
    alpha : float or list of float
        Significance level(s), must be in (0, 1)
    method : str
        Conformal method identifier
    n_samples : int
        Number of samples in dataset

    Raises
    ------
    ValueError
        If parameters are invalid
    """
    # Validate alpha
    alphas = [alpha] if isinstance(alpha, (int, float)) else alpha
    for a in alphas:
        if not 0 < a < 1:
            raise ValueError(
                f"alpha must be between 0 and 1, got {a}"
            )

    # Validate method
    valid_methods = {'auto', 'split', 'cv+', 'jackknife+', 'jackknife-minmax', 'enbpi', 'cqr'}
    if method not in valid_methods:
        raise ValueError(
            f"method must be one of {valid_methods}, got '{method}'"
        )

    # Warn for small datasets with certain methods
    if method == 'split' and n_samples < 300:
        import warnings
        warnings.warn(
            f"Split conformal with small dataset (n={n_samples}) may produce "
            f"wide intervals. Consider 'cv+' or 'jackknife+' for better coverage.",
            UserWarning
        )

    # Warn for large datasets with Jackknife+
    if method == 'jackknife+' and n_samples > 5000:
        import warnings
        warnings.warn(
            f"Jackknife+ requires O(n) model fits (n={n_samples}). "
            f"This may be slow. Consider 'split' or 'cv+' for efficiency.",
            UserWarning
        )


def format_conformal_predictions(
    predictions: np.ndarray,
    intervals: np.ndarray,
    alpha: Union[float, List[float]],
    method: str
) -> pd.DataFrame:
    """
    Format conformal predictions into standardized DataFrame.

    Parameters
    ----------
    predictions : ndarray
        Point predictions, shape (n_samples,)
    intervals : ndarray
        Prediction intervals:
        - Single alpha: shape (n_samples, 2, 1) with [:, 0, 0] = lower, [:, 1, 0] = upper
        - Multiple alphas: shape (n_samples, 2, n_alpha)
    alpha : float or list of float
        Significance level(s)
    method : str
        Conformal method used

    Returns
    -------
    DataFrame
        Formatted predictions with columns:
        - .pred: point predictions
        - .pred_lower_{coverage}: lower bounds (if single alpha, no suffix)
        - .pred_upper_{coverage}: upper bounds (if single alpha, no suffix)
        - .conf_method: method identifier
        - .conf_alpha: significance level (if single alpha)
    """
    alphas = [alpha] if isinstance(alpha, (int, float)) else alpha
    n_alphas = len(alphas)

    # Base DataFrame
    result = pd.DataFrame({
        '.pred': predictions,
        '.conf_method': method
    })

    # Add interval columns
    if n_alphas == 1:
        # Single alpha: simple column names
        result['.pred_lower'] = intervals[:, 0, 0]
        result['.pred_upper'] = intervals[:, 1, 0]
        result['.conf_alpha'] = alphas[0]
        result['.conf_coverage'] = 1 - alphas[0]
    else:
        # Multiple alphas: suffixed column names
        for i, a in enumerate(alphas):
            coverage = int((1 - a) * 100)
            result[f'.pred_lower_{coverage}'] = intervals[:, 0, i]
            result[f'.pred_upper_{coverage}'] = intervals[:, 1, i]

    return result


def get_mapie_cv_config(method: str, n_samples: int, **kwargs) -> Dict[str, Any]:
    """
    Get cross-validation configuration for MAPIE.

    Parameters
    ----------
    method : str
        Conformal method ('split', 'cv+', 'jackknife+', etc.)
    n_samples : int
        Number of training samples
    **kwargs : dict
        Override defaults (cv, n_resamplings, etc.)

    Returns
    -------
    dict
        Configuration dict for MAPIE
    """
    config = {}

    if method == 'split':
        config['method'] = 'base'
        config['cv'] = 'split'

    elif method == 'cv+':
        config['method'] = 'plus'
        # Default to 5 folds for medium datasets, 10 for larger
        default_cv = 10 if n_samples > 5000 else 5
        config['cv'] = kwargs.get('cv', default_cv)

    elif method == 'jackknife+':
        config['method'] = 'plus'
        config['cv'] = -1  # Leave-one-out

    elif method == 'jackknife-minmax':
        config['method'] = 'minmax'
        config['cv'] = -1

    elif method == 'enbpi':
        config['method'] = 'enbpi'
        config['n_resamplings'] = kwargs.get('n_resamplings', 10)
        # BlockBootstrap configuration handled separately

    elif method == 'cqr':
        config['method'] = 'cqr'
        config['cv'] = kwargs.get('cv', 'split')

    # Add parallelization if available
    config['n_jobs'] = kwargs.get('n_jobs', -1)

    return config


def estimate_seasonal_period(
    data: pd.DataFrame,
    date_col: str
) -> int:
    """
    Estimate seasonal period from time series data.

    Parameters
    ----------
    data : DataFrame
        Time series data
    date_col : str
        Date column name

    Returns
    -------
    int
        Estimated period (e.g., 7 for daily with weekly seasonality,
        12 for monthly with annual seasonality), or 10 as default
    """
    if date_col not in data.columns:
        return 10  # Default fallback

    dates = pd.to_datetime(data[date_col])
    if len(dates) < 2:
        return 10  # Default fallback

    # Infer frequency
    try:
        freq = pd.infer_freq(dates)
    except (ValueError, TypeError):
        return 10  # Default fallback

    if freq is None:
        return 10  # Default fallback

    # Map frequency to seasonal periods
    # Handle both old and new pandas frequency codes
    freq_map = {
        'D': 7,      # Daily → weekly seasonality
        'W': 52,     # Weekly → annual seasonality
        'MS': 12,    # Month start → annual seasonality
        'M': 12,     # Month end → annual seasonality
        'ME': 12,    # Month end (new pandas) → annual seasonality
        'Q': 4,      # Quarter → annual seasonality
        'QE': 4,     # Quarter end (new pandas) → annual seasonality
        'H': 24,     # Hourly → daily seasonality (deprecated)
        'h': 24,     # Hourly → daily seasonality (new pandas)
        'T': 60,     # Minute → hourly seasonality (deprecated)
        'min': 60,   # Minute → hourly seasonality (new pandas)
        'S': 60,     # Second → minute seasonality
        'Y': 1,      # Yearly → no sub-annual seasonality
        'YE': 1,     # Year end (new pandas) → no sub-annual seasonality
    }

    # Extract base frequency code (first character or full code)
    # Handle multi-character codes like 'MS', 'QE', etc.
    for code in ['MS', 'ME', 'QE', 'min', 'YE']:
        if freq.startswith(code):
            return freq_map.get(code, 10)

    # Single character codes
    return freq_map.get(freq[0] if freq else None, 10)  # Default to 10


def create_block_bootstrap(
    n_blocks: Optional[int] = None,
    n_resamplings: int = 10,
    overlapping: bool = False,
    random_state: Optional[int] = None
) -> "BlockBootstrap":
    """
    Create block bootstrap configuration for time series.

    Block bootstrap preserves temporal structure by resampling blocks
    of consecutive observations rather than individual points.

    Parameters
    ----------
    n_blocks : int, optional
        Number of blocks (default: 10, or seasonal period if detected)
    n_resamplings : int, default=10
        Number of bootstrap samples
    overlapping : bool, default=False
        Whether blocks can overlap (False avoids data leakage)
    random_state : int, optional
        Random seed

    Returns
    -------
    BlockBootstrap
        Configured bootstrap sampler
    """
    from mapie.subsample import BlockBootstrap

    return BlockBootstrap(
        n_resamplings=n_resamplings,
        n_blocks=n_blocks or 10,
        overlapping=overlapping,
        random_state=random_state
    )
