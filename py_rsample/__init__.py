"""
py-rsample: Time series resampling and cross-validation

Provides tools for creating train/test splits for time series data with
support for rolling/expanding windows and period parsing.

Main Functions:
    initial_time_split: Create single chronological train/test split
    time_series_cv: Create multiple CV folds with rolling/expanding windows

Main Classes:
    Split: Immutable split with train/test indices
    RSplit: Wrapper providing training()/testing() methods
    TimeSeriesCV: Iterable container of CV splits

Example:
    >>> from py_rsample import initial_time_split, time_series_cv
    >>>
    >>> # Simple train/test split
    >>> split = initial_time_split(data, prop=0.75, date_column="date")
    >>> train = split.training()
    >>> test = split.testing()
    >>>
    >>> # Cross-validation with expanding window
    >>> cv = time_series_cv(
    ...     data=data,
    ...     date_column="date",
    ...     initial="2 years",
    ...     assess="3 months",
    ...     cumulative=True
    ... )
    >>> for fold in cv:
    ...     train = fold.training()
    ...     test = fold.testing()
"""

from py_rsample.split import Split, RSplit, training, testing
from py_rsample.initial_split import initial_time_split
from py_rsample.time_series_cv import time_series_cv, TimeSeriesCV

# Alias for non-time-series data (matches R's rsample API)
initial_split = initial_time_split

__all__ = [
    "Split",
    "RSplit",
    "training",
    "testing",
    "initial_split",
    "initial_time_split",
    "time_series_cv",
    "TimeSeriesCV",
]

__version__ = "0.1.0"
