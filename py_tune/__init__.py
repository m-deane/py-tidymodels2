"""
py-tune: Hyperparameter tuning for py-tidymodels

Provides tidymodels-style hyperparameter optimization with grid search,
random search, and cross-validation evaluation.
"""

from .tune import (
    # Parameter Markers
    tune,
    TuneParameter,

    # Grid Generation
    grid_regular,
    grid_random,

    # Tuning Functions
    fit_resamples,
    tune_grid,

    # Results
    TuneResults,

    # Workflow Finalization
    finalize_workflow,
)

__all__ = [
    "tune",
    "TuneParameter",
    "grid_regular",
    "grid_random",
    "fit_resamples",
    "tune_grid",
    "TuneResults",
    "finalize_workflow",
]

__version__ = "0.1.0"
