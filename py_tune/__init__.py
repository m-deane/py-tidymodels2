"""
py-tune: Hyperparameter tuning for py-tidymodels

Provides tidymodels-style hyperparameter optimization with grid search,
racing methods, and advanced optimization algorithms.
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

from .racing import (
    # Racing Control
    RaceControl,
    control_race,

    # Statistical Tests
    filter_parameters_anova,
    filter_parameters_bt,
)

from .tune_race_anova import (
    # Racing Methods
    tune_race_anova,
)

from .tune_race_win_loss import (
    tune_race_win_loss,
)

__all__ = [
    # Parameter Markers
    "tune",
    "TuneParameter",

    # Grid Generation
    "grid_regular",
    "grid_random",

    # Tuning Functions
    "fit_resamples",
    "tune_grid",
    "tune_race_anova",
    "tune_race_win_loss",

    # Results
    "TuneResults",

    # Racing Control
    "RaceControl",
    "control_race",
    "filter_parameters_anova",
    "filter_parameters_bt",

    # Workflow Finalization
    "finalize_workflow",
]

__version__ = "0.2.0"
