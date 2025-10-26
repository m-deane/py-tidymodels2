"""
py-parsnip: Unified model specification interface

This package provides:
- ModelSpec: Immutable model specification
- ModelFit: Fitted model container
- Engine registry: Pluggable backend system (sklearn, statsmodels, etc.)
- Standardized model interface across all engines
- Parameter translation from tidymodels → engine-specific

Key functions:
- linear_reg(), logistic_reg(), rand_forest(), etc.
- set_engine(), set_mode()
- fit(), predict()
"""

from py_parsnip.model_spec import ModelSpec, ModelFit
from py_parsnip.models.linear_reg import linear_reg
from py_parsnip.models.prophet_reg import prophet_reg
from py_parsnip.models.arima_reg import arima_reg
from py_parsnip.models.rand_forest import rand_forest

# Import engines to trigger registration
import py_parsnip.engines  # noqa: F401

__all__ = [
    "ModelSpec",
    "ModelFit",
    "linear_reg",
    "prophet_reg",
    "arima_reg",
    "rand_forest",
]
__version__ = "0.1.0"
