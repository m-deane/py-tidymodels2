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
from py_parsnip.models.exp_smoothing import exp_smoothing
from py_parsnip.models.seasonal_reg import seasonal_reg
from py_parsnip.models.rand_forest import rand_forest
from py_parsnip.models.recursive_reg import recursive_reg
from py_parsnip.models.null_model import null_model
from py_parsnip.models.naive_reg import naive_reg
from py_parsnip.models.boost_tree import boost_tree
from py_parsnip.models.arima_boost import arima_boost
from py_parsnip.models.prophet_boost import prophet_boost
from py_parsnip.models.mars import mars
from py_parsnip.models.poisson_reg import poisson_reg
from py_parsnip.models.gen_additive_mod import gen_additive_mod
from py_parsnip.models.decision_tree import decision_tree
from py_parsnip.models.nearest_neighbor import nearest_neighbor
from py_parsnip.models.svm_rbf import svm_rbf
from py_parsnip.models.svm_linear import svm_linear
from py_parsnip.models.mlp import mlp

# Import engines to trigger registration
import py_parsnip.engines  # noqa: F401

__all__ = [
    "ModelSpec",
    "ModelFit",
    "linear_reg",
    "prophet_reg",
    "arima_reg",
    "exp_smoothing",
    "seasonal_reg",
    "rand_forest",
    "recursive_reg",
    "null_model",
    "naive_reg",
    "boost_tree",
    "arima_boost",
    "prophet_boost",
    "mars",
    "poisson_reg",
    "gen_additive_mod",
    "decision_tree",
    "nearest_neighbor",
    "svm_rbf",
    "svm_linear",
    "mlp",
]
__version__ = "0.1.0"
