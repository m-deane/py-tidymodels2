"""Recipe steps for py-recipes"""

from py_recipes.steps.normalize import StepNormalize, PreparedStepNormalize
from py_recipes.steps.dummy import StepDummy, PreparedStepDummy
from py_recipes.steps.impute import (
    StepImputeMean,
    PreparedStepImputeMean,
    StepImputeMedian,
    PreparedStepImputeMedian,
    StepImputeMode,
    PreparedStepImputeMode,
    StepImputeKnn,
    PreparedStepImputeKnn,
    StepImputeLinear,
    PreparedStepImputeLinear,
)
from py_recipes.steps.mutate import StepMutate, PreparedStepMutate
from py_recipes.steps.timeseries import (
    StepLag,
    PreparedStepLag,
    StepDiff,
    PreparedStepDiff,
    StepPctChange,
    PreparedStepPctChange,
    StepRolling,
    PreparedStepRolling,
    StepDate,
    PreparedStepDate,
)
from py_recipes.steps.feature_selection import (
    StepPCA,
    PreparedStepPCA,
    StepSelectCorr,
    PreparedStepSelectCorr,
)
from py_recipes.steps.transformations import (
    StepLog,
    PreparedStepLog,
    StepSqrt,
    PreparedStepSqrt,
    StepBoxCox,
    PreparedStepBoxCox,
    StepYeoJohnson,
    PreparedStepYeoJohnson,
)
from py_recipes.steps.scaling import (
    StepCenter,
    PreparedStepCenter,
    StepScale,
    PreparedStepScale,
    StepRange,
    PreparedStepRange,
)
from py_recipes.steps.filters import (
    StepZv,
    PreparedStepZv,
    StepNzv,
    PreparedStepNzv,
    StepLinComb,
    PreparedStepLinComb,
    StepFilterMissing,
    PreparedStepFilterMissing,
)
from py_recipes.steps.naomit import (
    StepNaOmit,
    PreparedStepNaOmit,
)
from py_recipes.steps.categorical_extended import (
    StepOther,
    PreparedStepOther,
    StepNovel,
    PreparedStepNovel,
    StepIndicateNa,
    PreparedStepIndicateNa,
    StepInteger,
    PreparedStepInteger,
)
from py_recipes.steps.basis import (
    StepBs,
    PreparedStepBs,
    StepNs,
    PreparedStepNs,
    StepPoly,
    PreparedStepPoly,
    StepHarmonic,
    PreparedStepHarmonic,
)
from py_recipes.steps.interactions import (
    StepInteract,
    PreparedStepInteract,
    StepRatio,
    PreparedStepRatio,
)
from py_recipes.steps.discretization import (
    StepDiscretize,
    PreparedStepDiscretize,
    StepCut,
    PreparedStepCut,
)
from py_recipes.steps.reduction import (
    StepIca,
    PreparedStepIca,
    StepKpca,
    PreparedStepKpca,
    StepPls,
    PreparedStepPls,
)
from py_recipes.steps.timeseries_extended import (
    StepHoliday,
    PreparedStepHoliday,
    StepFourier,
    PreparedStepFourier,
    StepTimeseriesSignature,
    PreparedStepTimeseriesSignature,
    StepLead,
    PreparedStepLead,
    StepEwm,
    PreparedStepEwm,
    StepExpanding,
    PreparedStepExpanding,
)
from py_recipes.steps.feature_selection_advanced import (
    StepVip,
    PreparedStepVip,
    StepBoruta,
    PreparedStepBoruta,
    StepRfe,
    PreparedStepRfe,
)
from py_recipes.steps.financial_oscillators import (
    StepOscillators,
    PreparedStepOscillators,
)

__all__ = [
    # Basic steps
    "StepNormalize",
    "PreparedStepNormalize",
    "StepDummy",
    "PreparedStepDummy",
    "StepMutate",
    "PreparedStepMutate",
    # Imputation steps
    "StepImputeMean",
    "PreparedStepImputeMean",
    "StepImputeMedian",
    "PreparedStepImputeMedian",
    "StepImputeMode",
    "PreparedStepImputeMode",
    "StepImputeKnn",
    "PreparedStepImputeKnn",
    "StepImputeLinear",
    "PreparedStepImputeLinear",
    # Time series steps
    "StepLag",
    "PreparedStepLag",
    "StepDiff",
    "PreparedStepDiff",
    "StepPctChange",
    "PreparedStepPctChange",
    "StepRolling",
    "PreparedStepRolling",
    "StepDate",
    "PreparedStepDate",
    # Feature selection steps
    "StepPCA",
    "PreparedStepPCA",
    "StepSelectCorr",
    "PreparedStepSelectCorr",
    # Transformation steps
    "StepLog",
    "PreparedStepLog",
    "StepSqrt",
    "PreparedStepSqrt",
    "StepBoxCox",
    "PreparedStepBoxCox",
    "StepYeoJohnson",
    "PreparedStepYeoJohnson",
    # Scaling steps
    "StepCenter",
    "PreparedStepCenter",
    "StepScale",
    "PreparedStepScale",
    "StepRange",
    "PreparedStepRange",
    # Filter steps
    "StepZv",
    "PreparedStepZv",
    "StepNzv",
    "PreparedStepNzv",
    "StepLinComb",
    "PreparedStepLinComb",
    "StepFilterMissing",
    "PreparedStepFilterMissing",
    "StepNaOmit",
    "PreparedStepNaOmit",
    # Extended categorical steps
    "StepOther",
    "PreparedStepOther",
    "StepNovel",
    "PreparedStepNovel",
    "StepIndicateNa",
    "PreparedStepIndicateNa",
    "StepInteger",
    "PreparedStepInteger",
    # Basis function steps
    "StepBs",
    "PreparedStepBs",
    "StepNs",
    "PreparedStepNs",
    "StepPoly",
    "PreparedStepPoly",
    "StepHarmonic",
    "PreparedStepHarmonic",
    # Interaction steps
    "StepInteract",
    "PreparedStepInteract",
    "StepRatio",
    "PreparedStepRatio",
    # Discretization steps
    "StepDiscretize",
    "PreparedStepDiscretize",
    "StepCut",
    "PreparedStepCut",
    # Advanced dimensionality reduction steps
    "StepIca",
    "PreparedStepIca",
    "StepKpca",
    "PreparedStepKpca",
    "StepPls",
    "PreparedStepPls",
    # Extended time series steps
    "StepHoliday",
    "PreparedStepHoliday",
    "StepFourier",
    "PreparedStepFourier",
    "StepTimeseriesSignature",
    "PreparedStepTimeseriesSignature",
    "StepLead",
    "PreparedStepLead",
    "StepEwm",
    "PreparedStepEwm",
    "StepExpanding",
    "PreparedStepExpanding",
    # Advanced feature selection steps
    "StepVip",
    "PreparedStepVip",
    "StepBoruta",
    "PreparedStepBoruta",
    "StepRfe",
    "PreparedStepRfe",
    # Financial indicator steps
    "StepOscillators",
    "PreparedStepOscillators",
]
