"""Recipe steps for py-recipes"""

from py_recipes.steps.remove import StepRm, StepSelect
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
from py_recipes.steps.filter_supervised import (
    step_filter_anova,
    step_filter_rf_importance,
    step_filter_mutual_info,
    step_filter_roc_auc,
    step_filter_chisq,
    step_select_shap,
    step_select_permutation,
    StepFilterAnova,
    StepFilterRfImportance,
    StepFilterMutualInfo,
    StepFilterRocAuc,
    StepFilterChisq,
    StepSelectShap,
    StepSelectPermutation,
)
from py_recipes.steps.splitwise import (
    StepSplitwise,
)
from py_recipes.steps.feature_extraction import (
    StepSafe,
    StepSafeV2,
)
from py_recipes.steps.interaction_detection import (
    StepEIX,
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
    # Column selection steps
    "StepRm",
    "StepSelect",
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
    # Supervised filter steps
    "StepFilterAnova",
    "StepFilterRfImportance",
    "StepFilterMutualInfo",
    "StepFilterRocAuc",
    "StepFilterChisq",
    "StepSelectShap",
    "StepSelectPermutation",
    # Adaptive transformation steps
    "StepSplitwise",
    # Feature extraction steps
    "StepSafe",
    "StepSafeV2",
    "StepEIX",
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
