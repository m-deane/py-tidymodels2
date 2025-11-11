"""
py-recipes: Feature engineering and preprocessing for py-tidymodels

Provides tools for specifying preprocessing pipelines that:
- Are fitted on training data (prep)
- Can be applied to new data (bake)
- Prevent data leakage
- Integrate with py-workflows

Main Components:
    recipe(): Create a new recipe
    Recipe: Recipe specification
    PreparedRecipe: Fitted recipe

Recipe Steps:
    step_normalize(): Normalize numeric columns
    step_dummy(): Create dummy variables
    step_impute_mean(): Impute missing values with mean
    step_impute_median(): Impute missing values with median
    step_mutate(): Custom transformations

Example:
    >>> from py_recipes import recipe
    >>> from py_workflows import workflow
    >>> from py_parsnip import linear_reg
    >>>
    >>> # Create recipe
    >>> rec = (
    ...     recipe()
    ...     .step_impute_mean()
    ...     .step_normalize()
    ...     .step_dummy(["category"])
    ... )
    >>>
    >>> # Use in workflow
    >>> wf = (
    ...     workflow()
    ...     .add_recipe(rec)
    ...     .add_model(linear_reg().set_engine("sklearn"))
    ... )
    >>> wf_fit = wf.fit(train_data)
    >>> predictions = wf_fit.predict(test_data)
"""

from py_recipes.recipe import Recipe, PreparedRecipe, recipe

# Import steps for convenience
from py_recipes.steps import (
    StepNormalize,
    StepDummy,
    StepImputeMean,
    StepImputeMedian,
    StepMutate,
)

# Import time series steps
from py_recipes.steps.timeseries import (
    StepLag,
    StepDate,
)

# Import supervised filter steps
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

# Import feature extraction steps
from py_recipes.steps.feature_extraction import (
    step_safe_v2,
    StepSafeV2,
)

# Import selectors
from py_recipes.selectors import (
    # Type selectors
    all_numeric,
    all_nominal,
    all_integer,
    all_float,
    all_string,
    all_datetime,
    # Predictor/Outcome selectors
    all_predictors,
    all_outcomes,
    all_numeric_predictors,
    all_nominal_predictors,
    # Pattern selectors
    starts_with,
    ends_with,
    contains,
    matches,
    # Utility selectors
    everything,
    one_of,
    none_of,
    where,
    # Combination selectors
    union,
    intersection,
    difference,
    # Role selector
    has_role as has_role_selector,
    # Helper
    resolve_selector,
)

__all__ = [
    "recipe",
    "Recipe",
    "PreparedRecipe",
    "StepNormalize",
    "StepDummy",
    "StepImputeMean",
    "StepImputeMedian",
    "StepMutate",
    "StepLag",
    "StepDate",
    # Supervised filter steps
    "step_filter_anova",
    "step_filter_rf_importance",
    "step_filter_mutual_info",
    "step_filter_roc_auc",
    "step_filter_chisq",
    "StepFilterAnova",
    "StepFilterRfImportance",
    "StepFilterMutualInfo",
    "StepFilterRocAuc",
    "StepFilterChisq",
    # Feature extraction steps
    "step_safe_v2",
    "StepSafeV2",
    # Selectors
    "all_numeric",
    "all_nominal",
    "all_integer",
    "all_float",
    "all_string",
    "all_datetime",
    "all_predictors",
    "all_outcomes",
    "all_numeric_predictors",
    "all_nominal_predictors",
    "starts_with",
    "ends_with",
    "contains",
    "matches",
    "everything",
    "one_of",
    "none_of",
    "where",
    "union",
    "intersection",
    "difference",
    "has_role_selector",
    "resolve_selector",
]

__version__ = "0.1.0"
