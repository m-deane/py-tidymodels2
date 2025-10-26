"""
Random Forest model specification

Supports multiple engines:
- sklearn: RandomForestRegressor, RandomForestClassifier

Parameters (tidymodels naming):
- mtry: Number of variables randomly sampled as candidates at each split
- trees: Number of trees in the forest
- min_n: Minimum number of data points in a node required to split
"""

from typing import Optional
from py_parsnip.model_spec import ModelSpec


def rand_forest(
    mtry: Optional[int] = None,
    trees: Optional[int] = None,
    min_n: Optional[int] = None,
    engine: str = "sklearn",
) -> ModelSpec:
    """
    Create a random forest model specification.

    Args:
        mtry: Number of variables to sample at each split
            - For sklearn: maps to 'max_features'
            - Default: sqrt(n_features) for classification, n_features/3 for regression
        trees: Number of trees in the forest
            - For sklearn: maps to 'n_estimators'
            - Default: 500
        min_n: Minimum number of data points in a node to split
            - For sklearn: maps to 'min_samples_split'
            - Default: 2
        engine: Computational engine to use (default "sklearn")

    Returns:
        ModelSpec for random forest

    Examples:
        >>> # Default random forest (mode must be set before fitting)
        >>> spec = rand_forest()
        >>> spec = spec.set_mode("regression")

        >>> # Custom parameters
        >>> spec = rand_forest(mtry=5, trees=1000, min_n=10)
        >>> spec = spec.set_mode("classification")

        >>> # For regression
        >>> spec = rand_forest(trees=500).set_mode("regression")

        >>> # For classification
        >>> spec = rand_forest(trees=500).set_mode("classification")
    """
    # Build args dict (only include non-None values)
    args = {}
    if mtry is not None:
        args["mtry"] = mtry
    if trees is not None:
        args["trees"] = trees
    if min_n is not None:
        args["min_n"] = min_n

    return ModelSpec(
        model_type="rand_forest",
        engine=engine,
        mode="unknown",  # Must be set via set_mode()
        args=args,
    )
