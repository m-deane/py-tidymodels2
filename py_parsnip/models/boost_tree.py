"""
Boosted Tree model specification

Supports multiple gradient boosting engines:
- xgboost: XGBoost gradient boosting
- lightgbm: LightGBM gradient boosting
- catboost: CatBoost gradient boosting

Parameters (tidymodels naming):
- trees: Number of boosting iterations (n_estimators)
- tree_depth: Maximum tree depth (max_depth)
- learn_rate: Step size shrinkage (learning_rate)
- mtry: Features per split (max_features/colsample_bytree)
- min_n: Minimum samples in leaf (min_child_weight/min_data_in_leaf)
- loss_reduction: Minimum loss reduction (gamma/min_split_gain)
- sample_size: Row sampling fraction (subsample)
- stop_iter: Early stopping rounds (early_stopping_rounds)
"""

from typing import Optional
from py_parsnip.model_spec import ModelSpec


def boost_tree(
    trees: Optional[int] = None,
    tree_depth: Optional[int] = None,
    learn_rate: Optional[float] = None,
    mtry: Optional[int] = None,
    min_n: Optional[int] = None,
    loss_reduction: Optional[float] = None,
    sample_size: Optional[float] = None,
    stop_iter: Optional[int] = None,
    engine: str = "xgboost",
) -> ModelSpec:
    """
    Create a boosted tree model specification.

    Args:
        trees: Number of boosting iterations (default depends on engine)
            - Maps to 'n_estimators' in all engines
        tree_depth: Maximum tree depth (default depends on engine)
            - Maps to 'max_depth' in all engines
        learn_rate: Step size shrinkage for updates (0 to 1)
            - Maps to 'learning_rate' in all engines
        mtry: Number of features to sample per split
            - XGBoost: maps to 'colsample_bytree' (as fraction)
            - LightGBM: maps to 'colsample_bytree' (as fraction)
            - CatBoost: maps to 'max_features' (as integer)
        min_n: Minimum number of samples in leaf
            - XGBoost: maps to 'min_child_weight'
            - LightGBM: maps to 'min_data_in_leaf'
            - CatBoost: maps to 'min_data_in_leaf'
        loss_reduction: Minimum loss reduction required for split
            - XGBoost: maps to 'gamma'
            - LightGBM: maps to 'min_split_gain'
            - CatBoost: not directly supported
        sample_size: Fraction of observations to sample per tree (0 to 1)
            - Maps to 'subsample' in all engines
        stop_iter: Number of iterations for early stopping
            - Maps to 'early_stopping_rounds' in all engines
        engine: Computational engine to use (default "xgboost")

    Returns:
        ModelSpec for boosted tree regression

    Examples:
        >>> # Basic XGBoost model
        >>> spec = boost_tree(trees=100, tree_depth=6)

        >>> # LightGBM with learning rate
        >>> spec = boost_tree(trees=100, learn_rate=0.1, engine="lightgbm")

        >>> # CatBoost with early stopping
        >>> spec = boost_tree(trees=1000, stop_iter=50, engine="catboost")

        >>> # Full parameter specification
        >>> spec = boost_tree(
        ...     trees=100,
        ...     tree_depth=6,
        ...     learn_rate=0.1,
        ...     mtry=5,
        ...     min_n=10,
        ...     loss_reduction=0.01,
        ...     sample_size=0.8,
        ...     stop_iter=10
        ... )
    """
    # Build args dict (only include non-None values)
    args = {}
    if trees is not None:
        args["trees"] = trees
    if tree_depth is not None:
        args["tree_depth"] = tree_depth
    if learn_rate is not None:
        args["learn_rate"] = learn_rate
    if mtry is not None:
        args["mtry"] = mtry
    if min_n is not None:
        args["min_n"] = min_n
    if loss_reduction is not None:
        args["loss_reduction"] = loss_reduction
    if sample_size is not None:
        args["sample_size"] = sample_size
    if stop_iter is not None:
        args["stop_iter"] = stop_iter

    return ModelSpec(
        model_type="boost_tree",
        engine=engine,
        mode="regression",
        args=args,
    )
