"""
Bagged Tree model specification

Supports sklearn engine with BaggingRegressor/BaggingClassifier
using decision trees as base estimators.

Parameters (tidymodels naming):
- trees: Number of bootstrap samples/trees (default: 25)
- min_n: Minimum number of data points in a node required to split
- cost_complexity: Complexity parameter for pruning (ccp_alpha)
- tree_depth: Maximum tree depth
"""

from typing import Optional
from py_parsnip.model_spec import ModelSpec


def bag_tree(
    trees: Optional[int] = None,
    min_n: Optional[int] = None,
    cost_complexity: Optional[float] = None,
    tree_depth: Optional[int] = None,
    engine: str = "sklearn",
) -> ModelSpec:
    """
    Create a bagged tree model specification.

    Bagging (Bootstrap Aggregating) creates an ensemble of decision trees
    by training each tree on a bootstrap sample of the data. Reduces variance
    and overfitting compared to a single tree.

    Args:
        trees: Number of bootstrap samples/trees to aggregate
            - For sklearn: maps to 'n_estimators'
            - Default: 25
        min_n: Minimum number of data points in a node to split
            - For sklearn: maps to 'min_samples_split' in base estimator
            - Default: 2
        cost_complexity: Complexity parameter for pruning (>= 0)
            - For sklearn: maps to 'ccp_alpha' in base estimator
            - Default: 0.0 (no pruning)
        tree_depth: Maximum tree depth (None means unlimited)
            - For sklearn: maps to 'max_depth' in base estimator
            - Default: None (trees grow until pure leaves)
        engine: Computational engine to use (default "sklearn")

    Returns:
        ModelSpec for bagged trees

    Examples:
        >>> # Default bagged tree (mode must be set before fitting)
        >>> spec = bag_tree()
        >>> spec = spec.set_mode("regression")

        >>> # Custom parameters
        >>> spec = bag_tree(trees=50, min_n=5, tree_depth=10)
        >>> spec = spec.set_mode("classification")

        >>> # For regression with pruning
        >>> spec = bag_tree(trees=100, cost_complexity=0.01).set_mode("regression")

        >>> # For classification with depth limit
        >>> spec = bag_tree(trees=75, tree_depth=15).set_mode("classification")
    """
    # Build args dict (only include non-None values)
    args = {}
    if trees is not None:
        args["trees"] = trees
    if min_n is not None:
        args["min_n"] = min_n
    if cost_complexity is not None:
        args["cost_complexity"] = cost_complexity
    if tree_depth is not None:
        args["tree_depth"] = tree_depth

    return ModelSpec(
        model_type="bag_tree",
        engine=engine,
        mode="unknown",  # Must be set via set_mode()
        args=args,
    )
