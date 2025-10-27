"""
Decision Tree model specification

Supports multiple engines:
- sklearn: DecisionTreeRegressor

Parameters (tidymodels naming):
- tree_depth: Maximum tree depth
- min_n: Minimum number of data points in a node required to split
- cost_complexity: Complexity parameter for pruning (ccp_alpha)
"""

from typing import Optional
from py_parsnip.model_spec import ModelSpec


def decision_tree(
    tree_depth: Optional[int] = None,
    min_n: Optional[int] = None,
    cost_complexity: Optional[float] = None,
    engine: str = "sklearn",
) -> ModelSpec:
    """
    Create a decision tree model specification.

    Args:
        tree_depth: Maximum depth of the tree
            - For sklearn: maps to 'max_depth'
            - Default: None (nodes expanded until leaves are pure)
        min_n: Minimum number of data points in a node to split
            - For sklearn: maps to 'min_samples_split'
            - Default: 2
        cost_complexity: Complexity parameter for pruning
            - For sklearn: maps to 'ccp_alpha'
            - Default: 0.0 (no pruning)
        engine: Computational engine to use (default "sklearn")

    Returns:
        ModelSpec for decision tree

    Examples:
        >>> # Default decision tree
        >>> spec = decision_tree()

        >>> # Constrained tree with max depth
        >>> spec = decision_tree(tree_depth=5)

        >>> # Tree with minimum samples per split
        >>> spec = decision_tree(min_n=10)

        >>> # Tree with pruning
        >>> spec = decision_tree(cost_complexity=0.01)

        >>> # Fully customized tree
        >>> spec = decision_tree(tree_depth=10, min_n=5, cost_complexity=0.001)
    """
    # Build args dict (only include non-None values)
    args = {}
    if tree_depth is not None:
        args["tree_depth"] = tree_depth
    if min_n is not None:
        args["min_n"] = min_n
    if cost_complexity is not None:
        args["cost_complexity"] = cost_complexity

    return ModelSpec(
        model_type="decision_tree",
        engine=engine,
        mode="regression",
        args=args,
    )
