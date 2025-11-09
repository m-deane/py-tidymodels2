"""
RuleFit model specification

Supports interpretable rule-based modeling via imodels library.

Parameters (tidymodels naming):
- max_rules: Maximum number of rules to generate
- tree_depth: Maximum depth of trees for rule extraction
- penalty: L1 regularization penalty
- tree_generator: Algorithm for tree generation
"""

from typing import Optional
from py_parsnip.model_spec import ModelSpec


def rule_fit(
    max_rules: Optional[int] = None,
    tree_depth: Optional[int] = None,
    penalty: Optional[float] = None,
    tree_generator: Optional[str] = None,
    engine: str = "imodels",
) -> ModelSpec:
    """
    Create a RuleFit model specification.

    RuleFit is an interpretable rule-based model that combines:
    1. Linear model on original features
    2. Rules extracted from decision trees
    3. L1 regularization for sparsity

    Args:
        max_rules: Maximum number of rules to generate
            - For imodels: maps to 'max_rules'
            - Default: 10
        tree_depth: Maximum depth for tree generation
            - For imodels: maps to 'tree_size'
            - Default: 3
        penalty: L1 regularization penalty
            - For imodels: maps to 'alpha'
            - Default: 0.0 (no regularization)
        tree_generator: Algorithm for tree generation
            - For imodels: maps to 'tree_generator'
            - Default: None (uses default generator)
        engine: Computational engine to use (default "imodels")

    Returns:
        ModelSpec for RuleFit

    Examples:
        >>> # Default RuleFit
        >>> spec = rule_fit()

        >>> # RuleFit with more rules
        >>> spec = rule_fit(max_rules=20)

        >>> # RuleFit with deeper trees
        >>> spec = rule_fit(tree_depth=5)

        >>> # RuleFit with L1 regularization
        >>> spec = rule_fit(penalty=0.01)

        >>> # Fully customized RuleFit
        >>> spec = rule_fit(max_rules=15, tree_depth=4, penalty=0.001)
    """
    # Build args dict (only include non-None values)
    args = {}
    if max_rules is not None:
        args["max_rules"] = max_rules
    if tree_depth is not None:
        args["tree_depth"] = tree_depth
    if penalty is not None:
        args["penalty"] = penalty
    if tree_generator is not None:
        args["tree_generator"] = tree_generator

    return ModelSpec(
        model_type="rule_fit",
        engine=engine,
        mode="unknown",  # Will be set via set_mode()
        args=args,
    )
