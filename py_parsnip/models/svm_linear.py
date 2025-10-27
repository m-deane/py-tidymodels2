"""
Support Vector Machine (Linear kernel) model specification

Supports multiple engines:
- sklearn: LinearSVR

Parameters (tidymodels naming):
- cost: Cost of constraint violation (regularization parameter)
- margin: Epsilon in the epsilon-SVR model
"""

from typing import Optional
from py_parsnip.model_spec import ModelSpec


def svm_linear(
    cost: Optional[float] = None,
    margin: Optional[float] = None,
    engine: str = "sklearn",
) -> ModelSpec:
    """
    Create a support vector machine model specification with linear kernel.

    Args:
        cost: Regularization parameter (C)
            - For sklearn: maps to 'C'
            - Controls trade-off between smooth decision boundary and classifying training points
            - Higher values = less regularization (fit training data more closely)
            - Default: 1.0
        margin: Epsilon parameter in epsilon-SVR
            - For sklearn: maps to 'epsilon'
            - Specifies epsilon-tube within which no penalty is associated
            - Default: 0.0
        engine: Computational engine to use (default "sklearn")

    Returns:
        ModelSpec for SVM with linear kernel

    Examples:
        >>> # Default linear SVM
        >>> spec = svm_linear()

        >>> # Linear SVM with higher cost (less regularization)
        >>> spec = svm_linear(cost=10.0)

        >>> # Linear SVM with custom epsilon tube
        >>> spec = svm_linear(margin=0.1)

        >>> # Fully customized linear SVM
        >>> spec = svm_linear(cost=5.0, margin=0.05)
    """
    # Build args dict (only include non-None values)
    args = {}
    if cost is not None:
        args["cost"] = cost
    if margin is not None:
        args["margin"] = margin

    return ModelSpec(
        model_type="svm_linear",
        engine=engine,
        mode="regression",
        args=args,
    )
