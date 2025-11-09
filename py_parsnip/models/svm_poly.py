"""
Support Vector Machine (polynomial kernel) model specification

Supports multiple engines:
- sklearn: SVR/SVC with polynomial kernel

Parameters (tidymodels naming):
- cost: Cost of constraint violation (regularization parameter)
- degree: Degree of the polynomial kernel
- scale_factor: Polynomial kernel coefficient (gamma)
- margin: Epsilon in the epsilon-SVR model
"""

from typing import Optional
from py_parsnip.model_spec import ModelSpec


def svm_poly(
    cost: Optional[float] = None,
    degree: Optional[int] = None,
    scale_factor: Optional[float] = None,
    margin: Optional[float] = None,
    engine: str = "sklearn",
) -> ModelSpec:
    """
    Create a support vector machine model specification with polynomial kernel.

    Args:
        cost: Regularization parameter (C)
            - For sklearn: maps to 'C'
            - Controls trade-off between smooth decision boundary and classifying training points
            - Higher values = less regularization (fit training data more closely)
            - Default: 1.0
        degree: Degree of the polynomial kernel
            - For sklearn: maps to 'degree'
            - Polynomial degree (e.g., 2 for quadratic, 3 for cubic)
            - Default: 3
        scale_factor: Polynomial kernel coefficient
            - For sklearn: maps to 'gamma'
            - Defines influence of single training example
            - Higher values = more influence, tighter fit
            - Default: "scale" (1 / (n_features * X.var()))
        margin: Epsilon parameter in epsilon-SVR
            - For sklearn: maps to 'epsilon'
            - Specifies epsilon-tube within which no penalty is associated
            - Default: 0.1
        engine: Computational engine to use (default "sklearn")

    Returns:
        ModelSpec for SVM with polynomial kernel

    Examples:
        >>> # Default SVM-Poly (cubic)
        >>> spec = svm_poly()

        >>> # Quadratic SVM
        >>> spec = svm_poly(degree=2)

        >>> # SVM with higher cost (less regularization)
        >>> spec = svm_poly(cost=10.0, degree=3)

        >>> # SVM with specific gamma
        >>> spec = svm_poly(scale_factor=0.1, degree=4)

        >>> # SVM with custom epsilon tube
        >>> spec = svm_poly(margin=0.2, degree=2)

        >>> # Fully customized SVM-Poly
        >>> spec = svm_poly(cost=5.0, degree=3, scale_factor=0.05, margin=0.15)
    """
    # Build args dict (only include non-None values)
    args = {}
    if cost is not None:
        args["cost"] = cost
    if degree is not None:
        args["degree"] = degree
    if scale_factor is not None:
        args["scale_factor"] = scale_factor
    if margin is not None:
        args["margin"] = margin

    return ModelSpec(
        model_type="svm_poly",
        engine=engine,
        mode="regression",
        args=args,
    )
