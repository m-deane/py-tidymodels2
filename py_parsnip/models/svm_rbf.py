"""
Support Vector Machine (RBF kernel) model specification

Supports multiple engines:
- sklearn: SVR with RBF kernel

Parameters (tidymodels naming):
- cost: Cost of constraint violation (regularization parameter)
- rbf_sigma: RBF kernel coefficient (gamma)
- margin: Epsilon in the epsilon-SVR model
"""

from typing import Optional
from py_parsnip.model_spec import ModelSpec


def svm_rbf(
    cost: Optional[float] = None,
    rbf_sigma: Optional[float] = None,
    margin: Optional[float] = None,
    engine: str = "sklearn",
) -> ModelSpec:
    """
    Create a support vector machine model specification with RBF kernel.

    Args:
        cost: Regularization parameter (C)
            - For sklearn: maps to 'C'
            - Controls trade-off between smooth decision boundary and classifying training points
            - Higher values = less regularization (fit training data more closely)
            - Default: 1.0
        rbf_sigma: RBF kernel coefficient
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
        ModelSpec for SVM with RBF kernel

    Examples:
        >>> # Default SVM-RBF
        >>> spec = svm_rbf()

        >>> # SVM with higher cost (less regularization)
        >>> spec = svm_rbf(cost=10.0)

        >>> # SVM with specific gamma
        >>> spec = svm_rbf(rbf_sigma=0.1)

        >>> # SVM with custom epsilon tube
        >>> spec = svm_rbf(margin=0.2)

        >>> # Fully customized SVM-RBF
        >>> spec = svm_rbf(cost=5.0, rbf_sigma=0.05, margin=0.15)
    """
    # Build args dict (only include non-None values)
    args = {}
    if cost is not None:
        args["cost"] = cost
    if rbf_sigma is not None:
        args["rbf_sigma"] = rbf_sigma
    if margin is not None:
        args["margin"] = margin

    return ModelSpec(
        model_type="svm_rbf",
        engine=engine,
        mode="regression",
        args=args,
    )
