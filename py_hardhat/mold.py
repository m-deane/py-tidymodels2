"""
mold(): Convert formula + data → model-ready format

The mold() function is the core of hardhat's preprocessing. It:
1. Parses the formula using patsy
2. Creates design matrices for outcomes and predictors
3. Captures metadata in a Blueprint
4. Returns MoldedData ready for model fitting

This ensures consistent data structure and prevents common pitfalls like:
- Misaligned columns between train and test
- Unseen factor levels in test data
- Type mismatches
"""

from typing import Optional, Dict, Any, List
import pandas as pd
import patsy
from patsy import dmatrices, dmatrix

from py_hardhat.blueprint import Blueprint, MoldedData


def mold(
    formula: str,
    data: pd.DataFrame,
    intercept: bool = True,
    indicators: str = "traditional",
) -> MoldedData:
    """
    Convert formula + data into model-ready format with Blueprint.

    Args:
        formula: R-style formula string (e.g., "y ~ x1 + x2")
        data: pandas DataFrame containing the variables
        intercept: Whether to include intercept (default True)
        indicators: Categorical encoding strategy:
            - "traditional": One-hot encoding with reference category dropped
            - "none": Keep categorical as-is (for tree-based models)

    Returns:
        MoldedData containing:
            - outcomes: DataFrame with outcome variable(s)
            - predictors: DataFrame with predictor variables (design matrix)
            - blueprint: Blueprint for applying same preprocessing to new data
            - extras: Dict with additional metadata

    Example:
        >>> data = pd.DataFrame({
        ...     "sales": [100, 200, 300],
        ...     "price": [10, 20, 30],
        ...     "category": ["A", "B", "A"]
        ... })
        >>> molded = mold("sales ~ price + category", data)
        >>> molded.predictors.columns
        Index(['Intercept', 'price', 'category[B]'], dtype='object')

    Raises:
        ValueError: If formula is invalid or references missing columns
    """

    # Parse formula and create design matrices using patsy
    try:
        # Create design matrices and capture design info
        y_mat, X_mat = dmatrices(
            formula,
            data,
            return_type="dataframe",
            NA_action="raise",  # Fail on missing values - user should handle this
        )

        # Extract design info from the matrices for later use in forge()
        predictor_design_info = X_mat.design_info
        outcome_design_info = y_mat.design_info

    except Exception as e:
        raise ValueError(f"Failed to parse formula '{formula}': {str(e)}") from e

    # Handle intercept option
    if not intercept and "Intercept" in X_mat.columns:
        X_mat = X_mat.drop(columns=["Intercept"])

    # Handle indicators option
    if indicators == "none":
        # For tree-based models, we might want to keep categoricals as-is
        # This is a simplified version - full implementation would need more work
        pass  # TODO: Implement proper categorical preservation

    # Extract roles from formula
    outcome_cols = list(y_mat.columns)
    predictor_cols = list(X_mat.columns)

    roles = {
        "outcome": outcome_cols,
        "predictor": predictor_cols,
    }

    # Extract factor levels for categorical variables
    factor_levels = _extract_factor_levels(data, X_mat)

    # Extract column types
    ptypes = {col: str(data[col].dtype) for col in data.columns}

    # Create Blueprint
    blueprint = Blueprint(
        formula=formula,
        roles=roles,
        factor_levels=factor_levels,
        column_order=predictor_cols,
        ptypes=ptypes,
        intercept=intercept,
        indicators=indicators,
        design_info=predictor_design_info,
        outcome_design_info=outcome_design_info,
    )

    # Create MoldedData
    molded = MoldedData(
        outcomes=y_mat,
        predictors=X_mat,
        blueprint=blueprint,
        extras={},
    )

    return molded


def _extract_factor_levels(
    data: pd.DataFrame, design_matrix: pd.DataFrame
) -> Dict[str, List[Any]]:
    """
    Extract categorical variable levels from design matrix.

    This captures the factor levels used in one-hot encoding so we can
    enforce them in forge() to prevent new levels in test data.

    Args:
        data: Original data DataFrame
        design_matrix: Design matrix created by patsy

    Returns:
        Dict mapping categorical column names to their levels
    """
    factor_levels = {}

    # Identify categorical columns
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        if col in data.columns:
            # Get unique levels, sorted for consistency
            levels = sorted(data[col].unique())
            factor_levels[col] = levels

    return factor_levels
