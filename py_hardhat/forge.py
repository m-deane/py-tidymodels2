"""
forge(): Apply Blueprint to new data for prediction

The forge() function applies a Blueprint created by mold() to new data.
It ensures that test/prediction data has exactly the same structure as
training data by:
1. Enforcing factor levels (error on unseen categories)
2. Aligning columns in the same order
3. Applying the same formula transformations
4. Validating data types

This prevents common errors like:
- New factor levels appearing in test data
- Columns in different order
- Missing columns
- Type mismatches
"""

from typing import Optional
import pandas as pd
import patsy
from patsy import dmatrix, build_design_matrices

from py_hardhat.blueprint import Blueprint, MoldedData


def forge(
    new_data: pd.DataFrame,
    blueprint: Blueprint,
    outcomes: bool = False,
) -> MoldedData:
    """
    Apply Blueprint to new data for prediction.

    Args:
        new_data: pandas DataFrame with new data to process
        blueprint: Blueprint created by mold() during training
        outcomes: Whether to expect outcome variables in new_data (default False)

    Returns:
        MoldedData containing:
            - outcomes: DataFrame with outcome variable(s) if outcomes=True, else None
            - predictors: DataFrame with predictor variables (design matrix)
            - blueprint: Original blueprint (unchanged)
            - extras: Dict with additional metadata

    Example:
        >>> # Training
        >>> train = pd.DataFrame({
        ...     "sales": [100, 200, 300],
        ...     "price": [10, 20, 30],
        ...     "category": ["A", "B", "A"]
        ... })
        >>> molded = mold("sales ~ price + category", train)
        >>>
        >>> # Prediction
        >>> test = pd.DataFrame({
        ...     "price": [15, 25],
        ...     "category": ["A", "B"]
        ... })
        >>> forged = forge(test, molded.blueprint)
        >>> forged.predictors.columns
        Index(['Intercept', 'price', 'category[B]'], dtype='object')

    Raises:
        ValueError: If new_data has unseen factor levels or missing columns
    """

    # Validate factor levels (no new levels allowed)
    _validate_factor_levels(new_data, blueprint)

    # Validate required columns are present
    _validate_columns(new_data, blueprint)

    # Create design matrix using stored design info for consistency
    try:
        # Use patsy's build_design_matrices with stored design_info
        # This ensures identical structure even if some categorical levels are missing
        if outcomes:
            # Has outcome variable - build both outcome and predictor matrices
            (y_mat,) = build_design_matrices(
                [blueprint.outcome_design_info], new_data, return_type="dataframe"
            )
            (X_mat,) = build_design_matrices(
                [blueprint.design_info], new_data, return_type="dataframe"
            )
            outcomes_df = y_mat
        else:
            # No outcome variable - build predictor matrix only
            (X_mat,) = build_design_matrices(
                [blueprint.design_info], new_data, return_type="dataframe"
            )
            outcomes_df = None

    except Exception as e:
        raise ValueError(
            f"Failed to apply blueprint to new data: {str(e)}"
        ) from e

    # Handle intercept option
    if not blueprint.intercept and "Intercept" in X_mat.columns:
        X_mat = X_mat.drop(columns=["Intercept"])

    # Ensure columns are in the same order as training
    X_mat = _align_columns(X_mat, blueprint)

    # Create MoldedData
    forged = MoldedData(
        outcomes=outcomes_df,
        predictors=X_mat,
        blueprint=blueprint,
        extras={},
    )

    return forged


def _validate_factor_levels(data: pd.DataFrame, blueprint: Blueprint) -> None:
    """
    Validate that categorical variables don't have new levels.

    Raises:
        ValueError: If new levels are found
    """
    for col, expected_levels in blueprint.factor_levels.items():
        if col in data.columns:
            actual_levels = set(data[col].unique())
            expected_levels_set = set(expected_levels)

            # Check for new levels
            new_levels = actual_levels - expected_levels_set
            if new_levels:
                raise ValueError(
                    f"New factor levels found in '{col}': {new_levels}. "
                    f"Expected levels: {expected_levels_set}"
                )


def _validate_columns(data: pd.DataFrame, blueprint: Blueprint) -> None:
    """
    Validate that required columns are present in new_data.

    Raises:
        ValueError: If required columns are missing
    """
    # Get all columns referenced in the formula (excluding outcome)
    required_cols = set()
    for role, cols in blueprint.roles.items():
        if role != "outcome":  # Don't require outcome for prediction
            # Extract original column names (before encoding)
            for col in cols:
                # Strip encoding artifacts like [T.B] from patsy
                base_col = col.split("[")[0]
                if base_col != "Intercept":
                    required_cols.add(base_col)

    # Get outcome columns to exclude from factor_levels check
    outcome_cols = set()
    for col in blueprint.roles.get("outcome", []):
        base_col = col.split("[")[0]
        outcome_cols.add(base_col)

    # Also check factor_levels keys (excluding outcome variables)
    for factor_col in blueprint.factor_levels.keys():
        if factor_col not in outcome_cols:
            required_cols.add(factor_col)

    # Remove duplicates and check what's missing
    missing_cols = required_cols - set(data.columns)
    if missing_cols:
        raise ValueError(
            f"Required columns missing from new_data: {missing_cols}. "
            f"Available columns: {list(data.columns)}"
        )


def _extract_predictor_formula(formula: str) -> str:
    """
    Extract predictor side of formula (right side of ~).

    Args:
        formula: Full formula like "y ~ x1 + x2"

    Returns:
        Predictor formula like "x1 + x2"
    """
    parts = formula.split("~")
    if len(parts) != 2:
        raise ValueError(f"Invalid formula format: {formula}")

    predictor_formula = parts[1].strip()
    return predictor_formula


def _align_columns(X_mat: pd.DataFrame, blueprint: Blueprint) -> pd.DataFrame:
    """
    Ensure design matrix columns are in the same order as training.

    Args:
        X_mat: Design matrix from patsy
        blueprint: Blueprint with expected column order

    Returns:
        Reordered DataFrame with columns matching blueprint

    Raises:
        ValueError: If columns don't match blueprint
    """
    expected_cols = blueprint.column_order
    actual_cols = list(X_mat.columns)

    # Check for missing columns - add them with zeros if they're categorical dummy columns
    # This happens when test data doesn't have all categorical levels from training
    missing = set(expected_cols) - set(actual_cols)
    if missing:
        # Add missing columns with zeros (expected for categorical variables)
        for col in missing:
            X_mat[col] = 0.0

    # Check for extra columns
    extra = set(actual_cols) - set(expected_cols)
    if extra:
        raise ValueError(
            f"Design matrix has unexpected columns: {extra}. "
            f"This shouldn't happen - check blueprint consistency."
        )

    # Reorder to match blueprint
    return X_mat[expected_cols]
