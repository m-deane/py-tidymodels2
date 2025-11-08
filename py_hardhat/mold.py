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
import re

from py_hardhat.blueprint import Blueprint, MoldedData


def _expand_dot_formula(formula: str, data: pd.DataFrame) -> str:
    """
    Expand '.' wildcard in formula to explicit column names.

    The '.' in R-style formulas means "all columns except the outcome".
    Patsy 1.0.1 doesn't support this syntax and raises a SyntaxError when
    it tries to parse '.' as Python code. This function preprocesses the
    formula to expand '.' into explicit column names before passing to patsy.

    Args:
        formula: Formula string (e.g., "y ~ .")
        data: DataFrame with columns to expand

    Returns:
        Expanded formula string (e.g., "y ~ x1 + x2 + x3")

    Example:
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'y': [5, 6]})
        >>> _expand_dot_formula('y ~ .', df)
        'y ~ A + B'

    Notes:
        - Column names that aren't valid Python identifiers are wrapped with Q()
        - Handles formulas like "y ~ ." and "y ~ . - x1"
        - Does not expand '.' if it's not present in the RHS
    """
    # Check if formula contains the tilde separator
    if '~' not in formula:
        return formula

    # Split into LHS (outcome) and RHS (predictors)
    lhs, rhs = formula.split('~', 1)
    lhs = lhs.strip()
    rhs = rhs.strip()

    # If RHS doesn't contain '.', return as-is
    if '.' not in rhs:
        return formula

    # Extract outcome column name(s) from LHS
    # Handle simple cases like "y" or complex cases like "y1 + y2"
    # For wrapped names like 'Q("my var")', extract the actual column name
    outcome_cols = []
    for part in lhs.split('+'):
        part = part.strip()
        # Check if it's a Q() wrapped name
        q_match = re.match(r'Q\(["\'](.+?)["\']\)', part)
        if q_match:
            outcome_cols.append(q_match.group(1))
        else:
            outcome_cols.append(part)

    # Get all columns except outcomes
    predictor_cols = [col for col in data.columns if col not in outcome_cols]

    # Quote column names that aren't valid Python identifiers
    def quote_if_needed(name: str) -> str:
        """Wrap column name in Q() if it's not a valid Python identifier"""
        if name.isidentifier():
            return name
        else:
            # Escape any quotes in the name
            escaped_name = name.replace('"', '\\"')
            return f'Q("{escaped_name}")'

    quoted_predictors = [quote_if_needed(col) for col in predictor_cols]

    # Handle different cases of '.' usage
    if rhs.strip() == '.':
        # Simple case: "y ~ ."
        expanded_rhs = ' + '.join(quoted_predictors)
    elif rhs.strip().startswith('.'):
        # Cases like ". - x1" or ". + I(x^2)"
        # Replace leading '.' with the column list
        rest_of_formula = rhs[1:].strip()
        if quoted_predictors:
            expanded_rhs = ' + '.join(quoted_predictors) + (' ' + rest_of_formula if rest_of_formula else '')
        else:
            expanded_rhs = rest_of_formula.lstrip('+').strip()
    else:
        # Complex case: "x1 + . + log(x2)" (rare but possible)
        # Simple replacement
        expanded_rhs = rhs.replace('.', ' + '.join(quoted_predictors) if quoted_predictors else '')

    # Clean up any double operators or extra spaces
    expanded_rhs = re.sub(r'\+\s*\+', '+', expanded_rhs)
    expanded_rhs = re.sub(r'\s+', ' ', expanded_rhs).strip()

    return f'{lhs} ~ {expanded_rhs}'


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

    # Early validation: Check if formula references columns with spaces
    # This catches cases where outcome column has spaces before patsy parsing
    import re

    # Extract potential column names from raw formula (before expansion)
    # Look for multi-word tokens that might be column names
    raw_tokens = re.findall(r'[\w\s]+', formula)
    raw_cols_with_spaces = [token.strip() for token in raw_tokens
                            if ' ' in token.strip() and token.strip() in data.columns]

    if raw_cols_with_spaces:
        raise ValueError(
            f"Column names used in formula cannot contain spaces. Found {len(raw_cols_with_spaces)} invalid column(s):\n"
            f"  {raw_cols_with_spaces[:5]}\n"  # Show first 5
            f"Please rename these columns before using them in formulas. Example:\n"
            f"  data = data.rename(columns={{'{raw_cols_with_spaces[0]}': '{raw_cols_with_spaces[0].replace(' ', '_')}'}})\n"
            f"Or use: data.columns = data.columns.str.replace(' ', '_')"
        )

    # Patsy doesn't support '.' notation, so we expand it manually
    expanded_formula = _expand_dot_formula(formula, data)

    # Validate only columns referenced in the expanded formula have no spaces
    # Extract column names from expanded formula
    referenced_cols = []

    # Extract from Q() wrapped names: Q("column name")
    q_wrapped = re.findall(r'Q\(["\'](.+?)["\']\)', expanded_formula)
    referenced_cols.extend(q_wrapped)

    # Extract regular Python identifiers (exclude Q to avoid duplicates)
    regular_ids = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expanded_formula)
    # Filter to only actual column names in the data
    referenced_cols.extend([col for col in regular_ids if col in data.columns and col != 'Q'])

    # Remove duplicates and check for spaces
    referenced_cols = list(set(referenced_cols))
    invalid_cols = [col for col in referenced_cols if ' ' in col]

    if invalid_cols:
        raise ValueError(
            f"Column names used in formula cannot contain spaces. Found {len(invalid_cols)} invalid column(s):\n"
            f"  {invalid_cols[:5]}\n"  # Show first 5
            f"Please rename these columns before using them in formulas. Example:\n"
            f"  data = data.rename(columns={{'{invalid_cols[0]}': '{invalid_cols[0].replace(' ', '_')}'}})\n"
            f"Or use: data.columns = data.columns.str.replace(' ', '_')"
        )

    # Parse formula and create design matrices using patsy
    try:
        # Create design matrices and capture design info
        y_mat, X_mat = dmatrices(
            expanded_formula,
            data,
            return_type="dataframe",
            NA_action="raise",  # Fail on missing values - user should handle this
        )

        # Extract design info from the matrices for later use in forge()
        predictor_design_info = X_mat.design_info
        outcome_design_info = y_mat.design_info

    except Exception as e:
        raise ValueError(
            f"Failed to parse formula '{formula}': {str(e)}\n"
            f"(Expanded to: '{expanded_formula}')"
        ) from e

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
