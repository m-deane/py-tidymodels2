"""
Model signature inference and validation for MLflow models.
"""

import pandas as pd
from typing import Optional, Union
from mlflow.models.signature import infer_signature, ModelSignature


def infer_model_signature(
    model_input: pd.DataFrame,
    model_output: Union[pd.DataFrame, pd.Series],
) -> ModelSignature:
    """
    Infer MLflow model signature from example input/output.

    Args:
        model_input: Example input DataFrame
        model_output: Example output DataFrame or Series

    Returns:
        ModelSignature object
    """
    return infer_signature(model_input, model_output)


def get_input_example(data: pd.DataFrame, n_rows: int = 5) -> pd.DataFrame:
    """
    Get input example for signature inference.

    Args:
        data: Input DataFrame
        n_rows: Number of rows to include in example

    Returns:
        DataFrame with first n_rows
    """
    return data.head(n_rows)


def validate_signature(
    signature: Optional[ModelSignature],
    input_data: pd.DataFrame
) -> None:
    """
    Validate that input data conforms to model signature.

    Args:
        signature: Model signature (or None to skip validation)
        input_data: Input data to validate

    Raises:
        ValueError: If input doesn't match signature
    """
    if signature is None:
        return

    # Get expected columns from signature
    expected_cols = [inp.name for inp in signature.inputs]
    actual_cols = list(input_data.columns)

    # Check for missing columns
    missing = set(expected_cols) - set(actual_cols)
    if missing:
        raise ValueError(
            f"Input data missing required columns: {missing}. "
            f"Expected: {expected_cols}"
        )

    # Check for extra columns (warning only)
    extra = set(actual_cols) - set(expected_cols)
    if extra:
        import warnings
        warnings.warn(
            f"Input data contains extra columns not in signature: {extra}",
            UserWarning
        )
