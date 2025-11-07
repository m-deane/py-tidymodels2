"""
Manual Regression model specification

A model where coefficients are manually specified rather than fitted.
Useful for:
- Comparing with pre-existing forecasts
- Benchmarking against known models
- Testing specific coefficient values
- Domain expert knowledge incorporation
"""

from typing import Optional, Dict, Union, List
from py_parsnip.model_spec import ModelSpec


def manual_reg(
    coefficients: Optional[Dict[str, float]] = None,
    intercept: Optional[float] = None,
    engine: str = "parsnip",
) -> ModelSpec:
    """
    Create a manual regression model with user-specified coefficients.

    This model type allows you to manually specify coefficients instead of
    fitting them from data. Useful for:
    - Comparing with external/pre-existing models
    - Testing specific coefficient combinations
    - Incorporating domain expert knowledge
    - Benchmarking against known baselines

    Args:
        coefficients: Dictionary mapping variable names to coefficient values
            Example: {"x1": 2.5, "x2": -1.3, "x3": 0.8}
        intercept: Intercept/constant term (default: 0.0)
        engine: Computational engine (default "parsnip")

    Returns:
        ModelSpec for manual regression

    Examples:
        >>> # Manual model from domain knowledge
        >>> spec = manual_reg(
        ...     coefficients={"temperature": 1.5, "humidity": -0.3},
        ...     intercept=10.0
        ... )
        >>> fit = spec.fit(data, 'sales ~ temperature + humidity')
        >>> predictions = fit.predict(test_data)

        >>> # Compare with pre-existing forecast
        >>> # You know an external model uses these coefficients:
        >>> external_model = manual_reg(
        ...     coefficients={"marketing_spend": 2.1, "seasonality": 0.8},
        ...     intercept=5.0
        ... )
        >>> fit = external_model.fit(train, 'revenue ~ marketing_spend + seasonality')
        >>> outputs, coefficients, stats = fit.extract_outputs()

        >>> # Benchmark against simple baseline
        >>> baseline = manual_reg(
        ...     coefficients={"x": 1.0},  # Simple 1:1 relationship
        ...     intercept=0.0
        ... )

    Notes:
        - The "fit" method validates coefficients match formula variables
        - Predictions are calculated as: y_pred = intercept + sum(coef_i * x_i)
        - extract_outputs() returns standard three-DataFrame format
        - No actual fitting occurs - coefficients are fixed at specified values
        - Useful for comparing py-tidymodels with external forecasting tools

    See Also:
        - null_model(): Baseline models (mean, median, last)
        - naive_reg(): Simple time series baselines
    """
    # Default to empty coefficients if not provided
    if coefficients is None:
        coefficients = {}

    # Default intercept to 0.0 if not provided
    if intercept is None:
        intercept = 0.0

    # Validate coefficients
    if not isinstance(coefficients, dict):
        raise TypeError(
            f"coefficients must be a dict, got {type(coefficients).__name__}"
        )

    # Validate all coefficient values are numeric
    for var, coef in coefficients.items():
        if not isinstance(coef, (int, float)):
            raise TypeError(
                f"Coefficient for '{var}' must be numeric, got {type(coef).__name__}"
            )

    # Validate intercept
    if not isinstance(intercept, (int, float)):
        raise TypeError(
            f"intercept must be numeric, got {type(intercept).__name__}"
        )

    # Build args dict
    args = {
        "coefficients": coefficients,
        "intercept": float(intercept),
    }

    return ModelSpec(
        model_type="manual_reg",
        engine=engine,
        mode="regression",
        args=args,
    )
