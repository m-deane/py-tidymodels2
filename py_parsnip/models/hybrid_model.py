"""
Generic Hybrid Model specification

This is a flexible hybrid model that can combine any two models.
Unlike specific hybrids (arima_boost, prophet_boost), this allows arbitrary
model combinations with different strategies.

Strategies:
1. "residual": Train model2 on residuals from model1
   - Final prediction = model1_pred + model2_pred
2. "sequential": Train models on different time periods
   - Use model1 for early period, model2 for later period
3. "weighted": Weighted combination of both predictions
   - Final prediction = weight1 * model1_pred + weight2 * model2_pred
4. "custom_data": Train models on different/overlapping datasets
   - Pass dict with 'model1' and 'model2' keys to fit()
   - Each model trains on its own dataset (can overlap)
   - Final prediction blends both models

Use Cases:
- Combine linear + non-linear models
- Combine different time series models
- Train models on different periods (regime changes)
- Train models on overlapping/different date ranges
- Create custom ensembles
"""

from typing import Optional, Union, Literal
from py_parsnip.model_spec import ModelSpec


def hybrid_model(
    model1: Optional[ModelSpec] = None,
    model2: Optional[ModelSpec] = None,
    strategy: Literal["residual", "sequential", "weighted", "custom_data"] = "residual",
    weight1: float = 0.5,
    weight2: float = 0.5,
    split_point: Optional[Union[int, float, str]] = None,
    blend_predictions: str = "weighted",
    engine: str = "generic_hybrid",
) -> ModelSpec:
    """
    Create a generic hybrid model combining two arbitrary models.

    This provides flexibility to combine any two models with different
    strategies for how they interact.

    Strategies:
    -----------
    1. **residual** (default):
       - Train model1 on y
       - Train model2 on residuals from model1
       - Prediction = model1_pred + model2_pred
       - Use case: Capture what model1 misses

    2. **sequential**:
       - Train model1 on early period (before split_point)
       - Train model2 on later period (after split_point)
       - Use model1 predictions before split, model2 after
       - Use case: Handle regime changes, structural breaks

    3. **weighted**:
       - Train both models on same data
       - Prediction = weight1 * model1_pred + weight2 * model2_pred
       - Use case: Simple ensemble, reduce variance

    4. **custom_data**:
       - Train model1 on data['model1']
       - Train model2 on data['model2']
       - Datasets can have different/overlapping date ranges
       - Prediction blends both models based on blend_predictions
       - Use case: Different training periods, adaptive learning

    Args:
        model1: First model specification (e.g., linear_reg())
        model2: Second model specification (e.g., rand_forest())
        strategy: How to combine models ("residual", "sequential", "weighted", "custom_data")
        weight1: Weight for model1 in weighted/custom_data strategy (default 0.5)
        weight2: Weight for model2 in weighted/custom_data strategy (default 0.5)
        split_point: For sequential strategy - where to split periods
            - int: row index
            - float: proportion (0.0 to 1.0)
            - str: date string (e.g., "2020-06-01")
        blend_predictions: For custom_data strategy - how to combine predictions
            - "weighted": weight1 * pred1 + weight2 * pred2 (default)
            - "avg": simple average (0.5 * pred1 + 0.5 * pred2)
            - "model1": use only model1 predictions
            - "model2": use only model2 predictions
        engine: Computational engine (default "generic_hybrid")

    Returns:
        ModelSpec for hybrid model

    Examples:
        >>> # Residual strategy: Linear + Random Forest on residuals
        >>> spec = hybrid_model(
        ...     model1=linear_reg(),
        ...     model2=rand_forest(),
        ...     strategy="residual"
        ... )

        >>> # Sequential strategy: Different models for different periods
        >>> spec = hybrid_model(
        ...     model1=linear_reg(),
        ...     model2=rand_forest(),
        ...     strategy="sequential",
        ...     split_point="2020-06-01"  # Regime change date
        ... )

        >>> # Weighted strategy: Simple ensemble
        >>> spec = hybrid_model(
        ...     model1=linear_reg(),
        ...     model2=svm_rbf(),
        ...     strategy="weighted",
        ...     weight1=0.6,
        ...     weight2=0.4
        ... )

        >>> # Custom data strategy: Different/overlapping training periods
        >>> spec = hybrid_model(
        ...     model1=linear_reg(),
        ...     model2=rand_forest(),
        ...     strategy="custom_data",
        ...     blend_predictions="weighted",
        ...     weight1=0.4,  # Less weight on older model
        ...     weight2=0.6   # More weight on recent model
        ... )
        >>> # Fit with separate datasets (can overlap)
        >>> early_data = df[df['date'] < '2020-07-01']
        >>> later_data = df[df['date'] >= '2020-04-01']  # 3 months overlap
        >>> fit = spec.fit({'model1': early_data, 'model2': later_data}, 'y ~ x')

        >>> # Standard fit to data (non-custom_data strategies)
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2020-01-01', periods=100),
        ...     'y': range(100),
        ...     'x': range(100)
        ... })
        >>> fit = spec.fit(df, 'y ~ x')

    Notes:
        - Model specs are stored in args, actual fitting happens in engine
        - For sequential strategy, split_point is required
        - For custom_data strategy, pass dict to fit(): {'model1': df1, 'model2': df2}
        - Weights in weighted/custom_data strategy should sum to 1.0 for interpretability
        - Both models must be compatible with the same formula/data

    References:
        Wolpert, D. H. (1992). Stacked generalization. Neural networks, 5(2), 241-259.
    """
    # Validate strategy
    valid_strategies = ["residual", "sequential", "weighted", "custom_data"]
    if strategy not in valid_strategies:
        raise ValueError(f"strategy must be one of {valid_strategies}, got '{strategy}'")

    # Validate models
    if model1 is None or model2 is None:
        raise ValueError("Both model1 and model2 are required")

    # Validate sequential strategy
    if strategy == "sequential" and split_point is None:
        raise ValueError("split_point is required for sequential strategy")

    # Validate weighted and custom_data strategies
    if strategy in ["weighted", "custom_data"]:
        if not (0 <= weight1 <= 1 and 0 <= weight2 <= 1):
            raise ValueError("Weights must be between 0 and 1")
        # Warn if weights don't sum to 1
        if abs(weight1 + weight2 - 1.0) > 0.01:
            import warnings
            warnings.warn(
                f"Weights sum to {weight1 + weight2:.2f}, not 1.0. "
                "Consider normalizing for interpretability."
            )

    # Validate blend_predictions for custom_data
    if strategy == "custom_data":
        valid_blend_types = ["weighted", "avg", "model1", "model2"]
        if blend_predictions not in valid_blend_types:
            raise ValueError(
                f"blend_predictions must be one of {valid_blend_types}, got '{blend_predictions}'"
            )

    # Build args dict
    args = {
        "model1_spec": model1,
        "model2_spec": model2,
        "strategy": strategy,
        "weight1": weight1,
        "weight2": weight2,
        "split_point": split_point,
        "blend_predictions": blend_predictions,
    }

    return ModelSpec(
        model_type="hybrid_model",
        engine=engine,
        mode="regression",  # Currently only regression supported
        args=args,
    )
