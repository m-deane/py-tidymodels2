"""
Panel Regression model specification (Linear Mixed Effects Models)

Panel regression models are used for grouped/hierarchical data where observations
within groups are correlated. Also known as:
- Linear Mixed Effects Models
- Multilevel Models
- Hierarchical Linear Models
- Random Effects Models

Use cases:
- Multi-store sales forecasting (stores are groups)
- Multi-patient clinical trials (patients are groups)
- Multi-region economic data (regions are groups)
- Longitudinal/panel data (subjects measured over time)

Supports:
- Random intercepts (default): Each group has its own baseline level
- Random slopes: Each group has its own slope for a specified variable
- Random intercepts + slopes: Both group-specific intercepts and slopes

Engine:
- statsmodels: MixedLM (Linear Mixed Effects Model)

Parameters:
- random_effects: Type of random effects ("intercept", "slope", "both", None)
- intercept: Whether to fit an intercept term (default True)
- penalty: Regularization penalty (future enhancement)
"""

from typing import Optional
from py_parsnip.model_spec import ModelSpec


def panel_reg(
    random_effects: Optional[str] = "intercept",
    penalty: Optional[float] = None,
    intercept: bool = True,
    engine: str = "statsmodels",
) -> ModelSpec:
    """
    Create a panel regression model specification with random effects.

    Panel regression (mixed linear effects models) are used for data with
    grouped/hierarchical structure where observations within groups are
    correlated. Useful for:
    - Multi-store sales forecasting (stores are groups)
    - Multi-patient clinical trials (patients are groups)
    - Multi-region economic data (regions are groups)
    - Longitudinal/panel data (subjects measured over time)

    Args:
        random_effects: Random effects specification (default: "intercept")
            - "intercept": Random intercepts only (default)
                Each group has its own baseline level
            - "slope": Random slopes (requires slope_var parameter)
                Each group has its own slope for specified variable
            - "both": Random intercepts + random slopes
                Each group has its own baseline and slope
            - None: No random effects (use linear_reg instead)
        penalty: Regularization penalty for fixed effects (future: not in initial implementation)
        intercept: Whether to fit an intercept term (default True)
        engine: Computational engine to use (default "statsmodels")

    Returns:
        ModelSpec for panel regression

    Examples:
        >>> # Random intercepts only (most common)
        >>> spec = panel_reg()
        >>> wf = workflow().add_formula("sales ~ price").add_model(spec)
        >>> fit = wf.fit_global(data, group_col='store_id')

        >>> # Random intercepts + random slopes for 'time'
        >>> spec = panel_reg(random_effects="both")
        >>> spec = spec.set_args(slope_var='time')
        >>> wf = workflow().add_formula("health ~ time + treatment").add_model(spec)
        >>> fit = wf.fit_global(data, group_col='patient_id')

        >>> # Use with recipe preprocessing
        >>> rec = recipe().step_normalize(all_numeric())
        >>> wf = workflow().add_recipe(rec).add_model(panel_reg())
        >>> fit = wf.fit_global(data, group_col='store_id')

        >>> # Compare in WorkflowSet
        >>> models = [linear_reg(), panel_reg()]
        >>> wf_set = WorkflowSet.from_cross(formulas, models)
        >>> results = wf_set.fit_global(data, group_col='store_id')
        >>> ranked = results.rank_results('rmse')

    Notes:
        - Must use fit_global(data, group_col='...') to specify group column
        - Group column must have at least 2 groups
        - Each group must have at least 2 observations
        - Random slopes require .set_args(slope_var='variable_name')
        - Provides ICC (Intraclass Correlation Coefficient) in stats output
        - Works with recipe preprocessing (group column preserved)
    """
    args = {
        "intercept": intercept,
        "random_effects": random_effects,
    }
    if penalty is not None:
        args["penalty"] = penalty

    return ModelSpec(
        model_type="panel_reg",
        engine=engine,
        mode="regression",
        args=args,
    )
