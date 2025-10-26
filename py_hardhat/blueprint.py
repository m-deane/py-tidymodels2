"""
Blueprint: Immutable preprocessing metadata for consistent data handling

The Blueprint captures all preprocessing decisions made during mold() so they
can be consistently applied to new data during forge().
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import pandas as pd


@dataclass(frozen=True)
class Blueprint:
    """
    Immutable preprocessing blueprint created by mold().

    Stores all metadata needed to consistently transform new data:
    - Formula specification
    - Column roles (outcome, predictor, time_index, group)
    - Factor levels for categorical variables
    - Column order and types
    - Patsy design info for consistent categorical encoding

    Attributes:
        formula: Original formula string (e.g., "y ~ x1 + x2")
        roles: Dict mapping role names to column lists
        factor_levels: Dict mapping categorical columns to their levels
        column_order: Ordered list of predictor column names
        ptypes: Dict mapping column names to pandas dtypes
        intercept: Whether intercept was included in formula
        indicators: Categorical encoding strategy ("traditional" or "none")
        design_info: Patsy DesignInfo for predictors (for consistent encoding)
        outcome_design_info: Patsy DesignInfo for outcomes (optional)

    Example:
        >>> blueprint = Blueprint(
        ...     formula="sales ~ price + category",
        ...     roles={"outcome": ["sales"], "predictor": ["price", "category"]},
        ...     factor_levels={"category": ["A", "B", "C"]},
        ...     column_order=["price", "category"],
        ...     ptypes={"price": "float64", "category": "object"},
        ...     intercept=True,
        ...     indicators="traditional",
        ...     design_info=None,  # Patsy DesignInfo object
        ...     outcome_design_info=None
        ... )
    """

    formula: str
    roles: Dict[str, List[str]]
    factor_levels: Dict[str, List[Any]]
    column_order: List[str]
    ptypes: Dict[str, str]
    intercept: bool = True
    indicators: str = "traditional"
    design_info: Optional[Any] = None  # patsy.DesignInfo
    outcome_design_info: Optional[Any] = None  # patsy.DesignInfo

    def __post_init__(self):
        """Validate blueprint after creation"""
        valid_roles = {"outcome", "predictor", "time_index", "group"}
        invalid = set(self.roles.keys()) - valid_roles
        if invalid:
            raise ValueError(f"Invalid roles: {invalid}. Must be one of {valid_roles}")

        valid_indicators = {"traditional", "none"}
        if self.indicators not in valid_indicators:
            raise ValueError(f"indicators must be one of {valid_indicators}")


@dataclass
class MoldedData:
    """
    Container for model-ready data produced by mold() or forge().

    Contains separate DataFrames for outcomes and predictors, plus metadata.

    Attributes:
        outcomes: DataFrame with outcome variable(s) - None for prediction data
        predictors: DataFrame with predictor variables (design matrix)
        blueprint: Blueprint containing preprocessing metadata
        extras: Dict containing additional data (time_index, groups, etc.)

    Example:
        >>> molded = MoldedData(
        ...     outcomes=pd.DataFrame({"sales": [100, 200, 300]}),
        ...     predictors=pd.DataFrame({
        ...         "Intercept": [1, 1, 1],
        ...         "price": [10, 20, 30],
        ...         "category[B]": [0, 1, 0]
        ...     }),
        ...     blueprint=blueprint,
        ...     extras={"time_index": pd.Series([0, 1, 2])}
        ... )
    """

    outcomes: Optional[pd.DataFrame]
    predictors: pd.DataFrame
    blueprint: Blueprint
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate molded data after creation"""
        if self.outcomes is not None and len(self.outcomes) != len(self.predictors):
            raise ValueError(
                f"outcomes and predictors must have same length. "
                f"Got {len(self.outcomes)} vs {len(self.predictors)}"
            )
