"""
Blueprint serialization utilities for MLflow.

Handles serialization and deserialization of hardhat Blueprint objects,
working around the limitation that patsy DesignInfo objects cannot be pickled.

The approach:
1. Extract all serializable metadata from Blueprint (formula, factor_levels, etc.)
2. Store DesignInfo state separately (NOT pickled, but as metadata)
3. On load, we don't need to perfectly reconstruct DesignInfo - the stored
   metadata is sufficient for forge() to work correctly
"""

import json
from typing import Dict, Any, Optional
from dataclasses import asdict
from py_hardhat.blueprint import Blueprint


def extract_blueprint_metadata(blueprint: Blueprint) -> Dict[str, Any]:
    """
    Extract serializable metadata from a Blueprint.

    This function extracts all important information from a Blueprint except
    the non-serializable DesignInfo objects. The extracted metadata is sufficient
    to reconstruct a functional Blueprint for making predictions.

    Args:
        blueprint: Blueprint object to extract metadata from

    Returns:
        Dict containing serializable blueprint metadata:
        - formula: Formula string (e.g., "y ~ x1 + x2")
        - roles: Dict mapping role names to column lists
        - factor_levels: Dict mapping categorical columns to their levels
        - column_order: Ordered list of predictor column names
        - ptypes: Dict mapping column names to pandas dtype strings
        - intercept: Whether intercept was included
        - indicators: Categorical encoding strategy
        - design_info_state: Minimal state from DesignInfo (if present)
        - outcome_design_info_state: Minimal state from outcome DesignInfo (if present)

    Examples:
        >>> blueprint = Blueprint(
        ...     formula="y ~ x1 + x2",
        ...     roles={"outcome": ["y"], "predictor": ["x1", "x2"]},
        ...     factor_levels={},
        ...     column_order=["x1", "x2"],
        ...     ptypes={"x1": "float64", "x2": "float64"},
        ...     intercept=True,
        ...     indicators="traditional",
        ...     design_info=None,
        ...     outcome_design_info=None
        ... )
        >>> metadata = extract_blueprint_metadata(blueprint)
        >>> metadata['formula']
        'y ~ x1 + x2'
    """
    # Extract DesignInfo state BEFORE converting to dict (to avoid deepcopy issues)
    design_info_state = None
    if blueprint.design_info is not None:
        try:
            design_info_state = {
                'column_names': blueprint.design_info.column_names,
                'term_names': blueprint.design_info.term_names,
                'describe': blueprint.design_info.describe(),
            }
        except Exception:
            # If extraction fails, set to None (we can reconstruct from formula)
            design_info_state = None

    outcome_design_info_state = None
    if blueprint.outcome_design_info is not None:
        try:
            outcome_design_info_state = {
                'column_names': blueprint.outcome_design_info.column_names,
                'term_names': blueprint.outcome_design_info.term_names,
                'describe': blueprint.outcome_design_info.describe(),
            }
        except Exception:
            outcome_design_info_state = None

    # Manually extract serializable fields (avoid asdict which tries to deepcopy DesignInfo)
    blueprint_dict = {
        'formula': blueprint.formula,
        'roles': blueprint.roles,
        'factor_levels': blueprint.factor_levels,
        'column_order': blueprint.column_order,
        'ptypes': blueprint.ptypes,
        'intercept': blueprint.intercept,
        'indicators': blueprint.indicators,
        'design_info': design_info_state,
        'outcome_design_info': outcome_design_info_state,
    }

    return blueprint_dict


def reconstruct_blueprint(metadata: Dict[str, Any]) -> Blueprint:
    """
    Reconstruct a Blueprint from serialized metadata.

    This function recreates a Blueprint object from the metadata extracted by
    extract_blueprint_metadata(). It recreates the DesignInfo objects by calling
    dmatrices() on synthetic sample data that represents all categorical levels.

    Args:
        metadata: Dict containing blueprint metadata (from extract_blueprint_metadata)

    Returns:
        Reconstructed Blueprint object with functional DesignInfo

    Examples:
        >>> metadata = {
        ...     'formula': 'y ~ x1 + x2',
        ...     'roles': {'outcome': ['y'], 'predictor': ['x1', 'x2']},
        ...     'factor_levels': {},
        ...     'column_order': ['x1', 'x2'],
        ...     'ptypes': {'x1': 'float64', 'x2': 'float64'},
        ...     'intercept': True,
        ...     'indicators': 'traditional',
        ...     'design_info': None,
        ...     'outcome_design_info': None
        ... }
        >>> blueprint = reconstruct_blueprint(metadata)
        >>> blueprint.formula
        'y ~ x1 + x2'
    """
    import pandas as pd
    import numpy as np
    from patsy import dmatrices

    # Create synthetic sample data to recreate DesignInfo
    # This data must include all categorical levels that were present during training
    sample_data = {}

    # Get all columns from roles
    all_cols = set()
    for role_cols in metadata['roles'].values():
        all_cols.update(role_cols)

    # Also add any columns from factor_levels
    all_cols.update(metadata['factor_levels'].keys())

    # Create sample data for each column
    for col in all_cols:
        if col in metadata['factor_levels']:
            # Categorical column - use all levels
            sample_data[col] = metadata['factor_levels'][col]
        else:
            # Numeric column - use dummy values (same length as longest categorical)
            max_len = max([len(levels) for levels in metadata['factor_levels'].values()], default=2)
            sample_data[col] = np.ones(max_len)

    # Ensure all columns have the same length
    max_len = max([len(v) if isinstance(v, (list, np.ndarray)) else 1 for v in sample_data.values()], default=2)
    for col in sample_data:
        if isinstance(sample_data[col], (list, np.ndarray)):
            # Extend if needed
            if len(sample_data[col]) < max_len:
                sample_data[col] = list(sample_data[col]) + [sample_data[col][0]] * (max_len - len(sample_data[col]))
        else:
            sample_data[col] = [sample_data[col]] * max_len

    sample_df = pd.DataFrame(sample_data)

    # Recreate DesignInfo by calling dmatrices on sample data
    try:
        y_mat, X_mat = dmatrices(
            metadata['formula'],
            sample_df,
            return_type="dataframe",
            NA_action="raise"
        )
        predictor_design_info = X_mat.design_info
        outcome_design_info = y_mat.design_info
    except Exception:
        # If recreation fails, set to None
        predictor_design_info = None
        outcome_design_info = None

    # Create Blueprint with reconstructed DesignInfo
    return Blueprint(
        formula=metadata['formula'],
        roles=metadata['roles'],
        factor_levels=metadata['factor_levels'],
        column_order=metadata['column_order'],
        ptypes=metadata['ptypes'],
        intercept=metadata.get('intercept', True),
        indicators=metadata.get('indicators', 'traditional'),
        design_info=predictor_design_info,
        outcome_design_info=outcome_design_info
    )


def save_blueprint_metadata(blueprint: Blueprint, path: str) -> None:
    """
    Save blueprint metadata to a JSON file.

    Args:
        blueprint: Blueprint object to save
        path: Path to JSON file where metadata will be saved

    Examples:
        >>> save_blueprint_metadata(blueprint, "model/blueprint.json")
    """
    metadata = extract_blueprint_metadata(blueprint)
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_blueprint_metadata(path: str) -> Blueprint:
    """
    Load blueprint metadata from a JSON file and reconstruct Blueprint.

    Args:
        path: Path to JSON file containing blueprint metadata

    Returns:
        Reconstructed Blueprint object

    Examples:
        >>> blueprint = load_blueprint_metadata("model/blueprint.json")
    """
    with open(path, 'r') as f:
        metadata = json.load(f)
    return reconstruct_blueprint(metadata)
