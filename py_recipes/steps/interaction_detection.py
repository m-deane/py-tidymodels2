"""
EIX (Explain Interactions in XGBoost) - Tree-based interaction detection and feature selection.

This module implements the EIX algorithm for py-tidymodels recipes, based on the R EIX package.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Literal, Union
import pandas as pd
import numpy as np

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


@dataclass
class StepEIX:
    """
    EIX - Explain Interactions in XGBoost/LightGBM for feature selection.

    Analyzes tree structure to identify important variable interactions and creates
    interaction features based on tree model gain.

    Parameters
    ----------
    tree_model : xgboost.Booster or lightgbm.Booster
        Pre-fitted tree-based model (XGBoost or LightGBM). REQUIRED.
    outcome : str
        Outcome variable name (needed for data validation). REQUIRED.
    option : {'variables', 'interactions', 'both'}
        What to extract:
        - 'variables': Only single variable importance
        - 'interactions': Only interaction importance
        - 'both': Both variables and interactions (default)
    top_n : int, optional
        Select top N most important features/interactions. None = keep all.
    min_gain : float
        Minimum sumGain threshold for keeping features (default: 0.0)
    create_interactions : bool
        Whether to create interaction features (parent × child) (default: True)
    keep_original_cols : bool
        Keep original columns alongside EIX features (default: False)
    skip : bool
        Skip this step during prep/bake (default: False)
    id : str, optional
        Unique identifier for this step

    Notes
    -----
    - Requires pre-fitted XGBoost or LightGBM model
    - Model must be trained on the same variables in the data
    - Creates interaction features by multiplying parent × child variables
    - Interactions are identified where child gain > parent gain

    Examples
    --------
    >>> from xgboost import XGBRegressor
    >>> from py_recipes import recipe
    >>>
    >>> # Fit tree model (REQUIRED)
    >>> tree_model = XGBRegressor(n_estimators=100, max_depth=3)
    >>> tree_model.fit(X_train, y_train)
    >>>
    >>> # Basic usage - find and create top interactions
    >>> rec = recipe().step_eix(
    ...     tree_model=tree_model,
    ...     outcome='target',
    ...     option='interactions',
    ...     top_n=10
    ... )
    >>>
    >>> # Conservative - only strong interactions
    >>> rec = recipe().step_eix(
    ...     tree_model=tree_model,
    ...     outcome='sales',
    ...     option='interactions',
    ...     min_gain=0.1,
    ...     top_n=5
    ... )
    """

    tree_model: Any
    outcome: str
    option: Literal['variables', 'interactions', 'both'] = 'both'
    top_n: Optional[int] = None
    min_gain: float = 0.0
    create_interactions: bool = True
    keep_original_cols: bool = False
    skip: bool = False
    id: Optional[str] = None

    # Internal state (set during prep)
    _is_prepped: bool = field(default=False, init=False, repr=False)
    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _interactions_to_create: List[Dict] = field(default_factory=list, init=False, repr=False)
    _importance_table: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _original_columns: List[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        """Validate parameters after initialization."""
        # Check dependencies
        if not HAS_XGB and not HAS_LGB:
            raise ImportError(
                "xgboost or lightgbm package required for step_eix(). "
                "Install with: pip install xgboost lightgbm"
            )

        # Validate tree_model
        if self.tree_model is None:
            raise ValueError("tree_model is required for step_eix()")

        if not self._is_tree_model_fitted():
            raise ValueError(
                "tree_model must be fitted before step_eix(). "
                "Call tree_model.fit(X, y) first."
            )

        # Validate outcome
        if not self.outcome:
            raise ValueError("outcome is required for step_eix()")

        # Validate option
        if self.option not in ['variables', 'interactions', 'both']:
            raise ValueError(
                f"option must be 'variables', 'interactions', or 'both', got '{self.option}'"
            )

        # Validate numeric parameters
        if self.top_n is not None and self.top_n <= 0:
            raise ValueError(f"top_n must be positive, got {self.top_n}")

        if self.min_gain < 0:
            raise ValueError(f"min_gain must be non-negative, got {self.min_gain}")

    def _is_tree_model_fitted(self) -> bool:
        """Check if tree model is fitted."""
        model = self.tree_model

        # XGBoost check
        if HAS_XGB and isinstance(model, (xgb.Booster, xgb.XGBModel)):
            if isinstance(model, xgb.Booster):
                # Booster - check if it has trees
                try:
                    model.get_dump()
                    return True
                except:
                    return False
            else:
                # XGBModel (XGBRegressor, XGBClassifier)
                return hasattr(model, '_Booster') and model._Booster is not None

        # LightGBM check
        if HAS_LGB and isinstance(model, (lgb.Booster, lgb.LGBMModel)):
            if isinstance(model, lgb.Booster):
                return model.num_trees() > 0
            else:
                return hasattr(model, '_Booster') and model._Booster is not None

        # Unknown model type
        raise TypeError(
            f"tree_model must be XGBoost or LightGBM model, got {type(model)}"
        )

    def _extract_trees_dataframe(self) -> pd.DataFrame:
        """Extract tree structure as DataFrame from XGBoost or LightGBM model."""
        model = self.tree_model

        # XGBoost
        if HAS_XGB and isinstance(model, (xgb.Booster, xgb.XGBModel)):
            if isinstance(model, xgb.XGBModel):
                booster = model.get_booster()
            else:
                booster = model

            # Get trees as dataframe
            trees_df = booster.trees_to_dataframe()
            return trees_df

        # LightGBM
        if HAS_LGB and isinstance(model, (lgb.Booster, lgb.LGBMModel)):
            if isinstance(model, lgb.LGBMModel):
                booster = model.booster_
            else:
                booster = model

            # Get trees as dataframe
            trees_df = booster.trees_to_dataframe()
            return trees_df

        raise TypeError(f"Unsupported model type: {type(model)}")

    def _calculate_interactions(self, trees_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate interaction importance from tree structure.

        Identifies interactions where:
        1. A variable (parent) splits into another variable (child)
        2. Child gain > parent gain (strong interaction signal)
        3. Parent and child are different variables
        """
        interactions_list = []

        # Process each tree
        for tree_id in trees_df['Tree'].unique():
            tree = trees_df[trees_df['Tree'] == tree_id].copy()

            # Skip trees with only one node
            if len(tree) <= 1:
                continue

            # For each non-leaf node, find its children
            non_leaf = tree[tree['Feature'] != 'Leaf'].copy()

            for idx, node in non_leaf.iterrows():
                parent_feature = node['Feature']
                parent_gain = node.get('Gain', node.get('Split', 0.0))

                # Get left and right children
                left_child_id = node['Yes']
                right_child_id = node['No']

                # Process left child
                left_child = tree[tree['Node'] == left_child_id]
                if not left_child.empty:
                    left_row = left_child.iloc[0]
                    if left_row['Feature'] != 'Leaf':
                        child_feature = left_row['Feature']
                        child_gain = left_row.get('Gain', left_row.get('Split', 0.0))

                        # Check if this is a strong interaction
                        if child_feature != parent_feature and child_gain > parent_gain:
                            interactions_list.append({
                                'Parent': parent_feature,
                                'Child': child_feature,
                                'gain': child_gain,
                                'tree': tree_id
                            })

                # Process right child
                right_child = tree[tree['Node'] == right_child_id]
                if not right_child.empty:
                    right_row = right_child.iloc[0]
                    if right_row['Feature'] != 'Leaf':
                        child_feature = right_row['Feature']
                        child_gain = right_row.get('Gain', right_row.get('Split', 0.0))

                        # Check if this is a strong interaction
                        if child_feature != parent_feature and child_gain > parent_gain:
                            interactions_list.append({
                                'Parent': parent_feature,
                                'Child': child_feature,
                                'gain': child_gain,
                                'tree': tree_id
                            })

        if not interactions_list:
            return pd.DataFrame(columns=['Parent', 'Child', 'sumGain', 'frequency'])

        # Aggregate interactions
        interactions_df = pd.DataFrame(interactions_list)

        # Create interaction name
        interactions_df['interaction'] = (
            interactions_df['Parent'] + ':' + interactions_df['Child']
        )

        # Aggregate by interaction
        importance = interactions_df.groupby(['Parent', 'Child']).agg({
            'gain': ['sum', 'count', 'mean']
        }).reset_index()

        importance.columns = ['Parent', 'Child', 'sumGain', 'frequency', 'meanGain']
        importance = importance.sort_values('sumGain', ascending=False)

        return importance

    def _calculate_variable_importance(self, trees_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate single variable importance from tree structure."""
        # Get all non-leaf nodes
        variables = trees_df[trees_df['Feature'] != 'Leaf'].copy()

        if variables.empty:
            return pd.DataFrame(columns=['Feature', 'sumGain', 'frequency', 'meanGain'])

        # Use 'Gain' column if available, otherwise 'Split'
        gain_col = 'Gain' if 'Gain' in variables.columns else 'Split'

        # Aggregate by feature
        importance = variables.groupby('Feature').agg({
            gain_col: ['sum', 'count', 'mean']
        }).reset_index()

        importance.columns = ['Feature', 'sumGain', 'frequency', 'meanGain']
        importance = importance.sort_values('sumGain', ascending=False)

        return importance

    def prep(self, data: pd.DataFrame, training: bool = True) -> "StepEIX":
        """
        Prepare the step by analyzing tree structure and identifying interactions.

        Parameters
        ----------
        data : pd.DataFrame
            Training data
        training : bool
            Whether this is training data (always True for prep)

        Returns
        -------
        StepEIX
            Prepared step
        """
        if self.skip:
            self._is_prepped = True
            return self

        # Validate outcome exists
        if self.outcome not in data.columns:
            raise ValueError(f"Outcome column '{self.outcome}' not found in data")

        # Store original columns (excluding outcome)
        self._original_columns = [col for col in data.columns if col != self.outcome]

        # Extract tree structure
        trees_df = self._extract_trees_dataframe()

        # Calculate importance based on option
        if self.option == 'variables':
            importance = self._calculate_variable_importance(trees_df)
            importance['type'] = 'variable'

        elif self.option == 'interactions':
            importance = self._calculate_interactions(trees_df)
            importance['type'] = 'interaction'

        else:  # 'both'
            var_importance = self._calculate_variable_importance(trees_df)
            var_importance['type'] = 'variable'

            inter_importance = self._calculate_interactions(trees_df)
            inter_importance['type'] = 'interaction'

            # Combine (need to align columns)
            var_importance['Parent'] = None
            var_importance['Child'] = None
            var_importance = var_importance.rename(columns={'Feature': 'name'})

            inter_importance['name'] = (
                inter_importance['Parent'] + ':' + inter_importance['Child']
            )

            # Combine both
            importance = pd.concat([var_importance, inter_importance], ignore_index=True)
            importance = importance.sort_values('sumGain', ascending=False)

        # Filter by min_gain
        importance = importance[importance['sumGain'] >= self.min_gain]

        # Select top N
        if self.top_n is not None:
            importance = importance.head(self.top_n)

        # Store importance table
        self._importance_table = importance.copy()

        # Identify features to select and interactions to create
        self._selected_features = []
        self._interactions_to_create = []

        for _, row in importance.iterrows():
            if row['type'] == 'variable':
                feature_name = row.get('name', row.get('Feature'))
                if feature_name in self._original_columns:
                    self._selected_features.append(feature_name)

            elif row['type'] == 'interaction':
                parent = row['Parent']
                child = row['Child']

                # Add parent and child to selected features
                if parent in self._original_columns and parent not in self._selected_features:
                    self._selected_features.append(parent)
                if child in self._original_columns and child not in self._selected_features:
                    self._selected_features.append(child)

                # Record interaction to create
                if self.create_interactions:
                    self._interactions_to_create.append({
                        'parent': parent,
                        'child': child,
                        'name': f"{parent}_x_{child}",
                        'sumGain': row['sumGain']
                    })

        self._is_prepped = True
        return self

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the EIX transformation to data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to transform

        Returns
        -------
        pd.DataFrame
            Transformed data with selected features and created interactions
        """
        if not self._is_prepped:
            return data.copy()

        result = pd.DataFrame(index=data.index)

        # Add selected features
        for feature in self._selected_features:
            if feature in data.columns:
                result[feature] = data[feature]

        # Create interaction features
        for interaction in self._interactions_to_create:
            parent = interaction['parent']
            child = interaction['child']
            name = interaction['name']

            if parent in data.columns and child in data.columns:
                result[name] = data[parent] * data[child]

        # Keep original columns if requested
        if self.keep_original_cols:
            for col in self._original_columns:
                if col in data.columns and col not in result.columns:
                    result[col] = data[col]

        # Always preserve outcome column
        if self.outcome in data.columns:
            result[self.outcome] = data[self.outcome]

        return result

    def get_importance(self) -> pd.DataFrame:
        """
        Get the importance table calculated during prep.

        Returns
        -------
        pd.DataFrame
            Importance table with sumGain, frequency, and meanGain

        Raises
        ------
        RuntimeError
            If called before prep()
        """
        if not self._is_prepped:
            raise RuntimeError("Must call prep() before get_importance()")

        return self._importance_table.copy()

    def get_interactions(self) -> List[Dict]:
        """
        Get the list of interactions that will be created.

        Returns
        -------
        list of dict
            List of interactions with parent, child, name, and sumGain

        Raises
        ------
        RuntimeError
            If called before prep()
        """
        if not self._is_prepped:
            raise RuntimeError("Must call prep() before get_interactions()")

        return self._interactions_to_create.copy()
