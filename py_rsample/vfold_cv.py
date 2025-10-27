"""
V-fold cross-validation for general data (not time series specific)

Provides standard k-fold cross-validation splits for model evaluation.
"""

from typing import Optional, List
import pandas as pd
import numpy as np
from dataclasses import dataclass

from py_rsample.split import RSplit


@dataclass
class VFoldCV:
    """
    Container for V-fold cross-validation splits.

    Provides k-fold cross-validation where data is randomly partitioned into
    v groups (folds). Each fold serves as the assessment/test set once while
    the remaining folds form the analysis/training set.

    Attributes:
        splits: List of RSplit objects, one per fold
        v: Number of folds
        stratified: Whether stratification was used
        strata: Column name used for stratification (if any)
    """
    splits: List[RSplit]
    v: int
    stratified: bool = False
    strata: Optional[str] = None

    def __iter__(self):
        """Iterate over splits"""
        return iter(self.splits)

    def __len__(self):
        """Number of splits"""
        return len(self.splits)

    def __getitem__(self, idx):
        """Get split by index"""
        return self.splits[idx]


def vfold_cv(
    data: pd.DataFrame,
    v: int = 10,
    repeats: int = 1,
    strata: Optional[str] = None,
    seed: Optional[int] = None
) -> VFoldCV:
    """
    Create V-fold cross-validation splits.

    Randomly splits data into v groups (folds) for cross-validation. Each fold
    serves as the test set once while the remaining folds form the training set.

    Args:
        data: DataFrame to split
        v: Number of folds (default: 10)
        repeats: Number of times to repeat the v-fold partitioning (default: 1)
        strata: Optional column name for stratified sampling
        seed: Random seed for reproducibility

    Returns:
        VFoldCV object containing all splits

    Examples:
        >>> # Basic 5-fold CV
        >>> folds = vfold_cv(data, v=5)
        >>>
        >>> # 10-fold CV with stratification
        >>> folds = vfold_cv(data, v=10, strata='outcome')
        >>>
        >>> # Repeated CV
        >>> folds = vfold_cv(data, v=5, repeats=3)
        >>>
        >>> # Iterate over folds
        >>> for fold in folds:
        ...     train = fold.training()
        ...     test = fold.testing()
    """
    if v < 2:
        raise ValueError("v must be at least 2")

    if seed is not None:
        np.random.seed(seed)

    n = len(data)
    splits = []

    for repeat in range(repeats):
        if strata is not None:
            # Stratified sampling
            fold_assignments = _stratified_fold_assignment(data, strata, v)
        else:
            # Random assignment to folds
            indices = np.arange(n)
            np.random.shuffle(indices)
            fold_assignments = np.array_split(indices, v)

        # Create RSplit for each fold
        for fold_idx in range(v):
            # Test set is the current fold
            if strata is not None:
                test_indices = fold_assignments[fold_idx]
            else:
                test_indices = fold_assignments[fold_idx]

            # Train set is all other folds
            if strata is not None:
                train_indices = np.concatenate([fold_assignments[i] for i in range(v) if i != fold_idx])
            else:
                train_indices = np.concatenate([fold_assignments[i] for i in range(v) if i != fold_idx])

            # Create Split and wrap in RSplit
            from py_rsample.split import Split
            split_id = f"Fold{fold_idx+1:02d}" if repeats == 1 else f"Repeat{repeat+1}_Fold{fold_idx+1:02d}"
            split_obj = Split(
                data=data,
                in_id=train_indices,
                out_id=test_indices,
                id=split_id
            )
            rsplit = RSplit(split_obj)
            splits.append(rsplit)

    return VFoldCV(
        splits=splits,
        v=v,
        stratified=strata is not None,
        strata=strata
    )


def _stratified_fold_assignment(
    data: pd.DataFrame,
    strata_col: str,
    v: int
) -> List[np.ndarray]:
    """
    Create stratified fold assignments.

    Ensures each fold has approximately the same distribution of the strata variable.

    Args:
        data: DataFrame
        strata_col: Column name to stratify by
        v: Number of folds

    Returns:
        List of arrays, one per fold, containing row indices
    """
    if strata_col not in data.columns:
        raise ValueError(f"Strata column '{strata_col}' not found in data")

    # Get stratification variable
    strata = data[strata_col]

    # Initialize fold assignments
    fold_lists = [[] for _ in range(v)]

    # For each stratum, assign indices to folds
    for stratum_value in strata.unique():
        # Get indices for this stratum
        stratum_indices = data[strata == stratum_value].index.tolist()

        # Shuffle within stratum
        np.random.shuffle(stratum_indices)

        # Assign to folds in round-robin fashion
        for i, idx in enumerate(stratum_indices):
            fold_idx = i % v
            fold_lists[fold_idx].append(idx)

    # Convert to arrays and shuffle each fold
    fold_arrays = []
    for fold_list in fold_lists:
        fold_array = np.array(fold_list)
        np.random.shuffle(fold_array)
        fold_arrays.append(fold_array)

    return fold_arrays
