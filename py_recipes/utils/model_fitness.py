"""
Model fitness evaluation utilities for genetic algorithm feature selection.

This module provides functions to evaluate feature subsets using parsnip models
and cross-validation for robust performance estimation.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Callable
from sklearn.model_selection import KFold
from py_yardstick import (
    rmse, mae, mape, r_squared,
    accuracy, precision, recall, f_meas
)


class ModelFitnessEvaluator:
    """
    Evaluate fitness of feature subsets using machine learning models.

    This class handles:
    - Model training and prediction
    - Cross-validation for robust estimates
    - Multiple evaluation metrics
    - Regression and classification tasks

    Parameters
    ----------
    data : pd.DataFrame
        Training data with outcome and predictors
    outcome_col : str
        Name of outcome column
    model_spec : ModelSpec
        Parsnip model specification to use for evaluation
    metric : str
        Metric to optimize ('rmse', 'mae', 'r_squared', 'accuracy', etc.)
    maximize : bool
        Whether to maximize metric (True) or minimize (False)
    cv_folds : int, default=5
        Number of cross-validation folds. If 1, uses train-only evaluation
    random_state : Optional[int]
        Random seed for CV splits

    Attributes
    ----------
    feature_names_ : List[str]
        Names of all candidate features
    n_features_ : int
        Number of candidate features
    """

    def __init__(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        model_spec,
        metric: str = "rmse",
        maximize: bool = False,
        cv_folds: int = 5,
        random_state: Optional[int] = None
    ):
        self.data = data
        self.outcome_col = outcome_col
        self.model_spec = model_spec
        self.metric = metric.lower()
        self.maximize = maximize
        self.cv_folds = cv_folds
        self.random_state = random_state

        # Extract feature names (all columns except outcome)
        self.feature_names_ = [col for col in data.columns if col != outcome_col]
        self.n_features_ = len(self.feature_names_)

        # Validate metric
        self._validate_metric()

    def _validate_metric(self):
        """Validate that metric is supported."""
        supported_metrics = {
            # Regression metrics
            'rmse', 'mae', 'mape', 'r_squared', 'rsq', 'r2',
            # Classification metrics
            'accuracy', 'precision', 'recall', 'f_meas', 'f1'
        }

        if self.metric not in supported_metrics:
            raise ValueError(
                f"Unsupported metric '{self.metric}'. "
                f"Supported: {sorted(supported_metrics)}"
            )

    def _compute_metric(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Compute specified metric given true and predicted values.

        Parameters
        ----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values

        Returns
        -------
        score : float
            Metric score
        """
        # Convert to pandas Series if needed
        if isinstance(y_true, np.ndarray):
            y_true = pd.Series(y_true)
        if isinstance(y_pred, np.ndarray):
            y_pred = pd.Series(y_pred, index=y_true.index)

        # Compute metric
        if self.metric in ['rmse']:
            result = rmse(y_true, y_pred)
        elif self.metric in ['mae']:
            result = mae(y_true, y_pred)
        elif self.metric in ['mape']:
            result = mape(y_true, y_pred)
        elif self.metric in ['r_squared', 'rsq', 'r2']:
            result = r_squared(y_true, y_pred)
        elif self.metric in ['accuracy']:
            result = accuracy(y_true, y_pred)
        elif self.metric in ['precision']:
            result = precision(y_true, y_pred)
        elif self.metric in ['recall']:
            result = recall(y_true, y_pred)
        elif self.metric in ['f_meas', 'f1']:
            result = f_meas(y_true, y_pred)
        else:
            raise ValueError(f"Metric '{self.metric}' not implemented")

        # Extract scalar value from result DataFrame
        return result.iloc[0]["value"]

    def evaluate_features(
        self,
        feature_indices: np.ndarray
    ) -> float:
        """
        Evaluate a feature subset using the model and metric.

        Parameters
        ----------
        feature_indices : np.ndarray
            Binary array indicating which features to include (1) or exclude (0)

        Returns
        -------
        fitness : float
            Fitness score (higher is better, regardless of metric direction)
        """
        # Convert binary array to feature names
        selected_features = [
            self.feature_names_[i]
            for i in range(len(feature_indices))
            if feature_indices[i] == 1
        ]

        if len(selected_features) == 0:
            # No features selected - return worst possible fitness
            return -np.inf

        # Build formula
        formula = f"{self.outcome_col} ~ {' + '.join(selected_features)}"

        try:
            if self.cv_folds <= 1:
                # No CV - train and evaluate on same data
                score = self._evaluate_no_cv(formula)
            else:
                # Cross-validation
                score = self._evaluate_with_cv(formula)

            # Convert to fitness (higher is better)
            if self.maximize:
                fitness = score
            else:
                # For metrics to minimize (RMSE, MAE), invert and scale
                # Use 1 / (1 + score) to ensure positive fitness
                fitness = 1.0 / (1.0 + abs(score))

            return fitness

        except Exception as e:
            # Model fitting failed - return very low fitness
            return -np.inf

    def _evaluate_no_cv(self, formula: str) -> float:
        """Evaluate on training data without CV (for debugging/speed)."""
        fit = self.model_spec.fit(self.data, formula)
        predictions = fit.predict(self.data)

        y_true = self.data[self.outcome_col]
        y_pred = predictions[".pred"]

        return self._compute_metric(y_true, y_pred)

    def _evaluate_with_cv(self, formula: str) -> float:
        """Evaluate using cross-validation."""
        kf = KFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )

        scores = []
        for train_idx, val_idx in kf.split(self.data):
            train_data = self.data.iloc[train_idx].copy()
            val_data = self.data.iloc[val_idx].copy()

            # Fit model on fold
            fit = self.model_spec.fit(train_data, formula)

            # Predict on validation fold
            predictions = fit.predict(val_data)

            y_true = val_data[self.outcome_col]
            y_pred = predictions[".pred"]

            score = self._compute_metric(y_true, y_pred)
            scores.append(score)

        # Return mean score across folds
        return np.mean(scores)

    def create_fitness_function(self) -> Callable[[np.ndarray], float]:
        """
        Create a fitness function for the genetic algorithm.

        Returns
        -------
        fitness_fn : Callable
            Function that takes a binary chromosome and returns fitness
        """
        def fitness_fn(chromosome: np.ndarray) -> float:
            return self.evaluate_features(chromosome)

        return fitness_fn


def create_model_fitness_evaluator(
    data: pd.DataFrame,
    outcome_col: str,
    model_spec,
    metric: str = "rmse",
    maximize: bool = False,
    cv_folds: int = 5,
    random_state: Optional[int] = None
) -> Callable[[np.ndarray], float]:
    """
    Convenience function to create a fitness evaluator.

    Parameters
    ----------
    data : pd.DataFrame
        Training data
    outcome_col : str
        Outcome column name
    model_spec : ModelSpec
        Parsnip model specification
    metric : str
        Metric to optimize
    maximize : bool
        Whether to maximize metric
    cv_folds : int
        CV folds
    random_state : Optional[int]
        Random seed

    Returns
    -------
    fitness_fn : Callable
        Fitness function for GA

    Examples
    --------
    >>> from py_parsnip import linear_reg
    >>> fitness_fn = create_model_fitness_evaluator(
    ...     data=train_data,
    ...     outcome_col='y',
    ...     model_spec=linear_reg(),
    ...     metric='rmse',
    ...     cv_folds=5
    ... )
    >>> # Use with GeneticAlgorithm
    >>> ga = GeneticAlgorithm(n_features=10, fitness_function=fitness_fn)
    >>> best_features, best_fitness, history = ga.evolve()
    """
    evaluator = ModelFitnessEvaluator(
        data=data,
        outcome_col=outcome_col,
        model_spec=model_spec,
        metric=metric,
        maximize=maximize,
        cv_folds=cv_folds,
        random_state=random_state
    )
    return evaluator.create_fitness_function()
