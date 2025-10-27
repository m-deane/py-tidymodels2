"""
K-Nearest Neighbors model specification

Supports multiple engines:
- sklearn: KNeighborsRegressor

Parameters (tidymodels naming):
- neighbors: Number of neighbors to use
- weight_func: Weight function for neighbors
- dist_power: Power parameter for Minkowski distance
"""

from typing import Optional
from py_parsnip.model_spec import ModelSpec


def nearest_neighbor(
    neighbors: Optional[int] = None,
    weight_func: Optional[str] = None,
    dist_power: Optional[float] = None,
    engine: str = "sklearn",
) -> ModelSpec:
    """
    Create a k-nearest neighbors model specification.

    Args:
        neighbors: Number of neighbors to use for predictions
            - For sklearn: maps to 'n_neighbors'
            - Default: 5
        weight_func: Weight function used in prediction
            - For sklearn: maps to 'weights'
            - Options: "uniform" (all neighbors weighted equally),
                      "distance" (closer neighbors have more influence)
            - Default: "uniform"
        dist_power: Power parameter for Minkowski distance metric
            - For sklearn: maps to 'p'
            - p=1: Manhattan distance, p=2: Euclidean distance
            - Default: 2 (Euclidean)
        engine: Computational engine to use (default "sklearn")

    Returns:
        ModelSpec for k-nearest neighbors

    Examples:
        >>> # Default KNN (5 neighbors, uniform weights, Euclidean distance)
        >>> spec = nearest_neighbor()

        >>> # KNN with 10 neighbors
        >>> spec = nearest_neighbor(neighbors=10)

        >>> # Distance-weighted KNN
        >>> spec = nearest_neighbor(weight_func="distance")

        >>> # KNN with Manhattan distance
        >>> spec = nearest_neighbor(dist_power=1)

        >>> # Fully customized KNN
        >>> spec = nearest_neighbor(neighbors=7, weight_func="distance", dist_power=2)
    """
    # Build args dict (only include non-None values)
    args = {}
    if neighbors is not None:
        args["neighbors"] = neighbors
    if weight_func is not None:
        args["weight_func"] = weight_func
    if dist_power is not None:
        args["dist_power"] = dist_power

    return ModelSpec(
        model_type="nearest_neighbor",
        engine=engine,
        mode="regression",
        args=args,
    )
