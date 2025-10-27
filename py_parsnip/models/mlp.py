"""
Multi-Layer Perceptron (Neural Network) model specification

Supports multiple engines:
- sklearn: MLPRegressor

Parameters (tidymodels naming):
- hidden_units: Number of hidden units (tuple or int for single layer)
- penalty: L2 regularization penalty (alpha)
- epochs: Maximum number of training iterations
- learn_rate: Initial learning rate
- activation: Activation function for hidden layers
"""

from typing import Optional, Union, Tuple
from py_parsnip.model_spec import ModelSpec


def mlp(
    hidden_units: Optional[Union[int, Tuple[int, ...]]] = None,
    penalty: Optional[float] = None,
    epochs: Optional[int] = None,
    learn_rate: Optional[float] = None,
    activation: Optional[str] = None,
    engine: str = "sklearn",
) -> ModelSpec:
    """
    Create a multi-layer perceptron (neural network) model specification.

    Args:
        hidden_units: Number of hidden units in each layer
            - For sklearn: maps to 'hidden_layer_sizes'
            - Can be int (single hidden layer) or tuple (multiple layers)
            - Example: 10 = one layer with 10 units, (10, 5) = two layers with 10 and 5 units
            - Default: (100,)
        penalty: L2 regularization parameter
            - For sklearn: maps to 'alpha'
            - Controls weight penalty to prevent overfitting
            - Default: 0.0001
        epochs: Maximum number of training iterations
            - For sklearn: maps to 'max_iter'
            - Default: 200
        learn_rate: Initial learning rate for weight updates
            - For sklearn: maps to 'learning_rate_init'
            - Default: 0.001
        activation: Activation function for hidden layers
            - For sklearn: maps to 'activation'
            - Options: "relu", "tanh", "logistic", "identity"
            - Default: "relu"
        engine: Computational engine to use (default "sklearn")

    Returns:
        ModelSpec for multi-layer perceptron

    Examples:
        >>> # Default MLP (single hidden layer with 100 units)
        >>> spec = mlp()

        >>> # MLP with single hidden layer of 50 units
        >>> spec = mlp(hidden_units=50)

        >>> # MLP with two hidden layers (100 and 50 units)
        >>> spec = mlp(hidden_units=(100, 50))

        >>> # MLP with regularization
        >>> spec = mlp(penalty=0.01)

        >>> # MLP with more training epochs
        >>> spec = mlp(epochs=500)

        >>> # MLP with tanh activation
        >>> spec = mlp(activation="tanh")

        >>> # Fully customized MLP
        >>> spec = mlp(
        ...     hidden_units=(100, 50, 25),
        ...     penalty=0.001,
        ...     epochs=300,
        ...     learn_rate=0.01,
        ...     activation="relu"
        ... )
    """
    # Build args dict (only include non-None values)
    args = {}
    if hidden_units is not None:
        args["hidden_units"] = hidden_units
    if penalty is not None:
        args["penalty"] = penalty
    if epochs is not None:
        args["epochs"] = epochs
    if learn_rate is not None:
        args["learn_rate"] = learn_rate
    if activation is not None:
        args["activation"] = activation

    return ModelSpec(
        model_type="mlp",
        engine=engine,
        mode="regression",
        args=args,
    )
