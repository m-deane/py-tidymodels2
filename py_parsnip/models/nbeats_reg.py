"""
NBEATS regression model specification.

NBEATS (Neural Basis Expansion Analysis for Time Series) is a deep learning
architecture specifically designed for univariate time series forecasting.

Key Features:
- Interpretable decomposition: Separate trend and seasonality stacks
- Pure univariate: No exogenous variables (use generic stacks if needed)
- Backward/forward casting: Provides both forecast and backcast
- Doubly residual architecture: Hierarchical decomposition

Architecture:
- Stack types:
  - 'trend': Polynomial basis for trend component
  - 'seasonality': Harmonic basis for seasonal patterns
  - 'generic': Learned basis (no constraints)
- Each stack contains multiple blocks
- Blocks share weights within stack (optional)

When to Use NBEATS:
- Pure univariate forecasting (no exogenous variables)
- Interpretable trend/seasonality decomposition needed
- Strong periodic patterns (daily, weekly, yearly)
- Sufficient training data (100+ observations)

Parameters (tidymodels naming):
- horizon: Forecast horizon (number of steps ahead)
- input_size: Lookback window (default: 2*horizon)
- n_harmonics: Number of harmonics for seasonality stack
- n_polynomials: Polynomial degree for trend stack
- stack_types: List of stack types to use
- n_blocks: Number of blocks per stack
- mlp_units: Hidden layer sizes for each stack
- share_weights_in_stack: Whether blocks share weights within stack
- dropout_prob_theta: Dropout probability for theta layers
- activation: Activation function ('ReLU', 'Softplus', 'Tanh', 'SELU', etc.)
- learning_rate: Learning rate for optimizer
- max_steps: Maximum training steps
- batch_size: Batch size for training
- early_stop_patience_steps: Early stopping patience (None = no early stop)
- loss: Loss function ('MAE', 'MSE', 'MAPE', 'SMAPE')
- device: Compute device ('auto', 'cuda', 'mps', 'cpu')
- random_seed: Random seed for reproducibility
- validation_split: Proportion of data for validation

References:
    Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2019).
    "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting."
    ICLR 2020. https://arxiv.org/abs/1905.10437
"""

from typing import Optional, Literal, List, Union
from py_parsnip.model_spec import ModelSpec


def nbeats_reg(
    horizon: int = 1,
    input_size: Optional[int] = None,
    n_harmonics: int = 2,
    n_polynomials: int = 2,
    stack_types: List[str] = None,
    n_blocks: List[int] = None,
    mlp_units: List[List[int]] = None,
    share_weights_in_stack: bool = False,
    dropout_prob_theta: float = 0.0,
    activation: str = 'ReLU',
    learning_rate: float = 1e-3,
    max_steps: int = 1000,
    batch_size: int = 32,
    early_stop_patience_steps: Optional[int] = None,
    loss: Literal['MAE', 'MSE', 'MAPE', 'SMAPE'] = 'MAE',
    device: Literal['auto', 'cuda', 'mps', 'cpu'] = 'auto',
    random_seed: int = 1,
    validation_split: float = 0.2,
    engine: str = "neuralforecast",
) -> ModelSpec:
    """
    Create an NBEATS regression model specification.

    NBEATS is a deep learning architecture for univariate time series forecasting
    that uses interpretable basis expansion to decompose forecasts into trend and
    seasonality components.

    Parameters
    ----------
    horizon : int, default=1
        Forecast horizon (number of steps ahead).
    input_size : int, optional
        Lookback window size. If None, defaults to 2*horizon.
        Larger values capture longer-term patterns but increase computation.
    n_harmonics : int, default=2
        Number of Fourier terms for seasonality stack.
        Higher values capture more complex seasonal patterns.
    n_polynomials : int, default=2
        Polynomial degree for trend stack.
        Higher values capture more flexible trends (linear=1, quadratic=2, cubic=3, etc.).
    stack_types : list of str, optional
        Types of stacks to use. Default is ['trend', 'seasonality'].
        Options: 'trend', 'seasonality', 'generic'.
        - Use ['trend', 'seasonality'] for interpretable decomposition
        - Use ['generic'] for maximum flexibility (no interpretability)
        - Order matters: earlier stacks capture patterns first
    n_blocks : list of int, optional
        Number of blocks per stack. Default is [1, 1] (one block per stack).
        Must match length of stack_types. More blocks = more capacity.
    mlp_units : list of list of int, optional
        Hidden layer sizes for each stack. Default is [[512, 512], [512, 512]].
        Must match length of stack_types.
        Format: [[stack1_layer1, stack1_layer2, ...], [stack2_layer1, ...], ...]
    share_weights_in_stack : bool, default=False
        If True, blocks within a stack share weights (reduces parameters).
        If False, each block has independent weights (more capacity).
    dropout_prob_theta : float, default=0.0
        Dropout probability for theta layers (0.0 = no dropout).
        Range: [0.0, 1.0]. Higher values reduce overfitting but may hurt performance.
    activation : str, default='ReLU'
        Activation function for hidden layers.
        Options: 'ReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'PReLU', 'Sigmoid'.
    learning_rate : float, default=1e-3
        Learning rate for Adam optimizer.
    max_steps : int, default=1000
        Maximum number of training steps.
    batch_size : int, default=32
        Batch size for training.
    early_stop_patience_steps : int, optional
        Number of validation steps with no improvement before early stopping.
        If None, no early stopping is used (trains for max_steps).
    loss : {'MAE', 'MSE', 'MAPE', 'SMAPE'}, default='MAE'
        Loss function for training.
        - MAE: Mean Absolute Error (robust to outliers)
        - MSE: Mean Squared Error (penalizes large errors)
        - MAPE: Mean Absolute Percentage Error (scale-independent)
        - SMAPE: Symmetric MAPE (bounded, handles zeros better)
    device : {'auto', 'cuda', 'mps', 'cpu'}, default='auto'
        Compute device for training.
        - 'auto': Automatically select best available (CUDA > MPS > CPU)
        - 'cuda': NVIDIA GPU (requires CUDA-enabled PyTorch)
        - 'mps': Apple Silicon GPU (M1/M2, requires PyTorch >= 1.12)
        - 'cpu': CPU only
    random_seed : int, default=1
        Random seed for reproducibility.
    validation_split : float, default=0.2
        Proportion of training data to use for validation.
        Range: (0.0, 1.0). Validation is used for early stopping and monitoring.
    engine : str, default="neuralforecast"
        Computational engine (currently only "neuralforecast" supported).

    Returns
    -------
    ModelSpec
        Model specification for NBEATS regression.

    Examples
    --------
    >>> # Basic NBEATS with interpretable decomposition
    >>> spec = nbeats_reg(horizon=7, input_size=14)
    >>> fit = spec.fit(train_data, "sales ~ date")
    >>>
    >>> # Extract trend and seasonality components
    >>> outputs, coeffs, stats = fit.extract_outputs()
    >>> trend = outputs[outputs['component'] == 'trend']['value']
    >>> seasonality = outputs[outputs['component'] == 'seasonality']['value']
    >>>
    >>> # Generic NBEATS (maximum flexibility, no decomposition)
    >>> spec = nbeats_reg(
    ...     horizon=7,
    ...     stack_types=['generic'],
    ...     n_blocks=[3],
    ...     mlp_units=[[1024, 1024]]
    ... )
    >>>
    >>> # Deep NBEATS with multiple blocks
    >>> spec = nbeats_reg(
    ...     horizon=7,
    ...     stack_types=['trend', 'seasonality', 'generic'],
    ...     n_blocks=[2, 2, 2],
    ...     mlp_units=[[512, 512], [512, 512], [512, 512]]
    ... )
    >>>
    >>> # Long-horizon forecasting
    >>> spec = nbeats_reg(
    ...     horizon=30,
    ...     input_size=90,  # 3 months lookback
    ...     n_harmonics=4,  # Capture complex seasonality
    ...     n_polynomials=3,  # Cubic trend
    ...     max_steps=5000,  # More training
    ...     early_stop_patience_steps=200
    ... )
    >>>
    >>> # GPU training with regularization
    >>> spec = nbeats_reg(
    ...     horizon=7,
    ...     dropout_prob_theta=0.1,  # Add dropout
    ...     device='cuda',
    ...     batch_size=128,  # Larger batches on GPU
    ...     learning_rate=5e-4
    ... )

    Notes
    -----
    NBEATS Architecture:
        - Doubly residual: Each block outputs forecast + backcast
        - Backcast removes explained patterns from input (for next block)
        - Final forecast = sum of all block forecasts

    Interpretable vs Generic Stacks:
        - Interpretable stacks ('trend', 'seasonality'):
          * Constrained basis functions (polynomial, harmonic)
          * Components can be visualized separately
          * Better generalization with less data
          * Recommended for most use cases

        - Generic stacks:
          * Learned basis functions (no constraints)
          * Maximum flexibility
          * Requires more training data
          * Use when patterns are complex/unknown

    Exogenous Variables:
        NBEATS is designed for pure univariate forecasting and does NOT support
        exogenous variables in the standard architecture. If you need to include
        external regressors, consider:
        - NHITS model (supports exogenous variables)
        - TFT model (supports static/time-varying covariates)
        - Hybrid models (e.g., NBEATS for trend + XGBoost for exogenous effects)

    Computational Requirements:
        - Minimum: 100+ observations for training
        - Recommended: 500+ observations for complex patterns
        - GPU strongly recommended for horizon > 7 or max_steps > 1000
        - Memory usage scales with: input_size × mlp_units × n_blocks

    Hyperparameter Tuning Tips:
        - Start with defaults: stack_types=['trend', 'seasonality']
        - Increase n_harmonics if strong seasonality (daily→7, weekly→4, yearly→12)
        - Increase n_polynomials for non-linear trends (2→3)
        - Increase n_blocks for complex patterns (1→2 or 3)
        - Increase mlp_units for more capacity ([512,512]→[1024,1024])
        - Add dropout if overfitting (0.0→0.1 or 0.2)
        - Tune learning_rate if not converging (1e-3→5e-4 or 5e-3)

    See Also
    --------
    nhits_reg : NHITS model (multi-scale architecture, supports exogenous)
    prophet_reg : Prophet model (interpretable, handles holidays)
    arima_reg : ARIMA model (classical time series)
    """
    # Set defaults for stack configuration
    if stack_types is None:
        stack_types = ['trend', 'seasonality']

    if n_blocks is None:
        n_blocks = [1] * len(stack_types)

    if mlp_units is None:
        mlp_units = [[512, 512]] * len(stack_types)

    # Validate stack configuration
    if len(n_blocks) != len(stack_types):
        raise ValueError(
            f"n_blocks length ({len(n_blocks)}) must match stack_types length ({len(stack_types)}). "
            f"Got n_blocks={n_blocks}, stack_types={stack_types}"
        )

    if len(mlp_units) != len(stack_types):
        raise ValueError(
            f"mlp_units length ({len(mlp_units)}) must match stack_types length ({len(stack_types)}). "
            f"Got mlp_units={mlp_units}, stack_types={stack_types}"
        )

    # Validate stack types
    valid_stack_types = ['trend', 'seasonality', 'generic']
    invalid_stacks = [s for s in stack_types if s not in valid_stack_types]
    if invalid_stacks:
        raise ValueError(
            f"Invalid stack types: {invalid_stacks}. "
            f"Must be one of {valid_stack_types}."
        )

    # Set default input_size if not provided
    if input_size is None:
        input_size = 2 * horizon

    # Build arguments dictionary
    args = {
        "h": horizon,  # NeuralForecast uses 'h' for horizon
        "input_size": input_size,
        "n_harmonics": n_harmonics,
        "n_polynomials": n_polynomials,
        "stack_types": stack_types,
        "n_blocks": n_blocks,
        "mlp_units": mlp_units,
        "share_weights_in_stack": share_weights_in_stack,
        "dropout_prob_theta": dropout_prob_theta,
        "activation": activation,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "early_stop_patience_steps": early_stop_patience_steps,
        "loss": loss,
        "device": device,
        "random_seed": random_seed,
        "validation_split": validation_split,
    }

    return ModelSpec(
        model_type="nbeats_reg",
        engine=engine,
        mode="regression",
        args=args,
    )
