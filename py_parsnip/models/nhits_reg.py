"""
NHITS (Neural Hierarchical Interpolation for Time Series) model specification.

NHITS is a deep learning model for time series forecasting that uses:
- Multi-rate input processing with hierarchical interpolation
- Stack-based architecture for multi-scale pattern capture
- Efficient long-horizon forecasting

The model decomposes the time series at different frequencies using
a hierarchical structure with expressiveness pooling and interpolation.

Key Features:
- Handles complex seasonal patterns
- Efficient for long-horizon forecasting
- Multi-scale temporal pattern capture
- Fast training with GPU support

Parameters (tidymodels naming):
- horizon: Forecast horizon (number of steps ahead)
- input_size: Lookback window (auto-calculated if None)
- n_freq_downsample: Downsampling rates for multi-scale processing
- n_blocks: Number of blocks per stack
- mlp_units: MLP hidden units per stack
- n_pool_kernel_size: Pooling kernel sizes
- n_theta_hidden: Theta network hidden units
- pooling_mode: Pooling operation ('MaxPool1d' or 'AvgPool1d')
- interpolation_mode: Interpolation method ('linear', 'nearest', 'cubic')
- dropout_prob_theta: Dropout probability for theta networks
- activation: Activation function ('ReLU', 'Tanh', etc.)
- learning_rate: Optimizer learning rate
- max_steps: Maximum training steps
- batch_size: Training batch size
- early_stop_patience_steps: Early stopping patience (None disables)
- loss: Loss function ('MAE', 'MSE', 'MAPE', 'SMAPE')
- device: Compute device ('auto', 'cpu', 'cuda', 'mps')
- random_seed: Random seed for reproducibility
- validation_split: Validation data proportion

Reference:
    Challu, C., Olivares, K. G., Oreshkin, B. N., Ramirez, F. G., Canseco, M. M., & Dubrawski, A. (2022).
    N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting.
    arXiv preprint arXiv:2201.12886.
"""

from typing import Optional, List, Literal
from py_parsnip.model_spec import ModelSpec


def nhits_reg(
    horizon: int = 1,
    input_size: Optional[int] = None,
    n_freq_downsample: List[int] = None,
    n_blocks: Optional[List[int]] = None,
    mlp_units: List[List[int]] = None,
    n_pool_kernel_size: Optional[List[int]] = None,
    n_theta_hidden: Optional[List[int]] = None,
    pooling_mode: str = 'MaxPool1d',
    interpolation_mode: str = 'linear',
    dropout_prob_theta: float = 0.0,
    activation: str = 'ReLU',
    learning_rate: float = 1e-3,
    max_steps: int = 1000,
    batch_size: int = 32,
    early_stop_patience_steps: Optional[int] = None,
    loss: Literal['MAE', 'MSE', 'MAPE', 'SMAPE'] = 'MAE',
    device: Literal['auto', 'cpu', 'cuda', 'mps'] = 'auto',
    random_seed: int = 1,
    validation_split: float = 0.2,
    engine: str = "neuralforecast",
) -> ModelSpec:
    """
    Create an NHITS (Neural Hierarchical Interpolation for Time Series) model specification.

    NHITS uses a hierarchical architecture with multi-rate input processing and
    interpolation to efficiently capture patterns at different temporal scales.
    It's particularly effective for long-horizon forecasting.

    Parameters
    ----------
    horizon : int, default=1
        Forecast horizon (number of steps ahead to predict).
        Must be positive integer.
    input_size : int, optional
        Lookback window size (number of historical steps used for prediction).
        If None, automatically set to max(2 * horizon, 7 * frequency).
    n_freq_downsample : list of int, optional
        Downsampling rates for hierarchical processing.
        Default is [8, 4, 1] for 3-stack architecture.
        Each value represents the pooling factor for that stack.
    n_blocks : list of int, optional
        Number of blocks per stack. Must match length of n_freq_downsample.
        Default is [1, 1, 1]. More blocks increase model capacity.
    mlp_units : list of list of int, optional
        MLP hidden units for each stack's blocks.
        Default is [[512, 512], [512, 512], [512, 512]].
        Inner lists define hidden layer sizes for each stack.
    n_pool_kernel_size : list of int, optional
        Pooling kernel sizes for each stack.
        Default is [8, 4, 1] (matches n_freq_downsample).
        Must match length of n_freq_downsample.
    n_theta_hidden : list of int, optional
        Hidden units for theta networks (basis expansion).
        Default is [256, 256, 256].
        Must match length of n_freq_downsample.
    pooling_mode : str, default='MaxPool1d'
        Pooling operation: 'MaxPool1d' or 'AvgPool1d'.
        MaxPool captures peaks, AvgPool smooths.
    interpolation_mode : str, default='linear'
        Interpolation method for upsampling: 'linear', 'nearest', or 'cubic'.
        Linear is recommended for most time series.
    dropout_prob_theta : float, default=0.0
        Dropout probability for theta networks (0.0 to 1.0).
        Higher values reduce overfitting but may hurt performance.
    activation : str, default='ReLU'
        Activation function: 'ReLU', 'Tanh', 'LeakyReLU', 'GELU', etc.
        ReLU is standard; Tanh/GELU for smoother activations.
    learning_rate : float, default=1e-3
        Optimizer learning rate. Typical range: 1e-4 to 1e-2.
        Lower for stable convergence, higher for faster training.
    max_steps : int, default=1000
        Maximum training steps (epochs). More steps improve fit but risk overfitting.
        Use early_stop_patience_steps to prevent overfitting.
    batch_size : int, default=32
        Training batch size. Larger batches (64-128) are more stable;
        smaller batches (16-32) may generalize better.
    early_stop_patience_steps : int, optional
        Early stopping patience (number of steps without improvement).
        If None, no early stopping (trains for max_steps).
        Recommended: 50-100 for stability.
    loss : {'MAE', 'MSE', 'MAPE', 'SMAPE'}, default='MAE'
        Loss function for training:
        - 'MAE': Mean Absolute Error (robust to outliers)
        - 'MSE': Mean Squared Error (penalizes large errors)
        - 'MAPE': Mean Absolute Percentage Error (relative errors)
        - 'SMAPE': Symmetric MAPE (bounded, scale-independent)
    device : {'auto', 'cpu', 'cuda', 'mps'}, default='auto'
        Compute device:
        - 'auto': Automatically select GPU if available
        - 'cpu': Force CPU (slower but always available)
        - 'cuda': NVIDIA GPU (requires CUDA-enabled PyTorch)
        - 'mps': Apple Silicon GPU (M1/M2 Macs)
    random_seed : int, default=1
        Random seed for reproducibility. Use same seed for consistent results.
    validation_split : float, default=0.2
        Proportion of training data for validation (0.0 to 1.0).
        Used for early stopping and hyperparameter tuning.
    engine : str, default='neuralforecast'
        Computational engine. Currently only 'neuralforecast' supported.

    Returns
    -------
    ModelSpec
        NHITS model specification ready for fitting.

    Examples
    --------
    >>> from py_parsnip import nhits_reg
    >>> import pandas as pd
    >>>
    >>> # Basic NHITS model with defaults (horizon=1)
    >>> spec = nhits_reg()
    >>> fit = spec.fit(train_data, 'sales ~ date')
    >>> predictions = fit.predict(test_data)
    >>>
    >>> # Multi-step forecasting (7-day ahead)
    >>> spec = nhits_reg(horizon=7, input_size=28)
    >>> fit = spec.fit(train_data, 'sales ~ date')
    >>>
    >>> # With exogenous variables (price, promotions)
    >>> spec = nhits_reg(horizon=7)
    >>> fit = spec.fit(train_data, 'sales ~ price + promo + date')
    >>>
    >>> # Custom architecture (deeper model)
    >>> spec = nhits_reg(
    ...     horizon=14,
    ...     n_freq_downsample=[16, 8, 4, 1],  # 4 stacks
    ...     mlp_units=[[1024, 512], [1024, 512], [512, 512], [512, 512]],
    ...     n_blocks=[2, 2, 2, 2],  # 2 blocks per stack
    ...     max_steps=2000,
    ...     early_stop_patience_steps=100
    ... )
    >>>
    >>> # Long-horizon forecasting with GPU
    >>> spec = nhits_reg(
    ...     horizon=28,
    ...     input_size=84,  # 3x horizon
    ...     max_steps=3000,
    ...     batch_size=64,
    ...     device='cuda',  # Use GPU
    ...     early_stop_patience_steps=100
    ... )
    >>>
    >>> # Regularized model (prevent overfitting)
    >>> spec = nhits_reg(
    ...     horizon=7,
    ...     dropout_prob_theta=0.3,  # 30% dropout
    ...     learning_rate=5e-4,  # Lower LR
    ...     validation_split=0.3  # More validation data
    ... )
    >>>
    >>> # CPU-only training (no GPU)
    >>> spec = nhits_reg(horizon=7, device='cpu')

    Notes
    -----
    **Architecture:**
        NHITS uses a hierarchical stack architecture where each stack processes
        the input at a different temporal resolution. The stacks are combined
        via interpolation to produce the final forecast.

    **Multi-scale Processing:**
        n_freq_downsample=[8, 4, 1] means:
        - Stack 1: Processes input at 1/8 resolution (captures long-term trends)
        - Stack 2: Processes input at 1/4 resolution (medium-term patterns)
        - Stack 3: Processes input at full resolution (short-term details)

    **GPU Requirements:**
        - CUDA: Requires PyTorch with CUDA support (NVIDIA GPU)
        - MPS: Requires PyTorch 1.12+ (Apple Silicon M1/M2)
        - Automatic fallback to CPU if GPU unavailable

    **Memory Considerations:**
        - Larger batch_size requires more memory
        - Deeper architectures (more stacks/blocks) use more memory
        - Long input_size increases memory usage
        - Use device='cpu' if GPU memory insufficient

    **Hyperparameter Tuning:**
        Key parameters to tune:
        1. horizon: Match your forecasting need
        2. input_size: Typically 2-5x horizon
        3. learning_rate: Start with 1e-3, adjust based on convergence
        4. max_steps: Increase until validation loss plateaus
        5. early_stop_patience_steps: 50-100 is typical

    **Data Requirements:**
        - Minimum: input_size + horizon observations
        - Recommended: 500+ observations for stable training
        - Validation split reduces effective training data

    See Also
    --------
    nbeats_reg : Alternative deep learning model (non-hierarchical)
    tft_reg : Transformer-based forecasting model
    prophet_reg : Additive time series model (interpretable)
    arima_reg : Statistical time series model
    """
    # Set default values for architecture parameters
    if n_freq_downsample is None:
        n_freq_downsample = [8, 4, 1]

    if n_blocks is None:
        n_blocks = [1] * len(n_freq_downsample)

    if mlp_units is None:
        mlp_units = [[512, 512]] * len(n_freq_downsample)

    if n_pool_kernel_size is None:
        n_pool_kernel_size = n_freq_downsample.copy()

    if n_theta_hidden is None:
        n_theta_hidden = [256] * len(n_freq_downsample)

    # Validate horizon
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")

    # Validate input_size if provided
    if input_size is not None and input_size <= 0:
        raise ValueError(f"input_size must be positive, got {input_size}")

    # Validate validation_split
    if not (0 < validation_split < 1):
        raise ValueError(
            f"validation_split must be between 0 and 1, got {validation_split}"
        )

    # Validate architectural consistency
    n_stacks = len(n_freq_downsample)
    if len(n_blocks) != n_stacks:
        raise ValueError(
            f"n_blocks length ({len(n_blocks)}) must match "
            f"n_freq_downsample length ({n_stacks})"
        )
    if len(mlp_units) != n_stacks:
        raise ValueError(
            f"mlp_units length ({len(mlp_units)}) must match "
            f"n_freq_downsample length ({n_stacks})"
        )
    if len(n_pool_kernel_size) != n_stacks:
        raise ValueError(
            f"n_pool_kernel_size length ({len(n_pool_kernel_size)}) must match "
            f"n_freq_downsample length ({n_stacks})"
        )
    if len(n_theta_hidden) != n_stacks:
        raise ValueError(
            f"n_theta_hidden length ({len(n_theta_hidden)}) must match "
            f"n_freq_downsample length ({n_stacks})"
        )

    # Store parameters
    args = {
        "h": horizon,  # NeuralForecast uses 'h' for horizon
        "input_size": input_size,
        "n_freq_downsample": n_freq_downsample,
        "n_blocks": n_blocks,
        "mlp_units": mlp_units,
        "n_pool_kernel_size": n_pool_kernel_size,
        "n_theta_hidden": n_theta_hidden,
        "pooling_mode": pooling_mode,
        "interpolation_mode": interpolation_mode,
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
        model_type="nhits_reg",
        engine=engine,
        mode="regression",
        args=args,
    )
