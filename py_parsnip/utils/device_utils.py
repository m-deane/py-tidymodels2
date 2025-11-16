"""
Device management utilities for deep learning models.

This module provides GPU/CPU device detection and management for PyTorch-based
deep learning engines (NeuralForecast). Supports CUDA, MPS (Apple Silicon), and CPU.
"""

from typing import List, Optional, Dict, Any
import warnings
from contextlib import contextmanager


def detect_available_devices() -> List[str]:
    """
    Detect all available compute devices on the system.

    Checks for CUDA GPUs, Apple MPS (Metal Performance Shaders), and CPU.
    Returns devices in priority order: CUDA > MPS > CPU.

    Returns
    -------
    list of str
        List of available device names in priority order.
        Possible values: ['cuda', 'mps', 'cpu']
        Always returns at least ['cpu'].

    Examples
    --------
    >>> devices = detect_available_devices()
    >>> print(devices)
    ['cuda', 'cpu']  # On NVIDIA GPU system
    >>> print(devices)
    ['mps', 'cpu']   # On Apple Silicon Mac
    >>> print(devices)
    ['cpu']          # On CPU-only system
    """
    devices = []

    try:
        import torch

        # Check for CUDA
        if torch.cuda.is_available():
            devices.append('cuda')

        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append('mps')

        # CPU is always available
        devices.append('cpu')

    except ImportError:
        # PyTorch not installed, default to CPU
        warnings.warn(
            "PyTorch not installed. Deep learning models will not be available. "
            "Install with: pip install torch",
            ImportWarning
        )
        devices.append('cpu')

    return devices


def get_optimal_device(prefer_gpu: bool = True) -> str:
    """
    Select the optimal compute device for model training/inference.

    Automatically selects the best available device based on hardware:
    - CUDA GPU (NVIDIA) - highest priority if available
    - MPS (Apple Silicon) - second priority
    - CPU - fallback if no GPU available

    Parameters
    ----------
    prefer_gpu : bool, default=True
        If True, prefer GPU over CPU when available.
        If False, always return 'cpu' (useful for debugging or memory constraints).

    Returns
    -------
    str
        Name of the optimal device: 'cuda', 'mps', or 'cpu'.

    Examples
    --------
    >>> # On NVIDIA GPU system
    >>> device = get_optimal_device()
    >>> print(device)
    'cuda'
    >>>
    >>> # Force CPU usage
    >>> device = get_optimal_device(prefer_gpu=False)
    >>> print(device)
    'cpu'
    >>>
    >>> # On Apple Silicon Mac
    >>> device = get_optimal_device()
    >>> print(device)
    'mps'
    """
    if not prefer_gpu:
        return 'cpu'

    devices = detect_available_devices()

    # Return first available device (already in priority order)
    # Priority: cuda > mps > cpu
    return devices[0]


def check_gpu_memory(min_gb: float = 1.0, device: str = 'cuda') -> bool:
    """
    Check if GPU has sufficient available memory.

    Verifies that the specified GPU device has at least the minimum required
    memory available. Useful for preventing OOM errors during training.

    Parameters
    ----------
    min_gb : float, default=1.0
        Minimum required GPU memory in gigabytes.
    device : str, default='cuda'
        Device to check. Must be 'cuda' or 'mps'.
        Note: MPS memory checking is limited on Apple Silicon.

    Returns
    -------
    bool
        True if sufficient memory is available, False otherwise.
        Always returns True for CPU device (no memory constraints).

    Warnings
    --------
    UserWarning
        If GPU memory is below the minimum threshold.

    Examples
    --------
    >>> # Check if GPU has at least 2GB available
    >>> if check_gpu_memory(min_gb=2.0):
    ...     device = 'cuda'
    ... else:
    ...     device = 'cpu'
    ...     print("Insufficient GPU memory, falling back to CPU")
    >>>
    >>> # Always returns True for CPU
    >>> check_gpu_memory(device='cpu')
    True
    """
    if device == 'cpu':
        return True  # No memory constraints for CPU

    try:
        import torch

        if device == 'cuda':
            if not torch.cuda.is_available():
                warnings.warn(
                    f"CUDA device requested but not available. Use device='cpu' instead.",
                    UserWarning
                )
                return False

            # Get available memory in GB
            gpu_props = torch.cuda.get_device_properties(0)
            total_memory_gb = gpu_props.total_memory / (1024 ** 3)

            # Get currently allocated memory
            allocated_memory_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
            available_memory_gb = total_memory_gb - allocated_memory_gb

            if available_memory_gb < min_gb:
                warnings.warn(
                    f"GPU memory low: {available_memory_gb:.2f}GB available, "
                    f"{min_gb:.2f}GB required. Consider using device='cpu'.",
                    UserWarning
                )
                return False

            return True

        elif device == 'mps':
            # MPS doesn't provide detailed memory information
            # Apple Silicon memory is shared, so check system memory instead
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                warnings.warn(
                    f"MPS device requested but not available. Use device='cpu' instead.",
                    UserWarning
                )
                return False

            # For MPS, we can't easily check memory, so assume it's available
            # User should monitor system memory separately
            return True

        else:
            raise ValueError(f"Unknown device: {device}. Must be 'cuda', 'mps', or 'cpu'.")

    except ImportError:
        warnings.warn(
            "PyTorch not installed. Cannot check GPU memory. Defaulting to CPU.",
            ImportWarning
        )
        return False


def get_device_info() -> Dict[str, Any]:
    """
    Get detailed information about available compute devices.

    Returns comprehensive device information including:
    - Available devices
    - Optimal device
    - GPU memory (if CUDA available)
    - Device counts

    Returns
    -------
    dict
        Dictionary containing device information:
        - 'available_devices': List of available device names
        - 'optimal_device': Recommended device for model training
        - 'cuda_available': Whether CUDA is available
        - 'cuda_device_count': Number of CUDA devices (0 if none)
        - 'cuda_device_name': Name of primary CUDA device (None if unavailable)
        - 'cuda_memory_gb': Total memory of primary CUDA device in GB (None if unavailable)
        - 'mps_available': Whether MPS is available
        - 'torch_version': PyTorch version (None if not installed)

    Examples
    --------
    >>> info = get_device_info()
    >>> print(info)
    {
        'available_devices': ['cuda', 'cpu'],
        'optimal_device': 'cuda',
        'cuda_available': True,
        'cuda_device_count': 1,
        'cuda_device_name': 'NVIDIA GeForce RTX 3090',
        'cuda_memory_gb': 24.0,
        'mps_available': False,
        'torch_version': '2.0.1'
    }
    """
    info = {
        'available_devices': detect_available_devices(),
        'optimal_device': get_optimal_device(),
        'cuda_available': False,
        'cuda_device_count': 0,
        'cuda_device_name': None,
        'cuda_memory_gb': None,
        'mps_available': False,
        'torch_version': None,
    }

    try:
        import torch

        info['torch_version'] = torch.__version__

        # CUDA information
        if torch.cuda.is_available():
            info['cuda_available'] = True
            info['cuda_device_count'] = torch.cuda.device_count()

            # Get primary device info
            gpu_props = torch.cuda.get_device_properties(0)
            info['cuda_device_name'] = gpu_props.name
            info['cuda_memory_gb'] = round(gpu_props.total_memory / (1024 ** 3), 2)

        # MPS information
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info['mps_available'] = True

    except ImportError:
        pass  # Leave defaults

    return info


@contextmanager
def device_context(device: str):
    """
    Context manager for safe device operations.

    Ensures proper cleanup and error handling when working with GPU devices.
    Automatically clears GPU cache on exit if using CUDA.

    Parameters
    ----------
    device : str
        Device to use within context: 'cuda', 'mps', or 'cpu'.

    Yields
    ------
    str
        The device string for use within the context.

    Examples
    --------
    >>> with device_context('cuda') as device:
    ...     # Train model on GPU
    ...     model = train_on_device(device)
    ...     # GPU cache automatically cleared on exit
    >>>
    >>> # Automatic fallback to CPU if GPU unavailable
    >>> requested_device = 'cuda'
    >>> with device_context(requested_device) as actual_device:
    ...     if actual_device != requested_device:
    ...         print(f"Fell back to {actual_device}")
    ...     model = train_on_device(actual_device)
    """
    try:
        import torch

        # Validate device availability
        if device == 'cuda' and not torch.cuda.is_available():
            warnings.warn(
                f"CUDA requested but not available. Falling back to CPU.",
                UserWarning
            )
            device = 'cpu'

        elif device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            warnings.warn(
                f"MPS requested but not available. Falling back to CPU.",
                UserWarning
            )
            device = 'cpu'

        yield device

    finally:
        # Cleanup: clear GPU cache if using CUDA
        try:
            import torch
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def validate_device(device: str) -> str:
    """
    Validate and sanitize device string.

    Checks if requested device is available and returns a valid device string.
    Falls back to CPU if requested device is unavailable.

    Parameters
    ----------
    device : str
        Requested device: 'cuda', 'mps', 'cpu', or 'auto'.
        Use 'auto' for automatic device selection.

    Returns
    -------
    str
        Validated device string: 'cuda', 'mps', or 'cpu'.

    Raises
    ------
    ValueError
        If device string is invalid (not in ['cuda', 'mps', 'cpu', 'auto']).

    Examples
    --------
    >>> # Auto-select optimal device
    >>> device = validate_device('auto')
    >>> print(device)
    'cuda'  # On GPU system
    >>>
    >>> # Validate specific device
    >>> device = validate_device('cuda')
    >>> print(device)
    'cuda'  # If CUDA available
    'cpu'   # If CUDA not available (with warning)
    """
    valid_devices = ['cuda', 'mps', 'cpu', 'auto']

    if device not in valid_devices:
        raise ValueError(
            f"Invalid device: '{device}'. Must be one of {valid_devices}."
        )

    if device == 'auto':
        return get_optimal_device()

    # Validate device availability
    available = detect_available_devices()

    if device not in available:
        warnings.warn(
            f"Device '{device}' not available. Falling back to CPU. "
            f"Available devices: {available}",
            UserWarning
        )
        return 'cpu'

    return device
