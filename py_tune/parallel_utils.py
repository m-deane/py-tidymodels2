"""
Parallel processing utilities for py-tidymodels2.

Provides helpers for CPU detection, n_jobs validation, and progress tracking.
"""

import os
import warnings
from typing import Optional


def get_cpu_count() -> int:
    """
    Get the number of available CPU cores.

    Returns:
        Number of CPU cores (minimum 1 even if detection fails)
    """
    try:
        cpu_count = os.cpu_count()
        return cpu_count if cpu_count is not None else 1
    except Exception:
        return 1


def validate_n_jobs(n_jobs: Optional[int], task_count: int, verbose: bool = False) -> int:
    """
    Validate and resolve n_jobs parameter with warnings.

    Args:
        n_jobs: Number of parallel jobs
            - None or 1: Sequential execution
            - -1: Use all CPU cores
            - N > 1: Use N cores
        task_count: Number of tasks to execute
        verbose: If True, print validation messages

    Returns:
        Resolved n_jobs value (positive integer or 1 for sequential)

    Warnings:
        - If n_jobs > available CPU cores
        - If n_jobs > task_count (inefficient)
        - If n_jobs=-1 but only 1-2 cores available (limited benefit)
    """
    if n_jobs is None or n_jobs == 1:
        return 1

    cpu_count = get_cpu_count()

    # Resolve -1 to all cores
    if n_jobs == -1:
        effective_jobs = cpu_count

        # Warn if limited cores available
        if cpu_count <= 2:
            warnings.warn(
                f"n_jobs=-1 requested but only {cpu_count} CPU core(s) available. "
                f"Parallel execution may not provide significant speedup.",
                UserWarning,
                stacklevel=3
            )
    elif n_jobs < -1:
        # scikit-learn convention: -2 means all cores - 1, etc.
        effective_jobs = max(1, cpu_count + n_jobs + 1)
        if verbose:
            print(f"Resolved n_jobs={n_jobs} to {effective_jobs} cores (total: {cpu_count})")
    else:
        effective_jobs = n_jobs

    # Warn if requesting more cores than available
    if effective_jobs > cpu_count:
        warnings.warn(
            f"n_jobs={effective_jobs} requested but only {cpu_count} CPU core(s) available. "
            f"This may cause oversubscription and reduced performance. "
            f"Consider using n_jobs={cpu_count} or n_jobs=-1.",
            UserWarning,
            stacklevel=3
        )

    # Warn if more workers than tasks (inefficient)
    if effective_jobs > task_count:
        warnings.warn(
            f"n_jobs={effective_jobs} is greater than task count ({task_count}). "
            f"Only {task_count} worker(s) will be utilized. "
            f"Consider using n_jobs={min(task_count, cpu_count)} for efficiency.",
            UserWarning,
            stacklevel=3
        )

    return effective_jobs


def get_joblib_backend() -> str:
    """
    Get the appropriate joblib backend for the current platform.

    Returns:
        Backend name ('loky' for all platforms, 'threading' for fallback)

    Notes:
        - 'loky' is the default and works on Windows, macOS, and Linux
        - 'loky' uses multiprocessing with robust process management
        - 'threading' is available as fallback but limited by GIL
    """
    # loky is the most robust backend and works on all platforms
    # It handles serialization better than standard multiprocessing
    return 'loky'


def check_windows_compatibility(verbose: bool = False) -> bool:
    """
    Check if parallel execution is compatible with Windows.

    Args:
        verbose: If True, print compatibility information

    Returns:
        True if compatible (always True with loky backend)

    Notes:
        - Windows requires 'loky' or 'threading' backend
        - 'fork' backend is not available on Windows
        - Objects must be picklable for process-based parallelism
    """
    import sys
    is_windows = sys.platform.startswith('win')

    if verbose and is_windows:
        print("Running on Windows: using 'loky' backend for multiprocessing")
        print("Note: All objects passed to parallel workers must be picklable")

    # loky backend is always compatible
    return True


def format_parallel_info(n_jobs: int, task_count: int, task_type: str) -> str:
    """
    Format informative message about parallel execution.

    Args:
        n_jobs: Resolved number of jobs
        task_count: Number of tasks
        task_type: Description of tasks (e.g., "CV folds", "workflows")

    Returns:
        Formatted info string

    Example:
        >>> format_parallel_info(4, 10, "CV folds")
        'Processing 10 CV folds using 4 cores (loky backend)'
    """
    backend = get_joblib_backend()
    cpu_count = get_cpu_count()

    if n_jobs == 1:
        return f"Processing {task_count} {task_type} (sequential)"
    else:
        core_info = f"{n_jobs}/{cpu_count} cores" if n_jobs < cpu_count else f"{n_jobs} cores"
        return f"Processing {task_count} {task_type} using {core_info} ({backend} backend)"
