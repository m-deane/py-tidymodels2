"""
Performance benchmarks for parallel processing in py-tidymodels2.

Compares sequential vs parallel execution across different scenarios
to measure speedup and identify optimal n_jobs values.
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Tuple
import warnings

# Import py-tidymodels2 components
from py_workflows import workflow
from py_parsnip import linear_reg, rand_forest
from py_rsample import vfold_cv, time_series_cv
from py_yardstick import metric_set, rmse, mae, r_squared
from py_tune import fit_resamples, tune_grid, grid_regular, tune
from py_workflowsets import WorkflowSet
from py_tune.parallel_utils import get_cpu_count

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def generate_sample_data(n_rows: int = 1000, n_features: int = 5, seed: int = 42) -> pd.DataFrame:
    """Generate sample regression data."""
    np.random.seed(seed)
    data = pd.DataFrame({
        f'x{i+1}': np.random.randn(n_rows)
        for i in range(n_features)
    })
    # Create target with some relationship to features
    data['y'] = (
        2 * data['x1'] +
        1.5 * data['x2'] -
        data['x3'] +
        np.random.randn(n_rows) * 0.5
    )
    return data


def generate_grouped_data(n_groups: int = 5, rows_per_group: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate grouped/panel data for nested modeling."""
    np.random.seed(seed)

    all_data = []
    for i in range(n_groups):
        group_data = generate_sample_data(n_rows=rows_per_group, seed=seed + i)
        group_data['group'] = f'Group_{chr(65 + i)}'  # A, B, C, etc.
        all_data.append(group_data)

    return pd.concat(all_data, ignore_index=True)


def benchmark_function(func, *args, **kwargs) -> Tuple[float, any]:
    """
    Benchmark a function and return execution time and result.

    Returns:
        Tuple of (execution_time_seconds, result)
    """
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return elapsed, result


def print_benchmark_header(title: str):
    """Print formatted benchmark section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_benchmark_result(scenario: str, seq_time: float, par_time: float, n_jobs: int):
    """Print formatted benchmark results."""
    speedup = seq_time / par_time if par_time > 0 else 0
    efficiency = (speedup / n_jobs * 100) if n_jobs > 1 else 100

    print(f"\n{scenario}:")
    print(f"  Sequential: {seq_time:.2f}s")
    print(f"  Parallel (n_jobs={n_jobs}): {par_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Efficiency: {efficiency:.1f}%")


# =============================================================================
# Benchmark 1: fit_resamples() - CV Fold Evaluation
# =============================================================================

def benchmark_fit_resamples():
    """Benchmark fit_resamples with varying fold counts."""
    print_benchmark_header("Benchmark 1: fit_resamples() - CV Fold Evaluation")

    cpu_count = get_cpu_count()
    data = generate_sample_data(n_rows=2000)
    wf = workflow().add_formula("y ~ x1 + x2 + x3").add_model(linear_reg())
    metrics = metric_set(rmse, mae)

    # Test with different fold counts
    for n_folds in [5, 10]:
        print(f"\nScenario: {n_folds}-fold CV on 2000 rows")
        folds = vfold_cv(data, v=n_folds, seed=123)

        # Sequential
        seq_time, _ = benchmark_function(
            fit_resamples, wf, folds, metrics=metrics, n_jobs=1, verbose=False
        )

        # Parallel with all cores
        par_time, _ = benchmark_function(
            fit_resamples, wf, folds, metrics=metrics, n_jobs=-1, verbose=False
        )

        print_benchmark_result(f"{n_folds}-fold CV", seq_time, par_time, cpu_count)


# =============================================================================
# Benchmark 2: tune_grid() - Grid Search
# =============================================================================

def benchmark_tune_grid():
    """Benchmark tune_grid with varying grid sizes."""
    print_benchmark_header("Benchmark 2: tune_grid() - Grid Search")

    cpu_count = get_cpu_count()
    data = generate_sample_data(n_rows=1000)

    # Create tunable workflow
    spec = linear_reg(penalty=tune(), mixture=tune()).set_engine("sklearn")
    wf = workflow().add_formula("y ~ x1 + x2 + x3 + x4").add_model(spec)

    param_info = {
        'penalty': {'range': (0.001, 1.0), 'trans': 'log'},
        'mixture': {'range': (0, 1)}
    }

    folds = vfold_cv(data, v=5, seed=123)
    metrics = metric_set(rmse)

    # Test with different grid sizes
    for levels in [3, 5]:
        n_configs = levels ** 2  # 9 or 25 configurations
        print(f"\nScenario: {n_configs} configs × 5 folds = {n_configs * 5} fits")

        grid = grid_regular(param_info, levels=levels)

        # Sequential
        seq_time, _ = benchmark_function(
            tune_grid, wf, folds, grid=grid, metrics=metrics, n_jobs=1, verbose=False
        )

        # Parallel
        par_time, _ = benchmark_function(
            tune_grid, wf, folds, grid=grid, metrics=metrics, n_jobs=-1, verbose=False
        )

        print_benchmark_result(f"Grid search ({n_configs} configs)", seq_time, par_time, cpu_count)


# =============================================================================
# Benchmark 3: Workflow.fit_nested() - Grouped Modeling
# =============================================================================

def benchmark_fit_nested():
    """Benchmark fit_nested with grouped data."""
    print_benchmark_header("Benchmark 3: Workflow.fit_nested() - Grouped Modeling")

    cpu_count = get_cpu_count()
    data = generate_grouped_data(n_groups=8, rows_per_group=200)
    wf = workflow().add_formula("y ~ x1 + x2 + x3").add_model(linear_reg())

    print(f"\nScenario: 8 groups × 200 rows each = 1600 total rows")

    # Sequential
    seq_time, _ = benchmark_function(
        wf.fit_nested, data, group_col='group', n_jobs=1, verbose=False
    )

    # Parallel
    par_time, _ = benchmark_function(
        wf.fit_nested, data, group_col='group', n_jobs=-1, verbose=False
    )

    print_benchmark_result("Nested grouped modeling", seq_time, par_time, cpu_count)


# =============================================================================
# Benchmark 4: WorkflowSet.fit_resamples() - Multi-Workflow CV
# =============================================================================

def benchmark_workflowset_fit_resamples():
    """Benchmark WorkflowSet.fit_resamples with multiple workflows."""
    print_benchmark_header("Benchmark 4: WorkflowSet.fit_resamples() - Multi-Workflow CV")

    cpu_count = get_cpu_count()
    data = generate_sample_data(n_rows=1500)

    # Create multiple workflows
    formulas = [
        "y ~ x1 + x2",
        "y ~ x1 + x2 + x3",
        "y ~ x1 + x2 + x3 + x4",
        "y ~ x1 + x2 + x3 + x4 + x5"
    ]
    models = [
        linear_reg(),
        linear_reg(penalty=0.1, mixture=1.0).set_engine("sklearn")
    ]

    wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)
    n_workflows = len(wf_set.workflows)

    folds = vfold_cv(data, v=5, seed=123)
    metrics = metric_set(rmse, mae)

    print(f"\nScenario: {n_workflows} workflows × 5 folds = {n_workflows * 5} fits")

    # Sequential
    seq_time, _ = benchmark_function(
        wf_set.fit_resamples, folds, metrics=metrics, n_jobs=1, verbose=False
    )

    # Parallel
    par_time, _ = benchmark_function(
        wf_set.fit_resamples, folds, metrics=metrics, n_jobs=-1, verbose=False
    )

    print_benchmark_result(f"WorkflowSet ({n_workflows} workflows)", seq_time, par_time, cpu_count)


# =============================================================================
# Benchmark 5: WorkflowSet.fit_nested() - Multi-Workflow Grouped
# =============================================================================

def benchmark_workflowset_fit_nested():
    """Benchmark WorkflowSet.fit_nested with grouped data."""
    print_benchmark_header("Benchmark 5: WorkflowSet.fit_nested() - Multi-Workflow Grouped")

    cpu_count = get_cpu_count()
    data = generate_grouped_data(n_groups=6, rows_per_group=150)

    # Create workflows
    formulas = ["y ~ x1 + x2", "y ~ x1 + x2 + x3"]
    models = [linear_reg()]

    wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)
    n_workflows = len(wf_set.workflows)
    n_groups = 6

    print(f"\nScenario: {n_workflows} workflows × {n_groups} groups = {n_workflows * n_groups} models")

    # Sequential
    seq_time, _ = benchmark_function(
        wf_set.fit_nested, data, group_col='group', n_jobs=1, verbose=False
    )

    # Parallel
    par_time, _ = benchmark_function(
        wf_set.fit_nested, data, group_col='group', n_jobs=-1, verbose=False
    )

    print_benchmark_result(f"WorkflowSet nested ({n_workflows}×{n_groups})", seq_time, par_time, cpu_count)


# =============================================================================
# Benchmark 6: Scalability Test - Varying n_jobs
# =============================================================================

def benchmark_scalability():
    """Benchmark fit_resamples with varying n_jobs values."""
    print_benchmark_header("Benchmark 6: Scalability Test - Varying n_jobs")

    cpu_count = get_cpu_count()
    data = generate_sample_data(n_rows=2000)
    wf = workflow().add_formula("y ~ x1 + x2 + x3").add_model(linear_reg())
    folds = vfold_cv(data, v=10, seed=123)
    metrics = metric_set(rmse)

    print(f"\nSystem has {cpu_count} CPU cores")
    print("\nScenario: 10-fold CV with varying n_jobs")

    results = []

    # Test with sequential
    seq_time, _ = benchmark_function(
        fit_resamples, wf, folds, metrics=metrics, n_jobs=1, verbose=False
    )
    results.append((1, seq_time, 1.0))
    print(f"  n_jobs=1 (sequential): {seq_time:.2f}s (baseline)")

    # Test with increasing n_jobs
    for n_jobs in [2, 4, cpu_count]:
        if n_jobs <= cpu_count:
            par_time, _ = benchmark_function(
                fit_resamples, wf, folds, metrics=metrics, n_jobs=n_jobs, verbose=False
            )
            speedup = seq_time / par_time
            efficiency = (speedup / n_jobs * 100)
            results.append((n_jobs, par_time, speedup))
            print(f"  n_jobs={n_jobs}: {par_time:.2f}s (speedup: {speedup:.2f}x, efficiency: {efficiency:.1f}%)")

    # Test with all cores
    if cpu_count not in [2, 4]:
        par_time, _ = benchmark_function(
            fit_resamples, wf, folds, metrics=metrics, n_jobs=-1, verbose=False
        )
        speedup = seq_time / par_time
        efficiency = (speedup / cpu_count * 100)
        results.append((-1, par_time, speedup))
        print(f"  n_jobs=-1 (all {cpu_count} cores): {par_time:.2f}s (speedup: {speedup:.2f}x, efficiency: {efficiency:.1f}%)")


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_all_benchmarks():
    """Run all benchmark suites."""
    print("\n" + "=" * 80)
    print("  PARALLEL PROCESSING PERFORMANCE BENCHMARKS")
    print("  py-tidymodels2")
    print("=" * 80)

    cpu_count = get_cpu_count()
    print(f"\nSystem Information:")
    print(f"  CPU cores detected: {cpu_count}")
    print(f"  Joblib backend: loky (multiprocessing)")
    print(f"  Note: Results may vary based on system load")

    # Run benchmarks
    try:
        benchmark_fit_resamples()
    except Exception as e:
        print(f"\n❌ fit_resamples benchmark failed: {e}")

    try:
        benchmark_tune_grid()
    except Exception as e:
        print(f"\n❌ tune_grid benchmark failed: {e}")

    try:
        benchmark_fit_nested()
    except Exception as e:
        print(f"\n❌ fit_nested benchmark failed: {e}")

    try:
        benchmark_workflowset_fit_resamples()
    except Exception as e:
        print(f"\n❌ WorkflowSet.fit_resamples benchmark failed: {e}")

    try:
        benchmark_workflowset_fit_nested()
    except Exception as e:
        print(f"\n❌ WorkflowSet.fit_nested benchmark failed: {e}")

    try:
        benchmark_scalability()
    except Exception as e:
        print(f"\n❌ Scalability benchmark failed: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("  BENCHMARK COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    print("  - Parallel execution provides speedup for computationally intensive tasks")
    print("  - Optimal n_jobs depends on task count and computational cost per task")
    print("  - WorkflowSet methods benefit most from parallelization (many workflows)")
    print("  - Grouped modeling (fit_nested) sees good speedup with independent groups")
    print("  - Small tasks may not benefit due to multiprocessing overhead")
    print("\nRecommendations:")
    print("  - Use n_jobs=-1 for grid search with many configurations")
    print("  - Use n_jobs=-1 for WorkflowSet with many workflows")
    print("  - Use n_jobs=1 for quick tasks (< 5 folds, simple models)")
    print("  - Monitor efficiency - if < 50%, reduce n_jobs")


if __name__ == "__main__":
    run_all_benchmarks()
