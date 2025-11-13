"""
Tests for WorkflowSet conformal prediction integration (Phase 5).

Tests the compare_conformal() method for comparing conformal intervals
across multiple workflows.
"""

import pytest
import pandas as pd
import numpy as np
from py_workflowsets import WorkflowSet
from py_parsnip import linear_reg
from py_workflows import workflow


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n = 300

    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'x3': np.random.randn(n)
    })
    data['y'] = 10 + 2*data['x1'] + 3*data['x2'] + np.random.randn(n) * 0.5

    return data


def test_compare_conformal_basic(sample_data):
    """Test basic conformal comparison across workflows."""
    # Create workflow set with different formulas
    wf_set = WorkflowSet.from_cross(
        preproc=["y ~ x1", "y ~ x1 + x2", "y ~ x1 + x2 + x3"],
        models=[linear_reg()]
    )

    # Compare conformal intervals
    comparison = wf_set.compare_conformal(sample_data, alpha=0.05, method='split')

    # Check output structure
    assert isinstance(comparison, pd.DataFrame)
    assert len(comparison) == 3  # 3 workflows

    # Check required columns
    required_cols = [
        'wflow_id', 'model', 'preprocessor', 'conf_method',
        'avg_interval_width', 'median_interval_width',
        'coverage', 'n_predictions'
    ]
    for col in required_cols:
        assert col in comparison.columns, f"Missing column: {col}"

    # Check all workflows succeeded
    assert comparison['conf_method'].eq('split').all()
    assert comparison['avg_interval_width'].notna().all()
    assert comparison['coverage'].notna().all()

    # Check sorted by interval width
    assert comparison['avg_interval_width'].is_monotonic_increasing or \
           comparison['avg_interval_width'].iloc[0] <= comparison['avg_interval_width'].iloc[-1]


def test_compare_conformal_different_models(sample_data):
    """Test conformal comparison across different model types."""
    # Create workflow set with different models
    wf_set = WorkflowSet.from_cross(
        preproc=["y ~ x1 + x2"],
        models=[
            linear_reg(),
            linear_reg(penalty=0.1, mixture=1.0)  # Lasso
        ]
    )

    # Compare conformal intervals
    comparison = wf_set.compare_conformal(sample_data, alpha=0.05, method='split')

    # Check output
    assert len(comparison) == 2
    assert set(comparison['model'].unique()) == {'linear_reg'}
    assert comparison['avg_interval_width'].notna().all()


def test_compare_conformal_auto_method(sample_data):
    """Test conformal comparison with auto method selection."""
    wf_set = WorkflowSet.from_cross(
        preproc=["y ~ x1", "y ~ x1 + x2"],
        models=[linear_reg()]
    )

    # Use auto method selection
    comparison = wf_set.compare_conformal(sample_data, alpha=0.05, method='auto')

    # Check method was auto-selected (should be split or cv+ for this data size)
    assert comparison['conf_method'].iloc[0] in ['split', 'cv+', 'jackknife+']


def test_compare_conformal_coverage_calculation(sample_data):
    """Test that coverage is calculated correctly."""
    wf_set = WorkflowSet.from_cross(
        preproc=["y ~ x1 + x2"],
        models=[linear_reg()]
    )

    comparison = wf_set.compare_conformal(sample_data, alpha=0.05, method='split')

    # Coverage should be calculated (data has actuals)
    assert comparison['coverage'].notna().all()

    # Coverage should be reasonable (target 95% for alpha=0.05)
    coverage = comparison['coverage'].iloc[0]
    assert 0.85 <= coverage <= 1.0, f"Coverage {coverage:.1%} out of expected range"


def test_compare_conformal_interval_width_comparison(sample_data):
    """Test that interval widths are comparable across workflows."""
    # Create workflows with different preprocessing
    wf_set = WorkflowSet.from_cross(
        preproc=[
            "y ~ x1",           # Simple model
            "y ~ x1 + x2 + x3"  # Full model
        ],
        models=[linear_reg()]
    )

    comparison = wf_set.compare_conformal(sample_data, alpha=0.05, method='split')

    # Both should have valid interval widths
    assert len(comparison) == 2
    assert comparison['avg_interval_width'].notna().all()

    # Widths should be positive
    assert (comparison['avg_interval_width'] > 0).all()


def test_compare_conformal_with_interaction_term(sample_data):
    """Test conformal comparison including interaction term."""
    wf_set = WorkflowSet.from_cross(
        preproc=[
            "y ~ x1 + x2",
            "y ~ x1 + x2 + I(x1*x2)"  # With interaction
        ],
        models=[linear_reg()]
    )

    comparison = wf_set.compare_conformal(sample_data, alpha=0.05, method='split')

    assert len(comparison) == 2
    assert comparison['conf_method'].eq('split').all()
    assert comparison['avg_interval_width'].notna().all()


def test_compare_conformal_n_predictions(sample_data):
    """Test that n_predictions is correct."""
    wf_set = WorkflowSet.from_cross(
        preproc=["y ~ x1 + x2"],
        models=[linear_reg()]
    )

    comparison = wf_set.compare_conformal(sample_data, alpha=0.05, method='split')

    # Number of predictions should equal number of rows in data
    assert comparison['n_predictions'].iloc[0] == len(sample_data)


def test_compare_conformal_sorted_output(sample_data):
    """Test that output is sorted by interval width (tightest first)."""
    # Create workflows that should have different interval widths
    wf_set = WorkflowSet.from_cross(
        preproc=[
            "y ~ x1",
            "y ~ x1 + x2",
            "y ~ x1 + x2 + x3"
        ],
        models=[linear_reg()]
    )

    comparison = wf_set.compare_conformal(sample_data, alpha=0.05, method='split')

    # Check sorted ascending by avg_interval_width
    widths = comparison['avg_interval_width'].values
    assert all(widths[i] <= widths[i+1] for i in range(len(widths)-1))


def test_compare_conformal_workflow_info_preserved(sample_data):
    """Test that workflow info (model, preprocessor) is preserved."""
    wf_set = WorkflowSet.from_cross(
        preproc=["y ~ x1", "y ~ x1 + x2"],
        models=[linear_reg()]
    )

    comparison = wf_set.compare_conformal(sample_data, alpha=0.05, method='split')

    # Check that workflow info matches
    assert set(comparison['model'].unique()) == {'linear_reg'}
    assert set(comparison['preprocessor'].unique()) == {'formula'}
