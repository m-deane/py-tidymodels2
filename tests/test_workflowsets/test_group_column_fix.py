"""
Test to verify library-wide group column fix.

Tests that extract_outputs() correctly sets both:
1. Specific group column (e.g., 'country', 'store_id')
2. Generic 'group' column

Both should contain the same actual group values, not 'global'.
"""

import pandas as pd
import numpy as np
import pytest
from py_workflows import Workflow
from py_workflowsets import WorkflowSet
from py_parsnip import linear_reg, rand_forest


@pytest.fixture
def grouped_data():
    """Create simple grouped dataset."""
    np.random.seed(42)
    n_per_group = 50

    data = pd.DataFrame({
        'country': np.repeat(['USA', 'Germany', 'Japan'], n_per_group),
        'x1': np.random.randn(n_per_group * 3) * 10 + 50,
        'x2': np.random.randn(n_per_group * 3) * 5 + 20,
        'y': np.random.randn(n_per_group * 3) * 100 + 500
    })

    return data


def test_nested_workflow_fit_group_column(grouped_data):
    """Test NestedWorkflowFit.extract_outputs() sets group column correctly."""
    # Create workflow
    wf = Workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())

    # Fit nested (per-group models)
    fit = wf.fit_nested(grouped_data, group_col='country')

    # Extract outputs
    outputs, coefs, stats = fit.extract_outputs()

    # Verify both columns exist
    assert 'country' in outputs.columns, "Specific column 'country' should exist"
    assert 'group' in outputs.columns, "Generic 'group' column should exist"

    # Verify they match
    assert (outputs['country'] == outputs['group']).all(), \
        "Specific 'country' and generic 'group' columns should match"

    # Verify actual group values (not 'global')
    unique_groups = outputs['group'].unique()
    assert 'global' not in unique_groups, \
        f"Group column should not contain 'global', got: {unique_groups}"
    assert set(unique_groups) == {'USA', 'Germany', 'Japan'}, \
        f"Group column should contain actual country names, got: {unique_groups}"

    # Same checks for coefficients
    assert 'country' in coefs.columns, "Coefs should have 'country' column"
    assert 'group' in coefs.columns, "Coefs should have 'group' column"
    assert (coefs['country'] == coefs['group']).all(), \
        "Coefs: 'country' and 'group' should match"
    assert 'global' not in coefs['group'].unique(), \
        "Coefs: group should not be 'global'"

    # Same checks for stats
    assert 'country' in stats.columns, "Stats should have 'country' column"
    assert 'group' in stats.columns, "Stats should have 'group' column"
    assert (stats['country'] == stats['group']).all(), \
        "Stats: 'country' and 'group' should match"
    assert 'global' not in stats['group'].unique(), \
        "Stats: group should not be 'global'"

    print("✅ NestedWorkflowFit.extract_outputs() group column fix verified")


def test_workflowset_nested_results_group_column(grouped_data):
    """Test WorkflowSetNestedResults.extract_outputs() sets group column correctly."""
    # Create WorkflowSet with multiple workflows
    wf_set = WorkflowSet.from_cross(
        preproc=["y ~ x1", "y ~ x1 + x2"],
        models=[linear_reg(), rand_forest(trees=10).set_mode('regression')]
    )

    # Fit all workflows on all groups
    results = wf_set.fit_nested(grouped_data, group_col='country')

    # Extract outputs
    outputs, coefs, stats = results.extract_outputs()

    # Verify both columns exist
    assert 'country' in outputs.columns, "Outputs should have 'country' column"
    assert 'group' in outputs.columns, "Outputs should have 'group' column"
    assert 'wflow_id' in outputs.columns, "Outputs should have 'wflow_id' column"

    # Verify they match
    assert (outputs['country'] == outputs['group']).all(), \
        "Outputs: 'country' and 'group' columns should match"

    # Verify actual group values (not 'global')
    unique_groups = outputs['group'].unique()
    assert 'global' not in unique_groups, \
        f"Outputs: group should not contain 'global', got: {unique_groups}"
    assert set(unique_groups) == {'USA', 'Germany', 'Japan'}, \
        f"Outputs: group should contain actual country names, got: {unique_groups}"

    # Same checks for coefficients
    assert 'country' in coefs.columns, "Coefs should have 'country' column"
    assert 'group' in coefs.columns, "Coefs should have 'group' column"
    assert (coefs['country'] == coefs['group']).all(), \
        "Coefs: 'country' and 'group' should match"
    assert 'global' not in coefs['group'].unique(), \
        "Coefs: group should not be 'global'"

    # Stats from WorkflowSetNestedResults uses collect_metrics() which only has 'group' column
    # This is expected behavior - the specific column name is only in outputs/coefs
    assert 'group' in stats.columns, "Stats should have 'group' column"

    # Verify 'group' column has actual values (not 'global')
    assert 'global' not in stats['group'].unique(), \
        "Stats: group should not be 'global'"
    assert set(stats['group'].unique()) == {'USA', 'Germany', 'Japan'}, \
        f"Stats: group should contain actual country names, got: {stats['group'].unique()}"

    print("✅ WorkflowSetNestedResults.extract_outputs() group column fix verified")


def test_group_column_filtering(grouped_data):
    """Test that filtering by 'group' column works as expected."""
    # Create workflow
    wf = Workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())

    # Fit nested
    fit = wf.fit_nested(grouped_data, group_col='country')

    # Extract outputs
    outputs, _, _ = fit.extract_outputs()

    # Test filtering by generic 'group' column
    usa_outputs = outputs[outputs['group'] == 'USA']
    germany_outputs = outputs[outputs['group'] == 'Germany']
    japan_outputs = outputs[outputs['group'] == 'Japan']

    # Verify filtering works
    assert len(usa_outputs) > 0, "Should find USA data"
    assert len(germany_outputs) > 0, "Should find Germany data"
    assert len(japan_outputs) > 0, "Should find Japan data"

    # Verify filtered data is correct
    assert (usa_outputs['country'] == 'USA').all(), \
        "USA filter should only return USA data"
    assert (germany_outputs['country'] == 'Germany').all(), \
        "Germany filter should only return Germany data"
    assert (japan_outputs['country'] == 'Japan').all(), \
        "Japan filter should only return Japan data"

    # Verify total equals original
    total_filtered = len(usa_outputs) + len(germany_outputs) + len(japan_outputs)
    assert total_filtered == len(outputs), \
        "Filtered data should equal original data"

    print("✅ Group column filtering works correctly")


def test_different_group_column_names(grouped_data):
    """Test fix works with different group column names."""
    # Rename 'country' to 'store_id' for variety
    data = grouped_data.rename(columns={'country': 'store_id'})

    # Create workflow
    wf = Workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())

    # Fit nested with different group column name
    fit = wf.fit_nested(data, group_col='store_id')

    # Extract outputs
    outputs, coefs, stats = fit.extract_outputs()

    # Verify both columns exist
    assert 'store_id' in outputs.columns, "Should have 'store_id' column"
    assert 'group' in outputs.columns, "Should have 'group' column"

    # Verify they match
    assert (outputs['store_id'] == outputs['group']).all(), \
        "store_id and group should match"

    # Verify not 'global'
    assert 'global' not in outputs['group'].unique(), \
        "Group should not be 'global'"

    print("✅ Fix works with different group column names")


if __name__ == "__main__":
    # Run tests manually
    import sys
    sys.path.insert(0, '/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels')

    # Create test data
    data = grouped_data()

    print("\n" + "="*60)
    print("Testing Library-Wide Group Column Fix")
    print("="*60 + "\n")

    try:
        test_nested_workflow_fit_group_column(data)
        print()
        test_workflowset_nested_results_group_column(data)
        print()
        test_group_column_filtering(data)
        print()
        test_different_group_column_names(data)
        print()
        print("="*60)
        print("✅ ALL TESTS PASSED - Group column fix working correctly!")
        print("="*60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
