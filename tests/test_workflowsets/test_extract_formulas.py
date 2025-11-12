"""
Test WorkflowSetNestedResults.extract_formulas() method
"""

import pandas as pd
import numpy as np
from py_workflowsets import WorkflowSet
from py_parsnip import linear_reg, rand_forest


def test_extract_formulas():
    """Test that extract_formulas returns DataFrame of formulas per workflow per group."""
    # Create grouped data
    np.random.seed(42)
    n_per_group = 50

    data = pd.DataFrame({
        'country': np.repeat(['USA', 'Germany'], n_per_group),
        'x1': np.random.randn(n_per_group * 2) * 10 + 50,
        'x2': np.random.randn(n_per_group * 2) * 5 + 20,
        'y': np.random.randn(n_per_group * 2) * 100 + 500
    })

    # Create WorkflowSet with multiple workflows
    wf_set = WorkflowSet.from_cross(
        preproc=["y ~ x1", "y ~ x1 + x2"],
        models=[linear_reg(), rand_forest(trees=10).set_mode('regression')]
    )

    # Fit all workflows on all groups
    results = wf_set.fit_nested(data, group_col='country')

    # Extract formulas
    formulas_df = results.extract_formulas()

    # Verify it's a DataFrame
    assert isinstance(formulas_df, pd.DataFrame), "extract_formulas should return a DataFrame"

    # Verify columns
    expected_cols = {'wflow_id', 'group', 'formula', 'n_features', 'preprocessor', 'model'}
    assert set(formulas_df.columns) == expected_cols, \
        f"Expected columns {expected_cols}, got {set(formulas_df.columns)}"

    # Verify we have entries for all workflow-group combinations
    n_workflows = len(wf_set.workflows)
    n_groups = data['country'].nunique()
    expected_rows = n_workflows * n_groups
    assert len(formulas_df) == expected_rows, \
        f"Should have {expected_rows} rows (workflows × groups), got {len(formulas_df)}"

    # Verify all workflow IDs are present
    unique_wf_ids = formulas_df['wflow_id'].unique()
    expected_wf_ids = list(wf_set.workflows.keys())
    assert set(unique_wf_ids) == set(expected_wf_ids), \
        f"Missing workflow IDs: {set(expected_wf_ids) - set(unique_wf_ids)}"

    # Verify all groups are present
    unique_groups = formulas_df['group'].unique()
    expected_groups = ['USA', 'Germany']
    assert set(unique_groups) == set(expected_groups), \
        f"Expected groups {expected_groups}, got {list(unique_groups)}"

    # Verify formulas are strings with '~'
    assert formulas_df['formula'].apply(lambda x: isinstance(x, str)).all(), \
        "All formulas should be strings"
    assert formulas_df['formula'].apply(lambda x: '~' in x).all(), \
        "All formulas should contain '~'"

    # Verify correct formulas for each prep
    prep_1_df = formulas_df[formulas_df['wflow_id'].str.startswith('prep_1')]
    prep_2_df = formulas_df[formulas_df['wflow_id'].str.startswith('prep_2')]

    # All prep_1 workflows should have "y ~ x1"
    for _, row in prep_1_df.iterrows():
        formula = row['formula']
        assert 'x1' in formula, f"{row['wflow_id']} should include x1"
        assert 'x2' not in formula, f"{row['wflow_id']} should not include x2 (formula: {formula})"

    # All prep_2 workflows should have "y ~ x1 + x2"
    for _, row in prep_2_df.iterrows():
        formula = row['formula']
        assert 'x1' in formula, f"{row['wflow_id']} should include x1"
        assert 'x2' in formula, f"{row['wflow_id']} should include x2"

    # Verify formulas are consistent across groups (when per_group_prep=False)
    formula_uniqueness = formulas_df.groupby('wflow_id')['formula'].nunique()
    assert (formula_uniqueness == 1).all(), \
        "With per_group_prep=False, formulas should be same across groups"

    # Verify n_features column has correct counts
    assert (formulas_df['n_features'] >= 0).all(), "n_features should be non-negative"

    # For prep_1 (y ~ x1), should have 1 feature
    prep_1_rows = formulas_df[formulas_df['wflow_id'].str.startswith('prep_1')]
    assert (prep_1_rows['n_features'] == 1).all(), \
        f"prep_1 should have 1 feature, got: {prep_1_rows['n_features'].unique()}"

    # For prep_2 (y ~ x1 + x2), should have 2 features
    prep_2_rows = formulas_df[formulas_df['wflow_id'].str.startswith('prep_2')]
    assert (prep_2_rows['n_features'] == 2).all(), \
        f"prep_2 should have 2 features, got: {prep_2_rows['n_features'].unique()}"

    print(f"✅ extract_formulas() returned {len(formulas_df)} rows")
    print("\nSample:")
    print(formulas_df.head(10).to_string())


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels')

    print("\n" + "="*60)
    print("Testing WorkflowSetNestedResults.extract_formulas()")
    print("="*60 + "\n")

    try:
        test_extract_formulas()
        print("\n" + "="*60)
        print("✅ TEST PASSED")
        print("="*60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
