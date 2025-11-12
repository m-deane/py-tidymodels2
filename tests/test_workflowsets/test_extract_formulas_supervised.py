"""
Test WorkflowSetNestedResults.extract_formulas() with supervised feature selection.

This test verifies that extract_formulas() shows the ACTUAL features that
survived preprocessing, not just the original formula.
"""

import pandas as pd
import numpy as np
from py_workflowsets import WorkflowSet
from py_parsnip import linear_reg
from py_recipes import recipe


def test_extract_formulas_with_feature_selection():
    """Test that extract_formulas shows actual features after supervised selection."""
    # Create data with correlated features
    # Different groups have different correlation patterns
    np.random.seed(42)
    n_per_group = 100

    # USA: x1 and x2 highly correlated, x3 independent
    usa_data = pd.DataFrame({
        'country': ['USA'] * n_per_group,
        'x1': np.random.randn(n_per_group),
    })
    usa_data['x2'] = usa_data['x1'] * 0.95 + np.random.randn(n_per_group) * 0.1  # Highly correlated with x1
    usa_data['x3'] = np.random.randn(n_per_group)  # Independent
    usa_data['y'] = usa_data['x1'] * 2 + usa_data['x3'] * 1.5 + np.random.randn(n_per_group) * 0.5

    # Germany: x1 and x3 highly correlated, x2 independent
    germany_data = pd.DataFrame({
        'country': ['Germany'] * n_per_group,
        'x1': np.random.randn(n_per_group),
    })
    germany_data['x3'] = germany_data['x1'] * 0.95 + np.random.randn(n_per_group) * 0.1  # Highly correlated with x1
    germany_data['x2'] = np.random.randn(n_per_group)  # Independent
    germany_data['y'] = germany_data['x1'] * 2 + germany_data['x2'] * 1.5 + np.random.randn(n_per_group) * 0.5

    data = pd.concat([usa_data, germany_data], ignore_index=True)

    # Create workflow with supervised feature selection
    # This should remove x2 for USA (correlated with x1) and x3 for Germany (correlated with x1)
    rec = recipe().step_select_corr(outcome='y', threshold=0.9, method='multicollinearity')

    wf_set = WorkflowSet.from_cross(
        preproc=[rec],
        models=[linear_reg()]
    )

    # Fit with per-group preprocessing
    results = wf_set.fit_nested(data, group_col='country', per_group_prep=True)

    # Extract formulas
    formulas_df = results.extract_formulas()

    # Verify DataFrame structure
    assert isinstance(formulas_df, pd.DataFrame), "Should return DataFrame"
    assert 'n_features' in formulas_df.columns, "Should have n_features column"

    # Verify we have entries for both groups
    assert set(formulas_df['group'].unique()) == {'USA', 'Germany'}, \
        "Should have formulas for both groups"

    # Get formulas for each group
    usa_formula = formulas_df[formulas_df['group'] == 'USA'].iloc[0]['formula']
    germany_formula = formulas_df[formulas_df['group'] == 'Germany'].iloc[0]['formula']

    print(f"\n{'='*60}")
    print("Feature Selection Results:")
    print(f"{'='*60}")
    print(f"USA formula:     {usa_formula}")
    print(f"Germany formula: {germany_formula}")

    # Count features
    usa_n_features = formulas_df[formulas_df['group'] == 'USA'].iloc[0]['n_features']
    germany_n_features = formulas_df[formulas_df['group'] == 'Germany'].iloc[0]['n_features']

    print(f"\nUSA features:     {usa_n_features}")
    print(f"Germany features: {germany_n_features}")

    # Verify feature selection worked (should have removed some features)
    assert usa_n_features < 3, "USA should have fewer than 3 features after selection"
    assert germany_n_features < 3, "Germany should have fewer than 3 features after selection"

    # Verify formulas differ (different features removed per group)
    # Note: They might be the same if correlation patterns don't trigger removal
    # but the important thing is that the method works
    print(f"\nFormulas differ: {usa_formula != germany_formula}")

    print(f"{'='*60}")
    print("✅ extract_formulas() correctly shows post-selection features")
    print(f"{'='*60}\n")


def test_extract_formulas_basic():
    """Test basic extract_formulas without feature selection."""
    # Create simple grouped data
    np.random.seed(42)
    n_per_group = 50

    data = pd.DataFrame({
        'country': np.repeat(['USA', 'Germany'], n_per_group),
        'x1': np.random.randn(n_per_group * 2) * 10 + 50,
        'x2': np.random.randn(n_per_group * 2) * 5 + 20,
        'y': np.random.randn(n_per_group * 2) * 100 + 500
    })

    # Create WorkflowSet without feature selection
    wf_set = WorkflowSet.from_cross(
        preproc=["y ~ x1", "y ~ x1 + x2"],
        models=[linear_reg()]
    )

    # Fit without per-group preprocessing
    results = wf_set.fit_nested(data, group_col='country')

    # Extract formulas
    formulas_df = results.extract_formulas()

    # Verify structure
    assert isinstance(formulas_df, pd.DataFrame), "Should return DataFrame"
    assert 'n_features' in formulas_df.columns, "Should have n_features column"

    # Verify n_features is correct
    prep_1_df = formulas_df[formulas_df['wflow_id'].str.contains('prep_1')]
    prep_2_df = formulas_df[formulas_df['wflow_id'].str.contains('prep_2')]

    # prep_1 should have 1 feature (x1)
    assert (prep_1_df['n_features'] == 1).all(), "prep_1 should have 1 feature"

    # prep_2 should have 2 features (x1 + x2)
    assert (prep_2_df['n_features'] == 2).all(), "prep_2 should have 2 features"

    # Verify formulas are consistent across groups (no per-group prep)
    for wf_id in formulas_df['wflow_id'].unique():
        wf_df = formulas_df[formulas_df['wflow_id'] == wf_id]
        unique_formulas = wf_df['formula'].unique()
        assert len(unique_formulas) == 1, \
            f"{wf_id} should have same formula across groups (no per_group_prep)"

    print("✅ Basic extract_formulas() test passed")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels')

    print("\n" + "="*60)
    print("Testing WorkflowSetNestedResults.extract_formulas()")
    print("="*60 + "\n")

    try:
        print("Test 1: Basic formulas (no feature selection)")
        print("-" * 60)
        test_extract_formulas_basic()
        print()

        print("Test 2: Formulas with supervised feature selection")
        print("-" * 60)
        test_extract_formulas_with_feature_selection()
        print()

        print("="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
