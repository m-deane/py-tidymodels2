"""
Test WorkflowSetNestedResults.extract_formulas() method
"""

import pandas as pd
import numpy as np
from py_workflowsets import WorkflowSet
from py_parsnip import linear_reg, rand_forest


def test_extract_formulas():
    """Test that extract_formulas returns dict of formulas."""
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
    formulas = results.extract_formulas()

    # Verify it's a dict
    assert isinstance(formulas, dict), "extract_formulas should return a dict"

    # Verify it has entries for all workflows
    expected_wf_ids = list(wf_set.workflows.keys())
    assert len(formulas) == len(expected_wf_ids), \
        f"Should have {len(expected_wf_ids)} formulas, got {len(formulas)}"

    # Verify all workflow IDs are present
    for wf_id in expected_wf_ids:
        assert wf_id in formulas, f"Missing formula for {wf_id}"

    # Verify formulas are strings
    for wf_id, formula in formulas.items():
        assert isinstance(formula, str), f"Formula for {wf_id} should be string, got {type(formula)}"
        assert '~' in formula, f"Formula for {wf_id} should contain '~', got: {formula}"

    # Verify correct formulas
    prep_1_formulas = {k: v for k, v in formulas.items() if k.startswith('prep_1')}
    prep_2_formulas = {k: v for k, v in formulas.items() if k.startswith('prep_2')}

    # All prep_1 workflows should have "y ~ x1"
    for wf_id, formula in prep_1_formulas.items():
        assert 'x1' in formula, f"{wf_id} should include x1"
        assert 'x2' not in formula, f"{wf_id} should not include x2 (formula: {formula})"

    # All prep_2 workflows should have "y ~ x1 + x2"
    for wf_id, formula in prep_2_formulas.items():
        assert 'x1' in formula, f"{wf_id} should include x1"
        assert 'x2' in formula, f"{wf_id} should include x2"

    print(f"✅ extract_formulas() returned {len(formulas)} formulas")
    for wf_id, formula in formulas.items():
        print(f"  {wf_id}: {formula}")


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
