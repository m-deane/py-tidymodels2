"""
Test WorkflowSet.from_cross() with custom recipe names.

Tests dict and tuple formats for automatically using recipe variable names
in workflow IDs instead of generic prep_1, prep_2, etc.
"""

import pandas as pd
import numpy as np
import pytest
from py_workflowsets import WorkflowSet
from py_parsnip import linear_reg, rand_forest
from py_recipes import recipe


def test_from_cross_dict_format():
    """Test from_cross with dict format for custom preprocessor names."""
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'country': np.repeat(['USA', 'Germany'], 50),
        'x1': np.random.randn(100) * 10 + 50,
        'x2': np.random.randn(100) * 5 + 20,
        'y': np.random.randn(100) * 100 + 500
    })

    # Create named recipes
    rec_lags = recipe().step_lag(['x1'], lags=[2])
    rec_normalize = recipe().step_normalize(['x1', 'x2'])

    # Create WorkflowSet with dict format
    wf_set = WorkflowSet.from_cross(
        preproc={'rec_lags': rec_lags, 'rec_normalize': rec_normalize},
        models=[linear_reg(), rand_forest(trees=10).set_mode('regression')]
    )

    # Verify workflow IDs use custom names
    assert len(wf_set) == 4  # 2 recipes × 2 models

    wf_ids = list(wf_set.workflows.keys())

    # Should have rec_lags in IDs
    lags_ids = [wf_id for wf_id in wf_ids if 'rec_lags' in wf_id]
    assert len(lags_ids) == 2, f"Expected 2 rec_lags workflows, got {len(lags_ids)}"

    # Should have rec_normalize in IDs
    norm_ids = [wf_id for wf_id in wf_ids if 'rec_normalize' in wf_id]
    assert len(norm_ids) == 2, f"Expected 2 rec_normalize workflows, got {len(norm_ids)}"

    # Verify full ID format
    assert 'rec_lags_linear_reg_1' in wf_ids
    assert 'rec_lags_rand_forest_2' in wf_ids
    assert 'rec_normalize_linear_reg_1' in wf_ids
    assert 'rec_normalize_rand_forest_2' in wf_ids

    # Verify info DataFrame has correct option names
    assert 'rec_lags' in wf_set.info['option'].values
    assert 'rec_normalize' in wf_set.info['option'].values


def test_from_cross_tuple_format():
    """Test from_cross with tuple format for custom preprocessor names."""
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'x1': np.random.randn(100),
        'x2': np.random.randn(100),
        'y': np.random.randn(100)
    })

    # Create named recipes
    rec_pca = recipe().step_pca(num_comp=2)
    rec_poly = recipe().step_poly(['x1'], degree=2)

    # Create WorkflowSet with tuple format
    wf_set = WorkflowSet.from_cross(
        preproc=[('rec_pca', rec_pca), ('rec_poly', rec_poly)],
        models=[linear_reg()]
    )

    # Verify workflow IDs use custom names
    assert len(wf_set) == 2  # 2 recipes × 1 model

    wf_ids = list(wf_set.workflows.keys())

    # Verify full ID format
    assert 'rec_pca_linear_reg_1' in wf_ids
    assert 'rec_poly_linear_reg_1' in wf_ids

    # Verify info DataFrame
    assert 'rec_pca' in wf_set.info['option'].values
    assert 'rec_poly' in wf_set.info['option'].values


def test_from_cross_list_format_backward_compatible():
    """Test that list format still works with generic prep_N IDs (backward compatible)."""
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'x1': np.random.randn(100),
        'x2': np.random.randn(100),
        'y': np.random.randn(100)
    })

    # Create recipes as list (old approach)
    rec1 = recipe().step_normalize(['x1'])
    rec2 = recipe().step_scale(['x2'])

    # Create WorkflowSet with list format
    wf_set = WorkflowSet.from_cross(
        preproc=[rec1, rec2],
        models=[linear_reg()]
    )

    # Verify workflow IDs use generic prep_N names
    wf_ids = list(wf_set.workflows.keys())

    assert 'prep_1_linear_reg_1' in wf_ids
    assert 'prep_2_linear_reg_1' in wf_ids

    # Should NOT have custom names
    assert not any('rec' in wf_id and 'prep' not in wf_id for wf_id in wf_ids)


def test_from_cross_formulas_dict_format():
    """Test from_cross with dict format for formulas."""
    # Create WorkflowSet with formula dict
    wf_set = WorkflowSet.from_cross(
        preproc={'minimal': "y ~ x1", 'full': "y ~ x1 + x2"},
        models=[linear_reg()]
    )

    # Verify workflow IDs use custom names
    wf_ids = list(wf_set.workflows.keys())

    assert 'minimal_linear_reg_1' in wf_ids
    assert 'full_linear_reg_1' in wf_ids

    # Verify info DataFrame
    assert 'minimal' in wf_set.info['option'].values
    assert 'full' in wf_set.info['option'].values


def test_from_cross_mixed_formulas_recipes_dict():
    """Test from_cross with dict containing both formulas and recipes."""
    rec_lags = recipe().step_lag(['x1'], lags=[3])

    wf_set = WorkflowSet.from_cross(
        preproc={
            'formula_minimal': "y ~ x1",
            'rec_lags': rec_lags
        },
        models=[linear_reg()]
    )

    # Verify workflow IDs
    wf_ids = list(wf_set.workflows.keys())

    assert 'formula_minimal_linear_reg_1' in wf_ids
    assert 'rec_lags_linear_reg_1' in wf_ids

    # Verify preprocessor types in info
    info_df = wf_set.info

    formula_row = info_df[info_df['option'] == 'formula_minimal'].iloc[0]
    assert formula_row['preprocessor'] == 'formula'

    recipe_row = info_df[info_df['option'] == 'rec_lags'].iloc[0]
    assert recipe_row['preprocessor'] == 'recipe'


def test_from_cross_dict_preserves_order():
    """Test that dict format preserves key order (Python 3.7+)."""
    rec_a = recipe().step_normalize(['x1'])
    rec_b = recipe().step_scale(['x2'])
    rec_c = recipe().step_center(['x1', 'x2'])

    wf_set = WorkflowSet.from_cross(
        preproc={'rec_a': rec_a, 'rec_b': rec_b, 'rec_c': rec_c},
        models=[linear_reg()]
    )

    # Get workflow IDs in order
    wf_ids = list(wf_set.workflows.keys())

    # Extract prep names from IDs
    prep_names = [wf_id.split('_linear_reg')[0] for wf_id in wf_ids]

    # Verify order matches dict order
    assert prep_names == ['rec_a', 'rec_b', 'rec_c']


def test_from_cross_tuple_with_multiple_models():
    """Test tuple format with multiple models to verify ID generation."""
    rec_simple = recipe().step_normalize(['x1'])
    rec_complex = recipe().step_pca(num_comp=3)

    wf_set = WorkflowSet.from_cross(
        preproc=[('rec_simple', rec_simple), ('rec_complex', rec_complex)],
        models=[linear_reg(), rand_forest(trees=5).set_mode('regression')]
    )

    # Verify all 4 workflow IDs (2 recipes × 2 models)
    wf_ids = list(wf_set.workflows.keys())
    assert len(wf_ids) == 4

    expected_ids = [
        'rec_simple_linear_reg_1',
        'rec_simple_rand_forest_2',
        'rec_complex_linear_reg_1',
        'rec_complex_rand_forest_2'
    ]

    for expected_id in expected_ids:
        assert expected_id in wf_ids, f"Expected {expected_id} in workflow IDs"


def test_from_cross_ids_parameter_still_works():
    """Test that ids parameter still works with list format (backward compatible)."""
    rec1 = recipe().step_normalize(['x1'])
    rec2 = recipe().step_scale(['x2'])

    # Use ids parameter (old approach)
    wf_set = WorkflowSet.from_cross(
        preproc=[rec1, rec2],
        models=[linear_reg()],
        ids=['custom_1', 'custom_2']
    )

    wf_ids = list(wf_set.workflows.keys())

    assert 'custom_1_linear_reg_1' in wf_ids
    assert 'custom_2_linear_reg_1' in wf_ids


def test_from_cross_empty_list_handling():
    """Test that empty list creates empty WorkflowSet."""
    # Empty preprocessor list creates empty WorkflowSet
    wf_set = WorkflowSet.from_cross(
        preproc=[],
        models=[linear_reg()]
    )

    # Should have no workflows
    assert len(wf_set) == 0
    assert len(wf_set.workflows) == 0
    assert len(wf_set.info) == 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels')

    print("\n" + "="*60)
    print("Testing WorkflowSet.from_cross() Custom Names")
    print("="*60 + "\n")

    # Run tests
    test_from_cross_dict_format()
    print("✅ Dict format test passed")

    test_from_cross_tuple_format()
    print("✅ Tuple format test passed")

    test_from_cross_list_format_backward_compatible()
    print("✅ List format backward compatibility test passed")

    test_from_cross_formulas_dict_format()
    print("✅ Formulas dict format test passed")

    test_from_cross_mixed_formulas_recipes_dict()
    print("✅ Mixed formulas/recipes dict test passed")

    test_from_cross_dict_preserves_order()
    print("✅ Dict order preservation test passed")

    test_from_cross_tuple_with_multiple_models()
    print("✅ Tuple with multiple models test passed")

    test_from_cross_ids_parameter_still_works()
    print("✅ ids parameter backward compatibility test passed")

    test_from_cross_empty_list_handling()
    print("✅ Empty list handling test passed")

    print("\n" + "="*60)
    print("✅ ALL CUSTOM NAMES TESTS PASSED")
    print("="*60)
