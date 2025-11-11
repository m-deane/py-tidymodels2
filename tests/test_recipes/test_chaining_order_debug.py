"""
Debug test: Understand the order of operations for chained steps.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from py_recipes import recipe
from py_workflows import workflow
from py_parsnip import linear_reg


def test_simple_chain():
    """Simplest possible chain to understand order."""

    # Create simple data
    np.random.seed(42)
    n = 100

    data = pd.DataFrame({
        'group': ['A'] * 50 + ['B'] * 50,
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'x3': np.random.randn(n),
        'x4': np.random.randn(n),
        'x5': np.random.randn(n),
    })
    data['y'] = data['x1'] * 2 + data['x2'] * 1.5 + np.random.randn(n) * 0.1

    train = data.iloc[:80]
    test = data.iloc[80:]

    print("Train data columns:", train.columns.tolist())
    print("Test data columns:", test.columns.tolist())

    # Try different step orders

    # Order 1: Normalize BEFORE selection (WRONG?)
    print("\n" + "="*70)
    print("ORDER 1: Normalize → Select (normalize sees all, select filters)")
    print("="*70)

    rec1 = (
        recipe()
        .step_normalize()  # Should normalize x1-x5
        .step_filter_rf_importance(outcome='y', top_n=2, trees=30)  # Select 2
    )

    wf1 = workflow().add_recipe(rec1).add_model(linear_reg())

    try:
        fit1 = wf1.fit_nested(train, group_col='group', per_group_prep=True)
        print("  ✓ Fit succeeded")

        # Check what columns the prepared recipe has
        group_a_prep = fit1.group_fits['A'].pre
        print(f"  Group A prepared recipe has {len(group_a_prep.steps)} steps")

        # Try to predict
        preds1 = fit1.predict(test)
        print(f"  ✓ Predict succeeded, shape: {preds1.shape}")

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Order 2: Select BEFORE normalize (RIGHT?)
    print("\n" + "="*70)
    print("ORDER 2: Select → Normalize (select filters first, normalize sees filtered)")
    print("="*70)

    rec2 = (
        recipe()
        .step_filter_rf_importance(outcome='y', top_n=2, trees=30)  # Select 2 first
        .step_normalize()  # Normalize only selected columns
    )

    wf2 = workflow().add_recipe(rec2).add_model(linear_reg())

    try:
        fit2 = wf2.fit_nested(train, group_col='group', per_group_prep=True)
        print("  ✓ Fit succeeded")

        preds2 = fit2.predict(test)
        print(f"  ✓ Predict succeeded, shape: {preds2.shape}")

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Order 3: Normalize specific columns, then select
    print("\n" + "="*70)
    print("ORDER 3: Normalize(specific) → Select")
    print("="*70)

    rec3 = (
        recipe()
        .step_normalize(['x1', 'x2', 'x3', 'x4', 'x5'])  # Explicit columns
        .step_filter_rf_importance(outcome='y', top_n=2, trees=30)
    )

    wf3 = workflow().add_recipe(rec3).add_model(linear_reg())

    try:
        fit3 = wf3.fit_nested(train, group_col='group', per_group_prep=True)
        print("  ✓ Fit succeeded")

        preds3 = fit3.predict(test)
        print(f"  ✓ Predict succeeded, shape: {preds3.shape}")

    except Exception as e:
        print(f"  ✗ FAILED: {e}")


if __name__ == "__main__":
    test_simple_chain()
