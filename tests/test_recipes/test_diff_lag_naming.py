"""
Verification test for step_diff column naming with "lag" included.

This test demonstrates the improved column naming that includes "lag"
to make it clearer what the differencing parameters mean.
"""

import pandas as pd
from py_recipes import recipe


def test_diff_column_naming_examples():
    """Demonstrate the new step_diff column naming convention."""

    # Create sample data
    data = pd.DataFrame({
        'sales': [100, 102, 105, 103, 108, 110, 107, 109],
        'price': [50, 51, 52, 51, 53, 54, 53, 55]
    })

    print("\n" + "="*70)
    print("step_diff COLUMN NAMING - WITH 'LAG' INCLUDED")
    print("="*70)

    # Example 1: Default differencing (lag=1, differences=1)
    print("\n1. Default Differencing (lag=1, differences=1):")
    print("-" * 70)
    rec1 = recipe().step_diff(columns=["sales"])
    transformed1 = rec1.prep(data).bake(data)
    print(f"   Original column: 'sales'")
    print(f"   Transformed column: 'sales_diff_lag_1'")
    print(f"   Columns in data: {transformed1.columns.tolist()}")
    assert "sales_diff_lag_1" in transformed1.columns
    print("   ✓ Column name includes 'lag' for clarity")

    # Example 2: Custom lag (lag=2)
    print("\n2. Custom Lag (lag=2, differences=1):")
    print("-" * 70)
    rec2 = recipe().step_diff(columns=["sales"], lag=2)
    transformed2 = rec2.prep(data).bake(data)
    print(f"   Original column: 'sales'")
    print(f"   Transformed column: 'sales_diff_lag_2'")
    print(f"   Columns in data: {transformed2.columns.tolist()}")
    assert "sales_diff_lag_2" in transformed2.columns
    print("   ✓ Lag parameter (2) clearly indicated")

    # Example 3: Second-order differencing (lag=1, differences=2)
    print("\n3. Second-Order Differencing (lag=1, differences=2):")
    print("-" * 70)
    rec3 = recipe().step_diff(columns=["sales"], lag=1, differences=2)
    transformed3 = rec3.prep(data).bake(data)
    print(f"   Original column: 'sales'")
    print(f"   Transformed column: 'sales_diff_lag_1_order_2'")
    print(f"   Columns in data: {transformed3.columns.tolist()}")
    assert "sales_diff_lag_1_order_2" in transformed3.columns
    print("   ✓ Both lag (1) and order (2) clearly indicated")

    # Example 4: Custom lag with second-order differencing (lag=7, differences=2)
    print("\n4. Weekly Lag with Second-Order Differencing (lag=7, differences=2):")
    print("-" * 70)
    rec4 = recipe().step_diff(columns=["sales"], lag=7, differences=2)
    transformed4 = rec4.prep(data).bake(data)
    print(f"   Original column: 'sales'")
    print(f"   Transformed column: 'sales_diff_lag_7_order_2'")
    print(f"   Columns in data: {transformed4.columns.tolist()}")
    assert "sales_diff_lag_7_order_2" in transformed4.columns
    print("   ✓ Weekly lag (7) and second order (2) clearly indicated")

    # Example 5: Multiple columns
    print("\n5. Multiple Columns with Default Differencing:")
    print("-" * 70)
    rec5 = recipe().step_diff(columns=["sales", "price"])
    transformed5 = rec5.prep(data).bake(data)
    print(f"   Original columns: ['sales', 'price']")
    print(f"   Transformed columns: ['sales_diff_lag_1', 'price_diff_lag_1']")
    print(f"   Columns in data: {transformed5.columns.tolist()}")
    assert "sales_diff_lag_1" in transformed5.columns
    assert "price_diff_lag_1" in transformed5.columns
    print("   ✓ Both columns have 'lag' in their names")

    print("\n" + "="*70)
    print("COMPARISON: OLD vs NEW NAMING")
    print("="*70)
    print("\nOLD NAMING (Ambiguous):")
    print("  - sales_diff_1        ← What does '1' mean? Lag? Order?")
    print("  - sales_diff_2        ← Is this lag=2 or differences=2?")
    print("  - sales_diff_1_2      ← Unclear what each number means")
    print("\nNEW NAMING (Clear):")
    print("  - sales_diff_lag_1         ← Clearly lag=1")
    print("  - sales_diff_lag_2         ← Clearly lag=2")
    print("  - sales_diff_lag_1_order_2 ← Clearly lag=1, order=2")
    print("\n" + "="*70)
    print("ALL NAMING TESTS PASSED! ✓")
    print("="*70)


if __name__ == "__main__":
    test_diff_column_naming_examples()
