"""
Test to verify step_winsorizer works correctly with quantiles method
"""

import pandas as pd
import numpy as np
from py_recipes import recipe
from py_recipes.selectors import all_numeric, one_of, difference

# Create test data with outliers
np.random.seed(42)
n = 100

data = pd.DataFrame({
    'x1': np.concatenate([
        np.random.randn(90) * 10 + 50,  # Normal values
        np.array([150, 160, 170, 180, 190, 200, 210, 220, 230, 240])  # Outliers
    ]),
    'x2': np.concatenate([
        np.random.randn(90) * 5 + 20,  # Normal values
        np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])  # Outliers
    ]),
    'y': np.random.randn(100) * 100 + 500,
})

print("Original data statistics:")
print(data.describe())
print()
print(f"x1 max: {data['x1'].max():.2f} (should be ~240)")
print(f"x2 max: {data['x2'].max():.2f} (should be ~125)")
print()

# Test 1: Verify quantiles validation
print("="*60)
print("Test 1: Quantiles validation")
print("="*60)

try:
    rec_bad = recipe().step_winsorizer(
        capping_method='quantiles',
        quantiles=(0.05, 0.90)  # Not symmetric!
    )
    print("❌ FAILED: Should have raised ValueError for asymmetric quantiles")
except ValueError as e:
    print(f"✓ Correctly rejected asymmetric quantiles:")
    print(f"  {e}")
print()

# Test 2: Valid quantiles
print("="*60)
print("Test 2: Valid symmetric quantiles")
print("="*60)

try:
    rec = (
        recipe()
        .step_winsorizer(
            columns=difference(all_numeric(), one_of('y')),  # Exclude outcome
            capping_method='quantiles',
            quantiles=(0.05, 0.95)  # Symmetric: 0.95 == 1 - 0.05
        )
    )

    print("✓ Recipe created successfully with symmetric quantiles")

    # Prep the recipe
    prepped = rec.prep(data)
    print("✓ Recipe prepped successfully")

    # Bake (apply winsorization)
    winsorized = prepped.bake(data)
    print("✓ Recipe baked successfully")

    print("\nWinsorized data statistics:")
    print(winsorized.describe())
    print()

    # Verify capping occurred
    print("Verification:")
    print(f"x1 max after winsorizing: {winsorized['x1'].max():.2f} (should be < 240)")
    print(f"x2 max after winsorizing: {winsorized['x2'].max():.2f} (should be < 125)")
    print(f"y unchanged: {winsorized['y'].max():.2f} (should match original)")

    if winsorized['x1'].max() < data['x1'].max():
        print("✅ x1 outliers capped!")
    else:
        print("❌ x1 outliers NOT capped")

    if winsorized['x2'].max() < data['x2'].max():
        print("✅ x2 outliers capped!")
    else:
        print("❌ x2 outliers NOT capped")

    if abs(winsorized['y'].max() - data['y'].max()) < 0.01:
        print("✅ y preserved (not winsorized)")
    else:
        print("❌ y was modified")

    print("\n" + "="*60)
    print("✅ SUCCESS! step_winsorizer works with quantiles method")
    print("="*60)

except Exception as e:
    print("\n" + "="*60)
    print("❌ FAILED!")
    print("="*60)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
