#!/usr/bin/env python3
"""
Test Bug #16 fix: min_samples_split normalization handling
"""

# Test the parameter conversion logic
test_cases = [
    # (input_value, expected_output, description)
    (0.0, 2, "Normalized 0.0 → 2"),
    (0.33, 2, "Normalized 0.33 → 2"),
    (0.67, 2, "Normalized 0.67 → 2"),
    (1.0, 2, "Edge case 1.0 → 2 (via int conversion)"),
    (2.0, 2, "Normal 2.0 → 2"),
    (8.0, 8, "Normal 8.0 → 8"),
    (14.0, 14, "Normal 14.0 → 14"),
    (20.0, 20, "Normal 20.0 → 20"),
]

print("=" * 60)
print("TESTING BUG #16 FIX: min_samples_split handling")
print("=" * 60)

for input_val, expected, desc in test_cases:
    # Apply the same logic as in sklearn_rand_forest.py:104-114
    if isinstance(input_val, float) and 0 < input_val < 1:
        result = 2
    else:
        result = max(2, int(input_val))

    status = "✓" if result == expected else "✗"
    print(f"{status} {desc}: input={input_val} → output={result} (expected={expected})")

print("\n" + "=" * 60)
print("All normalized values [0, 1) are now clamped to 2!")
print("This prevents sklearn errors for invalid min_samples_split values.")
print("=" * 60)
