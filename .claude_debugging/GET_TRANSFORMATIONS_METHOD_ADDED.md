# get_transformations() Method Added to StepSafeV2

**Date:** 2025-11-10
**Status:** ✅ FIXED

## Issue

User encountered `AttributeError` when running notebook cell:

```python
safe_step = rec_safe_prepped.prepared_steps[0]
transformations = safe_step.get_transformations()  # ❌ AttributeError

# AttributeError: 'StepSafeV2' object has no attribute 'get_transformations'
```

**Notebook Cell:** `_md/forecasting_recipes.ipynb`, Cell 78 (In[54])

## Root Cause

`StepSafeV2` was missing the `get_transformations()` method that existed in the old `StepSafe` class. The notebook expected this method for inspecting transformation details after `prep()`.

## Solution

Added `get_transformations()` method to `StepSafeV2` class with **backward compatibility** for old naming conventions.

### Implementation

**File:** `py_recipes/steps/feature_extraction.py:1852-1891`

```python
def get_transformations(self) -> Dict[str, Any]:
    """
    Get transformation metadata for all variables.

    Returns
    -------
    dict
        Transformation information for each variable including:
        - For numeric: type, original_name, changepoints (alias for thresholds),
                      thresholds, intervals (alias for new_names), new_names
        - For categorical: type, original_name, levels,
                          merged_levels (alias for new_names), new_names

    Note: Provides both old and new naming for backward compatibility.
    """
    if not self._is_prepared:
        raise ValueError("Step must be prepared before accessing transformations")

    transformations = {}
    for var in self._variables:
        info = {
            'type': var['type'],
            'original_name': var['original_name']
        }

        if var['type'] == 'numeric':
            # New naming
            info['thresholds'] = var['thresholds']
            info['new_names'] = var['new_names']
            # Old naming (backward compatibility)
            info['changepoints'] = var['thresholds']
            info['intervals'] = var['new_names']
        else:  # categorical
            # New naming
            info['levels'] = var['levels']
            info['new_names'] = var['new_names']
            # Old naming (backward compatibility)
            info['merged_levels'] = var['new_names']

        transformations[var['original_name']] = info

    return transformations
```

## Naming Conventions

### Numeric Variables

| Old Name (StepSafe) | New Name (StepSafeV2) | Description |
|---------------------|----------------------|-------------|
| `changepoints` | `thresholds` | Threshold values detected by PDP |
| `intervals` | `new_names` | Binary indicator feature names |

**Both names now available** for backward compatibility.

### Categorical Variables

| Old Name (StepSafe) | New Name (StepSafeV2) | Description |
|---------------------|----------------------|-------------|
| `levels` | `levels` | Original categorical levels |
| `merged_levels` | `new_names` | Merged/grouped level names |

**Both names now available** for backward compatibility.

## Usage Example

```python
from sklearn.ensemble import GradientBoostingRegressor
from py_recipes import recipe

# Create unfitted model
surrogate = GradientBoostingRegressor(n_estimators=100)

# Create recipe with step_safe_v2
rec = recipe().step_safe_v2(
    surrogate_model=surrogate,
    outcome='target',
    penalty=10.0,
    max_thresholds=5,
    top_n=30
)

# Prep the recipe
rec_prepped = rec.prep(train_data)

# Access transformation details
safe_step = rec_prepped.prepared_steps[0]
transformations = safe_step.get_transformations()

# Inspect transformations
for var, info in transformations.items():
    if info['type'] == 'numeric':
        # Can use either old or new naming
        print(f"{var}: {len(info['changepoints'])} thresholds")
        print(f"  Thresholds: {info['thresholds']}")
        print(f"  Features: {info['new_names']}")
    else:
        print(f"{var}: {len(info['levels'])} levels → {len(info['merged_levels'])} groups")
```

## Test Results

```bash
$ python3 test_get_transformations.py

Prepping recipe...

Getting transformations...

=== Transformations ===

Variable: x1
  Type: numeric
  Thresholds: [-0.1269562917797126]
  Changepoints (alias): [-0.1269562917797126]
  New names: ['x1_gt_0_13']

Variable: x2
  Type: numeric
  Thresholds: [0.08410716994683427]
  Changepoints (alias): [0.08410716994683427]
  New names: ['x2_gt_0_08']

Variable: x3
  Type: categorical
  Levels: ['A', 'B', 'C']
  Merged levels: ['x3_B', 'x3_C']

✓ Test passed!
```

## Notebook Compatibility

The notebook cell (Cell 78) now works without modification:

```python
# Access the prepared step to see transformation details
safe_step = rec_safe_prepped.prepared_steps[0]
transformations = safe_step.get_transformations()  # ✅ Now works!

print("\n=== SAFE Transformation Summary ===\n")
for var, info in transformations.items():
    var_type = info['type']

    if var_type == 'numeric':
        changepoints = info['changepoints']  # Uses old naming
        print(f"  {var:15} → {len(changepoints)} changepoints detected")
    else:
        levels = info['levels']
        merged = info['merged_levels']  # Uses old naming
        print(f"  {var:15} → Merged {len(levels)} levels into {len(merged)} groups")
```

## Additional Methods Available

Along with `get_transformations()`, the following inspection methods are available on `StepSafeV2`:

1. **`get_feature_importances()`** - Returns DataFrame of feature importances
   ```python
   importances = safe_step.get_feature_importances()
   # Returns: DataFrame with columns ['feature', 'importance']
   ```

## Backward Compatibility

✅ **Full backward compatibility maintained:**
- Notebooks using old naming (`changepoints`, `intervals`, `merged_levels`) continue to work
- New code can use new naming (`thresholds`, `new_names`)
- Both names reference the same underlying data

## Files Modified

1. **`py_recipes/steps/feature_extraction.py`**
   - Added `get_transformations()` method to `StepSafeV2` class (lines 1852-1891)
   - 40 lines added

2. **`.claude_debugging/RECIPE_STEPS_UNFITTED_MODELS_COMPLETE.md`**
   - Updated documentation to include inspection methods
   - Added section describing `get_transformations()` and `get_feature_importances()`

## Related Documentation

- **Implementation:** `.claude_debugging/RECIPE_STEPS_UNFITTED_MODELS_COMPLETE.md`
- **Migration Guide:** `.claude_debugging/FORECASTING_RECIPES_MIGRATION_COMPLETE.md`
- **Test File:** `tests/test_recipes/test_safe_v2.py` (21/21 tests passing)

## Conclusion

**Status:** ✅ FIXED

The notebook cell now executes successfully. The `get_transformations()` method provides full inspection capabilities for SAFE transformations while maintaining backward compatibility with existing notebook code that expects the old naming conventions.
