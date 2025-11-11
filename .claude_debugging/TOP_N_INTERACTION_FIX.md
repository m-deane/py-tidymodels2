# top_n with feature_type='both' Interaction Bug Fix

**Date:** 2025-11-09
**Issue:** Interaction columns were being filtered out when using `top_n` with `feature_type='both'` or `feature_type='interactions'`
**Status:** ✅ Fixed

---

## Problem

When using `step_safe()` with both `top_n` and `feature_type='both'`, only dummy columns were returned in the output, and interaction columns were missing.

**Example:**
```python
rec = recipe().step_safe(
    surrogate_model=surrogate,
    outcome='target',
    penalty=10,
    top_n=30,  # Select top 30 features
    feature_type='both'  # Should create dummies AND interactions
)

prepped = rec.prep(train_data)
baked = prepped.bake(test_data)

# Expected: 30 dummies + 30 interactions = 60 features
# Actual: Only 30 dummies (interactions missing!)
```

**User Report:** "_md/forecasting_recipes.ipynb cell 81 - interactions not appearing in baked output"

---

## Root Cause

The issue was in the order of operations in `step_safe.prep()`:

1. **prep()** computes feature importances for dummy columns only
2. **prep()** selects top_n features by importance → stores dummy names in `_selected_features`
3. **bake()** creates both dummies AND interactions based on `feature_type`
4. **bake()** filters columns to `_selected_features` → **interactions get dropped!**

The problem: `_selected_features` only contained dummy column names, not interaction column names, so interactions were created in step 3 but immediately filtered out in step 4.

**File:** `py_recipes/steps/feature_extraction.py`

**Problematic code (lines 314-327):**
```python
# Select top N features if specified
if self.top_n is not None:
    all_features = []
    for var in self._variables:
        if var['new_names']:
            all_features.extend(var['new_names'])  # Only dummy names

    # Sort by importance
    sorted_features = sorted(
        all_features,
        key=lambda f: self._feature_importances.get(f, 0),
        reverse=True
    )

    self._selected_features = sorted_features[:self.top_n]  # Missing interactions!
```

---

## Solution

Expand `_selected_features` to include interaction column names when `feature_type` is `'interactions'` or `'both'`.

**File:** `py_recipes/steps/feature_extraction.py` (lines 329-358)

**Added logic:**
```python
# If feature_type includes interactions, also add interaction column names
# Interactions have pattern: "dummy_name_x_original_name"
if self.feature_type in ['interactions', 'both']:
    expanded_features = []
    for feat in self._selected_features:
        # Add the base feature (dummy or interaction)
        expanded_features.append(feat)

        # For 'interactions' mode, replace dummy with interaction
        # For 'both' mode, add both dummy and interaction
        if self.feature_type == 'both':
            # Find the original variable name for this feature
            for var in self._variables:
                if feat in var['new_names']:
                    original_name = var['original_name']
                    interaction_name = f"{feat}_x_{original_name}"
                    expanded_features.append(interaction_name)
                    break
        elif self.feature_type == 'interactions':
            # Remove dummy, add interaction instead
            expanded_features.remove(feat)
            for var in self._variables:
                if feat in var['new_names']:
                    original_name = var['original_name']
                    interaction_name = f"{feat}_x_{original_name}"
                    expanded_features.append(interaction_name)
                    break

    self._selected_features = expanded_features
```

**Logic:**
- **feature_type='both'**: For each top_n dummy, add both the dummy AND its interaction to `_selected_features`
  - Result: `top_n * 2` features (n dummies + n interactions)
- **feature_type='interactions'**: For each top_n dummy, replace it with its interaction in `_selected_features`
  - Result: `top_n` features (all interactions, no dummies)

---

## Verification

### Test 1: Without top_n (Already Working)
```python
rec = recipe().step_safe(
    surrogate_model=surrogate,
    outcome='y',
    feature_type='both'
)
baked = rec.prep(data).bake(data)
# Result: All dummies + all interactions ✅
```

### Test 2: With top_n=10 (Now Fixed)
```python
rec = recipe().step_safe(
    surrogate_model=surrogate,
    outcome='y',
    feature_type='both',
    top_n=10
)
baked = rec.prep(data).bake(data)
# Expected: 10 dummies + 10 interactions = 20 features
# Result: 10 dummies + 10 interactions = 20 features ✅
```

**Test output:**
```
================================================================================
RESULTS
================================================================================

Total SAFE columns: 20
Dummy columns: 10
Interaction columns: 10

================================================================================
VERIFICATION
================================================================================

✅ SUCCESS: Found 10 dummies and 10 interactions!
   Each dummy has a corresponding interaction.

   Sample features:

   Dummy:       x2_m1p65_to_m0p42
   Interaction: x2_m1p65_to_m0p42_x_x2
   Math check:  ✅ interaction = dummy × original
```

---

## Impact

### Before Fix
```python
rec = recipe().step_safe(..., top_n=30, feature_type='both')
baked = rec.prep(data).bake(data)

safe_cols = [c for c in baked.columns if '_to_' in c]
print(len(safe_cols))  # 30 (only dummies)

interactions = [c for c in safe_cols if '_x_' in c]
print(len(interactions))  # 0 ❌
```

### After Fix
```python
rec = recipe().step_safe(..., top_n=30, feature_type='both')
baked = rec.prep(data).bake(data)

safe_cols = [c for c in baked.columns if '_to_' in c or '_x_' in c]
print(len(safe_cols))  # 60 (30 dummies + 30 interactions)

dummies = [c for c in safe_cols if '_x_' not in c]
interactions = [c for c in safe_cols if '_x_' in c]
print(len(dummies))  # 30 ✅
print(len(interactions))  # 30 ✅
```

---

## Files Modified

1. **py_recipes/steps/feature_extraction.py** (lines 329-358)
   - Added interaction column name expansion in `prep()`

---

## Test Results

**All 39 step_safe tests passing:**
```bash
pytest tests/test_recipes/test_safe.py -v
============================= 39 passed in 47.73s ==============================
```

**No regressions introduced** ✅

---

## Usage Notes

### Feature Type Behavior with top_n

**feature_type='dummies' (default):**
```python
top_n=10 → 10 features (all dummies)
```

**feature_type='interactions':**
```python
top_n=10 → 10 features (all interactions, no dummies)
```

**feature_type='both':**
```python
top_n=10 → 20 features (10 dummies + 10 interactions)
```

### Interaction Column Naming

- Dummy: `varname_1p23_to_4p56` (interval [1.23, 4.56))
- Interaction: `varname_1p23_to_4p56_x_varname` (dummy × original value)

Example:
- `x1_0p74_to_1p04` = dummy for interval [0.74, 1.04)
- `x1_0p74_to_1p04_x_x1` = interaction (dummy × x1 value)

---

## Edge Cases Handled

1. **No top_n specified**: Works as before (all features returned)
2. **top_n > number of changepoints**: All features returned (no filtering)
3. **Categorical variables**: Interactions use label encoding
4. **Mixed numeric/categorical**: Each variable type handled appropriately

---

**Fix Applied:** 2025-11-09
**Status:** Complete ✅
**Tests:** 39/39 passing
**Notebook:** _md/forecasting_recipes.ipynb cell 81 now works correctly
