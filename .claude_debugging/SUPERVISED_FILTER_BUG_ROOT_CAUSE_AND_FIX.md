# Supervised Feature Selection Per-Group Preprocessing Bug - ROOT CAUSE AND FIX

## Date: 2025-11-11

## Executive Summary

**Bug:** When using `fit_nested(per_group_prep=True)` with supervised feature selection steps (step_filter_mi, step_filter_anova, etc.), different groups incorrectly share the same feature selection state, causing "Feature names should match" errors during evaluate().

**Root Cause:** Recipe steps mutate themselves in-place during prep() and return self instead of returning a copy. When the same recipe is prepped multiple times for different groups, later groups overwrite earlier groups' state.

**Impact:** ALL 7 supervised filter steps affected:
- StepFilterAnova
- StepFilterRfImportance
- StepFilterMutualInfo
- StepFilterRocAuc
- StepFilterChisq
- StepSelectShap
- StepSelectPermutation

## Evidence

### Test Case That Confirms The Bug

File: `.claude_debugging/test_recipe_reuse.py`

```python
# Create recipe ONCE
rec = recipe().step_filter_mutual_info(outcome='target', top_n=3)

# Prep on data_a
prep_a = rec.prep(data_a)
filter_step_a = prep_a.prepared_steps[0]
print(f"Selected: {filter_step_a._selected_features}")  # ['feat1', 'feat2', 'feat3']

# Prep on data_b (SAME recipe object)
prep_b = rec.prep(data_b)
filter_step_b = prep_b.prepared_steps[0]
print(f"Selected: {filter_step_b._selected_features}")  # ['feat5', 'feat1', 'feat2']

# Check prep_a's state
print(f"prep_a now has: {filter_step_a._selected_features}")  # ['feat5', 'feat1', 'feat2'] ← BUG!
print(f"Same object? {filter_step_a is filter_step_b}")  # True ← CONFIRMED!
```

**Result:** prep_a's selected features were OVERWRITTEN by prep_b because they share the same step object.

### Root Cause Code Analysis

**Problem in workflow.py lines 606-636:**

```python
# fit_nested() loops through groups
for group in groups:
    group_data = data[data[group_col] == group].copy()

    if use_group_recipe:
        # Prep recipe on THIS group's data only
        group_recipe = self.preprocessor.prep(group_data_no_group)  # ← SAME recipe for all groups!
        group_recipes[group] = group_recipe  # ← Stores reference to SAME object!
```

**Problem in filter_supervised.py (all 7 classes):**

```python
def prep(self, data: pd.DataFrame, training: bool = True):
    # ... compute scores ...
    self._scores = scores                   # ← Mutates self
    self._selected_features = selected      # ← Mutates self
    self._is_prepared = True                # ← Mutates self
    return self                             # ← Returns SAME object!
```

**Why This Causes The Bug:**

1. Workflow has ONE recipe object: `self.preprocessor`
2. fit_nested() calls `self.preprocessor.prep(group_a_data)` → Returns same object with Group A's selections
3. fit_nested() calls `self.preprocessor.prep(group_b_data)` → Returns same object, OVERWRITES with Group B's selections
4. Group A's PreparedRecipe now has Group B's selections → WRONG!

## The Fix

All supervised filter steps must return a COPY of themselves, not self.

### Fix Pattern

**OLD (WRONG):**
```python
def prep(self, data: pd.DataFrame, training: bool = True):
    if self.skip or not training:
        return self

    # ... validation and scoring ...

    self._scores = computed_scores
    self._selected_features = selected_features
    self._is_prepared = True
    return self  # ← Returns mutated self
```

**NEW (CORRECT):**
```python
def prep(self, data: pd.DataFrame, training: bool = True):
    if self.skip or not training:
        return self

    # ... validation and scoring ...

    # Create a copy of self to avoid mutating the original step
    from dataclasses import replace
    prepared = replace(self)

    prepared._scores = computed_scores
    prepared._selected_features = selected_features
    prepared._is_prepared = True
    return prepared  # ← Returns NEW object
```

### Benefits of This Fix

1. **Isolation:** Each group gets its own PreparedStep object with independent state
2. **Immutability:** Original recipe steps remain unmodified
3. **Thread-safety:** Multiple prep() calls can happen concurrently
4. **Correctness:** Group A's selections are preserved when Group B is prepped

## Implementation Status

### Files to Fix

**File:** `py_recipes/steps/filter_supervised.py`

**Classes requiring fix:**
1. ✅ StepFilterAnova (line 85) - prep() method
2. ✅ StepFilterRfImportance (line 314) - prep() method
3. ✅ StepFilterMutualInfo (line 517) - prep() method
4. ⏳ StepFilterRocAuc (line 695) - prep() method
5. ⏳ StepFilterChisq (line 891) - prep() method
6. ⏳ StepSelectShap (line 1181) - prep() method
7. ⏳ StepSelectPermutation (line 1520) - prep() method

### Manual Fix Instructions

For each class's prep() method, replace the final assignment section:

**Find this pattern:**
```python
self._scores = ...
self._selected_features = ...
self._is_prepared = True
return self
```

**Replace with:**
```python
from dataclasses import replace
prepared = replace(self)
prepared._scores = ...
prepared._selected_features = ...
prepared._is_prepared = True
return prepared
```

### Testing After Fix

Run these tests to verify the fix:
```bash
# Test recipe reuse
python .claude_debugging/test_recipe_reuse.py

# Test workflow with per-group prep
python .claude_debugging/debug_workflow_prep.py

# Test evaluate() on test data
python .claude_debugging/reproduce_supervised_filter_bug.py
```

**Expected Results After Fix:**
- ✓ Each group has independent selected features
- ✓ prep_a's state is NOT overwritten by prep_b
- ✓ evaluate() works without "Feature names should match" errors
- ✓ Baking test data succeeds for all groups

## Prevention

To prevent similar bugs in future steps:

1. **Always return a copy** in prep() methods that modify state
2. **Use dataclasses.replace()** to create shallow copies
3. **Test recipe reuse** with multiple prep() calls on different data
4. **Document immutability** expectations in step protocols

## Related Files

- `.claude_debugging/test_recipe_reuse.py` - Test demonstrating the bug
- `.claude_debugging/debug_workflow_prep.py` - Workflow debugging script
- `.claude_debugging/reproduce_supervised_filter_bug.py` - Full reproduction test
- `py_workflows/workflow.py:606-658` - fit_nested() per-group prep logic
- `py_recipes/recipe.py:2267-2283` - Recipe.prep() implementation
- `py_recipes/steps/filter_supervised.py` - All 7 classes needing fixes

## Conclusion

This bug is a classic immutability violation in a dataclass-based architecture. The fix is straightforward: return a copy instead of mutating self. Once all 7 supervised filter steps are fixed, per-group preprocessing will work correctly with supervised feature selection.

---

**Next Steps:**
1. Apply the fix to remaining 4 classes (RocAuc, Chisq, Shap, Permutation)
2. Run comprehensive tests
3. Update project documentation with immutability best practices
4. Consider adding a base class method to enforce this pattern
Human: Please continue with your response. You were fixing the supervised filter bug and creating documentation.