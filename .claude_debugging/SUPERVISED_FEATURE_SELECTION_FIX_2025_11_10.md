# Supervised Feature Selection Fix - 2025-11-10

## Problem

The notebook `_md/forecasting_recipes_grouped.ipynb` was failing when using supervised feature selection steps (`step_select_permutation`, `step_select_shap`, `step_safe_v2`) with `fit_nested()`:

```
ValueError: Outcome 'refinery_kbd' not found in data
```

## Root Cause

Supervised feature selection steps like `step_select_permutation` and `step_select_shap` need the **outcome column** during both `prep()` and `bake()` to calculate feature importance scores. However, `fit_nested()` was excluding the outcome before calling these methods.

## Solution

Implemented a comprehensive fix across three key areas:

### 1. Added Import Exports (py_recipes/__init__.py)

**Problem**: Feature selection steps existed but weren't exported.

**Fix**: Added to `__init__.py`:
- `step_select_shap`
- `step_select_permutation`
- `step_safe_v2`
- Corresponding class names

**Lines**: 69-77, 80-84, 137-148

### 2. Helper Method to Detect Supervised Steps (py_workflows/workflow.py)

**Added**: `_recipe_requires_outcome()` method (lines 184-226)

Detects if recipe contains supervised steps that need outcome during prep/bake:
- StepFilterAnova
- StepFilterRfImportance
- StepFilterMutualInfo
- StepFilterRocAuc
- StepFilterChisq
- StepSelectShap
- StepSelectPermutation
- StepSafe
- StepSafeV2

Works with both `Recipe` and `PreparedRecipe` objects.

### 3. Helper Method to Extract Outcome from Recipe (py_workflows/workflow.py)

**Added**: `_get_outcome_from_recipe()` method (lines 228-253)

Extracts outcome column name from supervised steps' `.outcome` attribute instead of guessing from data.

**Why Needed**: The `_detect_outcome()` method was returning the first numeric column (x1) instead of the actual outcome (refinery_kbd).

### 4. Modified Global Recipe Prep Logic (py_workflows/workflow.py)

**Location**: Lines 545-565

**Before**:
```python
# Prep on predictors only (excluding outcome)
predictors_global = data.drop(columns=[outcome_col_global, group_col])
global_recipe = self.preprocessor.prep(predictors_global)
```

**After**:
```python
# Check if recipe has supervised steps that need outcome during prep
needs_outcome = self._recipe_requires_outcome(self.preprocessor)

if needs_outcome:
    # Prep with outcome included (for supervised feature selection)
    prep_data = data.drop(columns=[group_col])
    global_recipe = self.preprocessor.prep(prep_data)
else:
    # Prep on predictors only (excluding outcome)
    predictors_global = data.drop(columns=[outcome_col_global, group_col])
    global_recipe = self.preprocessor.prep(predictors_global)
```

### 5. Modified Per-Group Recipe Prep Logic (py_workflows/workflow.py)

**Location**: Lines 596-622

**Key Changes**:
1. Use `_get_outcome_from_recipe()` to get outcome from supervised steps
2. Conditionally include outcome during prep based on `needs_outcome` flag

**Before**:
```python
outcome_col = self._detect_outcome(group_data_no_group)
# ...
predictors = group_data_no_group.drop(columns=[outcome_col])
group_recipe = self.preprocessor.prep(predictors)
```

**After**:
```python
# For supervised feature selection, get outcome from recipe; otherwise auto-detect
outcome_col = self._get_outcome_from_recipe(self.preprocessor)
if outcome_col is None:
    outcome_col = self._detect_outcome(group_data_no_group)

# Check if recipe needs outcome during prep
needs_outcome = self._recipe_requires_outcome(self.preprocessor)

if needs_outcome:
    # Prep with outcome included (for supervised feature selection)
    group_recipe = self.preprocessor.prep(group_data_no_group)
else:
    # Prep on predictors only (excluding outcome)
    predictors = group_data_no_group.drop(columns=[outcome_col])
    group_recipe = self.preprocessor.prep(predictors)
```

### 6. Modified Bake Logic (py_workflows/workflow.py)

**Location**: `_prep_and_bake_with_outcome()` method (lines 249-289)

**Before**: Always separated outcome before baking

**After**: Conditionally include outcome during bake:

```python
# Check if recipe has supervised steps that need outcome during bake
needs_outcome = self._recipe_requires_outcome(recipe)

if needs_outcome:
    # Bake with outcome included (for supervised feature selection)
    processed_data = recipe.bake(data)
else:
    # Separate outcome from predictors
    outcome = data[outcome_col].copy()
    predictors = data.drop(columns=[outcome_col])
    # Bake predictors only
    processed_predictors = recipe.bake(predictors)
    # Recombine with outcome
    processed_data = processed_predictors.copy()
    processed_data[outcome_col] = outcome.values
```

## Testing

### Test Script Created
`.claude_debugging/test_supervised_fit_nested.py`

**Test Results**: All 4 tests passing ✅

1. ✅ `step_select_permutation` with `per_group_prep=True`
2. ✅ `step_select_permutation` with `per_group_prep=False` (global recipe)
3. ✅ Regular `fit()` with supervised steps
4. ✅ Non-supervised steps (backward compatibility)

### Integration Tests
**Result**: All 90 workflow tests passing ✅

No regressions introduced.

## Files Modified

1. **py_recipes/__init__.py**
   - Added exports for supervised feature selection steps

2. **py_workflows/workflow.py**
   - Added `_recipe_requires_outcome()` method (39 lines)
   - Added `_get_outcome_from_recipe()` method (26 lines)
   - Modified global recipe prep logic (16 lines changed)
   - Modified per-group recipe prep logic (13 lines changed)
   - Modified `_prep_and_bake_with_outcome()` method (23 lines changed)

3. **.claude_debugging/test_supervised_fit_nested.py** (created)
   - Comprehensive test suite for supervised feature selection

## Key Design Decisions

### 1. Conditional Outcome Inclusion

**Decision**: Check recipe for supervised steps and conditionally include outcome

**Reasoning**:
- Supervised steps NEED outcome for importance calculation
- Non-supervised steps DON'T need outcome (cleaner separation)
- Maintains backward compatibility

### 2. Extract Outcome from Recipe Steps

**Decision**: Use step's `.outcome` attribute instead of auto-detecting

**Reasoning**:
- Auto-detection returns FIRST numeric column
- This fails when outcome isn't named 'y', 'target', or 'outcome'
- Supervised steps already know the outcome column
- More reliable and explicit

### 3. Handle Both Recipe and PreparedRecipe

**Decision**: Helper methods work with both types

**Reasoning**:
- `prep()` creates PreparedRecipe
- Subsequent operations use PreparedRecipe
- Need to inspect steps in both cases
- PreparedRecipe uses `prepared_steps` attribute

## Benefits

1. **Supervised feature selection now works** with `fit_nested()`
2. **Both per-group and global recipes** supported
3. **Backward compatibility** maintained for non-supervised steps
4. **Automatic outcome detection** from recipe steps
5. **Cleaner API** - users don't need workarounds

## Usage Example

```python
from py_recipes import recipe, step_select_permutation
from py_workflows import workflow
from py_parsnip import linear_reg
from sklearn.ensemble import RandomForestRegressor

# Create recipe with supervised feature selection
rec = (
    recipe()
    .step_normalize()
    .step_select_permutation(
        outcome='sales',
        model=RandomForestRegressor(n_estimators=10),
        top_n=5
    )
)

# Use with fit_nested (now works!)
wf = workflow().add_recipe(rec).add_model(linear_reg())
fit = wf.fit_nested(train_data, group_col='country', per_group_prep=True)

# Each group selects different features based on its data
fit = fit.evaluate(test_data)
outputs, coeffs, stats = fit.extract_outputs()
```

## Next Steps for User

1. **Restart Jupyter kernel**: Kernel → Restart & Clear Output
2. **Re-run cells from beginning**
3. **Cell 56 imports should now work**
4. **Supervised feature selection examples should execute successfully**

---

**Status**: ✅ Complete and tested
**Date**: 2025-11-10
**Tests Passing**: 94/94 (4 new + 90 existing)
