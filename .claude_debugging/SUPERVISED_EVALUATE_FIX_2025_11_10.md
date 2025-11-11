# Supervised Feature Selection evaluate() Fix - 2025-11-10

## Problem

When using supervised feature selection steps (e.g., `step_filter_anova`) followed by `evaluate()` on test data, the workflow failed with:

```
ValueError: The feature names should match those that were passed during fit.
Feature names seen at fit time, yet now missing:
- bakken_coking_usmc
- brent_cracking_nw_europe
- es_sider_cracking_med
- refinery_kbd  ← The outcome column
- x30_70_wcs_bakken_cracking_usmc
```

## Root Cause

The issue occurred in `WorkflowFit.evaluate()` when processing test data:

1. **During Training** (with supervised steps):
   - Recipe is prepped WITH outcome column (needed for feature importance calculation)
   - `step_filter_anova` selects features and keeps them + outcome
   - `step_normalize` is fitted on selected features + outcome

2. **During Test Evaluation** (BEFORE fix):
   - `WorkflowFit.evaluate()` separated outcome from predictors
   - Tried to bake predictors ONLY through PreparedRecipe
   - `step_normalize` expected all columns it saw during prep (including outcome)
   - **Result**: sklearn's StandardScaler failed because outcome column was missing

3. **Additional Issue**: Outcome column detection
   - `_detect_outcome()` was called on test data to guess outcome column name
   - For supervised steps, should use `_get_outcome_from_recipe()` instead
   - **Result**: Wrong outcome column name detected (e.g., 'x1' instead of 'refinery_kbd')

## Solution

Applied two fixes in `WorkflowFit.evaluate()` method:

### Fix 1: Conditional Outcome Inclusion During Bake (Lines 927-943)

Check if recipe has supervised steps and include outcome during baking if needed:

```python
# Check if outcome exists in test data
if outcome_col in test_data.columns:
    # Check if recipe has supervised steps that need outcome during bake
    needs_outcome = self.workflow._recipe_requires_outcome(self.pre)

    if needs_outcome:
        # Bake with outcome included (for supervised feature selection)
        processed_test_data = self.pre.bake(test_data)
    else:
        # Separate outcome from predictors
        outcome = test_data[outcome_col].copy()
        predictors = test_data.drop(columns=[outcome_col])

        # Bake predictors only
        processed_predictors = self.pre.bake(predictors)

        # Recombine with outcome (align by index to handle step_naomit)
        processed_test_data = processed_predictors.copy()
        processed_test_data[outcome_col] = outcome.loc[processed_predictors.index].values
```

**Why This Works**:
- Supervised steps were prepped WITH outcome, so they expect it during baking
- Non-supervised steps were prepped WITHOUT outcome, so we exclude it
- Maintains backward compatibility for standard recipes

### Fix 2: Outcome Detection from Recipe (Lines 922-926)

Get outcome column name from recipe instead of auto-detecting:

```python
# Detect outcome column if not provided
if outcome_col is None:
    # For supervised steps, get outcome from recipe; otherwise auto-detect
    outcome_col = self.workflow._get_outcome_from_recipe(self.pre)
    if outcome_col is None:
        outcome_col = self.workflow._detect_outcome(test_data)
```

**Why This Matters**:
- Supervised steps have `.outcome` attribute with correct column name
- Auto-detection on baked data can return wrong column (first numeric instead of actual outcome)
- More reliable and explicit

## Files Modified

### py_workflows/workflow.py (2 changes)

**Change 1: Lines 922-926** - Outcome detection from recipe
```python
# BEFORE:
if outcome_col is None:
    outcome_col = self.workflow._detect_outcome(test_data)

# AFTER:
if outcome_col is None:
    outcome_col = self.workflow._get_outcome_from_recipe(self.pre)
    if outcome_col is None:
        outcome_col = self.workflow._detect_outcome(test_data)
```

**Change 2: Lines 927-943** - Conditional bake with outcome
```python
# BEFORE:
if outcome_col in test_data.columns:
    outcome = test_data[outcome_col].copy()
    predictors = test_data.drop(columns=[outcome_col])
    processed_predictors = self.pre.bake(predictors)
    processed_test_data = processed_predictors.copy()
    processed_test_data[outcome_col] = outcome.values  # Always separated

# AFTER:
if outcome_col in test_data.columns:
    needs_outcome = self.workflow._recipe_requires_outcome(self.pre)
    if needs_outcome:
        processed_test_data = self.pre.bake(test_data)  # With outcome
    else:
        # Separate and recombine (standard path)
        ...
```

## Testing

### New Test Created
`.claude_debugging/test_supervised_evaluate_fix.py`

**Test Coverage**:
1. `step_filter_anova` + `step_normalize` with per-group prep - ✅ PASS
2. `step_filter_rf_importance` + `step_normalize` with per-group prep - ✅ PASS
3. Supervised steps with global recipe - ✅ PASS

### Regression Testing
**All 90 workflow tests passing** ✅

No regressions introduced.

## Impact

### Before Fix
- ❌ `evaluate()` failed for supervised feature selection + normalization
- ❌ Wrong outcome column detected after baking
- ❌ Test data evaluation broken for supervised recipes

### After Fix
- ✅ `evaluate()` works correctly with supervised feature selection
- ✅ Correct outcome column extracted from recipe
- ✅ Both per-group and global recipes work
- ✅ Backward compatible with non-supervised recipes

## Technical Details

### Supervised Steps That Need Outcome During Bake

Steps detected by `_recipe_requires_outcome()`:
- `StepFilterAnova`
- `StepFilterRfImportance`
- `StepFilterMutualInfo`
- `StepFilterRocAuc`
- `StepFilterChisq`
- `StepSelectShap`
- `StepSelectPermutation`
- `StepSafe`
- `StepSafeV2`

### Why Supervised Steps Keep Outcome

Supervised feature selection steps:
1. **During prep()**: Calculate feature importance using outcome
2. **During bake()**: Select top N features, keep them + outcome
3. **Return**: `data[selected_features + [outcome]]`

The outcome must flow through the entire recipe so:
- Downstream steps can use it if needed
- Model fitting receives it
- Not transformed by predictor-only steps (like normalization)

### Example Flow

**Recipe**: `step_filter_anova(outcome='y', top_n=3) → step_normalize(all_numeric_predictors())`

**Training (10 features → 3 selected)**:
1. Prep step_filter_anova on [x1...x10, y]:
   - Calculates ANOVA F-scores
   - Selects [x2, x5, x7]
2. Bake: Returns [x2, x5, x7, y]
3. Prep step_normalize on [x2, x5, x7, y]:
   - `all_numeric_predictors()` selects [x2, x5, x7] (excludes y)
   - Fits scaler on these 3 features
4. Bake: Returns [x2_norm, x5_norm, x7_norm, y]

**Test Evaluation (BEFORE fix)**:
1. evaluate() separates y from predictors
2. Tries to bake [x1...x10] (predictors only)
3. step_filter_anova keeps [x2, x5, x7]
4. step_normalize expects [x2, x5, x7, y] but gets [x2, x5, x7]
5. **ERROR**: Feature names don't match

**Test Evaluation (AFTER fix)**:
1. evaluate() detects supervised steps
2. Bakes [x1...x10, y] (with outcome)
3. step_filter_anova keeps [x2, x5, x7, y]
4. step_normalize transforms [x2, x5, x7], preserves y
5. **SUCCESS**: Returns [x2_norm, x5_norm, x7_norm, y]

## Related Fixes

This fix builds on previous supervised feature selection work:
- `.claude_debugging/SUPERVISED_FEATURE_SELECTION_FIX_2025_11_10.md` - Training with supervised steps
- `.claude_debugging/STEP_NAOMIT_INDEX_ALIGNMENT_FIX.md` - Index alignment for row removal

## Usage Example

```python
from py_recipes import recipe
from py_workflows import workflow
from py_parsnip import linear_reg

# Recipe with supervised feature selection
rec = (
    recipe()
    .step_filter_anova(outcome='sales', top_n=10)
    .step_normalize()
)

wf = workflow().add_recipe(rec).add_model(linear_reg())

# Fit
fit = wf.fit_nested(train_data, group_col='store', per_group_prep=True)

# Evaluate (now works correctly!)
fit = fit.evaluate(test_data)

# Extract results
outputs, coeffs, stats = fit.extract_outputs()
```

---

**Status**: ✅ Complete and tested
**Date**: 2025-11-10
**Tests Passing**: 90/90 workflow tests + 3 new supervised evaluate tests
**Impact**: Critical fix for supervised feature selection with evaluate()
