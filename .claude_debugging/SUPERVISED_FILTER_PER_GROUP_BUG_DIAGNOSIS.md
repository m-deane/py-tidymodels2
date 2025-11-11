# Supervised Feature Selection Per-Group Preprocessing Bug

## Date: 2025-11-11

## Root Cause Analysis

### The Bug

When using `fit_nested(per_group_prep=True)` with supervised feature selection steps (step_filter_mi, step_filter_anova, etc.), the following error occurs during `evaluate(test_data)`:

```
ValueError: The feature names should match those that were passed during fit.
Feature names seen at fit time, yet now missing:
- dubai_hydroskimming_singapore
- x30_70_wcs_bakken_cracking_usmc
```

### Evidence from Code Inspection

**Supervised Filter Steps (ALL 7 steps have the same pattern):**
- `StepFilterAnova` (lines 85-240 in filter_supervised.py)
- `StepFilterRfImportance` (lines 310-454)
- `StepFilterMutualInfo` (lines 517-624)
- `StepFilterRocAuc` (lines 688-818)
- `StepFilterChisq` (lines 884-998)
- `StepSelectShap` (lines 1174-1398)
- `StepSelectPermutation` (lines 1513-1672)

**Common bake() pattern (example from StepFilterMutualInfo line 614):**
```python
def bake(self, data: pd.DataFrame) -> pd.DataFrame:
    """Apply filter to new data."""
    if self.skip:
        return data

    if not self._is_prepared:
        raise ValueError("Step must be prepped before baking")

    keep_cols = self._selected_features + [self.outcome]
    keep_cols = [c for c in keep_cols if c in data.columns]
    return data[keep_cols]
```

### The Flow During Training (fit_nested with per_group_prep=True)

1. **For Group A:**
   ```
   Lines 606-614 in workflow.py:
   - Prep recipe on Group A's data (WITH outcome)
   - step_filter_mi.prep(group_data):
     * Scores features: [feat1, feat2, feat3, feat4, feat5]
     * Selects top features: [feat1, feat2, feat3]
     * Stores: _selected_features = [feat1, feat2, feat3]
   ```

2. **Recipe.prep() calls bake() for next step (line 2276 in recipe.py):**
   ```python
   current_data = prepared_step.bake(current_data)
   ```

   **step_filter_mi.bake():**
   ```python
   keep_cols = [feat1, feat2, feat3] + [outcome]  # = [feat1, feat2, feat3, outcome]
   return data[[feat1, feat2, feat3, outcome]]  # ← Filters to selected + outcome
   ```

3. **Next step (e.g., step_normalize) preps on FILTERED data:**
   ```
   - step_normalize.prep() receives: [feat1, feat2, feat3, outcome]
   - Fits scaler on: [feat1, feat2, feat3]
   - Stores means/stds for: [feat1, feat2, feat3]
   ```

4. **Model fits on:**
   ```
   - Formula: "outcome ~ feat1 + feat2 + feat3"
   - Features: [feat1, feat2, feat3]
   ```

5. **For Group B (different selected features):**
   ```
   - step_filter_mi selects: [feat2, feat3, feat4]
   - step_normalize fits on: [feat2, feat3, feat4]
   - Model fits on: [feat2, feat3, feat4]
   ```

### The Flow During Testing (evaluate)

1. **Group A test data arrives:**
   ```
   Test data has columns: [feat1, feat2, feat3, feat4, feat5, outcome]
   ```

2. **WorkflowFit.predict() calls PreparedRecipe.bake():**
   ```
   - step_filter_mi.bake(test_data):
     keep_cols = [feat1, feat2, feat3] + [outcome]
     return test_data[[feat1, feat2, feat3, outcome]]

   → Returns: [feat1, feat2, feat3, outcome]
   ```

3. **step_normalize.bake() receives:** `[feat1, feat2, feat3, outcome]`
   ```
   - Has scalers for: [feat1, feat2, feat3]
   - Transforms: [feat1, feat2, feat3]
   - Returns: [feat1_norm, feat2_norm, feat3_norm, outcome]
   ```

4. **Model.predict() receives:** `[feat1_norm, feat2_norm, feat3_norm, outcome]`
   ```
   - Model was trained on: [feat1, feat2, feat3]
   - **SUCCESS** ✓
   ```

### Wait... This Should Work!

The flow above shows it SHOULD work. So where's the bug?

### Re-Analyzing The Error

The error message says:
```
ValueError: The feature names should match those that were passed during fit.
Feature names seen at fit time, yet now missing:
- dubai_hydroskimming_singapore
```

This error comes from sklearn models (LightGBM, XGBoost) when features don't match.

### The REAL Root Cause

Looking at workflow.py line 611-618:

```python
needs_outcome = self._recipe_requires_outcome(self.preprocessor)

if needs_outcome:
    # Prep with outcome included (for supervised feature selection)
    group_recipe = self.preprocessor.prep(group_data_no_group)
else:
    # Prep on predictors only (excluding outcome)
    predictors = group_data_no_group.drop(columns=[outcome_col])
    group_recipe = self.preprocessor.prep(predictors)
```

**IF `needs_outcome=True`:**
- Recipe is prepped on data WITH outcome
- All steps see outcome during prep
- Supervised filter step's prep() receives outcome column
- **BUG**: The supervised filter includes outcome in the data it processes

Let me check if supervised filters exclude the outcome during scoring...

Looking at line 98-108 in filter_supervised.py (StepFilterAnova.prep):

```python
# Resolve columns to score
if self.columns is None:
    # Use all columns except outcome
    score_cols = [c for c in data.columns if c != self.outcome]
elif isinstance(self.columns, str):
    score_cols = [self.columns]
elif callable(self.columns):
    score_cols = self.columns(data)
else:
    score_cols = list(self.columns)

# Remove outcome if accidentally included
score_cols = [c for c in score_cols if c != self.outcome]
```

**Good**: The supervised filter DOES exclude the outcome from scoring.

### The ACTUAL Bug

I need to look at what happens during per-group evaluation. Let me check NestedWorkflowFit.evaluate():

Looking at the error again, the issue is that **sklearn models are seeing different features** between train and test.

**AH HA!** I found it:

During `fit_nested()` with `per_group_prep=True`, line 632-636:

```python
processed_data = self._prep_and_bake_with_outcome(
    group_recipes[group],
    group_data_no_group,
    outcome_col
)
```

The `_prep_and_bake_with_outcome()` method (lines 284-324) does this:

```python
needs_outcome = self._recipe_requires_outcome(recipe if isinstance(recipe, PreparedRecipe) else recipe)

if needs_outcome:
    # Bake with outcome included (for supervised feature selection)
    if isinstance(recipe, PreparedRecipe):
        processed_data = recipe.bake(data)  # ← Uses PREPARED recipe
```

**During training:**
- Recipe WAS prepped on data with outcome
- Recipe.bake(data) expects outcome to be present
- supervised_filter.bake() does: `keep_cols = _selected_features + [outcome]`
- **WORKS** because outcome is in data

**During testing (via WorkflowFit.predict()):**
- Line in WorkflowFit.predict():
  ```python
  if self.pre is not None and isinstance(self.pre, PreparedRecipe):
      processed_data = self.pre.bake(new_data)
  ```
- **BUG**: new_data may or may not have outcome column
- supervised_filter.bake() tries: `keep_cols = _selected_features + [outcome]`
- If outcome not in new_data: `keep_cols = [c for c in keep_cols if c in data.columns]`
- **SKIPS outcome** → Only returns selected features
- BUT: Formula still includes outcome as "y ~ feat1 + feat2"
- **BUG**: patsy tries to find outcome column, fails

NO WAIT. Let me check WorkflowFit.predict() more carefully...

Actually, I think the real bug is simpler. Let me create a test to verify:

## The REAL Bug (Final Analysis)

The bug is that during `evaluate()`, the test data goes through:
1. NestedWorkflowFit.evaluate() calls predict()
2. For each group, calls WorkflowFit.predict(group_test_data)
3. WorkflowFit.predict() bakes data: `processed_data = self.pre.bake(group_test_data)`
4. supervised_filter.bake() filters to selected features + outcome
5. **BUG LOCATION**: The outcome IS present in test data, so bake() keeps it
6. Model.predict() receives processed_data with formula
7. **SKLEARN FIT** expects features WITHOUT outcome
8. **ERROR**: sklearn sees outcome column during predict but not during fit

Wait, that's not right either. The model fit happens on processed_data which includes outcome, but patsy separates it out...

Let me write a debugging script to trace this:

