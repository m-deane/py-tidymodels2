# Complete Recipe Step Audit - November 11, 2025

## Executive Summary

**Total Steps Audited:** 68 step classes across 23 files
**SAFE Steps:** 64 (94%)
**BROKEN Steps:** 4 (6%)
**Missing Steps:** 2 (StepSelectShap, StepSelectPermutation)

## Audit Methodology

Systematic check of every `prep()` method in all recipe steps:

**BROKEN Pattern (Bug):**
```python
def prep(self, data, training=True):
    self._attribute = value  # Mutates self
    return self                # Returns mutated self
```

**SAFE Patterns:**
```python
# Pattern 1: Returns PreparedStepXXX
def prep(self, data, training=True) -> "PreparedStepXXX":
    return PreparedStepXXX(...)  # Different class

# Pattern 2: Uses replace(self)
def prep(self, data, training=True):
    prepared = replace(self)
    prepared._attr = value
    return prepared

# Pattern 3: No state stored
def prep(self, data, training=True):
    return self  # No mutation, trivial case
```

## Broken Steps Requiring Fix

### 1. StepRm (remove.py)
**Location:** `py_recipes/steps/remove.py:54-76`
**Priority:** HIGH (commonly used)
**Issue:** Lines 61-75 mutate `self._columns_to_remove` and `self._is_prepared`

**Current Code:**
```python
def prep(self, data: pd.DataFrame, training: bool = True):
    if self.skip or not training:
        return self

    # Resolve columns
    if isinstance(self.columns, str):
        self._columns_to_remove = [self.columns]  # ❌ MUTATION
    # ... more mutations
    self._is_prepared = True  # ❌ MUTATION
    return self  # ❌ RETURNS MUTATED SELF
```

**Fix Required:**
```python
def prep(self, data: pd.DataFrame, training: bool = True):
    if self.skip or not training:
        return self

    # Resolve columns locally
    if isinstance(self.columns, str):
        columns_to_remove = [self.columns]
    # ... resolve locally

    # Create new instance
    prepared = replace(self)
    prepared._columns_to_remove = columns_to_remove
    prepared._is_prepared = True
    return prepared
```

### 2. StepSelect (remove.py)
**Location:** `py_recipes/steps/remove.py:133-155`
**Priority:** HIGH (commonly used)
**Issue:** Lines 140-154 mutate `self._columns_to_keep` and `self._is_prepared`

**Current Code:**
```python
def prep(self, data: pd.DataFrame, training: bool = True):
    if self.skip or not training:
        return self

    # Resolve columns
    if isinstance(self.columns, str):
        self._columns_to_keep = [self.columns]  # ❌ MUTATION
    # ... more mutations
    self._is_prepared = True  # ❌ MUTATION
    return self  # ❌ RETURNS MUTATED SELF
```

**Fix Required:** Same pattern as StepRm - use `replace(self)`

### 3. StepSafe (feature_extraction.py)
**Location:** `py_recipes/steps/feature_extraction.py:239-356`
**Priority:** MEDIUM (advanced feature)
**Issue:** Lines 264, 296-309, 313-348 mutate multiple attributes:
- `self._original_columns` (line 264)
- `self._variables` (lines 301, 309)
- `self._feature_importances` (line 315)
- `self._selected_features` (line 331)
- `self._is_prepared` (line 354)

**Current Code:**
```python
def prep(self, data: pd.DataFrame, training: bool = True):
    if self.skip or not training:
        return self

    X = data.drop(columns=[self.outcome]).copy()
    self._original_columns = list(X.columns)  # ❌ MUTATION

    # ... lots of processing

    for idx, col in enumerate(self._original_columns):
        if col in numeric_cols:
            var = self._fit_numeric_variable(...)
            self._variables.append(var)  # ❌ MUTATION
        else:
            var = self._fit_categorical_variable(...)
            self._variables.append(var)  # ❌ MUTATION

    self._compute_feature_importances(X_transformed, outcome_series)  # ❌ MUTATION

    if self.top_n is not None:
        # ... compute
        self._selected_features = sorted_features[:self.top_n]  # ❌ MUTATION

    self._is_prepared = True  # ❌ MUTATION
    return self  # ❌ RETURNS MUTATED SELF
```

**Fix Required:**
```python
def prep(self, data: pd.DataFrame, training: bool = True):
    if self.skip or not training:
        return self

    # All computation happens locally
    X = data.drop(columns=[self.outcome]).copy()
    original_columns = list(X.columns)
    variables = []

    # ... process into local variables
    for idx, col in enumerate(original_columns):
        if col in numeric_cols:
            var = self._fit_numeric_variable(...)
            variables.append(var)
        # ... etc

    # Compute importances (may need to modify this helper)
    feature_importances = self._compute_feature_importances_pure(X_transformed, outcome_series)

    # Select features locally
    if self.top_n is not None:
        selected_features = sorted_features[:self.top_n]
    else:
        selected_features = []

    # Create new instance at the end
    prepared = replace(self)
    prepared._original_columns = original_columns
    prepared._variables = variables
    prepared._feature_importances = feature_importances
    prepared._selected_features = selected_features
    prepared._is_prepared = True
    return prepared
```

**Note:** `_compute_feature_importances()` method at line 315 likely also mutates `self._feature_importances`. This helper method needs refactoring to return values instead of mutating.

### 4. StepEIX (interaction_detection.py)
**Location:** `py_recipes/steps/interaction_detection.py:321-420`
**Priority:** MEDIUM (advanced feature)
**Issue:** Lines 338, 346, 388, 391-392, 405-407, 412-417, 419 mutate multiple attributes:
- `self._is_prepped` (line 338, 419)
- `self._original_columns` (line 346)
- `self._importance_table` (line 388)
- `self._selected_features` (lines 391, 398, 405-407)
- `self._interactions_to_create` (lines 392, 412-417)

**Current Code:**
```python
def prep(self, data: pd.DataFrame, training: bool = True) -> "StepEIX":
    if self.skip:
        self._is_prepped = True  # ❌ MUTATION
        return self

    # Store original columns
    self._original_columns = [col for col in data.columns if col != self.outcome]  # ❌ MUTATION

    # Extract and process
    trees_df = self._extract_trees_dataframe()
    importance = self._calculate_variable_importance(trees_df)
    # ... processing

    # Store results
    self._importance_table = importance.copy()  # ❌ MUTATION
    self._selected_features = []  # ❌ MUTATION
    self._interactions_to_create = []  # ❌ MUTATION

    for _, row in importance.iterrows():
        if row['type'] == 'variable':
            self._selected_features.append(feature_name)  # ❌ MUTATION
        elif row['type'] == 'interaction':
            self._selected_features.append(parent)  # ❌ MUTATION
            self._interactions_to_create.append({...})  # ❌ MUTATION

    self._is_prepped = True  # ❌ MUTATION
    return self  # ❌ RETURNS MUTATED SELF
```

**Fix Required:**
```python
def prep(self, data: pd.DataFrame, training: bool = True) -> "StepEIX":
    if self.skip:
        prepared = replace(self)
        prepared._is_prepped = True
        return prepared

    # Store original columns locally
    original_columns = [col for col in data.columns if col != self.outcome]

    # Extract and process locally
    trees_df = self._extract_trees_dataframe()
    importance = self._calculate_variable_importance(trees_df)
    # ... processing

    # Build results locally
    importance_table = importance.copy()
    selected_features = []
    interactions_to_create = []

    for _, row in importance.iterrows():
        if row['type'] == 'variable':
            selected_features.append(feature_name)
        elif row['type'] == 'interaction':
            selected_features.append(parent)
            interactions_to_create.append({...})

    # Create new instance
    prepared = replace(self)
    prepared._original_columns = original_columns
    prepared._importance_table = importance_table
    prepared._selected_features = selected_features
    prepared._interactions_to_create = interactions_to_create
    prepared._is_prepped = True
    return prepared
```

## Missing Steps (ImportError Issue)

### StepSelectShap
**Status:** DOES NOT EXIST
**Referenced in:**
- `py_recipes/steps/__init__.py:229` - exported in `__all__`
- `py_recipes/recipe.py:880` - import statement
- `py_recipes/__init__.py:63` - import statement
- `tests/test_recipes/test_select_shap.py:10` - test imports it

**Error:** `ImportError: cannot import name 'StepSelectShap' from 'py_recipes.steps.filter_supervised'`

### StepSelectPermutation
**Status:** DOES NOT EXIST
**Referenced in:**
- `py_recipes/steps/__init__.py:230` - exported in `__all__`
- `py_recipes/recipe.py:930` - import statement
- `py_recipes/__init__.py:63` - import statement
- `tests/test_recipes/test_select_permutation.py:10` - test imports it

**Error:** `ImportError: cannot import name 'StepSelectPermutation' from 'py_recipes.steps.filter_supervised'`

**Action Required:** These steps need to be implemented OR removed from all import statements.

## Safe Steps (64 Total)

### Basis Functions (4 steps)
- ✅ StepBs - Returns PreparedStepBs
- ✅ StepNs - Returns PreparedStepNs
- ✅ StepPoly - Returns PreparedStepPoly
- ✅ StepHarmonic - Returns PreparedStepHarmonic

### Categorical Extended (4 steps)
- ✅ StepOther - Returns PreparedStepOther
- ✅ StepNovel - Returns PreparedStepNovel
- ✅ StepUnknown - Returns PreparedStepUnknown
- ✅ StepIndicateNa - Returns PreparedStepIndicateNa
- ✅ StepInteger - Returns PreparedStepInteger

### Discretization (3 steps)
- ✅ StepDiscretize - Returns PreparedStepDiscretize
- ✅ StepCut - Returns PreparedStepCut
- ✅ StepPercentile - Returns PreparedStepPercentile

### Dummy Encoding (1 step)
- ✅ StepDummy - Returns PreparedStepDummy

### Feature Extraction (1 step)
- ✅ StepSafeV2 - Uses replace(self) pattern (FIXED 2025-11-10)

### Feature Selection (2 steps)
- ✅ StepPCA - Returns PreparedStepPCA
- ✅ StepSelectCorr - Returns PreparedStepSelectCorr

### Feature Selection Advanced (3 steps)
- ✅ StepVip - Uses replace(self) pattern (FIXED 2025-11-10)
- ✅ StepBoruta - Uses replace(self) pattern (FIXED 2025-11-10)
- ✅ StepRfe - Uses replace(self) pattern (FIXED 2025-11-10)

### Supervised Filters (5 steps)
- ✅ StepFilterAnova - Uses replace(self) pattern (FIXED 2025-11-10)
- ✅ StepFilterRfImportance - Uses replace(self) pattern (FIXED 2025-11-10)
- ✅ StepFilterMutualInfo - Uses replace(self) pattern (FIXED 2025-11-10)
- ✅ StepFilterRocAuc - Uses replace(self) pattern (FIXED 2025-11-10)
- ✅ StepFilterChisq - Uses replace(self) pattern (FIXED 2025-11-10)

### Filters (4 steps)
- ✅ StepZv - Returns PreparedStepZv
- ✅ StepNzv - Returns PreparedStepNzv
- ✅ StepLinComb - Returns PreparedStepLinComb
- ✅ StepFilterMissing - Returns PreparedStepFilterMissing

### Financial Oscillators (1 step)
- ✅ StepOscillators - Returns PreparedStepOscillators

### Imputation (7 steps)
- ✅ StepImputeMean - Returns PreparedStepImputeMean
- ✅ StepImputeMedian - Returns PreparedStepImputeMedian
- ✅ StepImputeMode - Returns PreparedStepImputeMode
- ✅ StepImputeKnn - Returns PreparedStepImputeKnn
- ✅ StepImputeLinear - Returns PreparedStepImputeLinear
- ✅ StepImputeBag - Returns PreparedStepImputeBag
- ✅ StepImputeRoll - Returns PreparedStepImputeRoll

### Interactions (2 steps)
- ✅ StepInteract - Returns PreparedStepInteract
- ✅ StepRatio - Returns PreparedStepRatio

### Mutate (1 step)
- ✅ StepMutate - Returns PreparedStepMutate

### NaOmit (1 step)
- ✅ StepNaOmit - Returns PreparedStepNaOmit

### Normalize (1 step)
- ✅ StepNormalize - Returns PreparedStepNormalize

### Reduction (3 steps)
- ✅ StepIca - Returns PreparedStepIca
- ✅ StepKpca - Returns PreparedStepKpca
- ✅ StepPls - Returns PreparedStepPls

### Scaling (3 steps)
- ✅ StepCenter - Returns PreparedStepCenter
- ✅ StepScale - Returns PreparedStepScale
- ✅ StepRange - Returns PreparedStepRange

### Splitwise (1 step)
- ✅ StepSplitwise - Uses replace(self) pattern (FIXED 2025-11-10)

### Timeseries (5 steps)
- ✅ StepLag - Returns PreparedStepLag
- ✅ StepDiff - Returns PreparedStepDiff
- ✅ StepPctChange - Returns PreparedStepPctChange
- ✅ StepRolling - Returns PreparedStepRolling
- ✅ StepDate - Returns PreparedStepDate

### Timeseries Extended (6 steps)
- ✅ StepHoliday - Returns PreparedStepHoliday
- ✅ StepFourier - Returns PreparedStepFourier
- ✅ StepTimeseriesSignature - Returns PreparedStepTimeseriesSignature
- ✅ StepLead - Returns PreparedStepLead
- ✅ StepEwm - Returns PreparedStepEwm
- ✅ StepExpanding - Returns PreparedStepExpanding

### Transformations (5 steps)
- ✅ StepLog - Returns PreparedStepLog
- ✅ StepSqrt - Returns PreparedStepSqrt
- ✅ StepBoxCox - Returns PreparedStepBoxCox
- ✅ StepYeoJohnson - Returns PreparedStepYeoJohnson
- ✅ StepInverse - Returns PreparedStepInverse

## Fix Priority Ranking

1. **IMMEDIATE (HIGH):**
   - StepRm - Very commonly used
   - StepSelect - Very commonly used

2. **SOON (MEDIUM):**
   - StepSafe - Advanced feature, likely used in SAFE notebooks/demos
   - StepEIX - Advanced feature, likely used in EIX notebooks/demos

3. **MISSING STEPS:**
   - Remove imports for StepSelectShap and StepSelectPermutation OR implement them

## Recommended Fix Order

1. Fix StepRm and StepSelect (same file, straightforward fix)
2. Remove/implement StepSelectShap and StepSelectPermutation
3. Fix StepSafe (requires refactoring helper method)
4. Fix StepEIX (straightforward fix, similar to supervised filters)

## Testing Requirements

After each fix:
1. Run step-specific tests: `pytest tests/test_recipes/test_<stepfile>.py -v`
2. Run full recipe test suite: `pytest tests/test_recipes/ -v`
3. Test in notebook if demo exists
4. Verify `.prep().prep()` idempotency

## Code References

- Audit script: `.claude_debugging/audit_all_steps.py`
- Previous fixes: `.claude_debugging/SUPERVISED_FEATURE_SELECTION_FIX_2025_11_10.md`
- Safe pattern: `.claude_debugging/STEP_SAFE_REPLACEMENT_COMPLETE.md`

## Summary Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| Total Steps | 68 | 100% |
| Safe Steps | 64 | 94% |
| Broken Steps | 4 | 6% |
| Missing Steps | 2 | - |

**Conclusion:** The vast majority (94%) of recipe steps are correctly implemented. Only 4 steps need mutation fixes, and 2 missing steps need to be addressed. This is a very healthy codebase state.
