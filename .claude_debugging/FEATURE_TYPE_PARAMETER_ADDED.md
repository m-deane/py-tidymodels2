# Feature Type Parameter Added to step_safe() and step_splitwise()

**Date:** 2025-11-09
**Status:** ✅ Complete
**Tests:** 73 passing (39 SAFE + 34 Splitwise, including 17 new tests)

---

## Summary

Added `feature_type` parameter to both `step_safe()` and `step_splitwise()`, allowing users to control whether to create:
- **'dummies'**: Binary dummy variables only (default, backward compatible)
- **'interactions'**: Interaction features (dummy × original_value) only
- **'both'**: Both binary dummies and interactions

This provides flexibility for modeling different relationships between features and the outcome.

---

## Motivation

**User Request:** "add an option to both steps to return either the binary dummies and interactions (so the binary dummy multiplied by the original feature), just the interactions, or just the binary dummies"

**Use Cases:**
1. **Interactions for linear models**: Capture both the threshold effect AND the magnitude effect
   - Binary dummy: "Is x above threshold?"
   - Interaction: "How much above threshold is x?"

2. **Interpretable non-linearities**: Model piecewise linear relationships
   - Different slopes before/after threshold
   - Common in economics and social sciences

3. **Feature engineering flexibility**: Choose transformation based on model type
   - Tree models: Use dummies only (already capture interactions)
   - Linear models: Use interactions or both (need explicit interactions)

---

## Implementation Details

### 1. step_splitwise() Changes

**File:** `py_recipes/steps/splitwise.py`

**Parameter Added:**
```python
feature_type: Literal['dummies', 'interactions', 'both'] = 'dummies'
```

**Bake Method Logic:**
```python
# Save original values before transformation
original_values = result[col].copy()

# Create dummy variable
dummy = (result[col] >= threshold).astype(int)

# Apply feature_type logic
if self.feature_type == 'dummies':
    result[dummy_name] = dummy
    result = result.drop(columns=[col])
elif self.feature_type == 'interactions':
    interaction_name = f"{dummy_name}_x_{col}"
    result[interaction_name] = dummy * original_values
    result = result.drop(columns=[col])
else:  # 'both'
    result[dummy_name] = dummy
    interaction_name = f"{dummy_name}_x_{col}"
    result[interaction_name] = dummy * original_values
    result = result.drop(columns=[col])
```

**Column Naming:**
- Dummy: `x_ge_5p0000` (x >= 5.0)
- Interaction: `x_ge_5p0000_x_x` (dummy × x)

**Applies to:**
- Single-split transformations: `x_ge_threshold`
- Double-split transformations: `x_between_lower_upper`

---

### 2. step_safe() Changes

**File:** `py_recipes/steps/feature_extraction.py`

**Parameter Added:**
```python
feature_type: Literal['dummies', 'interactions', 'both'] = 'dummies'
```

**Numeric Variable Transformation:**
```python
# Create DataFrame with dummies
dummies_df = pd.DataFrame(transformed, columns=new_names)

# Handle feature_type
if self.feature_type == 'dummies':
    return dummies_df
elif self.feature_type == 'interactions':
    # Create interactions: dummy * original_value
    interactions_df = pd.DataFrame()
    original_values = X_col.values
    for col in dummies_df.columns:
        interaction_name = f"{col}_x_{var['original_name']}"
        interactions_df[interaction_name] = dummies_df[col] * original_values
    return interactions_df
else:  # 'both'
    # Return both dummies and interactions
    result_df = dummies_df.copy()
    original_values = X_col.values
    for col in dummies_df.columns:
        interaction_name = f"{col}_x_{var['original_name']}"
        result_df[interaction_name] = dummies_df[col] * original_values
    return result_df
```

**Categorical Variable Transformation:**
For categorical variables, interactions use label encoding:
```python
# For categorical, use label encoding for interactions
label_encoded = pd.factorize(X_col)[0]
interactions_df = pd.DataFrame()
for col in dummies.columns:
    interaction_name = f"{col}_x_{var['original_name']}"
    interactions_df[interaction_name] = dummies[col] * label_encoded
```

**Column Naming:**
- Dummy: `x1_cp_1`, `x1_cp_2` (changepoints 1, 2)
- Interaction: `x1_cp_1_x_x1`, `x1_cp_2_x_x1`

---

### 3. Recipe Integration

**File:** `py_recipes/recipe.py`

Both `step_splitwise()` and `step_safe()` methods updated to accept `feature_type` parameter:

```python
def step_splitwise(
    self,
    outcome: str,
    transformation_mode: str = 'univariate',
    min_support: float = 0.1,
    min_improvement: float = 3.0,
    criterion: str = 'AIC',
    feature_type: str = 'dummies',  # NEW
    exclude_vars: Optional[List[str]] = None,
    columns: Union[None, str, List[str], Callable] = None
) -> "Recipe":
    """
    Args:
        feature_type: 'dummies' (binary only), 'interactions' (dummy*value), or 'both'
    """
```

---

## Usage Examples

### Example 1: step_splitwise with Interactions

```python
from py_recipes import recipe
import pandas as pd
import numpy as np

# Data with threshold relationship
np.random.seed(42)
n = 200
x = np.random.randn(n)
y = 5 * (x > 0).astype(int) + 2 * x + np.random.randn(n) * 0.5

data = pd.DataFrame({'x': x, 'y': y})

# Create dummies only (default)
rec_dummies = recipe().step_splitwise(
    outcome='y',
    feature_type='dummies'
)
prepped = rec_dummies.prep(data)
baked = prepped.bake(data)
# Result: x_ge_0p0000 (binary 0/1)

# Create interactions only
rec_interactions = recipe().step_splitwise(
    outcome='y',
    feature_type='interactions'
)
prepped = rec_interactions.prep(data)
baked = prepped.bake(data)
# Result: x_ge_0p0000_x_x (dummy × original x value)

# Create both
rec_both = recipe().step_splitwise(
    outcome='y',
    feature_type='both'
)
prepped = rec_both.prep(data)
baked = prepped.bake(data)
# Result: x_ge_0p0000 (dummy) + x_ge_0p0000_x_x (interaction)
```

### Example 2: step_safe with Interactions

```python
from py_recipes import recipe
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np

# Train surrogate model
surrogate = GradientBoostingRegressor(n_estimators=50, random_state=42)
surrogate.fit(X_train, y_train)

# Create recipe with SAFE + interactions
rec = recipe().step_safe(
    surrogate_model=surrogate,
    outcome='target',
    penalty=3.0,
    feature_type='both',  # Create both dummies and interactions
    top_n=10
)

prepped = rec.prep(train_data)
baked = prepped.bake(test_data)

# Result columns:
# - x1_cp_1 (dummy)
# - x1_cp_1_x_x1 (interaction)
# - x1_cp_2 (dummy)
# - x1_cp_2_x_x1 (interaction)
```

### Example 3: Using with Workflows

```python
from py_workflows import workflow
from py_parsnip import linear_reg

# Workflow with interactions
rec = recipe().step_splitwise(
    outcome='sales',
    feature_type='interactions'
)

wf = workflow().add_recipe(rec).add_model(linear_reg())

fit = wf.fit(train_data)
predictions = fit.predict(test_data)
```

---

## Mathematical Background

### Interaction Interpretation

For a threshold at value `t`:

**Dummy only (feature_type='dummies'):**
```
y = β₀ + β₁·I(x ≥ t) + ε

I(x ≥ t) = 1 if x ≥ t, else 0
```
- Constant shift above threshold
- Same effect regardless of how far above threshold

**Interaction only (feature_type='interactions'):**
```
y = β₀ + β₁·[I(x ≥ t) × x] + ε
```
- No effect below threshold (interaction = 0)
- Linear effect above threshold (interaction = x)
- Effect proportional to x value

**Both (feature_type='both'):**
```
y = β₀ + β₁·I(x ≥ t) + β₂·[I(x ≥ t) × x] + ε
```
- Piecewise linear relationship
- Jump at threshold (β₁)
- Different slope after threshold (β₂)

---

## Test Coverage

### New Tests Added: 17 total

**step_splitwise (8 tests):**
1. `test_feature_type_dummies_default` - Default behavior
2. `test_feature_type_interactions_only` - Interactions only
3. `test_feature_type_both` - Both types
4. `test_feature_type_invalid` - Validation
5. `test_interaction_values_correct` - Verify math
6. `test_recipe_with_feature_type_interactions` - Recipe integration
7. `test_recipe_with_feature_type_both` - Recipe integration
8. `test_double_split_with_interactions` - Double-split case

**step_safe (9 tests):**
1. `test_feature_type_dummies_default` - Default behavior
2. `test_feature_type_interactions_only` - Interactions only
3. `test_feature_type_both` - Both types
4. `test_feature_type_invalid` - Validation
5. `test_interaction_values_correct` - Verify math
6. `test_recipe_with_feature_type_interactions` - Recipe integration
7. `test_recipe_with_feature_type_both` - Recipe integration
8. `test_categorical_with_interactions` - Categorical case
9. `test_top_n_with_feature_types` - Works with top_n

### Test Results
```bash
tests/test_recipes/test_safe.py - 39 passed (30 original + 9 new)
tests/test_recipes/test_splitwise.py - 34 passed (26 original + 8 new)
============================= 73 passed in 46.74s ==============================
```

---

## Backward Compatibility

✅ **Fully backward compatible**

- Default `feature_type='dummies'` maintains existing behavior
- All existing code continues to work without changes
- All original tests still pass

---

## Files Modified

### Core Implementation (3 files)
1. **py_recipes/steps/splitwise.py** (lines 95, 126-130, 396-484)
   - Added `feature_type` parameter
   - Updated `bake()` for all three modes
   - Added parameter validation

2. **py_recipes/steps/feature_extraction.py** (lines 122, 169-173, 714-841)
   - Added `feature_type` parameter
   - Updated `_transform_numeric_variable()`
   - Updated `_transform_categorical_variable()`
   - Added parameter validation

3. **py_recipes/recipe.py** (lines 841-997)
   - Updated `step_splitwise()` method signature
   - Updated `step_safe()` method signature
   - Added examples to docstrings

### Tests (2 files)
1. **tests/test_recipes/test_splitwise.py** (+155 lines)
   - Added `TestStepSplitwiseFeatureTypes` class
   - 8 comprehensive tests

2. **tests/test_recipes/test_safe.py** (+211 lines)
   - Added `TestStepSafeFeatureTypes` class
   - 9 comprehensive tests

---

## Technical Notes

### Interaction Value Calculation

**For numeric variables:**
```python
interaction = dummy × original_value
```
- If dummy = 0: interaction = 0 (below threshold)
- If dummy = 1: interaction = original_value (above threshold)

**For categorical variables:**
```python
label_encoded = pd.factorize(X_col)[0]  # 0, 1, 2, ...
interaction = dummy × label_encoded
```
- Provides numeric representation for categorical interactions

### Column Naming Strategy

Both steps use sanitized naming for patsy compatibility:
- Replace negative sign with 'm': `-5.0` → `m5p0000`
- Replace decimal point with 'p': `5.0` → `5p0000`
- Format to 4 decimal places: `5.123456` → `5p1235`

This ensures column names work in formulas without quoting.

---

## Performance Considerations

**Memory Impact:**
- `feature_type='dummies'`: No change
- `feature_type='interactions'`: Same memory as dummies (1 feature per changepoint)
- `feature_type='both'`: 2× memory (2 features per changepoint)

**Computation Impact:**
- Minimal overhead (one multiplication per interaction)
- Prep phase unchanged (decision logic same for all feature_type values)

---

## Future Enhancements (Optional)

1. **Polynomial Interactions**: Add higher-order terms
   ```python
   feature_type='polynomial'  # dummy, dummy*x, dummy*x²
   ```

2. **Custom Interaction Functions**: User-defined transformations
   ```python
   feature_type='custom'
   interaction_fn=lambda dummy, x: dummy * np.log(x + 1)
   ```

3. **Automatic Feature Selection**: Choose feature_type based on model type
   ```python
   feature_type='auto'  # 'dummies' for trees, 'both' for linear
   ```

---

**Implementation Date:** 2025-11-09
**Status:** Complete and tested ✅
**Total Tests:** 73 passing (39 SAFE + 34 Splitwise)
**Backward Compatible:** Yes ✅
