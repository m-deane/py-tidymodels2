# Workflow Extract Methods Enhancement

**Date:** 2025-11-09
**Enhancement:** Added `extract_formula()` and `extract_preprocessed_data()` methods to WorkflowFit
**Status:** ✅ COMPLETE

---

## Problem

Users needed convenient methods to:
1. Extract the formula used by a fitted workflow
2. Get the transformed/preprocessed data to inspect what the model actually sees

**User Request:**
> "I also want to add to the library an extract_formula() method that will return the formula used by the workflow, i also want a convenience function to extract the transformed train and test data using the recipe in the workflow applied to a dataset"

---

## Solution

Added two new methods to the `WorkflowFit` class:

### 1. `extract_formula()` - Get Formula Used

Returns the formula string used for model fitting, whether specified explicitly or auto-generated from a recipe.

```python
wf_fit = workflow().add_formula("sales ~ price").add_model(spec).fit(train)
formula = wf_fit.extract_formula()
print(formula)  # 'sales ~ price'
```

### 2. `extract_preprocessed_data()` - Get Transformed Data

Applies the fitted preprocessor to data and returns the transformed DataFrame that the model actually sees.

```python
# Extract transformed training data
train_transformed = wf_fit.extract_preprocessed_data(train)

# Extract transformed test data
test_transformed = wf_fit.extract_preprocessed_data(test)

# Inspect what the model sees
print(train_transformed.columns)
print(train_transformed.head())
```

---

## Implementation Details

### Modified Files

**`py_workflows/workflow.py`:**

1. **Updated WorkflowFit dataclass (line 413):**
   ```python
   @dataclass
   class WorkflowFit:
       workflow: Workflow
       pre: Any
       fit: ModelFit
       post: Optional[Any] = None
       formula: Optional[str] = None  # NEW: Store formula
   ```

2. **Updated Workflow.fit() method (line 252):**
   ```python
   return WorkflowFit(
       workflow=self,
       pre=fitted_preprocessor,
       fit=model_fit,
       post=self.post,
       formula=formula  # NEW: Pass formula
   )
   ```

3. **Added extract_formula() method (lines 544-566):**
   ```python
   def extract_formula(self) -> str:
       """Extract the formula used for model fitting."""
       if self.formula is None:
           raise ValueError("No formula stored in WorkflowFit")
       return self.formula
   ```

4. **Added extract_preprocessed_data() method (lines 568-615):**
   ```python
   def extract_preprocessed_data(self, data: pd.DataFrame) -> pd.DataFrame:
       """Apply the fitted preprocessor to data and return transformed data."""
       if isinstance(self.pre, str):
           # Formula - use mold() to get preprocessed data
           from py_hardhat import mold
           molded = mold(self.pre, data)
           result = molded.predictors.copy()
           if molded.outcomes is not None and not molded.outcomes.empty:
               for col in molded.outcomes.columns:
                   result[col] = molded.outcomes[col]
           return result
       elif isinstance(self.pre, PreparedRecipe):
           # Recipe - use bake()
           return self.pre.bake(data)
       else:
           raise ValueError(f"Unknown preprocessor type: {type(self.pre)}")
   ```

---

## Usage Examples

### Example 1: Extract Formula (Explicit)

```python
from py_workflows import workflow
from py_parsnip import linear_reg

# Workflow with explicit formula
wf = (
    workflow()
    .add_formula("sales ~ price + advertising")
    .add_model(linear_reg().set_engine("sklearn"))
)

fit = wf.fit(train_data)

# Extract formula
formula = fit.extract_formula()
print(formula)
# Output: 'sales ~ price + advertising'
```

### Example 2: Extract Formula (Auto-Generated from Recipe)

```python
from py_workflows import workflow
from py_parsnip import linear_reg
from py_recipes import recipe

# Workflow with recipe (auto-generates formula)
rec = recipe().step_normalize().step_dummy()
wf = (
    workflow()
    .add_recipe(rec)
    .add_model(linear_reg().set_engine("sklearn"))
)

fit = wf.fit(train_data)

# Extract auto-generated formula
formula = fit.extract_formula()
print(formula)
# Output: 'y ~ x1 + x2 + x3 + category_A + category_B'
```

### Example 3: Inspect Transformed Data (Formula)

```python
# Fit workflow with formula
wf = workflow().add_formula("y ~ x1 + x2").add_model(spec)
fit = wf.fit(train)

# Extract transformed training data
train_transformed = fit.extract_preprocessed_data(train)
print(train_transformed.columns)
# Output: ['Intercept', 'x1', 'x2', 'y']

print(train_transformed.head())
#    Intercept        x1        x2         y
# 0        1.0  0.496714 -0.138264  0.647689
# 1        1.0 -0.234153  1.523030 -0.234137
# ...
```

### Example 4: Inspect Transformed Data (Recipe)

```python
from py_recipes import recipe

# Create recipe with normalization
rec = (
    recipe()
    .step_normalize()  # Normalize all numeric predictors
    .step_dummy()      # Encode categoricals
)

wf = workflow().add_recipe(rec).add_model(spec)
fit = wf.fit(train)

# Extract transformed data
train_transformed = fit.extract_preprocessed_data(train)

# Verify normalization applied
print(f"x1 mean: {train_transformed['x1'].mean():.4f}")  # ≈ 0
print(f"x1 std: {train_transformed['x1'].std():.4f}")    # ≈ 1

# Extract test data with same transformations
test_transformed = fit.extract_preprocessed_data(test)
```

### Example 5: Debug Preprocessing Pipeline

```python
# Complex recipe with multiple steps
rec = (
    recipe()
    .step_impute_median()
    .step_normalize()
    .step_pca(num_comp=3)
)

wf = workflow().add_recipe(rec).add_model(spec)
fit = wf.fit(train)

# Check transformations at each stage
print("Original data:")
print(train[['x1', 'x2', 'x3']].head())

print("\nTransformed data:")
transformed = fit.extract_preprocessed_data(train)
print(transformed.head())

print("\nNew columns after PCA:")
print([col for col in transformed.columns if col.startswith('PC')])
# Output: ['PC1', 'PC2', 'PC3']
```

---

## Use Cases

### 1. **Debugging Preprocessing**
Inspect what transformations were applied to your data:
```python
transformed = fit.extract_preprocessed_data(train)
print(f"Original shape: {train.shape}")
print(f"Transformed shape: {transformed.shape}")
print(f"New columns: {set(transformed.columns) - set(train.columns)}")
```

### 2. **Understanding Model Inputs**
See exactly what features the model receives:
```python
transformed = fit.extract_preprocessed_data(test)
print("Features model sees:")
print(transformed.drop(columns=['y']).describe())
```

### 3. **Reproducibility**
Store formula for documentation or reproduction:
```python
formula = fit.extract_formula()
print(f"Model formula: {formula}")
# Save to file or documentation
```

### 4. **Manual Analysis**
Perform manual analysis on transformed features:
```python
import matplotlib.pyplot as plt

transformed = fit.extract_preprocessed_data(train)

# Plot transformed features
plt.scatter(transformed['PC1'], transformed['PC2'], c=transformed['y'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA-Transformed Data')
plt.show()
```

---

## Behavior Details

### extract_formula()
- **With formula workflow**: Returns exact formula string provided
- **With recipe workflow**: Returns auto-generated formula (e.g., `"y ~ x1 + x2 + x3"`)
- **Datetime columns**: Excluded from auto-generated formulas (prevents categorical encoding errors)
- **Raises ValueError**: If no formula stored (shouldn't happen in normal usage)

### extract_preprocessed_data()
- **With formula workflow**:
  - Uses `mold()` from py_hardhat
  - Returns predictors + outcomes combined
  - Includes "Intercept" column (added by patsy)
- **With recipe workflow**:
  - Uses `bake()` from fitted recipe
  - Returns all transformed columns
  - Includes outcome column unchanged
- **Preserves index**: Output DataFrame has same index as input
- **Works with any data**: Can apply to train, test, or new data

---

## Integration with Existing Methods

The new methods complement existing extract methods:

```python
# Existing methods
model_fit = fit.extract_fit_parsnip()      # Get ModelFit
preprocessor = fit.extract_preprocessor()   # Get PreparedRecipe or formula
model_spec = fit.extract_spec_parsnip()    # Get ModelSpec
outputs, coeffs, stats = fit.extract_outputs()  # Get model outputs

# NEW methods
formula = fit.extract_formula()            # Get formula string
transformed = fit.extract_preprocessed_data(data)  # Get transformed data
```

---

## Test Results

All tests passing:

```
✅ Test 1: extract_formula() with explicit formula
   Formula extracted: 'y ~ x1 + x2'

✅ Test 2: extract_formula() with recipe (auto-generated)
   Formula extracted: 'y ~ x1 + x2 + x3'

✅ Test 3: extract_preprocessed_data() with formula
   Train data transformed: (80, 4) with columns ['Intercept', 'x1', 'x2', 'y']
   Test data transformed: (20, 4)

✅ Test 4: extract_preprocessed_data() with recipe
   Train data transformed: (80, 4) with normalized data
   x1 mean: 0.0000 (normalized)
   x1 std: 1.0063 (normalized)
```

---

## Backward Compatibility

✅ **Fully backward compatible**

- Existing WorkflowFit code: No changes needed
- New formula parameter has default: `formula: Optional[str] = None`
- New methods are additive, not breaking
- All existing tests pass without modification

---

## Summary

Two new convenience methods added to WorkflowFit:

1. **`extract_formula()`**:
   - Returns the formula used for model fitting
   - Works with both explicit formulas and recipe-generated formulas
   - Useful for documentation and reproducibility

2. **`extract_preprocessed_data(data)`**:
   - Applies fitted preprocessing to any data
   - Returns transformed DataFrame
   - Useful for debugging, inspection, and manual analysis
   - Works with both formula and recipe workflows

**Status:** COMPLETE - Methods implemented, tested, and documented.

---

## Documentation Files

- `.claude_debugging/WORKFLOW_EXTRACT_METHODS.md` (this file)
- `py_workflows/workflow.py` - Implementation
- Comprehensive docstrings with examples

**Restart Jupyter Kernel:** After reinstalling package, restart kernel to pick up changes.
