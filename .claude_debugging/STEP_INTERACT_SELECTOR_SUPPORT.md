# step_interact() Selector Support

**Date:** 2025-11-09
**Feature:** Added selector support to `step_interact()`
**Status:** ✅ COMPLETE

---

## Overview

`step_interact()` now supports selector functions like `all_numeric_predictors()`, automatically creating all pairwise interaction terms for selected columns.

---

## New Usage

### Option 1: Explicit Pairs (Original)
```python
rec = recipe().step_interact([("x1", "x2"), ("x1", "x3")])
# Creates: x1_x_x2, x1_x_x3
```

### Option 2: List of Columns (Original)
```python
rec = recipe().step_interact(["x1", "x2", "x3"])
# Creates: x1_x_x2, x1_x_x3, x2_x_x3 (all pairs)
```

### Option 3: Selector Function (NEW!)
```python
from py_recipes.selectors import all_numeric_predictors

rec = recipe().step_interact(all_numeric_predictors())
# Creates all pairwise interactions between numeric predictors
# With 3 predictors: creates 3 interactions (3 choose 2)
# With 4 predictors: creates 6 interactions (4 choose 2)
# With 5 predictors: creates 10 interactions (5 choose 2)
```

---

## Examples

### Example 1: All Numeric Predictors

```python
import pandas as pd
from py_recipes import recipe
from py_recipes.selectors import all_numeric_predictors

data = pd.DataFrame({
    'x1': [1, 2, 3],
    'x2': [4, 5, 6],
    'x3': [7, 8, 9],
    'target': [10, 11, 12]
})

rec = recipe().step_interact(all_numeric_predictors())
prepped = rec.prep(data)
baked = prepped.bake(data)

print(baked.columns)
# Output: ['x1', 'x2', 'x3', 'target',
#          'x1_x_x2', 'x1_x_x3', 'x2_x_x3']
```

### Example 2: Pattern Matching

```python
from py_recipes.selectors import starts_with

# Create interactions for all features starting with 'temp'
rec = recipe().step_interact(starts_with('temp_'))

# If you have: temp_morning, temp_noon, temp_evening
# Creates: temp_morning_x_temp_noon, temp_morning_x_temp_evening,
#          temp_noon_x_temp_evening
```

### Example 3: Custom Separator

```python
rec = recipe().step_interact(
    all_numeric_predictors(),
    separator="_times_"
)

# Creates: x1_times_x2, x1_times_x3, x2_times_x3
```

### Example 4: Combined with Other Steps

```python
rec = (recipe()
    .step_normalize(all_numeric_predictors())           # Normalize first
    .step_interact(all_numeric_predictors())            # Create interactions
    .step_corr(threshold=0.9)                           # Remove multicollinearity
)

# Pipeline: normalize → create all interactions → remove correlated
```

---

## Combinatorial Growth

Number of interaction terms created:

| Number of Predictors | Interactions (n choose 2) |
|---------------------|---------------------------|
| 2 | 1 |
| 3 | 3 |
| 4 | 6 |
| 5 | 10 |
| 6 | 15 |
| 7 | 21 |
| 10 | 45 |
| 20 | 190 |

**Formula:** For n predictors, creates n×(n-1)/2 interactions

**Warning:** With many predictors, this can create a large number of features. Consider:
- Using `step_corr()` afterward to remove redundant features
- Using specific column lists or pattern matching instead of `all_numeric_predictors()`
- Using feature selection steps

---

## Implementation Details

### Files Modified

1. **`py_recipes/steps/interactions.py`**
   - Added `Union[List[tuple], List[str], Callable]` type hint (line 29)
   - Added selector resolution in `prep()` method (lines 43-56)
   - Updated docstring with examples (lines 22-26)

2. **`py_recipes/recipe.py`**
   - Updated `step_interact()` signature to accept `Callable` (line 1148)
   - Updated docstring with selector examples (lines 1155-1172)
   - Removed manual combination logic (now handled in StepInteract.prep())

### Code Flow

```python
# User code:
rec = recipe().step_interact(all_numeric_predictors())

# During prep():
1. StepInteract.prep() is called with data
2. Detects input is callable (selector function)
3. Resolves selector: resolve_selector(all_numeric_predictors(), data)
   → Returns: ['x1', 'x2', 'x3']
4. Creates all pairs: combinations(['x1', 'x2', 'x3'], 2)
   → Returns: [('x1', 'x2'), ('x1', 'x3'), ('x2', 'x3')]
5. Validates columns exist in data
6. Creates feature names: ['x1_x_x2', 'x1_x_x3', 'x2_x_x3']

# During bake():
1. For each pair, multiply columns
2. Add interaction features to DataFrame
```

---

## Test Results

All tests passing:

```
[Test 1] Explicit pairs
✅ PASS: Explicit pairs work

[Test 2] List of columns
✅ PASS: All pairwise interactions created (3 choose 2 = 3)

[Test 3] Selector function (NEW)
✅ PASS: Selector created all pairwise interactions (3 choose 2 = 3)
✅ PASS: Interaction values are correct

[Test 4] Custom separator
✅ PASS: Custom separator works

[Test 5] Combinatorial explosion check
✅ PASS: Correct number of interactions for 7 predictors (21)
```

---

## Available Selectors

You can use any selector with `step_interact()`:

**Numeric Selectors:**
- `all_numeric()` - All numeric columns
- `all_numeric_predictors()` - All numeric columns except outcome
- `all_integer()` - Integer columns only
- `all_double()` - Float columns only

**Pattern Matching:**
- `starts_with(prefix)` - Columns starting with prefix
- `ends_with(suffix)` - Columns ending with suffix
- `contains(substring)` - Columns containing substring
- `matches(pattern)` - Regex pattern matching

**Custom:**
- `where(lambda s: condition)` - Custom predicate function

**Example with Pattern Matching:**
```python
# Create interactions only for lag features
rec = recipe().step_interact(contains('_lag'))

# Create interactions for temperature variables
rec = recipe().step_interact(starts_with('temp_'))
```

---

## Comparison: step_poly() vs step_interact()

| Feature | step_poly() | step_interact() |
|---------|-------------|-----------------|
| **Purpose** | Polynomial terms (x², x³) | Multiplicative interactions (x₁×x₂) |
| **Single column** | x → x² | (needs 2+ columns) |
| **Multiple columns** | x, y → x², y² | x, y → x×y |
| **Degree 2** | x² + y² | x×y |
| **Degree 2 + interactions** | x², y², x×y | x×y (always) |
| **Selector support** | ✅ Yes | ✅ Yes (NEW) |
| **inplace parameter** | ✅ Yes | ❌ No (always adds) |

**Use Together:**
```python
rec = (recipe()
    # Create both polynomials and interactions manually
    .step_poly(all_numeric_predictors(), degree=2, inplace=False)
    .step_interact(all_numeric_predictors())
)

# OR use step_poly with interactions enabled:
rec = recipe().step_poly(
    all_numeric_predictors(),
    degree=2,
    include_interactions=True,  # Creates both x² and x×y
    inplace=False
)
```

---

## Performance Considerations

### Memory Usage

With n predictors:
- Polynomial terms: n features (x₁², x₂², ...)
- Interaction terms: n×(n-1)/2 features (x₁×x₂, x₁×x₃, ...)
- Both: n + n×(n-1)/2 features

**Example:**
- 10 predictors → 10 + 45 = 55 features
- 20 predictors → 20 + 190 = 210 features

### Recommendations

1. **For many predictors**, use feature selection:
   ```python
   rec = (recipe()
       .step_interact(all_numeric_predictors())
       .step_corr(threshold=0.9)  # Remove redundant
       .step_pca(num_comp=10)      # Reduce dimensions
   )
   ```

2. **For specific interactions**, use explicit pairs:
   ```python
   rec = recipe().step_interact([
       ("temp", "humidity"),
       ("speed", "distance"),
       ("price", "quantity")
   ])
   ```

3. **For pattern-based**, use selectors:
   ```python
   rec = recipe().step_interact(starts_with("lag_"))
   ```

---

## Backward Compatibility

✅ **Fully backward compatible**

- Existing code with explicit pairs works unchanged
- Existing code with column lists works unchanged
- New selector support adds functionality without breaking changes

---

## Related Features

Works seamlessly with:
- ✅ All selector functions
- ✅ Custom separators
- ✅ Recipe chaining
- ✅ Workflow integration
- ✅ Feature selection steps

---

## Summary

`step_interact()` now accepts:

1. **Explicit pairs**: `[("x1", "x2"), ("x1", "x3")]`
2. **Column lists**: `["x1", "x2", "x3"]` → creates all pairs
3. **Selectors**: `all_numeric_predictors()` → creates all pairs

All three modes create multiplicative interaction terms (x₁ × x₂) for the specified or selected columns.

**Status:** COMPLETE - Feature implemented, tested, and documented.
