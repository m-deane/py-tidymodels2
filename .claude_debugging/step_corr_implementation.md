# step_corr() Implementation Summary

**Date:** 2025-11-07
**Component:** py_recipes
**Feature:** Correlation-based feature filtering

## Overview

Implemented `step_corr()` for removing highly correlated features from datasets. This step helps reduce multicollinearity by identifying pairs of features with correlation above a threshold and removing one from each pair.

## Implementation Details

### Files Modified

1. **`py_recipes/steps/feature_selection.py`**
   - Added `StepCorr` class
   - Added `PreparedStepCorr` class
   - Added import for `resolve_selector`

2. **`py_recipes/recipe.py`**
   - Added `step_corr()` method

3. **`py_recipes/steps/__init__.py`**
   - Exported `StepCorr` and `PreparedStepCorr`

### Files Created

1. **`tests/test_recipes/test_step_corr.py`**
   - 23 comprehensive tests covering all functionality
   - All tests passing

2. **`examples/step_corr_demo.py`**
   - 7 example scenarios demonstrating usage patterns

## Algorithm

The correlation filtering algorithm:

1. **Column Selection**: Resolves columns using `resolve_selector()` to support:
   - `None`: All numeric columns (default)
   - `str`: Single column name
   - `List[str]`: Explicit list of columns
   - `Callable`: Selector functions like `all_numeric()`, `all_predictors()`

2. **Correlation Calculation**: Computes correlation matrix using specified method:
   - `pearson`: Linear correlation (default)
   - `spearman`: Monotonic correlation (rank-based)
   - `kendall`: Tau correlation

3. **Pair Identification**: Finds all pairs with `abs(correlation) > threshold`

4. **Feature Removal**: For each correlated pair:
   - Calculates mean absolute correlation with all other features
   - Removes the feature with **higher** mean correlation
   - Ensures each feature is only removed once

5. **Non-numeric Preservation**: All non-numeric columns are preserved unchanged

## Key Features

### 1. Flexible Column Selection

```python
# Default: all numeric columns
recipe().step_corr(threshold=0.9)

# Specific columns
recipe().step_corr(columns=['x1', 'x2', 'x3'])

# Using selectors
from py_recipes import all_numeric, all_numeric_predictors
recipe().step_corr(columns=all_numeric())
recipe().step_corr(columns=all_numeric_predictors())
```

### 2. Multiple Correlation Methods

```python
# Pearson: linear relationships (default)
recipe().step_corr(threshold=0.9, method='pearson')

# Spearman: monotonic relationships
recipe().step_corr(threshold=0.9, method='spearman')

# Kendall: robust to outliers
recipe().step_corr(threshold=0.9, method='kendall')
```

### 3. Customizable Thresholds

```python
# Conservative: remove only very high correlations
recipe().step_corr(threshold=0.95)

# Standard: default threshold
recipe().step_corr(threshold=0.9)

# Aggressive: remove moderate correlations
recipe().step_corr(threshold=0.7)
```

### 4. Pipeline Integration

```python
# Chain with other steps
rec = (recipe(data)
       .step_normalize()           # Normalize first
       .step_corr(threshold=0.9)   # Then remove correlations
       .step_pca(num_comp=5))      # Finally apply PCA
```

## Usage Examples

### Basic Usage

```python
import pandas as pd
import numpy as np
from py_recipes import recipe

# Create data with correlated features
np.random.seed(42)
x1 = np.random.randn(100)
x2 = x1 + np.random.randn(100) * 0.01  # Highly correlated
x3 = np.random.randn(100)              # Independent

data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})

# Remove correlations
rec = recipe(data).step_corr(threshold=0.9)
rec_prepped = rec.prep(data)
result = rec_prepped.bake(data)

print(f"Original: {data.shape[1]} features")
print(f"After step_corr: {result.shape[1]} features")
# Output: Original: 3 features
#         After step_corr: 2 features (x1 or x2 removed)
```

### With Multiple Correlated Groups

```python
# Create multiple groups of correlated features
base1 = np.random.randn(100)
base2 = np.random.randn(100)

data = pd.DataFrame({
    'g1_x1': base1,
    'g1_x2': base1 + np.random.randn(100) * 0.01,
    'g1_x3': base1 + np.random.randn(100) * 0.01,
    'g2_x1': base2,
    'g2_x2': base2 + np.random.randn(100) * 0.01,
    'independent': np.random.randn(100)
})

rec = recipe(data).step_corr(threshold=0.9)
result = rec.prep(data).bake(data)

# Removes 2 from group 1, 1 from group 2, keeps independent
print(f"Retained: {result.shape[1]} features")
# Output: Retained: 3 features
```

### Selective Filtering

```python
# Only check specific columns
rec = recipe(data).step_corr(
    columns=['x1', 'x2', 'x3'],  # Only these columns
    threshold=0.85
)

# Or use selectors
from py_recipes import all_numeric
rec = recipe(data).step_corr(
    columns=all_numeric(),
    threshold=0.9
)
```

## Comparison with step_select_corr()

The existing `step_select_corr()` serves a different purpose:

| Feature | `step_corr()` | `step_select_corr()` |
|---------|---------------|---------------------|
| **Purpose** | Remove multicollinearity | Select relevant features |
| **Outcome Required** | No | Yes |
| **Criteria** | Correlation between predictors | Correlation with outcome |
| **Use Case** | Preprocessing before modeling | Feature selection |
| **Method** | Removes one from each correlated pair | Removes based on outcome correlation |

Example:

```python
# step_corr: removes multicollinearity
recipe().step_corr(threshold=0.9)

# step_select_corr: keeps features correlated with outcome
recipe().step_select_corr(outcome='y', threshold=0.3, method='outcome')
```

## Test Coverage

### Test Suite: `tests/test_recipes/test_step_corr.py`

**23 tests covering:**

1. **Basic Functionality (5 tests)**
   - Basic correlation filtering
   - No removal when uncorrelated
   - Different thresholds
   - Column subset selection
   - Perfect correlation

2. **Correlation Methods (3 tests)**
   - Pearson correlation
   - Spearman correlation
   - Kendall correlation

3. **Complex Scenarios (5 tests)**
   - Multiple correlated pairs
   - Three-way correlation
   - Negative correlation
   - Higher mean correlation removal
   - Multiple groups

4. **Integration (5 tests)**
   - Non-numeric column preservation
   - Selector function usage
   - Chaining with other steps
   - New data baking
   - Column order preservation

5. **Edge Cases (5 tests)**
   - Single column
   - Empty columns
   - Missing columns in bake
   - Direct PreparedStepCorr usage
   - Empty removal list

**All 23 tests passing.**

## Performance Characteristics

### Time Complexity

- **prep()**: O(n * m^2) where n = rows, m = selected columns
  - Correlation matrix calculation: O(n * m^2)
  - Pair comparison: O(m^2)

- **bake()**: O(1)
  - Simple column dropping

### Space Complexity

- **prep()**: O(m^2) for correlation matrix
- **bake()**: O(n * m) for data copy

### Recommendations

- For large datasets (m > 100 features):
  - Use `columns` parameter to limit checked features
  - Consider higher threshold values
  - Use `method='pearson'` (fastest)

## Design Decisions

### 1. Removal Strategy

**Decision**: Remove feature with **higher** mean correlation

**Rationale**:
- Feature with higher mean correlation is more redundant across the dataset
- Keeping feature with lower mean correlation preserves more unique information
- Aligns with R's caret::findCorrelation default behavior

### 2. Absolute Correlation

**Decision**: Use `abs(correlation)` for threshold comparison

**Rationale**:
- Negative correlation indicates multicollinearity (e.g., x and -x)
- Both positive and negative correlation reduce model stability
- Common practice in multicollinearity detection

### 3. One-Time Removal

**Decision**: Each feature can only be removed once

**Rationale**:
- Prevents cascading removals
- More predictable behavior
- Preserves as many features as possible while meeting threshold

### 4. Column Selection Integration

**Decision**: Use `resolve_selector()` for column specification

**Rationale**:
- Consistent with other py_recipes steps
- Supports all selector patterns (None, str, list, callable)
- Enables powerful column selection like `all_numeric_predictors()`

## Documentation

### Docstrings

All classes and methods have comprehensive docstrings with:
- Description
- Parameters with types
- Return values
- Usage examples

### Examples

Created `examples/step_corr_demo.py` with 7 scenarios:
1. Basic correlation filtering
2. Different thresholds
3. Correlation methods
4. Column selection
5. Chaining with other steps
6. Multiple correlated groups
7. Comparison with step_select_corr()

## Integration with py-tidymodels

### Recipe Integration

```python
from py_recipes import recipe
from py_workflows import workflow
from py_parsnip import linear_reg

# Create preprocessing recipe
rec = (recipe()
       .step_normalize()
       .step_corr(threshold=0.9)
       .step_dummy(['category']))

# Use in workflow
wf = (workflow()
      .add_recipe(rec)
      .add_model(linear_reg().set_engine("sklearn")))

# Fit and predict
wf_fit = wf.fit(train_data, formula="y ~ .")
predictions = wf_fit.predict(test_data)
```

### Tuning Support

Compatible with py-tune for threshold optimization:

```python
from py_tune import tune, tune_grid, grid_regular

# Mark threshold for tuning
rec = recipe().step_corr(threshold=tune())

# Define grid
grid = grid_regular({
    'threshold': {'range': (0.7, 0.95), 'levels': 5}
})

# Tune
results = tune_grid(workflow, resamples, grid=grid)
best = results.select_best('rmse')
```

## Future Enhancements

Potential improvements for future versions:

1. **VIF-Based Removal**: Add option to use Variance Inflation Factor instead of correlation
2. **Cluster-Based Removal**: Remove features by correlation clusters
3. **Importance Weighting**: Consider feature importance when choosing which to remove
4. **Parallel Processing**: Speed up correlation calculation for large datasets
5. **Progress Reporting**: Add verbose mode for large feature sets

## Summary

The `step_corr()` implementation:

- **Completes the requirement** for correlation-based feature filtering
- **Follows py_recipes patterns** for step implementation
- **Provides flexible usage** via selectors and parameters
- **Has comprehensive test coverage** (23 tests, all passing)
- **Integrates seamlessly** with existing py-tidymodels components
- **Includes clear documentation** and practical examples

The feature is production-ready and can be used immediately in modeling workflows.
