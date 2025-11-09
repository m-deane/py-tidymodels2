# step_safe() Quick Reference

**Status:** ✅ Production Ready | **Tests:** 29/29 Passing | **Date:** 2025-11-09

---

## Quick Start

```python
from py_recipes import recipe
from py_workflows import workflow
from py_parsnip import linear_reg
from sklearn.ensemble import GradientBoostingRegressor

# Fit surrogate model (REQUIRED)
surrogate = GradientBoostingRegressor(n_estimators=100, max_depth=3)
surrogate.fit(train_data.drop('target', axis=1), train_data['target'])

# Basic usage
rec = recipe().step_safe(
    surrogate_model=surrogate,
    outcome='target',
    penalty=3.0
)

wf = workflow().add_recipe(rec).add_model(linear_reg())
fit = wf.fit(train_data)
preds = fit.predict(test_data)
```

---

## What It Does

Uses a complex surrogate model to transform features into interpretable intervals/categories:
- **Numeric variables**: Detects changepoints in partial dependence plots → intervals
- **Categorical variables**: Clusters similar levels → merged categories
- **Output**: One-hot encoded transformed features with p-1 scheme

---

## Key Parameters

| Parameter | Default | Description | Typical Range |
|-----------|---------|-------------|---------------|
| `surrogate_model` | **required** | Pre-fitted surrogate model | - |
| `outcome` | **required** | Outcome variable name | - |
| `penalty` | 3.0 | Changepoint penalty (higher = fewer) | 0.1-10.0 |
| `pelt_model` | 'l2' | Cost function for Pelt algorithm | 'l2', 'l1', 'rbf' |
| `no_changepoint_strategy` | 'median' | What to do if no changepoints | 'median', 'drop' |
| `keep_original_cols` | False | Keep original columns too | True/False |
| `top_n` | None | Select top N important features | int or None |
| `grid_resolution` | 1000 | PDP grid points | 100-5000 |

**Conservative:** Higher `penalty`, use `'l1'`, `'drop'` strategy
**Aggressive:** Lower `penalty`, use `'l2'`, `'median'` strategy

---

## Common Patterns

### 1. Basic Transformation
```python
rec = recipe().step_safe(
    surrogate_model=fitted_surrogate,
    outcome='price'
)
```

### 2. Conservative (Fewer Features)
```python
rec = recipe().step_safe(
    surrogate_model=fitted_surrogate,
    outcome='sales',
    penalty=10.0,  # Fewer changepoints
    no_changepoint_strategy='drop'  # Remove features with no changepoints
)
```

### 3. Feature Selection (Top N)
```python
rec = recipe().step_safe(
    surrogate_model=fitted_surrogate,
    outcome='revenue',
    penalty=2.0,  # More features initially
    top_n=10  # Select top 10 by importance
)
```

### 4. Keep Original Features
```python
rec = recipe().step_safe(
    surrogate_model=fitted_surrogate,
    outcome='target',
    keep_original_cols=True  # Keep x1, x2, x3 alongside SAFE features
)
```

### 5. Inspect Transformations
```python
prepped = rec.prep(train_data)
safe_step = prepped.prepared_steps[0]

# Get transformations
transformations = safe_step.get_transformations()
for var, info in transformations.items():
    print(f"{var}: {info['type']}")
    if info['type'] == 'numeric':
        print(f"  Changepoints: {info['changepoints']}")
    else:
        print(f"  Merged levels: {info['merged_levels']}")

# Get feature importances
importances = safe_step.get_feature_importances()
print(importances.head())
```

---

## Column Naming Convention

Transformed columns use patsy-safe naming:

| Transformation | Column Name Example | Meaning |
|----------------|---------------------|---------|
| Numeric (positive) | `x1_0p50_to_1p23` | 0.50 ≤ x1 < 1.23 |
| Numeric (negative) | `x2_m1p50_to_0p00` | -1.50 ≤ x2 < 0.00 |
| Numeric (last interval) | `x3_2p34_to_Inf` | x3 ≥ 2.34 |
| Categorical | `category_A_B` | Levels A and B merged |

**Naming:** `-` → `m` (minus), `.` → `p` (point), `_to_` separates intervals

---

## When to Use

✅ **Use when:**
- Want to transfer knowledge from complex model to simple model
- Need interpretable features from black-box model
- Have no domain knowledge for manual feature engineering
- Want data-driven threshold detection
- Complex model overfits but captures useful patterns

❌ **Don't use when:**
- Surrogate model is not fitted properly
- Very small datasets (< 50 observations)
- Simple linear relationships (no complex patterns to extract)
- Computational cost is prohibitive (large datasets)

---

## Workflow Integration

### With Linear Model
```python
from py_parsnip import linear_reg

rec = recipe().step_safe(
    surrogate_model=gradient_boosting_surrogate,
    outcome='y',
    penalty=3.0
)

wf = workflow().add_recipe(rec).add_model(linear_reg())
fit = wf.fit(train)
```

### With Other Models
```python
from py_parsnip import rand_forest

# SAFE features can be used with any model
wf = workflow().add_recipe(rec).add_model(
    rand_forest().set_mode('regression')
)
```

### Combined with Other Steps
```python
rec = (
    recipe()
    .step_safe(
        surrogate_model=surrogate,
        outcome='y',
        top_n=15
    )
    .step_normalize(all_numeric_predictors())
    .step_pca(num_comp=10)
)
```

---

## Surrogate Model Requirements

**Must have:**
- ✅ `predict()` method for regression
- ✅ OR `predict_proba()` method for classification
- ✅ Be pre-fitted before creating step_safe()

**Recommended surrogate models:**
- GradientBoostingRegressor (sklearn)
- RandomForestRegressor (sklearn)
- XGBRegressor (xgboost)
- LGBMRegressor (lightgbm)
- CatBoostRegressor (catboost)

**Example:**
```python
from sklearn.ensemble import GradientBoostingRegressor

# MUST fit before step_safe()
surrogate = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1
)
surrogate.fit(X_train, y_train)

# Now can use in step_safe()
rec = recipe().step_safe(
    surrogate_model=surrogate,  # ✅ Fitted
    outcome='y'
)
```

---

## Troubleshooting

### No changepoints detected
- **Try:** Lower `penalty` to 1.0-2.0
- **Try:** Change `pelt_model` to 'l1' or 'rbf'
- **Check:** Data has non-linear patterns (scatter plots)

### Too many features created
- **Try:** Increase `penalty` to 5.0+
- **Try:** Use `top_n` parameter
- **Try:** Set `no_changepoint_strategy='drop'`

### Sklearn feature name errors
- **Fixed automatically:** Column reordering handled internally
- **If persists:** Check surrogate was fitted on same data structure

### Workflow can't find outcome
- **Fixed automatically:** Outcome column preserved
- **If persists:** Check outcome column name matches

### Slow performance
- **Try:** Reduce `grid_resolution` to 500 or 300
- **Try:** Use `top_n` to limit features
- **Try:** Use simpler surrogate model (fewer trees)

---

## Dependencies

**Required packages:**
```bash
pip install ruptures kneed scipy
```

- `ruptures`: Pelt changepoint detection
- `kneed`: Elbow detection for clustering
- `scipy`: Hierarchical clustering (usually already installed)

---

## Performance

- **Speed:** O(p × n × grid_resolution) where p = variables, n = observations
- **300 obs × 3 vars × 1000 grid:** ~2-3 seconds
- **1000 obs × 10 vars × 1000 grid:** ~30-40 seconds
- **Memory:** Moderate - stores PDPs (O(p × grid_resolution))

**Tips for speed:**
- Reduce `grid_resolution` for faster computation
- Use `top_n` to limit output features
- Use smaller/simpler surrogate model

---

## Comparison with step_splitwise()

| Aspect | step_safe() | step_splitwise() |
|--------|-------------|------------------|
| **Approach** | Surrogate model PDP | Direct outcome-based |
| **Flexibility** | Any surrogate model | Decision trees only |
| **Complexity** | Higher | Lower |
| **Categorical** | Yes (clustering) | No |
| **Dependencies** | ruptures, kneed, scipy | sklearn only |
| **Speed** | Slower (PDP computation) | Faster |
| **Use case** | Transfer complex → simple | Direct threshold detection |

**When to use SAFE:** Want to leverage existing complex model
**When to use SplitWise:** Want direct, fast threshold detection

---

## Test Coverage

**29 tests, all passing in 29.58 seconds**

- Parameter validation: 8 tests
- Prep functionality: 4 tests
- Bake functionality: 5 tests
- Recipe integration: 3 tests
- Categorical handling: 1 test
- Edge cases: 3 tests
- Feature importances: 3 tests
- Workflow integration: 2 tests

---

## Files

- **Implementation:** `py_recipes/steps/feature_extraction.py` (731 lines)
- **Tests:** `tests/test_recipes/test_safe.py` (591 lines)
- **Full Docs:** `.claude_debugging/STEP_SAFE_IMPLEMENTATION.md`

---

## Reference

SAFE Library: https://github.com/ModelOriented/SAFE

---

**Version:** 1.0 | **Status:** Production Ready ✅
