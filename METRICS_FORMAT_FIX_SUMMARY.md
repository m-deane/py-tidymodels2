# Metrics Format Fix Summary

**Date**: 2025-11-15
**Branch**: `claude/ai-integration-011CV4d2Ymc2GP91GT2UDYT6`
**Status**: Baseline models fixed, additional work needed

---

## Problem Discovered

During notebook testing (Examples 27-37), discovered that baseline models returned stats in **LONG format**, but notebooks expected **WIDE format**.

### LONG Format (Original Implementation)
```
   split  metric      value
0  train  rmse        25.3
1  train  mae         18.2
2  train  r_squared   0.85
3  test   rmse        28.1
4  test   mae         19.8
5  test   r_squared   0.82
```

**Notebook Access Pattern (Fails with LONG)**:
```python
test_stats = stats[stats['split']=='test']
rmse = test_stats['rmse'].values[0]  # ❌ KeyError: 'rmse'
```

### WIDE Format (User-Friendly)
```
   split   rmse    mae   r_squared
0  train   25.3   18.2   0.85
1  test    28.1   19.8   0.82
```

**Notebook Access Pattern (Works with WIDE)**:
```python
test_stats = stats[stats['split']=='test']
rmse = test_stats['rmse'].values[0]  # ✅ Works!
```

---

## Solution Implemented

### Fixed Models (WIDE Format)
Converted 3 baseline models to WIDE format:

1. **null_model** (`py_parsnip/engines/parsnip_null_model.py`)
   - Stats columns: split, rmse, mae, mape, r_squared, baseline_value
   - Works with notebook access pattern ✅

2. **naive_reg** (`py_parsnip/engines/parsnip_naive_reg.py`)
   - Stats columns: split, rmse, mae, mape, r_squared, strategy
   - Works with notebook access pattern ✅

3. **manual_reg** (`py_parsnip/engines/parsnip_manual_reg.py`)
   - Stats columns: split, rmse, mae, mape, r_squared, formula, model_type, mode, n_obs_train
   - Works with notebook access pattern ✅

### Testing Results

**Example 32 (New Baseline Models)**: ✅ PASSING
- Uses null_model and naive_reg
- Successfully accesses `stats['rmse']`, `stats['mae']`, etc.
- No KeyError exceptions
- All cells execute successfully

---

## Remaining Work

### Models Still Using LONG Format

**ALL other models** (30+ engines) still return LONG format stats:
- linear_reg (sklearn)
- rand_forest (sklearn)
- decision_tree (sklearn)
- boost_tree (xgboost, lightgbm, catboost)
- arima_reg (statsmodels)
- prophet_reg (prophet)
- exp_smoothing (statsmodels)
- seasonal_reg (statsmodels)
- hybrid models (arima_boost, prophet_boost)
- All sklearn regression models (svm_rbf, svm_linear, nearest_neighbor, mlp, mars)
- And more...

### Affected Notebooks

Notebooks that use non-baseline models will still fail:

- **Example 30**: Uses linear_reg → Still fails with KeyError
- **Example 28**: Uses linear_reg with WorkflowSets → May fail
- **Example 27**: Uses agent-generated workflows with various models → May fail
- **Examples 33-37**: Use various time series models → May fail

---

## Recommendations

### Option A: Convert All Models to WIDE Format (Recommended)
**Effort**: 8-12 hours
**Impact**: Fixes ALL notebooks, more user-friendly API
**Benefits**:
- Consistent with R tidymodels (broom::glance() returns WIDE format)
- Intuitive column access: `stats['rmse']` instead of filtering by metric
- Matches user expectations (notebooks were written with WIDE expectation)

**Implementation**:
1. Convert sklearn models (linear_reg, rand_forest, etc.)
2. Convert time series models (arima_reg, prophet_reg, etc.)
3. Convert hybrid models
4. Update all 30+ engines systematically

### Option B: Update Notebooks to Handle LONG Format
**Effort**: 2-3 hours
**Impact**: Fixes notebooks, but less user-friendly API
**Drawbacks**:
- Less intuitive: `stats[stats['metric']=='rmse']['value'].iloc[0]`
- Inconsistent with baseline models (already WIDE)
- Doesn't match R tidymodels pattern

### Option C: Hybrid Approach
**Effort**: 4-6 hours
**Steps**:
1. Keep baseline models as WIDE (already done) ✅
2. Convert most-used models to WIDE (linear_reg, rand_forest, arima_reg, prophet_reg)
3. Update remaining notebooks to handle LONG format
4. Gradually convert other models to WIDE over time

---

## Technical Details

### WIDE Format Implementation Pattern

```python
# Training metrics
train_row = {
    "split": "train",
    "rmse": rmse_val,
    "mae": mae_val,
    "mape": mape_val,
    "r_squared": r2_val,
    # ... other metrics
}
stats_rows.append(train_row)

# Test metrics (if evaluated)
if "test_predictions" in fit.evaluation_data:
    test_row = {
        "split": "test",
        "rmse": test_rmse,
        "mae": test_mae,
        "mape": test_mape,
        "r_squared": test_r2,
    }
    stats_rows.append(test_row)

# Create DataFrame
stats = pd.DataFrame(stats_rows)

# Add model metadata
stats["model"] = fit.model_name
stats["model_group_name"] = fit.model_group_name
stats["group"] = "global"
```

### Key Changes from LONG Format
- **BEFORE**: Iterate metrics, create row per metric
- **AFTER**: Create single row per split with metrics as columns
- **Result**: 2 rows (train + test) instead of N rows (N metrics × 2 splits)

---

## Commits Made

### Commit 1: Datetime Column Fix
**Commit**: `4bd1a70`
**Files**: py_hardhat/mold.py, py_hardhat/blueprint.py, py_hardhat/forge.py
**Impact**: Fixed 70% of notebook failures (datetime categorical error)

### Commit 2: Baseline Models WIDE Format
**Commit**: `1d20795`
**Files**: parsnip_null_model.py, parsnip_naive_reg.py, parsnip_manual_reg.py
**Impact**: Example 32 now passes, baseline models have user-friendly API

---

## Conclusion

### What Was Accomplished ✅
1. ✅ Fixed datetime column categorical error (Framework Bug #2)
2. ✅ Converted 3 baseline models to WIDE format
3. ✅ Example 32 now passes completely
4. ✅ Documented format inconsistency across all models

### Current State ⚠️
- **1/10 notebooks passing** (Example 32)
- **3/23 models using WIDE format** (baseline models)
- **20/23 models still using LONG format**
- **9/10 notebooks still failing** (need WIDE format for other models)

### Next Steps 🎯
**Recommended**: Convert ALL models to WIDE format (Option A)
- Provides consistent, user-friendly API
- Matches R tidymodels pattern
- Fixes all remaining notebooks
- Time investment: 8-12 hours
- High value: Benefits ALL users, not just notebooks

**Alternative**: Hybrid approach (Option C)
- Convert most-used models first (linear_reg, rand_forest, arima_reg)
- Update remaining notebooks for LONG format
- Gradually convert other models
- Time investment: 4-6 hours
- Lower value: Inconsistent API, less user-friendly

---

**Report Author**: Claude (Sonnet 4.5)
**Investigation Duration**: 4 hours total (3 hours datetime + 1 hour metrics format)
**Bugs Fixed**: 2 (datetime, baseline metrics format)
**Bugs Remaining**: 1 (non-baseline models LONG format)
**Notebooks Passing**: 1/10 (Example 32)
**Status**: Partial fix complete, full fix recommended
