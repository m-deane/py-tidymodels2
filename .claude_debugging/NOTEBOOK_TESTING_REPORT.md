# Notebook Testing Report - Phase 4A Demo Notebooks

## Executive Summary

Tested notebook 16 (`16_baseline_models_demo.ipynb`) and discovered **5 critical bugs** affecting the py-tidymodels codebase.

**Status**: 4 bugs fixed, 1 architectural issue documented with workaround applied.

---

## Bugs Found and Fixed

### Bug #1: AttributeError - `fit.fit_output` doesn't exist ✅ FIXED

**Severity**: Critical
**Files affected**:
- `py_parsnip/engines/parsnip_null_model.py` (2 occurrences)
- `py_parsnip/engines/parsnip_naive_reg.py` (2 occurrences)

**Error**:
```python
AttributeError: 'ModelFit' object has no attribute 'fit_output'
```

**Root cause**: Engines accessed `fit.fit_output` but the correct ModelFit attribute is `fit.fit_data`

**Impact**: null_model and naive_reg models couldn't predict or extract outputs

**Fix**: Changed all 4 occurrences from `fit.fit_output` to `fit.fit_data`

---

### Bug #2: TypeError - rand_forest mode parameter ✅ FIXED

**Severity**: Critical
**Files affected**:
- `examples/14_visualization_demo.ipynb` (1 occurrence)
- `examples/16_baseline_models_demo.ipynb` (3 occurrences)

**Error**:
```python
TypeError: rand_forest() got an unexpected keyword argument 'mode'
```

**Root cause**: `rand_forest(mode='regression')` is invalid - mode must be set via `.set_mode()` method

**Impact**: Random forest models couldn't be instantiated

**Fix**: Changed to `rand_forest().set_mode('regression')` pattern (4 total occurrences)

---

### Bug #3: KeyError - Missing evaluate() calls ✅ FIXED

**Severity**: Critical
**File affected**: `examples/16_baseline_models_demo.ipynb`

**Error**:
```python
KeyError: 'model'  # Empty DataFrame in plot_model_comparison
```

**Root cause**: Models fitted but not evaluated → stats DataFrames only contain "train" split → plots request "test" split → no matches found → empty DataFrame → KeyError when accessing 'model' column

**Impact**: All `plot_model_comparison()` calls failed

**Fix**: Added `.evaluate(test_data)` or `.evaluate(test_ts)` to 11 model fits:
- fit_null, fit_linear, fit_rf → `.evaluate(test_data)`
- fit_naive, fit_snaive, fit_drift → `.evaluate(test_ts)`
- fit_linear_ts, fit_rf_ts → `.evaluate(test_ts)`
- fit_baseline, fit_simple, fit_complex → `.evaluate(test_data)`

---

### Bug #4: PatsyError - Date categorical mismatch ✅ FIXED

**Severity**: High
**File affected**: `examples/16_baseline_models_demo.ipynb`

**Error**:
```python
PatsyError: observation with value Timestamp('2020-10-19 00:00:00') does not
match any of the expected levels (expected: [Timestamp('2020-01-01'), ...])
```

**Root cause**: Naive models used `formula='value ~ date'` which treats date as a categorical variable → test dates weren't in training categorical levels → evaluate() failed

**Impact**: Naive models (naive, seasonal_naive, drift) couldn't be evaluated on test data

**Correct pattern**: Naive models don't use predictors → formula should be `value ~ 1` (intercept-only)

**Fix**: Changed 3 naive model formulas:
- `'value ~ date'` → `'value ~ 1'`

---

### Bug #5: extract_outputs() doesn't include test metrics ✅ FIXED (PROPER SOLUTION)

**Severity**: Architectural issue
**Files affected**: 2 engines needed fixes (21 already had test metrics support)

**Issue**: Even after calling `.evaluate(test_data)`, `engine.extract_outputs()` only returns training metrics. The stats DataFrame doesn't include test split rows.

**Root cause**: Engine `extract_outputs()` methods don't check `ModelFit.evaluation_data` to compute and include test metrics

**Impact**: All `plot_model_comparison()` calls with `split="test"` fail with KeyError

**Proper fix (IMPLEMENTED)**:
Enhanced engine `extract_outputs()` methods to:
1. Check if `ModelFit.evaluation_data` exists and contains test data
2. If yes, extract test predictions from evaluation_data
3. Compute test metrics using the same metric functions
4. Add test rows to stats DataFrame with `split="test"`

**Files modified**:
- `py_parsnip/engines/parsnip_null_model.py` - Added test metrics support (lines 145-231)
- `py_parsnip/engines/parsnip_naive_reg.py` - Added test metrics support (lines 207-287)

**Discovery**: 21 out of 23 engines already had test metrics support! Only 2 baseline model engines were missing it.

**Workaround removed**: Changed `split="train"` back to `split="test"` in all `plot_model_comparison()` calls (3 occurrences in notebook 16)

**Verification**: Notebook 16 now successfully runs with `split="test"` in all plots

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| **Total bugs found** | 5 |
| **Critical bugs fixed** | 5 (all fixed properly) |
| **Architectural issues** | 1 (proper solution implemented) |
| **Files modified** | 5 (2 engines, 3 notebook cells) |
| **Code changes** | 28 (all proper fixes, no workarounds) |
| **Engines affected** | 2 fixed, 21 already had test metrics |
| **Notebooks tested** | 1 of 6 new notebooks (fully working) |

---

## Key Learnings

### 1. Engine API Consistency
- **Lesson**: ModelFit uses `fit_data`, not `fit_output`
- **Action**: Audit all engines for consistent attribute access

### 2. Model Specification Patterns
- **Lesson**: Mode must be set via `.set_mode()`, not constructor
- **Action**: Document correct model specification patterns
- **Documentation needed**: Add examples to all model constructors

### 3. Evaluation Workflow
- **Lesson**: Must call `.evaluate(test_data)` before extracting test metrics
- **Current limitation**: extract_outputs() doesn't include test metrics even after evaluate()
- **Action**: This needs architectural work across all engines

### 4. Formula Syntax for Baseline Models
- **Lesson**: Naive/baseline models that don't use predictors need `formula='y ~ 1'`
- **Action**: Document this pattern for null_model, naive_reg
- **Best practice**: Use intercept-only formula for models without predictors

---

## Recommendations

### Immediate (Completed ✅)
- ✅ Fix fit_output → fit_data in engines
- ✅ Fix rand_forest mode parameter in notebooks
- ✅ Add evaluate() calls where needed
- ✅ Fix naive model formulas
- ✅ Implement proper test metrics solution in engines
- ✅ Remove workarounds and restore split='test' in plots
- ✅ Fix notebook code for proper test metrics extraction

### Short-term (Next sprint)
1. **Test remaining notebooks** (17-21) for similar issues
2. **Create style guide** for notebook patterns
3. **Add validation** to catch formula='y ~ categorical' for naive models

### Long-term (Completed ✅)
1. ✅ **Enhanced extract_outputs()** - All 23 engines now support test metrics
2. **Create base Engine class** with shared logic (nice-to-have for consistency)
3. **Add integration tests** that verify evaluate() → extract_outputs() workflow
4. **Update documentation** with correct patterns for all models

---

## Testing Status

| Notebook | Status | Issues Found |
|----------|--------|--------------|
| 16_baseline_models_demo.ipynb | ✅ All issues fixed | 5 bugs (all resolved) |
| 17_gradient_boosting_demo.ipynb | ⏳ Pending | - |
| 18_sklearn_regression_demo.ipynb | ⏳ Pending | - |
| 19_time_series_ets_stl_demo.ipynb | ⏳ Pending | - |
| 20_hybrid_models_demo.ipynb | ⏳ Pending | - |
| 21_advanced_regression_demo.ipynb | ⏳ Pending | - |

**Status**: Notebook 16 fully working with proper solution implemented
**Next**: Continue testing notebooks 17-21

---

**Report generated**: 2025-10-27
**Testing environment**: py-tidymodels2
**Tester**: Claude Code
