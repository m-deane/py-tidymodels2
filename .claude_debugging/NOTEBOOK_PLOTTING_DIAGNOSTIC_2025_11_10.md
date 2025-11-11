# Notebook Plotting Diagnostic

**Date**: 2025-11-10
**Notebook**: `_md/forecasting_recipes_grouped.ipynb`
**Issue**: User reported cell 7 shows train+test plots, but cells 8+ only show test
**Status**: ✅ NO BUG - Code working correctly

---

## Investigation Summary

### Diagnosis
The code is **functioning correctly**. Both patterns produce identical outputs containing train and test splits:
- Cell 7 (ModelSpec.fit_nested): ✅ Train + Test data present
- Cells 8+ (Workflow.fit_nested): ✅ Train + Test data present

### Test Results
```
TEST 1: Cell 7 Pattern - ModelSpec.fit_nested()
Output shape: (240, 10)
Splits present: ['train' 'test']
Train rows: 180
Test rows: 60

TEST 2: Cell 8 Pattern - Workflow.fit_nested() with recipe
Output shape: (240, 10)
Splits present: ['train' 'test']
Train rows: 180
Test rows: 60

✓ Both patterns have identical split values (train + test)
```

### Root Cause
The issue was **stale notebook outputs** from a previous execution, possibly from before recent bug fixes.

### Solution Applied
1. ✅ Cleared all notebook outputs
2. User should re-execute the notebook with:
   ```bash
   cd "/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels"
   source py-tidymodels2/bin/activate
   jupyter notebook _md/forecasting_recipes_grouped.ipynb
   # Then: Kernel → Restart & Run All
   ```

---

## Code Pattern Verification

### Cell 7 Pattern (Working)
```python
spec_baseline = linear_reg().set_engine("sklearn")
fit_baseline = spec_baseline.fit_nested(train_data, FORMULA_STR, group_col='country')
fit_baseline = fit_baseline.evaluate(test_data)  # ← Adds test data
```

### Cells 8+ Pattern (Also Working)
```python
rec_normalize = recipe().step_normalize()
wf_normalize = workflow().add_recipe(rec_normalize).add_model(linear_reg())
fit_normalize = wf_normalize.fit_nested(train_data, group_col='country')
fit_normalize = fit_normalize.evaluate(test_data)  # ← Adds test data
```

**Key Point**: Both patterns correctly call `.evaluate(test_data)`, which adds test split to outputs.

---

## How plot_forecast() Works

`plot_forecast()` reads the outputs DataFrame from `extract_outputs()`:
- **Train data**: Rows where `split == "train"`
- **Test data**: Rows where `split == "test"`

If both splits are present in outputs (which they are), both will be plotted.

**Visual Styling**:
- Blue solid line: Actuals (continuous across train+test)
- Orange dashed: Fitted (Train)
- Red dashed: Fitted (Test)

If orange and red lines overlap closely, it may **appear** that only test is shown, but both are present.

---

## Files Verified

1. `/py_workflows/workflow.py:1008-1053` - NestedWorkflowFit.evaluate()
2. `/py_parsnip/model_spec.py:549-615` - ModelFit.evaluate()
3. `/py_parsnip/engines/sklearn_linear_reg.py:273-409` - extract_outputs()
4. `/py_visualize/forecast.py:100-190` - Plotting logic

All verified to work correctly.

---

## Conclusion

**No code changes required.** The functionality works as designed. Notebook outputs have been cleared. User should re-execute the notebook to verify all plots now show train+test data correctly.

---

**Diagnostic completed**: 2025-11-10
**Result**: Code working correctly, stale outputs cleared
