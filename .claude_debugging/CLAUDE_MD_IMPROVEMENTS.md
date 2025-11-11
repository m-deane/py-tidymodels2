# Suggested Improvements to CLAUDE.md

**Date**: 2025-11-10
**Status**: Recommendations for CLAUDE.md updates

---

## Summary

The existing CLAUDE.md is comprehensive and well-structured. The following additions would incorporate recent work from 2025-11-10 sessions:

---

## Additions to Include

### 1. Add to "Layer 4: py-workflows" Section

After line 305 (`extract_preprocessed_data(data)` - **NEW:**), add:

```markdown
**Model Naming Methods (NEW - 2025-11-10):**
- **Purpose**: Label models with custom names for easy identification in multi-model comparisons
- **Methods**:
  - `.add_model_name(name)`: Set custom model name (appears in "model" column of extract_outputs())
  - `.add_model_group_name(group_name)`: Set model group for organization (appears in "model_group_name" column)
- **Usage**:
  ```python
  wf = (
      workflow()
      .add_recipe(rec)
      .add_model(linear_reg())
      .add_model_name("baseline")
      .add_model_group_name("linear_models")
  )

  fit = wf.fit(train_data)
  outputs, _, _ = fit.extract_outputs()
  print(outputs["model"].unique())            # ['baseline']
  print(outputs["model_group_name"].unique()) # ['linear_models']
  ```
- **Benefits**:
  - Clear model identification in multi-model comparisons
  - Easy filtering and grouping of related models
  - Better visualization labels
  - Simplified result analysis
- **Code References**:
  - `py_workflows/workflow.py:54-59` - Added fields to Workflow dataclass
  - `py_workflows/workflow.py:123-173` - Methods implementation
  - `py_workflows/workflow.py:370-376, 543-549, 582-588, 620-626` - fit() and fit_nested() updates
  - `.claude_debugging/MODEL_NAME_METHODS_2025_11_10.md` - Complete documentation

**NestedWorkflowFit.extract_preprocessed_data() (NEW - 2025-11-10):**
- **Purpose**: Inspect preprocessed data after recipe transformations for grouped models
- **Method**:
  ```python
  processed_data = nested_fit.extract_preprocessed_data(data, split='train')
  ```
- **Returns**: DataFrame with all groups combined, showing transformed features
- **Key Features**:
  - Handles per-group and shared preprocessing
  - Preserves group column and metadata (date, split)
  - Consistent column ordering (date first, group_col second)
  - Shows what each group's model actually sees after transformations
- **Use Cases**:
  - Debug preprocessing pipelines (verify normalization, PCA, feature selection)
  - Compare preprocessing across groups
  - Verify feature engineering steps
  - Document transformations for reporting
- **Code References**:
  - `py_workflows/workflow.py:1396-1496` - Method implementation
  - `.claude_debugging/EXTRACT_PREPROCESSED_DATA_METHOD_2025_11_10.md` - Complete documentation
```

### 2. Add to "Layer 5: py-recipes" Section

After line 388 (before **Files:**), add:

```markdown
**Feature Selection Steps (Demonstrated 2025-11-10):**

Four feature selection methods available, especially powerful with per-group preprocessing:

1. **step_filter_rf_importance()** - Random Forest feature importance
   - ⚡⚡⚡ Speed: Fast (~10 seconds)
   - ⭐⭐ Interpretability: Good
   - Best for: Initial screening, high-dimensional data
   - Example: `.step_filter_rf_importance(outcome='y', top_n=3)`

2. **step_select_permutation()** - Permutation importance
   - ⚡⚡ Speed: Medium (~30 seconds)
   - ⭐⭐⭐ Interpretability: Very Good
   - Best for: Model-agnostic selection, robust to overfitting
   - Example: `.step_select_permutation(outcome='y', model=RandomForestRegressor(), top_n=3)`

3. **step_select_shap()** - SHAP value-based selection
   - ⚡ Speed: Slow (~60 seconds)
   - ⭐⭐⭐ Interpretability: Very Good
   - Best for: Detailed explanations, tree-based models
   - Example: `.step_select_shap(outcome='y', model=GradientBoostingRegressor(), top_n=4)`

4. **step_safe_v2()** - Surrogate Assisted Feature Extraction
   - 🐌 Speed: Very Slow (~2-5 minutes)
   - ⭐⭐⭐⭐ Interpretability: Excellent
   - Best for: Interpretable rules, threshold discovery
   - Creates threshold-based features (e.g., `brent_gt_50`, `wti_lt_70`)
   - Example: `.step_safe_v2(surrogate_model=GradientBoostingRegressor(), outcome='y', top_n=5)`

**Per-Group Feature Selection:**
When using `per_group_prep=True` in workflows, each group can select different features:
```python
rec = recipe().step_normalize().step_select_permutation(outcome='y', model=model, top_n=3)
wf = workflow().add_recipe(rec).add_model(linear_reg())
fit = wf.fit_nested(data, group_col='country')  # Different features per group!

# Inspect selected features per group
processed = fit.extract_preprocessed_data(train_data)
for group in processed['country'].unique():
    features = [col for col in processed[processed['country']==group].columns
               if col not in ['date', 'country', 'y', 'split']]
    print(f"{group}: {features}")
```

**Notebook Example**: `_md/forecasting_recipes_grouped.ipynb` cells 88-100 demonstrate all four methods with per-group selection

**Code References**:
- `py_recipes/steps/filter_supervised.py` - step_select_permutation(), step_select_shap()
- `py_recipes/steps/feature_extraction.py` - step_safe_v2()
- `.claude_debugging/FEATURE_SELECTION_GROUPED_EXAMPLE_2025_11_10.md` - Complete documentation
```

### 3. Add New Section After "step_poly() and Patsy XOR Errors"

Add this new critical implementation note:

```markdown
### all_numeric_predictors() Selector Limitation

**Problem:** The `all_numeric_predictors()` selector only excludes hardcoded outcome names: `{'y', 'target', 'outcome'}`. Custom outcome names (e.g., 'refinery_kbd', 'sales') are NOT excluded and will be incorrectly included in preprocessing.

**Impact:**
```python
# Data with custom outcome name
data = pd.DataFrame({
    'x1': [...], 'x2': [...],
    'refinery_kbd': [...]  # Custom outcome name
})

# Using all_numeric_predictors()
rec = recipe().step_normalize(all_numeric_predictors())
prepped = rec.prep(data)
processed = prepped.bake(data)

# ❌ PROBLEM: 'refinery_kbd' gets normalized along with predictors!
```

**Solutions:**

1. **Use workflows** (RECOMMENDED):
   - Workflows automatically preserve outcomes during recipe prep
   - Uses `_prep_and_bake_with_outcome()` method internally
   ```python
   wf = workflow().add_recipe(rec).add_model(linear_reg())
   fit = wf.fit(train_data)  # Outcome preserved automatically
   ```

2. **Manually exclude outcome**:
   ```python
   rec = recipe().step_normalize([col for col in data.columns if col not in ['refinery_kbd', 'date']])
   ```

3. **Use specific column names** instead of selectors

**Why Workflows Are Better:**
- Automatic outcome detection from formula
- Outcome excluded during recipe prep
- Outcome preserved during baking
- Outcome added back to processed data
- Works with grouped data (`per_group_prep=True`)

**Code References:**
- `py_recipes/selectors.py:437-460` - all_numeric_predictors() with hardcoded exclusions
- `py_workflows/workflow.py:121-179` - Outcome preservation helpers in workflows
```

### 4. Update "Project Status and Planning" Section

Replace line 1143 "**Last Updated:** 2025-11-09" with:

```markdown
**Last Updated:** 2025-11-10 (Recent: model naming, preprocessing extraction, feature selection demo)
```

Add after line 1154:

```markdown
**Recent Enhancements (2025-11-10):**
- ✅ **Model naming methods**: Added `.add_model_name()` and `.add_model_group_name()` for custom model labels
- ✅ **Preprocessing inspection**: Added `.extract_preprocessed_data()` for NestedWorkflowFit
- ✅ **Feature selection demo**: Comprehensive example in notebook with 4 methods (permutation, SHAP, SAFE, RF importance)
- ✅ **step_poly() fix**: Replaced `^` with `_pow_` in column names to prevent patsy XOR errors
```

### 5. Update Test Counts

Replace line 1144 "**Total Tests Passing:** 762+ tests" with:

```markdown
**Total Tests Passing:** 775+ tests across all packages (762 base + 3 new verification + 10 new panel)
```

---

## Optional: Consolidate .claude/ CLAUDE.md

The `.claude/CLAUDE.md` file appears to be an older template with generic guidance. Consider:

**Option 1:** Delete `.claude/CLAUDE.md` (template no longer needed)
**Option 2:** Add note at top of `.claude/CLAUDE.md`:
```markdown
# Note: This file is deprecated

Please refer to the main CLAUDE.md in the project root for current project guidance.

This file contains legacy workflow templates and is kept for reference only.
```

---

## Files Referenced in Additions

New documentation files created in 2025-11-10 sessions:
- `.claude_debugging/STEP_POLY_CARET_FIX_2025_11_10.md`
- `.claude_debugging/MODEL_NAME_METHODS_2025_11_10.md`
- `.claude_debugging/EXTRACT_PREPROCESSED_DATA_METHOD_2025_11_10.md`
- `.claude_debugging/FEATURE_SELECTION_GROUPED_EXAMPLE_2025_11_10.md`
- `.claude_debugging/SESSION_SUMMARY_POLY_NAMES_EXTRACT_2025_11_10.md`

---

## Implementation Priority

**High Priority:**
1. Add model naming methods documentation (widely useful)
2. Add extract_preprocessed_data() documentation (debugging essential)
3. Update "Recent Enhancements" and "Last Updated"

**Medium Priority:**
4. Add feature selection steps overview
5. Add all_numeric_predictors() limitation warning

**Low Priority:**
6. Update test counts
7. Consolidate/deprecate .claude/ CLAUDE.md

---

## Reasoning

The existing CLAUDE.md is already excellent. These additions:
- Document new features from recent sessions
- Warn about common pitfalls (all_numeric_predictors limitation)
- Provide quick reference for feature selection methods
- Keep the "Last Updated" and test counts accurate

The additions maintain the existing structure and tone, focusing on practical guidance for future Claude instances.
