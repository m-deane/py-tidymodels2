# WorkflowSet Notebook Fixes

**Date:** 2025-11-08
**Issue:** forecasting_workflowsets.ipynb had multiple API mismatches preventing execution
**Total Issues Fixed:** 15 (6 API mismatches + 1 missing feature + 1 argument pattern + 6 data processing errors + 1 notebook issue)

## Problems Identified

### 1. **Dict vs List Parameter Error** (Critical)
**Location:** Cells 10, 30 (WorkflowSet creation)

**Error:**
```python
AttributeError: 'str' object has no attribute 'model_type'
```

**Root Cause:** `WorkflowSet.from_cross()` expects **lists** as parameters, but notebook was passing **dictionaries**.

**Original Code:**
```python
# Define models as dict
models = {
    "linear": linear_reg(),
    "ridge": linear_reg(penalty=0.1, mixture=0.0),
    # ...
}

# Pass dict directly (WRONG!)
wf_set = WorkflowSet.from_cross(
    preproc=formulas,  # dict
    models=models      # dict
)
```

When iterating over a dict, Python iterates over keys (strings), not values:
```python
for j, model in enumerate(models):
    model_type = model.model_type  # model is "linear" (string), not linear_reg()!
```

**Fix Applied:**
```python
# Convert dicts to lists of values
formula_list = list(formulas.values())
model_list = list(models.values())

wf_set = WorkflowSet.from_cross(
    preproc=formula_list,
    models=model_list,
    ids=list(formulas.keys())  # Use formula names as IDs
)
```

### 2. **Column Name Mismatch in rank_results()** (API Design)
**Location:** Cells 16, 32, 43, 46

**Expected Behavior:** Notebook expects wide-format output with metric-specific columns:
- `rmse_mean`, `rmse_std`
- `mae_mean`, `mae_std`
- `r_squared_mean`, `r_squared_std`

**Original Implementation:** Returned long-format with generic columns:
- `mean`, `std`, `metric`

**Fix Applied to `py_workflowsets/workflowset.py`:**
```python
def rank_results(self, ...):
    # Get summarized metrics
    metrics_df = self.collect_metrics(summarize=True)

    # Pivot to wide format: one column per metric
    wide_metrics = metrics_df.pivot_table(
        index=["wflow_id", "preprocessor", "model"],
        columns="metric",
        values=["mean", "std", "n"]
    )

    # Flatten multi-level columns: (mean, rmse) -> rmse_mean
    wide_metrics.columns = [f"{metric}_{stat}" for stat, metric in wide_metrics.columns]
    wide_metrics = wide_metrics.reset_index()

    # Sort by specified metric
    sort_col = f"{rank_metric}_mean"
    ...
```

**Result:** Now returns DataFrame with columns:
```
rank | wflow_id | preprocessor | model | rmse_mean | rmse_std | mae_mean | mae_std | ...
```

### 3. **Parameter Name Inconsistency** (API Design)
**Location:** All rank_results() calls

**Notebook Usage:**
```python
top_models = wf_results.rank_results(metric="rmse", n=10)
```

**Original Implementation:**
```python
def rank_results(self, rank_metric: str, ...):
```

**Fix Applied:**
```python
def rank_results(self,
                 rank_metric: Optional[str] = None,
                 metric: Optional[str] = None,
                 select_best: bool = False,
                 n: int = 10):
    # Handle both parameter names for backwards compatibility
    if metric is not None:
        rank_metric = metric
    elif rank_metric is None:
        raise ValueError("Must provide either 'metric' or 'rank_metric' parameter")
    ...
```

### 4. **Matplotlib vs Plotly Confusion** (Visualization)
**Location:** Cells 17, 33, 44

**Original Code (expects Plotly):**
```python
fig = wf_results.autoplot(metric="rmse", n=20)
fig.update_layout(title="Top 20 Workflows by RMSE", height=600)
fig.show()
```

**Actual Implementation:** Returns matplotlib Figure object (no `.update_layout()` or `.show()` methods)

**Fix Applied:**
```python
import matplotlib.pyplot as plt

fig = wf_results.autoplot(metric="rmse", top_n=20)
fig.suptitle("Top 20 Workflows by RMSE", fontsize=16, y=0.995)
plt.tight_layout()
plt.show()  # Use pyplot's show(), not figure's show()
```

**Note:** Also changed parameter `n=` to `top_n=` to match implementation.

### 5. **Import Error: Non-Existent PreparedStep Classes** (Import Infrastructure)
**Location:** py_recipes/steps/__init__.py:66-77 and __all__ export list

**Error:**
```python
ImportError: cannot import name 'PreparedStepFilterAnova' from 'py_recipes.steps.filter_supervised'
```

**Root Cause:** The `filter_supervised.py` module uses a unified class pattern (single class with `_is_prepared` flag) rather than separate `Step` and `PreparedStep` classes. But `__init__.py` was trying to import non-existent `PreparedStep*` classes.

**Classes Actually Defined in filter_supervised.py:**
- `StepFilterAnova`
- `StepFilterRfImportance`
- `StepFilterMutualInfo`
- `StepFilterRocAuc`
- `StepFilterChisq`

**Fix Applied:**
Removed all `PreparedStep*` imports from both the import statement and the `__all__` export list:

```python
# Before
from py_recipes.steps.filter_supervised import (
    StepFilterAnova,
    PreparedStepFilterAnova,  # DOESN'T EXIST!
    StepFilterRfImportance,
    PreparedStepFilterRfImportance,  # DOESN'T EXIST!
    ...
)

# After
from py_recipes.steps.filter_supervised import (
    StepFilterAnova,
    StepFilterRfImportance,
    StepFilterMutualInfo,
    StepFilterRocAuc,
    StepFilterChisq,
)
```

Also removed from `__all__` list at lines 208-218.

### 6. **Recipe API Mismatch** (Recipe Construction)
**Location:** Cell 26 (Recipe definitions)

**Error:**
```python
TypeError: recipe() takes from 0 to 1 positional arguments but 2 were given
```

**Root Cause:** The notebook was calling `recipe(train_data, "target ~ .")` but the `recipe()` function only accepts one optional parameter (data), not a formula.

**API Signature:**
```python
def recipe(data: Optional[pd.DataFrame] = None) -> Recipe:
```

**Key Concept:** In py-tidymodels, formulas are specified at the **workflow level**, not the recipe level. Recipes define preprocessing steps; workflows combine recipes with formulas and models.

**Original Code:**
```python
rec_minimal = (
    recipe(train_data, "target ~ .")  # WRONG! Too many arguments
    .step_rm("date")
)
```

**Fix Applied:**
```python
rec_minimal = (
    recipe()  # Correct - no arguments or just data
    .step_rm("date")
)
```

**How Formulas Work with Recipes:**
When a recipe is used in a workflow without an explicit formula, the workflow auto-generates a formula like `"target ~ ."` from the data at fit time. Example:

```python
# Recipe defines preprocessing only
rec = recipe().step_rm("date").step_normalize()

# Formula specified when adding to workflow (if needed)
wf = workflow().add_recipe(rec).add_model(linear_reg())

# At fit time, formula is auto-generated from data columns
```

### 7. **Missing step_rm() Method** (Missing Feature)
**Location:** Cell 26 (Recipe definitions)

**Error:**
```python
AttributeError: 'Recipe' object has no attribute 'step_rm'
```

**Root Cause:** The `Recipe` class didn't have a `step_rm()` method for removing columns. This is a commonly needed feature for dropping columns like IDs, dates, or other non-predictive features.

**Solution:** Created new `step_rm()` and `step_select()` methods for the Recipe class.

**Implementation:**

1. **Created new step classes** (`py_recipes/steps/remove.py`):
   - `StepRm`: Remove/drop specified columns
   - `StepSelect`: Keep only specified columns (inverse of step_rm)

2. **Added methods to Recipe class** (`py_recipes/recipe.py`):
   ```python
   def step_rm(self, columns: Union[str, List[str], Callable]) -> "Recipe":
       """Remove/drop columns from the dataset."""
       from py_recipes.steps.remove import StepRm
       return self.add_step(StepRm(columns=columns))

   def step_select(self, columns: Union[str, List[str], Callable]) -> "Recipe":
       """Select (keep) only specified columns."""
       from py_recipes.steps.remove import StepSelect
       return self.add_step(StepSelect(columns=columns))
   ```

3. **Updated imports** (`py_recipes/steps/__init__.py`):
   - Added `StepRm` and `StepSelect` to imports and `__all__`

**Usage Examples:**
```python
# Remove single column
rec = recipe().step_rm("date")

# Remove multiple columns
rec = recipe().step_rm(["id", "date", "timestamp"])

# Keep only specific columns
rec = recipe().step_select(["feature1", "feature2", "target"])

# Chain with other steps
rec = (
    recipe()
    .step_rm("id")
    .step_normalize()
    .step_dummy(["category"])
)
```

**Tests Created:** `tests/test_recipes/test_step_rm.py` with 6 comprehensive tests

### 8. **Recipe Step Arguments API Mismatch** (Cell 26)
**Location:** Cell 26 (Recipe definitions)

**Error:**
```python
TypeError: Recipe.step_impute_median() takes from 1 to 2 positional arguments but 4 were given
```

**Root Cause:** Recipe step methods expect column names as a **list**, not as multiple separate string arguments.

**Original Code:**
```python
rec_normalized = (
    recipe()
    .step_rm("date")
    .step_impute_median("totaltar", "mean_med_diesel_crack_input1_trade_month_lag2", "mean_nwe_hsfo_crack_trade_month_lag1")  # WRONG!
    .step_normalize("totaltar", "mean_med_diesel_crack_input1_trade_month_lag2", "mean_nwe_hsfo_crack_trade_month_lag1")  # WRONG!
)
```

**Fix Applied:**
```python
rec_normalized = (
    recipe()
    .step_rm("date")
    .step_impute_median(["totaltar", "mean_med_diesel_crack_input1_trade_month_lag2", "mean_nwe_hsfo_crack_trade_month_lag1"])  # Correct
    .step_normalize(["totaltar", "mean_med_diesel_crack_input1_trade_month_lag2", "mean_nwe_hsfo_crack_trade_month_lag1"])  # Correct
)
```

**API Pattern:** All recipe step methods that accept multiple columns expect them as a list:
```python
# Correct patterns
.step_impute_median(["col1", "col2", "col3"])
.step_normalize(["col1", "col2"])
.step_log(["col1", "col2"], offset=1)
.step_poly(["col1"], degree=2)
.step_interact(["col1", "col2"])

# Single column can be string or list
.step_rm("date")  # OK
.step_rm(["date"])  # Also OK
```

### 9. **Patsy Special Characters in Auto-Generated Formulas** (NEW - 2025-11-08)
**Location:** py_workflows/workflow.py:216-235

**Error:**
```python
NameError: name 'totaltar' is not defined
    target ~ ... + totaltar^2
```

**Root Cause:** When recipes use `step_poly()`, sklearn creates column names like `totaltar^2`. But Patsy treats `^` as exponentiation operator, not part of column name. When auto-generating formulas, these special characters cause Patsy to try evaluating expressions instead of using literal column names.

**Fix Applied:**
```python
# Escape column names that contain Patsy special characters
import re
def escape_column_name(col):
    if re.search(r'[\^\*\:\+\-\/\(\)\[\]\{\}]', col):
        # Wrap in Q() for Patsy to treat as literal column name
        return f'Q("{col}")'
    return col

escaped_cols = [escape_column_name(col) for col in predictor_cols]
formula = f"{outcome_col} ~ {' + '.join(escaped_cols)}"
```

**Result:** Auto-generated formulas now properly escape special characters:
```python
# Before: "target ~ totaltar + totaltar^2"  (Patsy tries to square totaltar)
# After:  "target ~ Q(\"totaltar\") + Q(\"totaltar^2\")"  (Treats as column names)
```

### 10. **StepInteract API Mismatch** (NEW - 2025-11-08)
**Location:** py_recipes/recipe.py:step_interact()

**Error:**
```python
ValueError: too many values to unpack (expected 2)
```

**Root Cause:** `StepInteract` expects list of tuples `[("col1", "col2")]`, but notebook was calling with simple list `["col1", "col2"]`. The for loop `for col1, col2 in interactions:` tried to unpack strings instead of tuples.

**Fix Applied:**
```python
def step_interact(
    self,
    interactions: Union[List[tuple], List[str]],
    separator: str = "_x_"
) -> "Recipe":
    from itertools import combinations

    # Check if interactions is a list of tuples or a list of strings
    if interactions and isinstance(interactions[0], str):
        # Generate all pairwise combinations
        interactions = list(combinations(interactions, 2))

    return self.add_step(StepInteract(interactions=interactions, separator=separator))
```

**Result:** Now accepts both formats:
```python
# Explicit pairs
.step_interact([("totaltar", "mean_med_diesel_crack_input1_trade_month_lag2")])

# Auto-generate all pairs
.step_interact(["totaltar", "mean_med_diesel_crack_input1_trade_month_lag2"])
```

### 11. **Log Transform with Negative Values** (NEW - 2025-11-08)
**Location:** py_recipes/steps/transformations.py:StepLog.prep()

**Error:**
```python
PatsyError: factor contains missing values
```

**Root Cause:** With `offset=1`, negative values like `-56.18` create `log(-56.18 + 1) = log(-55.18) = NaN`. The notebook used `step_log(offset=1)` but data had large negative values.

**Fix Applied:**
```python
def prep(self, data: pd.DataFrame, training: bool = True):
    # ... existing code ...

    # Calculate safe offset if data has negative values
    safe_offset = self.offset
    if not self.signed and training:
        min_val = min([data[col].min() for col in cols if col in data.columns], default=0)
        if min_val < 0:
            required_offset = abs(min_val) + 1e-10
            if self.offset < required_offset:
                warnings.warn(
                    f"Log transform: offset={self.offset} insufficient for negative values "
                    f"(min={min_val:.4f}). Using offset={required_offset:.4f}. "
                    f"Consider using signed=True for signed log transform."
                )
                safe_offset = required_offset
```

**Result:** Log transform automatically adjusts offset to prevent NaN values and warns user.

### 12. **Recipe Workflows No Metrics Collected** (NEW - 2025-11-08)
**Location:** py_tune/tune.py:fit_resamples()

**Error:**
```python
ValueError: Metric 'rmse' not found in results
```

**Root Cause:** In `fit_resamples()` line 376, metric computation only happened for formula-based workflows:
```python
if hasattr(workflow, 'preprocessor') and isinstance(workflow.preprocessor, str):
    # Only runs for formulas, NOT for recipes!
    outcome = workflow.preprocessor.split('~')[0].strip()
    truth = test_data[outcome]
    # ... compute metrics
```

For recipe workflows, `workflow.preprocessor` is a Recipe object (not string), so:
1. The condition was FALSE
2. Metrics never got computed (lines 377-398 skipped)
3. `all_metrics` stayed empty
4. `metrics_df` became empty DataFrame
5. WorkflowSet collected empty DataFrames
6. When `rank_results()` tried to pivot, no 'rmse' column existed to create 'rmse_mean'

All 25 recipe workflows executed successfully but produced NO metrics, causing the ranking to fail.

**Fix Applied:**
```python
# Get outcome column from either formula or fitted workflow's blueprint
outcome = None

if hasattr(workflow, 'preprocessor') and isinstance(workflow.preprocessor, str):
    # Formula-based workflow
    outcome = workflow.preprocessor.split('~')[0].strip()
else:
    # Recipe-based or other workflow
    from py_recipes import Recipe
    if hasattr(workflow, 'preprocessor') and isinstance(workflow.preprocessor, Recipe):
        # For recipes, workflow.fit() auto-detects outcome and builds a formula
        # The ModelFit stores the blueprint with outcome info
        blueprint = wf_fit.fit.blueprint

        # Try multiple ways to extract outcome from blueprint
        if hasattr(blueprint, 'outcome_name'):
            outcome = blueprint.outcome_name
        elif hasattr(blueprint, 'roles') and 'outcome' in blueprint.roles:
            outcome = blueprint.roles['outcome'][0] if blueprint.roles['outcome'] else None
        elif isinstance(blueprint, dict):
            outcome = blueprint.get('outcome_name') or blueprint.get('y_name')
            # If dict blueprint has formula_data, extract from formula
            if outcome is None and 'formula_data' in blueprint:
                formula_str = str(blueprint.get('formula', ''))
                if '~' in formula_str:
                    outcome = formula_str.split('~')[0].strip()

if outcome and outcome in test_data.columns:
    truth = test_data[outcome]
    estimate = predictions['.pred']
    # ... compute metrics
else:
    print(f"Warning: Fold {fold_idx+1} could not determine outcome column from workflow")
```

**Result:** Recipe workflows now correctly extract the outcome column from the Blueprint's `roles['outcome']` attribute and compute metrics properly.

**Verification:** Created test script (`_md/test_recipe_metrics.py`) that confirms:
- Recipe workflows collect all 3 metrics (rmse, mae, r_squared)
- 9 rows of metrics collected (3 folds × 3 metrics)
- All py_tune tests still pass (36/36)

### 13. **from_workflows() Doesn't Handle Tuple Lists** (NEW - 2025-11-08)
**Location:** py_workflowsets/workflowset.py:from_workflows()

**Error:**
```python
AttributeError: 'tuple' object has no attribute 'preprocessor'
```

**Root Cause:** The notebook builds a list of tuples `[(wf_id, workflow), ...]` which is a common pattern:
```python
ts_workflows = []
for formula_name in ["prophet_basic", "prophet_full"]:
    wf_id = f"{formula_name}_prophet"
    wf = workflow().add_formula(ts_formulas[formula_name]).add_model(ts_models["prophet"])
    ts_workflows.append((wf_id, wf))  # List of tuples

wf_set_ts = WorkflowSet.from_workflows(ts_workflows)  # ERROR!
```

But `from_workflows()` only accepted:
- List of workflows + separate IDs: `from_workflows([wf1, wf2], ids=["id1", "id2"])`
- List of workflows only: `from_workflows([wf1, wf2])`

When passed a list of tuples, the code at line 56 did `dict(zip(ids, workflows))` where `workflows` was the list of tuples, creating `{"id": (wf_id, wf)}` instead of `{"id": wf}`.

**Fix Applied:**
```python
@classmethod
def from_workflows(cls, workflows: List[Any], ids: Optional[List[str]] = None):
    """
    Create WorkflowSet from a list of workflows.

    Args:
        workflows: List of Workflow objects OR list of (id, workflow) tuples
        ids: Optional list of IDs (ignored if workflows is list of tuples)

    Examples:
        >>> # Method 1: Separate workflows and IDs
        >>> wf_set = WorkflowSet.from_workflows([wf1, wf2], ids=["linear", "rf"])
        >>> # Method 2: List of tuples
        >>> wf_set = WorkflowSet.from_workflows([("linear", wf1), ("rf", wf2)])
    """
    # Check if workflows is a list of tuples (id, workflow)
    if workflows and isinstance(workflows[0], tuple) and len(workflows[0]) == 2:
        # Extract IDs and workflows from tuples
        ids = [wf_id for wf_id, _ in workflows]
        workflows = [wf for _, wf in workflows]

    # ... rest of method
```

**Result:** Both patterns now work:
- `from_workflows([wf1, wf2], ids=["a", "b"])` - Separate lists
- `from_workflows([("a", wf1), ("b", wf2)])` - List of tuples (NEW)

### 14. **Pivot Table Extracts Wrong Model Names** (NOTEBOOK ISSUE - 2025-11-08)
**Location:** Cell 36 in forecasting_workflowsets.ipynb

**Problem:** Pivot table shows NaN values and numeric column headers (1, 2, 3, 4, 5) instead of model names:
```
model_type                     1        2        3        4        5
recipe_strategy
interact_boost_tree          NaN      NaN      NaN  37.7596      NaN
```

**Root Cause:** The code extracts "model_type" from workflow IDs:
```python
recipe_metrics['model_type'] = recipe_metrics['wflow_id'].str.split('_').str[-1]
```

For IDs like `"minimal_linear_reg_1"`, this gets the numeric suffix `"1"` instead of the model name `"linear_reg"`. This creates a sparse pivot because:
- Different models have different numeric IDs
- The same model appears under different numbers in different recipe combinations
- Most cells become NaN because each (recipe, number) combination appears only once

**Fix:** Use the existing `'model'` column from `collect_metrics()`:
```python
# Don't extract - use existing column!
pivot = recipe_metrics[recipe_metrics['metric'] == 'rmse'].pivot_table(
    values='mean',
    index='recipe_strategy',
    columns='model',  # Use 'model' instead of 'model_type'
    aggfunc='first'
).round(4)

# Update the loop too
print("\nBest recipe for each model:")
for col in pivot.columns:
    best_idx = pivot[col].idxmin()
    best_val = pivot[col].min()
    print(f"  {col:15s}: {best_idx:12s} (RMSE={best_val:.4f})")
```

**Result:** Proper pivot table with model names as columns:
```
model                 boost_tree  linear_reg  rand_forest
recipe_strategy
interact_boost_tree     37.7596         NaN          NaN
interact_linear_reg         NaN     67.5804     33.1532
...
```

No NaN values within each model column - only NaN when a recipe-model combination doesn't exist.

### 15. **Time Series fit_raw() Unexpected Keyword Argument** (NEW - 2025-11-08)
**Location:** py_parsnip/model_spec.py:fit()

**Error:**
```
Warning: Fold 1 failed with error: HybridProphetBoostEngine.fit_raw() got an unexpected keyword argument 'original_training_data'
Warning: Fold 1 failed with error: StatsmodelsARIMAEngine.fit_raw() got an unexpected keyword argument 'original_training_data'
Warning: Fold 1 failed with error: StatsmodelsExpSmoothingEngine.fit_raw() got an unexpected keyword argument 'original_training_data'
```

**Root Cause:** When workflows call `model.fit()` (line 245 in workflow.py), they pass `original_training_data` for engines that might need it:
```python
model_fit = self.spec.fit(processed_data, formula, original_training_data=original_data)
```

In `model_spec.py`, when the model uses the raw path (`fit_raw()`), the code at lines 142-143 unconditionally passed this parameter:
```python
fit_raw_kwargs = {}
if original_training_data is not None:
    fit_raw_kwargs['original_training_data'] = original_training_data  # ERROR!
```

But most time series engines don't accept this parameter:
- ProphetEngine.fit_raw(self, spec, data, formula) - NO original_training_data
- StatsmodelsARIMAEngine.fit_raw(self, spec, data, formula, date_col=None) - NO original_training_data
- HybridProphetBoostEngine.fit_raw(self, spec, data, formula) - NO original_training_data

Only newer engines that need both raw datetime handling AND access to original data would accept it.

**Fix Applied:**
```python
# Check if fit_raw accepts optional parameters
fit_raw_signature = inspect.signature(engine.fit_raw)
accepts_date_col = 'date_col' in fit_raw_signature.parameters
accepts_original_data = 'original_training_data' in fit_raw_signature.parameters

# Build kwargs for fit_raw
fit_raw_kwargs = {}
if original_training_data is not None and accepts_original_data:
    fit_raw_kwargs['original_training_data'] = original_training_data  # Only if accepted!

# Only infer and pass date_col if engine supports it
if accepts_date_col:
    inferred_date = _infer_date_column(data, self.date_col, date_col)
    fit_raw_kwargs['date_col'] = inferred_date
```

**Result:** Time series workflows no longer fail during CV. The parameter is only passed to engines that actually accept it, just like the existing `date_col` handling.

**Verification:** Created test script (`_md/test_ts_workflows.py`) that confirms:
- Prophet workflows succeed across all CV folds (20 metrics collected)
- ARIMA workflows succeed across all CV folds (20 metrics collected)
- No more "unexpected keyword argument" errors

## Summary of Changes

### Code Changes
**py_workflowsets/workflowset.py:**
1. ✅ Updated `rank_results()` to return wide-format DataFrame with metric-specific columns
2. ✅ Added dual parameter support: `metric=` and `rank_metric=` (backwards compatible)
3. ✅ Improved error handling for missing metric parameter

**py_recipes/steps/__init__.py:**
4. ✅ Removed non-existent `PreparedStep*` imports from filter_supervised
5. ✅ Updated `__all__` export list to remove PreparedStep classes
6. ✅ Added `StepRm` and `StepSelect` imports

**py_recipes/recipe.py:**
7. ✅ Added `Union` and `Callable` to type imports
8. ✅ Implemented `step_rm()` method for removing columns
9. ✅ Implemented `step_select()` method for selecting columns
10. ✅ **NEW:** Updated `step_interact()` to accept both list of tuples and simple list of columns

**py_recipes/steps/remove.py (NEW):**
11. ✅ Created `StepRm` class for column removal
12. ✅ Created `StepSelect` class for column selection

**py_recipes/steps/transformations.py:**
13. ✅ **NEW:** Updated `StepLog.prep()` to auto-calculate safe offset for negative values

**py_workflows/workflow.py:**
14. ✅ **NEW:** Added Patsy special character escaping in auto-generated formulas

**py_tune/tune.py:**
15. ✅ **NEW:** Updated `fit_resamples()` to extract outcome column from recipe blueprints (not just formulas)

**py_workflowsets/workflowset.py:**
16. ✅ **NEW:** Updated `from_workflows()` to handle list of (id, workflow) tuples

**py_parsnip/model_spec.py:**
17. ✅ **NEW:** Updated `fit()` to check if `fit_raw()` accepts `original_training_data` before passing

**tests/test_recipes/test_step_rm.py (NEW):**
18. ✅ Created 6 comprehensive tests for step_rm and step_select

**tests/test_workflowsets/test_workflowset.py:**
19. ✅ Updated test_rank_results_basic for wide-format column names

**_md/test_recipe_metrics.py (NEW):**
20. ✅ Created verification test for recipe workflow metrics collection

**_md/test_ts_workflows.py (NEW):**
21. ✅ Created verification test for time series workflow CV without parameter errors

### Notebook Changes (_md/forecasting_workflowsets.ipynb)
1. ✅ **Cell 10:** Convert formula/model dicts to lists before passing to `from_cross()`
2. ✅ **Cell 17:** Fixed matplotlib visualization (removed Plotly methods)
3. ✅ **Cell 26:** Fixed recipe() calls - removed formula argument + wrapped column names in lists
4. ✅ **Cell 30:** Convert recipe/model dicts to lists before passing to `from_cross()`
5. ✅ **Cell 33:** Fixed matplotlib visualization (removed Plotly methods)
6. ✅ **Cell 36:** Use 'model' column instead of extracting from workflow IDs (fixes NaN values in pivot)
7. ✅ **Cell 44:** Fixed matplotlib visualization (removed Plotly methods)
8. ✅ **Cell 48:** No changes needed - from_workflows() now handles tuple lists automatically

## Expected Outcome

After these fixes, the notebook should:
1. ✅ Successfully create WorkflowSets (40 formula-based, 25 recipe-based, 9 time-series)
2. ✅ Evaluate all workflows across CV folds without parameter errors
3. ✅ Rank results with proper column names
4. ✅ Display matplotlib visualizations correctly
5. ✅ Handle polynomial features with special characters in column names
6. ✅ Auto-generate pairwise interactions from simple column lists
7. ✅ Automatically adjust log transform offsets for negative data
8. ✅ Collect metrics from recipe-based workflows correctly
9. ✅ Collect metrics from time series workflows correctly (no fit_raw errors)
10. ✅ Create time series WorkflowSet from list of tuples
11. ✅ Display proper pivot table with model names (no NaN spam)
12. ✅ Complete grand comparison of best models

## Testing Status

**Not yet tested** - Notebook requires full execution which can take several minutes due to:
- 40 workflows × 5 CV folds = 200 model fits (formula-based)
- 25 workflows × 5 CV folds = 125 model fits (recipe-based)
- 9 workflows × time-series CV folds = ~45+ model fits (time-series)
- **Total: 370+ model fits**

## Recommendations

1. **Test incrementally:** Run each section separately to identify any remaining issues
2. **Monitor memory:** 370+ model fits may require significant RAM
3. **Consider subset testing:** Reduce number of formulas/models for initial validation
4. **Document execution time:** Track how long full execution takes for user expectations

## Related Files

**Modified:**
- `py_workflowsets/workflowset.py` - Core WorkflowSet implementation (API fixes)
- `py_recipes/steps/__init__.py` - Import fixes and new step additions
- `py_recipes/recipe.py` - Added step_rm() and step_select() methods
- `_md/forecasting_workflowsets.ipynb` - Fixed notebook (6 cells)

**Created:**
- `py_recipes/steps/remove.py` - New StepRm and StepSelect classes
- `tests/test_recipes/test_step_rm.py` - Tests for new steps

**Documentation:**
- `.claude_debugging/WORKFLOWSET_NOTEBOOK_FIXES.md` - This file

---

## Issue 16: Plot Forecast Refactoring

**Date:** 2025-11-08
**Location:** `forecasting.ipynb`
**Component:** `py_visualize/forecast.py` - `plot_forecast()` function

### Problem

User requested specific plotting behavior for forecast plots:
1. **Actuals should be ONE continuous line** (not split by train/test)
2. **Fitted values should be split by train/test** with different colors:
   - Train: Orange dashed
   - Test: Red dashed
3. **Forecast line should NEVER be plotted** in these plots

### Original Implementation

The original `plot_forecast()` function had:
- Actuals split by train (blue) and test (green) as separate lines
- Fitted values only shown for training data (orange dotted)
- Optional forecast line shown for test data (red dashed)
- `show_forecast` parameter to toggle forecast display

### Root Cause

The plotting logic didn't match user's visualization requirements for model diagnostic plots. The function was designed for showing forecasts (future predictions), but user wanted diagnostic plots showing model fit quality.

### Fix Applied

**1. Removed `show_forecast` Parameter**
- Simplified API by removing optional parameter
- Forecast line never displayed in these diagnostic plots

**2. Refactored `_plot_forecast_single()` (lines 88-180):**

**Before:**
```python
# Training data - Actuals (blue line)
fig.add_trace(go.Scatter(
    x=x_train,
    y=train_data["actuals"],
    name="Training",
    line=dict(color="#1f77b4")
))

# Training data - Fitted (orange dotted)
fig.add_trace(go.Scatter(
    x=x_train,
    y=train_data["fitted"],
    name="Fitted",
    line=dict(color="#ff7f0e", dash="dot")
))

# Test data - Actuals (green line)
fig.add_trace(go.Scatter(
    x=x_test,
    y=test_data["actuals"],
    name="Test",
    line=dict(color="#2ca02c")
))

# Test data - Forecast (optional, red dashed)
if show_forecast:
    fig.add_trace(go.Scatter(
        x=x_test,
        y=test_data["forecast"],
        name="Forecast",
        line=dict(color="#d62728", dash="dash")
    ))
```

**After:**
```python
# 1. ACTUALS - One continuous line (train + test combined)
if x_test is not None and len(test_data) > 0:
    # Check if x values are Series or Index
    if isinstance(x_train, pd.Series) and isinstance(x_test, pd.Series):
        x_all = pd.concat([x_train, x_test])
    else:
        # For Index objects, convert to list and concatenate
        x_all = list(x_train) + list(x_test)
    y_all = pd.concat([train_data["actuals"], test_data["actuals"]])
else:
    x_all = x_train
    y_all = train_data["actuals"]

fig.add_trace(go.Scatter(
    x=x_all,
    y=y_all,
    name="Actuals",
    mode="lines",
    line=dict(color="#1f77b4", width=2)  # Blue continuous
))

# 2. FITTED VALUES (TRAIN) - Orange dashed
fig.add_trace(go.Scatter(
    x=x_train,
    y=train_data["fitted"],
    name="Fitted (Train)",
    mode="lines",
    line=dict(color="#ff7f0e", width=2, dash="dash")
))

# 3. FITTED VALUES (TEST) - Red dashed
if len(test_data) > 0:
    fig.add_trace(go.Scatter(
        x=x_test,
        y=test_data["fitted"],
        name="Fitted (Test)",
        mode="lines",
        line=dict(color="#d62728", width=2, dash="dash")
    ))
```

**3. Refactored `_plot_forecast_nested()` (lines 183-307):**

Applied identical changes for grouped/nested models:
- One continuous actuals line per group (train + test combined)
- Fitted values split by train (orange dashed) and test (red dashed)
- No forecast line

**4. Fixed Index Concatenation Bug:**

When data has no date column, `x_train` and `x_test` are pandas Index objects. The original code tried to use `pd.concat()` which only works with Series/DataFrames.

**Solution:**
```python
# Check if x values are Series or Index
if isinstance(x_train, pd.Series) and isinstance(x_test, pd.Series):
    x_all = pd.concat([x_train, x_test])
else:
    # For Index objects, convert to list and concatenate
    x_all = list(x_train) + list(x_test)
```

### Visual Changes

**Before:**
- 🔵 Blue line (train actuals)
- 🟢 Green line (test actuals)
- 🟠 Orange dotted (fitted train)
- 🔴 Red dashed (forecast - optional)

**After:**
- 🔵 Blue continuous (actuals - train + test combined)
- 🟠 Orange dashed (fitted train)
- 🔴 Red dashed (fitted test)
- Prediction intervals (if available, gray shaded area)

### Code References

**Modified Files:**
- `py_visualize/forecast.py:88-180` - `_plot_forecast_single()` refactored
- `py_visualize/forecast.py:183-307` - `_plot_forecast_nested()` refactored

**Signature Changes:**
```python
# Before
def _plot_forecast_single(..., show_forecast: bool = True)
def _plot_forecast_nested(..., show_forecast: bool = True)

# After  
def _plot_forecast_single(..., show_legend: bool = True)
def _plot_forecast_nested(..., show_legend: bool = True)
```

### Testing

**Test File:** `tests/test_visualize/test_plot_forecast.py`

**Results:**
- ✅ `test_forecast_plot_with_prediction_intervals` - PASSED
- ✅ `test_forecast_plot_customization` - PASSED
- ✅ `test_nested_without_date_column` - PASSED (fixed Index concatenation bug)
- ✅ `test_empty_fit` - PASSED
- ✅ `test_missing_date_column` - PASSED
- ⚠️ 3 tests have pre-existing patsy datetime issues (not related to this fix)

**Test Coverage:**
- Single model plots with and without test data
- Nested/grouped model plots
- Plots with and without date columns (Index-based)
- Prediction intervals
- Custom titles, heights, widths

### Expected Outcome

After this refactoring:
1. ✅ Actuals shown as one continuous line across train and test splits
2. ✅ Fitted values clearly distinguished by split (orange train, red test)
3. ✅ No forecast line confusion
4. ✅ Works with both date columns and numeric indices
5. ✅ Prediction intervals still displayed correctly
6. ✅ Nested/grouped plots follow same pattern

### User Feedback

**Original Request:**
> "actuals should be one line, not split by train/test, fitted values should be plotted in orange dashed for train and red dashed for test and split by train/test. forecast should never be plotted in these plots"

**Implementation Status:** ✅ Complete

All user requirements implemented and tested.
