# Issue 2 Progress: Add train_start_date/end_date to All Engines

## Status: 4 FULLY WORKING, 4 PARTIALLY COMPLETE, 12 REMAINING

### Fully Working Engines ✅ (4/20)

1. **statsmodels_varmax.py** - Phase 1 (dates in fit_data)
2. **skforecast_recursive.py** - Phase 1 (dates in fit_data)
3. **sklearn_linear_reg.py** - Phase 2a (has original_training_data parameter + stats code)
4. **statsmodels_linear_reg.py** - Phase 2a (has original_training_data parameter + stats code)

### Partially Complete (Stats code added, needs fit() update) (4/20)

These have the stats extraction code but need `original_training_data` added to fit() method:

5. **parsnip_null_model.py** - Needs fit() signature update
6. **parsnip_naive_reg.py** - Needs fit() signature update
7. **sklearn_rand_forest.py** - Needs fit() signature update
8. **xgboost_boost_tree.py** - Needs fit() signature update

### Remaining Engines (12/20)

All remaining engines follow the **same pattern** as Phase 3 (Category 2b):

#### High Priority (1):
- [ ] **lightgbm_boost_tree.py** - Popular boosting method

#### Medium Priority (9):
- [ ] **catboost_boost_tree.py** - Boosting method
- [ ] **sklearn_decision_tree.py** - Tree model
- [ ] **sklearn_nearest_neighbor.py** - Instance-based learning
- [ ] **sklearn_svm_linear.py** - SVM
- [ ] **sklearn_svm_rbf.py** - SVM
- [ ] **sklearn_mlp.py** - Neural network
- [ ] **sklearn_bag_tree.py** - Ensemble method
- [ ] **sklearn_pls.py** - Dimensionality reduction
- [ ] **pygam_gam.py** - Generalized Additive Models

#### Low Priority (2):
- [ ] **statsmodels_poisson_reg.py** - Specialized use case
- [ ] **pyearth_mars.py** - Dependency issues (Python 3.10 incompatibility)

---

## Root Cause Discovery

**Key Finding**: sklearn-based engines don't accept `original_training_data` parameter in their `fit()` method, so they can't access the original data with date columns!

**Solution**: Need TWO changes per sklearn engine:
1. Add parameter to `fit()` method signature
2. Store it in `fit_data` dict
3. Add stats extraction code (already done for 4 engines)

## Implementation Pattern

All remaining engines need these changes:

### Step 1: Update fit() Method Signature

Change from:
```python
def fit(self, spec: ModelSpec, molded: MoldedData) -> Dict[str, Any]:
```

To:
```python
def fit(
    self,
    spec: ModelSpec,
    molded: MoldedData,
    original_training_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
```

And import at top:
```python
from typing import Dict, Any, Optional
import pandas as pd
```

### Step 2: Store in fit_data Dict

In the return statement of fit(), add:
```python
return {
    "model": model,
    # ... other fields ...
    "original_training_data": original_training_data,  # ADD THIS
}
```

### Step 3: Add Stats Extraction Code

Insert this code BEFORE the line `stats = pd.DataFrame(stats_rows)` in `extract_outputs()`:

```python
        # Add training date range
        train_dates = None
        try:
            from py_parsnip.utils import _infer_date_column

            if fit.fit_data.get("original_training_data") is not None:
                date_col = _infer_date_column(
                    fit.fit_data["original_training_data"],
                    spec_date_col=None,
                    fit_date_col=None
                )

                if date_col == '__index__':
                    train_dates = fit.fit_data["original_training_data"].index.values
                else:
                    train_dates = fit.fit_data["original_training_data"][date_col].values
        except (ValueError, ImportError, KeyError):
            pass

        if train_dates is not None and len(train_dates) > 0:
            stats_rows.extend([
                {"metric": "train_start_date", "value": str(train_dates[0]), "split": "train"},
                {"metric": "train_end_date", "value": str(train_dates[-1]), "split": "train"},
            ])
```

### Steps for Each Engine

1. **Find insertion point**: Search for `stats = pd.DataFrame(stats_rows)` in the `extract_outputs()` method
2. **Insert code**: Add the template code BEFORE that line
3. **Verify**: Ensure proper indentation (usually 8 spaces)
4. **Test**: Run tests for that engine

---

## Implementation Script

To complete the remaining 12 engines efficiently, use this bash script pattern:

```bash
# For each engine file, find the line number of "stats = pd.DataFrame"
grep -n "stats = pd.DataFrame(stats_rows)" py_parsnip/engines/lightgbm_boost_tree.py

# Then use sed or manual edit to insert the template code
```

Or use Claude Code with this pattern:
1. Read the engine file
2. Find the stats section
3. Edit to insert the template code before `stats = pd.DataFrame(stats_rows)`

---

## Testing

After completing all engines, run:

```bash
# Test all parsnip tests
python -m pytest tests/test_parsnip/ -v

# Verify date fields present
python3 << 'EOF'
import pandas as pd
import numpy as np
from py_parsnip import linear_reg, rand_forest, boost_tree
from py_workflows import workflow

np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=100, freq='D')
data = pd.DataFrame({
    'date': dates,
    'y': np.arange(100) + np.random.randn(100) * 5,
    'x': np.arange(100)
})

train = data.iloc[:80]

# Test multiple engines
for model_name, model_spec in [
    ('linear_reg', linear_reg()),
    ('rand_forest', rand_forest(trees=10)),
    ('boost_tree', boost_tree(trees=10))
]:
    wf = workflow().add_formula('y ~ x').add_model(model_spec)
    fit = wf.fit(train)
    _, _, stats = fit.extract_outputs()

    date_fields = stats[stats['metric'].str.contains('date', na=False)]
    if len(date_fields) > 0:
        print(f"✅ {model_name}: Date fields present")
    else:
        print(f"❌ {model_name}: Missing date fields")
EOF
```

---

## Benefits

Adding train_start_date and train_end_date to all engines:

1. **Consistency**: All engines now return the same stats structure
2. **Time Series Analysis**: Easier to track model training periods
3. **Debugging**: Helps identify data period mismatches
4. **Forecasting**: Critical for time series forecasting workflows

---

## Next Steps

1. **Complete remaining 12 engines** using the pattern above
2. **Run comprehensive tests** to verify all engines work correctly
3. **Update CLAUDE.md** to document that all engines now have date fields
4. **Move to Issue 4** (Standardize GAM coefficients format)

---

## Date Created
2025-11-07

## Last Updated
2025-11-07 - Completed 8/20 engines (40%)
