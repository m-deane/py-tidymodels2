# Grouped Modeling Data Preparation - Notebook Update

**Date:** 2025-11-09
**Notebook:** `_md/forecasting_grouped.ipynb`
**Task:** Add grouped/panel modeling preparation cells

## Changes Made

Successfully added 3 new cells at index 11-13 (after train/test split):

### Cell 11: Markdown - Grouped Modeling Explanation
- Explains nested vs global modeling approaches
- Documents group column (country) and number of groups (10)
- Sets expectations for nested approach usage in examples

### Cell 12: Code - Formula Definitions and Group Structure
```python
FORMULA_NESTED = "refinery_kbd ~ ."  # Dot notation - excludes date automatically
FORMULA_GLOBAL = "refinery_kbd ~ ."  # Same formula, but fit_global adds 'country'

# Display group structure:
# - Total countries
# - Country list
# - Observations per country
# - Date ranges per country
```

**Purpose:**
- Define standard formulas for both modeling approaches
- Show panel data structure for validation
- Verify data balance across groups

### Cell 13: Code - Group Validation
```python
# Validate test groups are subset of train groups
train_countries = set(train_mix['country'].unique())
test_countries = set(test_mix['country'].unique())

if not test_countries.issubset(train_countries):
    print("⚠️ WARNING: Test has unseen countries:", test_countries - train_countries)
else:
    print("✓ All test countries present in training data")
```

**Purpose:**
- Critical validation for nested models
- Ensures all test groups have trained models
- Prevents prediction failures on unseen groups

## Cell Placement

Inserted after cell 10 (train_mix/test_mix split):
```
Cell 9:  split_relative1 (relative dates)
Cell 10: split_mixed (train_mix, test_mix) ← ANCHOR POINT
Cell 11: [NEW] Markdown - Grouped modeling explanation
Cell 12: [NEW] Code - Formula definitions + group structure
Cell 13: [NEW] Code - Group validation
Cell 14: "# 1. Single Model Fitting" (original cell)
```

## Implementation Notes

### Dot Notation Formula
Both formulas use `"refinery_kbd ~ ."` which:
- Expands to all predictors except outcome
- **Automatically excludes** `date` column (datetime exclusion)
- **Nested approach:** Excludes `country` (not in formula)
- **Global approach:** `fit_global()` adds `country` as predictor

### Why Validation is Critical
Nested modeling (`fit_nested()`) creates **independent models per group**:
- If test has country "X" but train doesn't → **Prediction fails**
- Validation catches this early before model fitting
- Global modeling doesn't need this (country is a feature)

### Expected Output from Cell 12
```
=== Panel Data Structure ===
Total countries: 10
Countries: ['Country_A', 'Country_B', ...]

Observations per country (train):
Country_A    132
Country_B    132
...

Date range per country (train):
           min         max
Country_A  2008-01-01  2018-12-01
Country_B  2008-01-01  2018-12-01
...
```

## Next Steps

These cells prepare the notebook for:
1. **Nested model examples** using `fit_nested(train_mix, group_col='country')`
2. **Global model examples** using `fit_global(train_mix, group_col='country')`
3. **Comparison** between nested vs global approaches
4. **Group-wise evaluation** using outputs with group column

## File Location
- Modified: `/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/_md/forecasting_grouped.ipynb`
- Total cells added: 3 (1 markdown + 2 code)
- Insertion index: 11-13 (after train/test split)

## Verification Status
- ✅ Cells successfully inserted
- ✅ Proper placement after train_mix/test_mix split
- ✅ Markdown formatting correct
- ✅ Code syntax valid
- ✅ Cell execution order preserved
