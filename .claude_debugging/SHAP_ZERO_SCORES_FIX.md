# SHAP Zero Scores Bug Fix

**Date:** 2025-11-09
**Issue:** All SHAP importance scores were showing as 0 in cell 84
**Status:** ✅ FIXED

## Root Cause

The SHAP step has a try-except block that catches exceptions and returns zero scores. When SHAP calculation fails due to feature mismatch between the model and the data, it silently returns zeros without showing the error.

### Feature Mismatch Problem

```python
# Model was trained on these features:
X_train = train_data.drop(['target', 'date'], axis=1)  # May include non-numeric columns
model.fit(X_train, y_train)

# But SHAP step might see different features after preprocessing:
# - Categorical columns might be one-hot encoded
# - Column order might differ
# - Feature names might not match model.feature_names_in_
```

## Fixes Applied

### 1. Improved Error Handling in SHAP Step

**File:** `py_recipes/steps/filter_supervised.py`

**Line 1297-1308:** Enhanced error message to show feature mismatch details:
```python
except Exception as e:
    error_msg = (
        f"SHAP calculation failed: {e}\n"
        f"Model type: {type(self.model).__name__}\n"
        f"Model features: {getattr(self.model, 'feature_names_in_', 'N/A')}\n"
        f"Data features: {X_sample.columns.tolist()}\n"
        "Using zero scores as fallback."
    )
    warnings.warn(error_msg, UserWarning)
```

### 2. Feature Alignment Check

**File:** `py_recipes/steps/filter_supervised.py`

**Line 1244-1265:** Added pre-check to align features before SHAP calculation:
```python
# Verify model features match data features
if hasattr(self.model, 'feature_names_in_'):
    model_features = list(self.model.feature_names_in_)
    data_features = X_sample.columns.tolist()

    if set(model_features) != set(data_features):
        missing_in_data = set(model_features) - set(data_features)
        extra_in_data = set(data_features) - set(model_features)

        if missing_in_data:
            warnings.warn(
                f"Model was trained on features not in data: {missing_in_data}. "
                "Using zero scores for mismatched features.",
                UserWarning
            )
            return {col: 0.0 for col in X.columns}

        # Only use features the model knows about, in the correct order
        if extra_in_data or model_features != data_features:
            X_sample = X_sample[[col for col in model_features if col in data_features]]
```

### 3. Fixed Notebook Example

**File:** `_md/forecasting_recipes.ipynb` (Cell 84)

**Changes:**
- Explicitly filter to numeric columns only before training model
- Add diagnostic output showing features used
- Add check for zero scores with helpful error message

**New code pattern:**
```python
# Get predictors (exclude target and date)
X_train = train_data.drop(['target', 'date'], axis=1)
y_train = train_data['target']

# Ensure we're using numeric features only
numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
X_train = X_train[numeric_cols]

print(f"Training features ({len(numeric_cols)}): {numeric_cols[:5]}...")

# Fit model
shap_model = XGBRegressor(...)
shap_model.fit(X_train, y_train)

# ... later, after prep ...

# Check if SHAP calculation succeeded
if shap_step._scores and any(score > 0 for score in shap_step._scores.values()):
    # Display scores
    ...
else:
    print("⚠️ SHAP calculation returned zero scores")
    print("  This may indicate a feature mismatch between model and data")
```

## How It Works Now

### Before Fix
1. Model trained on `X_train` (may include mixed types)
2. SHAP step tries to compute importance
3. Feature mismatch causes exception
4. Exception caught silently, returns zeros
5. User sees all scores = 0 with no explanation

### After Fix
1. Model trained explicitly on numeric features only
2. Feature names printed for verification
3. SHAP step checks if features match
4. If mismatch: clear warning with details
5. If exception: detailed error message with feature lists
6. Notebook checks for zero scores and shows diagnostic message

## Testing the Fix

Run the updated notebook cell 84. You should now see:

**If successful:**
```
=== Fitting Tree Model for SHAP Analysis ===
Training features (50): ['x1', 'x2', 'x3', 'x4', 'x5']...
✓ Tree model fitted with 100 estimators
  Training R²: 0.8523

=== Computing SHAP Values ===

=== SHAP Feature Importance (Top 10) ===
   feature  shap_importance
0       x1           0.4523
1       x2           0.3211
...
```

**If feature mismatch:**
```
UserWarning: Model was trained on features not in data: {'old_feature'}
⚠️ SHAP calculation returned zero scores
  This may indicate a feature mismatch between model and data
```

**If SHAP fails:**
```
UserWarning: SHAP calculation failed: <error details>
Model type: XGBRegressor
Model features: ['x1', 'x2', 'x3']
Data features: ['x1', 'x2', 'x4']  # x3 missing, x4 extra!
Using zero scores as fallback.
```

## Prevention

To avoid this issue in future examples:

1. **Always use numeric features only:**
   ```python
   numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
   X_numeric = X[numeric_cols]
   model.fit(X_numeric, y)
   ```

2. **Check model features match data:**
   ```python
   if hasattr(model, 'feature_names_in_'):
       print(f"Model expects: {list(model.feature_names_in_)}")
   print(f"Data has: {X.columns.tolist()}")
   ```

3. **Enable warnings in notebooks:**
   ```python
   import warnings
   warnings.filterwarnings('default')  # Show all warnings
   ```

## Related Files

- Implementation: `py_recipes/steps/filter_supervised.py`
  - Lines 1244-1265: Feature alignment check
  - Lines 1297-1308: Enhanced error handling
- Notebook: `_md/forecasting_recipes.ipynb`
  - Cell 84: Fixed SHAP example with diagnostics
- Tests: `tests/test_recipes/test_select_shap.py` (all 11 tests still passing)

## Impact

- Users now get clear error messages when features don't match
- SHAP step automatically aligns features when possible
- Notebook example explicitly uses numeric features only
- Diagnostic output helps debug feature mismatch issues

## Summary

The fix ensures that:
1. Feature mismatches are detected early and clearly communicated
2. Features are automatically aligned when possible
3. Users see helpful error messages instead of silent zeros
4. The notebook example follows best practices for feature selection
