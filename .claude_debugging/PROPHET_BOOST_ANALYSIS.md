# Prophet Boost Implementation Analysis: R vs Python

**Investigation Date:** 2025-11-05
**Objective:** Analyze R modeltime prophet_boost implementation to diagnose sharp residual increase in test set for Python implementation

---

## Executive Summary

After analyzing the R modeltime reference implementation, I've identified **critical differences** between the R and Python implementations that likely explain the sharp residual increase in test data. The key issue is **feature engineering for XGBoost** - the R implementation uses actual exogenous regressors while the Python implementation generates cyclical time features.

---

## R Implementation Architecture

### File Structure
- **Model Specification:** `R/parsnip-prophet_boost.R`
- **Engine Registration:** `R/parsnip-prophet_boost_data.R`
- **Core Implementation:** `prophet_xgboost_fit_impl()` function (lines 410-587)
- **Prediction:** `prophet_xgboost_predict_impl()` function (lines 649-690)
- **Tests:** `tests/testthat/test-algo-prophet_boost.R`

### Two-Stage Hybrid Strategy

**Stage 1: Prophet Model**
```r
# Line 496-514
fit_prophet <- prophet::prophet(
    df = df,  # Contains y and ds (date)
    growth = growth,
    changepoints = changepoints,
    n.changepoints = n.changepoints,
    # ... other parameters
    fit = TRUE
)
```

**Stage 2: XGBoost on Residuals**
```r
# Lines 517-519 - Calculate Prophet residuals
prophet_fitted    <- stats::predict(fit_prophet, df) %>% dplyr::pull(yhat)
prophet_residuals <- outcome - prophet_fitted

# Lines 524-539 - Fit XGBoost on residuals using XREGS
if (!is.null(xreg_tbl)) {
    fit_xgboost <- xgboost_impl(
        x = xreg_tbl,           # CRITICAL: Uses actual exogenous regressors
        y = prophet_residuals,  # Target is Prophet residuals
        max_depth = max_depth,
        nrounds = nrounds,
        eta  = eta,
        ...
    )
    xgboost_fitted <- xgboost_predict(fit_xgboost, newdata = xreg_tbl)
} else {
    fit_xgboost    <- NULL
    xgboost_fitted <- rep(0, length(prophet_residuals))  # Zero if no xregs
}
```

**Prediction Combination**
```r
# Lines 560-562 - Final fitted values
.fitted = prophet_fitted + xgboost_fitted
.residuals = .actual - .fitted
```

---

## Critical Discovery: XREG Handling

### R Implementation (Lines 476-478)
```r
# XREGS - Clean names, get xreg recipe, process predictors
xreg_recipe <- create_xreg_recipe(predictor, prepare = TRUE)
xreg_tbl    <- juice_xreg_recipe(xreg_recipe, format = "tbl")
```

**What this does:**
1. Extracts **actual exogenous regressors** from the formula (e.g., `month(date)`, `as.numeric(date)`, fourier terms)
2. Processes them through a recipe (encoding, transformation)
3. Passes them to XGBoost as features

**Example from tests (line 119):**
```r
fit(log(value) ~ date + as.numeric(date) + factor(month(date, label = TRUE), ordered = F),
    data = training(splits))
```

The formula includes:
- `date` - Used by Prophet
- `as.numeric(date)` - **Exogenous regressor for XGBoost**
- `factor(month(...))` - **Exogenous regressor for XGBoost**

### R Implementation: No XREGs Case (Lines 541-544)
```r
} else {
    fit_xgboost       <- NULL
    xgboost_fitted    <- rep(0, length(prophet_residuals))  # ZEROS!
}
```

**Key Insight:** When there are NO exogenous regressors (formula is just `y ~ date`), XGBoost contributes ZERO to predictions. This is intentional - Prophet handles all time-based patterns.

---

## Python Implementation Analysis

### File: `py_parsnip/engines/hybrid_prophet_boost.py`

**Feature Engineering (Lines 58-139)**
```python
@staticmethod
def _create_xgb_features(dates: pd.Series, date_min: pd.Timestamp = None) -> np.ndarray:
    """
    Create cyclical time features for XGBoost to capture seasonality.

    Features include:
    - days_since_start: Linear time trend
    - day_of_week, day_of_month, day_of_year, month, week_of_year
    - Cyclical encodings (sin/cos) for:
      - day_of_week (weekly patterns)
      - day_of_year (yearly patterns)
      - month (monthly patterns)
    """
    # ... generates 12 time-based features from dates alone
```

**Fitting (Lines 238-262)**
```python
# Create cyclical time features for XGBoost
dates = pd.to_datetime(prophet_df["ds"])
X_boost = self._create_xgb_features(dates, date_min=None)  # GENERATED features

# Target for XGBoost is Prophet residuals
y_boost = prophet_residuals

# Fit XGBoost
xgb_model = XGBRegressor(**xgb_params)
xgb_model.fit(X_boost, y_boost)
```

**Prediction (Lines 372-383)**
```python
# Create cyclical time features for XGBoost (same features as training)
dates = pd.to_datetime(date_series)
X_boost = self._create_xgb_features(dates, date_min=date_min)

# Get XGBoost predictions
xgb_pred = xgb_model.predict(X_boost)
```

---

## ROOT CAUSE IDENTIFIED

### Problem 1: XGBoost Always Runs (Even When Inappropriate)

**R Implementation:**
- If formula is `y ~ date` (no xregs), XGBoost contributes **zeros**
- XGBoost only activates when you explicitly provide exogenous regressors

**Python Implementation:**
- ALWAYS generates 12 cyclical time features from dates
- XGBoost ALWAYS trains and predicts, even when not needed
- This causes **extrapolation issues** when predicting beyond training range

### Problem 2: Feature Engineering Mismatch

**R Implementation:**
- Uses **actual exogenous variables** from formula (domain-specific features)
- Examples: `as.numeric(date)`, `month(date)`, `fourier_vec(date, period=12)`
- These are **meaningful features** that generalize to test data

**Python Implementation:**
- Generates **generic cyclical encodings** from dates alone
- Features like `sin_day_of_year`, `cos_month`, `days_since_start`
- Problem: `days_since_start` uses training minimum date as baseline
- **Extrapolation Issue:** When test dates >> training dates, `days_since_start` becomes very large
- XGBoost trained on `days_since_start` in range [0, 365] but predicts on [365, 730]
- This causes **systematic bias** in residuals

### Problem 3: Date Range Extrapolation

**R Implementation:**
```r
# Line 676 - Prediction uses same xreg_tbl processing
xreg_tbl <- bake_xreg_recipe(xreg_recipe, new_data, format = "tbl")
```
- Exogenous features are **recalculated** for new data
- Features like `month(date)` are **cyclical by nature** (Jan-Dec repeats)
- Linear features like `as.numeric(date)` continue naturally

**Python Implementation:**
```python
# Line 242 - Uses training data's minimum date
X_boost = self._create_xgb_features(dates, date_min=None)  # date_min from training

# Line 380 - Prediction uses SAME date_min from training
X_boost = self._create_xgb_features(dates, date_min=date_min)  # STORED from training
```
- `days_since_start` calculated as `(test_date - train_min_date).days`
- For test data far from training, this becomes **out-of-distribution**
- XGBoost sees values it was never trained on

---

## Why Test Residuals Increase Sharply

### Training Phase:
1. Prophet captures trend + seasonality → leaves residuals
2. XGBoost trains on features in range: `days_since_start = [0, 365]`
3. XGBoost learns patterns like: "at day 180, residual is typically +5"
4. Combined model: `pred = prophet_pred + xgb_pred` → low training residuals

### Test Phase (30+ days after training):
1. Prophet still captures trend + seasonality correctly
2. XGBoost gets features: `days_since_start = [395, 425]` (out of range!)
3. XGBoost **extrapolates** poorly on unseen `days_since_start` values
4. XGBoost predictions become **systematically biased** (too high or too low)
5. Combined model: `pred = prophet_pred + biased_xgb_pred` → **sharp residual increase**

### Mathematical Explanation:
```
Training residuals = actuals - (prophet_pred + xgb_pred)
                   = actuals - prophet_pred - xgb_pred
                   ≈ 0  (XGBoost corrects Prophet errors)

Test residuals = actuals - (prophet_pred + xgb_pred_extrapolated)
               = actuals - prophet_pred - extrapolated_xgb_pred
               ≠ 0  (extrapolated XGBoost adds noise instead of correction)
```

---

## R Test Expectations

From `tests/testthat/test-algo-prophet_boost.R`:

### Test WITHOUT Exogenous Regressors (Lines 42-108)
```r
model_fit <- model_spec %>%
    fit(log(value) ~ date, data = training(splits))  # NO xregs

# Lines 100-106 - Accuracy expectations
resid <- testing(splits)$value - exp(predictions_tbl$.value)

# - Max Error less than 1500
expect_lte(max(abs(resid)), 1500)

# - MAE less than 700
expect_lte(mean(abs(resid)), 700)
```

**Interpretation:** Even without xregs, model should maintain reasonable accuracy on test set

### Test WITH Exogenous Regressors (Lines 112-197)
```r
model_fit <- model_spec %>%
    fit(log(value) ~ date + as.numeric(date) + factor(month(date, label = TRUE), ordered = F),
        data = training(splits))

# Lines 189-195 - Same accuracy expectations
resid <- testing(splits)$value - exp(predictions_tbl$.value)

# - Max Error less than 1500
expect_lte(max(abs(resid)), 1500)

# - MAE less than 700
expect_lte(mean(abs(resid)), 700)
```

**Interpretation:** With xregs, model should still maintain same accuracy (xregs help XGBoost)

---

## Recommendations for Python Implementation

### Option 1: Match R Behavior (Recommended)

**Modify `fit_raw()` to extract exogenous regressors from formula:**

```python
def fit_raw(self, spec: ModelSpec, data: pd.DataFrame, formula: str, date_col: str = None):
    # Parse formula to extract BOTH outcome and exogenous variables
    outcome_name, exog_vars = _parse_ts_formula(formula, inferred_date_col)

    if exog_vars:
        # Extract exogenous features from data
        X_boost = data[exog_vars].copy()
        # Apply encoding/transformation as needed
        X_boost = self._process_xregs(X_boost)
    else:
        # NO exogenous regressors -> XGBoost contributes zero
        X_boost = None

    # Stage 2: Fit XGBoost (only if xregs exist)
    if X_boost is not None:
        xgb_model = XGBRegressor(**xgb_params)
        xgb_model.fit(X_boost, prophet_residuals)
        xgb_fitted = xgb_model.predict(X_boost)
    else:
        xgb_model = None
        xgb_fitted = np.zeros(len(prophet_residuals))  # ZEROS like R
```

**Benefits:**
- Matches R behavior exactly
- Only uses XGBoost when meaningful features provided
- Avoids extrapolation issues
- Generalizes better to test data

### Option 2: Fix Cyclical Feature Engineering

**If keeping auto-generated features, fix the extrapolation issue:**

```python
def _create_xgb_features(dates: pd.Series, date_min: pd.Timestamp = None) -> np.ndarray:
    # REMOVE days_since_start (linear extrapolation issue)
    # days_since_start = (dates - date_min).dt.days  # REMOVE THIS

    # Keep ONLY cyclical features (these repeat and don't extrapolate)
    day_of_week = dates.dt.dayofweek  # 0-6 (repeats weekly)
    day_of_month = dates.dt.day  # 1-31 (repeats monthly)
    month = dates.dt.month  # 1-12 (repeats yearly)

    # Cyclical encodings (naturally periodic)
    sin_day_of_week = np.sin(2 * np.pi * day_of_week / 7)
    cos_day_of_week = np.cos(2 * np.pi * day_of_week / 7)
    sin_month = np.sin(2 * np.pi * month / 12)
    cos_month = np.cos(2 * np.pi * month / 12)

    # ONLY cyclical features (no linear trend)
    features = np.column_stack([
        day_of_week, day_of_month, month,
        sin_day_of_week, cos_day_of_week,
        sin_month, cos_month
    ])

    return features
```

**Benefits:**
- Cyclical features don't extrapolate poorly
- Test data features are in same range as training
- Still captures seasonal patterns XGBoost can learn

**Drawbacks:**
- Loses ability to capture trend changes (but Prophet handles trend anyway)

### Option 3: Hybrid Approach with Exogenous Support

**Best of both worlds:**

```python
def fit_raw(self, spec: ModelSpec, data: pd.DataFrame, formula: str, date_col: str = None):
    # Parse formula
    outcome_name, exog_vars = _parse_ts_formula(formula, inferred_date_col)

    # Create feature matrix
    if exog_vars:
        # Use user-provided exogenous regressors
        X_boost = data[exog_vars].copy()
        X_boost = self._process_xregs(X_boost)
    else:
        # Fallback: Generate ONLY cyclical features (no linear)
        dates = pd.to_datetime(data[inferred_date_col])
        X_boost = self._create_cyclical_features(dates)  # NEW: only cyclical

    # Always fit XGBoost (either on xregs or cyclical features)
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(X_boost, prophet_residuals)
```

---

## Key Differences Summary

| Aspect | R Implementation | Python Implementation | Impact |
|--------|------------------|----------------------|--------|
| **XGBoost Features** | Actual exogenous regressors from formula | Auto-generated cyclical time features | High |
| **No XREGs Behavior** | XGBoost contributes zeros | XGBoost still trains on time features | High |
| **Feature Extrapolation** | Exogenous features recalculated for new data | `days_since_start` uses training baseline | **Critical** |
| **Test Generalization** | Features in-distribution | Features out-of-distribution | **Critical** |
| **Residual Pattern** | Stable across train/test | Sharp increase in test set | **Observed Issue** |

---

## Validation Strategy

### To Confirm Diagnosis:

1. **Test with NO exogenous regressors:**
   ```python
   spec = prophet_boost(trees=100)
   fit = spec.fit(data, "sales ~ date")  # Only date
   ```
   - Check if test residuals still spike
   - If yes → confirms auto-feature issue

2. **Test with exogenous regressors:**
   ```python
   # Add month as exogenous regressor
   data['month_num'] = data['date'].dt.month
   fit = spec.fit(data, "sales ~ date + month_num")
   ```
   - Check if test residuals improve
   - If yes → confirms need for actual xregs

3. **Remove days_since_start:**
   - Temporarily modify `_create_xgb_features()` to exclude `days_since_start`
   - Check if test residuals stabilize
   - If yes → confirms extrapolation issue

---

## Conclusion

The **sharp residual increase in test set** is caused by:

1. **XGBoost always running** (even when formula has no exogenous regressors)
2. **Auto-generated time features** that don't match R's exogenous regressor approach
3. **Linear feature extrapolation** (`days_since_start`) going out of distribution

**Recommended Fix:** Implement Option 1 (Match R Behavior) to:
- Only use XGBoost when exogenous regressors provided
- Extract actual features from formula (not auto-generate)
- Avoid extrapolation issues on test data

This will align Python implementation with R's proven architecture and resolve the residual spike issue.

---

## References

- **R Source:** `/reference/modeltime-master/R/parsnip-prophet_boost.R`
- **Python Source:** `py_parsnip/engines/hybrid_prophet_boost.py`
- **R Tests:** `/reference/modeltime-master/tests/testthat/test-algo-prophet_boost.R`
- **Python Tests:** `tests/test_parsnip/test_prophet_boost.py`
