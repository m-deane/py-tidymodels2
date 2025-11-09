# step_fourier() Column Naming Enhancement

**Date:** 2025-11-09
**Issue:** Fourier feature column names didn't clearly show the period and k values
**Status:** ✅ FIXED

---

## Problem

The `step_fourier()` function was creating column names like:
- `fourier_sin_1`, `fourier_cos_1`
- `fourier_sin_2`, `fourier_cos_2`
- etc.

While the k value was technically included (the `_1`, `_2` part), the naming wasn't explicit enough and didn't include the period information, making it difficult to:
1. Identify which period each Fourier feature corresponds to
2. Distinguish between multiple `step_fourier()` calls with different periods
3. Understand the harmonic number (k value) at a glance

---

## Solution

Enhanced the column naming convention to include both the period and k value explicitly:

### New Format
```
{prefix}p{period}_k{k}_sin
{prefix}p{period}_k{k}_cos
```

### Examples

**Monthly seasonality (period=12, K=3):**
```
fourier_p12_k1_sin, fourier_p12_k1_cos
fourier_p12_k2_sin, fourier_p12_k2_cos
fourier_p12_k3_sin, fourier_p12_k3_cos
```

**Yearly seasonality (period=365, K=2):**
```
fourier_p365_k1_sin, fourier_p365_k1_cos
fourier_p365_k2_sin, fourier_p365_k2_cos
```

**Fractional periods (period=12.5, K=2):**
```
fourier_p12.5_k1_sin, fourier_p12.5_k1_cos
fourier_p12.5_k2_sin, fourier_p12.5_k2_cos
```

**Custom prefix (prefix="season_", period=12, K=2):**
```
season_p12_k1_sin, season_p12_k1_cos
season_p12_k2_sin, season_p12_k2_cos
```

---

## Benefits

### 1. Clear Period Identification
You can immediately see which period each feature represents:
- `fourier_p12_k1_sin` → period of 12 (monthly)
- `fourier_p365_k1_sin` → period of 365 (yearly)

### 2. Clear Harmonic Number (k)
The k value is explicitly labeled:
- `k1` → first harmonic (fundamental frequency)
- `k2` → second harmonic (double frequency)
- `k3` → third harmonic (triple frequency)

### 3. Multiple Periods Support
When using multiple `step_fourier()` calls with different periods, features are clearly distinguishable:

```python
rec = (
    recipe()
    .step_fourier("date", period=12, K=3)   # Monthly seasonality
    .step_fourier("date", period=4, K=2)    # Quarterly seasonality
)
```

Creates:
```
fourier_p12_k1_sin, fourier_p12_k1_cos, fourier_p12_k2_sin, ...
fourier_p4_k1_sin, fourier_p4_k1_cos, fourier_p4_k2_sin, ...
```

### 4. Easy Filtering
Can easily filter features by period or k value:

```python
# Get all monthly features (period=12)
monthly_features = [col for col in df.columns if 'p12_' in col]

# Get all first harmonics (k=1)
first_harmonics = [col for col in df.columns if '_k1_' in col]

# Get all sine components
sine_features = [col for col in df.columns if col.endswith('_sin')]
```

---

## Implementation Details

### Files Modified

**`py_recipes/steps/timeseries_extended.py`**

1. **prep() method** (lines 281-288):
   ```python
   # Generate feature names
   # Format: {prefix}p{period}_k{k}_sin / cos
   # Example: fourier_p12_k1_sin, fourier_p12_k1_cos
   feature_names = []
   period_label = str(int(self.period)) if self.period == int(self.period) else f"{self.period:.1f}"
   for k in range(1, self.K + 1):
       feature_names.append(f"{self.prefix}p{period_label}_k{k}_sin")
       feature_names.append(f"{self.prefix}p{period_label}_k{k}_cos")
   ```

2. **bake() method** - Updated in 3 locations:
   - Lines 360-377: pytimetk success path
   - Lines 381-386: pytimetk fallback path
   - Lines 399-404: Manual creation (no pytimetk)

### Period Label Logic

Integer periods display without decimals:
- `period=12` → `p12`
- `period=365` → `p365`

Fractional periods show one decimal place:
- `period=12.5` → `p12.5`
- `period=7.3` → `p7.3`

---

## Usage Examples

### Basic Usage
```python
from py_recipes import recipe

# Create recipe with Fourier features
rec = recipe().step_fourier("date", period=12, K=3)

# Prep and bake
prepped = rec.prep(train_data)
transformed = prepped.bake(train_data)

# Result columns:
# fourier_p12_k1_sin, fourier_p12_k1_cos
# fourier_p12_k2_sin, fourier_p12_k2_cos
# fourier_p12_k3_sin, fourier_p12_k3_cos
```

### Multiple Periods
```python
# Capture both monthly and yearly seasonality
rec = (
    recipe()
    .step_fourier("date", period=12, K=3, prefix="monthly_")
    .step_fourier("date", period=365, K=2, prefix="yearly_")
)

# Creates:
# monthly_p12_k1_sin, monthly_p12_k1_cos, ...
# yearly_p365_k1_sin, yearly_p365_k1_cos, ...
```

### In Workflows
```python
from py_workflows import workflow
from py_parsnip import linear_reg

wf = (
    workflow()
    .add_recipe(
        recipe()
        .step_fourier("date", period=12, K=5)
        .step_normalize(all_numeric_predictors())
    )
    .add_model(linear_reg())
)

fit = wf.fit(train_data)
```

---

## Test Results

All tests passing:

```
✅ Test 1: period=12, K=3
   Created: fourier_p12_k1_sin, fourier_p12_k1_cos, fourier_p12_k2_sin, ...

✅ Test 2: period=365, K=2
   Created: fourier_p365_k1_sin, fourier_p365_k1_cos, ...

✅ Test 3: period=12.5, K=2 (fractional)
   Created: fourier_p12.5_k1_sin, fourier_p12.5_k1_cos, ...

✅ Test 4: Custom prefix
   Created: season_p12_k1_sin, season_p12_k1_cos, ...
```

---

## Backward Compatibility

### Breaking Change Notice

This is a **breaking change** for existing code that references Fourier column names.

**Old names:**
```python
fourier_sin_1, fourier_cos_1
fourier_sin_2, fourier_cos_2
```

**New names:**
```python
fourier_p12_k1_sin, fourier_p12_k1_cos
fourier_p12_k2_sin, fourier_p12_k2_cos
```

### Migration

If you have code that explicitly references Fourier column names, update them:

```python
# Old code
df['fourier_sin_1']

# New code (if period=12)
df['fourier_p12_k1_sin']

# Or use pattern matching
[col for col in df.columns if col.startswith('fourier_p12_k1_')]
```

---

## Related Documentation

- `py_recipes/steps/timeseries_extended.py` - Implementation
- `FORECASTING_RECIPES_NOTEBOOK_EXPANSION.md` - Notebook examples
- `QUICK_RECIPE_REFERENCE.md` - Recipe step reference

---

## Summary

✅ **FIXED** - Fourier feature names now explicitly include period and k values
- Format: `{prefix}p{period}_k{k}_sin` / `{prefix}p{period}_k{k}_cos`
- Examples: `fourier_p12_k1_sin`, `fourier_p365_k2_cos`
- Benefits: Clear identification, multi-period support, easy filtering
- All tests passing
- Breaking change: Update code that references old column names

**Example:**
```python
rec = recipe().step_fourier("date", period=12, K=3)
# Creates: fourier_p12_k1_sin, fourier_p12_k1_cos,
#          fourier_p12_k2_sin, fourier_p12_k2_cos,
#          fourier_p12_k3_sin, fourier_p12_k3_cos
```
