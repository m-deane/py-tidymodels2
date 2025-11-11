# Analysis: forecasting_grouped.ipynb

## Purpose
This notebook demonstrates all 23+ models from py-tidymodels on **grouped/panel time series data** with multiple countries. The data has a "country" column that creates natural groups for per-country modeling.

## Data Structure

### Source Data
- **File**: `__data/refinery_margins.csv`
- **Group Column**: `country` (10 countries: Algeria, Denmark, Germany, Italy, Netherlands, Norway, Romania, Russian Federation, Turkey, United Kingdom)
- **Date Column**: `date` (monthly data from 2006-01-01 onwards)
- **Target**: `refinery_kbd` (refinery capacity in thousands of barrels per day)
- **Features**: 18+ columns including:
  - Oil prices: `brent`, `dubai`, `wti`
  - Refinery margins: `brent_cracking_nw_europe`, `brent_hydroskimming_nw_europe`, etc.
  - Regional variations: NW Europe, Mediterranean, Singapore, US Gulf Coast, US Mid-Continent

### Data Shape
- **Panel structure**: Multiple countries over time
- **Each row**: One country-date observation
- **Total observations**: ~2000+ rows (10 countries × ~200 months)

## Current Notebook Organization

### Section 1: Data Import & Splitting (Cells 1-13)
- **Cell 2**: Import data
- **Cell 3**: Load raw data from CSV
- **Cell 7**: Standard split using `initial_split()` → `train_data`, `test_data`
- **Cell 10**: Time-based split using `initial_time_split()` → `train_mix`, `test_mix`
- **Cell 13**: Define formula: `FORMULA_STR = "refinery_kbd ~ ."`

**Current Issue**: Data contains all countries mixed together. No per-country splitting.

### Section 2: Single Model Fitting (Cells 14-82)
**Pattern**: Each model follows this structure:
```python
spec_<model> = <model_function>(<params>)
fit_<model> = spec_<model>.fit(train_data, FORMULA_STR)
fit_<model> = fit_<model>.evaluate(test_data)
outputs, coefs, stats = fit_<model>.extract_outputs()
fig = plot_forecast(fit_<model>, title="...")
fig.show()
```

**Models Included** (39 total specifications):
1. Prophet (cell 15)
2. ARIMA (cell 17)
3. OLS - Statsmodels (cell 19)
4. OLS - sklearn (cell 22)
5. ElasticNet (cell 23)
6. Ridge Regression (cell 24)
7. Lasso Regression (cell 25)
8. GAM - Generalized Additive Models (cell 26)
9. Prophet (duplicate, cell 27)
10. Exponential Smoothing (cell 28)
11. ARIMA Boost (cell 29)
12. Prophet Boost (cell 30)
13. Boost Tree - XGBoost (cell 31)
14. Null Model - Mean (cell 32)
15. Naive Model - Naive (cell 33)
16. Naive Model - Seasonal Naive (cell 34)
17. Naive Model - Drift (cell 35)
18. Null Model - Median (cell 36)
19. Hybrid Model (cell 39)
20. Manual Regression (cell 41)
21. Boost Tree - XGBoost (cell 44, different params)
22. Decision Tree (cell 44)
23. Random Forest (cell 46)
24. XGBoost (cell 48)
25. CatBoost (cell 50)
26. SVM RBF (cell 52)
27. SVM Linear (cell 54)
28. k-NN (k=5) (cell 56)
29. k-NN (k=10) (cell 57)
30. MLP Neural Network (cell 59)
31. Poisson Regression (cell 61)
32. MARS (commented out, cell 63)
33. STL Seasonal Decomposition (cell 65)
34. ARIMA Non-seasonal (cell 67)
35. Auto ARIMA (commented out, cell 69)
36. ETS - Additive (cell 71)
37. ETS - Multiplicative (cell 73)
38. Prophet - High Flexibility (cell 75)
39. Prophet - Low Flexibility (cell 77)
40. Recursive Forecasting (cell 79)

**Current Issue**: All models fit on MIXED data (all countries together). Should fit per-country models.

### Section 3: Workflows (Cells 83-90)
- **Cell 84**: Formula-based workflow
- **Cell 85-86**: Recipe-based workflow (commented out)
- **Cell 87**: Method chaining example
- **Cell 89**: Recipe + workflow example

**Current Issue**: Workflows also use mixed data. Should demonstrate grouped workflows.

## What Needs to Change

### 1. Data Splitting Must Be Group-Aware
**Current code**:
```python
split = initial_split(df, prop=0.75, seed=123)
train_data = training(split)
test_data = testing(split)
```

**Required change**: Split per country to maintain group integrity
```python
# Option A: Split within each country
train_list = []
test_list = []
for country in df['country'].unique():
    country_data = df[df['country'] == country]
    split = initial_time_split(
        country_data, 
        date_column="date",
        prop=0.75
    )
    train_list.append(training(split))
    test_list.append(testing(split))

train_data = pd.concat(train_list, ignore_index=True)
test_data = pd.concat(test_list, ignore_index=True)

# Option B: Use global split but ensure per-country modeling later
```

### 2. Model Fitting Must Use fit_nested() or fit_global()
**Current code (39 instances)**:
```python
fit_prophet = spec_prophet.fit(train_data, FORMULA_STR)
fit_prophet = fit_prophet.evaluate(test_data)
```

**Required changes**:

**Option A - Nested (Per-Country Models)**:
```python
# For direct model.fit():
fit_prophet_nested = spec_prophet.fit_nested(train_data, FORMULA_STR, group_col='country')
fit_prophet_nested = fit_prophet_nested.evaluate(test_data, group_col='country')

# For workflows:
wf = workflow().add_formula(FORMULA_STR).add_model(spec_prophet)
fit_wf_nested = wf.fit_nested(train_data, group_col='country')
fit_wf_nested = fit_wf_nested.evaluate(test_data)  # group_col preserved
```

**Option B - Global (Single Model with Country as Feature)**:
```python
# For direct model.fit():
fit_prophet_global = spec_prophet.fit_global(train_data, FORMULA_STR, group_col='country')

# For workflows:
wf = workflow().add_formula(FORMULA_STR).add_model(spec_prophet)
fit_wf_global = wf.fit_global(train_data, group_col='country')
```

### 3. Extract Outputs Will Include Group Column
**Current code**:
```python
outputs_prophet, coefs_prophet, stats_prophet = fit_prophet.extract_outputs()
```

**After change** (works the same, but outputs now have 'group' column):
```python
outputs_prophet, coefs_prophet, stats_prophet = fit_prophet_nested.extract_outputs()

# outputs_prophet will have columns:
# [..., 'model', 'model_group_name', 'group']
# where 'group' = country name

# Filter by country:
algeria_outputs = outputs_prophet[outputs_prophet['group'] == 'Algeria']
```

### 4. Visualization Must Be Group-Aware
**Current code**:
```python
fig = plot_forecast(fit_prophet, title="Sales Forecast")
fig.show()
```

**Required change**: Either plot all groups or filter by group
```python
# Option A: Plot all countries (may be crowded)
fig = plot_forecast(fit_prophet_nested, title="Refinery Forecast - All Countries")
fig.show()

# Option B: Plot specific country
algeria_fit = fit_prophet_nested.filter_group('Algeria')  # If this method exists
fig = plot_forecast(algeria_fit, title="Refinery Forecast - Algeria")
fig.show()

# Option C: Manual filtering
outputs_algeria = outputs_prophet[outputs_prophet['group'] == 'Algeria']
# Then plot outputs_algeria separately
```

## Specific Code Chunks Needing Updates

### High Priority (Core Modeling)
1. **Cell 7, 10**: Data splitting → Add per-country split logic
2. **Cell 13**: Formula → Consider adding country-specific features or excluding country from predictors
3. **Cells 15, 17, 19, 22-39, 44-82**: All 39 model fitting chunks → Add `.fit_nested()` or `.fit_global()`
4. **Cells 84, 89**: Workflow examples → Add `.fit_nested()` or `.fit_global()`

### Medium Priority (Visualization)
5. All `plot_forecast()` and `plot_residuals()` calls → Add group filtering or multi-group plotting

### Low Priority (Output Inspection)
6. All `display(outputs)` statements → Consider groupby operations to show per-country metrics

## Model Categories Represented

### Time Series Specific (11 models)
- Prophet (4 variations)
- ARIMA (3 variations)
- Exponential Smoothing (3 variations)
- STL Decomposition (1)

### Hybrid Time Series (3 models)
- ARIMA Boost
- Prophet Boost
- Hybrid Model (generic)

### Machine Learning - Tree-Based (5 models)
- Decision Tree
- Random Forest
- XGBoost
- CatBoost
- Boost Tree

### Machine Learning - Linear (7 models)
- OLS (2 engines)
- ElasticNet
- Ridge
- Lasso
- GAM
- Poisson Regression

### Machine Learning - Other (6 models)
- SVM RBF
- SVM Linear
- k-NN (2 variations)
- MLP Neural Network
- MARS (commented out)

### Baseline Models (5 models)
- Null Model (2 variations)
- Naive Model (3 variations)

### Special Models (2 models)
- Manual Regression
- Recursive Forecasting

## Recommended Update Strategy

### Phase 1: Add Grouped Modeling Examples
1. Keep first 3 models (Prophet, ARIMA, sklearn) as **GLOBAL** examples
   - Show how single model learns from all countries
   - Demonstrate `fit_global()` approach
   
2. Convert next 3 models to **NESTED** examples
   - Show per-country modeling
   - Demonstrate `fit_nested()` approach
   - Show group filtering in outputs

3. Add explanatory markdown before each approach:
   ```markdown
   ## Global Approach: Single Model Across All Countries
   This approach fits ONE model using data from all countries. 
   The country variable can be included as a feature.
   ```
   
   ```markdown
   ## Nested Approach: Separate Model Per Country
   This approach fits 10 independent models (one per country).
   Useful when country patterns differ significantly.
   ```

### Phase 2: Update Remaining Models
4. Update all remaining 33+ models to use nested or global approach
5. Consider organizing by modeling strategy:
   - Section 1.1: Global Models (5-10 examples)
   - Section 1.2: Nested Models (remaining 30+ examples)

### Phase 3: Update Workflows
6. Add workflow examples for both approaches:
   - Global workflow with country as feature
   - Nested workflow with per-country models

### Phase 4: Enhanced Visualization
7. Add group-aware visualization examples:
   - Multi-country overlay plots
   - Faceted plots (one panel per country)
   - Country comparison plots

## Formula Considerations

### Current Formula
```python
FORMULA_STR = "refinery_kbd ~ ."
```

This will include ALL columns as predictors, including "country".

### For Nested Approach
```python
# Country is already split, exclude it from formula
FORMULA_STR = "refinery_kbd ~ . -country"
```

### For Global Approach
```python
# Keep country as predictor (or let fit_global handle it)
FORMULA_STR = "refinery_kbd ~ ."
# OR explicitly include it
FORMULA_STR = "refinery_kbd ~ brent + wti + ... + country"
```

## Success Criteria

After updates, notebook should demonstrate:
1. ✅ Per-country data splitting
2. ✅ Both global and nested modeling approaches
3. ✅ Group column in outputs DataFrames
4. ✅ Group-aware visualizations
5. ✅ Clear explanation of when to use each approach
6. ✅ All 39+ models working with grouped data

## Notes

- **Recursive forecasting (cell 79)** uses `train_indexed` which needs special handling
- **Auto ARIMA and MARS** are commented out (compatibility issues)
- **Workflows section** is partially commented out - needs completion
- Some models have duplicate entries (e.g., Prophet appears multiple times with different parameters)
