# Complete Recipe Steps Reference

**py-tidymodels Recipe Step Library - Comprehensive Documentation**

This reference covers all 71+ recipe steps with complete parameter signatures and usage examples.

---

## Table of Contents

1. [Time Series Steps](#time-series-steps)
2. [Financial Steps](#financial-steps)
3. [Imputation Steps](#imputation-steps)
4. [Normalization & Scaling](#normalization--scaling)
5. [Transformations](#transformations)
6. [Categorical Encoding](#categorical-encoding)
7. [Feature Selection](#feature-selection)
8. [Supervised Filters](#supervised-filters)
9. [Adaptive Transformations](#adaptive-transformations)
10. [Data Quality Filters](#data-quality-filters)
11. [Dimensionality Reduction](#dimensionality-reduction)
12. [Basis Functions](#basis-functions)
13. [Discretization](#discretization)
14. [Interactions & Ratios](#interactions--ratios)
15. [Row Operations](#row-operations)
16. [Column Selectors](#column-selectors)

---

## Time Series Steps

### Basic Time Series Features

#### `step_lag(columns=None, lags=[1])`
Create lagged features for autoregressive modeling.

**Parameters:**
- `columns` (list or None): Columns to lag (None = all numeric)
- `lags` (list): List of lag periods (e.g., [1, 2, 7, 30])

**Example:**
```python
# Multiple lags for autoregressive features
.step_lag(["sales", "temperature"], lags=[1, 2, 3, 7])

# Weekly and monthly lags
.step_lag(["target"], lags=[7, 14, 28, 30])
```

**Creates:** Original column with `_lag_N` suffix

---

#### `step_diff(columns=None, lag=1, differences=1)`
Difference features for stationarity.

**Parameters:**
- `columns` (list or None): Columns to difference (None = all numeric)
- `lag` (int): Period for differencing (default: 1)
- `differences` (int): Number of differencing operations (default: 1)

**Example:**
```python
# First difference for trend removal
.step_diff(["sales"], lag=1, differences=1)

# Seasonal differencing
.step_diff(["sales"], lag=12, differences=1)

# Second-order differencing
.step_diff(["sales"], lag=1, differences=2)
```

**Creates:** Original column with `_diff` suffix

---

#### `step_pct_change(columns=None, periods=1)`
Percentage change features.

**Parameters:**
- `columns` (list or None): Columns to transform (None = all numeric)
- `periods` (int): Periods for change calculation (default: 1)

**Example:**
```python
# Daily percent change
.step_pct_change(["price"], periods=1)

# Week-over-week percent change
.step_pct_change(["sales"], periods=7)
```

**Creates:** Original column with `_pct_change` suffix

---

#### `step_rolling(columns=None, window=1, stats=["mean"])`
Rolling window statistics.

**Parameters:**
- `columns` (list or None): Columns to apply rolling windows (None = all numeric)
- `window` (int): Size of rolling window
- `stats` (list): Statistics to compute - options: "mean", "std", "min", "max", "sum"

**Example:**
```python
# 7-day moving average and std
.step_rolling(["sales"], window=7, stats=["mean", "std"])

# Multiple windows for different horizons
.step_rolling(["temperature"], window=3, stats=["mean", "min", "max"])
.step_rolling(["temperature"], window=7, stats=["mean"])
```

**Creates:** Original column with `_rolling_{window}_{stat}` suffix

---

#### `step_date(column, features=["year", "month", "day", "dayofweek"], keep_original_date=False)`
Extract datetime features from a date column.

**Parameters:**
- `column` (str): Datetime column to extract features from
- `features` (list): Features to extract
- `keep_original_date` (bool): Keep original date column (default: False)

**Available Features:**
- Basic: `year`, `month`, `day`, `hour`, `minute`, `second`
- Calendar: `dayofweek`, `dayofyear`, `quarter`, `week`
- Flags: `is_weekend`, `is_month_start`, `is_month_end`, `is_quarter_start`, `is_quarter_end`, `is_year_start`, `is_year_end`

**Example:**
```python
# Common calendar features
.step_date("date", features=["year", "month", "quarter", "dayofweek"])

# Weekend and month-end flags
.step_date("date", features=["is_weekend", "is_month_end"])

# Keep original date column
.step_date("date", features=["year", "month"], keep_original_date=True)
```

**Creates:** New columns with feature names

---

### Extended Time Series Features

#### `step_lead(columns, leads, prefix="lead_")`
Create lead (future) features.

**Parameters:**
- `columns` (list): Columns to create leads for
- `leads` (list): List of lead periods (e.g., [1, 2, 7])
- `prefix` (str): Prefix for created columns (default: "lead_")

**Example:**
```python
# Future values for target planning
.step_lead(["demand"], leads=[1, 2, 3, 7], prefix="future_")

# Multiple horizons
.step_lead(["price"], leads=[1, 7, 30])
```

**Creates:** `{prefix}{column}_lead_{N}`

---

#### `step_ewm(columns, span=10, statistics=["mean"], prefix="ewm_")`
Exponentially weighted moving features.

**Parameters:**
- `columns` (list): Columns to apply EWM
- `span` (int): Span for exponential weighting (default: 10)
- `statistics` (list): Statistics to compute - "mean", "std", "var"
- `prefix` (str): Prefix for created columns (default: "ewm_")

**Example:**
```python
# Exponentially weighted moving average
.step_ewm(["sales"], span=7, statistics=["mean"])

# EWM mean and std for volatility
.step_ewm(["price"], span=20, statistics=["mean", "std"])

# Short-term vs long-term EWM
.step_ewm(["target"], span=5, statistics=["mean"], prefix="short_")
.step_ewm(["target"], span=20, statistics=["mean"], prefix="long_")
```

**Creates:** `{prefix}{statistic}_{column}_span_{span}`

---

#### `step_expanding(columns, statistics=["mean"], prefix="expanding_", min_periods=1)`
Expanding window (cumulative) features.

**Parameters:**
- `columns` (list): Columns to apply expanding window
- `statistics` (list): Statistics to compute - "mean", "std", "sum", "min", "max"
- `prefix` (str): Prefix for created columns (default: "expanding_")
- `min_periods` (int): Minimum periods required (default: 1)

**Example:**
```python
# Cumulative average
.step_expanding(["revenue"], statistics=["mean"])

# Running total
.step_expanding(["orders"], statistics=["sum"])

# Cumulative min/max range
.step_expanding(["price"], statistics=["min", "max"], min_periods=5)
```

**Creates:** `{prefix}{statistic}_{column}`

---

#### `step_fourier(date_column, period, K=5, prefix="fourier_")`
Fourier features for seasonality patterns.

**Parameters:**
- `date_column` (str or list): Date or numeric time index column
- `period` (int/float): Seasonality period (e.g., 365 for yearly, 12 for monthly)
- `K` (int): Number of Fourier term pairs (default: 5)
- `prefix` (str): Prefix for created columns (default: "fourier_")

**Example:**
```python
# Annual seasonality (daily data)
.step_fourier("date", period=365, K=5)

# Monthly seasonality
.step_fourier("date", period=12, K=3)

# Multiple seasonal patterns
.step_fourier("date", period=7, K=2, prefix="weekly_")
.step_fourier("date", period=365, K=5, prefix="yearly_")
```

**Creates:** `{prefix}sin_{period}_{k}` and `{prefix}cos_{period}_{k}` pairs

---

#### `step_timeseries_signature(date_column, features=None, prefix="")`
Comprehensive time-based features from a single date column.

**Parameters:**
- `date_column` (str or list): Column containing dates
- `features` (list or None): List of features to extract (None = all)
- `prefix` (str): Prefix for created columns (default: "")

**Default Features:** year, month, day, hour, minute, second, quarter, day_of_week, day_of_year, week_of_year, is_weekend, is_month_start, is_month_end, is_quarter_start, is_quarter_end, is_year_start, is_year_end

**Example:**
```python
# All time features at once
.step_timeseries_signature("date")

# Selected features with prefix
.step_timeseries_signature("date",
    features=["month", "quarter", "is_weekend"],
    prefix="ts_")
```

**Creates:** Multiple feature columns based on selections

---

#### `step_holiday(date_column, country="US", holidays=None, prefix="holiday_")`
Add holiday indicator features.

**Parameters:**
- `date_column` (str or list): Column containing dates
- `country` (str): Country code (e.g., "US", "UK", "CA", "DE")
- `holidays` (list or None): List of specific holidays (None = all major holidays)
- `prefix` (str): Prefix for created columns (default: "holiday_")

**Example:**
```python
# US holidays
.step_holiday("date", country="US")

# UK holidays with custom prefix
.step_holiday("date", country="UK", prefix="uk_hol_")

# Specific holidays only
.step_holiday("date", country="US",
    holidays=["Christmas", "Thanksgiving"])
```

**Requires:** `pytimetk` package
**Creates:** Binary indicator columns for each holiday

---

## Financial Steps

#### `step_oscillators(date_col, close_col, group_col=None, rsi_periods=[14], macd=True, macd_fast=12, macd_slow=26, macd_signal=9, stochastic=False, stoch_period=14, stoch_smooth_k=3, high_col=None, low_col=None, handle_na="keep")`
Financial momentum indicators (RSI, MACD, Stochastic Oscillator).

**Parameters:**
- `date_col` (str): Name of date/time column
- `close_col` (str): Name of close price column
- `group_col` (str or None): Optional grouping column (e.g., ticker symbol)
- `rsi_periods` (list): RSI periods to calculate (default: [14])
- `macd` (bool): Calculate MACD indicators (default: True)
- `macd_fast/slow/signal` (int): MACD parameters (default: 12/26/9)
- `stochastic` (bool): Calculate Stochastic Oscillator (default: False)
- `stoch_period` (int): Stochastic period (default: 14)
- `stoch_smooth_k` (int): Stochastic smoothing (default: 3)
- `high_col/low_col` (str or None): Required for Stochastic
- `handle_na` (str): How to handle NaN - "keep", "drop", "fill_forward"

**Example:**
```python
# RSI only
.step_oscillators("date", "close", rsi_periods=[7, 14, 21])

# MACD and RSI
.step_oscillators("date", "close",
    rsi_periods=[14],
    macd=True,
    macd_fast=12, macd_slow=26, macd_signal=9)

# Full technical indicators including Stochastic
.step_oscillators("date", "close",
    group_col="ticker",
    rsi_periods=[14],
    macd=True,
    stochastic=True,
    high_col="high",
    low_col="low",
    handle_na="fill_forward")
```

**Requires:** `pytimetk` package
**Creates:** RSI, MACD, MACD Signal, MACD Histogram, Stochastic %K/%D columns

---

## Imputation Steps

#### `step_impute_mean(columns=None)`
Replace missing values with training mean.

**Parameters:**
- `columns` (list, str, or None): Columns to impute (None = all numeric with NA)

**Example:**
```python
# All numeric columns with missing values
.step_impute_mean()

# Specific columns
.step_impute_mean(["age", "income"])

# With selector
.step_impute_mean(all_numeric_predictors())
```

---

#### `step_impute_median(columns=None)`
Replace missing values with training median (robust to outliers).

**Parameters:**
- `columns` (list, str, or None): Columns to impute (None = all numeric with NA)

**Example:**
```python
# All numeric columns (preferred over mean for skewed data)
.step_impute_median()

# Specific columns
.step_impute_median(["price", "quantity"])
```

---

#### `step_impute_mode(columns=None)`
Replace missing values with most frequent value.

**Parameters:**
- `columns` (list, str, or None): Columns to impute (None = all columns with NA)

**Example:**
```python
# Works for both numeric and categorical
.step_impute_mode(["category", "region"])

# All columns with missing values
.step_impute_mode()
```

---

#### `step_impute_knn(columns=None, neighbors=5, weights="uniform")`
K-Nearest Neighbors imputation based on similar observations.

**Parameters:**
- `columns` (list, str, or None): Columns to impute (None = all numeric with NA)
- `neighbors` (int): Number of neighbors to use (default: 5)
- `weights` (str): Weight function - "uniform" or "distance"

**Example:**
```python
# Basic KNN imputation
.step_impute_knn(["temperature", "humidity"], neighbors=5)

# Distance-weighted neighbors
.step_impute_knn(all_numeric_predictors(),
    neighbors=10,
    weights="distance")
```

**Note:** More computationally expensive than mean/median

---

#### `step_impute_linear(columns=None, limit=None, limit_direction="both")`
Linear interpolation for time series data.

**Parameters:**
- `columns` (list, str, or None): Columns to impute (None = all numeric with NA)
- `limit` (int or None): Maximum consecutive NAs to fill (None = no limit)
- `limit_direction` (str): Direction - "forward", "backward", "both"

**Example:**
```python
# Basic interpolation
.step_impute_linear(["sensor_reading"])

# Limit consecutive fills
.step_impute_linear(["temperature"], limit=3, limit_direction="forward")

# All numeric columns
.step_impute_linear()
```

**Best for:** Time series with missing values in sequences

---

#### `step_impute_bag(columns=None, impute_with=None, trees=25, seed_val=None)`
Bagged tree models for sophisticated imputation.

**Parameters:**
- `columns` (list, str, or None): Columns to impute (None = all columns with NA)
- `impute_with` (list or None): Predictor columns (None = all other columns)
- `trees` (int): Number of bagged trees (default: 25)
- `seed_val` (int or None): Random seed for reproducibility

**Example:**
```python
# Let algorithm choose predictors
.step_impute_bag(["income", "age"], trees=50)

# Specify predictor columns
.step_impute_bag(["salary"],
    impute_with=["age", "education", "experience"],
    trees=100,
    seed_val=42)
```

**Note:** Most sophisticated but slowest imputation method

---

#### `step_impute_roll(columns=None, window=5, statistic=None)`
Rolling window statistic imputation for time series.

**Parameters:**
- `columns` (list, str, or None): Columns to impute (None = all numeric with NA)
- `window` (int): Size of rolling window (must be odd integer >= 3, default: 5)
- `statistic` (callable or None): Function to compute (default: np.nanmedian)

**Example:**
```python
# Default median rolling imputation
.step_impute_roll(["sensor_data"], window=7)

# Custom statistic
.step_impute_roll(["temperature"],
    window=5,
    statistic=np.nanmean)
```

**Best for:** Time series with sporadic missing values

---

## Normalization & Scaling

#### `step_normalize(columns=None, method="zscore")`
Normalize using sklearn scalers.

**Parameters:**
- `columns` (list, str, or None): Columns to normalize (None = all numeric)
- `method` (str): "zscore" (StandardScaler) or "minmax" (MinMaxScaler)

**Example:**
```python
# Z-score normalization (mean=0, std=1)
.step_normalize(all_numeric_predictors())

# Min-max scaling (0 to 1)
.step_normalize(all_numeric_predictors(), method="minmax")

# Specific columns
.step_normalize(["age", "income", "score"])
```

**Note:** Automatically excludes datetime columns

---

#### `step_center(columns=None)`
Center to have mean of zero (subtracts mean).

**Parameters:**
- `columns` (list, str, or None): Columns to center (None = all numeric)

**Example:**
```python
# Center all numeric predictors
.step_center(all_numeric_predictors())

# Center specific columns
.step_center(["x1", "x2", "x3"])
```

**Note:** Does not scale variance

---

#### `step_scale(columns=None)`
Scale to have standard deviation of one (divides by std).

**Parameters:**
- `columns` (list, str, or None): Columns to scale (None = all numeric)

**Example:**
```python
# Scale all numeric predictors
.step_scale(all_numeric_predictors())

# Often combined with step_center for full normalization
.step_center(all_numeric_predictors())
.step_scale(all_numeric_predictors())
```

**Note:** Does not center mean

---

#### `step_range(columns=None, min_val=0.0, max_val=1.0)`
Scale to custom range.

**Parameters:**
- `columns` (list, str, or None): Columns to scale (None = all numeric)
- `min_val` (float): Target minimum (default: 0)
- `max_val` (float): Target maximum (default: 1)

**Example:**
```python
# Scale to [0, 1]
.step_range(all_numeric_predictors())

# Scale to [-1, 1]
.step_range(all_numeric_predictors(), min_val=-1, max_val=1)

# Scale to custom range
.step_range(["probability"], min_val=0, max_val=100)
```

---

## Transformations

#### `step_log(columns=None, base=np.e, offset=0.0, signed=False, inplace=True)`
Logarithmic transformation for skewed data.

**Parameters:**
- `columns` (list, str, or None): Columns to transform (None = all numeric)
- `base` (float): Logarithm base (default: np.e for natural log)
- `offset` (float): Value added before transformation (default: 0)
- `signed` (bool): If True, preserves sign for negative values (default: False)
- `inplace` (bool): Replace original columns (True) or create new (False)

**Example:**
```python
# Natural log with offset for zeros
.step_log(["price", "quantity"], offset=1, inplace=True)

# Log base 10
.step_log(["sales"], base=10, offset=1)

# Signed log for data with negatives
.step_log(["returns"], signed=True, inplace=True)

# Create new columns (keeps originals)
.step_log(["income"], inplace=False)  # Creates income_log
```

---

#### `step_sqrt(columns=None, inplace=True)`
Square root transformation.

**Parameters:**
- `columns` (list, str, or None): Columns to transform (None = all numeric)
- `inplace` (bool): Replace original columns (True) or create new (False)

**Example:**
```python
# Transform in place
.step_sqrt(["count", "total"], inplace=True)

# Create new columns
.step_sqrt(["area"], inplace=False)  # Creates area_sqrt
```

**Note:** Only works on non-negative values

---

#### `step_inverse(columns=None, offset=0.0, inplace=True)`
Inverse transformation (1/x).

**Parameters:**
- `columns` (list, str, or None): Columns to transform (None = all numeric)
- `offset` (float): Value added before inversion to avoid division by zero
- `inplace` (bool): Replace original columns (True) or create new (False)

**Example:**
```python
# Basic inverse
.step_inverse(["rate"], offset=0.001, inplace=True)

# Create new columns
.step_inverse(["frequency"], inplace=False)  # Creates frequency_inv
```

---

#### `step_boxcox(columns=None, lambdas=None, limits=(-5, 5), inplace=True)`
Box-Cox power transformation for positive values.

**Parameters:**
- `columns` (list, str, or None): Columns to transform (None = all numeric)
- `lambdas` (dict or None): Column → lambda mapping (None = estimate from data)
- `limits` (tuple): (lower, upper) limits for lambda search (default: (-5, 5))
- `inplace` (bool): Replace original columns (True) or create new (False)

**Example:**
```python
# Auto-estimate lambda
.step_boxcox(all_numeric_predictors(), inplace=True)

# Specify lambdas
.step_boxcox(["sales"], lambdas={"sales": 0.5})

# Custom lambda search range
.step_boxcox(["price"], limits=(-2, 2))
```

**Requirement:** All values must be positive

---

#### `step_yeojohnson(columns=None, lambdas=None, limits=(-5, 5), inplace=True)`
Yeo-Johnson transformation (handles negative values).

**Parameters:**
- `columns` (list, str, or None): Columns to transform (None = all numeric)
- `lambdas` (dict or None): Column → lambda mapping (None = estimate from data)
- `limits` (tuple): (lower, upper) limits for lambda search (default: (-5, 5))
- `inplace` (bool): Replace original columns (True) or create new (False)

**Example:**
```python
# Preferred over Box-Cox when data has negatives
.step_yeojohnson(all_numeric_predictors(), inplace=True)

# Specify lambdas
.step_yeojohnson(["profit"], lambdas={"profit": 1.5})

# Create new columns
.step_yeojohnson(["returns"], inplace=False)
```

**Advantage:** Works with negative values (unlike Box-Cox)

---

## Categorical Encoding

#### `step_dummy(columns, one_hot=True)`
Create dummy variables from categorical columns.

**Parameters:**
- `columns` (list, str, or callable): Categorical columns to encode
- `one_hot` (bool): Use one-hot encoding (True) or integer encoding (False)

**Example:**
```python
# One-hot encoding (creates k-1 columns)
.step_dummy(["region", "category"])

# Integer encoding
.step_dummy(["status"], one_hot=False)

# With selector
.step_dummy(all_nominal_predictors())
```

**Note:** Automatically excludes datetime columns, handles novel levels in forge

---

#### `step_other(columns=None, threshold=0.05, other_label="other")`
Pool infrequent categorical levels into "other" category.

**Parameters:**
- `columns` (list, str, or None): Columns to process (None = all categorical)
- `threshold` (float): Minimum frequency to keep level (default: 0.05)
- `other_label` (str): Label for pooled category (default: "other")

**Example:**
```python
# Pool levels below 5% frequency
.step_other(["city", "occupation"], threshold=0.05)

# More aggressive pooling (10% threshold)
.step_other(["product_type"], threshold=0.10, other_label="rare")

# All categorical columns
.step_other()
```

**Use case:** Reduce dimensionality from high-cardinality categoricals

---

#### `step_novel(columns=None, novel_label="new")`
Handle novel categorical levels in new data.

**Parameters:**
- `columns` (list, str, or None): Columns to process (None = all categorical)
- `novel_label` (str): Label assigned to novel levels (default: "new")

**Example:**
```python
# Assign novel levels to "new"
.step_novel(["category", "region"])

# Custom label
.step_novel(["product"], novel_label="unknown")
```

**Use case:** Prevent errors when test data has unseen categories

---

#### `step_unknown(columns=None, unknown_label="_unknown_")`
Assign missing categorical values to specific level.

**Parameters:**
- `columns` (list, str, or None): Columns to process (None = all categorical)
- `unknown_label` (str): Label for missing values (default: "_unknown_")

**Example:**
```python
# Replace NA with "_unknown_"
.step_unknown(["category", "status"])

# Custom label
.step_unknown(["product_type"], unknown_label="missing")
```

**Difference from impute_mode:** Explicitly treats NA as a category level

---

#### `step_indicate_na(columns=None, prefix="na_ind")`
Create binary indicator columns for missing values.

**Parameters:**
- `columns` (list, str, or None): Columns to check (None = all with NA)
- `prefix` (str): Prefix for indicator columns (default: "na_ind")

**Example:**
```python
# Create indicators before imputing
.step_indicate_na(["income", "age"])
.step_impute_median(["income", "age"])

# Custom prefix
.step_indicate_na(all_numeric_predictors(), prefix="missing_")
```

**Use case:** Preserve "missingness" as a signal before imputation

---

#### `step_integer(columns=None, zero_based=True)`
Integer encode categorical variables.

**Parameters:**
- `columns` (list, str, or None): Categorical columns (None = all categorical)
- `zero_based` (bool): Use zero-based indexing (default: True)

**Example:**
```python
# Zero-based encoding (0, 1, 2, ...)
.step_integer(["priority", "status"])

# One-based encoding (1, 2, 3, ...)
.step_integer(["grade"], zero_based=False)
```

**Use case:** Tree-based models, ordinal relationships

---

## Feature Selection

#### `step_pca(columns=None, num_comp=None, threshold=None)`
Principal Component Analysis for dimensionality reduction.

**Parameters:**
- `columns` (list, str, or None): Columns to transform (None = all numeric)
- `num_comp` (int or None): Number of components to keep
- `threshold` (float or None): Variance threshold (alternative to num_comp)

**Example:**
```python
# Keep 5 components
.step_pca(all_numeric_predictors(), num_comp=5)

# Keep components explaining 95% variance
.step_pca(all_numeric_predictors(), threshold=0.95)

# Specific columns
.step_pca(["x1", "x2", "x3", "x4"], num_comp=2)
```

**Creates:** PC1, PC2, ... columns

---

#### `step_select_corr(outcome, threshold=0.9, method="multicollinearity", corr_method="pearson")`
Correlation-based feature selection.

**Parameters:**
- `outcome` (str): Outcome column name (required)
- `threshold` (float): Correlation threshold (default: 0.9)
- `method` (str): "multicollinearity" (remove inter-feature correlation) or "outcome" (keep features correlated with outcome)
- `corr_method` (str): "pearson", "spearman", or "kendall"

**Example:**
```python
# Remove multicollinear features
.step_select_corr(outcome="target", threshold=0.9, method="multicollinearity")

# Keep features correlated with outcome
.step_select_corr(outcome="target", threshold=0.3, method="outcome")

# Spearman correlation for non-linear relationships
.step_select_corr(outcome="target", corr_method="spearman")
```

---

## Supervised Filters

#### `step_filter_anova(outcome, threshold=None, top_n=None, top_p=None, use_pvalue=True, columns=None)`
ANOVA F-test feature selection (fast, linear relationships).

**Parameters:**
- `outcome` (str): Outcome variable name (required)
- `threshold` (float or None): Minimum score to keep feature
- `top_n` (int or None): Keep top N features
- `top_p` (float or None): Keep top proportion (e.g., 0.5 for top 50%)
- `use_pvalue` (bool): Use -log10(p-value) as score (default: True)
- `columns` (list, str, or None): Columns to filter (None = all numeric)

**Example:**
```python
# Keep top 50% of features
.step_filter_anova(outcome="target", top_p=0.5)

# Keep top 10 features
.step_filter_anova(outcome="target", top_n=10)

# Threshold on F-statistic
.step_filter_anova(outcome="target", threshold=5.0, use_pvalue=False)
```

**Best for:** Quick linear feature selection

---

#### `step_filter_rf_importance(outcome, threshold=None, top_n=None, top_p=None, trees=100, mtry=None, min_n=2, random_state=None, columns=None)`
Random Forest feature importance filter (powerful, captures non-linearity).

**Parameters:**
- `outcome` (str): Outcome variable name (required)
- `threshold/top_n/top_p` (specify ONE): Selection mode
- `trees` (int): Number of trees (default: 100)
- `mtry` (int or None): Features per split (None = sqrt(n_features))
- `min_n` (int): Min samples per leaf (default: 2)
- `random_state` (int or None): Random seed
- `columns` (list, str, or None): Columns to filter

**Example:**
```python
# Keep top 10 most important features
.step_filter_rf_importance(outcome="target", top_n=10, trees=200)

# Keep top 30% of features
.step_filter_rf_importance(outcome="target", top_p=0.3, trees=100)

# Reproducible results
.step_filter_rf_importance(outcome="target", top_n=15, random_state=42)
```

**Best for:** Non-linear relationships, interactions

---

#### `step_filter_mutual_info(outcome, threshold=None, top_n=None, top_p=None, n_neighbors=3, random_state=None, columns=None)`
Mutual information filter (captures non-linear dependencies).

**Parameters:**
- `outcome` (str): Outcome variable name (required)
- `threshold/top_n/top_p` (specify ONE): Selection mode
- `n_neighbors` (int): Neighbors for MI estimation (default: 3)
- `random_state` (int or None): Random seed
- `columns` (list, str, or None): Columns to filter

**Example:**
```python
# Keep features with MI > 0.1
.step_filter_mutual_info(outcome="target", threshold=0.1)

# Keep top 20 features
.step_filter_mutual_info(outcome="target", top_n=20, n_neighbors=5)

# Keep top 40% of features
.step_filter_mutual_info(outcome="target", top_p=0.4)
```

**Best for:** Non-linear relationships, doesn't assume linearity

---

#### `step_filter_roc_auc(outcome, threshold=None, top_n=None, top_p=None, multiclass_strategy='ovr', columns=None)`
ROC AUC filter for classification tasks.

**Parameters:**
- `outcome` (str): Outcome variable name (required, must be categorical)
- `threshold/top_n/top_p` (specify ONE): Selection mode
- `multiclass_strategy` (str): "ovr" (one-vs-rest) or "ovo" (one-vs-one)
- `columns` (list, str, or None): Columns to filter

**Example:**
```python
# Keep features with AUC > 0.7
.step_filter_roc_auc(outcome="class", threshold=0.7)

# Keep top 15 discriminative features
.step_filter_roc_auc(outcome="class", top_n=15)

# Multiclass with one-vs-one strategy
.step_filter_roc_auc(outcome="species",
    top_n=10,
    multiclass_strategy='ovo')
```

**Use case:** Binary or multiclass classification

---

#### `step_filter_chisq(outcome, threshold=None, top_n=None, top_p=None, method='chisq', use_pvalue=True, columns=None)`
Chi-squared/Fisher exact test for categorical outcomes.

**Parameters:**
- `outcome` (str): Outcome variable name (required, categorical)
- `threshold/top_n/top_p` (specify ONE): Selection mode
- `method` (str): "chisq" or "fisher"
- `use_pvalue` (bool): Use -log10(p-value) as score (default: True)
- `columns` (list, str, or None): Columns to filter

**Example:**
```python
# Chi-squared test, keep top 50%
.step_filter_chisq(outcome="diagnosis", top_p=0.5)

# Fisher exact test for small samples
.step_filter_chisq(outcome="rare_event", method='fisher', top_n=10)

# Threshold on chi-square statistic
.step_filter_chisq(outcome="category",
    threshold=10.0,
    use_pvalue=False)
```

**Use case:** Categorical predictors and outcomes

---

## Adaptive Transformations

#### `step_splitwise(outcome, transformation_mode="univariate", min_support=0.1, min_improvement=3.0, criterion="AIC", exclude_vars=None, columns=None)`
Adaptive dummy encoding using data-driven threshold detection via shallow decision trees.

**Parameters:**
- `outcome` (str): Outcome variable name (**required** for supervised transformation)
- `transformation_mode` (str): "univariate" (each predictor independent) or "iterative" (not yet implemented)
- `min_support` (float): Minimum fraction in each dummy group (default: 0.1, range: 0-0.5)
- `min_improvement` (float): Minimum AIC/BIC improvement for dummy over linear (default: 3.0)
- `criterion` (str): Model selection criterion - "AIC" or "BIC" (default: "AIC")
- `exclude_vars` (list or None): Variables forced to stay linear (no transformation)
- `columns` (selector or None): Columns to consider (None = all numeric except outcome)

**Decision Process:**
For each numeric predictor, SplitWise automatically decides:
- **Linear**: Keep predictor unchanged (if relationship is linear)
- **Single-split dummy**: Create binary `x >= threshold` (if threshold effect detected)
- **Double-split dummy**: Create binary `lower < x < upper` (if U-shaped pattern detected)

Decision based on AIC/BIC comparison using shallow decision trees (max_depth=2).

**Example:**
```python
# Basic usage - automatic transformation
.step_splitwise(outcome="target", min_improvement=2.0)

# Conservative (fewer transformations)
.step_splitwise(
    outcome="sales",
    min_improvement=5.0,  # Higher threshold
    criterion="BIC"        # More conservative than AIC
)

# Keep specific variables linear
.step_splitwise(
    outcome="price",
    exclude_vars=["year", "month"],  # Domain knowledge: time is linear
    min_improvement=2.0
)

# More balanced splits required
.step_splitwise(
    outcome="revenue",
    min_support=0.15,      # At least 15% in each group
    min_improvement=2.0
)

# Inspect transformation decisions after prep
rec_prepped = recipe().step_splitwise(outcome="y").prep(train_data)
splitwise_step = rec_prepped.prepared_steps[0]
decisions = splitwise_step.get_decisions()

for var, info in decisions.items():
    print(f"{var}: {info['decision']}")  # "linear", "single_split", or "double_split"
```

**Column Naming:**
Transformed columns use sanitized names (patsy-compatible):
- `x_ge_0p5234` - Single split: x >= 0.5234
- `x_ge_m1p2345` - Single split: x >= -1.2345
- `x_between_m0p5_1p2` - Double split: -0.5 < x < 1.2

**Naming convention:** `-` → `m` (minus), `.` → `p` (point)

**Use cases:**
- Non-linear relationships in predictors
- Interpretable threshold effects (e.g., "sales increase when temp > 20°C")
- Data-driven transformation decisions
- Alternative to manual dummy encoding or splines

**Advantages:**
- Automatic threshold detection via trees
- AIC/BIC model selection prevents overfitting
- Interpretable binary splits
- Support constraint ensures balanced groups
- Works with any downstream model

**Comparison with alternatives:**
- vs. **Manual dummies**: Data-driven threshold selection, not arbitrary
- vs. **Splines**: More interpretable thresholds, fewer parameters
- vs. **Polynomials**: Clearer interpretation, robust to outliers

**Reference:** Kurbucz et al. (2025). SplitWise Regression: Stepwise Modeling with Adaptive Dummy Encoding. arXiv:2505.15423

**Note:** This is a supervised step - requires outcome during prep()

---

#### `step_safe(surrogate_model, outcome, penalty=3.0, pelt_model='l2', no_changepoint_strategy='median', keep_original_cols=False, top_n=None, grid_resolution=1000)`
Surrogate Assisted Feature Extraction (SAFE) for interpretable models using complex surrogate models.

**Parameters:**
- `surrogate_model` (**required**): Pre-fitted surrogate model (must have `predict()` or `predict_proba()`)
- `outcome` (str): Outcome variable name (**required** for supervised transformation)
- `penalty` (float): Changepoint penalty - higher = fewer changepoints (default: 3.0, range: 0.1-10.0)
- `pelt_model` (str): Cost function for Pelt algorithm - "l2", "l1", or "rbf" (default: "l2")
- `no_changepoint_strategy` (str): What to do when no changepoints detected - "median" (split at median) or "drop" (remove feature) (default: "median")
- `keep_original_cols` (bool): Keep original columns alongside SAFE features (default: False)
- `top_n` (int or None): Select only top N most important features (default: None = all)
- `grid_resolution` (int): Number of points for PDP grid (default: 1000, range: 100-5000)

**Algorithm:**
SAFE uses a complex surrogate model to guide feature transformation for simple interpretable models:

**For Numeric Variables:**
1. Create 1000-point grid from min to max
2. Compute partial dependence plot (PDP) using surrogate model
3. Apply Pelt algorithm for changepoint detection
4. Create intervals based on changepoints: `[-Inf, cp1)`, `[cp1, cp2)`, ..., `[cpN, Inf)`
5. One-hot encode intervals with p-1 scheme

**For Categorical Variables:**
1. Compute PDP for each category level
2. Apply hierarchical clustering (Ward linkage) on PDP values
3. Use KneeLocator for optimal cluster count
4. Merge similar categories based on model response

**Example:**
```python
from sklearn.ensemble import GradientBoostingRegressor

# Fit surrogate model (REQUIRED before step_safe)
surrogate = GradientBoostingRegressor(n_estimators=100, max_depth=3)
surrogate.fit(train_data.drop('target', axis=1), train_data['target'])

# Basic usage
.step_safe(
    surrogate_model=surrogate,
    outcome='target',
    penalty=3.0
)

# Conservative (fewer features)
.step_safe(
    surrogate_model=surrogate,
    outcome='sales',
    penalty=10.0,                    # Higher penalty = fewer changepoints
    no_changepoint_strategy='drop'   # Remove features with no changepoints
)

# Feature selection (top N)
.step_safe(
    surrogate_model=surrogate,
    outcome='revenue',
    penalty=2.0,                     # More features initially
    top_n=10                         # Select top 10 by importance
)

# Keep original features
.step_safe(
    surrogate_model=surrogate,
    outcome='target',
    keep_original_cols=True          # Keep x1, x2, x3 alongside SAFE features
)

# Inspect transformations after prep
rec_prepped = recipe().step_safe(
    surrogate_model=surrogate,
    outcome='y'
).prep(train_data)

safe_step = rec_prepped.prepared_steps[0]

# Get transformation details
transformations = safe_step.get_transformations()
for var, info in transformations.items():
    print(f"{var}: {info['type']}")
    if info['type'] == 'numeric':
        print(f"  Changepoints: {info['changepoints']}")
    else:
        print(f"  Merged levels: {info['merged_levels']}")

# Get feature importances
importances = safe_step.get_feature_importances()
print(importances.head())
```

**Column Naming:**
Transformed columns use patsy-compatible sanitized names:
- `x1_0p50_to_1p23` - Numeric interval: 0.50 ≤ x1 < 1.23
- `x2_m1p50_to_0p00` - Negative interval: -1.50 ≤ x2 < 0.00
- `x3_2p34_to_Inf` - Last interval: x3 ≥ 2.34
- `category_A_B` - Categorical: merged levels A and B

**Naming convention:** `-` → `m` (minus), `.` → `p` (point), `_to_` separates intervals

**Use cases:**
- Transfer knowledge from complex model to simple model
- Create interpretable features from black-box model responses
- Data-driven threshold detection without domain knowledge
- Extract useful patterns from overfit complex models
- Alternative to manual feature engineering

**Advantages:**
- Leverages any sklearn-compatible surrogate model
- Automatic threshold detection via changepoint analysis
- Handles both numeric and categorical variables
- Feature importance and top-N selection
- Interpretable interval/category transformations
- Works with any downstream model

**Comparison with alternatives:**
- vs. **step_splitwise()**: SAFE uses surrogate model responses (more general), SplitWise uses outcome directly (more direct)
- vs. **Manual engineering**: Data-driven transformation, no domain knowledge required
- vs. **Splines/Polynomials**: More interpretable intervals, fewer parameters
- vs. **Tree models**: Extracts knowledge to simple model, maintains interpretability

**When to use SAFE:**
- ✅ Want to transfer knowledge from complex model to simple model
- ✅ Need interpretable features from black-box model
- ✅ Have no domain knowledge for manual engineering
- ✅ Want data-driven threshold detection
- ✅ Complex model overfits but captures useful patterns

**When NOT to use SAFE:**
- ❌ Surrogate model is not fitted properly
- ❌ Very small datasets (< 50 observations)
- ❌ Simple linear relationships (no complex patterns to extract)
- ❌ Computational cost is prohibitive (large datasets)

**Performance considerations:**
- Speed: O(p × n × grid_resolution) for p variables, n observations
- 300 obs × 3 vars × 1000 grid: ~2-3 seconds
- Reduce `grid_resolution` for faster computation
- Use `top_n` to limit output features

**Dependencies:**
Requires: `ruptures`, `kneed`, `scipy`
```bash
pip install ruptures kneed
```

**Reference:** SAFE Library - https://github.com/ModelOriented/SAFE

**Note:** This is a supervised step - requires outcome during prep() and pre-fitted surrogate model

---

## Data Quality Filters

#### `step_zv(columns=None)`
Remove zero variance (constant) columns.

**Parameters:**
- `columns` (list, str, or None): Columns to check (None = all numeric)

**Example:**
```python
# Remove all constant columns
.step_zv()

# Check specific columns
.step_zv(["x1", "x2", "x3"])
```

**Use case:** Always run early in recipe

---

#### `step_nzv(columns=None, freq_cut=19.0, unique_cut=10.0)`
Remove near-zero variance columns.

**Parameters:**
- `columns` (list, str, or None): Columns to check (None = all numeric)
- `freq_cut` (float): Frequency ratio threshold (default: 19.0 = 95/5 ratio)
- `unique_cut` (float): Unique value percentage threshold (default: 10%)

**Example:**
```python
# Default thresholds
.step_nzv()

# More aggressive (catches more near-constant columns)
.step_nzv(freq_cut=10.0, unique_cut=5.0)

# More conservative
.step_nzv(freq_cut=50.0, unique_cut=20.0)
```

**Use case:** Remove nearly constant features that add little information

---

#### `step_lincomb(columns=None, threshold=1e-5)`
Remove linear combinations (redundant features).

**Parameters:**
- `columns` (list, str, or None): Columns to check (None = all numeric)
- `threshold` (float): Tolerance for linear dependency (default: 1e-5)

**Example:**
```python
# Remove linearly dependent features
.step_lincomb()

# More strict tolerance
.step_lincomb(threshold=1e-8)

# Specific columns
.step_lincomb(all_numeric_predictors())
```

**Use case:** Detect features that are exact linear combinations of others

---

#### `step_filter_missing(columns=None, threshold=0.5)`
Remove columns with high missing value proportion.

**Parameters:**
- `columns` (list, str, or None): Columns to check (None = all columns)
- `threshold` (float): Maximum proportion of missing values (default: 0.5)

**Example:**
```python
# Remove columns with >50% missing
.step_filter_missing()

# More aggressive (remove >30% missing)
.step_filter_missing(threshold=0.3)

# Check specific columns
.step_filter_missing(all_numeric_predictors(), threshold=0.4)
```

**Use case:** Run before imputation to remove unsalvageable columns

---

## Dimensionality Reduction

#### `step_ica(columns=None, num_comp=None, algorithm="parallel", max_iter=200)`
Independent Component Analysis.

**Parameters:**
- `columns` (list, str, or None): Columns to transform (None = all numeric)
- `num_comp` (int or None): Number of components
- `algorithm` (str): "parallel" or "deflation"
- `max_iter` (int): Maximum iterations (default: 200)

**Example:**
```python
# Basic ICA
.step_ica(all_numeric_predictors(), num_comp=5)

# Deflation algorithm
.step_ica(all_numeric_predictors(),
    num_comp=10,
    algorithm="deflation")
```

**Use case:** Blind source separation, non-Gaussian signals

---

#### `step_kpca(columns=None, num_comp=None, kernel="rbf", gamma=None)`
Kernel PCA for non-linear dimensionality reduction.

**Parameters:**
- `columns` (list, str, or None): Columns to transform (None = all numeric)
- `num_comp` (int or None): Number of components
- `kernel` (str): "linear", "poly", "rbf", "sigmoid"
- `gamma` (float or None): Kernel coefficient (None = auto)

**Example:**
```python
# RBF kernel
.step_kpca(all_numeric_predictors(), num_comp=5, kernel="rbf")

# Polynomial kernel
.step_kpca(all_numeric_predictors(),
    num_comp=3,
    kernel="poly")

# Custom gamma
.step_kpca(all_numeric_predictors(),
    num_comp=5,
    kernel="rbf",
    gamma=0.1)
```

**Use case:** Non-linear relationships, manifold learning

---

#### `step_pls(columns=None, outcome=None, num_comp=None)`
Partial Least Squares (supervised dimensionality reduction).

**Parameters:**
- `columns` (list, str, or None): Predictor columns (None = all numeric except outcome)
- `outcome` (str or None): Outcome column name (required)
- `num_comp` (int or None): Number of components

**Example:**
```python
# Basic PLS
.step_pls(all_numeric_predictors(), outcome="target", num_comp=5)

# Specific predictors
.step_pls(["x1", "x2", "x3", "x4"], outcome="y", num_comp=2)
```

**Use case:** High-dimensional regression, maximizes covariance with outcome

---

## Advanced Feature Selection

#### `step_vip(outcome, threshold=1.0, num_comp=2)`
Variable Importance in Projection via PLS.

**Parameters:**
- `outcome` (str): Outcome column name (required)
- `threshold` (float): VIP threshold (default: 1.0)
- `num_comp` (int): Number of PLS components (default: 2)

**Example:**
```python
# Keep features with VIP > 1.0
.step_vip(outcome="target", threshold=1.0, num_comp=5)

# More selective
.step_vip(outcome="sales", threshold=1.5, num_comp=3)
```

**Use case:** PLS-based feature importance

---

#### `step_boruta(outcome, max_iter=100, random_state=None, perc=100, alpha=0.05)`
Boruta all-relevant feature selection.

**Parameters:**
- `outcome` (str): Outcome column name (required)
- `max_iter` (int): Maximum iterations (default: 100)
- `random_state` (int or None): Random seed
- `perc` (int): Percentile for shadow importance (default: 100)
- `alpha` (float): P-value threshold (default: 0.05)

**Example:**
```python
# Default Boruta
.step_boruta(outcome="target", max_iter=100, random_state=42)

# More iterations for stability
.step_boruta(outcome="target", max_iter=200, alpha=0.01)
```

**Use case:** Find all relevant features, not just best subset

---

#### `step_rfe(outcome, n_features=None, step=1, estimator=None)`
Recursive Feature Elimination.

**Parameters:**
- `outcome` (str): Outcome column name (required)
- `n_features` (int or None): Number of features to select (None = select half)
- `step` (int): Features to remove per iteration (default: 1)
- `estimator` (object or None): Sklearn estimator (None = auto-select)

**Example:**
```python
# Select 10 features
.step_rfe(outcome="target", n_features=10)

# Remove 2 features per iteration
.step_rfe(outcome="target", n_features=15, step=2)

# Custom estimator
from sklearn.ensemble import RandomForestRegressor
.step_rfe(outcome="target",
    n_features=20,
    estimator=RandomForestRegressor(n_estimators=100))
```

**Use case:** Backward selection with model-based importance

---

## Basis Functions

#### `step_bs(column, degree=3, df=None, knots=None)`
B-spline basis functions.

**Parameters:**
- `column` (str): Column to create splines for
- `degree` (int): Spline degree (default: 3 for cubic)
- `df` (int or None): Degrees of freedom (number of basis functions)
- `knots` (int or None): Number of internal knots (alternative to df)

**Example:**
```python
# Cubic B-splines with 5 df
.step_bs("age", degree=3, df=5)

# Quadratic splines with 3 knots
.step_bs("temperature", degree=2, knots=3)
```

**Use case:** Non-linear relationships, smooth curves

---

#### `step_ns(column, df=None, knots=None)`
Natural spline basis functions (linear beyond boundaries).

**Parameters:**
- `column` (str): Column to create splines for
- `df` (int or None): Degrees of freedom
- `knots` (int or None): Number of internal knots (alternative to df)

**Example:**
```python
# Natural splines with 4 df
.step_ns("x", df=4)

# Natural splines with 2 knots
.step_ns("time", knots=2)
```

**Use case:** Better extrapolation behavior than B-splines

---

#### `step_poly(columns, degree=2, include_interactions=False, inplace=True)`
Polynomial features.

**Parameters:**
- `columns` (list, str, or callable): Column selector (supports selectors)
- `degree` (int): Maximum polynomial degree (default: 2)
- `include_interactions` (bool): Include cross terms (default: False)
- `inplace` (bool): Replace original columns or keep them (default: True)

**Example:**
```python
# Quadratic features (x, x^2)
.step_poly(["x1", "x2"], degree=2, include_interactions=False)

# Quadratic with interactions (x, x^2, x*y, y, y^2)
.step_poly(["x", "y"], degree=2, include_interactions=True)

# Selector support
.step_poly(all_numeric_predictors(), degree=3)

# Keep original columns
.step_poly(["feature"], degree=2, inplace=False)
```

**Note:** Automatically replaces spaces with underscores in feature names

---

#### `step_harmonic(column, frequency=1, period=1.0)`
Harmonic (Fourier) basis functions.

**Parameters:**
- `column` (str): Column to create harmonics for
- `frequency` (int): Number of harmonics/cycles (default: 1)
- `period` (float): Period of seasonality (default: 1.0)

**Example:**
```python
# Single harmonic
.step_harmonic("time", frequency=1, period=12)

# Multiple harmonics
.step_harmonic("day_of_year", frequency=3, period=365)
```

**Use case:** Periodic patterns, seasonality

---

## Discretization

#### `step_discretize(columns=None, num_breaks=4, method="quantile", labels=None)`
Discretize numeric columns into bins.

**Parameters:**
- `columns` (list, str, or None): Columns to discretize (None = all numeric)
- `num_breaks` (int): Number of bins (default: 4)
- `method` (str): "quantile" (equal-frequency) or "width" (equal-width)
- `labels` (list or None): Custom bin labels (None = auto-generate)

**Example:**
```python
# Quartiles
.step_discretize(["age", "income"], num_breaks=4, method="quantile")

# Equal-width bins
.step_discretize(["price"], num_breaks=5, method="width")

# Custom labels
.step_discretize(["score"],
    num_breaks=3,
    labels=["low", "medium", "high"])
```

**Note:** Automatically excludes datetime columns

---

#### `step_cut(columns, breaks, labels=None, include_lowest=True)`
Cut at specified thresholds.

**Parameters:**
- `columns` (list): Columns to cut
- `breaks` (dict): Column → list of breakpoints
- `labels` (dict or None): Column → list of labels
- `include_lowest` (bool): Include lowest value in first bin (default: True)

**Example:**
```python
# Custom breakpoints
.step_cut(
    columns=["age"],
    breaks={"age": [0, 18, 35, 50, 65, 100]},
    labels={"age": ["child", "young_adult", "adult", "middle_age", "senior"]}
)

# Multiple columns
.step_cut(
    columns=["temperature", "humidity"],
    breaks={
        "temperature": [0, 10, 20, 30],
        "humidity": [0, 30, 60, 100]
    }
)
```

---

#### `step_percentile(columns=None, num_breaks=100, as_integer=True)`
Convert to percentile ranks.

**Parameters:**
- `columns` (list, str, or None): Columns to transform (None = all numeric)
- `num_breaks` (int): Number of percentile bins (default: 100 for 0-100 scale)
- `as_integer` (bool): Return integer percentiles (default: True)

**Example:**
```python
# Convert to 0-100 percentiles
.step_percentile(["score", "rating"])

# Deciles (0-10 scale)
.step_percentile(["value"], num_breaks=10)

# Float percentiles
.step_percentile(["metric"], as_integer=False)
```

**Use case:** Normalize skewed distributions while preserving ranks

---

## Interactions & Ratios

#### `step_interact(interactions, separator="_x_")`
Create multiplicative interaction features.

**Parameters:**
- `interactions`: Can be:
  - List of column pairs: `[("x1", "x2"), ("x1", "x3")]`
  - List of columns: `["x1", "x2", "x3"]` (all pairwise)
  - Selector function: `all_numeric_predictors()` (all pairwise)
- `separator` (str): Separator for interaction names (default: "_x_")

**Example:**
```python
# Specific interactions
.step_interact([("temperature", "humidity"), ("price", "quantity")])

# All pairwise interactions from list
.step_interact(["x1", "x2", "x3"])  # Creates x1_x_x2, x1_x_x3, x2_x_x3

# All numeric predictor interactions
.step_interact(all_numeric_predictors())

# Custom separator
.step_interact([("a", "b")], separator="*")  # Creates a*b
```

**Creates:** New columns with multiplicative interactions

---

#### `step_ratio(ratios, offset=1e-10, separator="_per_")`
Create ratio features.

**Parameters:**
- `ratios` (list): List of (numerator, denominator) column pairs
- `offset` (float): Added to denominator to avoid division by zero (default: 1e-10)
- `separator` (str): Separator for ratio names (default: "_per_")

**Example:**
```python
# Single ratio
.step_ratio([("revenue", "cost")])  # Creates revenue_per_cost

# Multiple ratios
.step_ratio([
    ("sales", "inventory"),
    ("profit", "revenue"),
    ("clicks", "impressions")
])

# Custom separator and offset
.step_ratio([("numerator", "denominator")],
    offset=0.001,
    separator="_div_")
```

**Use case:** Financial ratios, rates, efficiency metrics

---

## Row Operations

#### `step_mutate(transformations)`
Create or modify columns using custom functions.

**Parameters:**
- `transformations` (dict): Column names → transformation functions

**Example:**
```python
import numpy as np

# Create new columns
.step_mutate({
    "log_value": lambda df: np.log(df["value"] + 1),
    "total": lambda df: df["quantity"] * df["price"],
    "is_weekend": lambda df: df["date"].dt.dayofweek >= 5
})

# Modify existing columns
.step_mutate({
    "age": lambda df: df["age"].clip(0, 100)
})
```

**Note:** Functions receive full DataFrame and return Series

---

#### `step_rm(columns)`
Remove columns from dataset.

**Parameters:**
- `columns` (str, list, or callable): Columns to remove

**Example:**
```python
# Remove specific columns
.step_rm(["temp_col", "unused"])

# Remove with selector
.step_rm(starts_with("temp_"))

# Single column
.step_rm("id")
```

---

#### `step_select(columns)`
Keep only specified columns (inverse of step_rm).

**Parameters:**
- `columns` (str, list, or callable): Columns to keep

**Example:**
```python
# Keep specific columns
.step_select(["x1", "x2", "target"])

# Keep with selector
.step_select(all_numeric())

# Keep single column
.step_select("important_feature")
```

---

#### `step_naomit(columns=None)`
Remove rows with missing values.

**Parameters:**
- `columns` (list, str, or None): Columns to check for NAs (None = all columns)

**Example:**
```python
# Remove rows with any NA
.step_naomit()

# Remove rows with NA in specific columns
.step_naomit(["target", "key_feature"])

# Common pattern after lag features
.step_lag(["sales"], lags=[1, 2, 3])
.step_naomit()  # Remove rows with NA from lagging
```

---

## Column Selectors

**Available Selectors:**

```python
# Role-based selectors
all_predictors()           # All features except outcome
all_outcomes()             # Outcome column(s)
all_numeric_predictors()   # Numeric predictors (excludes outcome, datetime)
all_nominal_predictors()   # Categorical predictors

# Type-based selectors
all_numeric()              # All numeric columns
all_nominal()              # All categorical columns
all_datetime()             # All datetime columns

# Role and type combination
has_role("predictor")      # Columns with specific role
has_type("numeric")        # Columns with specific type

# Pattern-based selectors
starts_with("prefix_")     # Columns starting with prefix
ends_with("_suffix")       # Columns ending with suffix
contains("substring")      # Columns containing substring
matches("regex_pattern")   # Columns matching regex

# Combining selectors
from py_recipes.selectors import selector_union, selector_intersection

selector_union(all_numeric(), starts_with("x_"))      # Numeric OR starts with "x_"
selector_intersection(all_predictors(), all_numeric())  # Numeric AND predictor
```

---

**Total Steps Documented:** 71+ (includes step_splitwise)
**Last Updated:** 2025-11-09
**Version:** py-tidymodels v1.0

**Recent Additions:**
- **step_splitwise()** - Adaptive dummy encoding via data-driven threshold detection (2025-11-09)
