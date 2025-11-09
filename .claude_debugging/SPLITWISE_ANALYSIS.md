# SplitWise Algorithm - Comprehensive Technical Analysis

**Citation:** Kurbucz, M. T., Tzivanakis, N., Aslam, N. S., & Sykulski, A. M. (2025). SplitWise Regression: Stepwise Modeling with Adaptive Dummy Encoding. arXiv preprint. https://arxiv.org/abs/2505.15423

**Package Version:** 1.0.2
**Source:** https://github.com/mtkurbucz/SplitWise
**License:** GPL-3

---

## Executive Summary

SplitWise is a hybrid stepwise regression approach that transforms numeric predictors into either:
1. **Single-split binary dummies** (e.g., `x >= threshold`)
2. **Double-split binary dummies** (e.g., `lower < x < upper`)
3. **Linear predictors** (unchanged)

The transformation decision is based on AIC/BIC improvement criteria, with two modes:
- **Univariate mode**: Each variable transformed independently
- **Iterative mode** (default): Variables transformed adaptively considering partial synergies with other predictors

---

## Algorithm Description

### High-Level Workflow

```
INPUT:
  - Data frame with numeric predictors X and response Y
  - Formula (e.g., mpg ~ .)

PROCESS:
  1. Decide transformation type for each variable (dummy vs linear)
     - Univariate: Independent decision per variable
     - Iterative: Stepwise with partial residual analysis
  2. Transform features according to decisions
  3. Fit full model on transformed features
  4. Apply stepwise selection (forward/backward/both)

OUTPUT:
  - Linear model with transformed variables
  - Transformation metadata (cutoff points, variable types)
```

---

## Core Algorithm Components

### 1. Univariate Transformation Mode

**File:** `R/decide_variable_type_univariate.R`

**Algorithm Steps:**

For each numeric predictor `x`:

1. **Fit shallow decision tree** on `Y ~ x`:
   - Use `rpart` with `maxdepth=2`
   - `minsplit = max(5, ceiling(min_support * n))`
   - Extract up to 2 split points from tree

2. **Evaluate 3 candidate models** via AIC/BIC:
   - **Linear model**: `Y ~ x` (continuous predictor)
   - **Single-split dummy**: `Y ~ dummy` where `dummy = 1 if x >= c1`
   - **Double-split dummy**: `Y ~ dummy` where `dummy = 1 if c1 < x < c2`

3. **Apply support constraint**:
   - For single-split: `mean(dummy) > min_support AND mean(dummy) < (1 - min_support)`
   - For double-split: Same constraint applies
   - Prevents creating tiny or overwhelming dummy groups

4. **Select best transformation**:
   - Compare AIC/BIC values
   - Only accept dummy if: `linear_AIC - dummy_AIC >= min_improvement`
   - Otherwise keep linear

5. **Handle exclusions**:
   - Variables in `exclude_vars` always remain linear

**Key Implementation Details:**

```r
# Single-split dummy encoding
dummy = as.numeric(x >= cutoff[1])

# Double-split dummy encoding (middle region)
dummy = as.numeric(x > cutoff[1] & x < cutoff[2])

# Support validation
support = mean(dummy)
valid = (support > min_support) && (support < (1 - min_support))
```

---

### 2. Iterative Transformation Mode (Default)

**File:** `R/decide_variable_type_iterative.R`

**Algorithm Steps:**

This is a **stepwise variable selection algorithm** that adaptively chooses each variable's best form.

**Initialization (Backward mode):**
1. For each variable, fit univariate model on partial residuals
2. Choose initial best form (linear vs dummy)
3. Add all variables to design matrix
4. Fit full model

**Main Iteration Loop:**

Repeat until no improvement found:

**Step A - Forward/Both Direction:**
- For each unused variable:
  - Fit `rpart` tree on **partial residuals** from current model
  - Extract splits and evaluate: linear, single-split, double-split
  - Track best improvement

**Step B - Backward/Both Direction:**
- For each variable in model:
  - Try **removing** the variable entirely
  - Try **switching** form (linear ↔ dummy, or change cutoffs)
  - Track best improvement

**Step C - Forward/Both Direction (form switching):**
- For each variable already in model:
  - Try switching to alternative form
  - Track best improvement

**Step D - Implement Best Change:**
- If improvement > `min_improvement`:
  - Update design matrix
  - Refit model
  - Update decisions dict
  - Continue loop

**Critical Feature - Partial Residuals:**

```r
# Partial residual calculation
get_partial_resid <- function(mod, x_col, var_name) {
  coefs <- coef(mod)
  if (var_name %in% names(coefs)) {
    # Variable in model: partial residual = residual + coefficient*x
    mod$residuals + coefs[[var_name]] * x_col
  } else {
    # Variable not in model: use full residuals
    mod$residuals
  }
}
```

This allows the tree to find splits **conditional on other variables already in the model**, capturing synergies.

---

### 3. Feature Transformation

**File:** `R/transform_features_univariate.R` and `R/transform_features_iterative.R`

**Univariate Transform Logic:**

```r
for each variable in decisions:
  if type == "linear":
    transformed[var_name] = X[var_name]

  else if type == "dummy":
    if length(cutoffs) == 1:
      # Single split: x >= c
      dummy = as.numeric(X[var_name] >= cutoffs[1])

    else if length(cutoffs) == 2:
      # Double split: c1 <= x <= c2 (range encoding)
      dummy = as.numeric(
        X[var_name] >= min(cutoffs) & X[var_name] <= max(cutoffs)
      )

    transformed[var_name + "_dummy"] = dummy
```

**Iterative Transform Logic:**

```r
for each variable in decisions:
  if type == "linear":
    transformed[var_name] = X[var_name]

  else if type == "dummy":
    # Always uses >= for cutoff (simpler than univariate)
    dummy = as.numeric(X[var_name] >= cutoff)
    transformed[var_name + "_dummy"] = dummy
```

**Note:** There's a subtle difference in dummy encoding:
- Univariate uses `>=` for lower bound, `<=` for upper bound (inclusive range)
- Iterative uses `>=` for single cutoff
- Double-split in iterative uses `c1 < x < c2` (exclusive bounds)

---

### 4. Stepwise Selection

**File:** `R/splitwise.R` (lines 147-182)

After transformation, applies standard `stats::step()`:

```r
if direction == "forward":
  initial_model = lm(Y ~ 1, data=transformed)  # Intercept only
else:
  initial_model = lm(Y ~ ., data=transformed)  # Full model

scope = {
  lower: Y ~ 1,
  upper: Y ~ all_transformed_vars
}

step_model = step(
  object = initial_model,
  scope = scope,
  direction = direction,  # "forward", "backward", "both"
  k = k,  # 2 for AIC, log(n) for BIC
  steps = 1000  # max iterations
)
```

---

## Key Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `formula` | Formula | Required | - | Response and predictors (e.g., `mpg ~ .`) |
| `data` | DataFrame | Required | - | Data containing variables in formula |
| `transformation_mode` | String | "iterative" | {"iterative", "univariate"} | How to decide transformations |
| `direction` | String | "backward" | {"forward", "backward", "both"} | Stepwise selection strategy |
| `min_support` | Float | 0.1 | (0, 0.5) | Minimum fraction of observations in either dummy group |
| `min_improvement` | Float | 3.0 | [0, ∞) | Minimum AIC/BIC improvement required for dummy encoding |
| `criterion` | String | "AIC" | {"AIC", "BIC"} | Model selection criterion |
| `exclude_vars` | List[str] | NULL | - | Variables to keep linear (no dummy transformation) |
| `verbose` | Boolean | FALSE | {TRUE, FALSE} | Print debug information |
| `steps` | Integer | 1000 | [1, ∞) | Maximum stepwise iterations |
| `k` | Float | 2.0 | [0, ∞) | Penalty for degrees of freedom (2=AIC, log(n)=BIC) |

**Parameter Relationships:**
- If `criterion="BIC"`, should typically set `k=log(n)` for consistency
- Larger `min_support` → fewer dummy transformations (more conservative)
- Larger `min_improvement` → requires stronger evidence for dummies
- `exclude_vars` useful for keeping interpretable variables linear

---

## Input/Output Specifications

### Inputs

**Required:**
- `formula`: R formula object (e.g., `mpg ~ cyl + disp + hp`)
- `data`: Pandas DataFrame with numeric columns

**Constraints:**
- All predictors must be numeric (no categorical/factor variables allowed)
- Predictors can have missing values (handled by `lm` fitting)
- Response must be numeric (continuous regression)

### Outputs

**Primary Output:** `splitwise_lm` S3 object inheriting from `lm`

**Structure:**
```r
splitwise_lm = {
  # Standard lm components
  coefficients: named vector,
  residuals: vector,
  fitted.values: vector,
  ...

  # SplitWise-specific metadata
  splitwise_info: {
    transformation_mode: "iterative" | "univariate",
    decisions: {
      var_name: {
        type: "linear" | "dummy",
        cutoff: numeric vector (length 1 or 2) or NULL,
        tree_model: rpart object or NULL
      },
      ...
    },
    final_data: DataFrame (transformed),
    call: matched call object
  }
}
```

**Decisions Dictionary:**

For each variable in final model:
- `type`: "linear" or "dummy"
- `cutoff`:
  - NULL if linear
  - Single value if single-split dummy (e.g., `[5.5]`)
  - Two values if double-split dummy (e.g., `[2.0, 8.0]`)
- `tree_model`: The fitted `rpart` object (only in univariate mode)

---

## Code Structure

### Main Functions

1. **`splitwise()`** - Main entry point
   - Validates arguments
   - Routes to univariate or iterative mode
   - Applies stepwise selection
   - Returns `splitwise_lm` object

2. **`decide_variable_type_univariate()`** - Univariate decision logic
   - Fits `rpart` tree per variable
   - Compares linear vs dummy forms
   - Returns decisions dict

3. **`decide_variable_type_iterative()`** - Iterative decision logic
   - Stepwise selection with partial residuals
   - Adaptive form switching
   - Returns decisions dict

4. **`transform_features_univariate()`** - Apply univariate transformations
   - Creates dummy columns based on cutoffs
   - Returns transformed DataFrame

5. **`transform_features_iterative()`** - Apply iterative transformations
   - Creates dummy columns based on cutoffs
   - Returns transformed DataFrame

6. **S3 Methods:**
   - `print.splitwise_lm()` - Custom print showing dummy rules
   - `summary.splitwise_lm()` - Extended summary with transformations
   - `predict.splitwise_lm()` - Prediction with automatic transformation
   - `coef.splitwise_lm()` - Extract coefficients
   - `fitted.splitwise_lm()` - Extract fitted values
   - `residuals.splitwise_lm()` - Extract residuals

---

## Critical Implementation Details

### 1. Decision Tree Configuration

```r
rpart.control(
  maxdepth = 2,           # Shallow trees only (max 2 splits)
  minsplit = max(5, ceiling(min_support * n))  # Minimum obs per split
)
```

**Why maxdepth=2?**
- Allows at most 2 split points
- Prevents overfitting
- Captures simple non-linearities

**Why minsplit constraint?**
- Ensures splits respect `min_support` parameter
- Prevents pathological splits on outliers

### 2. Support Validation

```r
support = mean(dummy_column)
valid = (support > min_support) && (support < (1 - min_support))
```

**Purpose:**
- Prevents dummy variables with extreme imbalance
- E.g., with `min_support=0.1`, dummy must have 10%-90% split
- Ensures sufficient observations in both groups for stable estimation

### 3. Dummy Encoding Semantics

**Single-Split Dummy:**
```r
# Threshold encoding: high values → 1
dummy = as.numeric(x >= threshold)
```

**Double-Split Dummy:**
```r
# UNIVARIATE: Inclusive range (values in [c1, c2] → 1)
dummy = as.numeric(x >= c1 & x <= c2)

# ITERATIVE: Exclusive middle (values in (c1, c2) → 1)
dummy = as.numeric(x > c1 & x < c2)
```

**Interpretation:**
- Single-split captures "high vs low" effect
- Double-split captures "middle vs extremes" effect
- Useful when relationship is non-monotonic

### 4. Partial Residual Analysis (Iterative Mode)

**Key Insight:** When deciding how to transform variable `x_i`:
1. Remove `x_i` from current model
2. Compute residuals from model without `x_i`
3. Fit tree on `residuals ~ x_i`
4. Extract splits that explain **remaining unexplained variance**

This captures marginal contribution of `x_i` given other predictors.

### 5. Prediction with Transformations

```r
predict.splitwise_lm(object, newdata) {
  decisions = object$splitwise_info$decisions

  # Apply same transformations to newdata
  for var in decisions:
    if type == "dummy":
      newdata[var + "_dummy"] = encode_dummy(newdata[var], cutoffs)

  # Use standard lm prediction
  predict.lm(object, newdata_transformed)
}
```

**Critical:** Must store transformation rules to apply to test data.

---

## Edge Cases and Error Handling

### 1. Non-Numeric Predictors

```r
# Error raised if any predictor is not numeric
if (!is.numeric(X[[var]])):
  stop("SplitWise requires all predictors to be numeric")
```

**Python Implementation:** Check dtypes before transformation.

### 2. Zero-Variance Predictors

```r
if length(unique(x_vec[!is.na(x_vec)])) == 1:
  # Skip dummy transformation, use linear (will be dropped in stepwise)
  decisions[[var]] = {type: "linear", cutoff: NULL}
```

### 3. All-NA Predictors

```r
if all(is.na(x_vec)):
  # Skip dummy transformation
  decisions[[var]] = {type: "linear", cutoff: NULL}
```

### 4. No Valid Splits Found

```r
if tree$splits is NULL or nrow(tree$splits) == 0:
  # rpart couldn't find valid splits
  # Use linear form
  decisions[[var]] = {type: "linear", cutoff: NULL}
```

### 5. Support Constraint Violations

```r
if support <= min_support or support >= (1 - min_support):
  # Dummy would be too imbalanced
  # Set AIC/BIC to Inf (won't be selected)
  dummy_aic = Inf
```

---

## Performance Characteristics

### Computational Complexity

**Univariate Mode:**
- O(p × n log n) where p = number of predictors
- Each predictor: fit one `rpart` tree + 3 linear models
- Fast and parallelizable

**Iterative Mode:**
- O(p² × n log n × iterations) worst case
- Each iteration: evaluate all variables, fit trees, compare models
- Slower but more accurate
- Typical iterations: 5-20 for most datasets

**Memory:**
- Stores original data + transformed data
- O(n × p) for data
- O(p) for decisions metadata

### Scalability

**Recommended Use:**
- n: 50 - 100,000 observations (tested on mtcars with n=32)
- p: 2 - 100 predictors
- For large p, consider univariate mode or feature pre-screening

---

## Comparison with Standard Stepwise

**Benchmark Results (from test-02-benchmark.R):**

On mtcars dataset:
```
Direction    Transformation Mode   SplitWise AIC   Step AIC
----------------------------------------------------------------
backward     univariate           [lower]          [baseline]
backward     iterative            [lower]          [baseline]
forward      univariate           [lower]          [baseline]
forward      iterative            [lower]          [baseline]
both         univariate           [lower]          [baseline]
both         iterative            [lower]          [baseline]
```

**Key Finding:** Mean SplitWise AIC ≤ Mean standard stepwise AIC

**Advantages of SplitWise:**
- Captures non-linear relationships automatically
- More interpretable than polynomial terms
- Often better predictive performance
- Automatic threshold discovery

---

## Python Implementation Considerations

### For py-recipes Step

**Recommended Design:**

```python
from py_recipes import recipe

rec = (recipe(data, "mpg ~ .")
    .step_splitwise(
        all_numeric(),
        transformation_mode="iterative",  # or "univariate"
        min_support=0.1,
        min_improvement=3.0,
        criterion="AIC",
        exclude=[],  # variable names to exclude
        outcome=None  # auto-detect from formula
    ))

prepped = rec.prep()
transformed = prepped.bake(new_data)
```

**Key Components Needed:**

1. **Decision Tree Fitter**
   - Use `sklearn.tree.DecisionTreeRegressor` with `max_depth=2`
   - Or `rpy2` to call R's `rpart` directly

2. **Model Comparison**
   - Fit OLS models with `statsmodels.api.OLS`
   - Compute AIC/BIC via `statsmodels`

3. **State Storage**
   - Store `decisions` dict in step state
   - Store cutoffs for each variable
   - Store variable types (linear vs dummy)

4. **Baking Logic**
   - Apply same cutoffs to new data
   - Create dummy columns matching training
   - Handle missing variables gracefully

**Challenges:**
- Requires access to outcome variable (supervised transformation)
- Decision logic is complex (iterative mode especially)
- Integration with recipes' tidy data flow

**Simplifications for MVP:**
- Start with univariate mode only
- Skip stepwise selection (just do transformations)
- Later add iterative mode if needed

---

## Usage Examples

### Example 1: Univariate with Backward Selection

```r
library(SplitWise)
data(mtcars)

model = splitwise(
  mpg ~ cyl + disp + hp + wt,
  data = mtcars,
  transformation_mode = "univariate",
  direction = "backward",
  min_support = 0.15,
  min_improvement = 2.0,
  criterion = "AIC"
)

summary(model)
# Shows: which variables became dummies, cutoff values, coefficients
```

### Example 2: Iterative with Forward Selection

```r
model = splitwise(
  mpg ~ .,
  data = mtcars,
  transformation_mode = "iterative",
  direction = "forward",
  criterion = "BIC",
  k = log(nrow(mtcars)),  # BIC penalty
  exclude_vars = c("cyl")  # Keep cylinder count linear
)

print(model)
# Shows: stepwise path, final dummy encodings
```

### Example 3: Prediction

```r
# Fit on training data
train_model = splitwise(mpg ~ ., data = mtcars[1:25,])

# Predict on test data (transformations applied automatically)
preds = predict(train_model, newdata = mtcars[26:32,])
```

---

## References

[1] Kurbucz, M. T., Tzivanakis, N., Aslam, N. S., & Sykulski, A. M. (2025). SplitWise Regression: Stepwise Modeling with Adaptive Dummy Encoding. arXiv:2505.15423. https://arxiv.org/abs/2505.15423

[2] SplitWise R Package. GitHub: https://github.com/mtkurbucz/SplitWise

[3] Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). Classification and regression trees. CRC press.

---

## Summary for Python Implementation

**Core Algorithm:**
1. For each numeric predictor, fit shallow decision tree (depth=2)
2. Extract 1-2 split points from tree
3. Create candidate dummy variables:
   - Single-split: `x >= threshold`
   - Double-split: `lower < x < upper`
4. Compare AIC/BIC: linear vs single-split vs double-split
5. Keep best form if improvement >= `min_improvement`
6. Apply support constraint: dummy groups must be >= `min_support` fraction

**Python Recipe Step Signature:**

```python
step_splitwise(
    terms: Selector,
    transformation_mode: str = "iterative",
    min_support: float = 0.1,
    min_improvement: float = 3.0,
    criterion: str = "AIC",
    exclude: List[str] = None,
    outcome: str = None,
    role: str = "predictor",
    trained: bool = False,
    decisions: Dict = None,
    skip: bool = False,
    id: str = None
)
```

**State to Store (prep phase):**
```python
self.decisions = {
    "var1": {
        "type": "dummy",
        "cutoff": [5.5],  # single split
        "original_name": "var1",
        "dummy_name": "var1_dummy"
    },
    "var2": {
        "type": "dummy",
        "cutoff": [2.0, 8.0],  # double split
        "original_name": "var2",
        "dummy_name": "var2_dummy"
    },
    "var3": {
        "type": "linear",
        "cutoff": None,
        "original_name": "var3"
    }
}
```

**Bake Phase Logic:**
```python
for var_name, decision in self.decisions.items():
    if decision["type"] == "linear":
        # Keep original column
        continue

    elif decision["type"] == "dummy":
        cutoffs = decision["cutoff"]

        if len(cutoffs) == 1:
            # Single-split
            data[decision["dummy_name"]] = (
                data[var_name] >= cutoffs[0]
            ).astype(int)

        elif len(cutoffs) == 2:
            # Double-split (middle region)
            data[decision["dummy_name"]] = (
                (data[var_name] > cutoffs[0]) &
                (data[var_name] < cutoffs[1])
            ).astype(int)

        # Drop original column (replaced by dummy)
        data.drop(columns=[var_name], inplace=True)
```

This provides a complete foundation for implementing `step_splitwise` in the py-recipes package.
