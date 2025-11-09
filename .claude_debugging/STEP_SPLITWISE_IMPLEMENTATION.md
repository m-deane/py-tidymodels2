# step_splitwise() Implementation - Complete Documentation

**Implementation Date:** 2025-11-09
**Based on:** SplitWise R package (Kurbucz et al., 2025)
**Reference:** arXiv:2505.15423

---

## Overview

`step_splitwise()` implements adaptive dummy encoding for numeric predictors using shallow decision trees. The method automatically determines whether to transform each numeric predictor into binary dummy variables (with 1-2 split points) or keep them as continuous linear predictors.

**Key Innovation:** Data-driven transformation decisions based on AIC/BIC model comparison, balancing model fit with parsimony.

---

## Algorithm

### Core Methodology

For each numeric predictor, SplitWise:

1. **Fit Shallow Decision Tree**
   - Uses `DecisionTreeRegressor(max_depth=2)` to find potential split points
   - Tree depth limited to prevent overfitting
   - Minimum leaf size enforced for stable splits

2. **Extract Split Points**
   - Traverses tree structure to extract thresholds
   - Filters to points within data range
   - Removes duplicates

3. **Compare Three Options**
   - **Linear**: Keep original numeric predictor unchanged
   - **Single-split**: Binary dummy `x >= threshold`
   - **Double-split**: Binary dummy `lower < x < upper` (middle region)

4. **Select Best Transformation**
   - Fits linear regression for each option
   - Computes AIC or BIC
   - Selects transformation with best criterion value (lowest AIC/BIC)
   - Requires minimum improvement threshold to prefer dummy over linear

5. **Apply Support Constraint**
   - Rejects splits creating highly imbalanced groups
   - Ensures both groups have at least `min_support` fraction of observations
   - Default: 10% minimum in each group

### AIC/BIC Computation

```python
# Number of parameters
k = 2  # Intercept + slope

# Log-likelihood (assuming normal errors)
sigma2 = rss / n
log_likelihood = -0.5 * n * (log(2π) + log(sigma2) + 1)

# Information criteria
AIC = -2 * log_likelihood + 2 * k
BIC = -2 * log_likelihood + k * log(n)
```

**BIC vs AIC:** BIC penalizes complexity more heavily, leading to more conservative transformations (fewer dummy variables).

---

## Parameters

### Required

- **`outcome`** (str): Name of outcome variable for supervised transformation

### Optional

- **`transformation_mode`** (str, default='univariate')
  - `'univariate'`: Each predictor evaluated independently ✅ **Implemented**
  - `'iterative'`: Adaptive with partial residuals ❌ **Not yet implemented**

- **`min_support`** (float, default=0.1)
  - Minimum fraction of observations in each dummy group
  - Range: (0, 0.5)
  - Higher values = more balanced splits required

- **`min_improvement`** (float, default=3.0)
  - Minimum AIC/BIC improvement to prefer dummy over linear
  - Higher values = more conservative (fewer transformations)
  - Value of 3.0 matches typical "strong evidence" threshold

- **`criterion`** (str, default='AIC')
  - Model selection criterion: 'AIC' or 'BIC'
  - BIC more conservative than AIC

- **`exclude_vars`** (list of str, optional)
  - Variables forced to stay linear (no transformation)
  - Useful for known linear relationships

- **`columns`** (selector, optional)
  - Which columns to consider for transformation
  - If None, uses all numeric predictors except outcome
  - Supports selector functions (e.g., `all_numeric()`)

- **`skip`** (bool, default=False)
  - Skip this step during prep/bake

- **`id`** (str, optional)
  - Unique identifier for this step

---

## Usage Examples

### Basic Usage

```python
from py_recipes import recipe
from py_workflows import workflow
from py_parsnip import linear_reg
import pandas as pd
import numpy as np

# Create data with non-linear relationships
np.random.seed(42)
n = 300
x1 = np.random.uniform(-5, 5, n)
x2 = np.random.uniform(-3, 3, n)
x3 = np.random.randn(n)

y = (
    10 * (x1 > 0).astype(int) +  # Threshold effect
    5 * ((x2 < -1) | (x2 > 1)).astype(int) +  # U-shaped
    2 * x3 +  # Linear
    np.random.randn(n) * 0.5
)

data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'y': y})

# Create recipe with splitwise
rec = (
    recipe()
    .step_splitwise(outcome='y', min_improvement=2.0)
)

# Build workflow
wf = (
    workflow()
    .add_recipe(rec)
    .add_model(linear_reg())
)

# Fit and predict
fit = wf.fit(data.iloc[:200])
preds = fit.predict(data.iloc[200:])
```

### Custom Parameters

```python
# More conservative transformation (higher improvement threshold)
rec = (
    recipe()
    .step_splitwise(
        outcome='sales',
        min_improvement=5.0,  # Require stronger evidence
        min_support=0.15,     # More balanced splits
        criterion='BIC'        # More conservative criterion
    )
)
```

### Excluding Variables

```python
# Force certain variables to stay linear
rec = (
    recipe()
    .step_splitwise(
        outcome='price',
        exclude_vars=['year', 'month'],  # Don't transform time variables
        min_improvement=2.0
    )
)
```

### Combined with Other Steps

```python
from py_recipes.selectors import all_numeric_predictors

# Splitwise followed by normalization
rec = (
    recipe()
    .step_splitwise(outcome='y', min_improvement=2.0)
    .step_normalize(all_numeric_predictors())  # Normalize remaining numeric vars
)
```

### Inspecting Transformation Decisions

```python
# Prep recipe to see transformation decisions
prepped = rec.prep(train_data)

# Access the prepared step
splitwise_step = prepped.prepared_steps[0]

# Get decisions for all variables
decisions = splitwise_step.get_decisions()

for var, info in decisions.items():
    decision = info['decision']
    cutoffs = info['cutoffs']

    if decision == 'linear':
        print(f"{var}: kept linear")
    elif decision == 'single_split':
        threshold = cutoffs['threshold']
        print(f"{var}: single split at {threshold:.4f}")
    elif decision == 'double_split':
        lower = cutoffs['lower']
        upper = cutoffs['upper']
        print(f"{var}: double split ({lower:.4f}, {upper:.4f})")
```

### Evaluating Model Performance

```python
# Split data
train = data.iloc[:200]
test = data.iloc[200:]

# Fit and evaluate
fit = wf.fit(train).evaluate(test)

# Extract outputs
outputs, coefficients, stats = fit.extract_outputs()

# View statistics
print(stats[stats['metric'] == 'rmse'])
```

---

## Implementation Details

### File Structure

1. **Core Implementation:** `py_recipes/steps/splitwise.py` (463 lines)
   - `StepSplitwise` class (dataclass)
   - `prep()` method - determines transformations
   - `bake()` method - applies transformations
   - `_decide_transformation_univariate()` - univariate decision logic
   - `_extract_split_points()` - tree threshold extraction
   - `_compute_aic()` - AIC/BIC calculation
   - `_sanitize_threshold()` - **NEW:** patsy-friendly column names
   - `get_decisions()` - inspect transformation decisions

2. **Recipe Integration:** `py_recipes/recipe.py` (lines 841-895)
   - `step_splitwise()` method added to Recipe class

3. **Registration:** `py_recipes/steps/__init__.py`
   - Imported and exported in `__all__`

4. **Tests:**
   - `tests/test_recipes/test_splitwise.py` - 26 unit tests
   - `tests/test_recipes/test_splitwise_workflow_integration.py` - 7 workflow tests
   - **Total: 33 tests, all passing**

### Key Design Decisions

#### 1. Supervised Transformation
Unlike unsupervised steps (normalize, scale), step_splitwise **requires the outcome variable** during prep(). This allows data-driven decisions based on predictive performance.

#### 2. State Storage Pattern
Uses dataclass fields with `init=False, repr=False` to store transformation state:
```python
_decisions: Dict[str, str]  # 'linear', 'single_split', or 'double_split'
_cutoffs: Dict[str, Dict[str, Any]]  # Threshold values for dummy creation
_original_columns: List[str]  # Columns considered for transformation
_is_prepared: bool  # Prep status flag
```

#### 3. Column Name Sanitization **CRITICAL**
Original threshold values can contain special characters (negative signs, decimals) that break patsy formula parsing:

**Problem:**
```python
# Negative threshold creates invalid column name
threshold = -0.0004
dummy_name = f"x1_ge_{threshold:.4f}"  # "x1_ge_-0.0004"
# Patsy error: '-' not valid in identifier
```

**Solution:**
```python
def _sanitize_threshold(self, value: float) -> str:
    formatted = f"{value:.4f}"
    sanitized = formatted.replace('-', 'm')  # Minus
    sanitized = sanitized.replace('.', 'p')  # Point
    return sanitized

# Result: "x1_ge_m0p0004" ✅ Valid identifier
```

**Examples:**
- `-0.0004` → `m0p0004`
- `0.9981` → `0p9981`
- `-0.9248` → `m0p9248`

#### 4. Univariate Mode Only
Iterative mode (with partial residuals) not yet implemented. The current univariate mode:
- Evaluates each predictor independently
- Faster computation
- Simpler interpretation
- Sufficient for most use cases

#### 5. Shallow Trees Prevent Overfitting
- `max_depth=2` limits complexity
- `min_samples_leaf=max(5, int(len(x) * 0.05))` ensures stable splits
- Deep trees would find spurious patterns

---

## Test Coverage

### Unit Tests (26 tests)

**TestStepSplitwiseBasics (6 tests):**
- ✅ Step creation with default parameters
- ✅ Step creation with custom parameters
- ✅ Invalid `min_support` validation (must be in (0, 0.5))
- ✅ Invalid `min_improvement` validation (must be >= 0)
- ✅ Invalid `criterion` validation (must be AIC or BIC)
- ✅ Iterative mode raises NotImplementedError

**TestStepSplitwisePrep (4 tests):**
- ✅ Basic prep functionality
- ✅ Error when outcome not found
- ✅ Excluding variables from transformation
- ✅ Transformation decisions stored correctly

**TestStepSplitwiseBake (5 tests):**
- ✅ Creates dummy variables with sanitized names
- ✅ Preserves linear variables unchanged
- ✅ Single-split dummy variables are binary
- ✅ Applies transformations to new data (test set)
- ✅ Skip parameter prevents transformation

**TestStepSplitwiseRecipe (2 tests):**
- ✅ Recipe integration with step_splitwise
- ✅ Combining with other recipe steps

**TestStepSplitwiseTransformations (5 tests):**
- ✅ Linear relationships kept unchanged
- ✅ Threshold relationships transformed to dummies
- ✅ min_support constraint prevents imbalanced splits
- ✅ AIC vs BIC criterion comparison
- ✅ get_decisions() method returns correct info

**TestStepSplitwiseEdgeCases (4 tests):**
- ✅ Error when no numeric columns available
- ✅ Handles very small datasets gracefully
- ✅ Constant predictors kept linear
- ✅ Missing values handled by removal during fit

### Workflow Integration Tests (7 tests)

**TestSplitwiseWorkflowIntegration (7 tests):**
- ✅ Basic workflow with step_splitwise
- ✅ Transformation decisions applied correctly
- ✅ Evaluation with train/test splits
- ✅ Combining splitwise with normalization step
- ✅ Excluding specific variables from transformation
- ✅ AIC vs BIC criterion comparison
- ✅ min_support parameter effect

**All 33 tests passing in 0.62 seconds**

---

## Common Patterns and Use Cases

### 1. Detecting Non-Linear Relationships

```python
# Prep recipe
prepped = recipe().step_splitwise(outcome='y').prep(data)

# Inspect decisions
decisions = prepped.prepared_steps[0].get_decisions()

# Find transformed variables (non-linear relationships)
non_linear = [
    var for var, info in decisions.items()
    if info['decision'] in ['single_split', 'double_split']
]

print(f"Non-linear predictors: {non_linear}")
```

### 2. Conservative vs Aggressive Transformation

```python
# Conservative: fewer transformations
conservative = recipe().step_splitwise(
    outcome='y',
    min_improvement=5.0,  # High threshold
    criterion='BIC'        # More penalty for complexity
)

# Aggressive: more transformations
aggressive = recipe().step_splitwise(
    outcome='y',
    min_improvement=1.0,  # Low threshold
    criterion='AIC'        # Less penalty for complexity
)
```

### 3. Mixed Modeling Strategy

```python
# Keep domain knowledge variables linear, let data decide others
rec = recipe().step_splitwise(
    outcome='sales',
    exclude_vars=['price', 'cost'],  # Economic theory: linear relationships
    min_improvement=2.0               # Data-driven for others
)
```

### 4. Interpretable Non-Linear Models

```python
# SplitWise creates interpretable thresholds
# Example output: "Sales increase when temperature > 20°C"

fit = workflow().add_recipe(rec).add_model(linear_reg()).fit(data)
outputs, coeffs, stats = fit.extract_outputs()

# Coefficients show threshold effects
print(coeffs[['variable', 'estimate', 'p_value']])
# temperature_ge_20p0000  coefficient: 1500.0  p<0.001
# ^ Sales boost when temp >= 20°C
```

---

## Comparison with Alternatives

### vs. Manual Dummy Encoding

**Manual:**
```python
data['high_income'] = (data['income'] > 50000).astype(int)
```
- Pro: Full control
- Con: Arbitrary threshold choice
- Con: Doesn't optimize for predictive performance

**SplitWise:**
```python
recipe().step_splitwise(outcome='purchase').prep(data)
```
- Pro: Data-driven threshold selection
- Pro: Automatic AIC/BIC optimization
- Pro: Considers multiple split options

### vs. Polynomial Terms

**Polynomial:**
```python
recipe().step_poly(['x1'], degree=2)
```
- Pro: Captures smooth non-linearities
- Con: Can be unstable at edges
- Con: Harder to interpret (quadratic, cubic effects)

**SplitWise:**
- Pro: Interpretable thresholds
- Pro: Robust to outliers (tree-based splits)
- Con: Piecewise constant (less smooth)

### vs. Splines

**Splines:**
```python
recipe().step_ns(['x1'], deg_free=3)
```
- Pro: Smooth non-linear curves
- Con: More parameters to estimate
- Con: Less interpretable (knot locations, basis functions)

**SplitWise:**
- Pro: Fewer parameters (1-2 dummies per variable)
- Pro: Clear threshold interpretation
- Pro: Automatic knot selection

---

## Known Limitations

### 1. Iterative Mode Not Implemented

**Current:** Univariate mode only (each predictor evaluated independently)
**Future:** Iterative mode with partial residuals (adaptive stepwise)

**Workaround:** Use univariate mode, which is sufficient for most cases.

### 2. Limited to Regression Outcomes

**Current:** Requires continuous outcome for AIC/BIC calculation
**Future:** Could extend to classification with appropriate criterion

**Workaround:** For classification, use step_discretize() or manual dummy encoding.

### 3. No Interaction Detection

**Current:** Each predictor transformed independently
**Future:** Could extend to detect predictor interactions

**Workaround:** Combine with step_interact() for explicit interactions.

### 4. Piecewise Constant Approximation

**Current:** Dummy variables create step functions
**Future:** Could combine with splines for smooth transitions

**Workaround:** For smooth non-linearities, use step_ns() or step_bs() instead.

---

## Performance Considerations

### Computational Complexity

**Per Variable:**
- Fit decision tree: O(n log n) for n observations
- Extract split points: O(tree nodes) ≈ O(1) for max_depth=2
- Evaluate transformations: O(n) per candidate split
- Total: O(n log n) per variable

**For p Variables:**
- O(p × n log n) total
- **Very efficient:** 300 observations × 3 variables in < 0.1 seconds

### Memory Usage

**During Prep:**
- Stores decisions and cutoffs: O(p) where p = number of variables
- Tree fitting: O(n) temporary memory
- **Minimal memory footprint**

**During Bake:**
- Applies transformations in-place when possible
- Creates new dummy columns: O(n × d) where d = number of dummies
- **Memory efficient:** No large intermediate structures

### Scaling Recommendations

- **< 1000 observations:** No issues, very fast
- **1000-10000 observations:** Still fast (< 1 second for 10 variables)
- **10000+ observations:** May want to sample for prep() if speed critical
- **100+ variables:** Consider excluding irrelevant variables with `columns` parameter

---

## Future Enhancements

### Priority 1: Iterative Mode

Implement adaptive transformation with partial residuals:
```python
# Future implementation
recipe().step_splitwise(
    outcome='y',
    transformation_mode='iterative'  # Not yet available
)
```

**Benefit:** Accounts for correlations between predictors during transformation.

### Priority 2: Classification Support

Extend to categorical outcomes:
```python
# Future: classification criterion
recipe().step_splitwise(
    outcome='species',
    criterion='deviance'  # For classification
)
```

**Benefit:** Adaptive dummy encoding for classification models.

### Priority 3: Ensemble Split Selection

Use multiple trees (random forest) to find robust split points:
```python
# Future: ensemble-based splits
recipe().step_splitwise(
    outcome='y',
    n_trees=100  # Multiple trees for stability
)
```

**Benefit:** More robust split point selection.

### Priority 4: Cross-Validation for Criterion

Use CV instead of AIC/BIC for model selection:
```python
# Future: CV-based selection
recipe().step_splitwise(
    outcome='y',
    criterion='cv',
    cv_folds=5
)
```

**Benefit:** More reliable criterion for small datasets.

---

## Troubleshooting

### Issue: All variables kept linear (no transformations)

**Possible Causes:**
1. `min_improvement` too high (default: 3.0)
2. Relationships are truly linear
3. Dataset too small for reliable tree fitting

**Solutions:**
- Lower `min_improvement` to 1.0 or 2.0
- Check data has non-linear patterns via scatter plots
- Ensure at least 50-100 observations for tree fitting

### Issue: Too many transformations (overfit)

**Possible Causes:**
1. `min_improvement` too low
2. Using AIC (less conservative than BIC)
3. Small dataset with noise

**Solutions:**
- Increase `min_improvement` to 5.0+
- Switch to `criterion='BIC'`
- Increase `min_support` to 0.2 (more balanced splits)

### Issue: Patsy formula errors with dummy columns

**Cause:** Column names with special characters
**Solution:** This is now **fixed** via `_sanitize_threshold()` method (replaces `-` with `m`, `.` with `p`)

**Before fix:**
```python
# Column: x1_ge_-0.0004  ❌ Invalid
```

**After fix:**
```python
# Column: x1_ge_m0p0004  ✅ Valid
```

### Issue: Highly imbalanced splits

**Possible Causes:**
1. `min_support` too low
2. Extreme outliers in data

**Solutions:**
- Increase `min_support` to 0.15 or 0.2
- Remove or cap extreme outliers before step_splitwise
- Consider step_filter_outliers() before transformation

---

## References

### Original Publication

Kurbucz, Marcell T.; Tzivanakis, Nikolaos; Aslam, Nilufer Sari; Sykulski, Adam M. (2025).
**SplitWise Regression: Stepwise Modeling with Adaptive Dummy Encoding.**
arXiv preprint: https://arxiv.org/abs/2505.15423

### Related Work

- **CART (Breiman et al., 1984):** Classification and Regression Trees - foundation for split detection
- **AIC (Akaike, 1974):** Information criterion for model selection
- **BIC (Schwarz, 1978):** Bayesian Information Criterion - more conservative than AIC
- **Splines (Hastie & Tibshirani, 1990):** Alternative approach to non-linear modeling

---

## Implementation History

**2025-11-09:** Initial implementation
- Core StepSplitwise class with univariate mode
- Recipe integration
- 26 unit tests
- 7 workflow integration tests
- Column name sanitization for patsy compatibility
- All 33 tests passing

**Files Created:**
- `py_recipes/steps/splitwise.py` (463 lines)
- `tests/test_recipes/test_splitwise.py` (428 lines)
- `tests/test_recipes/test_splitwise_workflow_integration.py` (285 lines)

**Files Modified:**
- `py_recipes/recipe.py` - Added step_splitwise() method (55 lines)
- `py_recipes/steps/__init__.py` - Registered StepSplitwise (3 lines)

**Total Implementation:** ~1200 lines of production code and tests

---

## Contact & Support

For issues or questions:
1. Check this documentation first
2. Review test files for usage examples
3. Consult the arXiv paper for methodology details
4. File GitHub issue for bugs or feature requests

---

**Documentation Version:** 1.0
**Last Updated:** 2025-11-09
**Status:** Production Ready ✅
