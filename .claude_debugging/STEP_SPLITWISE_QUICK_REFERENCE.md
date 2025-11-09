# step_splitwise() Quick Reference

**Status:** ✅ Production Ready | **Tests:** 33/33 Passing | **Date:** 2025-11-09

---

## Quick Start

```python
from py_recipes import recipe
from py_workflows import workflow
from py_parsnip import linear_reg

# Basic usage
rec = recipe().step_splitwise(outcome='y', min_improvement=2.0)
wf = workflow().add_recipe(rec).add_model(linear_reg())
fit = wf.fit(train_data)
preds = fit.predict(test_data)
```

---

## What It Does

Automatically transforms numeric predictors into either:
- **Binary dummies** (with 1-2 split points) if non-linear relationship detected
- **Linear predictors** (unchanged) if linear relationship

Decision based on AIC/BIC model comparison using shallow decision trees.

---

## Key Parameters

| Parameter | Default | Description | Typical Range |
|-----------|---------|-------------|---------------|
| `outcome` | **required** | Outcome variable name | - |
| `min_improvement` | 3.0 | Min AIC/BIC improvement for dummy | 1.0-5.0 |
| `min_support` | 0.1 | Min fraction in each group | 0.05-0.3 |
| `criterion` | 'AIC' | Model selection criterion | 'AIC', 'BIC' |
| `exclude_vars` | None | Variables to keep linear | list of str |

**Conservative:** Higher `min_improvement`, use `'BIC'`
**Aggressive:** Lower `min_improvement`, use `'AIC'`

---

## Common Patterns

### 1. Basic Transformation
```python
rec = recipe().step_splitwise(outcome='price')
```

### 2. Conservative (fewer transformations)
```python
rec = recipe().step_splitwise(
    outcome='sales',
    min_improvement=5.0,
    criterion='BIC'
)
```

### 3. Exclude Specific Variables
```python
rec = recipe().step_splitwise(
    outcome='revenue',
    exclude_vars=['year', 'month']  # Keep time vars linear
)
```

### 4. Inspect Transformation Decisions
```python
prepped = rec.prep(train_data)
decisions = prepped.prepared_steps[0].get_decisions()

for var, info in decisions.items():
    print(f"{var}: {info['decision']}")
    # Output: linear, single_split, or double_split
```

---

## Column Naming Convention

Dummy columns use sanitized naming (patsy-compatible):

| Transformation | Column Name Example |
|----------------|---------------------|
| Single-split (positive) | `x1_ge_0p5234` (x1 >= 0.5234) |
| Single-split (negative) | `x2_ge_m1p2345` (x2 >= -1.2345) |
| Double-split | `x3_between_m0p5_1p2` (-0.5 < x3 < 1.2) |
| Linear | `x4` (unchanged) |

**Naming:** `-` → `m` (minus), `.` → `p` (point)

---

## When to Use

✅ **Use when:**
- Non-linear relationships expected
- Want interpretable thresholds (e.g., "sales increase when temp > 20°C")
- Data-driven transformation decisions preferred
- Robust to outliers needed

❌ **Don't use when:**
- Smooth curves needed (use splines instead)
- Relationships known to be linear
- Very small datasets (< 50 observations)
- Classification outcomes (not yet supported)

---

## Alternatives

| Method | Use Case |
|--------|----------|
| `step_splitwise()` | Interpretable thresholds, automatic selection |
| `step_poly()` | Smooth polynomial curves |
| `step_ns()` / `step_bs()` | Flexible smooth splines |
| `step_discretize()` | Manual binning of numeric vars |
| Manual dummies | Full control over thresholds |

---

## Troubleshooting

### No transformations detected
- **Try:** Lower `min_improvement` to 1.0-2.0
- **Check:** Data has non-linear patterns (scatter plots)

### Too many transformations
- **Try:** Increase `min_improvement` to 5.0+
- **Try:** Switch to `criterion='BIC'`
- **Try:** Increase `min_support` to 0.2

### Highly imbalanced splits
- **Try:** Increase `min_support` to 0.15-0.2
- **Check:** Remove extreme outliers first

---

## Test Coverage

**Unit Tests (26):** Parameter validation, prep/bake, transformations, edge cases
**Workflow Tests (7):** End-to-end integration with workflows and models
**Total: 33 tests, all passing in 0.62 seconds**

---

## Files

- **Implementation:** `py_recipes/steps/splitwise.py` (463 lines)
- **Unit Tests:** `tests/test_recipes/test_splitwise.py` (428 lines)
- **Integration Tests:** `tests/test_recipes/test_splitwise_workflow_integration.py` (285 lines)
- **Full Docs:** `.claude_debugging/STEP_SPLITWISE_IMPLEMENTATION.md` (1050+ lines)

---

## Performance

- **Speed:** O(p × n log n) - Very fast even for large datasets
- **300 obs × 3 vars:** < 0.1 seconds
- **Memory:** Minimal - stores only decisions and cutoffs

---

## Reference

Kurbucz et al. (2025). SplitWise Regression: Stepwise Modeling with Adaptive Dummy Encoding.
arXiv: https://arxiv.org/abs/2505.15423

---

**Version:** 1.0 | **Status:** Production Ready ✅
