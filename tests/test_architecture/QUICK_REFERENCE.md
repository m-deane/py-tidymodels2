# Architecture Tests - Quick Reference

## Test Suite at a Glance

| Suite | Tests | Key Checks |
|-------|-------|------------|
| Dependencies | 17 | Layer hierarchy, circular imports |
| Engine Compliance | 14 | ABC inheritance, required methods |
| Immutability | 23 | Frozen dataclasses, builder pattern |
| Output Consistency | 19 | Three-DataFrame structure |
| Pattern Consistency | 19 | Factory patterns, recipe protocol |
| **TOTAL** | **92** | **Architecture enforcement** |

## Quick Test Commands

```bash
# Run all architecture tests
pytest tests/test_architecture/ -v

# Run specific suite
pytest tests/test_architecture/test_engine_compliance.py -v

# Run with failures only
pytest tests/test_architecture/ --tb=short -x

# Generate coverage report
pytest tests/test_architecture/ --cov --cov-report=html
```

## Common Violations and Fixes

### 1. Layer Dependency Violation
**Symptom:** `AssertionError: Layer X imports from Layer Y`

**Fix:**
```python
# ❌ WRONG: Lower layer importing higher layer
# In py_hardhat:
from py_parsnip import ModelSpec  # Layer 1 → Layer 2 violation

# ✅ CORRECT: Higher layer importing lower layer
# In py_parsnip:
from py_hardhat import Blueprint  # Layer 2 → Layer 1 OK
```

### 2. Engine Missing Methods
**Symptom:** `Engine missing extract_outputs()`

**Fix:**
```python
# ❌ WRONG: Incomplete engine
@register_engine("my_model", "my_engine")
class MyEngine(Engine):
    def fit(self, spec, molded):
        pass

# ✅ CORRECT: Complete engine
@register_engine("my_model", "my_engine")
class MyEngine(Engine):
    param_map = {}

    def fit(self, spec, molded):
        pass

    def predict(self, fit, molded, type):
        pass

    def extract_outputs(self, fit):
        return outputs_df, coefficients_df, stats_df
```

### 3. Immutability Violation
**Symptom:** `FrozenInstanceError: cannot assign to field`

**Fix:**
```python
# ❌ WRONG: Direct mutation
spec = linear_reg()
spec.engine = "statsmodels"  # Raises FrozenInstanceError

# ✅ CORRECT: Replacement method
spec = linear_reg()
new_spec = spec.set_engine("statsmodels")
```

### 4. Output Structure Mismatch
**Symptom:** `extract_outputs() should return 3 elements`

**Fix:**
```python
# ❌ WRONG: Missing DataFrame
def extract_outputs(self, fit):
    return outputs_df, coefficients_df

# ✅ CORRECT: Three DataFrames
def extract_outputs(self, fit):
    outputs_df = pd.DataFrame(...)
    coefficients_df = pd.DataFrame(...)
    stats_df = pd.DataFrame(...)
    return outputs_df, coefficients_df, stats_df
```

### 5. Missing Required Columns
**Symptom:** `outputs missing column 'fitted'`

**Fix:**
```python
# ❌ WRONG: Missing required column
outputs_df = pd.DataFrame({
    'actuals': actuals,
    'residuals': residuals
})

# ✅ CORRECT: All required columns
outputs_df = pd.DataFrame({
    'actuals': actuals,
    'fitted': fitted,
    'residuals': residuals,
    'forecast': forecast,
    'split': split
})
```

## Architecture Rules (Memorize These!)

### Rule 1: Layer Hierarchy
```
Layer 1: hardhat    (foundation - no dependencies)
Layer 2: parsnip    (depends on hardhat)
Layer 3: rsample    (independent)
Layer 4: workflows  (depends on parsnip, recipes)
Layer 5: recipes    (independent preprocessing)
Layer 6: yardstick  (independent metrics)
Layer 7: tune       (depends on workflows, rsample, yardstick)
Layer 8: workflowsets (depends on tune)
```

**Rule:** Lower layers NEVER import from higher layers.

### Rule 2: Specification Immutability
```python
# All specs are FROZEN dataclasses
ModelSpec   → frozen=True
Workflow    → frozen=True
Blueprint   → frozen=True

# Modifications return NEW instances
spec.set_engine()  → returns new ModelSpec
wf.add_model()     → returns new Workflow
```

### Rule 3: Engine Interface
```python
# ALL engines must implement:
class MyEngine(Engine):
    param_map = {...}  # Required attribute

    # Standard path OR raw path
    def fit(self, spec, molded) / fit_raw(self, spec, data, formula)
    def predict(self, fit, molded, type) / predict_raw(...)

    # Always required
    def extract_outputs(self, fit):
        return (outputs_df, coefficients_df, stats_df)
```

### Rule 4: Output Structure
```python
# extract_outputs() ALWAYS returns 3 DataFrames:

# 1. outputs: Observation-level results
outputs_df = pd.DataFrame({
    'actuals': [...],    # True values
    'fitted': [...],     # Predictions (never NaN!)
    'residuals': [...],  # actuals - fitted
    'forecast': [...],   # combine_first(actuals, fitted)
    'split': [...],      # 'train', 'test', 'forecast'
})

# 2. coefficients: Model parameters
coefficients_df = pd.DataFrame({
    'term': [...],       # Parameter name
    'estimate': [...],   # Parameter value
    # Optional: std_error, t_stat, p_value, conf_low, conf_high
})

# 3. stats: Model-level metrics
stats_df = pd.DataFrame({
    'metric': [...],     # Metric name (rmse, mae, r_squared)
    'value': [...],      # Numeric value
    'split': [...],      # 'train', 'test'
})
```

### Rule 5: Model Factory Pattern
```python
# Model factories MUST:
# 1. Return ModelSpec
# 2. Set correct model_type
# 3. Have default parameters (when possible)

def my_model(param1=default1, param2=default2, engine="default"):
    return ModelSpec(
        model_type="my_model",  # Must match function name
        engine=engine,
        mode="regression",
        args={"param1": param1, "param2": param2}
    )
```

### Rule 6: Recipe Step Protocol
```python
# Recipe steps MUST:
# 1. Have prep() method → returns PreparedStep
# 2. Have bake() method → returns DataFrame

class StepMyTransform:
    def prep(self, data: pd.DataFrame) -> PreparedStep:
        # Learn from training data
        return prepared_step

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        # Apply transformation
        return transformed_data
```

## Pre-Commit Checklist

Before committing code that touches core architecture:

- [ ] Run `pytest tests/test_architecture/ -v`
- [ ] All architecture tests pass
- [ ] No new layer dependency violations
- [ ] New engines implement full interface
- [ ] Specs remain immutable
- [ ] Output structure consistent
- [ ] Documentation updated

## When to Update Architecture Tests

Update tests when:
- Adding new architectural layer
- Changing core interfaces (Engine ABC)
- Modifying frozen dataclasses
- Adding new output requirements
- Establishing new patterns

Don't update tests to:
- Accommodate violations (fix code instead)
- Bypass immutability (by design)
- Relax layer rules (protect architecture)

## Test Failure Triage

1. **Read the assertion message** - tells you exactly what's wrong
2. **Check ARCHITECTURE_TEST_SUMMARY.md** - known issues documented
3. **Review passing examples** - see how other engines/components do it
4. **Check CLAUDE.md** - architectural patterns documented
5. **Ask for clarification** - tests should be clear, file issue if not

## Integration Points

Architecture tests integrate with:
- **CI/CD:** Run on every commit
- **Pre-commit hooks:** Optional local validation
- **Code review:** Reference in PR discussions
- **Documentation:** Living architecture documentation

## Performance

Architecture tests are **fast** (< 3 seconds):
- No external dependencies
- No network calls
- Mostly static analysis
- Safe to run frequently

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Pass rate | > 90% | 88% |
| Coverage | All engines | 33/33 ✓ |
| Layer violations | 0 | 2 |
| Missing methods | 0 | 0 ✓ |
| Documentation | Complete | ✓ |

## Resources

- Full documentation: `tests/test_architecture/README.md`
- Test results: `.claude_plans/ARCHITECTURE_TEST_SUMMARY.md`
- Project architecture: `CLAUDE.md`
- Engine registry: `py_parsnip/engine_registry.py`

---

**Remember:** Architecture tests are not obstacles - they're **guard rails** that keep the codebase maintainable as it grows. They document decisions and prevent accidental violations.
