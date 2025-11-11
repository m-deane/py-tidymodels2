# Architecture Compliance Tests

This directory contains comprehensive tests that enforce architectural constraints and design patterns across the py-tidymodels codebase.

## Purpose

Architecture tests prevent **architectural drift** by:
- Enforcing layer dependency rules
- Validating interface compliance
- Ensuring immutability patterns
- Verifying output consistency
- Checking implementation patterns

## Test Suites

### 1. Dependency Tests (`test_dependencies.py`)

**Purpose:** Enforce layered architecture and prevent circular dependencies.

**Tests:**
- Layer dependency constraints (8 layers)
- Circular import detection
- Module importability verification
- Upward dependency prevention

**Example violations caught:**
```python
# VIOLATION: Lower layer importing from higher layer
# py_hardhat importing py_parsnip (Layer 1 → Layer 2)
import py_parsnip  # ❌ FAILS TEST

# CORRECT: Higher layer importing from lower layer
# py_parsnip importing py_hardhat (Layer 2 → Layer 1)
import py_hardhat  # ✅ PASSES
```

**Run:**
```bash
pytest tests/test_architecture/test_dependencies.py -v
```

### 2. Engine Compliance Tests (`test_engine_compliance.py`)

**Purpose:** Verify all engines implement the Engine ABC correctly.

**Tests:**
- Engine registration verification
- ABC inheritance checks
- Required method validation (fit, predict, extract_outputs)
- Parameter translation verification
- Method signature validation
- Documentation completeness

**Example violations caught:**
```python
# VIOLATION: Engine missing required method
class BadEngine(Engine):
    def fit(self, spec, molded):
        pass
    # Missing predict() and extract_outputs()  # ❌ FAILS TEST

# CORRECT: Complete engine implementation
class GoodEngine(Engine):
    def fit(self, spec, molded):
        pass
    def predict(self, fit, molded, type):
        pass
    def extract_outputs(self, fit):
        pass  # ✅ PASSES
```

**Run:**
```bash
pytest tests/test_architecture/test_engine_compliance.py -v
```

### 3. Immutability Tests (`test_immutability.py`)

**Purpose:** Verify frozen dataclasses and immutable specification patterns.

**Tests:**
- ModelSpec frozen dataclass validation
- Workflow frozen dataclass validation
- Blueprint frozen dataclass validation
- Modification method returns new instances
- Builder pattern correctness
- Specification reuse safety

**Example violations caught:**
```python
# VIOLATION: Modifying frozen dataclass
spec = ModelSpec(model_type="linear_reg", engine="sklearn")
spec.engine = "statsmodels"  # ❌ FAILS TEST (FrozenInstanceError)

# CORRECT: Using replacement method
spec = ModelSpec(model_type="linear_reg", engine="sklearn")
new_spec = spec.set_engine("statsmodels")  # ✅ PASSES
```

**Run:**
```bash
pytest tests/test_architecture/test_immutability.py -v
```

### 4. Output Consistency Tests (`test_output_consistency.py`)

**Purpose:** Verify standardized three-DataFrame output structure.

**Tests:**
- extract_outputs() return type validation
- Outputs DataFrame structure (actuals, fitted, residuals, forecast, split)
- Coefficients DataFrame structure (term, estimate)
- Stats DataFrame structure (metric, value)
- Column type validation
- Cross-engine consistency

**Example violations caught:**
```python
# VIOLATION: Incorrect output structure
def extract_outputs(self, fit):
    return outputs_df, coefficients_df  # ❌ FAILS TEST (only 2 DataFrames)

# CORRECT: Three-DataFrame structure
def extract_outputs(self, fit):
    return outputs_df, coefficients_df, stats_df  # ✅ PASSES
```

**Run:**
```bash
pytest tests/test_architecture/test_output_consistency.py -v
```

### 5. Pattern Consistency Tests (`test_patterns.py`)

**Purpose:** Verify consistent implementation patterns across components.

**Tests:**
- Model factory function patterns
- Recipe step protocol compliance
- prep()/bake() workflow validation
- Chaining pattern verification
- Naming convention checks
- Error handling consistency

**Example violations caught:**
```python
# VIOLATION: Model factory not returning ModelSpec
def bad_model():
    return {"model_type": "bad_model"}  # ❌ FAILS TEST (returns dict)

# CORRECT: Model factory returns ModelSpec
def good_model():
    return ModelSpec(model_type="good_model")  # ✅ PASSES
```

**Run:**
```bash
pytest tests/test_architecture/test_patterns.py -v
```

## Running All Architecture Tests

```bash
# Run all architecture tests
pytest tests/test_architecture/ -v

# Run with detailed output
pytest tests/test_architecture/ -vv --tb=long

# Run with coverage
pytest tests/test_architecture/ --cov=py_parsnip --cov=py_workflows --cov-report=html

# Run specific test class
pytest tests/test_architecture/test_engine_compliance.py::TestEngineInterfaceCompliance -v

# Run specific test
pytest tests/test_architecture/test_immutability.py::TestModelSpecImmutability::test_model_spec_is_frozen -v
```

## Test Statistics

| Test Suite | Tests | Pass Rate | Purpose |
|------------|-------|-----------|---------|
| Dependencies | 17 | 88% | Layer architecture |
| Engine Compliance | 14 | 100% | Interface contracts |
| Immutability | 23 | 100% | Frozen dataclasses |
| Output Consistency | 19 | 79% | Output structure |
| Pattern Consistency | 19 | 84% | Implementation patterns |
| **TOTAL** | **92** | **88%** | **Architecture enforcement** |

## Integration with CI/CD

Add to `.github/workflows/test.yml`:

```yaml
- name: Run architecture tests
  run: |
    pytest tests/test_architecture/ -v --tb=short
  # Fail build on architecture violations
  continue-on-error: false
```

## Understanding Test Failures

### Common Failure Patterns

1. **Layer Dependency Violation:**
```
AssertionError: Layer 2 (py_parsnip) imports from Layer 6 (py_yardstick)
```
**Fix:** Refactor to remove upward dependency. Move shared code to lower layer or use dependency injection.

2. **Engine Missing Method:**
```
AssertionError: my_engine + sklearn: missing extract_outputs()
```
**Fix:** Implement the required method in your engine class.

3. **Immutability Violation:**
```
FrozenInstanceError: cannot assign to field 'engine'
```
**Fix:** Use replacement methods like `set_engine()` instead of direct assignment.

4. **Output Structure Mismatch:**
```
AssertionError: extract_outputs() should return 3 elements
```
**Fix:** Return tuple of (outputs_df, coefficients_df, stats_df).

5. **Missing Required Columns:**
```
AssertionError: outputs missing column 'fitted'
```
**Fix:** Ensure outputs DataFrame has all required columns.

## Best Practices

### When Adding New Engines

1. Run engine compliance tests first:
```bash
pytest tests/test_architecture/test_engine_compliance.py -v
```

2. Verify output structure:
```bash
pytest tests/test_architecture/test_output_consistency.py -v
```

3. Check your engine appears in registry:
```python
from py_parsnip.engine_registry import list_engines
print(list_engines())
```

### When Adding New Layers

1. Update dependency tests with new layer
2. Add layer to hierarchy in `test_layer_dependency_order()`
3. Add importability test for new layer
4. Document layer responsibilities

### When Modifying Core Classes

1. Run immutability tests if changing frozen classes
2. Run pattern tests if changing factory functions
3. Run output tests if changing extract_outputs()

## Architecture Decision Records

Key architectural decisions enforced by these tests:

1. **Frozen Specifications:** All specification objects (ModelSpec, Workflow, Blueprint) are frozen dataclasses to prevent accidental mutation.

2. **Layered Architecture:** 8 layers with strict dependency rules prevent circular dependencies and maintain clean separation of concerns.

3. **Engine ABC:** All model engines must implement the Engine ABC for consistent interface and predictable behavior.

4. **Three-DataFrame Output:** All engines return (outputs, coefficients, stats) for consistent model introspection.

5. **Builder Pattern:** Fluent interface with method chaining for ergonomic API.

## Maintenance

### Adding New Tests

1. Identify architectural constraint or pattern
2. Write test that fails when constraint violated
3. Verify test passes with current codebase
4. Add test to appropriate test suite
5. Update this README

### Updating Tests

1. When changing architecture, update tests first (TDD)
2. Ensure tests document WHY constraint exists
3. Add comments explaining failure scenarios
4. Update ARCHITECTURE_TEST_SUMMARY.md

## References

- **CLAUDE.md:** Project architecture documentation
- **ARCHITECTURE_TEST_SUMMARY.md:** Detailed test results and findings
- **py_parsnip/engine_registry.py:** Engine ABC definition
- **py_hardhat/blueprint.py:** Blueprint frozen dataclass
- **py_workflows/workflow.py:** Workflow frozen dataclass

## Questions?

If a test fails and you're unsure why:

1. Read the test docstring (explains what's being tested)
2. Check ARCHITECTURE_TEST_SUMMARY.md for known issues
3. Review CLAUDE.md for architectural patterns
4. Look at passing examples in other engines/components

The tests are designed to be **helpful guides** rather than obstacles. They document architectural decisions and prevent accidental violations.
