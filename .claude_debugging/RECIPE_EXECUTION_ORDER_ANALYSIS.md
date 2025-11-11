# py_recipes Step Execution Order Analysis

## Summary
Steps in py_recipes are executed in **STRICT SEQUENTIAL ORDER** with **NO AUTOMATIC REORDERING**. Both `prep()` and `bake()` follow identical execution patterns.

## Key Findings

### 1. Step Storage Mechanism
- **Location**: `Recipe.steps` (List[Any])
- **Type**: Simple Python list
- **Order**: Determined by `.add_step()` method calls
- **Immutability**: Recipe is a dataclass (not frozen), but convention is to not mutate

```python
@dataclass
class Recipe:
    steps: List[Any] = field(default_factory=list)
    
    def add_step(self, step: RecipeStep) -> "Recipe":
        """Add step to end of steps list"""
        self.steps.append(step)
        return self
```

### 2. Prep() Method - Execution Order

**Location**: `py_recipes/recipe.py:2249-2282`

```python
def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedRecipe":
    prepared_steps = []
    current_data = data.copy()

    for step in self.steps:  # ← Iterate in order added
        # 1. Prep the step
        prepared_step = step.prep(current_data, training=training)
        prepared_steps.append(prepared_step)

        # 2. Apply step to current data for next step
        current_data = prepared_step.bake(current_data)

    return PreparedRecipe(
        recipe=self,
        prepared_steps=prepared_steps,
        template=data
    )
```

**Execution Pattern**:
1. Iterate through `self.steps` **in the order they were added**
2. Call `.prep()` on each step with data transformed by previous steps
3. Immediately apply `.bake()` to transform data for next step
4. Store prepared steps in `PreparedRecipe.prepared_steps` **in same order**

**NO sorting, NO reordering, NO optimization**

### 3. Bake() Method - Execution Order

**Location**: `py_recipes/recipe.py:2307-2328`

```python
def bake(self, new_data: pd.DataFrame) -> pd.DataFrame:
    result = new_data.copy()

    for prepared_step in self.prepared_steps:  # ← Iterate in order prepped
        result = prepared_step.bake(result)

    return result
```

**Execution Pattern**:
1. Iterate through `self.prepared_steps` **in the order they were prepared during prep()**
2. Call `.bake()` on each step sequentially
3. Each step's output becomes the input for the next step

**Match**: bake() follows EXACT same order as prep()

### 4. Data Flow During prep()

Example with 3 steps:

```
Original Data
    ↓
Step 1.prep(data) → PreparedStep1
    ↓
Step 1.bake(data) → Data1 (transformed)
    ↓
Step 2.prep(Data1) → PreparedStep2
    ↓
Step 2.bake(Data1) → Data2 (transformed)
    ↓
Step 3.prep(Data2) → PreparedStep3
    ↓
Step 3.bake(Data2) → Data3 (transformed)
    ↓
Return PreparedRecipe with [PreparedStep1, PreparedStep2, PreparedStep3]
```

### 5. Data Flow During bake()

```
New Data
    ↓
PreparedStep1.bake(new_data) → Data1
    ↓
PreparedStep2.bake(Data1) → Data2
    ↓
PreparedStep3.bake(Data2) → Data3
    ↓
Return Data3
```

### 6. No Automatic Reordering

**Verified by checking**:
- No `sort()` or `sorted()` calls in recipe.py
- No filtering or reordering of steps
- No detection of step types (supervised vs unsupervised)
- No dependency tracking

**Examples of steps that execute in added order**:
1. `step_impute_median()` executes before `step_normalize()` if added first
2. `step_select_shap()` executes before `step_dummy()` if added first
3. Supervised steps (filter_*) execute at their added position, not first

**User is responsible for ordering**:
```python
# This works correctly
rec = (recipe()
    .step_impute_median()          # 1st
    .step_normalize()              # 2nd
    .step_filter_anova()           # 3rd - supervised
    .step_dummy())                 # 4th

# This might be problematic (filter after dummy)
rec = (recipe()
    .step_dummy()                  # 1st - creates many columns
    .step_filter_anova())          # 2nd - filters them
```

## Key Semantics

### Immutability Convention
- `Recipe` is NOT frozen (mutable)
- But convention is method chaining, creating "new" recipes conceptually
- Each step method returns `self` for chaining
- Steps list is modified in place

### Training vs Application
- Both `prep()` and `bake()` respect the `training` parameter
- Some steps skip fitting when `training=False`
- All steps apply transformation regardless

### Template Preservation
- `PreparedRecipe` stores original training data as `template`
- Used by `juice()` method: `bake(self.template)`
- Allows extracting transformed training data without re-fitting

## Code References

| Component | Location | Lines |
|-----------|----------|-------|
| Recipe class | `py_recipes/recipe.py` | 73-108 |
| Recipe.add_step() | `py_recipes/recipe.py` | 109-120 |
| Recipe.prep() | `py_recipes/recipe.py` | 2249-2282 |
| PreparedRecipe class | `py_recipes/recipe.py` | 2285-2305 |
| PreparedRecipe.bake() | `py_recipes/recipe.py` | 2307-2328 |
| PreparedRecipe.juice() | `py_recipes/recipe.py` | 2330-2340 |

## Implications

### What This Means

1. **Order Matters**: User-specified order is execution order
   - No intelligence about step dependencies
   - Same as R tidymodels recipes

2. **No Optimization**: Steps execute regardless of dependency
   - Can't skip redundant steps
   - No DAG (directed acyclic graph) optimization

3. **Deterministic**: Same recipe always produces same output
   - Reproducible preprocessing
   - Consistent between prep() and bake()

4. **No Supervised Step Ordering**: Supervised steps (step_select_shap) don't automatically move before feature creation
   - User must order correctly
   - Feature selection step must come after features are created

### Best Practices

```python
# GOOD: Logical order
rec = (recipe()
    .step_impute_median()           # Handle missing first
    .step_normalize()               # Scale numeric
    .step_dummy()                   # Encode categorical
    .step_filter_anova(outcome='y') # Select important
)

# RISKY: Filter before creation
rec = (recipe()
    .step_filter_anova(outcome='y') # Filter original features
    .step_poly(['x1', 'x2'])        # Create interactions
)  # Poly features created after filtering - likely not in final model!

# GOOD: Create features before filtering
rec = (recipe()
    .step_poly(['x1', 'x2'])        # Create interactions
    .step_filter_anova(outcome='y') # Filter all (including interactions)
)
```

## Testing Verification

Create a test to verify execution order:

```python
def test_execution_order():
    """Verify steps execute in added order"""
    import pandas as pd
    from py_recipes import recipe
    
    # Create recipe with distinct steps
    rec = (recipe()
        .step_mutate({'x_plus_1': lambda df: df['x'] + 1})  # x becomes x+1
        .step_mutate({'x_times_2': lambda df: df['x'] * 2})) # (x+1)*2 = 2x+2
    
    data = pd.DataFrame({'x': [1, 2, 3]})
    fitted = rec.prep(data)
    result = fitted.bake(data)
    
    # If order is correct: (1+1)*2=4, (2+1)*2=6, (3+1)*2=8
    assert all(result['x'] == [4, 6, 8])
```

