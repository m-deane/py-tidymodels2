# StepSafeV2 Interaction Features Added

**Date:** 2025-11-10
**Status:** ✅ COMPLETE

## Issue

User reported: "step_safe_v2 with the argument 'feature_type="both"' does not appear to be creating interactions"

### Root Cause

**Parameter Semantic Mismatch:**

The old `StepSafe` and new `StepSafeV2` both had a parameter called `feature_type`, but with **completely different meanings**:

**Old StepSafe (`feature_type`):**
- Controls **output type** (what features to create)
- Values: `'dummies'`, `'interactions'`, `'both'`
- `'dummies'`: Binary threshold indicators only
- `'interactions'`: Dummy × original value interactions only
- `'both'`: Both dummies AND interactions

**New StepSafeV2 (`feature_type`):**
- Controls **input type** (which variable types to process)
- Values: `'numeric'`, `'categorical'`, `'both'`
- `'numeric'`: Process only numeric variables
- `'categorical'`: Process only categorical variables
- `'both'`: Process both numeric AND categorical variables

**The Problem:**
StepSafeV2 had NO mechanism to create interaction features at all. It only created dummy variables (binary threshold indicators for numeric, one-hot encoding for categorical).

## Solution

Added a new parameter `output_mode` to StepSafeV2 to control feature output type, matching the old StepSafe's `feature_type` semantics:

**New Parameter: `output_mode`**
- Default: `'dummies'` (maintains backward compatibility)
- Values: `'dummies'`, `'interactions'`, `'both'`
- `'dummies'`: Binary dummy variables only
- `'interactions'`: Binary dummy × original feature interactions only
- `'both'`: Both dummies and interactions

Now StepSafeV2 has two separate controls:
1. **`feature_type`**: Which variable types to process (numeric/categorical/both)
2. **`output_mode`**: What features to create (dummies/interactions/both)

## Implementation Details

### Files Modified

1. **`py_recipes/steps/feature_extraction.py`**
   - Added `output_mode` parameter to `StepSafeV2` dataclass (line 1100)
   - Added validation in `__post_init__()` (lines 1140-1144)
   - Updated `_transform_numeric_variable()` to create interactions (lines 1731-1766)
   - Updated `_transform_categorical_variable()` to create interactions (lines 1768-1845)
   - Updated docstring (lines 1068-1072)

2. **`py_recipes/recipe.py`**
   - Added `output_mode` parameter to `step_safe_v2()` helper function (line 1105)
   - Updated docstring (line 1131)
   - Passed `output_mode` to StepSafeV2 constructor (line 1177)

### Code Changes

**StepSafeV2 Class Definition:**
```python
@dataclass
class StepSafeV2:
    surrogate_model: Any
    outcome: str
    penalty: float = 10.0
    top_n: Optional[int] = None
    max_thresholds: int = 5
    keep_original_cols: bool = True
    grid_resolution: int = 100
    feature_type: Literal['numeric', 'categorical', 'both'] = 'both'
    output_mode: Literal['dummies', 'interactions', 'both'] = 'dummies'  # NEW!
    columns: Union[None, str, List[str], Callable] = None
    skip: bool = False
    id: Optional[str] = None
```

**Numeric Variable Transformation with Interactions:**
```python
def _transform_numeric_variable(
    self, var: Dict[str, Any], X_col: pd.Series
) -> Optional[pd.DataFrame]:
    """Transform numeric variable into binary threshold indicators and/or interactions."""
    if not var.get('thresholds') or not var['new_names']:
        return None

    thresholds = var['thresholds']
    new_names = var['new_names']

    # Create binary indicators (feature > threshold)
    dummies_df = pd.DataFrame()
    for threshold_val, feat_name in zip(thresholds, new_names):
        dummies_df[feat_name] = (X_col > threshold_val).astype(int)

    # Handle output_mode
    if self.output_mode == 'dummies':
        return dummies_df
    elif self.output_mode == 'interactions':
        # Create interactions: dummy * original_value
        interactions_df = pd.DataFrame()
        original_values = X_col.values
        for col in dummies_df.columns:
            interaction_name = f"{col}_x_{var['original_name']}"
            interaction_name = self._sanitize_feature_name(interaction_name)
            interactions_df[interaction_name] = dummies_df[col] * original_values
        return interactions_df
    else:  # 'both'
        # Return both dummies and interactions
        result_df = dummies_df.copy()
        original_values = X_col.values
        for col in dummies_df.columns:
            interaction_name = f"{col}_x_{var['original_name']}"
            interaction_name = self._sanitize_feature_name(interaction_name)
            result_df[interaction_name] = dummies_df[col] * original_values
        return result_df
```

**Categorical Variable Transformation with Interactions:**
```python
def _transform_categorical_variable(
    self, var: Dict[str, Any], X_col: pd.Series
) -> Optional[pd.DataFrame]:
    """Transform categorical variable based on clusters and/or interactions."""

    # ... create dummies_df (either simple or cluster-based) ...

    # Handle output_mode
    if self.output_mode == 'dummies':
        return dummies_df
    elif self.output_mode == 'interactions':
        # For categorical, use label encoding for interactions
        label_encoded = pd.factorize(X_col)[0]
        interactions_df = pd.DataFrame()
        for col in dummies_df.columns:
            interaction_name = f"{col}_x_{var['original_name']}"
            interaction_name = self._sanitize_feature_name(interaction_name)
            interactions_df[interaction_name] = dummies_df[col] * label_encoded
        return interactions_df
    else:  # 'both'
        result_df = dummies_df.copy()
        label_encoded = pd.factorize(X_col)[0]
        for col in dummies_df.columns:
            interaction_name = f"{col}_x_{var['original_name']}"
            interaction_name = self._sanitize_feature_name(interaction_name)
            result_df[interaction_name] = dummies_df[col] * label_encoded
        return result_df
```

## Test Results

**Test Script:** `test_safe_v2_interactions.py`

```
=== Testing StepSafeV2 interaction feature creation ===

Test 1: output_mode='dummies' (default)
--------------------------------------------------
Columns created: 5
Sample columns: ['x1_gt_0_13', 'x2_gt_0_98', 'cat1_B', 'cat1_C', 'target']
Has interactions (e.g., '_x_'): False

Test 2: output_mode='interactions'
--------------------------------------------------
Columns created: 5
Sample columns: ['x1_gt_0_13_x_x1', 'x2_gt_0_98_x_x2', 'cat1_B_x_cat1', 'cat1_C_x_cat1', 'target']
Has interactions (e.g., '_x_'): True

Test 3: output_mode='both' (USER'S REQUEST)
--------------------------------------------------
Columns created: 9
Sample columns (first 10): ['x1_gt_0_13', 'x1_gt_0_13_x_x1', 'x2_gt_0_98', 'x2_gt_0_98_x_x2', 'cat1_B', 'cat1_C', 'cat1_B_x_cat1', 'cat1_C_x_cat1', 'target']
Has interactions (e.g., '_x_'): True

Breakdown:
  Dummy features: 4
  Interaction features: 4
  Total (excluding target): 8

Example interaction feature: x1_gt_0_13_x_x1
First 5 values: [0.4967141530112327, -0.0, 0.6476885381006925, 1.5230298564080254, -0.0]

=== VERIFICATION ===
--------------------------------------------------
✓ output_mode='dummies' correctly excludes interactions
✓ output_mode='interactions' correctly creates only interactions
✓ output_mode='both' correctly creates both dummies and interactions
✓ output_mode='both' creates ~2x features (1.8x)

✅ ALL TESTS PASSED - Interaction features working correctly!
```

## Usage Examples

### Example 1: Dummies Only (Default)
```python
from sklearn.ensemble import GradientBoostingRegressor
from py_recipes import recipe

surrogate = GradientBoostingRegressor(n_estimators=100)

rec = recipe().step_safe_v2(
    surrogate_model=surrogate,
    outcome='target',
    feature_type='both',      # Process numeric AND categorical
    output_mode='dummies'      # Create only dummies (DEFAULT)
)

# Result: Creates binary indicators like:
#   x1_gt_5, x1_gt_10, cat1_B, cat1_C
```

### Example 2: Interactions Only
```python
rec = recipe().step_safe_v2(
    surrogate_model=surrogate,
    outcome='target',
    feature_type='both',
    output_mode='interactions'  # Create only interactions
)

# Result: Creates interaction features like:
#   x1_gt_5_x_x1 = (x1 > 5) * x1
#   cat1_B_x_cat1 = (cat1 == 'B') * label_encode(cat1)
```

### Example 3: Both Dummies and Interactions (User's Request)
```python
rec = recipe().step_safe_v2(
    surrogate_model=surrogate,
    outcome='target',
    feature_type='both',
    output_mode='both'  # Create BOTH dummies AND interactions
)

# Result: Creates both types:
#   Dummies: x1_gt_5, x1_gt_10, cat1_B, cat1_C
#   Interactions: x1_gt_5_x_x1, x1_gt_10_x_x1, cat1_B_x_cat1, cat1_C_x_cat1
```

## Interaction Feature Semantics

### Numeric Variables
**Interaction = Dummy × Original Value**

Example for threshold x1 > 5:
- Dummy: `x1_gt_5 = (x1 > 5).astype(int)`
- Interaction: `x1_gt_5_x_x1 = (x1 > 5) * x1`

| x1 | x1_gt_5 (dummy) | x1_gt_5_x_x1 (interaction) |
|----|-----------------|---------------------------|
| 3  | 0               | 0                         |
| 7  | 1               | 7                         |
| 10 | 1               | 10                        |

**Why Useful:** Captures both the threshold effect AND the magnitude of the original value.

### Categorical Variables
**Interaction = Dummy × Label Encoded**

Example for categorical variable `color`:
- Dummies: `color_red`, `color_blue` (one-hot, drop_first)
- Label encoding: A=0, B=1, C=2
- Interaction: `color_red_x_color = color_red * label_encode(color)`

| color | color_red (dummy) | label_encode | color_red_x_color (interaction) |
|-------|-------------------|--------------|--------------------------------|
| A     | 0                 | 0            | 0                              |
| B     | 1                 | 1            | 1                              |
| C     | 0                 | 2            | 0                              |

**Why Useful:** Captures interaction between categorical membership and ordinal encoding.

## Backward Compatibility

**Default Behavior Unchanged:**
- `output_mode` defaults to `'dummies'`
- Existing code without `output_mode` parameter continues to work
- Only creates dummy variables (same as before the fix)

**To Get Interactions:**
- User must explicitly set `output_mode='interactions'` or `output_mode='both'`

## Related Documentation

- Original implementation: `.claude_debugging/RECIPE_STEPS_UNFITTED_MODELS_COMPLETE.md`
- Session fixes: `.claude_debugging/SESSION_FIXES_2025_11_10.md`
- Old StepSafe implementation: `py_recipes/steps/feature_extraction.py:lines-892-910` (numeric), `lines-920-937,966-982` (categorical)

## Conclusion

**Status:** ✅ COMPLETE

StepSafeV2 now supports interaction feature creation matching the functionality of the old StepSafe. User can now:

1. Set `output_mode='both'` to get both dummies and interactions
2. Set `output_mode='interactions'` to get only interactions
3. Set `output_mode='dummies'` (default) to get only dummies

The parameter naming is now clearer:
- `feature_type`: Controls which variable types to process (input selection)
- `output_mode`: Controls what features to create (output transformation)

All tests passing. Ready for production use.
