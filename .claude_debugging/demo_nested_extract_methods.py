"""
Demonstration of new NestedWorkflowFit extract methods.

Shows how to use extract_formula(), extract_spec_parsnip(),
extract_preprocessor(), and extract_fit_parsnip().
"""

import pandas as pd
import numpy as np
from py_workflows import workflow
from py_recipes import recipe
from py_parsnip import linear_reg

# Create sample grouped data
np.random.seed(42)
train_data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=100),
    'country': ['USA'] * 50 + ['UK'] * 50,
    'x1': np.random.randn(100) * 10 + 50,
    'x2': np.random.randn(100) * 5 + 20,
    'x3': np.random.randn(100) * 3 + 10,
    'sales': np.random.randn(100) * 15 + 100
})

test_data = pd.DataFrame({
    'date': pd.date_range('2020-04-10', periods=20),
    'country': ['USA'] * 10 + ['UK'] * 10,
    'x1': np.random.randn(20) * 10 + 50,
    'x2': np.random.randn(20) * 5 + 20,
    'x3': np.random.randn(20) * 3 + 10,
    'sales': np.random.randn(20) * 15 + 100
})

print("=" * 70)
print("DEMONSTRATION: NestedWorkflowFit Extract Methods")
print("=" * 70)

# ============================================================================
# Example 1: Extract Formula (Key Feature!)
# ============================================================================
print("\n" + "=" * 70)
print("Example 1: extract_formula() - Get formulas for all groups")
print("=" * 70)

wf = workflow().add_formula("sales ~ x1 + x2 + x3").add_model(linear_reg())
nested_fit = wf.fit_nested(train_data, group_col='country')

formulas = nested_fit.extract_formula()
print(f"\nFormulas by group:")
for group, formula in formulas.items():
    print(f"  {group}: {formula}")

# Check if all groups use same formula
if len(set(formulas.values())) == 1:
    print(f"\n✓ All groups use same formula: {list(formulas.values())[0]}")

# ============================================================================
# Example 2: Extract Spec
# ============================================================================
print("\n" + "=" * 70)
print("Example 2: extract_spec_parsnip() - Get shared model spec")
print("=" * 70)

spec = nested_fit.extract_spec_parsnip()
print(f"\nModel specification:")
print(f"  Type: {spec.model_type}")
print(f"  Engine: {spec.engine}")
print(f"  Mode: {spec.mode}")

# ============================================================================
# Example 3: Extract Preprocessor (All Groups)
# ============================================================================
print("\n" + "=" * 70)
print("Example 3: extract_preprocessor() - Get all preprocessors")
print("=" * 70)

preprocessors = nested_fit.extract_preprocessor()
print(f"\nPreprocessors by group:")
for group, prep in preprocessors.items():
    print(f"  {group}: {type(prep).__name__} = '{prep}'")

# ============================================================================
# Example 4: Extract Preprocessor (Single Group)
# ============================================================================
print("\n" + "=" * 70)
print("Example 4: extract_preprocessor(group='USA') - Get specific group")
print("=" * 70)

usa_preprocessor = nested_fit.extract_preprocessor(group='USA')
print(f"\nUSA preprocessor: {usa_preprocessor}")

# ============================================================================
# Example 5: Extract Fit (Single Group for Deep Inspection)
# ============================================================================
print("\n" + "=" * 70)
print("Example 5: extract_fit_parsnip(group='USA') - Deep dive into USA")
print("=" * 70)

# Evaluate first
nested_fit = nested_fit.evaluate(test_data)

usa_fit = nested_fit.extract_fit_parsnip(group='USA')
print(f"\nUSA ModelFit:")
print(f"  Model type: {usa_fit.spec.model_type}")
print(f"  Engine: {usa_fit.spec.engine}")

# Get detailed outputs for USA
outputs, coeffs, stats = usa_fit.extract_outputs()
print(f"\n  Performance:")
train_stats = stats[(stats['split']=='train') & (stats['metric']=='rmse')]
test_stats = stats[(stats['split']=='test') & (stats['metric']=='rmse')]
if not train_stats.empty:
    print(f"    Train RMSE: {train_stats['value'].values[0]:.2f}")
if not test_stats.empty:
    print(f"    Test RMSE: {test_stats['value'].values[0]:.2f}")

print(f"\n  Coefficients:")
for _, row in coeffs.head(5).iterrows():
    print(f"    {row['variable']}: {row['coefficient']:.3f}")

# ============================================================================
# Example 6: Extract All Fits and Compare
# ============================================================================
print("\n" + "=" * 70)
print("Example 6: extract_fit_parsnip() - Compare all groups")
print("=" * 70)

all_fits = nested_fit.extract_fit_parsnip()
print(f"\nPerformance comparison:")
for group, fit in sorted(all_fits.items()):
    _, _, stats = fit.extract_outputs()
    test_rmse_row = stats[(stats['split']=='test') & (stats['metric']=='rmse')]
    test_r2_row = stats[(stats['split']=='test') & (stats['metric']=='r_squared')]
    print(f"  {group}:")
    if not test_rmse_row.empty:
        print(f"    Test RMSE: {test_rmse_row['value'].values[0]:.2f}")
    if not test_r2_row.empty:
        print(f"    Test R²: {test_r2_row['value'].values[0]:.3f}")

# ============================================================================
# Example 7: With Recipe (Auto-Generated Formula)
# ============================================================================
print("\n" + "=" * 70)
print("Example 7: extract_formula() with recipe (auto-generated)")
print("=" * 70)

rec = recipe().step_normalize()
wf_recipe = workflow().add_recipe(rec).add_model(linear_reg())
nested_fit_recipe = wf_recipe.fit_nested(train_data, group_col='country')

formulas_recipe = nested_fit_recipe.extract_formula()
print(f"\nAuto-generated formulas:")
for group, formula in formulas_recipe.items():
    print(f"  {group}: {formula}")
    # Note: Should not include 'date' or 'country'

# ============================================================================
# Example 8: Systematic Group Analysis
# ============================================================================
print("\n" + "=" * 70)
print("Example 8: Systematic analysis of all group components")
print("=" * 70)

# Get all components
formulas = nested_fit.extract_formula()
preprocessors = nested_fit.extract_preprocessor()
spec = nested_fit.extract_spec_parsnip()
model_fits = nested_fit.extract_fit_parsnip()

print(f"\nComplete group analysis:")
print(f"  Shared model type: {spec.model_type}")
print(f"  Shared engine: {spec.engine}")
print(f"\n  Group-specific components:")

for group in sorted(formulas.keys()):
    print(f"\n  {group}:")
    print(f"    Formula: {formulas[group]}")
    print(f"    Preprocessor: {type(preprocessors[group]).__name__}")

    # Performance
    fit = model_fits[group]
    _, _, stats = fit.extract_outputs()
    test_rmse_row = stats[(stats['split']=='test') & (stats['metric']=='rmse')]
    if not test_rmse_row.empty:
        print(f"    Test RMSE: {test_rmse_row['value'].values[0]:.2f}")

print("\n" + "=" * 70)
print("✓ All examples completed successfully!")
print("=" * 70)
