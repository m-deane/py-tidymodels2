"""
SHAP Interpretability Demo

Demonstrates SHAP interpretability features with py-tidymodels.
Shows how to compute and analyze SHAP values for model explanations.
"""

import sys
import os

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from py_parsnip import linear_reg, rand_forest
from py_workflows import workflow
from py_recipes import recipe
from py_interpret import ShapEngine

# Set random seed for reproducibility
np.random.seed(42)

# Create sample regression data
n = 200
X1 = np.random.randn(n)
X2 = np.random.randn(n)
X3 = np.random.randn(n)
X4 = np.random.randn(n)
# True model: y = 2*X1 + 3*X2 - 1.5*X3 + noise (X4 is irrelevant)
y = 2 * X1 + 3 * X2 - 1.5 * X3 + np.random.randn(n) * 0.5

data = pd.DataFrame({
    'y': y,
    'X1': X1,
    'X2': X2,
    'X3': X3,
    'X4': X4  # Irrelevant variable
})

# Split into train/test
train = data.iloc[:150]
test = data.iloc[150:]

print("=" * 80)
print("SHAP Interpretability Demo")
print("=" * 80)

# Example 1: Basic SHAP with Linear Regression
print("\n1. Basic SHAP with Linear Regression")
print("-" * 40)

spec = linear_reg()
fit = spec.fit(train, 'y ~ X1 + X2 + X3 + X4')

# Compute SHAP values (auto-selects LinearExplainer)
shap_df = fit.explain(test, check_additivity=False)

print(f"SHAP DataFrame shape: {shap_df.shape}")
print(f"Columns: {shap_df.columns.tolist()}")
print(f"\nFirst 10 rows:")
print(shap_df.head(10))

# Global feature importance
print("\nGlobal Feature Importance (mean |SHAP|):")
importance = shap_df.groupby('variable')['abs_shap'].mean().sort_values(ascending=False)
for var, imp in importance.items():
    print(f"  {var}: {imp:.4f}")

# Example 2: SHAP with Random Forest
print("\n2. SHAP with Random Forest (TreeExplainer)")
print("-" * 40)

spec_rf = rand_forest(trees=50).set_mode('regression')
fit_rf = spec_rf.fit(train, 'y ~ X1 + X2 + X3 + X4')

# Compute SHAP (auto-selects TreeExplainer - fast!)
shap_rf = fit_rf.explain(test, check_additivity=False)

print("Feature Importance (Random Forest):")
importance_rf = shap_rf.groupby('variable')['abs_shap'].mean().sort_values(ascending=False)
for var, imp in importance_rf.items():
    print(f"  {var}: {imp:.4f}")

# Example 3: SHAP with Workflow and Recipe
print("\n3. SHAP with Workflow and Recipe")
print("-" * 40)

rec = recipe().step_normalize()
wf = workflow().add_recipe(rec).add_model(linear_reg())
wf_fit = wf.fit(train)

# SHAP on normalized features
shap_wf = wf_fit.explain(test, check_additivity=False)

print("Feature Importance (with normalization):")
importance_wf = shap_wf.groupby('variable')['abs_shap'].mean().sort_values(ascending=False)
for var, imp in importance_wf.items():
    print(f"  {var}: {imp:.4f}")

# Example 4: SHAP for Single Observation
print("\n4. SHAP for Single Observation (Waterfall-style)")
print("-" * 40)

single_obs = test.iloc[:1]
shap_single = fit.explain(single_obs, check_additivity=False)

print(f"Observation 0 prediction: {shap_single['prediction'].iloc[0]:.4f}")
print(f"Base value: {shap_single['base_value'].iloc[0]:.4f}")
print("\nFeature contributions:")
for _, row in shap_single.iterrows():
    sign = "+" if row['shap_value'] >= 0 else ""
    print(f"  {row['variable']}: {sign}{row['shap_value']:.4f} (value={row['feature_value']:.4f})")

# Verify additivity
total_shap = shap_single['shap_value'].sum()
base = shap_single['base_value'].iloc[0]
pred = shap_single['prediction'].iloc[0]
print(f"\nAdditivity check: sum(SHAP) + base = {total_shap + base:.4f}")
print(f"                  prediction        = {pred:.4f}")
print(f"                  difference        = {abs((total_shap + base) - pred):.6f}")

# Example 5: Grouped Model SHAP
print("\n5. SHAP for Grouped Models")
print("-" * 40)

# Create grouped data
grouped_data = []
for group in ['A', 'B']:
    n_group = 75
    X1_g = np.random.randn(n_group)
    X2_g = np.random.randn(n_group)
    X3_g = np.random.randn(n_group)

    # Different coefficients per group
    if group == 'A':
        y_g = 2 * X1_g + 3 * X2_g - 1.5 * X3_g + np.random.randn(n_group) * 0.5
    else:  # B
        y_g = 1 * X1_g + 4 * X2_g - 2 * X3_g + np.random.randn(n_group) * 0.5

    df = pd.DataFrame({
        'group_id': group,
        'y': y_g,
        'X1': X1_g,
        'X2': X2_g,
        'X3': X3_g
    })
    grouped_data.append(df)

grouped_df = pd.concat(grouped_data, ignore_index=True)
train_grouped = grouped_df.iloc[:100]
test_grouped = grouped_df.iloc[100:]

# Fit nested models
spec_nested = linear_reg()
nested_fit = spec_nested.fit_nested(train_grouped, 'y ~ X1 + X2 + X3', group_col='group_id')

# Compute SHAP per group
shap_grouped = nested_fit.explain(test_grouped, check_additivity=False)

print("Feature Importance by Group:")
importance_by_group = shap_grouped.groupby(['group', 'variable'])['abs_shap'].mean().unstack()
print(importance_by_group)

print("\nKey observations:")
print("  - Group A: X2 most important (true coef = 3.0)")
print("  - Group B: X2 most important (true coef = 4.0)")
print("  - X3 has higher importance in Group B (true coef = -2.0 vs -1.5)")

print("\n" + "=" * 80)
print("Demo complete! SHAP values provide model-agnostic feature importance.")
print("=" * 80)
