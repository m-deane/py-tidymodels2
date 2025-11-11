"""
Demonstration of improved SAFE feature importance calculation.

This script shows how the new LightGBM-based importance calculation
produces non-uniform scores based on actual predictive power, rather
than equal weights for all features from the same variable.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from py_recipes import recipe
from py_recipes.steps.feature_extraction import StepSafe

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic data with known structure
print("Creating synthetic data with known feature importance structure...")
n = 1000

# x1: Strong threshold effect at 50 (creates large y difference)
# x2: Weak relationship with y
# x3: Random noise (no relationship)
x1 = np.random.uniform(0, 100, n)
x2 = np.random.uniform(0, 100, n)
x3 = np.random.uniform(0, 100, n)

# Target: Strong effect from x1 > 50, weak effect from x2, no effect from x3
y = np.where(x1 > 50, 100, 20) + 0.1 * x2 + np.random.normal(0, 5, n)

data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'target': y})

print(f"Data shape: {data.shape}")
print(f"Target range: {y.min():.2f} to {y.max():.2f}")
print()

# Fit surrogate model
print("Fitting surrogate GradientBoostingRegressor...")
surrogate = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
X = data[['x1', 'x2', 'x3']]
surrogate.fit(X, data['target'])

print(f"Surrogate R² score: {surrogate.score(X, y):.4f}")
print()

# Create SAFE step with low penalty to get multiple changepoints
print("Creating SAFE transformation with penalty=1.5...")
step = StepSafe(
    surrogate_model=surrogate,
    outcome='target',
    penalty=1.5,  # Low penalty to get multiple thresholds
    feature_type='dummies'
)

# Prep the step
print("Preparing SAFE step (learning transformations)...")
step.prep(data, training=True)

# Get transformations
transformations = step.get_transformations()

print(f"\nTransformations learned:")
for var_name, info in transformations.items():
    if info['type'] == 'numeric':
        print(f"\n{var_name}:")
        print(f"  Changepoints: {[f'{cp:.2f}' for cp in info['changepoints']]}")
        print(f"  Number of intervals: {len(info['intervals'])}")

# Get feature importances
importances_df = step.get_feature_importances()

print(f"\n\nFeature Importances (Top 20):")
print("=" * 80)

for i, row in importances_df.head(20).iterrows():
    feat = row['feature']
    imp = row['importance']

    # Color code by importance
    if imp > 0.15:
        marker = "***"  # High importance
    elif imp > 0.05:
        marker = "** "  # Medium importance
    elif imp > 0.001:
        marker = "*  "  # Low importance
    else:
        marker = "   "  # Near zero

    print(f"{marker} {feat:50s} {imp:6.4f}")

# Analyze importances by variable
print("\n\nImportance Summary by Variable:")
print("=" * 80)

for var_name in ['x1', 'x2', 'x3']:
    var_features = importances_df[importances_df['feature'].str.startswith(f"{var_name}_")]

    if len(var_features) > 0:
        total_imp = var_features['importance'].sum()
        max_imp = var_features['importance'].max()
        min_imp = var_features['importance'].min()
        mean_imp = var_features['importance'].mean()
        std_imp = var_features['importance'].std()

        print(f"\n{var_name}:")
        print(f"  Number of features: {len(var_features)}")
        print(f"  Total importance:   {total_imp:.4f} (should be ~1.0 per variable)")
        print(f"  Max importance:     {max_imp:.4f}")
        print(f"  Min importance:     {min_imp:.4f}")
        print(f"  Mean importance:    {mean_imp:.4f}")
        print(f"  Std importance:     {std_imp:.4f}")

        # Check if uniform or non-uniform
        if std_imp < 0.01:
            print(f"  Distribution:       UNIFORM (all features ~equal)")
        else:
            print(f"  Distribution:       NON-UNIFORM (varies by predictive power)")

# Transform the data
print("\n\nApplying SAFE transformation to data...")
transformed_data = step.bake(data)

print(f"Transformed data shape: {transformed_data.shape}")
print(f"Number of SAFE features created: {transformed_data.shape[1] - 1}")  # -1 for target
print(f"Original features: {len(['x1', 'x2', 'x3'])}")

# Show sample of transformed data
print("\nSample of transformed features (first 5 rows, first 10 features):")
feature_cols = [col for col in transformed_data.columns if col != 'target']
print(transformed_data[feature_cols[:10]].head())

# Demonstrate top_n selection
print("\n\nDemonstrating top_n feature selection...")
step_top10 = StepSafe(
    surrogate_model=surrogate,
    outcome='target',
    penalty=1.5,
    top_n=10,  # Select only top 10 most important features
    feature_type='dummies'
)

step_top10.prep(data, training=True)
transformed_top10 = step_top10.bake(data)

print(f"With top_n=10:")
print(f"  Original features: {transformed_data.shape[1] - 1}")
print(f"  Selected features: {transformed_top10.shape[1] - 1}")
print(f"  Reduction: {100 * (1 - (transformed_top10.shape[1] - 1) / (transformed_data.shape[1] - 1)):.1f}%")

selected_features = [col for col in transformed_top10.columns if col != 'target']
print(f"\nTop 10 selected features:")
for feat in selected_features:
    imp = importances_df[importances_df['feature'] == feat]['importance'].values[0]
    print(f"  {feat:50s} {imp:6.4f}")

print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)
print("""
The improved SAFE implementation now uses LightGBM to calculate feature importance
based on ACTUAL PREDICTIVE POWER, not uniform distribution.

Key improvements:
1. Features with strong predictive power get higher importance scores
2. Uninformative thresholds get low/zero importance
3. top_n parameter now selects the MOST PREDICTIVE features
4. Importances sum to 1.0 within each variable group (normalization)
5. Graceful fallback to uniform distribution if LightGBM unavailable

Expected behavior in this example:
- x1 thresholds near 50 should have HIGH importance (strong effect)
- x2 thresholds should have LOW importance (weak effect)
- x3 thresholds should have NEAR-ZERO importance (random noise)
""")
