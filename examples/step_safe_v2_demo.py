"""
Demo: SAFE v2 - Surrogate Assisted Feature Extraction with UNFITTED model

This example demonstrates the new step_safe_v2() which:
1. Accepts UNFITTED surrogate model (fitted during prep())
2. Adds max_thresholds parameter to control threshold quantity
3. Sanitizes feature names for LightGBM compatibility
4. Recalculates importances on TRANSFORMED features
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from py_recipes import recipe

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data with nonlinear relationships
n = 500
data = pd.DataFrame({
    'x1': np.random.uniform(0, 10, n),
    'x2': np.random.uniform(-5, 5, n),
    'x3': np.random.uniform(0, 100, n),
    'x4': np.random.uniform(-10, 10, n),
    'cat1': np.random.choice(['A', 'B', 'C'], n),
    'cat2': np.random.choice(['Low', 'Medium', 'High'], n),
})

# Create outcome with complex nonlinear relationships
data['y'] = (
    2 * data['x1'] +
    5 * np.sin(data['x2']) +
    0.1 * data['x3'] ** 0.5 +
    np.where(data['x4'] > 0, 3 * data['x4'], -2 * data['x4']) +
    (data['cat1'] == 'A') * 5 +
    (data['cat2'] == 'High') * 3 +
    np.random.normal(0, 2, n)
)

# Split data
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

print("=" * 80)
print("SAFE v2 Demo: Surrogate Assisted Feature Extraction with UNFITTED Model")
print("=" * 80)
print(f"\nTraining data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"\nOriginal features: {list(data.columns[:-1])}")

# =============================================================================
# Example 1: Basic Usage - UNFITTED GradientBoosting surrogate
# =============================================================================
print("\n" + "=" * 80)
print("Example 1: Basic SAFE v2 with GradientBoosting Surrogate")
print("=" * 80)

# Create UNFITTED surrogate model
surrogate_gb = GradientBoostingRegressor(
    n_estimators=50,
    max_depth=5,
    random_state=42
)

# Create recipe with SAFE v2
# NOTE: Model is UNFITTED - it will be fitted during prep()
rec_v2 = recipe().step_safe_v2(
    surrogate_model=surrogate_gb,
    outcome='y',
    penalty=10.0,
    max_thresholds=5,
    keep_original_cols=False,
    grid_resolution=100
)

# Reset index before prep/bake to avoid issues
train_data_clean = train_data.reset_index(drop=True)
test_data_clean = test_data.reset_index(drop=True)

# Prep recipe (model fitted here)
prepped_v2 = rec_v2.prep(train_data_clean)

# Transform data
train_transformed = prepped_v2.bake(train_data_clean)
test_transformed = prepped_v2.bake(test_data_clean)

print(f"\nTransformed training data shape: {train_transformed.shape}")
print(f"Number of features created: {train_transformed.shape[1] - 1}")  # Exclude outcome
print(f"\nSample of transformed features:")
print(train_transformed.head())

# Get feature importances
step = prepped_v2.prepared_steps[0]
importances = step.get_feature_importances()
print(f"\nTop 10 most important transformed features:")
print(importances.head(10))

# =============================================================================
# Example 2: Feature Selection with top_n
# =============================================================================
print("\n" + "=" * 80)
print("Example 2: SAFE v2 with Feature Selection (top_n=10)")
print("=" * 80)

# Create UNFITTED surrogate
surrogate_rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=5,
    random_state=42
)

# Recipe with top_n selection
rec_top_n = recipe().step_safe_v2(
    surrogate_model=surrogate_rf,
    outcome='y',
    penalty=10.0,
    max_thresholds=5,
    top_n=10,  # Select only top 10 most important features
    keep_original_cols=False
)

prepped_top_n = rec_top_n.prep(train_data_clean)
train_top_n = prepped_top_n.bake(train_data_clean)

print(f"\nOriginal features: {train_data.shape[1] - 1}")
print(f"Selected features: {train_top_n.shape[1] - 1}")
print(f"\nSelected feature names:")
print([col for col in train_top_n.columns if col != 'y'])

# =============================================================================
# Example 3: Feature Type Selection
# =============================================================================
print("\n" + "=" * 80)
print("Example 3: SAFE v2 with Feature Type Control")
print("=" * 80)

# Only numeric features
surrogate_num = GradientBoostingRegressor(n_estimators=30, random_state=42)
rec_numeric = recipe().step_safe_v2(
    surrogate_model=surrogate_num,
    outcome='y',
    penalty=10.0,
    max_thresholds=3,
    feature_type='numeric',  # Only transform numeric features
    keep_original_cols=False
)

prepped_numeric = rec_numeric.prep(train_data_clean)
train_numeric = prepped_numeric.bake(train_data_clean)

print(f"\nNumeric features only:")
print(f"Transformed features: {train_numeric.shape[1] - 1}")

# Only categorical features
surrogate_cat = GradientBoostingRegressor(n_estimators=30, random_state=42)
rec_categorical = recipe().step_safe_v2(
    surrogate_model=surrogate_cat,
    outcome='y',
    penalty=10.0,
    feature_type='categorical',  # Only transform categorical features
    keep_original_cols=False
)

prepped_categorical = rec_categorical.prep(train_data_clean)
train_categorical = prepped_categorical.bake(train_data_clean)

print(f"\nCategorical features only:")
print(f"Transformed features: {train_categorical.shape[1] - 1}")

# =============================================================================
# Example 4: Compare Simple vs SAFE-transformed Model Performance
# =============================================================================
print("\n" + "=" * 80)
print("Example 4: Model Performance Comparison")
print("=" * 80)

from sklearn.metrics import mean_squared_error, r2_score

# Simple linear model on original features
X_train_orig = train_data.drop('y', axis=1)
X_test_orig = test_data.drop('y', axis=1)
y_train = train_data['y']
y_test = test_data['y']

# One-hot encode categoricals for simple model
X_train_simple = pd.get_dummies(X_train_orig, drop_first=True)
X_test_simple = pd.get_dummies(X_test_orig, drop_first=True)

# Align test columns with train
for col in X_train_simple.columns:
    if col not in X_test_simple.columns:
        X_test_simple[col] = 0

X_test_simple = X_test_simple[X_train_simple.columns]

lr_simple = LinearRegression()
lr_simple.fit(X_train_simple, y_train)
y_pred_simple = lr_simple.predict(X_test_simple)

rmse_simple = np.sqrt(mean_squared_error(y_test, y_pred_simple))
r2_simple = r2_score(y_test, y_pred_simple)

print(f"\nSimple Linear Model (original features):")
print(f"  Test RMSE: {rmse_simple:.4f}")
print(f"  Test R²: {r2_simple:.4f}")

# Linear model on SAFE-transformed features
X_train_safe = train_transformed.drop('y', axis=1)
X_test_safe = test_transformed.drop('y', axis=1)

lr_safe = LinearRegression()
lr_safe.fit(X_train_safe, y_train)
y_pred_safe = lr_safe.predict(X_test_safe)

rmse_safe = np.sqrt(mean_squared_error(y_test, y_pred_safe))
r2_safe = r2_score(y_test, y_pred_safe)

print(f"\nLinear Model with SAFE v2 Features:")
print(f"  Test RMSE: {rmse_safe:.4f}")
print(f"  Test R²: {r2_safe:.4f}")

improvement = ((rmse_simple - rmse_safe) / rmse_simple) * 100
print(f"\nImprovement: {improvement:.1f}% reduction in RMSE")

# =============================================================================
# Example 5: Feature Name Sanitization for LightGBM
# =============================================================================
print("\n" + "=" * 80)
print("Example 5: Feature Name Sanitization")
print("=" * 80)

# Create data with special characters in column names
data_special = train_data_clean.copy()
data_special['x.special'] = data_special['x1']
data_special['x-dash'] = data_special['x2']
data_special['x (paren)'] = data_special['x3']

surrogate_special = GradientBoostingRegressor(n_estimators=30, random_state=42)
rec_special = recipe().step_safe_v2(
    surrogate_model=surrogate_special,
    outcome='y',
    penalty=10.0,
    max_thresholds=3,
    keep_original_cols=False
)

prepped_special = rec_special.prep(data_special)
transformed_special = prepped_special.bake(data_special)

print(f"\nOriginal column names with special characters:")
print([c for c in data_special.columns if c not in train_data_clean.columns])

print(f"\nAll transformed column names are LightGBM-compatible:")
print(f"(Only alphanumeric and underscore)")
safe_names = all(
    all(c.isalnum() or c == '_' for c in col)
    for col in transformed_special.columns if col != 'y'
)
print(f"Validation: {safe_names}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("Summary: SAFE v2 Key Features")
print("=" * 80)
print("""
1. UNFITTED Model Support:
   - Pass unfitted sklearn-compatible model
   - Model fitted during prep() using recipe data
   - No need to pre-fit on training set

2. Threshold Control:
   - max_thresholds parameter limits thresholds per feature
   - Prevents feature explosion with high-variance data
   - Selects most important thresholds via PDP jumps

3. Feature Name Sanitization:
   - Automatic regex-based sanitization
   - Compatible with LightGBM and other strict libraries
   - Removes special characters, consecutive underscores

4. Transformed Feature Importances:
   - Uses LightGBM on TRANSFORMED features (not originals)
   - More accurate importance ranking
   - Enables better top_n selection

5. Flexible Feature Selection:
   - feature_type: 'numeric', 'categorical', or 'both'
   - top_n: Select N most important transformed features
   - columns: Target specific columns for transformation
""")

print("\n" + "=" * 80)
print("Demo Complete!")
print("=" * 80)
