"""
Example: Using Recipes vs Formulas in Workflows

This demonstrates the difference between formula-based and recipe-based workflows,
showing how recipes enable powerful feature engineering pipelines.
"""

import pandas as pd
import numpy as np
from py_workflows import workflow
from py_parsnip import linear_reg
from py_recipes import recipe
from py_recipes.selectors import all_numeric, all_nominal, all_predictors
from py_yardstick import rmse, mae, r_squared

# Create sample time series data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=100, freq='M')
df = pd.DataFrame({
    'date': dates,
    'target': np.random.randn(100).cumsum() + 100,
    'lag1': np.random.randn(100),
    'lag2': np.random.randn(100),
    'lag3': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'has_promo': np.random.choice([0, 1], 100)
})

train = df.iloc[:80]
test = df.iloc[80:]

print("="*70)
print("APPROACH 1: Formula-Based Workflow (Simple)")
print("="*70)

# Simple formula-based workflow
# Limitations: Can't easily normalize, encode categoricals, etc.
wf_formula = (
    workflow()
    .add_formula("target ~ lag1 + lag2 + lag3")
    .add_model(linear_reg())
)

fit_formula = wf_formula.fit(train)
fit_formula = fit_formula.evaluate(test)

outputs_formula, coefs_formula, stats_formula = fit_formula.extract_outputs()

print("\nFormula approach:")
print(f"  Formula: target ~ lag1 + lag2 + lag3")
print(f"  Features used: lag1, lag2, lag3 (raw values)")
print(f"  Test RMSE: {stats_formula[stats_formula['metric'] == 'rmse']['value'].iloc[1]:.4f}")

print("\n" + "="*70)
print("APPROACH 2: Recipe-Based Workflow (Powerful)")
print("="*70)

# Recipe-based workflow with advanced feature engineering
# Advantages: Normalization, encoding, imputation, feature creation, etc.

rec = (
    recipe(train, "target ~ .")  # Use all columns as predictors
    .step_rm("date")  # Remove date column from predictors
    .step_normalize(all_numeric())  # Normalize numeric features (z-score)
    .step_dummy(all_nominal())  # One-hot encode categorical variables
    # Could add more steps:
    # .step_impute_median(all_numeric())  # Handle missing values
    # .step_interact(["lag1", "lag2"])  # Create interaction terms
    # .step_pca(all_numeric(), num_comp=2)  # Dimensionality reduction
)

wf_recipe = (
    workflow()
    .add_recipe(rec)
    .add_model(linear_reg())
)

fit_recipe = wf_recipe.fit(train)
fit_recipe = fit_recipe.evaluate(test)

outputs_recipe, coefs_recipe, stats_recipe = fit_recipe.extract_outputs()

print("\nRecipe approach:")
print("  Steps:")
print("    1. Start with all columns: target ~ .")
print("    2. Remove date column")
print("    3. Normalize all numeric columns (z-score)")
print("    4. One-hot encode categorical columns")
print(f"  Features used: lag1, lag2, lag3, category_B, category_C, has_promo (normalized + encoded)")
print(f"  Test RMSE: {stats_recipe[stats_recipe['metric'] == 'rmse']['value'].iloc[1]:.4f}")

print("\n" + "="*70)
print("APPROACH 3: Recipe with Custom Feature Engineering")
print("="*70)

# More advanced recipe with feature engineering
rec_advanced = (
    recipe(train, "target ~ .")
    .step_rm("date")  # Remove date
    # Create polynomial features
    .step_poly("lag1", degree=2)  # Add lag1^2
    # Create interaction
    .step_interact(["lag1", "lag2"])  # Add lag1*lag2
    # Normalize everything
    .step_normalize(all_numeric())
    # Encode categoricals
    .step_dummy(all_nominal())
    # Log transform (would fail on negative values, so commented)
    # .step_log(["lag1"])
)

wf_advanced = (
    workflow()
    .add_recipe(rec_advanced)
    .add_model(linear_reg())
)

fit_advanced = wf_advanced.fit(train)
fit_advanced = fit_advanced.evaluate(test)

outputs_advanced, coefs_advanced, stats_advanced = fit_advanced.extract_outputs()

print("\nAdvanced recipe approach:")
print("  Steps:")
print("    1. Remove date column")
print("    2. Create polynomial features: lag1^2")
print("    3. Create interactions: lag1*lag2")
print("    4. Normalize all numeric features")
print("    5. One-hot encode categoricals")
print(f"  Features engineered: {len(coefs_advanced) - 1} total features")
print(f"  Test RMSE: {stats_advanced[stats_advanced['metric'] == 'rmse']['value'].iloc[1]:.4f}")

print("\n" + "="*70)
print("Key Differences: Formula vs Recipe")
print("="*70)

print("""
FORMULA APPROACH:
  ✓ Simple and quick for basic models
  ✓ R-style syntax (familiar to tidymodels users)
  ✗ Limited preprocessing (no normalization, encoding, etc.)
  ✗ Can't easily create engineered features
  ✗ Hard to reuse preprocessing across models

RECIPE APPROACH:
  ✓ Powerful feature engineering pipeline
  ✓ Normalize, encode, impute, transform data
  ✓ Create interactions, polynomials, splines
  ✓ Reusable preprocessing across models
  ✓ Automatically applied to test data
  ✓ 51 preprocessing steps available
  ✗ More verbose for simple cases

WHEN TO USE EACH:
  - Formula: Quick prototyping, simple linear models
  - Recipe: Production models, feature engineering, complex preprocessing
""")

print("\n" + "="*70)
print("Available Recipe Steps (51 total)")
print("="*70)

print("""
IMPUTATION (6 steps):
  step_impute_median()   - Fill missing with median
  step_impute_mean()     - Fill missing with mean
  step_impute_mode()     - Fill missing with mode
  step_impute_knn()      - KNN imputation
  step_impute_bag()      - Bagged tree imputation
  step_impute_linear()   - Linear model imputation

NORMALIZATION (4 steps):
  step_normalize()       - Z-score normalization
  step_range()           - Min-max scaling [0,1]
  step_center()          - Center to mean=0
  step_scale()           - Scale to sd=1

ENCODING (6 steps):
  step_dummy()           - One-hot encoding
  step_one_hot()         - One-hot (alternative)
  step_target_encode()   - Target encoding
  step_ordinal()         - Ordinal encoding
  step_bin()             - Discretization
  step_date()            - Extract date features

FEATURE ENGINEERING (8 steps):
  step_poly()            - Polynomial features
  step_interact()        - Interaction terms
  step_ns()              - Natural splines
  step_bs()              - B-splines
  step_pca()             - Principal components
  step_log()             - Log transform
  step_sqrt()            - Square root transform
  step_inverse()         - Inverse transform

FILTERING (6 steps):
  step_corr()            - Remove correlated features
  step_nzv()             - Remove zero-variance
  step_filter_missing()  - Remove columns with too many NAs
  step_select()          - Select specific columns
  step_rm()              - Remove columns
  step_filter()          - Row filtering

ROW OPERATIONS (6 steps):
  step_sample()          - Sample rows
  step_filter()          - Filter rows by condition
  step_slice()           - Select row ranges
  step_arrange()         - Sort rows
  step_shuffle()         - Shuffle rows
  step_lag()             - Create lagged features

TRANSFORMATIONS (6 steps):
  step_mutate()          - Create/modify columns
  step_discretize()      - Discretization
  step_cut()             - Cut into bins
  step_BoxCox()          - Box-Cox transform
  step_YeoJohnson()      - Yeo-Johnson transform
  step_other()           - Pool infrequent categories

TIME SERIES (4 steps):
  step_lag()             - Create lags
  step_diff()            - Differencing
  step_date()            - Extract date features
  step_timeseries_signature() - Comprehensive time features

And more! See documentation for full list.
""")

print("="*70)
print("Try it yourself! Edit and run this script.")
print("="*70)
